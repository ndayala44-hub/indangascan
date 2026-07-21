"""
Card detection and perspective rectification.

Strategy (in order of preference):
1. Contour analysis on an edge map — find convex quadrilaterals whose aspect
   ratio is close to the ISO ID-1 card ratio (~1.586). Each candidate is
   scored on area, aspect-ratio fit, and rectangularity; the best one wins.
   This handles multiple rectangular objects in frame.
2. Adaptive-threshold fallback for low-contrast backgrounds.
3. Minimum-area-rectangle fallback over the largest foreground region.

The winning quadrilateral is warped with a perspective transform to a
normalized, upright card of fixed dimensions.
"""

import logging

import cv2
import numpy as np

from app.config import settings
from app.core.errors import BlurryImageError, CardNotDetectedError, LowResolutionError

logger = logging.getLogger(__name__)

# Detection tuning constants
MIN_CARD_AREA_RATIO = 0.04   # candidate must cover >= 4% of the frame
MAX_CARD_AREA_RATIO = 0.995
ASPECT_TOLERANCE = 0.45      # relative deviation allowed from ID-1 ratio
BLUR_THRESHOLD = 45.0        # variance of Laplacian below this => too blurry


# --------------------------------------------------------------------------- #
# Quality gates
# --------------------------------------------------------------------------- #

def validate_image_quality(image: np.ndarray, side: str) -> None:
    """Reject inputs that cannot plausibly produce a good OCR result."""
    h, w = image.shape[:2]
    if min(h, w) < settings.min_image_dimension:
        logger.warning("Image too small", extra={"data": {"side": side, "w": w, "h": h}})
        raise LowResolutionError(f"{side}: {w}x{h} below minimum {settings.min_image_dimension}px")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    focus = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.debug("Focus measure", extra={"data": {"side": side, "laplacian_var": round(focus, 2)}})
    if focus < BLUR_THRESHOLD:
        raise BlurryImageError(f"{side}: laplacian variance {focus:.1f} < {BLUR_THRESHOLD}")


# --------------------------------------------------------------------------- #
# Candidate discovery
# --------------------------------------------------------------------------- #

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _quad_score(quad: np.ndarray, frame_area: float) -> float:
    """
    Confidence score in [0, 1] combining size, aspect-ratio fit and
    rectangularity. Used to rank competing rectangular objects.
    """
    ordered = _order_corners(quad)
    (tl, tr, br, bl) = ordered
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if min(w, h) < 1:
        return 0.0

    ratio = max(w, h) / min(w, h)
    aspect_err = abs(ratio - settings.card_aspect_ratio) / settings.card_aspect_ratio
    if aspect_err > ASPECT_TOLERANCE:
        return 0.0
    aspect_score = 1.0 - (aspect_err / ASPECT_TOLERANCE)

    area = cv2.contourArea(ordered)
    area_ratio = area / frame_area
    if not (MIN_CARD_AREA_RATIO <= area_ratio <= MAX_CARD_AREA_RATIO):
        return 0.0
    area_score = min(area_ratio / 0.35, 1.0)  # saturate once card fills ~35% of frame

    rect_score = area / (w * h + 1e-6)  # 1.0 for a perfect rectangle

    return 0.45 * aspect_score + 0.30 * area_score + 0.25 * min(rect_score, 1.0)


def _find_quads(mask: np.ndarray, frame_area: float) -> list[tuple[np.ndarray, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[np.ndarray, float]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        peri = cv2.arcLength(contour, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                score = _quad_score(approx, frame_area)
                if score > 0:
                    candidates.append((approx, score))
                break
        else:
            # Not a clean quad: fall back to its minimum-area rectangle.
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.int32)
            score = _quad_score(box, frame_area) * 0.85  # slight penalty
            if score > 0:
                candidates.append((box, score))
    return candidates


def _edge_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(cv2.dilate(edges, kernel, iterations=2), cv2.MORPH_CLOSE, kernel)


def _threshold_mask(gray: np.ndarray) -> np.ndarray:
    thresh = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def detect_and_rectify(image: np.ndarray, side: str) -> tuple[np.ndarray, float]:
    """
    Locate the ID card in `image`, crop it and correct perspective.

    Returns (rectified_card_bgr, detection_confidence). The rectified card is
    always landscape at settings.card_output_width and ID-1 aspect ratio.
    Raises CardNotDetectedError when no plausible card is found.
    """
    h, w = image.shape[:2]
    frame_area = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    candidates: list[tuple[np.ndarray, float]] = []

    # Pre-cropped uploads: if the frame itself is card-shaped, the whole frame
    # is a strong candidate — this stops inner rectangles (portrait window,
    # flag box, guilloche motifs) from winning on already-cropped photos.
    frame_ratio = max(h, w) / min(h, w)
    frame_aspect_err = abs(frame_ratio - settings.card_aspect_ratio) / settings.card_aspect_ratio
    if frame_aspect_err < 0.12:
        full_frame = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        candidates.append((full_frame, 0.80 * (1.0 - frame_aspect_err / 0.12) + 0.10))
        logger.debug("Full-frame candidate added", extra={"data": {"side": side, "aspect_err": round(frame_aspect_err, 3)}})

    for name, mask_fn in (("edges", _edge_mask), ("threshold", _threshold_mask)):
        found = _find_quads(mask_fn(gray), frame_area)
        logger.debug(
            "Detection pass", extra={"data": {"side": side, "method": name, "candidates": len(found)}}
        )
        candidates.extend(found)
        if any(score >= 0.75 for _, score in found):
            break  # strong hit; no need for the fallback pass

    if not candidates:
        logger.warning("Card not detected", extra={"data": {"side": side}})
        raise CardNotDetectedError(f"{side}: no card-like quadrilateral found")

    quad, confidence = max(candidates, key=lambda c: c[1])
    logger.info(
        "Card detected",
        extra={"data": {"side": side, "confidence": round(confidence, 3), "candidates": len(candidates)}},
    )
    return _warp(image, quad), confidence


def _warp(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp the quadrilateral to a normalized landscape card."""
    ordered = _order_corners(quad)
    (tl, tr, br, bl) = ordered
    src_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    src_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    out_w = settings.card_output_width
    out_h = int(round(out_w / settings.card_aspect_ratio))

    # If the detected quad is portrait, warp to portrait first, then rotate,
    # so text is never mirrored or squashed.
    if src_h > src_w:
        dst = np.array([[0, 0], [out_h - 1, 0], [out_h - 1, out_w - 1], [0, out_w - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        card = cv2.warpPerspective(image, matrix, (out_h, out_w))
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)  # orientation.py fixes 180° later
    else:
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        card = cv2.warpPerspective(image, matrix, (out_w, out_h))
    return card
