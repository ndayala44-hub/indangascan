"""
Image enhancement.

- `ocr_normalize` : the single canonical OCR image. Division normalization
  (image divided by a heavily blurred copy of itself) flattens the card's
  guilloche security pattern while preserving text strokes. Measured on real
  cards it beats CLAHE-style pipelines on both accuracy and speed - CLAHE
  amplifies the very background pattern that confuses the OCR engine, and
  non-local-means denoising costs seconds while adding nothing after
  division normalization.

- `ocr_fallback` : a CLAHE + contrast-stretch rendering used only when the
  primary pass reads poorly (flat, faded cards with no background pattern).

- `enhance_for_display` : the color image shown to the user, on a fast path
  (bilateral filter instead of non-local means).
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GLARE_V_THRESHOLD = 245      # HSV value above which a pixel counts as glare
GLARE_S_THRESHOLD = 40       # ...combined with low saturation
GLARE_MAX_AREA_RATIO = 0.10  # only inpaint if glare covers < 10% of the card


def reduce_glare(bgr: np.ndarray) -> np.ndarray:
    """Inpaint small saturated specular highlights (camera flash on laminate)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > GLARE_V_THRESHOLD) & (hsv[:, :, 1] < GLARE_S_THRESHOLD)).astype(np.uint8) * 255
    ratio = float(np.count_nonzero(mask)) / mask.size
    if 0 < ratio < GLARE_MAX_AREA_RATIO:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        logger.debug("Glare inpainting", extra={"data": {"glare_ratio": round(ratio, 4)}})
        return cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
    return bgr


def ocr_normalize(card: np.ndarray) -> np.ndarray:
    """Primary OCR image: upscale 2x, flatten background by division."""
    gray = cv2.cvtColor(reduce_glare(card), cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    background = cv2.GaussianBlur(up, (0, 0), sigmaX=25)
    return cv2.divide(up, background, scale=255)


def ocr_fallback(card: np.ndarray) -> np.ndarray:
    """Secondary rendering for flat or faded cards."""
    gray = cv2.cvtColor(reduce_glare(card), cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(up)
    p2, p98 = np.percentile(up, (2, 98))
    if p98 > p2:
        up = np.clip((up.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(up, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(up, 1.6, blurred, -0.6, 0)


def enhance_for_display(card: np.ndarray) -> np.ndarray:
    """Cleaned-up color card for on-screen verification (fast path)."""
    out = reduce_glare(card)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=40, sigmaSpace=40)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=2)
    return cv2.addWeighted(out, 1.4, blurred, -0.4, 0)
