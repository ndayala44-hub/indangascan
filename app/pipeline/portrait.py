"""
Portrait extraction from the front of the card.

Primary: OpenCV Haar cascade face detection on the rectified card, keeping
the largest face and padding it to a passport-style crop.

Fallback: the Rwandan ID has a fixed layout — the portrait occupies the
left portion of the card — so if no face is detected (heavy glare, faded
print) a layout-based crop is returned with a lower confidence.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Layout fallback region as fractions of the rectified card (x0, y0, x1, y1).
LAYOUT_REGION = (0.03, 0.28, 0.34, 0.95)
PAD_X, PAD_TOP, PAD_BOTTOM = 0.45, 0.55, 0.85  # padding around the detected face box


def extract_portrait(card: np.ndarray) -> tuple[np.ndarray | None, float, str]:
    """Return (portrait_bgr | None, confidence 0-1, method)."""
    h, w = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5, minSize=(int(h * 0.15), int(h * 0.15))
    )
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        x0 = max(0, int(x - fw * PAD_X))
        y0 = max(0, int(y - fh * PAD_TOP))
        x1 = min(w, int(x + fw * (1 + PAD_X)))
        y1 = min(h, int(y + fh * (1 + PAD_BOTTOM)))
        logger.info(
            "Portrait via face detection",
            extra={"data": {"faces": int(len(faces)), "box": [int(x), int(y), int(fw), int(fh)]}},
        )
        return card[y0:y1, x0:x1].copy(), 0.9, "face_detection"

    x0, y0, x1, y1 = (
        int(LAYOUT_REGION[0] * w), int(LAYOUT_REGION[1] * h),
        int(LAYOUT_REGION[2] * w), int(LAYOUT_REGION[3] * h),
    )
    crop = card[y0:y1, x0:x1].copy()
    # Sanity check: the region should contain visual structure, not flat color.
    if float(np.std(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))) < 12:
        logger.warning("Portrait not found (layout region is flat)")
        return None, 0.0, "none"
    logger.info("Portrait via layout fallback")
    return crop, 0.5, "layout_fallback"
