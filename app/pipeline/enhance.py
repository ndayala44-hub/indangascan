"""
Image enhancement for OCR.

Two outputs are produced from the upright card:

- `display` : a color image with glare reduction, denoising, contrast
  normalization and mild sharpening — what the user sees in the results view.
- `ocr`     : a grayscale, adaptively thresholded, upscaled image tuned to
  maximize OCR accuracy. OCR engines generally prefer dark text on a clean
  white background at ~300 DPI equivalent.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GLARE_V_THRESHOLD = 245     # HSV value above which a pixel counts as glare
GLARE_S_THRESHOLD = 40      # ...combined with low saturation
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


def enhance_for_display(card: np.ndarray) -> np.ndarray:
    """Cleaned-up color card for on-screen verification."""
    out = reduce_glare(card)
    out = cv2.fastNlMeansDenoisingColored(out, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=15)

    # Contrast normalization on luminance only (CLAHE), preserving color.
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Mild unsharp mask.
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=2)
    return cv2.addWeighted(out, 1.4, blurred, -0.4, 0)


def enhance_for_ocr(card: np.ndarray) -> np.ndarray:
    """Primary OCR image — first variant of enhance_variants()."""
    return enhance_variants(card)[0]


def enhance_variants(card: np.ndarray) -> list[np.ndarray]:
    """
    Multiple grayscale renderings of the card, each tuned for a different
    failure mode. OCR runs on all of them and the parser keeps the best
    value per field.

    Variant 1 — division normalization: dividing the image by a heavily
    blurred copy of itself flattens the guilloche security pattern printed
    across the whole card while preserving the darker text strokes. This is
    the workhorse for real Rwandan cards.

    Variant 2 — division normalization + unsharp mask, for slightly soft
    captures where the text strokes need reinforcing.

    Variant 3 — the classic denoise + CLAHE + percentile-stretch pipeline,
    which wins on flat, low-contrast or faded cards where there is no
    background pattern to amplify.
    """
    gray = cv2.cvtColor(reduce_glare(card), cv2.COLOR_BGR2GRAY)

    # Upscale once for all variants (~600 DPI equivalent helps small ID fonts).
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # --- Variant 1: background-flattening division normalization
    background = cv2.GaussianBlur(up, (0, 0), sigmaX=25)
    divided = cv2.divide(up, background, scale=255)
    v1 = cv2.fastNlMeansDenoising(divided, None, h=7, templateWindowSize=7, searchWindowSize=21)

    # --- Variant 2: v1 + unsharp mask
    blurred = cv2.GaussianBlur(v1, (0, 0), sigmaX=1.2)
    v2 = cv2.addWeighted(v1, 1.5, blurred, -0.5, 0)

    # --- Variant 3: denoise + CLAHE + robust brightness stretch + sharpen
    v3 = cv2.fastNlMeansDenoising(up, None, h=7, templateWindowSize=7, searchWindowSize=21)
    v3 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(v3)
    p2, p98 = np.percentile(v3, (2, 98))
    if p98 > p2:
        v3 = np.clip((v3.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    b3 = cv2.GaussianBlur(v3, (0, 0), sigmaX=1.5)
    v3 = cv2.addWeighted(v3, 1.6, b3, -0.6, 0)

    return [v1, v2, v3]
