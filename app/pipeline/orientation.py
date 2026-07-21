"""
Automatic orientation correction (0 / 90 / 180 / 270 degrees).

The rectified card from detector.py is already landscape, so the remaining
ambiguity is usually 0 vs 180 (and 90/270 for portrait-shaped inputs that
slipped through). Two complementary signals are used:

1. Tesseract OSD (orientation & script detection) — fast and usually right
   when there is enough text.
2. Brute-force fallback — OCR a downscaled copy at all four rotations and
   keep the rotation with the highest mean word confidence. Slower but very
   robust on low-text or stylized cards.
"""

import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,   # image was rotated 90 CW -> undo
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


def _apply_rotation(image: np.ndarray, angle: int) -> np.ndarray:
    op = _ROTATIONS.get(angle % 360)
    return image if op is None else cv2.rotate(image, op)


def _osd_angle(gray: np.ndarray) -> tuple[int, float] | None:
    """Ask Tesseract OSD for the text rotation. Returns (angle, confidence)."""
    try:
        osd = pytesseract.image_to_osd(gray, config="--psm 0 -c min_characters_to_try=40")
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        conf = float(re.search(r"Orientation confidence: ([\d.]+)", osd).group(1))
        return angle, conf
    except Exception as exc:  # OSD routinely fails on sparse text; not fatal.
        logger.debug("OSD unavailable", extra={"data": {"reason": str(exc)[:200]}})
        return None


def _mean_ocr_confidence(gray: np.ndarray) -> float:
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 6")
        confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) > 0]
        return float(np.mean(confs)) if len(confs) >= 3 else 0.0
    except Exception:
        return 0.0


def _readability_score(gray: np.ndarray) -> float:
    """
    Sum of (confidence x capped length) over confident alphabetic words.
    Real upright text produces long, confident dictionary-like words; rotated
    text and barcode noise produce short garbage. Run at full resolution —
    downscaling lets barcode texture masquerade as words.
    """
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 6")
    except Exception:
        return 0.0
    score = 0.0
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        conf = float(conf)
        if conf > 40 and len(word) >= 3 and re.fullmatch(r"[A-Za-z]+", word):
            score += conf * min(len(word), 8)
    return score


def correct_orientation(card: np.ndarray, side: str) -> tuple[np.ndarray, int]:
    """
    Return (upright_card, applied_rotation_degrees).

    The caller passes a rectified card; the user may have captured it upside
    down, sideways, or mirrored across devices — no manual rotation needed.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    osd = _osd_angle(gray)
    if osd and osd[1] >= 2.0:  # Tesseract OSD confidence is unbounded; >=2 is reliable
        angle, conf = osd
        if angle != 0:
            # OSD can be confidently wrong on barcode- or pattern-heavy sides;
            # trust a non-zero rotation only if it actually reads better.
            s0 = _readability_score(gray)
            s_rot = _readability_score(_apply_rotation(gray, angle))
            if s_rot <= s0:
                logger.info(
                    "OSD rotation rejected by readability check",
                    extra={"data": {"side": side, "osd_angle": angle,
                                    "score_0": round(s0), "score_rot": round(s_rot)}},
                )
                return card, 0
        logger.info("Orientation via OSD", extra={"data": {"side": side, "rotate": angle, "conf": conf}})
        return _apply_rotation(card, angle), angle

    # Fallback: readability-score all four rotations. Rotate away from the
    # as-captured orientation only on a clear margin — most users photograph
    # the card roughly upright, and near-tied scores are noise.
    scores = {angle: _readability_score(_apply_rotation(gray, angle)) for angle in (0, 90, 180, 270)}
    best = max(scores, key=scores.get)
    if best != 0 and scores[best] < scores[0] * 1.3 + 50:
        best = 0
    logger.info(
        "Orientation via brute force",
        extra={"data": {"side": side, "scores": {k: round(v) for k, v in scores.items()}, "rotate": best}},
    )
    return _apply_rotation(card, best), best
