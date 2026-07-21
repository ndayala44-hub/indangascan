"""
Automatic orientation correction.

The rectified card from detector.py is always landscape, so the realistic
ambiguity is 0 vs 180 degrees. Both are scored on a downscaled copy with a
readability measure that heavily weights the card's own bilingual
vocabulary (RWANDA, INDANGAMUNTU, Amazina, Gabo, ...). Plain word-shape
scoring is not enough: Tesseract confidently reads upside-down text as
garbage-but-alphabetic words, whereas the known vocabulary only ever
matches in the true orientation. If neither direction reads at all, 90 and
270 are scored as well (an edge case where the warp produced vertical text).
"""

import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}

SCORE_SCALE = 0.6       # downscale factor for scoring (speed/accuracy balance)
MIN_READABLE = 60.0     # below this for 0 and 180, try the vertical rotations
FLIP_MARGIN = 1.3       # a rotation must beat 0 deg by this factor to apply
KEYWORD_WEIGHT = 12.0   # multiplier applied to known card vocabulary hits

# Bilingual vocabulary printed on the Rwandan National ID (front and back).
CARD_KEYWORDS = {
    "repubulika", "rwanda", "republic", "indangamuntu", "national", "identity",
    "card", "amazina", "names", "itariki", "yavutseho", "date", "birth",
    "igitsina", "sex", "aho", "yatangiwe", "place", "issue", "gabo", "gore",
    "umukono", "nyirayo", "signature", "found", "please", "return", "nearest",
    "whoever", "uses", "this", "contrary", "law", "will", "punished",
    "uzayikoresha", "karita", "police",
    # Passport bio-data page vocabulary
    "pasiporo", "passport", "passeport", "pasipoti", "republique", "jamhuri",
    "surname", "nom", "prenoms", "nationality", "nationalite", "government",
    "naissance", "delivrance", "expiry", "authority", "kigali",
}


def _apply_rotation(image: np.ndarray, angle: int) -> np.ndarray:
    op = _ROTATIONS.get(angle % 360)
    return image if op is None else cv2.rotate(image, op)


def _readability_score(gray: np.ndarray) -> float:
    """Confidence-weighted word score with a strong bonus for card vocabulary."""
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 6")
    except Exception:
        return 0.0
    score = 0.0
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        conf = float(conf)
        if conf > 40 and len(word) >= 3 and re.fullmatch(r"[A-Za-z]+", word):
            weight = KEYWORD_WEIGHT if word.lower() in CARD_KEYWORDS else 1.0
            score += conf * min(len(word), 8) * weight
    return score


def correct_orientation(card: np.ndarray, side: str) -> tuple[np.ndarray, int]:
    """Return (upright_card, applied_rotation_degrees). No manual rotation needed."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=SCORE_SCALE, fy=SCORE_SCALE, interpolation=cv2.INTER_AREA)

    scores = {0: _readability_score(small), 180: _readability_score(_apply_rotation(small, 180))}
    if max(scores.values()) < MIN_READABLE:
        # Nothing reads horizontally - check the vertical orientations too.
        scores[90] = _readability_score(_apply_rotation(small, 90))
        scores[270] = _readability_score(_apply_rotation(small, 270))

    best = max(scores, key=scores.get)
    if best != 0 and scores[best] < scores[0] * FLIP_MARGIN + 50:
        best = 0  # near-tie: keep the as-captured orientation
    logger.info(
        "Orientation",
        extra={"data": {"side": side, "scores": {k: round(v) for k, v in scores.items()}, "rotate": best}},
    )
    return _apply_rotation(card, best), best
