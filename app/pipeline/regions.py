"""
Region-targeted OCR for the front of the Rwandan National ID.

After detection and rectification every card shares the same geometry, so
each field's value line sits in a known zone. Re-reading those zones with
Tesseract's single-line mode (--psm 7) and field-appropriate character
whitelists is both faster and markedly more accurate than relying on the
full-page pass alone: page segmentation can't wander, and the guilloche
between lines never enters the crop.

Region results are candidates, not authorities — the parser merges them
with the full-page pass and keeps whichever reads better, so a future card
revision with a shifted layout degrades gracefully to label-anchored
parsing instead of breaking.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    key: str                                  # parser field key this feeds
    box: tuple[float, float, float, float]    # y0, y1, x0, x1 as card fractions
    config: str


# Calibrated on rectified current-generation cards (landscape, ID-1 ratio).
FRONT_REGIONS: list[Region] = [
    Region("full_name", (0.385, 0.470, 0.28, 0.95), "--psm 7"),
    Region("date_of_birth", (0.535, 0.615, 0.24, 0.66),
           "--psm 7 -c tessedit_char_whitelist=0123456789/"),
    Region("sex", (0.640, 0.735, 0.28, 0.62), "--psm 7"),
    Region("place_of_issue", (0.640, 0.735, 0.28, 0.97), "--psm 7"),
    Region("national_id_number", (0.850, 1.000, 0.03, 1.00), "--psm 7"),
]


def read_regions(normalized_gray: np.ndarray) -> dict[str, tuple[str, float]]:
    """
    OCR each calibrated zone of the normalized (background-flattened) card.
    Returns {field_key: (raw_text, mean_word_confidence)}.
    """
    h, w = normalized_gray.shape[:2]
    out: dict[str, tuple[str, float]] = {}
    for region in FRONT_REGIONS:
        y0, y1, x0, x1 = region.box
        crop = normalized_gray[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        try:
            raw = pytesseract.image_to_string(crop, config=region.config).strip()
            if not raw:
                continue
            # Best-effort confidence; image_to_data segments small crops more
            # conservatively than image_to_string, so missing words are normal.
            data = pytesseract.image_to_data(
                crop, config=region.config, output_type=pytesseract.Output.DICT
            )
            confs = [float(c) for t, c in zip(data["text"], data["conf"])
                     if t.strip() and float(c) > 0]
            out[region.key] = (raw, float(np.mean(confs)) if confs else 55.0)
        except Exception:
            logger.exception("Region OCR failed", extra={"data": {"region": region.key}})
            continue
    logger.debug("Region OCR", extra={"data": {k: v[0] for k, v in out.items()}})
    return out
