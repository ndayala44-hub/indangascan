"""
Pipeline smoke tests with a synthetic Rwandan-ID-style card.

Run with:  python -m pytest tests/ -v      (or plain `python tests/test_pipeline.py`)

The synthetic card is rendered with the real bilingual field labels, then
photographed "badly" on purpose: pasted on a cluttered background, rotated,
perspective-skewed, and flipped upside down. The tests assert the pipeline
recovers the card, fixes orientation, and extracts the structured fields.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from app.pipeline import detector, orientation, enhance, ocr, parser, processor

CARD_W, CARD_H = 1000, 630


def make_front_card() -> np.ndarray:
    card = np.full((CARD_H, CARD_W, 3), (235, 225, 205), dtype=np.uint8)  # pale blue-ish
    cv2.rectangle(card, (0, 0), (CARD_W - 1, CARD_H - 1), (140, 120, 60), 6)

    def put(text, y, scale=1.0, bold=2, x=40, color=(30, 30, 30)):
        cv2.putText(card, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

    put("REPUBULIKA Y'U RWANDA / REPUBLIC OF RWANDA", 60, 0.85, 2)
    put("INDANGAMUNTU / IDENTITY CARD", 105, 0.8, 2)
    # portrait placeholder (left side)
    cv2.rectangle(card, (40, 180), (330, 560), (180, 170, 150), -1)
    cv2.circle(card, (185, 300), 70, (120, 110, 95), -1)          # head
    cv2.ellipse(card, (185, 480), (110, 120), 0, 180, 360, (120, 110, 95), -1)  # shoulders
    x = 370
    put("Amazina / Names", 210, 0.65, 1, x, (80, 80, 80))
    put("NDAYISHIMIYE Alain", 250, 0.85, 2, x)
    put("Itariki yavutseho / Date of Birth", 320, 0.65, 1, x, (80, 80, 80))
    put("15/03/1989", 360, 0.85, 2, x)
    put("Igitsina / Sex", 430, 0.65, 1, x, (80, 80, 80))
    put("Gabo / M", 470, 0.85, 2, x)
    put("Indangamuntu / National ID No", 540, 0.65, 1, x, (80, 80, 80))
    put("1 1989 8 0031866 1 85", 580, 0.9, 2, x)
    return card


def make_back_card() -> np.ndarray:
    card = np.full((CARD_H, CARD_W, 3), (230, 230, 225), dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (CARD_W - 1, CARD_H - 1), (140, 120, 60), 6)
    cv2.putText(card, "SIGNATURE OF THE HOLDER", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.putText(card, "NIDA123456789", (60, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    return card


def photograph(card: np.ndarray, angle_deg: float, upside_down: bool) -> np.ndarray:
    """Simulate a handheld photo: background clutter, rotation, perspective."""
    if upside_down:
        card = cv2.rotate(card, cv2.ROTATE_180)
    bg = np.random.randint(60, 110, (1400, 1900, 3), dtype=np.uint8)  # dark wooden desk
    ch, cw = card.shape[:2]
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    # mild perspective skew + rotation, placed off-center
    rot = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle_deg, 0.75)
    pts = cv2.transform(src.reshape(-1, 1, 2), rot).reshape(-1, 2)
    pts += np.float32([420, 330])
    pts += np.float32([[0, 0], [-25, 18], [12, -14], [20, 10]])  # skew corners
    matrix = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    warped = cv2.warpPerspective(card, matrix, (1900, 1400), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), matrix, (1900, 1400))
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    return out


def test_full_pipeline():
    front_photo = photograph(make_front_card(), angle_deg=8, upside_down=True)
    back_photo = photograph(make_back_card(), angle_deg=-5, upside_down=False)

    result = processor.process_id_card(front_photo, back_photo)

    assert result["status"] == "ok"
    assert result["images"]["front"].startswith("data:image/jpeg;base64,")
    fields = {f["key"]: f for f in result["fields"]}

    assert fields["national_id_number"]["value"] is not None, "NID not extracted"
    assert "1989" in fields["national_id_number"]["value"]
    assert fields["date_of_birth"]["value"] == "15/03/1989"
    assert fields["sex"]["value"] and "Male" in fields["sex"]["value"]
    assert fields["surname"]["value"] and "ndayishimiye" in fields["surname"]["value"].lower()
    assert all(v["passed"] for v in result["validations"]), result["validations"]
    print("\nExtracted:", {k: f["value"] for k, f in fields.items()})
    print("Timings: total", result["processing_ms"], "ms")


def test_card_not_detected():
    noise = np.random.randint(0, 255, (900, 1200, 3), dtype=np.uint8)
    try:
        detector.detect_and_rectify(noise, "front")
        raise AssertionError("expected CardNotDetectedError")
    except Exception as exc:
        assert type(exc).__name__ == "CardNotDetectedError"


if __name__ == "__main__":
    test_full_pipeline()
    test_card_not_detected()
    print("ALL TESTS PASSED")
