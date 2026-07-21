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




# --------------------------------------------------------------------------- #
# Passport tests (fictional data; check digits computed per ICAO 9303)
# --------------------------------------------------------------------------- #

def _fictional_mrz() -> tuple[str, str]:
    from app.pipeline.mrz import _check_digit
    doc = "PC1234567"
    birth, expiry, personal = "900215", "300101", "<" * 14
    l2 = (doc + _check_digit(doc) + "RWA" + birth + _check_digit(birth) + "F"
          + expiry + _check_digit(expiry) + personal + _check_digit(personal))
    l2 += _check_digit(doc + _check_digit(doc) + birth + _check_digit(birth)
                       + expiry + _check_digit(expiry) + personal + _check_digit(personal))
    return "PCRWAUWIMANA<<CLAUDINE<<<<<<<<<<<<<<<<<<<<<<", l2


def test_mrz_parser():
    from app.pipeline import mrz
    l1, l2 = _fictional_mrz()
    r = mrz.parse_td3(l1, l2)
    assert r.valid, r.checks
    assert r.passport_number == "PC1234567"
    assert r.surname == "Uwimana" and r.given_names == "Claudine"
    assert r.date_of_birth == "15/02/1990" and r.sex == "F" and r.date_of_expiry == "01/01/2030"

    # OCR-noise robustness: spaces inside fields, K-for-< filler misreads
    noisy2 = l2[:30] + "K<K" + l2[33:]
    r2 = mrz.parse_td3(l1, noisy2.replace("RWA", "RW A"))
    assert r2.passport_number == "PC1234567"


def test_synthetic_passport_page():
    from app.pipeline import processor
    l1, l2 = _fictional_mrz()
    page = np.full((900, 1300, 3), (235, 240, 240), dtype=np.uint8)

    def put(text, y, scale=0.9, bold=2, x=60, color=(30, 30, 30)):
        cv2.putText(page, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

    put("REPUBULIKA Y'U RWANDA / REPUBLIC OF RWANDA", 70, 0.9)
    put("PASIPORO / PASSPORT", 130, 0.8)
    put("Surname", 210, 0.55, 1, 60, (90, 90, 90)); put("UWIMANA", 250)
    put("Other names", 320, 0.55, 1, 60, (90, 90, 90)); put("CLAUDINE", 360)
    put("Date of birth", 430, 0.55, 1, 60, (90, 90, 90)); put("15 FEB/FEV 1990", 470)
    put("Date of issue", 540, 0.55, 1, 60, (90, 90, 90)); put("01 JAN/JAN 2020", 580)
    put("Issuing authority", 650, 0.55, 1, 60, (90, 90, 90)); put("GOVERNMENT OF RWANDA", 690)
    put("Place of issue", 740, 0.55, 1, 60, (90, 90, 90)); put("KIGALI", 780)
    put(l1, 840, 0.62, 1)
    put(l2, 880, 0.62, 1)

    res = processor.process_passport(page)
    fields = {f["key"]: f for f in res["fields"]}
    assert res["mrz"]["found"] and res["mrz"]["valid"], res["mrz"]
    assert fields["passport_number"]["value"] == "PC1234567"
    assert fields["surname"]["value"] == "Uwimana"
    assert fields["date_of_birth"]["value"] == "15/02/1990"
    assert fields["date_of_expiry"]["value"] == "01/01/2030"
    assert fields["sex"]["value"] == "F"
    assert fields["date_of_issue"]["value"] == "01/01/2020"
    print("Passport synthetic OK:", {k: f["value"] for k, f in fields.items()})




# --------------------------------------------------------------------------- #
# Verification: liveness logic + session store
# --------------------------------------------------------------------------- #

def test_liveness_logic():
    from app.verification.liveness import (
        FrameFeatures, verify_blink, verify_smile, verify_turn_left, verify_turn_right,
    )
    F = lambda **kw: FrameFeatures(**{"face": (100, 100, 200, 200), **kw})

    blink_seq = [F(eyes=2), F(eyes=2), F(eyes=0), F(eyes=0), F(eyes=2), F(eyes=2)]
    assert verify_blink(blink_seq)
    assert not verify_blink([F(eyes=2)] * 6), "eyes always open is not a blink"
    assert not verify_blink([F(eyes=0)] * 6), "eyes never open is not a blink"

    assert verify_smile([F(), F(smile=True), F(smile=True), F()])
    assert not verify_smile([F(smile=True)] + [F()] * 5), "one frame can be a false hit"

    turn_l = [F(), F(), FrameFeatures(face=None, profile_left=True),
              FrameFeatures(face=None, profile_left=True), F()]
    assert verify_turn_left(turn_l)
    assert not verify_turn_right(turn_l), "left turn must not satisfy the right challenge"

    drift_r = [F(face=(100, 100, 200, 200)), F(face=(160, 100, 200, 200)),
               F(face=(230, 100, 200, 200)), FrameFeatures(face=None)]
    assert verify_turn_right(drift_r), "rightward drift + face loss counts as a right turn"


def test_session_store_and_offer():
    import numpy as np
    from app.verification.session import SessionStore
    from app.pipeline.processor import _verification_offer

    s = SessionStore()
    emb = np.ones(8, dtype=np.float32)
    session = s.create("national_id", emb, "fallback_hog", ["blink", "smile"])
    assert s.get(session.token) is not None
    assert not session.liveness_passed
    session.completed = {"blink": True, "smile": True}
    assert session.liveness_passed
    s.delete(session.token)
    assert s.get(session.token) is None

    assert _verification_offer(None, "national_id") == {"available": False, "reason": "no_portrait"}

def test_head_pose_round_trip():
    """Rotate the 3D model by known angles, project, re-estimate: signs and
    magnitudes must survive the round trip (pins the pose conventions)."""
    import math
    from app.verification.pose import MODEL_POINTS, camera_matrix, estimate_pose
    W, H = 640, 480
    K = camera_matrix(W, H)

    def project(yaw_deg, pitch_deg):
        base = np.diag([1.0, -1.0, -1.0])
        y, p = math.radians(yaw_deg), math.radians(pitch_deg)
        ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        rvec, _ = cv2.Rodrigues(base @ ry @ rx)
        t = np.array([[0.0], [0.0], [400.0]])
        pts, _ = cv2.projectPoints(MODEL_POINTS, rvec, t, K, None)
        return pts.reshape(6, 2)

    for yaw_in, pitch_in in [(20, 0), (-20, 0), (0, 15), (0, -15), (18, 12)]:
        yaw, pitch, _ = estimate_pose(project(yaw_in, pitch_in), W, H)
        assert abs(yaw - yaw_in) < 1.0, (yaw_in, yaw)
        assert abs(pitch - pitch_in) < 1.0, (pitch_in, pitch)


def test_liveness_pose_logic():
    from app.verification.liveness import (
        FrameFeatures, verify_blink, verify_smile,
        verify_turn_left, verify_turn_right, verify_look_up, verify_look_down,
    )
    F = lambda **kw: FrameFeatures(**{"face": (0, 0, 100, 100), **kw})

    assert verify_blink([F(ear=0.30), F(ear=0.31), F(ear=0.12), F(ear=0.29), F(ear=0.30)])
    assert not verify_blink([F(ear=0.30)] * 6)
    assert verify_smile([F(mouth_ratio=0.42), F(mouth_ratio=0.42),
                         F(mouth_ratio=0.50), F(mouth_ratio=0.51)])

    left = [F(yaw=2.0), F(yaw=3.0), F(yaw=11.0), F(yaw=18.0), F(yaw=17.0)]
    right = [F(yaw=1.0), F(yaw=-9.0), F(yaw=-16.0), F(yaw=-17.0)]
    assert verify_turn_left(left) and not verify_turn_right(left)
    assert verify_turn_right(right) and not verify_turn_left(right)
    # Adaptive baseline: a user starting off-center is judged on the delta.
    off = [F(yaw=-8.0), F(yaw=-7.0), F(yaw=4.0), F(yaw=8.0), F(yaw=7.5)]
    assert verify_turn_left(off)
    # A full turn that rotates the face out of detection still passes.
    full = [F(yaw=2.0), F(yaw=9.0), FrameFeatures(), FrameFeatures(), FrameFeatures()]
    assert verify_turn_left(full)

    up = [F(pitch=0.0), F(pitch=1.0), F(pitch=9.0), F(pitch=13.0), F(pitch=12.0)]
    down = [F(pitch=2.0), F(pitch=-6.0), F(pitch=-11.0), F(pitch=-12.0)]
    assert verify_look_up(up) and not verify_look_down(up)
    assert verify_look_down(down) and not verify_look_up(down)

    still = [F(yaw=1.0, pitch=1.0)] * 6
    for v in (verify_turn_left, verify_turn_right, verify_look_up, verify_look_down):
        assert not v(still)


if __name__ == "__main__":
    test_full_pipeline()
    test_card_not_detected()
    test_mrz_parser()
    test_synthetic_passport_page()
    test_liveness_logic()
    test_session_store_and_offer()
    test_head_pose_round_trip()
    test_liveness_pose_logic()
    print("ALL TESTS PASSED")
