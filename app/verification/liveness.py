"""
Facial liveness detection (active challenge-response + passive checks).

The primary anti-spoofing mechanism is the randomized challenge: the server
picks the challenges, so a printed photo, a static injected image, or a
pre-recorded video cannot respond to instructions it has never seen. Each
challenge is verified server-side from a short burst of camera frames.

Per-frame features are extracted once (frontal face, eyes, smile, profile
orientation) and the challenge verifiers are pure functions over those
feature sequences - which keeps the decision logic unit-testable without a
camera.

Passive checks run on every burst:
- motion:   consecutive frames must differ inside the face region (rejects
            identical injected stills).
- presence: a face must be present in most frames of the burst.

Honest scope note: this defeats prints, static injections and pre-recorded
replays. It does not claim resistance to a live sophisticated deepfake
camera injection; that requires attested capture hardware or vendor-grade
passive liveness models, and is documented as future work.
"""

import logging
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.pipeline import face as face_module
from app.verification import pose

logger = logging.getLogger(__name__)

# Optional precision engine: MediaPipe FaceMesh gives 468 landmarks, from
# which eye aperture (EAR), mouth geometry and signed head yaw are computed
# deterministically. Falls back to Haar-cascade heuristics when absent.
# MediaPipe backend probing. The legacy Solutions FaceMesh API has been
# progressively removed from recent mediapipe releases; the supported path
# is the Tasks FaceLandmarker (needs models/face_landmarker.task, fetched by
# scripts/download_models.sh). Probe at import, in order of preference:
#   1. tasks FaceLandmarker  2. legacy FaceMesh  3. Haar cascades.
import os as _os

_MP = None
_TASK_PATH = _os.path.join(_os.getenv("FACE_MODEL_DIR", "models"), "face_landmarker.task")
_LANDMARKER_OPTS = None
_LEGACY_OK = False
LIVENESS_ENGINE = "cascades"
try:
    import mediapipe as mp
    _MP = mp
    try:
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision
        if _os.path.isfile(_TASK_PATH):
            _LANDMARKER_OPTS = _mp_vision.FaceLandmarkerOptions(
                base_options=_mp_python.BaseOptions(model_asset_path=_TASK_PATH),
                running_mode=_mp_vision.RunningMode.IMAGE, num_faces=1,
            )
            # Construct once to validate, then discard.
            _mp_vision.FaceLandmarker.create_from_options(_LANDMARKER_OPTS).close()
            LIVENESS_ENGINE = "mediapipe_tasks"
    except Exception:
        _LANDMARKER_OPTS = None
    if LIVENESS_ENGINE == "cascades":
        try:
            mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                min_detection_confidence=0.5).close()
            _LEGACY_OK = True
            LIVENESS_ENGINE = "mediapipe"
        except Exception:
            pass
except Exception:
    _MP = None
logger_probe = LIVENESS_ENGINE


def _new_facemesh():
    """One legacy FaceMesh per burst, STATIC mode (see extraction notes)."""
    return _MP.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5,
    )


# FaceMesh landmark indices (canonical face mesh topology).
_L_EYE = (33, 160, 158, 133, 153, 144)   # p1..p6 for EAR
_R_EYE = (362, 385, 387, 263, 373, 380)
_MOUTH_L, _MOUTH_R = 61, 291
_NOSE_TIP, _CHEEK_L, _CHEEK_R = 1, 234, 454

_SMILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
_PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

CHALLENGES = {
    "blink": "Blink slowly",
    "smile": "Smile",
    "turn_left": "Turn your head to your left",
    "turn_right": "Turn your head to your right",
    "look_up": "Tilt your head up",
    "look_down": "Tilt your head down",
}
# Pitch challenges need metric head pose; the cascade fallback can't judge them.
_CASCADE_CHALLENGES = ["blink", "smile", "turn_left", "turn_right"]

CHALLENGES_PER_SESSION = 2

# Degree thresholds, environment-tunable. Deltas are measured against each
# burst's own starting pose (adaptive baseline), so a user who begins
# slightly off-center is not penalized and no exaggerated motion is needed.
import os
TURN_DELTA_DEG = float(os.getenv("LIVENESS_TURN_DEG", 13.0))
PITCH_DELTA_DEG = float(os.getenv("LIVENESS_PITCH_DEG", 10.0))
MIN_FRAMES = 5
MAX_FRAMES = 16  # turn bursts send 12
# Set to true if a deployment's camera pipeline mirrors frames.
TURN_INVERTED = False


def pick_challenges() -> list[str]:
    pool = list(CHALLENGES) if LIVENESS_ENGINE.startswith("mediapipe") else _CASCADE_CHALLENGES
    return random.sample(pool, CHALLENGES_PER_SESSION)


# --------------------------------------------------------------------------- #
# Per-frame feature extraction
# --------------------------------------------------------------------------- #

@dataclass
class FrameFeatures:
    face: tuple[int, int, int, int] | None = None
    eyes: int = 0
    smile: bool = False
    profile_left: bool = False
    profile_right: bool = False
    motion: float = 0.0   # mean abs diff vs previous frame within face region
    # Landmark-based signals (MediaPipe engine only; None under cascades)
    ear: float | None = None          # eye aspect ratio - drops sharply on blink
    mouth_ratio: float | None = None  # mouth width / inter-cheek width
    yaw: float | None = None          # head yaw, degrees; + = subject's left
    pitch: float | None = None        # head pitch, degrees; + = looking up


def _ear(pts, idx) -> float:
    p = [pts[i] for i in idx]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float((v1 + v2) / (2.0 * h + 1e-6))


def _features_from_points(pts, bgr, gray, prev_gray) -> FrameFeatures:
    """Shared feature computation for both MediaPipe backends."""
    f = FrameFeatures()
    h, w = bgr.shape[:2]
    xs, ys = pts[:, 0], pts[:, 1]
    x0, y0 = int(max(0, xs.min())), int(max(0, ys.min()))
    x1, y1 = int(min(w, xs.max())), int(min(h, ys.max()))
    f.face = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    f.ear = (_ear(pts, _L_EYE) + _ear(pts, _R_EYE)) / 2.0
    cheek_w = float(np.linalg.norm(pts[_CHEEK_R] - pts[_CHEEK_L])) + 1e-6
    f.mouth_ratio = float(np.linalg.norm(pts[_MOUTH_R] - pts[_MOUTH_L])) / cheek_w
    image_pts = np.array([pts[pose.POSE_LANDMARKS[k]] for k in (
        "nose_tip", "chin", "right_eye_outer", "left_eye_outer",
        "mouth_right", "mouth_left")], dtype=np.float64)
    angles = pose.estimate_pose(image_pts, w, h)
    if angles:
        f.yaw, f.pitch = angles[0], angles[1]
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = cv2.absdiff(gray[y0:y1, x0:x1], prev_gray[y0:y1, x0:x1])
        f.motion = float(diff.mean()) if diff.size else 0.0
    return f


def _extract_tasks(frames: list[np.ndarray]) -> list[FrameFeatures]:
    from mediapipe.tasks.python import vision as mp_vision
    out: list[FrameFeatures] = []
    prev_gray = None
    landmarker = mp_vision.FaceLandmarker.create_from_options(_LANDMARKER_OPTS)
    try:
        for bgr in frames:
            h, w = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = _MP.Image(image_format=_MP.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)
            if res.face_landmarks:
                pts = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]],
                               dtype=np.float32)
                out.append(_features_from_points(pts, bgr, gray, prev_gray))
            else:
                out.append(FrameFeatures())
            prev_gray = gray
    finally:
        landmarker.close()
    return out


def _extract_mediapipe(frames: list[np.ndarray]) -> list[FrameFeatures]:
    out: list[FrameFeatures] = []
    prev_gray = None
    with _new_facemesh() as mesh:
        for bgr in frames:
            f = FrameFeatures()
            h, w = bgr.shape[:2]
            res = mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if res.multi_face_landmarks:
                pts = np.array(
                    [[lm.x * w, lm.y * h] for lm in res.multi_face_landmarks[0].landmark],
                    dtype=np.float32)
                f = _features_from_points(pts, bgr, gray, prev_gray)
            prev_gray = gray
            out.append(f)
    return out


def extract_features(frames: list[np.ndarray]) -> list[FrameFeatures]:
    if LIVENESS_ENGINE == "mediapipe_tasks":
        return _extract_tasks(frames)
    if LIVENESS_ENGINE == "mediapipe":
        return _extract_mediapipe(frames)
    return _extract_cascades(frames)


def _extract_cascades(frames: list[np.ndarray]) -> list[FrameFeatures]:
    out: list[FrameFeatures] = []
    prev_gray = None
    for bgr in frames:
        f = FrameFeatures()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)

        box = face_module.detect_largest_face(bgr)
        if box:
            x, y, w, h = box[:4]
            f.face = (x, y, w, h)
            roi = eq[y:y + h, x:x + w]
            f.eyes = len(face_module._EYES.detectMultiScale(
                roi[: h * 3 // 5], 1.05, 3, minSize=(max(8, w // 12),) * 2))
            f.smile = len(_SMILE.detectMultiScale(
                roi[h // 2:], 1.5, 12, minSize=(w // 4, h // 8))) > 0
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff = cv2.absdiff(gray[y:y + h, x:x + w], prev_gray[y:y + h, x:x + w])
                f.motion = float(diff.mean())

        # Profile orientation (independent of frontal success).
        f.profile_left = len(_PROFILE.detectMultiScale(eq, 1.1, 5, minSize=(70, 70))) > 0
        f.profile_right = len(_PROFILE.detectMultiScale(
            cv2.flip(eq, 1), 1.1, 5, minSize=(70, 70))) > 0

        prev_gray = gray
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Challenge verifiers (pure logic over feature sequences)
# --------------------------------------------------------------------------- #

def _face_presence(feats: list[FrameFeatures]) -> float:
    return sum(1 for f in feats if f.face) / max(len(feats), 1)


def verify_blink(feats: list[FrameFeatures]) -> bool:
    ears = [f.ear for f in feats if f.ear is not None]
    if len(ears) >= 4:
        # Landmark path: the eye aspect ratio dips sharply during a blink
        # relative to this burst's own open-eye baseline (person-agnostic).
        baseline = float(np.median(sorted(ears, reverse=True)[: max(2, len(ears) // 2)]))
        low = baseline * 0.62
        dipped = [i for i, e in enumerate(ears) if e < low]
        if not dipped:
            return False
        i = dipped[0]
        return any(e >= baseline * 0.85 for e in ears[:i]) and                any(e >= baseline * 0.85 for e in ears[i + 1:])
    # Cascade path
    open_before = closed = open_after = False
    for f in feats:
        if not f.face:
            continue
        if f.eyes >= 1 and not closed:
            open_before = True
        elif f.eyes == 0 and open_before:
            closed = True
        elif f.eyes >= 1 and closed:
            open_after = True
    return open_before and closed and open_after


def verify_smile(feats: list[FrameFeatures]) -> bool:
    ratios = [f.mouth_ratio for f in feats if f.mouth_ratio is not None]
    if len(ratios) >= 4:
        # A smile widens the mouth relative to the burst's neutral minimum.
        neutral = float(min(ratios))
        return sum(1 for r in ratios if r >= neutral * 1.10) >= 2
    return sum(1 for f in feats if f.smile) >= 2


def _pose_baseline(values: list[float]) -> float:
    """
    Adaptive per-burst baseline: the pose at the start of the burst, before
    the user moves. Recording begins as the instruction is given, so the
    first frames are neutral; on short sequences trust the first frame.
    """
    head = values[:3] if len(values) >= 6 else values[:1]
    return float(np.median(head))


def _verify_turn(feats: list[FrameFeatures], left: bool) -> bool:
    if TURN_INVERTED:
        left = not left
    yaws = [f.yaw for f in feats if f.yaw is not None]
    if len(yaws) >= 2:
        baseline = _pose_baseline(yaws)
        signed = [((y - baseline) if left else (baseline - y)) for y in yaws]
        peak = max(signed)
        correct = sum(1 for v in signed if v > TURN_DELTA_DEG * 0.75)
        wrong = sum(1 for v in signed if v < -TURN_DELTA_DEG)
        logger.info("Turn trace", extra={"data": {
            "left": left, "baseline_deg": round(baseline, 1),
            "delta_deg": [round(v, 1) for v in signed]}})
        if peak > TURN_DELTA_DEG and correct >= 2 and wrong < correct:
            return True
        # A full turn rotates the face out of the detector's range: accept a
        # correct-direction motion followed by losing the face mid-burst.
        present = [f.face is not None for f in feats]
        early_present = any(present[: max(2, len(present) // 3)])
        lost_later = sum(1 for p in present[len(present) // 2:] if not p) >= 2
        rising = signed[-1] - signed[0] if len(signed) >= 2 else 0.0
        if early_present and lost_later and (
            signed[-1] > TURN_DELTA_DEG * 0.4 or rising > TURN_DELTA_DEG * 0.4
        ):
            return True
        return False
    # Cascade path (unchanged)
    profile_hits = sum(
        1 for f in feats if (f.profile_left if left else f.profile_right) and not f.face
    )
    if profile_hits >= 1:
        return True
    xs = [f.face[0] + f.face[2] / 2 for f in feats if f.face]
    if len(xs) >= 3 and _face_presence(feats) < 0.9:
        drift = xs[-1] - xs[0]
        widths = [f.face[2] for f in feats if f.face]
        if abs(drift) > 0.15 * float(np.mean(widths)):
            return (drift < 0) if left else (drift > 0)
    return False


def verify_look_up(feats: list[FrameFeatures]) -> bool:
    return _verify_pitch(feats, up=True)


def verify_look_down(feats: list[FrameFeatures]) -> bool:
    return _verify_pitch(feats, up=False)


def _verify_pitch(feats: list[FrameFeatures], up: bool) -> bool:
    pitches = [f.pitch for f in feats if f.pitch is not None]
    if len(pitches) < 3:
        return False
    baseline = _pose_baseline(pitches)
    signed = [((p - baseline) if up else (baseline - p)) for p in pitches]
    correct = sum(1 for v in signed if v > PITCH_DELTA_DEG * 0.75)
    wrong = sum(1 for v in signed if v < -PITCH_DELTA_DEG)
    logger.info("Pitch trace", extra={"data": {
        "up": up, "baseline_deg": round(baseline, 1),
        "delta_deg": [round(v, 1) for v in signed]}})
    return max(signed) > PITCH_DELTA_DEG and correct >= 2 and wrong < correct


def verify_turn_left(feats: list[FrameFeatures]) -> bool:
    return _verify_turn(feats, left=True)


def verify_turn_right(feats: list[FrameFeatures]) -> bool:
    return _verify_turn(feats, left=False)


_VERIFIERS = {
    "blink": verify_blink,
    "smile": verify_smile,
    "turn_left": verify_turn_left,
    "turn_right": verify_turn_right,
    "look_up": verify_look_up,
    "look_down": verify_look_down,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def verify_challenge(challenge: str, frames: list[np.ndarray]) -> dict:
    """Verify one challenge over a burst of frames; returns a result dict."""
    if challenge not in _VERIFIERS:
        return {"passed": False, "reason": "unknown_challenge"}
    if not (MIN_FRAMES <= len(frames) <= MAX_FRAMES):
        return {"passed": False, "reason": f"need {MIN_FRAMES}-{MAX_FRAMES} frames"}

    try:
        feats = extract_features(frames)
    except Exception:
        logger.exception("Feature extraction failed",
                         extra={"data": {"challenge": challenge, "engine": LIVENESS_ENGINE}})
        return {"passed": False, "reason": "processing_error", "engine": LIVENESS_ENGINE}
    presence = _face_presence(feats)
    motions = [f.motion for f in feats[1:] if f.face]
    motion_ok = bool(motions) and max(motions) > 1.0  # identical stills => ~0

    # Turns legitimately lose the frontal face; other challenges must keep it.
    min_presence = 0.35 if challenge.startswith("turn") else 0.7
    if presence < min_presence:
        return {"passed": False, "reason": "face_not_visible", "face_presence": round(presence, 2)}
    if not motion_ok:
        return {"passed": False, "reason": "no_motion_detected", "face_presence": round(presence, 2)}

    passed = _VERIFIERS[challenge](feats)
    result = {
        "passed": passed,
        "reason": None if passed else "challenge_not_detected",
        "engine": LIVENESS_ENGINE,
        "face_presence": round(presence, 2),
        "motion": round(float(max(motions)), 2),
    }
    logger.info("Challenge verified", extra={"data": {"challenge": challenge, **result}})
    return result
