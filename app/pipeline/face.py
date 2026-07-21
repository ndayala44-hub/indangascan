"""
Face detection, alignment, embedding and matching.

Two engines, selected automatically at startup:

- PRODUCTION (preferred): OpenCV's YuNet face detector + SFace face
  recognizer, both running natively through cv2 with two small ONNX models.
  SFace cosine similarity with the OpenCV Zoo's published threshold gives
  production-grade 1:1 verification. Models are fetched once by
  scripts/download_models.sh (also wired into the devcontainer/Dockerfile).

- FALLBACK (no model files present): Haar-cascade detection + eye-aligned,
  illumination-normalized HOG feature cosine similarity. This keeps the full
  workflow functional with zero downloads, but classical features are NOT
  reliable for identity assurance; every response produced in this mode is
  tagged engine="fallback_hog" and the UI must present it as demo-grade.

Biometric data policy: portraits, live frames and embeddings live in process
memory only for the lifetime of a verification session; nothing is written
to disk and no image content is logged.
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("FACE_MODEL_DIR", "models")
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

# OpenCV Zoo published SFace cosine-distance threshold for verification.
SFACE_COSINE_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", 0.363))
# Fallback HOG similarity threshold - calibrated loose, demo-grade only.
HOG_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD_FALLBACK", 0.86))

_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_EYES = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
_ALT2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

_detector = None
_recognizer = None
ENGINE = "fallback_hog"

if os.path.isfile(YUNET_PATH) and os.path.isfile(SFACE_PATH):
    try:
        _detector = cv2.FaceDetectorYN_create(YUNET_PATH, "", (320, 320), 0.7, 0.3, 5000)
        _recognizer = cv2.FaceRecognizerSF_create(SFACE_PATH, "")
        ENGINE = "sface"
        logger.info("Face engine: SFace (production)")
    except Exception:
        logger.exception("Failed to load ONNX face models; using fallback engine")
if ENGINE == "fallback_hog":
    logger.warning(
        "Face engine: fallback HOG (demo-grade). Run scripts/download_models.sh "
        "to enable production SFace matching."
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def detect_largest_face(bgr: np.ndarray):
    """
    Return (x, y, w, h, landmarks|None) for the largest face, or None.
    Landmarks (YuNet only): right eye, left eye, nose, mouth corners.
    """
    if ENGINE == "sface":
        h, w = bgr.shape[:2]
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        return int(face[0]), int(face[1]), int(face[2]), int(face[3]), face

    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    faces = _FRONTAL.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    for x, y, w, h in sorted(faces, key=lambda f: f[2] * f[3], reverse=True):
        # Haar false-positives on texture: accept a candidate only with a
        # corroborating signal - a detectable eye, or consensus from a
        # second cascade over the same region.
        roi = gray[y:y + h, x:x + w]
        eyes = _EYES.detectMultiScale(roi, 1.05, 3, minSize=(max(8, w // 12),) * 2)
        if len(eyes) >= 1:
            return int(x), int(y), int(w), int(h), None
        pad = 10
        region = gray[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if len(_ALT2.detectMultiScale(region, 1.1, 4, minSize=(int(w * 0.6),) * 2)) >= 1:
            return int(x), int(y), int(w), int(h), None
    return None


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

def _fallback_align(bgr: np.ndarray, box) -> np.ndarray | None:
    """Eye-based rotation alignment + tight crop for the HOG fallback."""
    x, y, w, h, _ = box
    face = bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eyes = _EYES.detectMultiScale(cv2.equalizeHist(gray), 1.1, 5,
                                  minSize=(w // 8, w // 8))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
        c1 = (x1 + w1 / 2, y1 + h1 / 2)
        c2 = (x2 + w2 / 2, y2 + h2 / 2)
        angle = np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0]))
        if abs(angle) < 25:
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            face = cv2.warpAffine(face, m, (w, h))
    return face


def embed_face(bgr: np.ndarray) -> tuple[np.ndarray, str] | None:
    """Return (embedding, engine) for the largest face in the image, or None."""
    box = detect_largest_face(bgr)
    if box is None:
        return None

    if ENGINE == "sface":
        aligned = _recognizer.alignCrop(bgr, box[4])
        feature = _recognizer.feature(aligned)
        return feature, "sface"

    face = _fallback_align(bgr, box)
    if face is None:
        return None
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 112), interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    hog = cv2.HOGDescriptor((96, 112), (16, 16), (8, 8), (8, 8), 9)
    feature = hog.compute(gray).astype(np.float32).ravel()
    norm = np.linalg.norm(feature)
    return (feature / norm if norm > 0 else feature), "fallback_hog"


def match(embedding_a: np.ndarray, embedding_b: np.ndarray) -> tuple[float, bool, float]:
    """
    Compare two embeddings from the same engine.
    Returns (similarity 0..1-ish, verified, threshold_used).
    """
    if ENGINE == "sface":
        score = float(_recognizer.match(embedding_a, embedding_b,
                                        cv2.FaceRecognizerSF_FR_COSINE))
        return score, score >= SFACE_COSINE_THRESHOLD, SFACE_COSINE_THRESHOLD

    score = float(np.dot(embedding_a.ravel(), embedding_b.ravel()))
    return score, score >= HOG_THRESHOLD, HOG_THRESHOLD
