"""
Identity verification API.

Flow:
  1. POST /scan (existing) - when a portrait is found, the response now
     carries a `verification` object with a session token and the challenge
     list. The portrait's embedding is held server-side against that token.
  2. POST /verify/challenge - a burst of camera frames for one challenge;
     returns pass/fail with a reason for real-time UI feedback. Challenges
     may be retried until the session expires.
  3. POST /verify/complete - a final frame; requires all challenges passed,
     embeds the live face and matches it against the document portrait.
     The session is destroyed afterwards (single use).
"""

import datetime
import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.errors import AppError, CorruptedImageError
from app.pipeline import face as face_module
from app.verification import liveness
from app.verification.session import store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/verify")


class SessionError(AppError):
    code = "VERIFICATION_SESSION_INVALID"
    http_status = 400
    user_message = "This verification session is invalid or has expired. Please scan the document again."


class LivenessIncompleteError(AppError):
    code = "LIVENESS_INCOMPLETE"
    http_status = 400
    user_message = "Complete all liveness challenges before finishing verification."


class NoLiveFaceError(AppError):
    code = "NO_LIVE_FACE"
    http_status = 422
    user_message = "No face could be detected in the camera capture. Face the camera in good light and try again."


def _decode_frames(data: list[bytes]) -> list[np.ndarray]:
    frames = []
    for blob in data:
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise CorruptedImageError("camera frame failed to decode")
        frames.append(img)
    return frames


@router.post("/challenge")
async def challenge(token: str = Form(...), challenge: str = Form(...),
                    frames: list[UploadFile] = File(...)):
    session = store.get(token)
    if session is None or challenge not in session.challenges:
        raise SessionError(f"token or challenge invalid: {challenge}")
    blobs = [await f.read() for f in frames]
    decoded = _decode_frames(blobs)
    result = await run_in_threadpool(liveness.verify_challenge, challenge, decoded)
    if result["passed"]:
        session.completed[challenge] = True
    remaining = [c for c in session.challenges if not session.completed.get(c)]
    return {**result, "challenge": challenge, "remaining": remaining}


@router.post("/complete")
async def complete(token: str = Form(...), frame: UploadFile = File(...)):
    session = store.get(token)
    if session is None:
        raise SessionError("token invalid or expired")
    if not session.liveness_passed:
        raise LivenessIncompleteError(f"completed={session.completed}")

    blob = await frame.read()
    live = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if live is None:
        raise CorruptedImageError("final frame failed to decode")

    embedded = await run_in_threadpool(face_module.embed_face, live)
    if embedded is None:
        raise NoLiveFaceError()
    live_embedding, engine = embedded

    score, matched, threshold = face_module.match(session.portrait_embedding, live_embedding)
    reliable = engine == "sface"
    verified = bool(matched and reliable)

    result = {
        "status": "verified" if verified else "verification_failed",
        "verified": verified,
        "similarity_score": round(score, 4),
        "threshold": threshold,
        "liveness": {
            "passed": True,
            "challenges": session.challenges,
            "confidence": round(sum(session.completed.values()) / len(session.challenges), 2),
        },
        "engine": engine,
        "engine_reliable": reliable,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if not reliable:
        result["note"] = (
            "Similarity is indicative only: the production face-matching model is not "
            "installed, and identity is never confirmed in demo mode. "
            "Run scripts/download_models.sh to enable verified matching."
        )
    elif not matched:
        result["note"] = "The live face does not sufficiently match the document portrait."

    logger.info("Verification complete",
                extra={"data": {"verified": verified, "score": round(score, 4),
                                "engine": engine, "doc": session.document_type}})
    session.finished = True
    store.delete(token)
    return result
