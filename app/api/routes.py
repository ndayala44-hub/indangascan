"""
REST API routes.

POST /api/v1/scan   multipart form with `front` and `back` image files.
GET  /api/v1/health liveness probe (also reports the active OCR engine).
"""

import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.core.errors import (
    CorruptedImageError,
    FileTooLargeError,
    MissingImageError,
    UnsupportedFormatError,
)
from app.pipeline.processor import process_id_card, process_passport

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


async def _read_and_decode(upload: UploadFile | None, side: str) -> np.ndarray:
    if upload is None or not upload.filename:
        raise MissingImageError(f"{side} image not provided")

    ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
    if ext not in settings.allowed_extensions:
        raise UnsupportedFormatError(f"{side}: extension '.{ext}' not in {settings.allowed_extensions}")

    data = await upload.read()
    logger.info(
        "Upload received",
        extra={"data": {"side": side, "filename": upload.filename, "bytes": len(data)}},
    )
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"{side}: {len(data)} bytes > {settings.max_upload_bytes}",
            user_message=f"The {side} image exceeds the {settings.max_upload_mb} MB limit.",
        )
    if len(data) < 100:
        raise CorruptedImageError(f"{side}: file is empty or truncated")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise CorruptedImageError(f"{side}: cv2.imdecode returned None")
    return image


@router.post("/scan")
async def scan(
    front: UploadFile | None = File(None),
    back: UploadFile | None = File(None),
    doc_type: str = Form("national_id"),
):
    # The pipeline is CPU-bound OpenCV/Tesseract work; keep the event loop free.
    if doc_type == "passport":
        page_img = await _read_and_decode(front, "passport page")
        return await run_in_threadpool(process_passport, page_img)
    front_img = await _read_and_decode(front, "front")
    back_img = await _read_and_decode(back, "back")
    return await run_in_threadpool(process_id_card, front_img, back_img)


@router.get("/health")
async def health():
    from app.pipeline.face import ENGINE as face_engine
    from app.verification.liveness import LIVENESS_ENGINE
    return {"status": "ok", "ocr_engine": settings.ocr_engine,
            "face_engine": face_engine, "liveness_engine": LIVENESS_ENGINE,
            "max_upload_mb": settings.max_upload_mb}
