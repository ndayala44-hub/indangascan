"""
Pipeline orchestrator.

process_id_card() takes decoded front/back images and runs the full chain:

    quality gate -> detect & rectify -> orientation -> enhance -> OCR
                 -> parse fields -> extract portrait

Each stage is timed and logged; the returned dictionary is what the API
serializes to the client (processed images are base64-encoded JPEGs).
"""

import base64
import logging
import time
from typing import Any

import cv2
import numpy as np

from app.pipeline import detector, enhance, ocr, orientation, parser, portrait

logger = logging.getLogger(__name__)


def _b64_jpeg(image: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _process_side(image: np.ndarray, side: str) -> dict[str, Any]:
    timings: dict[str, float] = {}

    t = time.perf_counter()
    detector.validate_image_quality(image, side)
    card, det_conf = detector.detect_and_rectify(image, side)
    timings["detect_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    card, rotation = orientation.correct_orientation(card, side)
    timings["orientation_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    display = enhance.enhance_for_display(card)
    variants = enhance.enhance_variants(card)
    timings["enhance_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    ocr_results = [ocr.run_ocr(v, f"{side}/v{i+1}") for i, v in enumerate(variants)]
    timings["ocr_ms"] = (time.perf_counter() - t) * 1000

    logger.info(
        "Side processed",
        extra={"data": {"side": side, "rotation_applied": rotation,
                        "detection_confidence": round(float(det_conf), 3),
                        "timings_ms": {k: round(v) for k, v in timings.items()}}},
    )
    return {
        "card": card,
        "display": display,
        "ocr_results": ocr_results,
        "detection_confidence": float(det_conf),
        "rotation_applied": rotation,
        "timings": timings,
    }


def process_id_card(front: np.ndarray, back: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()

    front_res = _process_side(front, "front")
    back_res = _process_side(back, "back")

    parsed = parser.parse_card(front_res["ocr_results"], back_res["ocr_results"])

    face_img, face_conf, face_method = portrait.extract_portrait(front_res["card"])

    total_ms = (time.perf_counter() - started) * 1000
    logger.info("Scan complete", extra={"data": {"total_ms": round(total_ms)}})

    best_front = max(front_res["ocr_results"], key=lambda r: r.mean_confidence)
    best_back = max(back_res["ocr_results"], key=lambda r: r.mean_confidence)
    return {
        "status": "ok",
        "processing_ms": round(total_ms),
        "images": {
            "front": _b64_jpeg(front_res["display"]),
            "back": _b64_jpeg(back_res["display"]),
            "portrait": _b64_jpeg(face_img) if face_img is not None else None,
        },
        "portrait": {"found": face_img is not None, "confidence": float(face_conf), "method": face_method},
        "detection": {
            "front": {"confidence": round(float(front_res["detection_confidence"]), 3),
                      "rotation_applied": front_res["rotation_applied"]},
            "back": {"confidence": round(float(back_res["detection_confidence"]), 3),
                     "rotation_applied": back_res["rotation_applied"]},
        },
        "ocr_mean_confidence": {
            "front": round(float(best_front.mean_confidence), 1),
            "back": round(float(best_back.mean_confidence), 1),
        },
        **parsed,
    }
