"""
Pipeline orchestrator.

Front and back run in parallel (Tesseract is a subprocess, so threads give
true concurrency). Per side:

    quality gate -> detect & rectify -> orientation -> normalize
                 -> full OCR pass -> [front only] region OCR
                 -> [adaptive] fallback rendering if the primary read is weak

then field parsing merges the full pass, the fallback pass (when run) and
the region candidates, keeping the best-reading value per field.
"""

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from app.pipeline import detector, enhance, face as face_module, mrz, ocr, orientation, parser, passport_parser, portrait, regions
from app.verification import liveness
from app.verification.session import store as session_store

logger = logging.getLogger(__name__)

FALLBACK_CONFIDENCE = 45.0  # run the second rendering below this mean confidence


def _verification_offer(face_img: np.ndarray | None, document_type: str) -> dict[str, Any]:
    """
    If the document portrait embeds successfully, open a verification session
    keyed by a single-use token; only the embedding is retained server-side.
    """
    if face_img is None:
        return {"available": False, "reason": "no_portrait"}
    embedded = face_module.embed_face(face_img)
    if embedded is None:
        # Document portraits can be too small/degraded for the face gate.
        return {"available": False, "reason": "portrait_not_embeddable"}
    embedding, engine = embedded
    session = session_store.create(document_type, embedding, engine, liveness.pick_challenges())
    return {
        "available": True,
        "token": session.token,
        "challenges": [{"id": c, "instruction": liveness.CHALLENGES[c]} for c in session.challenges],
        "engine": engine,
        "engine_reliable": engine == "sface",
        "expires_in_seconds": 600,
    }


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
    normalized = enhance.ocr_normalize(card)
    timings["enhance_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    ocr_results = [ocr.run_ocr(normalized, f"{side}/primary")]
    if side == "front" and ocr_results[0].mean_confidence < FALLBACK_CONFIDENCE:
        logger.info("Primary read weak; running fallback rendering", extra={"data": {"side": side}})
        ocr_results.append(ocr.run_ocr(enhance.ocr_fallback(card), f"{side}/fallback"))
    timings["ocr_ms"] = (time.perf_counter() - t) * 1000

    region_texts: dict[str, tuple[str, float]] = {}
    if side == "front":
        t = time.perf_counter()
        region_texts = regions.read_regions(normalized)
        timings["regions_ms"] = (time.perf_counter() - t) * 1000

    logger.info(
        "Side processed",
        extra={"data": {"side": side, "rotation_applied": rotation,
                        "detection_confidence": round(float(det_conf), 3),
                        "timings_ms": {k: round(v) for k, v in timings.items()}}},
    )
    return {
        "card": card,
        "ocr_results": ocr_results,
        "region_texts": region_texts,
        "detection_confidence": float(det_conf),
        "rotation_applied": rotation,
    }


def _passport_page_bounds(ocr_result) -> tuple[int, int] | None:
    """Find (top, bottom) of the bio page from the primary pass's line boxes."""
    header_y = None
    mrz_bottom = None
    for line in ocr_result.lines:
        text = line.text.upper()
        if header_y is None and any(k in text for k in ("REPUBULIKA", "PASIPORO", "PASSEPORT")):
            header_y = line.top
        if line.text.count("<") >= 4:
            mrz_bottom = max(mrz_bottom or 0, line.bottom)
    if header_y is not None and mrz_bottom is not None and mrz_bottom > header_y:
        return header_y, mrz_bottom
    return None


def process_passport(page: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()

    detector.validate_image_quality(page, "passport")
    page, rotation = orientation.correct_orientation(page, "passport")
    normalized = enhance.ocr_normalize(page)

    ocr_results = [ocr.run_ocr(normalized, "passport/primary")]
    if ocr_results[0].mean_confidence < FALLBACK_CONFIDENCE:
        ocr_results.append(ocr.run_ocr(enhance.ocr_fallback(page), "passport/fallback"))

    mrz_result = mrz.parse_from_text(max(ocr_results, key=lambda r: r.mean_confidence).full_text)

    parsed = passport_parser.parse_passport(ocr_results, mrz_result)

    # Display crop: header-to-MRZ band of the page (2x normalized -> original px)
    bounds = _passport_page_bounds(ocr_results[0])
    if bounds:
        y0 = max(0, int(bounds[0] / 2) - 14)
        y1 = min(page.shape[0], int(bounds[1] / 2) + 14)
        page_crop = page[y0:y1]
    else:
        page_crop = page

    face_img, face_conf, face_method = portrait.extract_portrait(page_crop)
    display = enhance.enhance_for_display(page_crop)

    total_ms = (time.perf_counter() - started) * 1000
    logger.info("Passport scan complete", extra={"data": {"total_ms": round(total_ms),
                                                          "mrz_valid": mrz_result.valid}})
    best = max(ocr_results, key=lambda r: r.mean_confidence)
    return {
        "status": "ok",
        "document_type": "passport",
        "verification": _verification_offer(face_img, "passport"),
        "processing_ms": round(total_ms),
        "images": {
            "front": _b64_jpeg(display),
            "back": None,
            "portrait": _b64_jpeg(face_img) if face_img is not None else None,
        },
        "portrait": {"found": face_img is not None, "confidence": float(face_conf), "method": face_method},
        "detection": {"front": {"confidence": 1.0 if bounds else 0.5, "rotation_applied": rotation}},
        "ocr_mean_confidence": {"front": round(float(best.mean_confidence), 1)},
        **parsed,
    }


def process_id_card(front: np.ndarray, back: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as pool:
        front_future = pool.submit(_process_side, front, "front")
        back_future = pool.submit(_process_side, back, "back")
        front_res = front_future.result()
        back_res = back_future.result()

    parsed = parser.parse_card(
        front_res["ocr_results"], back_res["ocr_results"], front_res["region_texts"]
    )

    face_img, face_conf, face_method = portrait.extract_portrait(front_res["card"])

    front_display = enhance.enhance_for_display(front_res["card"])
    back_display = enhance.enhance_for_display(back_res["card"])

    total_ms = (time.perf_counter() - started) * 1000
    logger.info("Scan complete", extra={"data": {"total_ms": round(total_ms)}})

    best_front = max(front_res["ocr_results"], key=lambda r: r.mean_confidence)
    best_back = max(back_res["ocr_results"], key=lambda r: r.mean_confidence)
    return {
        "status": "ok",
        "document_type": "national_id",
        "verification": _verification_offer(face_img, "national_id"),
        "processing_ms": round(total_ms),
        "images": {
            "front": _b64_jpeg(front_display),
            "back": _b64_jpeg(back_display),
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
