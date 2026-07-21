"""
OCR engine abstraction.

The pipeline talks to a single `run_ocr()` function that returns text lines
with word-level confidences, keeping the rest of the system engine-agnostic.

- Tesseract (default): zero extra weight, ships in the Docker image.
- EasyOCR (optional):  set OCR_ENGINE=easyocr and `pip install easyocr`.
  Often stronger on stylized security fonts, at the cost of a large model
  download and slower cold start.

Swapping in PaddleOCR or a cloud service (Google Vision, Azure Document
Intelligence) only requires adding another `_run_*` function here.
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pytesseract

# Tesseract's internal OpenMP threading oversubscribes CPUs when the front
# and back sides run concurrently; a single thread per process is faster.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

from app.config import settings
from app.core.errors import OcrFailureError

logger = logging.getLogger(__name__)


@dataclass
class OcrLine:
    text: str
    confidence: float                     # 0-100, mean of word confidences
    words: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class OcrResult:
    lines: list[OcrLine]

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        confs = [l.confidence for l in self.lines]
        return float(np.mean(confs)) if confs else 0.0


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

def _run_tesseract(gray: np.ndarray) -> OcrResult:
    data = pytesseract.image_to_data(
        gray,
        lang=settings.tesseract_lang,
        config="--oem 3 --psm 6",  # LSTM engine, uniform block of text
        output_type=pytesseract.Output.DICT,
    )
    lines: dict[tuple, OcrLine] = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line = lines.setdefault(key, OcrLine(text="", confidence=0.0))
        line.words.append((word, conf))

    result_lines = []
    for line in lines.values():
        line.text = " ".join(w for w, _ in line.words)
        line.confidence = float(np.mean([c for _, c in line.words]))
        result_lines.append(line)
    return OcrResult(lines=result_lines)


_easyocr_reader = None  # lazy singleton — model load is expensive


def _run_easyocr(gray: np.ndarray) -> OcrResult:
    global _easyocr_reader
    import easyocr  # optional dependency

    if _easyocr_reader is None:
        logger.info("Loading EasyOCR model (first request only)")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)

    results = _easyocr_reader.readtext(gray, detail=1, paragraph=False)
    lines = [
        OcrLine(text=text, confidence=float(conf) * 100, words=[(text, float(conf) * 100)])
        for _, text, conf in results
        if text.strip()
    ]
    return OcrResult(lines=lines)


_ENGINES = {"tesseract": _run_tesseract, "easyocr": _run_easyocr}


def run_ocr(gray: np.ndarray, side: str) -> OcrResult:
    engine = _ENGINES.get(settings.ocr_engine, _run_tesseract)
    try:
        result = engine(gray)
    except Exception as exc:
        logger.exception("OCR engine failed", extra={"data": {"side": side, "engine": settings.ocr_engine}})
        raise OcrFailureError(f"{side}: {exc}") from exc

    logger.info(
        "OCR complete",
        extra={
            "data": {
                "side": side,
                "engine": settings.ocr_engine,
                "lines": len(result.lines),
                "mean_confidence": round(result.mean_confidence, 1),
            }
        },
    )
    logger.debug("OCR raw text", extra={"data": {"side": side, "text": result.full_text}})
    return result
