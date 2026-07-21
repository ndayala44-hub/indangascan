"""
Application configuration.

All settings are environment-driven so the same image works across
local development, Docker, and cloud deployment without code changes.
"""

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip().lower() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Upload constraints -------------------------------------------------
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 10)
    allowed_extensions: list[str] = field(
        default_factory=lambda: _env_list("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp")
    )
    min_image_dimension: int = _env_int("MIN_IMAGE_DIMENSION", 480)

    # --- Card geometry (ISO/IEC 7810 ID-1: 85.60 x 53.98 mm) ----------------
    card_output_width: int = _env_int("CARD_OUTPUT_WIDTH", 1000)
    card_aspect_ratio: float = 85.60 / 53.98  # ~1.586

    # --- OCR -----------------------------------------------------------------
    ocr_engine: str = os.getenv("OCR_ENGINE", "tesseract")  # tesseract | easyocr
    tesseract_lang: str = os.getenv("TESSERACT_LANG", "eng")
    low_confidence_threshold: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", 70))

    # --- Logging -------------------------------------------------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "logs")
    log_json: bool = os.getenv("LOG_JSON", "true").lower() == "true"

    # --- Server --------------------------------------------------------------
    cors_origins: list[str] = field(default_factory=lambda: _env_list("CORS_ORIGINS", "*"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
