"""
Structured logging for the whole application.

- JSON logs (one object per line) suitable for log aggregation, or plain
  text for local development (LOG_JSON=false).
- Rotating file handler + console handler.
- Every log record carries the request_id set by the API middleware, so a
  single upload can be traced end-to-end through detection, OCR and parsing.
"""

import json
import logging
import logging.handlers
import os
import sys
from contextvars import ContextVar

from app.config import settings

# Set per-request by middleware; safe under async concurrency.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        # Attach structured extras (e.g. logger.info("...", extra={"data": {...}}))
        if hasattr(record, "data"):
            payload["data"] = record.data
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging() -> None:
    os.makedirs(settings.log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()

    if settings.log_json:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(request_id)s | %(name)s | %(message)s"
        )

    console = logging.StreamHandler(sys.stdout)
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(settings.log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)

    # Quiet noisy third-party loggers.
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
