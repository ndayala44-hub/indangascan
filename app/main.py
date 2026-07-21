"""
Application entrypoint.

    uvicorn app.main:app --reload

Responsibilities kept here: app wiring, CORS, per-request ID + timing
middleware, the global error envelope, and serving the static frontend.
"""

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.verify_routes import router as verify_router
from app.config import settings
from app.core.errors import AppError
from app.core.logging_config import request_id_ctx, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Rwanda ID Scanner", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request ID to every log line and time the request."""
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request_id_ctx.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Ensure even middleware-level crashes are logged with the request ID.
        logger.exception("Unhandled error in request pipeline")
        response = JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR",
                               "message": AppError.user_message, "request_id": rid}},
        )
    duration = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = rid
    if request.url.path.startswith("/api"):
        logger.info(
            "Request finished",
            extra={"data": {"method": request.method, "path": request.url.path,
                            "status": response.status_code, "duration_ms": round(duration)}},
        )
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """Expected domain failures: friendly message out, technical detail to logs."""
    logger.warning(
        "Domain error", extra={"data": {"code": exc.code, "detail": exc.detail}},
        exc_info=exc if exc.http_status >= 500 else None,
    )
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": {"code": exc.code, "message": exc.user_message,
                           "request_id": request_id_ctx.get()}},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": AppError.user_message,
                           "request_id": request_id_ctx.get()}},
    )


app.include_router(router)
app.include_router(verify_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("app/static/index.html")
