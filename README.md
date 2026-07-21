# IndangaScan — Rwandan National ID Scanner (MVP)

A production-quality MVP web application, in the spirit of BlinkID, optimized for the Rwandan National Identity Card. Upload the front and back of a card and the system will detect the card, crop it, correct perspective and orientation, enhance the image, run OCR, extract structured fields with per-field confidence scores, and pull out the holder's portrait.

## What it does

Upload any handheld photo of the card, at any rotation, and the pipeline runs: quality gate (resolution and blur checks), card detection and perspective rectification, automatic orientation correction (0/90/180/270), image enhancement (glare inpainting, denoising, CLAHE contrast, brightness normalization, sharpening), OCR, structured field parsing, portrait extraction, and cross-validation of the extracted data against the redundancy encoded in the 16-digit National ID number (birth year and sex digits). Results are shown in a modal with the processed card images on the left and the extracted fields with confidence scores on the right; low-confidence fields are flagged for manual review.

Extracted front fields: National ID Number (Indangamuntu), Full Name, Surname, Given Names, Date of Birth, Sex, Nationality/holder status (derived from the ID number), Place of Issue. The back side is fully OCR'd and returned line by line, with the card serial pulled out where present. The Rwandan card does not print expiry date or issuing authority; the parser is spec-driven (see Extending below), so those fields can be added in minutes if a future card revision carries them.

## Architecture

```
app/
├── main.py                  FastAPI wiring, CORS, request-ID middleware, error envelope
├── config.py                Environment-driven settings
├── api/routes.py            POST /api/v1/scan, GET /api/v1/health
├── core/
│   ├── logging_config.py    Structured JSON logs, rotating files, per-request IDs
│   └── errors.py            Stable error codes mapped to friendly messages
├── pipeline/
│   ├── processor.py         Orchestrator (timed stages)
│   ├── detector.py          Card detection + perspective rectification (OpenCV)
│   ├── orientation.py       Tesseract OSD + brute-force rotation scoring
│   ├── enhance.py           Display and OCR enhancement pipelines
│   ├── ocr.py               Engine abstraction (Tesseract default, EasyOCR optional)
│   ├── parser.py            Spec-driven Rwandan ID field extraction + validation
│   └── portrait.py          Haar-cascade face crop with layout fallback
└── static/index.html        Responsive frontend (no build step required)
tests/test_pipeline.py       End-to-end smoke test on a synthetic warped card photo
```

Every request gets an `x-request-id` that appears on every log line it produces, so a single upload can be traced through detection, OCR and parsing. Logs are JSON lines (set `LOG_JSON=false` for plain text locally) written to both console and `logs/app.log` with rotation. Domain failures (card not detected, blurry image, file too large, unsupported format, corrupted upload, OCR failure, and so on) return a consistent envelope: `{"error": {"code", "message", "request_id"}}` with a user-friendly message, while the technical detail and stack trace go to the logs only.

The frontend is a single static page with hand-written CSS and vanilla JS served by the API itself, which keeps the MVP a one-container deployment with zero build tooling. It talks only to the REST API, so it can be swapped for a React/Next.js client (or a mobile app) without touching the backend.

## Run it locally

Requires Python 3.11+ and the Tesseract binary (`apt install tesseract-ocr` on Debian/Ubuntu, `brew install tesseract` on macOS).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://localhost:8000 — API docs live at `/api/docs`.

Run the test suite (renders a synthetic Rwandan-ID-style card, photographs it upside down at an angle with perspective skew on a cluttered background, and asserts the pipeline recovers every field):

```bash
python tests/test_pipeline.py        # or: pytest tests/ -v
```

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

## Deploy online for testing

The container runs anywhere Docker runs. Straightforward options:

- **Render / Railway / Fly.io**: point the service at this repo; the Dockerfile is detected automatically. One shared-CPU instance handles MVP traffic; give it 1 GB RAM (2 GB if you enable EasyOCR).
- **GitHub Codespaces** (quick shared testing): `uvicorn app.main:app --host 0.0.0.0 --port 8000`, then set port 8000 visibility to public.
- **GCP Cloud Run / Azure Container Apps / AWS App Runner**: build and push the image, deploy with `PORT=8000`, min instances 0 for a free-tier-friendly test deployment.

## Configuration

All via environment variables (see `.env.example`): `MAX_UPLOAD_MB`, `ALLOWED_EXTENSIONS`, `MIN_IMAGE_DIMENSION`, `OCR_ENGINE` (`tesseract` or `easyocr`), `TESSERACT_LANG`, `LOW_CONFIDENCE_THRESHOLD`, `LOG_LEVEL`, `LOG_JSON`, `CORS_ORIGINS`.

## Extending

- **Add a field**: append a `FieldSpec` in `app/pipeline/parser.py` with the bilingual anchor labels and/or a regex plus an optional normalizer. No other code changes.
- **Swap OCR engines**: add a `_run_*` function in `app/pipeline/ocr.py` (PaddleOCR, Google Vision, Azure Document Intelligence) and register it in `_ENGINES`.
- **Future integrations** the module boundaries are designed for: MRZ reading, QR/barcode decoding (the back of newer cards), face verification against the extracted portrait, liveness detection, batch processing (the `/scan` endpoint is stateless and thread-pooled), and mobile clients via the same REST API.

## API

`POST /api/v1/scan` — multipart form fields `front` and `back` (JPG/JPEG/PNG/WEBP). Returns processed images (base64 JPEG), extracted fields with confidences, cross-validation results, portrait, timings, and the raw back-side text. `GET /api/v1/health` reports the active OCR engine and upload limit.

## Privacy note

Processed images are returned to the browser and are not persisted server-side. Log lines record field values at INFO level for debugging; set `LOG_LEVEL=WARNING` in production if extracted personal data must not reach the logs, and review retention obligations under Rwanda's Data Privacy and Protection Law before storing any scan output.
