# IndangaScan — Rwandan National ID & Passport Scanner (MVP)

A production-quality MVP web application, in the spirit of BlinkID, optimized for Rwandan identity documents. It scans both the Rwandan National Identity Card (front and back) and the Rwandan passport bio-data page: detection, cropping, orientation correction, image enhancement, OCR, structured field extraction with per-field confidence scores, portrait extraction, and document-level cross-validation.

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
│   ├── processor.py         Orchestrator (front/back in parallel, timed stages)
│   ├── detector.py          Card detection + perspective rectification (OpenCV)
│   ├── orientation.py       Readability-scored rotation correction
│   ├── enhance.py           Division-normalization OCR prep + display pipeline
│   ├── ocr.py               Engine abstraction (Tesseract default, EasyOCR optional)
│   ├── regions.py           Calibrated per-field region OCR (single-line passes)
│   ├── mrz.py               ICAO 9303 TD3 MRZ parser: check digits + OCR repair
│   ├── parser.py            Spec-driven ID field extraction, region merge, NID cross-validation
│   ├── passport_parser.py   Passport VIZ extraction with MRZ authority/override
│   ├── face.py              Face detection/embedding/matching (SFace prod, HOG demo)
├── verification/
│   ├── liveness.py          Randomized challenge liveness + passive anti-spoofing
│   └── session.py           In-memory TTL biometric sessions (Redis-swappable)
├── api/verify_routes.py     /verify/challenge and /verify/complete endpoints
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

## Passport support

Select the document type in the UI (or pass `doc_type=passport` to `POST /api/v1/scan` with a single `front` image of the bio-data page). The pipeline locates the page (header-to-MRZ), corrects orientation, and extracts every field: passport number, surname, given names, nationality, date of birth, sex, place of birth, dates of issue and expiry, issuing authority, place of issue, and personal number, plus the raw MRZ with per-field check-digit results.

The MRZ is treated as authoritative: it is parsed per ICAO 9303 (TD3) with all five check digits computed, OCR confusions repaired positionally (O/0, I/1, filler '<' misread as K/R/C/E, line-length reconciliation), and every field it covers overrides the visual zone at verified confidence. The three visual-zone dates are disambiguated against the MRZ (the issue date is the one that is neither the checksummed birth nor expiry date). Fields the MRZ does not carry (place of birth, date of issue, issuing authority, place of issue) come from label-anchored visual-zone extraction with the standard low-confidence review flags.

## Identity verification (liveness + face matching)

After a successful scan that yields a document portrait, the response carries a single-use verification session token and a randomized set of liveness challenges. The UI's "Verify identity" flow opens the user's camera, guides them through each challenge with real-time pass/fail feedback. Challenges are verified server-side from short frame bursts; with MediaPipe installed (default in requirements.txt) verification uses face-mesh landmarks - eye aspect ratio for blinks, mouth geometry for smiles, signed head yaw for turns - which is deterministic and direction-aware. Without MediaPipe the system falls back to Haar-cascade heuristics, clearly labeled in responses, then captures a final frame and matches the live face against the document portrait.

Face matching runs on OpenCV's production stack: YuNet detection with SFace embeddings and the published cosine threshold. The two ONNX models are fetched once by `scripts/download_models.sh` (wired into the devcontainer and Dockerfile). Without them the system runs in an explicitly labeled demo mode: liveness still works, similarity is reported as indicative, and identity is never confirmed - measured tests showed classical features cannot separate same-person from impostor pairs reliably, so demo mode refuses to render a Verified verdict rather than render a false one.

Anti-spoofing scope, stated honestly: the randomized server-chosen challenge defeats printed photos, static image injections and pre-recorded video replays (which cannot respond to instructions chosen after recording); inter-frame motion checks reject identical injected stills. Real-time deepfake camera injection is out of scope for this MVP and would require attested capture or vendor-grade passive liveness models.

Biometric data handling: only face embeddings (not images) are retained server-side, in memory, for a 10-minute single-use session; nothing biometric is written to disk or logged. Review retention and consent obligations under Rwanda's Data Privacy and Protection Law before persisting any verification outcome.

## Performance and accuracy design

End-to-end processing runs in roughly 3 seconds on a single CPU core for a two-sided scan. The design choices that get there: front and back process in parallel (Tesseract runs as a subprocess, so threads give real concurrency, with OMP_THREAD_LIMIT=1 preventing core oversubscription); one OCR pass per side over a division-normalized image (dividing the card by a blurred copy of itself flattens the guilloche security pattern that otherwise wrecks recognition); a CLAHE fallback rendering runs only when the primary front read is weak; and orientation is settled by scoring real-word readability at 0 versus 180 degrees with an early exit when the captured orientation already reads strongly.

Accuracy comes from layering three sources per field and keeping the best: the full-page pass (label-anchored parsing), calibrated region OCR (single-line --psm 7 passes over each field's known zone on the rectified card, with digit whitelists where appropriate), and redundancy encoded in the 16-digit National ID number. The NID encodes the holder's birth year and sex, which the parser uses to cross-validate the printed fields, to repair partially degraded dates (a read of 03/12/189 with an NID year of 1989 is completed to 03/12/1989 and flagged for review), and to fall back to year-only when the printed date is unreadable. Every recovered or repaired value carries an explanatory note and a low-confidence flag so a human reviewer knows exactly what to verify.

## Extending

- **Add a field**: append a `FieldSpec` in `app/pipeline/parser.py` with the bilingual anchor labels and/or a regex plus an optional normalizer. No other code changes.
- **Swap OCR engines**: add a `_run_*` function in `app/pipeline/ocr.py` (PaddleOCR, Google Vision, Azure Document Intelligence) and register it in `_ENGINES`.
- **Future integrations** the module boundaries are designed for: MRZ reading, QR/barcode decoding (the back of newer cards), face verification against the extracted portrait, liveness detection, batch processing (the `/scan` endpoint is stateless and thread-pooled), and mobile clients via the same REST API.

## API

`POST /api/v1/scan` — multipart form fields `front` and `back` (JPG/JPEG/PNG/WEBP). Returns processed images (base64 JPEG), extracted fields with confidences, cross-validation results, portrait, timings, and the raw back-side text. `GET /api/v1/health` reports the active OCR engine and upload limit.

## Privacy note

Processed images are returned to the browser and are not persisted server-side. Log lines record field values at INFO level for debugging; set `LOG_LEVEL=WARNING` in production if extracted personal data must not reach the logs, and review retention obligations under Rwanda's Data Privacy and Protection Law before storing any scan output.
