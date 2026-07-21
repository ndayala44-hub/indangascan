"""
Structured field extraction for the Rwandan National Identity Card.

The card is bilingual (Kinyarwanda / English). The front of the current-
generation card carries:

    REPUBULIKA Y'U RWANDA / REPUBLIC OF RWANDA
    INDANGAMUNTU / IDENTITY CARD
    Amazina / Names
    Itariki yavutseho / Date of Birth
    Igitsina / Sex                (Gabo / M   or   Gore / F)
    Aho Yatangiwe / Place of Issue
    Indangamuntu / National ID No   e.g.  1 1989 8 0031866 1 85

The 16-digit National ID number itself encodes useful redundancy that we
exploit for cross-validation:

    [1]      1=citizen, 2=refugee, 3=foreign resident
    [2-5]    year of birth
    [6]      8=male, 7=female
    [7-13]   birth registration sequence
    [14]     issue count digit
    [15-16]  security digits

Extraction is spec-driven: FIELD_SPECS below is plain data, so adding a
field (or supporting a future card revision) means adding an entry, not
rewriting logic. Each extracted field carries the OCR confidence of the
line(s) it came from so the UI can flag low-confidence values for review.
"""

import logging
import re
from dataclasses import dataclass, field as dc_field
from difflib import SequenceMatcher
from typing import Any, Callable

from app.config import settings
from app.pipeline.ocr import OcrResult

logger = logging.getLogger(__name__)

NID_PATTERN = re.compile(r"\b([123])\s*((?:19|20)\d{2})\s*([78])\s*(\d{7})\s*(\d)\s*(\d{2})\b")
DATE_PATTERN = re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*((?:19|20)\d{2})\b")


@dataclass
class ExtractedField:
    key: str
    label: str
    value: str | None
    confidence: float
    low_confidence: bool = False
    notes: str | None = None
    raw: str = ""  # pre-normalization text, used for cross-variant quality scoring

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "confidence": round(self.confidence, 1),
            "low_confidence": self.low_confidence,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_label_line(ocr: OcrResult, labels: list[str]) -> tuple[int, str, float] | None:
    """
    Locate the line containing one of the (bilingual) labels, tolerating OCR
    noise via containment + fuzzy matching. Returns (line_index, label_hit, score).
    """
    best: tuple[int, str, float] | None = None
    for idx, line in enumerate(ocr.lines):
        text = line.text.lower()
        for label in labels:
            lab = label.lower()
            if lab in text:
                return idx, label, 1.0
            # Fuzzy: compare label against every window of the same word length.
            words = text.split()
            span = len(lab.split())
            for start in range(0, max(1, len(words) - span + 1)):
                window = " ".join(words[start : start + span])
                score = _similar(window, lab)
                if score >= 0.70 and (best is None or score > best[2]):
                    best = (idx, label, score)
    return best


def _looks_like_label_junk(remainder: str, labels: list[str]) -> bool:
    """
    On noisy scans the label itself OCRs imperfectly (e.g. 'Amazina' ->
    'Arnar@e'), leaving residue after stripping. If the remainder resembles
    any label word, it is residue, not a value.
    """
    tokens = {t for lab in labels for t in re.split(r"[\s/]+", lab) if len(t) > 2}
    for word in remainder.split():
        if any(_similar(word, t) > 0.55 for t in tokens):
            return True
    return False


def _value_after_label(ocr: OcrResult, labels: list[str]) -> tuple[str, float] | None:
    """
    Field values on the Rwandan card sit either after the label on the same
    line or on the line immediately below it.
    """
    hit = _find_label_line(ocr, labels)
    if hit is None:
        return None
    idx, label, _ = hit
    line = ocr.lines[idx]

    # Strip the bilingual label (and slash separators) from the line.
    remainder = line.text
    for token in re.split(r"\s*/\s*", label):
        remainder = re.sub(re.escape(token), "", remainder, flags=re.IGNORECASE)
    remainder = remainder.strip(" :/|-.\u2022")

    if len(remainder) >= 2 and not _looks_like_label_junk(remainder, labels):
        return remainder, line.confidence
    if idx + 1 < len(ocr.lines):
        nxt = ocr.lines[idx + 1]
        return nxt.text.strip(), nxt.confidence
    return None


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #

def _norm_date(raw: str) -> str | None:
    m = DATE_PATTERN.search(raw)
    if not m:
        return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mth <= 12):
        return None
    return f"{d:02d}/{mth:02d}/{y}"


def _norm_sex(raw: str) -> str | None:
    t = raw.lower()
    if "gabo" in t or re.search(r"\bm\b", t):
        return "Male (Gabo / M)"
    if "gore" in t or re.search(r"\bf\b", t):
        return "Female (Gore / F)"
    return None


def _name_quality(raw: str) -> float:
    """
    The card prints the surname in capitals followed by mixed-case given
    names ("NDAYISHIMIYE Alain"). A raw value matching that shape is almost
    certainly the real name rather than OCR residue from the label line.
    """
    return 1.0 if re.search(r"[A-Z']{3,}\s+[A-Z][a-z']{2,}", raw) else 0.0


def _norm_name(raw: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z'\- ]", " ", raw)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned.title() if len(cleaned) >= 3 else None


def _norm_place(raw: str) -> str | None:
    # Sex and Place of Issue share a line on the card ("Gabo / M  Kicukiro /
    # Niboye"); drop the sex fragment before cleaning.
    t = re.sub(r".*?\b(gabo|gore)\b\s*/?\s*[MF]?\b", "", raw, flags=re.IGNORECASE)
    return _norm_name(t if t.strip() else raw)


# --------------------------------------------------------------------------- #
# Field specifications (configurable)
# --------------------------------------------------------------------------- #

@dataclass
class FieldSpec:
    key: str
    label: str
    side: str                                   # "front" | "back"
    anchor_labels: list[str] = dc_field(default_factory=list)
    pattern: re.Pattern | None = None           # regex over the full OCR text
    normalizer: Callable[[str], str | None] | None = None
    quality: Callable[[str], float] | None = None  # scores the RAW value across variants
    required: bool = False


FIELD_SPECS: list[FieldSpec] = [
    FieldSpec(
        key="full_name", label="Full Name (Amazina)", side="front",
        anchor_labels=["Amazina / Names", "Amazina", "Names"],
        normalizer=_norm_name, quality=_name_quality, required=True,
    ),
    FieldSpec(
        key="date_of_birth", label="Date of Birth (Itariki yavutseho)", side="front",
        anchor_labels=["Itariki yavutseho / Date of Birth", "Itariki yavutseho", "Date of Birth"],
        pattern=DATE_PATTERN, normalizer=_norm_date, required=True,
    ),
    FieldSpec(
        key="sex", label="Sex (Igitsina)", side="front",
        anchor_labels=["Igitsina / Sex", "Igitsina", "Sex"],
        normalizer=_norm_sex, required=True,
    ),
    FieldSpec(
        key="place_of_issue", label="Place of Issue (Aho Yatangiwe)", side="front",
        anchor_labels=["Aho Yatangiwe / Place of Issue", "Aho Yatangiwe", "Place of Issue"],
        normalizer=_norm_place,
    ),
    FieldSpec(
        key="national_id_number", label="National ID Number (Indangamuntu)", side="front",
        anchor_labels=["Indangamuntu / National ID No", "National ID No", "Indangamuntu"],
        pattern=NID_PATTERN, required=True,
    ),
    # The reverse side carries the holder's signature, the card serial and
    # issue metadata; the generic back-extraction below captures all printed
    # lines, and this spec pulls out the machine-readable serial if present.
    FieldSpec(
        key="card_serial", label="Card Serial (Back)", side="back",
        pattern=re.compile(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{8,16}\b"),
    ),
]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _extract_field(spec: FieldSpec, ocr: OcrResult) -> ExtractedField:
    value: str | None = None
    raw_value = ""
    confidence = 0.0

    # 1) Label-anchored lookup (most precise).
    if spec.anchor_labels:
        hit = _value_after_label(ocr, spec.anchor_labels)
        if hit:
            raw, confidence = hit
            raw_value = raw
            if spec.pattern:
                # A pattern-bearing field must actually match — this rejects
                # false anchors (e.g. the card header also says "Indangamuntu").
                m = spec.pattern.search(raw)
                value = m.group(0) if m else None
                if value and spec.normalizer:
                    value = spec.normalizer(value)
            else:
                value = spec.normalizer(raw) if spec.normalizer else raw

    # 2) Pattern search across the full text (fallback / label-less fields).
    if value is None and spec.pattern:
        for line in ocr.lines:
            m = spec.pattern.search(line.text)
            if m:
                raw = m.group(0)
                raw_value = raw
                value = spec.normalizer(raw) if spec.normalizer else raw
                confidence = line.confidence
                break

    low = confidence < settings.low_confidence_threshold
    return ExtractedField(spec.key, spec.label, value, confidence,
                          low_confidence=low and value is not None, raw=raw_value)


def _decode_nid(nid_raw: str) -> dict[str, str]:
    """Decode the redundancy encoded in the 16-digit National ID number."""
    m = NID_PATTERN.search(nid_raw)
    if not m:
        return {}
    status = {"1": "Rwandan citizen", "2": "Refugee", "3": "Foreign resident"}.get(m.group(1), "Unknown")
    return {
        "formatted": f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)} {m.group(5)} {m.group(6)}",
        "holder_status": status,
        "birth_year": m.group(2),
        "sex_digit": "Male" if m.group(3) == "8" else "Female",
    }


def _best_across_variants(spec: FieldSpec, variants: list[OcrResult]) -> ExtractedField:
    """
    Extract the field from every OCR preprocessing variant and keep the best
    valid value, ranked by (spec quality of the raw text, OCR confidence).
    """
    def rank(f: ExtractedField) -> tuple[float, float]:
        q = spec.quality(f.raw) if (spec.quality and f.value) else 0.0
        return (q, f.confidence)

    best: ExtractedField | None = None
    for ocr in variants:
        cand = _extract_field(spec, ocr)
        if cand.value and (best is None or not best.value or rank(cand) > rank(best)):
            best = cand
        elif best is None:
            best = cand
    return best


def parse_card(front_ocrs: list[OcrResult], back_ocrs: list[OcrResult]) -> dict[str, Any]:
    """
    Run every field spec against the appropriate side across all OCR
    preprocessing variants, derive surname / given names, cross-validate
    against the encoded ID number, and return a JSON-serializable result
    including the raw back-side text.
    """
    ocrs_by_side = {"front": front_ocrs, "back": back_ocrs}
    fields: dict[str, ExtractedField] = {
        spec.key: _best_across_variants(spec, ocrs_by_side[spec.side]) for spec in FIELD_SPECS
    }
    back_ocr = max(back_ocrs, key=lambda r: r.mean_confidence)

    # Derived: Rwandan convention places the surname first on the Names line.
    name = fields["full_name"]
    surname = given = None
    if name.value:
        parts = name.value.split()
        surname, given = parts[0], " ".join(parts[1:]) or None
    fields["surname"] = ExtractedField("surname", "Surname", surname, name.confidence, name.low_confidence)
    fields["given_names"] = ExtractedField(
        "given_names", "Given Names", given, name.confidence, name.low_confidence
    )

    # Nationality is implied by the document + holder-status digit.
    nid = fields["national_id_number"]
    decoded = _decode_nid(nid.value) if nid.value else {}
    fields["nationality"] = ExtractedField(
        "nationality", "Nationality / Holder Status",
        decoded.get("holder_status", "Rwandan (implied by document)") if nid.value else None,
        nid.confidence, nid.low_confidence,
        notes="Derived from the first digit of the National ID number." if decoded else None,
    )
    if nid.value and decoded:
        nid.value = decoded["formatted"]

    # Cross-validation: DOB year and sex vs. what the ID number encodes.
    validations: list[dict[str, Any]] = []
    dob, sex = fields["date_of_birth"], fields["sex"]
    if decoded and not dob.value:
        # The printed date is unreadable, but the NID encodes the birth year.
        dob.value = f"{decoded['birth_year']} (year only)"
        dob.confidence = nid.confidence
        dob.low_confidence = True
        dob.notes = "Printed date unreadable; year recovered from the National ID number. Verify manually."
    elif decoded and dob.value:
        ok = dob.value.endswith(decoded["birth_year"])
        validations.append({"check": "birth_year_matches_id_number", "passed": ok})
        if not ok:
            dob.low_confidence = True
            dob.notes = f"Year disagrees with ID number (encodes {decoded['birth_year']}). Review."
    if decoded and sex.value:
        ok = decoded["sex_digit"].lower() in sex.value.lower()
        validations.append({"check": "sex_matches_id_number", "passed": ok})
        if not ok:
            sex.low_confidence = True
            sex.notes = f"Disagrees with ID number (encodes {decoded['sex_digit']}). Review."

    missing_required = [
        s.label for s in FIELD_SPECS if s.required and not fields[s.key].value
    ]
    logger.info(
        "Parsing complete",
        extra={
            "data": {
                "extracted": {k: f.value for k, f in fields.items()},
                "missing_required": missing_required,
                "validations": validations,
            }
        },
    )

    display_order = [
        "national_id_number", "full_name", "surname", "given_names", "date_of_birth",
        "sex", "nationality", "place_of_issue", "card_serial",
    ]
    return {
        "fields": [fields[k].as_dict() for k in display_order if k in fields],
        "back_text_lines": [
            {"text": l.text, "confidence": round(l.confidence, 1)} for l in back_ocr.lines
        ],
        "validations": validations,
        "missing_required": missing_required,
    }
