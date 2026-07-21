"""
Passport field extraction (Rwandan passport bio-data page).

The page carries two zones:

- VIZ (Visual Inspection Zone): the printed, human-readable fields with
  quadrilingual labels (Kinyarwanda / English / French / Swahili).
- MRZ (Machine Readable Zone): two checksummed OCR-B lines.

Strategy: extract everything from the VIZ with the same label-anchored
FieldSpec machinery used for the National ID, then let a check-digit-valid
MRZ override the fields it covers (passport number, names, date of birth,
sex, expiry, nationality). MRZ-verified values are the most reliable data
on the document; VIZ-only fields (place of birth, date of issue, issuing
authority, place of issue) stay with their OCR confidence and the usual
low-confidence flagging.
"""

import logging
import re
from typing import Any

from app.config import settings
from app.pipeline.mrz import MrzResult
from app.pipeline.ocr import OcrResult
from app.pipeline.parser import (
    ExtractedField,
    FieldSpec,
    _best_across_variants,
    _norm_name,
)

logger = logging.getLogger(__name__)

MRZ_CONFIDENCE = 99.0  # check-digit-verified values

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    # French variants that differ in the first three letters
    "FEV": 2, "AVR": 4, "MAI": 5, "JUI": 6, "AOU": 8, "DEC": 12,
}

VIZ_DATE_PATTERN = re.compile(r"\b[O0-9]{1,3}\s+([A-Z]{3})[A-Z]*(?:\s*/\s*[A-Z]+)?\s+((?:19|20)\d{2})\b")
PASSPORT_NO_PATTERN = re.compile(r"\bPC\s?\d{6,7}\b")


def _norm_viz_date(raw: str) -> str | None:
    """'06 JUL/JUIL 2022' (with O/0 confusion in the day) -> 06/07/2022."""
    m = VIZ_DATE_PATTERN.search(raw.upper())
    if not m:
        return None
    day_raw = re.match(r"\s*([O0-9]{1,3})", raw.upper().strip()).group(1)
    day_digits = day_raw.replace("O", "0")
    try:
        day = int(day_digits)
    except ValueError:
        return None
    month = _MONTHS.get(m.group(1))
    year = int(m.group(2))
    if not month or not (1 <= day <= 31):
        return None
    return f"{day:02d}/{month:02d}/{year}"


def _norm_upper_text(raw: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z\-/ ]", " ", raw)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -/")
    tokens = cleaned.split()
    while len(tokens) > 1 and len(tokens[0]) <= 2:  # leading OCR junk
        tokens = tokens[1:]
    cleaned = " ".join(tokens)
    return cleaned.upper() if len(cleaned) >= 3 else None


def _strip_trailing_date(raw: str) -> str:
    """
    Column merging puts a date after the value ('RUSIZI-NKANKA 05 JUL...');
    a mis-anchored label can make the whole value a date. Strip both.
    """
    out = re.sub(r"\s+[O0-9]{1,3}\s+[A-Z]{3}.*$", "", raw.upper()).strip()
    if VIZ_DATE_PATTERN.search(out) or re.match(r"^[O0-9]{1,3}\s+[A-Z]{3}", out):
        return ""
    return out


PASSPORT_FIELD_SPECS: list[FieldSpec] = [
    FieldSpec(
        key="passport_number", label="Passport Number", side="front",
        anchor_labels=["Passport No", "No. ya Pasiporo", "Nambari ya Pasipoti"],
        pattern=PASSPORT_NO_PATTERN, required=True,
    ),
    FieldSpec(
        key="surname", label="Surname", side="front",
        anchor_labels=["Surname", "Izina", "Nom"],
        normalizer=lambda raw: _norm_name(re.sub(r"\b[MF]\b\s*$", "", raw)),
        required=True,
    ),
    FieldSpec(
        key="given_names", label="Given Names", side="front",
        anchor_labels=["Other names", "Andi mazina", "Prenoms"],
        normalizer=_norm_name, required=True,
    ),
    FieldSpec(
        key="nationality", label="Nationality", side="front",
        anchor_labels=["Nationality", "Ubwenegihugu", "Nationalite"],
        pattern=re.compile(r"UMUNYARWANDA\s*/?\s*RWANDAN|RWANDAN", re.IGNORECASE),
        normalizer=_norm_upper_text,
    ),
    FieldSpec(
        key="date_of_birth", label="Date of Birth", side="front",
        anchor_labels=["Date of birth", "amavuko", "naissance"],
        pattern=VIZ_DATE_PATTERN, normalizer=_norm_viz_date, required=True,
    ),
    FieldSpec(
        key="sex", label="Sex", side="front",
        anchor_labels=["Igitsina", "Sex", "Jinsia"],
        pattern=re.compile(r"\b[MF]\b"),
    ),
    FieldSpec(
        key="place_of_birth", label="Place of Birth", side="front",
        anchor_labels=["Place of birth", "Aho yavukiye", "Lieu de naissance"],
        normalizer=lambda raw: _norm_upper_text(_strip_trailing_date(raw)),
    ),
    FieldSpec(
        key="date_of_issue", label="Date of Issue", side="front",
        anchor_labels=["Date of issue", "itangiweho", "delivrance", "kutolewa"],
        pattern=VIZ_DATE_PATTERN, normalizer=_norm_viz_date,
    ),
    FieldSpec(
        key="date_of_expiry", label="Date of Expiry", side="front",
        anchor_labels=["Date of expiry", "izarangiriraho", "expiration"],
        pattern=VIZ_DATE_PATTERN, normalizer=_norm_viz_date, required=True,
    ),
    FieldSpec(
        key="issuing_authority", label="Issuing Authority", side="front",
        anchor_labels=["Issuing authority", "Uyitanzeho", "Autorite de delivrance"],
        pattern=re.compile(r"GOVERNMENT\s+OF\s+RWANDA", re.IGNORECASE),
        normalizer=_norm_upper_text,
    ),
    FieldSpec(
        key="place_of_issue", label="Place of Issue", side="front",
        anchor_labels=["Place of issue", "Lieu de delivrance", "ilipotolewa"],
        pattern=re.compile(r"\bKIGALI\b", re.IGNORECASE),
        normalizer=lambda raw: _norm_upper_text(_strip_trailing_date(raw)),
    ),
    FieldSpec(
        key="personal_number", label="Personal Number", side="front",
        anchor_labels=["Personal No", "Personal number"],
    ),
]

# Fields a check-digit-valid MRZ overrides, mapped to (MrzResult attribute,
# the specific check that must pass, display formatter).
_MRZ_AUTHORITY: list[tuple[str, str, str]] = [
    ("passport_number", "passport_number", "document_number"),
    ("surname", "surname", "final"),
    ("given_names", "given_names", "final"),
    ("date_of_birth", "date_of_birth", "date_of_birth"),
    ("sex", "sex", "final"),
    ("date_of_expiry", "date_of_expiry", "date_of_expiry"),
    ("personal_number", "personal_number", "personal_number"),
]


def parse_passport(front_ocrs: list[OcrResult], mrz_result: MrzResult) -> dict[str, Any]:
    fields: dict[str, ExtractedField] = {
        spec.key: _best_across_variants(spec, front_ocrs) for spec in PASSPORT_FIELD_SPECS
    }

    # Disambiguate the three VIZ dates: OCR column-merging can attach the
    # wrong date to a label. With a valid MRZ, birth and expiry are known,
    # so the issue date must be the remaining VIZ date.
    validations: list[dict[str, Any]] = []
    if mrz_result.found:
        for check, passed in mrz_result.checks.items():
            validations.append({"check": f"mrz_{check}_check_digit", "passed": passed})

        for field_key, mrz_attr, check_name in _MRZ_AUTHORITY:
            mrz_value = getattr(mrz_result, mrz_attr)
            if mrz_value is None or not mrz_result.checks.get(check_name, False):
                continue
            viz = fields[field_key]
            if viz.value and _comparable(viz.value) == _comparable(str(mrz_value)):
                note = "Verified against the MRZ check digits (matches the printed value)."
                validations.append({"check": f"viz_matches_mrz_{field_key}", "passed": True})
            else:
                note = "Taken from the checksummed MRZ."
            fields[field_key] = ExtractedField(
                field_key, fields[field_key].label, str(mrz_value), MRZ_CONFIDENCE,
                low_confidence=False, notes=note, raw=str(mrz_value),
            )

        if mrz_result.nationality == "RWA" and mrz_result.checks.get("final"):
            fields["nationality"] = ExtractedField(
                "nationality", "Nationality", "RWANDAN (RWA)", MRZ_CONFIDENCE,
                notes="Verified against the MRZ.", raw="RWA",
            )

        # Issue date sanity: it can't equal birth or expiry.
        issue = fields["date_of_issue"]
        if issue.value and issue.value in (mrz_result.date_of_birth, mrz_result.date_of_expiry):
            issue.value = None
            issue.confidence = 0.0
        if not issue.value:
            candidate = _remaining_viz_date(front_ocrs, mrz_result)
            if candidate:
                fields["date_of_issue"] = ExtractedField(
                    "date_of_issue", "Date of Issue", candidate[0], candidate[1],
                    low_confidence=candidate[1] < settings.low_confidence_threshold,
                    notes="Identified as the visual-zone date that is neither the MRZ birth nor expiry date.",
                    raw=candidate[0],
                )

    missing_required = [
        s.label for s in PASSPORT_FIELD_SPECS if s.required and not fields[s.key].value
    ]
    logger.info(
        "Passport parsing complete",
        extra={"data": {"extracted": {k: f.value for k, f in fields.items()},
                        "mrz_valid": mrz_result.valid, "missing_required": missing_required}},
    )

    display_order = [
        "passport_number", "surname", "given_names", "nationality", "date_of_birth",
        "sex", "place_of_birth", "date_of_issue", "date_of_expiry",
        "issuing_authority", "place_of_issue", "personal_number",
    ]
    return {
        "fields": [fields[k].as_dict() for k in display_order],
        "mrz": {
            "found": mrz_result.found,
            "valid": mrz_result.valid,
            "lines": mrz_result.lines,
            "checks": mrz_result.checks,
        },
        "validations": validations,
        "missing_required": missing_required,
    }


def _comparable(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _remaining_viz_date(front_ocrs: list[OcrResult], mrz_result: MrzResult) -> tuple[str, float] | None:
    for ocr in front_ocrs:
        for line in ocr.lines:
            for m in VIZ_DATE_PATTERN.finditer(line.text.upper()):
                normalized = _norm_viz_date(m.group(0))
                if normalized and normalized not in (mrz_result.date_of_birth, mrz_result.date_of_expiry):
                    return normalized, line.confidence
    return None
