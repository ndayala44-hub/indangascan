"""
Machine Readable Zone (TD3) parsing for passports.

The MRZ is two 44-character lines in OCR-B, designed for machine reading:

    Line 1: P<TYP>ISSUER + SURNAME<<GIVEN<NAMES<...
    Line 2: DOCNUM(9) C NAT(3) BIRTH(6) C SEX EXPIRY(6) C PERSONAL(14) C FINAL

where C are check digits computed over the preceding field with weights
7-3-1. Because every critical field is checksummed, a validated MRZ is the
most reliable data on the document — the parser treats it as authoritative
over the visual zone and uses it to detect and repair visual OCR errors.

OCR confusions (O<->0, I<->1, S<->5, B<->8, Z<->2, G<->6) are repaired
positionally: numeric positions get letter->digit substitution and vice
versa, then check digits arbitrate.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_CHAR_VALUES = {**{str(d): d for d in range(10)},
                **{chr(ord("A") + i): 10 + i for i in range(26)}, "<": 0}
_WEIGHTS = (7, 3, 1)

_TO_DIGIT = str.maketrans("OQDIJLZSBG", "0001112586")
_TO_ALPHA = str.maketrans("0125868", "OIZSBGB")  # best-effort inverse (rarely needed)


def _check_digit(data: str) -> str:
    total = sum(_CHAR_VALUES.get(c, 0) * _WEIGHTS[i % 3] for i, c in enumerate(data))
    return str(total % 10)


def _yymmdd(raw: str, kind: str) -> str | None:
    if not re.fullmatch(r"\d{6}", raw):
        return None
    yy, mm, dd = int(raw[:2]), int(raw[2:4]), int(raw[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    # Pivot: birth years 19xx unless clearly 20xx; expiry always 20xx.
    century = 2000 if (kind == "expiry" or yy <= 30) else 1900
    return f"{dd:02d}/{mm:02d}/{century + yy}"


@dataclass
class MrzResult:
    found: bool = False
    valid: bool = False
    lines: list[str] = field(default_factory=list)
    document_type: str | None = None
    issuing_country: str | None = None
    surname: str | None = None
    given_names: str | None = None
    passport_number: str | None = None
    nationality: str | None = None
    date_of_birth: str | None = None   # DD/MM/YYYY
    sex: str | None = None             # M / F
    date_of_expiry: str | None = None  # DD/MM/YYYY
    personal_number: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def check_summary(self) -> str:
        return ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in self.checks.items())


def _clean_line(raw: str) -> str:
    """Uppercase, strip spaces (OCR splits MRZ words), keep MRZ alphabet."""
    return re.sub(r"[^A-Z0-9<]", "", raw.upper().replace(" ", ""))


def find_mrz_lines(text: str) -> list[str]:
    """Pick the two TD3 lines out of free OCR text (lines dense with '<')."""
    candidates = []
    for raw in text.splitlines():
        cleaned = _clean_line(raw)
        if len(cleaned) >= 30 and cleaned.count("<") >= 4:
            candidates.append(cleaned)
    return candidates[-2:] if len(candidates) >= 2 else candidates


def _fit_line2(line2: str) -> str:
    """
    Reconcile line 2 to exactly 44 characters. OCR misreads of the '<'
    filler can add or drop characters, which would shift the trailing check
    digits out of position under blind truncation/padding. The first 28
    positions are fixed-format; the last two characters are the personal-
    number check and the final check; everything between is the 14-character
    personal-number field, trimmed (dropping K/R filler misreads first) or
    padded to fit.
    """
    if len(line2) == 44:
        return line2
    if len(line2) < 31:
        return (line2 + "<" * 44)[:44]
    prefix, tail = line2[:28], line2[28:]
    middle, checks = tail[:-2], tail[-2:]
    while len(middle) > 14 and re.search(r"[KRCE]", middle):
        middle = re.sub(r"[KRCE]", "", middle, count=1)
    while len(middle) > 14 and "<" in middle:
        middle = middle.replace("<", "", 1)
    middle = (middle + "<" * 14)[:14]
    return prefix + middle + checks


def parse_td3(line1: str, line2: str) -> MrzResult:
    result = MrzResult(found=True, lines=[line1, line2])

    line1 = (line1 + "<" * 44)[:44]
    line2 = _fit_line2(line2)

    # ---- Line 1: document type, issuer, names -----------------------------
    result.document_type = line1[0:2].replace("<", "")
    result.issuing_country = line1[2:5].translate(_TO_ALPHA) if any(
        c.isdigit() for c in line1[2:5]) else line1[2:5]
    names = line1[5:].rstrip("<")
    if "<<" in names:
        surname, given = names.split("<<", 1)
    else:
        surname, given = names, ""

    def _name_tokens(part: str) -> str | None:
        # Drop filler misreads: '<' runs OCR'd as repeated letters (KKK...)
        # or garbled runs drawn from the usual '<' confusion set (CCESSE...).
        tokens = []
        for t in part.split("<"):
            if not t or re.fullmatch(r"(.)\1{2,}", t):
                continue
            if len(t) >= 6 and set(t) <= set("CEKRS"):
                continue
            tokens.append(t)
        return " ".join(tokens).title() or None

    result.surname = _name_tokens(surname)
    result.given_names = _name_tokens(given)

    # ---- Line 2: fixed positions with check digits ------------------------
    doc_raw, doc_check = line2[0:9], line2[9]
    nationality = line2[10:13]
    birth_raw, birth_check = line2[13:19], line2[19]
    sex = line2[20]
    expiry_raw, expiry_check = line2[21:27], line2[27]
    personal_raw, personal_check = line2[28:42], line2[42]
    final_check = line2[43]

    # Positional repair: date fields must be digits.
    birth_fixed = birth_raw.translate(_TO_DIGIT)
    expiry_fixed = expiry_raw.translate(_TO_DIGIT)

    # Filler repair: '<' is frequently OCR'd as K (or R); if the personal-
    # number check fails, try substituting those back and keep what passes.
    if _check_digit(personal_raw) != personal_check.translate(_TO_DIGIT):
        import itertools
        confusable = "KRCE"  # letters OCR commonly produces for '<' filler
        for n in range(1, len(confusable) + 1):
            done = False
            for combo in itertools.combinations(confusable, n):
                sub = personal_raw
                for ch in combo:
                    sub = sub.replace(ch, "<")
                if _check_digit(sub) == personal_check.translate(_TO_DIGIT):
                    personal_raw = sub
                    done = True
                    break
            if done:
                break

    result.checks = {
        "document_number": _check_digit(doc_raw) == doc_check,
        "date_of_birth": _check_digit(birth_fixed) == birth_check.translate(_TO_DIGIT),
        "date_of_expiry": _check_digit(expiry_fixed) == expiry_check.translate(_TO_DIGIT),
        "personal_number": _check_digit(personal_raw) == personal_check.translate(_TO_DIGIT)
        or (personal_raw.strip("<") == "" and personal_check in "<0"),
        "final": _check_digit(
            doc_raw + doc_check + birth_fixed + birth_check + expiry_fixed
            + expiry_check + personal_raw + personal_check
        ) == final_check.translate(_TO_DIGIT),
    }

    result.passport_number = doc_raw.rstrip("<") or None
    result.nationality = nationality if not any(c.isdigit() for c in nationality) else None
    result.date_of_birth = _yymmdd(birth_fixed, "birth")
    result.sex = sex if sex in ("M", "F") else None
    result.date_of_expiry = _yymmdd(expiry_fixed, "expiry")
    personal = personal_raw.rstrip("<")
    result.personal_number = personal or None
    result.valid = all(result.checks.values())

    logger.info(
        "MRZ parsed",
        extra={"data": {"valid": result.valid, "checks": result.checks,
                        "passport_number": result.passport_number}},
    )
    return result


def parse_from_text(text: str) -> MrzResult:
    lines = find_mrz_lines(text)
    if len(lines) < 2:
        logger.info("MRZ not found in OCR text")
        return MrzResult(found=False)
    return parse_td3(lines[0], lines[1])
