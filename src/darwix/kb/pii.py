"""PII identification, redaction and retrieval blocking.

The assessment asks to "identify and protect" PII. Identifying it is the easy
half. The protection is the design decision:

* PII-bearing records stay in the KB (they are real business data and the lead
  desk needs them) but are marked `retrieval_allowed = False`, so the voice
  agent physically cannot surface another customer's phone number no matter
  what it is asked. The block is enforced in the index query, not in a prompt.
* The stored `content` is the redacted text. The original is never copied into
  the KB, so a leak of the KB file is not a leak of customer data.
* PAN and Aadhaar are validated, not just pattern-matched. `ABCDE1234F` is a
  PAN shape; a 12-digit number is only an Aadhaar if it passes the Verhoeff
  checksum. Validating cuts false positives on invoice and account numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- patterns ---------------------------------------------------------------
PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_in": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    "phone_intl": re.compile(r"(?<!\d)\+(?:62|63)[\s-]?\d[\d\s-]{7,12}(?!\d)"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "aadhaar": re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
    "bank_account": re.compile(r"\b(?:a/?c|account)\s*(?:no\.?|number)?\s*[:#]?\s*(\d{9,18})\b", re.I),
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "gstin": re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b"),
    "pincode_address": re.compile(r"\b\d{6}\b(?=[^\d]*(?:india|road|street|nagar|colony))", re.I),
}

# Numbers that look like identifiers but are not personal data. Without this,
# every rupee amount and every year is a false positive.
SAFE_CONTEXT = re.compile(
    r"(?:INR|rs\.?|₹|turnover|amount|limit|crore|lakh|version|clause|section|page)\s*$",
    re.I,
)

MASKS = {
    "email": "[EMAIL_REDACTED]",
    "phone_in": "[PHONE_REDACTED]",
    "phone_intl": "[PHONE_REDACTED]",
    "pan": "[PAN_REDACTED]",
    "aadhaar": "[AADHAAR_REDACTED]",
    "bank_account": "[ACCOUNT_REDACTED]",
    "ifsc": "[IFSC_REDACTED]",
    "gstin": "[GSTIN_REDACTED]",
    "pincode_address": "[PINCODE_REDACTED]",
    "person_name": "[NAME_REDACTED]",
}

# Verhoeff tables - Aadhaar's checksum algorithm.
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    digits = [int(c) for c in reversed(number) if c.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, d in enumerate(digits):
        c = _D[c][_P[i % 8][d]]
    return c == 0


def pan_valid(value: str) -> bool:
    # 4th char encodes holder type; 'P' individual, 'C' company, 'F' firm, etc.
    return len(value) == 10 and value[3] in "ABCFGHLJPTK"


@dataclass
class PIIFinding:
    kind: str
    value: str
    start: int
    end: int


@dataclass
class PIIResult:
    has_pii: bool
    kinds: list[str] = field(default_factory=list)
    findings: list[PIIFinding] = field(default_factory=list)
    redacted: str = ""

    @property
    def count(self) -> int:
        return len(self.findings)


def _validated(kind: str, value: str) -> bool:
    if kind == "aadhaar":
        return verhoeff_valid(re.sub(r"\D", "", value))
    if kind == "pan":
        return pan_valid(value)
    return True


def scan(text: str) -> PIIResult:
    findings: list[PIIFinding] = []
    for kind, rx in PATTERNS.items():
        for m in rx.finditer(text):
            value = m.group(0)
            if SAFE_CONTEXT.search(text[max(0, m.start() - 20):m.start()]):
                continue
            if not _validated(kind, value):
                continue
            findings.append(PIIFinding(kind, value, m.start(), m.end()))
    findings.sort(key=lambda f: f.start)

    redacted_parts: list[str] = []
    cursor = 0
    for f in findings:
        if f.start < cursor:  # overlapping match, keep the first
            continue
        redacted_parts.append(text[cursor:f.start])
        redacted_parts.append(MASKS.get(f.kind, "[REDACTED]"))
        cursor = f.end
    redacted_parts.append(text[cursor:])

    kinds = sorted({f.kind for f in findings})
    return PIIResult(
        has_pii=bool(findings),
        kinds=kinds,
        findings=findings,
        redacted="".join(redacted_parts),
    )


NAME_FIELD = re.compile(r"(full_name|applicant name|customer name|name)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})")


def redact_names_in_fields(text: str) -> tuple[str, int]:
    """Names are only reliably detectable where the source labels them.

    A general NER pass would catch more and also redact "Fair Practices Code"
    and every city name in the corpus. Field-anchored redaction is narrower and
    correct; the residual risk is documented rather than hidden.
    """
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        return m.group(1) + ": " + MASKS["person_name"]

    return NAME_FIELD.sub(repl, text), n


def scan_and_redact(text: str) -> PIIResult:
    result = scan(text)
    redacted, names = redact_names_in_fields(result.redacted)
    if names:
        result.redacted = redacted
        result.has_pii = True
        if "person_name" not in result.kinds:
            result.kinds = sorted(result.kinds + ["person_name"])
    return result
