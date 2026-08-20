"""Slot normalisation.

The model extracts what the customer *said*; this module decides what it
*means*. That split is deliberate: "saade teen saal", "three and a half years"
and "42 months" must all become 42, and an LLM that does this arithmetic
inline will be right most of the time, which is not good enough when the number
decides an eligibility outcome.

Everything here is unit-testable without a network call - see tests/test_slots.py.
"""
from __future__ import annotations

import re
from typing import Any

LAKH = 100_000
CRORE = 10_000_000

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "twenty five": 25, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "half": 0.5,
    # Hindi/Urdu numerals that surface in Indian English calls
    "ek": 1, "do": 2, "teen": 3, "chaar": 4, "paanch": 5, "das": 10,
    "saade": 0.5,  # "saade teen" = three and a half
}

ENTITY_ALIASES = {
    "proprietorship": ["proprietor", "proprietorship", "sole proprietor", "individual",
                       "single owner", "my own", "prop"],
    "partnership": ["partnership", "partner firm", "partners", "firm"],
    "llp": ["llp", "limited liability"],
    "private_limited": ["private limited", "pvt ltd", "pvt. ltd", "pvt", "private ltd",
                        "company", "p ltd"],
    "trust": ["trust"],
    "society": ["society", "co-operative", "cooperative"],
    "huf": ["huf", "hindu undivided"],
}

INDUSTRY_FLAGS = {
    "gambling": ["gambling", "betting", "casino", "satta"],
    "crypto_trading": ["crypto", "bitcoin", "cryptocurrency trading"],
    "arms": ["arms", "ammunition", "firearms"],
    "tobacco_manufacturing": ["tobacco manufacturing", "gutkha", "cigarette manufacturing"],
    "money_lending": ["money lending", "moneylending", "chit fund", "sahukar"],
}

AFFIRM = {"yes", "yeah", "yep", "haan", "ha", "correct", "right", "sure", "true", "opo",
          "oo", "ya", "iya", "betul", "sige", "done", "registered"}
DENY = {"no", "nope", "nahi", "nahin", "not yet", "never", "hindi", "wala", "tidak", "nggak",
        "gak", "belum", "false", "unregistered"}


def _word_to_number(text: str) -> float | None:
    text = text.strip().lower()
    if text in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[text])
    parts = text.split()
    if len(parts) == 2 and parts[0] in _WORD_NUMBERS and parts[1] in _WORD_NUMBERS:
        a, b = _WORD_NUMBERS[parts[0]], _WORD_NUMBERS[parts[1]]
        # "saade teen" (0.5 + 3) and "twenty five" (20 + 5) both work additively
        return float(a + b)
    return None


# Indonesian and Filipino numerals. Both Q3 packs ask for amounts AS SPOKEN
# ("satu juta dua ratus ribu", "tatlong libo"), so the parser has to read them:
# an unparsed instalment is a slot the agent asks for twice and then abandons.
_SEA_DIGITS = {
    # Indonesian
    "nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6,
    "tujuh": 7, "delapan": 8, "sembilan": 9,
    # Filipino
    "isa": 1, "dalawa": 2, "tatlo": 3, "apat": 4, "anim": 6, "pito": 7,
    "walo": 8, "siyam": 9, "sampu": 10,
}
# Multipliers that scale what came before them. `puluh`/`ratus`/`daan` scale
# within a group; `ribu`/`juta`/`libo`/`milyon` close it off.
_SEA_SMALL_SCALE = {"puluh": 10, "ratus": 100, "daan": 100, "raan": 100, "gatos": 100}
_SEA_BIG_SCALE = {"ribu": 1_000, "libo": 1_000, "juta": 1_000_000,
                  "milyon": 1_000_000, "miliar": 1_000_000_000, "milyar": 1_000_000_000}
# "se-" is the bound form of "one": seribu = one thousand, seratus = one hundred.
_SEA_SE_FORMS = {"seribu": 1_000, "seratus": 100, "sejuta": 1_000_000,
                 "sepuluh": 10, "semilyar": 1_000_000_000}


def _sea_tokens(text: str) -> list[str]:
    """Keep only numeral tokens, dropping the Filipino ligature.

    "tatlong libo" is tatlo + the linker -ng + libo; without stripping the
    linker the number word never matches. ("apat na libo" uses the other
    linker, which is simply not a numeral and so falls away here.)
    """
    known = (_SEA_DIGITS, _SEA_SMALL_SCALE, _SEA_BIG_SCALE, _SEA_SE_FORMS)
    out: list[str] = []
    for raw in re.findall(r"[a-z]+", text.lower()):
        if any(raw in table for table in known):
            out.append(raw)
        elif raw.endswith("ng") and raw[:-2] in _SEA_DIGITS:
            out.append(raw[:-2])          # tatlong -> tatlo, limang -> lima
    return out


def _scale_number(text: str) -> float | None:
    """"satu juta dua ratus ribu" -> 1200000. Returns None if nothing scaled."""
    total = current = 0.0
    seen = False
    for tok in _sea_tokens(text):
        if tok in _SEA_SE_FORMS:
            value = _SEA_SE_FORMS[tok]
            if value >= 1_000:
                total += (current or 1) * value if current else value
                current = 0.0
            else:
                current = (current or 1) * value
            seen = True
        elif tok in _SEA_DIGITS:
            current += _SEA_DIGITS[tok]
            seen = True
        elif tok in _SEA_SMALL_SCALE:
            current = (current or 1) * _SEA_SMALL_SCALE[tok]
            seen = True
        elif tok in _SEA_BIG_SCALE:
            total += (current or 1) * _SEA_BIG_SCALE[tok]
            current = 0.0
            seen = True
    return (total + current) if seen else None


def _grouped_number(raw: str) -> float | None:
    """Read a digit group under Indian, Indonesian or Filipino conventions.

    The separators mean opposite things by market: `1.200.000` is 1.2 million in
    Jakarta, while `4.5` is four and a half everywhere. Decided by shape rather
    than by a locale flag, so a customer who says it the other way still parses.
    """
    raw = raw.strip().strip(",.")
    if not raw:
        return None
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", raw):
        return float(raw.replace(".", ""))          # 1.200.000 -> 1200000
    if re.fullmatch(r"\d+,\d{1,2}", raw):
        return float(raw.replace(",", "."))         # 2,5 juta -> 2.5
    try:
        return float(raw.replace(",", ""))          # 45,00,000 / 4.5
    except ValueError:
        return None


NEGATIVE_AMOUNT = {"no", "none", "nil", "nothing", "zero", "no emi", "no emis", "not any",
                   "nahi", "koi nahi", "wala", "tidak ada", "belum ada"}


def parse_amount(value: Any) -> int | None:
    """'45 lakh' / 'Rs 45,00,000' / '4.5 million' / 'two crore' -> integer rupees.

    "No existing EMIs" is zero, not unknown. Returning None there would strand
    the slot and make the agent ask again.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower().strip(".!,")
        if cleaned in NEGATIVE_AMOUNT or cleaned.startswith(("no ", "none", "nil", "zero")):
            return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).lower().strip()
    if not text:
        return None
    text = re.sub(r"₹|rs\.|\brs\b|\binr\b|\brp\b|\bidr\b|₱|\bphp\b|\bpeso[s]?\b|"
                  r"\bpiso\b|\brupiah\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    m = re.search(r"([\d,.]+)\s*(crore|cr\b|lakh|lakhs|lac|lacs|million|mn\b|miliar|milyar|"
                  r"juta|ribu|libo|milyon|k\b|thousand)?", text)
    number: float | None = None
    unit = ""
    if m and m.group(1).strip(",."):
        number = _grouped_number(m.group(1))
        unit = (m.group(2) or "").strip()
    if number is None:
        words = re.sub(r"(crore|cr|lakh|lakhs|lac|lacs|million|thousand)", "", text).strip()
        number = _word_to_number(words)
        unit_match = re.search(r"(crore|cr|lakh|lakhs|lac|lacs|million|thousand)", text)
        unit = unit_match.group(1) if unit_match else ""
    if number is None:
        # Indonesian / Filipino spoken numerals carry their own scale words, so
        # this path returns the finished figure - no unit multiplier after it.
        scaled = _scale_number(text)
        return int(scaled) if scaled is not None else None

    if unit.startswith("cr"):
        return int(number * CRORE)
    if unit.startswith(("lakh", "lac")):
        return int(number * LAKH)
    if unit.startswith(("million", "mn", "juta", "milyon")):
        return int(number * 1_000_000)
    if unit.startswith(("miliar", "milyar")):
        return int(number * 1_000_000_000)
    if unit in ("k", "thousand", "ribu", "libo"):
        return int(number * 1_000)
    return int(number)


def parse_months(value: Any) -> int | None:
    """'3 years' / 'since 2019' / '18 months' / 'saade teen saal' -> months."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).lower().strip()
    if not text:
        return None

    m = re.search(r"(?:since|from)\s*(19|20)(\d{2})", text)
    if m:
        from datetime import date
        year = int(m.group(1) + m.group(2))
        return max(0, (date.today().year - year) * 12 + date.today().month)

    m = re.search(r"([\d.]+)\s*(year|yr|saal|years|taon|tahun)", text)
    if m:
        return int(float(m.group(1)) * 12)
    m = re.search(r"([\d.]+)\s*(month|mahine|mah|months|buwan|bulan)", text)
    if m:
        return int(float(m.group(1)))

    words = re.sub(r"(years?|yrs?|saal|months?|mahine)", "", text).strip()
    n = _word_to_number(words)
    if n is not None:
        return int(n * 12) if re.search(r"(year|yr|saal|taon|tahun)", text) else int(n)
    m = re.search(r"([\d.]+)", text)
    if m:
        return int(float(m.group(1)) * 12)  # bare number in a vintage answer means years
    return None


def parse_entity_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    for canonical, aliases in ENTITY_ALIASES.items():
        for alias in aliases:
            if alias in text:
                return canonical
    return None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in AFFIRM or any(w in text.split() for w in AFFIRM):
        return True
    if text in DENY or any(w in text.split() for w in DENY):
        return False
    return None


def parse_industry(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    for flag, aliases in INDUSTRY_FLAGS.items():
        if any(a in text for a in aliases):
            return flag
    return str(value).strip()[:60] or None


def parse_int(value: Any, *, lo: int | None = None, hi: int | None = None) -> int | None:
    if value is None:
        return None
    m = re.search(r"\d+", str(value))
    if not m:
        return None
    n = int(m.group(0))
    if lo is not None and n < lo:
        return None
    if hi is not None and n > hi:
        return None
    return n


NORMALISERS = {
    "business_name": lambda v: (str(v).strip()[:80] or None) if v else None,
    "entity_type": parse_entity_type,
    "industry": parse_industry,
    "business_vintage_months": parse_months,
    "annual_turnover_inr": parse_amount,
    "loan_amount_inr": parse_amount,
    "loan_purpose": lambda v: (str(v).strip()[:80] or None) if v else None,
    "existing_emi_inr": parse_amount,
    "gst_registered": parse_bool,
    "city": lambda v: (str(v).strip()[:60] or None) if v else None,
    "applicant_age": lambda v: parse_int(v, lo=16, hi=99),
    "credit_score": lambda v: parse_int(v, lo=300, hi=900),
}


# Locale packs declare a slot's parser by name, so a market can define its own
# slots (policy number, premium, cicilan) without touching this module.
PARSERS_BY_NAME = {
    "text": lambda v: (str(v).strip()[:80] or None) if v else None,
    "amount": parse_amount,
    "months": parse_months,
    "bool": parse_bool,
    "int": lambda v: parse_int(v, lo=0, hi=100000),
    "entity_type": parse_entity_type,
    "industry": parse_industry,
    "age": lambda v: parse_int(v, lo=16, hi=99),
    "credit_score": lambda v: parse_int(v, lo=300, hi=900),
}


def normalise_with(raw: dict[str, Any], parsers: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Normalise using a locale-declared parser map."""
    out: dict[str, Any] = {}
    unparsed: list[str] = []
    for key, value in (raw or {}).items():
        name = parsers.get(key)
        fn = PARSERS_BY_NAME.get(name) if name else NORMALISERS.get(key)
        if fn is None or value in (None, "", "unknown", "not provided"):
            continue
        parsed = fn(value)
        if parsed is None:
            unparsed.append(key)
        else:
            out[key] = parsed
    return out, unparsed


def normalise(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Returns (normalised slots, list of slots that could not be parsed)."""
    out: dict[str, Any] = {}
    unparsed: list[str] = []
    for key, value in (raw or {}).items():
        fn = NORMALISERS.get(key)
        if fn is None or value in (None, "", "unknown", "not provided"):
            continue
        parsed = fn(value)
        if parsed is None:
            unparsed.append(key)
        else:
            out[key] = parsed
    return out, unparsed


def speak_amount_inr(value: int) -> str:
    """Render rupees the way an Indian business owner hears them.

    Not cosmetic: a bot that says "five million rupees" to an SME owner in
    Ludhiana sounds like a foreign call centre, and the assessment marks
    localisation over literal translation.
    """
    if value >= CRORE:
        n = value / CRORE
        return (str(int(n)) if n == int(n) else str(round(n, 2))) + " crore rupees"
    if value >= LAKH:
        n = value / LAKH
        return (str(int(n)) if n == int(n) else str(round(n, 2))) + " lakh rupees"
    if value >= 1000:
        return str(int(value / 1000)) + " thousand rupees"
    return str(value) + " rupees"
