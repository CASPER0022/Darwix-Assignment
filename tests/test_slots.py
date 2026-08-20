"""Slot parsing.

These are the conversions that decide a lending outcome, so they are tested
directly rather than trusted to the model that produced the raw string.
"""
import pytest

from darwix.voice.slots import (normalise, parse_amount, parse_bool, parse_entity_type,
                                parse_months, speak_amount_inr)


@pytest.mark.parametrize("raw,expected", [
    ("15 lakh", 1_500_000),
    ("15 lakhs", 1_500_000),
    ("Rs. 45,00,000", 4_500_000),
    ("1.2 crore", 12_000_000),
    ("two crore", 20_000_000),
    ("72 lakh rupees", 7_200_000),
    ("42 thousand", 42_000),
    ("₹ 8,50,000", 850_000),
    ("4.5 million", 4_500_000),
    # "no existing EMIs" is zero, not unknown - returning None here stranded
    # the slot and made the agent ask the same question twice
    ("no existing EMIs", 0),
    ("none", 0),
    ("nil", 0),
    ("", None),
    (None, None),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    # Indonesia. The id-ID pack asks for the instalment AS SPOKEN, so these are
    # the literal shapes the slot receives - all of which returned None before,
    # which made the agent ask for the instalment twice and then give up on it.
    ("satu juta dua ratus ribu", 1_200_000),
    ("dua ratus ribu", 200_000),
    ("tiga juta", 3_000_000),
    ("1.200.000", 1_200_000),          # '.' is the thousands separator here
    ("Rp 1.200.000 per bulan", 1_200_000),
    ("1.5 juta", 1_500_000),
    ("2,5 juta", 2_500_000),           # ',' is the decimal point here
    ("500 ribu", 500_000),
    ("dua miliar", 2_000_000_000),
    # Philippines. Same story in the fil-PH pack: "tatlong libo" is a premium.
    ("tatlong libo", 3_000),
    ("limang libo", 5_000),
    ("isang libo limang daan", 1_500),
    ("tatlong daan", 300),
    ("2500 pesos", 2_500),
    ("dalawang milyon", 2_000_000),
])
def test_parse_amount_id_and_ph(raw, expected):
    assert parse_amount(raw) == expected


def test_separator_conventions_do_not_collide():
    """The same characters mean opposite things by market, so both readings are
    pinned: an Indian grouping must not become an Indonesian one, or a 45 lakh
    turnover silently becomes 45."""
    assert parse_amount("45,00,000") == 4_500_000     # Indian grouping
    assert parse_amount("1.200.000") == 1_200_000     # Indonesian grouping
    assert parse_amount("4.5 million") == 4_500_000   # '.' as a decimal point
    assert parse_amount("2,5 juta") == 2_500_000      # ',' as a decimal point


@pytest.mark.parametrize("raw,expected", [
    ("6 years", 72),
    ("about 7 years now", 84),
    ("18 months", 18),
    ("3.5 years", 42),
    ("saade teen saal", 42),
    ("2 saal", 24),
])
def test_parse_months(raw, expected):
    assert parse_months(raw) == expected


def test_parse_months_from_start_year():
    from datetime import date
    months = parse_months("since 2019")
    expected = (date.today().year - 2019) * 12 + date.today().month
    assert months == expected


@pytest.mark.parametrize("raw,expected", [
    ("proprietorship", "proprietorship"),
    ("sole proprietor, just me", "proprietorship"),
    ("Pvt Ltd", "private_limited"),
    ("private limited company", "private_limited"),
    ("we are an LLP", "llp"),
    ("partnership firm", "partnership"),
    ("it's a trust", "trust"),
    ("no idea", None),
])
def test_parse_entity_type(raw, expected):
    assert parse_entity_type(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("yes", True), ("haan", True), ("opo", True), ("iya", True),
    ("no", False), ("nahi", False), ("belum", False), ("nggak", False),
    ("maybe later", None),
])
def test_parse_bool(raw, expected):
    assert parse_bool(raw) is expected


def test_normalise_reports_unparsed():
    slots, unparsed = normalise({
        "annual_turnover_inr": "72 lakhs",
        "business_vintage_months": "6 years",
        "entity_type": "something odd",
    })
    assert slots["annual_turnover_inr"] == 7_200_000
    assert slots["business_vintage_months"] == 72
    assert "entity_type" in unparsed
    assert "entity_type" not in slots


def test_normalise_ignores_unknown_keys():
    slots, _ = normalise({"favourite_colour": "blue", "city": "Surat"})
    assert slots == {"city": "Surat"}


@pytest.mark.parametrize("value,expected", [
    (1_500_000, "15 lakh rupees"),
    (12_000_000, "1.2 crore rupees"),
    (20_000_000, "2 crore rupees"),
    (42_000, "42 thousand rupees"),
])
def test_speak_amount_uses_indian_units(value, expected):
    # An SME owner in Ludhiana does not think in millions.
    assert speak_amount_inr(value) == expected
