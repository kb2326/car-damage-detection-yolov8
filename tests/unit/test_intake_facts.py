from datetime import date

import pytest

from claimlens.intake_agent.facts import FACTS, missing, validate_fact

TODAY = date(2026, 10, 3)
KINDS = ("overview", "damage_closeup", "plate")
COMPLETE = {
    "policy_id": "P-1001",
    "incident_date": "2026-10-02",
    "location": "Tesco car park",
    "what_happened": "Reversed into a bollard.",
    "object_hit": "a bollard",
    "other_party_involved": "no",
    "driver_is_policyholder": "yes",
    "driving_for_work": "no",
    "injuries": "no",
}
PHOTOS = {"overview": "a.jpg", "damage_closeup": "b.jpg", "plate": "c.jpg"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-01", "2026-10-01"),
        ("today", "2026-10-03"),
        ("Yesterday", "2026-10-02"),
        ("3 days ago", "2026-09-30"),
        ("unknown", "unknown"),
    ],
)
def test_dates_are_normalised(value: str, expected: str) -> None:
    assert validate_fact("incident_date", value, TODAY) == expected


@pytest.mark.parametrize("value", ["2026-12-01", "1999-12-31", "last spring"])
def test_bad_dates_are_errors(value: str) -> None:
    with pytest.raises(ValueError, match="incident_date"):
        validate_fact("incident_date", value, TODAY)


@pytest.mark.parametrize(("value", "expected"), [("Yes", "yes"), ("n", "no"), ("TRUE", "yes")])
def test_yes_no_facts(value: str, expected: str) -> None:
    assert validate_fact("driving_for_work", value, TODAY) == expected


def test_yes_no_rejects_other_words() -> None:
    with pytest.raises(ValueError, match="yes or no"):
        validate_fact("injuries", "maybe a bit", TODAY)


def test_unknown_names_and_required_unknowns_are_errors() -> None:
    with pytest.raises(ValueError, match="unknown fact"):
        validate_fact("favourite_colour", "blue", TODAY)
    with pytest.raises(ValueError, match="policy_id"):
        validate_fact("policy_id", "unknown", TODAY)
    with pytest.raises(ValueError, match="empty"):
        validate_fact("location", "   ", TODAY)


def test_text_is_trimmed_and_capped() -> None:
    assert validate_fact("location", "  car park  ", TODAY) == "car park"
    assert len(validate_fact("what_happened", "x" * 2000, TODAY)) == 600


def test_nothing_missing_when_complete() -> None:
    assert missing(COMPLETE, PHOTOS, {}, KINDS) == []
    assert set(FACTS) >= set(COMPLETE)


def test_missing_facts_and_photos_are_listed() -> None:
    facts = {k: v for k, v in COMPLETE.items() if k != "location"}
    assert missing(facts, {"overview": "a.jpg"}, {"plate": "too blurry"}, KINDS) == [
        "fact: location",
        "photo: damage_closeup",
    ]


def test_police_report_is_needed_for_theft_vandalism_or_injury() -> None:
    theft = {**COMPLETE, "what_happened": "My car was stolen and found damaged."}
    assert missing(theft, PHOTOS, {}, KINDS) == ["fact: police_report"]
    injured = {**COMPLETE, "injuries": "yes"}
    assert missing(injured, PHOTOS, {}, KINDS) == ["fact: police_report"]
    assert missing({**injured, "police_report": "none"}, PHOTOS, {}, KINDS) == []
