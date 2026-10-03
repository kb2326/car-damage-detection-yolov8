"""The facts the intake agent collects, their validation, and what is still missing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, timedelta

FACTS: dict[str, str] = {
    "policy_id": "the policy number, e.g. P-1001 (check it with lookup_policy)",
    "incident_date": "when it happened: a date, 'today', 'yesterday' or 'N days ago'",
    "location": "where it happened, in a few words",
    "what_happened": "the customer's own account, 1 to 3 sentences",
    "object_hit": "what the car hit or was hit by (another car, a post, nothing...)",
    "other_party_involved": "yes or no: was another person or vehicle involved",
    "driver_is_policyholder": "yes or no: was the policyholder driving",
    "driving_for_work": "yes or no: was the car being used for work, deliveries or paid trips",
    "injuries": "yes or no: was anyone hurt",
    "police_report": "a police report reference, or 'none' (only for theft, vandalism or injury)",
}
YES_NO = {"other_party_involved", "driver_is_policyholder", "driving_for_work", "injuries"}
NEVER_UNKNOWN = {"policy_id", "what_happened"}
REQUIRED = tuple(name for name in FACTS if name != "police_report")
MAX_TEXT = 600
_DAYS_AGO = re.compile(r"^(\d{1,3}) days? ago$")
_POLICE_WORDS = re.compile(r"theft|stolen|vandal|injur", re.IGNORECASE)


def _date(value: str, today: date) -> str:
    text = value.strip().lower()
    if text == "today":
        found = today
    elif text == "yesterday":
        found = today - timedelta(days=1)
    elif match := _DAYS_AGO.match(text):
        found = today - timedelta(days=int(match.group(1)))
    else:
        try:
            found = date.fromisoformat(text)
        except ValueError:
            raise ValueError(
                "incident_date must be a date (YYYY-MM-DD), 'today', 'yesterday' or "
                "'N days ago'; ask the customer for the date"
            ) from None
    if found > today or found.year < 2000:
        raise ValueError(f"incident_date {found} is in the future or before 2000")
    return found.isoformat()


def validate_fact(name: str, value: str, today: date) -> str:
    """The normalised value, or ValueError with a message the agent can act on."""
    if name not in FACTS:
        raise ValueError(f"unknown fact {name!r}; known facts: {', '.join(FACTS)}")
    text = " ".join(str(value).split())
    if not text:
        raise ValueError(f"{name} is empty")
    if text.lower() == "unknown":
        if name in NEVER_UNKNOWN:
            raise ValueError(f"{name} cannot be unknown; ask the customer again")
        return "unknown"
    if name == "incident_date":
        return _date(text, today)
    if name in YES_NO:
        answer = text.lower()
        if answer in ("yes", "y", "true"):
            return "yes"
        if answer in ("no", "n", "false"):
            return "no"
        raise ValueError(f"{name} must be yes or no (or unknown)")
    return text[:MAX_TEXT]


def missing(
    facts: Mapping[str, str],
    photos: Mapping[str, str],
    gaps: Mapping[str, str],
    kinds: Sequence[str],
) -> list[str]:
    """What the intake still needs: required facts, then photo kinds not received or given up."""
    needed = [f"fact: {name}" for name in REQUIRED if name not in facts]
    police = facts.get("injuries") == "yes" or bool(
        _POLICE_WORDS.search(facts.get("what_happened", ""))
    )
    if police and "police_report" not in facts:
        needed.append("fact: police_report")
    needed += [f"photo: {kind}" for kind in kinds if kind not in photos and kind not in gaps]
    return needed
