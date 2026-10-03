"""Three scripted customers talk to the real intake agent (Haiku). Opt-in: CLAIMLENS_LIVE=1.

The customer side is a simple keyword responder, so a run costs a few cents. It checks that the
agent collects every required fact, asks for the photos, and records driving_for_work correctly.
"""

import os
import random
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(os.environ.get("CLAIMLENS_LIVE") != "1", reason="live LLM test")

CUSTOMERS = {
    "simple": {
        "policy": "P-1001",
        "story": "I reversed into a bollard in the Tesco car park yesterday. No one else involved.",
        "when": "yesterday",
        "where": "the Tesco car park",
        "hit": "a bollard",
        "work": "no",
    },
    "vague": {
        "policy": "P-1002",
        "story": "Somebody's car and mine touched, I think. Not sure exactly.",
        "when": "2 days ago",
        "where": "outside my house",
        "hit": "another car",
        "work": "no",
    },
    "work": {
        "policy": "P-1005",
        "story": "I was delivering parcels for my courier job and clipped a post.",
        "when": "today",
        "where": "a delivery stop on the high street",
        "hit": "a post",
        "work": "yes",
    },
}


def _photo(path: Path) -> Path:
    rng = random.Random(str(path))
    image = Image.new("RGB", (640, 480))
    image.putdata([(rng.randrange(256),) * 3 for _ in range(640 * 480)])
    image.save(path)
    return path


def _answer(question: str, c: dict[str, str]) -> str:
    q = question.lower()
    rules = [
        (("policy",), c["policy"]),
        (("work", "business", "deliver", "paid", "job"), c["work"]),
        (("when", "date", "day"), c["when"]),
        (("where", "location"), c["where"]),
        (("hit", "collide", "object", "into"), c["hit"]),
        (("injur", "hurt"), "no"),
        (("who was driving", "driving", "driver", "you driving"), "yes, I was driving"),
        (
            ("anyone else", "other", "another", "third"),
            "no" if c["hit"] != "another car" else "yes",
        ),
        (("police",), "none"),
    ]
    for words, reply in rules:
        if any(w in q for w in words):
            return reply
    return c["story"]


@pytest.mark.parametrize("name", list(CUSTOMERS))
def test_a_scripted_customer_completes_intake(name: str, tmp_path: Path) -> None:
    from claimlens.events.store import SQLiteEventStore
    from claimlens.intake_agent.session import build_intake
    from tests.fakes import FakeDetector, make_test_deps

    root = Path(__file__).resolve().parents[2]
    customer = CUSTOMERS[name]
    store = SQLiteEventStore(tmp_path / "claims.db")
    deps = make_test_deps(tmp_path, store, FakeDetector())
    filed: list[str] = []
    from claimlens.llm.factory import build_gateway

    # Evaluation-style run: raise the daily cap for this test only (the session cap still applies).
    gateway = build_gateway(root / "config", root, per_day_usd=25.0)
    sessions = build_intake(
        root / "config", root, tmp_path / "intake.sqlite", lambda: deps, gateway=gateway
    )
    turn = sessions.start()
    for n in range(25):
        if turn.claim_id is not None:
            filed.append(turn.claim_id)
            break
        if turn.photo_kind is not None:
            turn = sessions.reply(turn.session_id, "here", _photo(tmp_path / f"{n}.png"))
        else:
            turn = sessions.reply(turn.session_id, _answer(turn.message, customer))
    assert filed, f"{name}: no claim after 25 turns; last message: {turn.message}"
    from uuid import UUID

    from claimlens.events.projection import fold

    state = fold(store.load(UUID(filed[0])))
    store.close()
    assert state.intake is not None
    facts = state.intake.facts
    print(name, state.intake.turns, "turns", facts)
    assert facts["policy_id"] == customer["policy"]
    assert facts["driving_for_work"] == customer["work"]
    assert set(state.intake.photo_kinds) == {"overview", "damage_closeup", "plate"}
