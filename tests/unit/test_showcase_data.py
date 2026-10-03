"""The committed showcase data (M8b): every claim verifies, every photo is credited."""

import json
import re
from pathlib import Path

from claimlens.events.store import SQLiteEventStore

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "showcase"


def test_every_sample_claim_verifies() -> None:
    store = SQLiteEventStore(DATA / "claims.db", read_only=True)
    try:
        claims = store.claim_ids()
        assert len(claims) == 5
        for claim in claims:
            assert store.load(claim)  # load verifies the hash chain
    finally:
        store.close()


def test_every_photo_is_credited_with_a_licence_and_source() -> None:
    credits = (DATA / "CREDITS.md").read_text(encoding="utf-8")
    blobs = sorted(p.name for p in (DATA / "blobs").iterdir())
    assert blobs
    for blob in blobs:
        (row,) = [line for line in credits.splitlines() if f"`{blob}`" in line]
        assert re.search(r"\[CC (BY|BY-SA|0)[^\]]*\]\(https?://", row), row
        assert "commons.wikimedia.org" in row


def test_no_published_photo_comes_from_a_restricted_source() -> None:
    credits = (DATA / "CREDITS.md").read_text(encoding="utf-8").lower()
    for word in ("cardd", "roboflow", "legacy", "course"):
        assert word not in credits


def test_the_transcript_is_a_plain_chat() -> None:
    transcript = json.loads((DATA / "transcript.json").read_text(encoding="utf-8"))
    assert len(transcript) >= 6
    assert {turn["role"] for turn in transcript} == {"agent", "customer"}
    assert all(set(turn) == {"role", "text"} for turn in transcript)
