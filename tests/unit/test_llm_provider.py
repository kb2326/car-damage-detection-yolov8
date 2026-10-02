import pytest

from claimlens.llm.provider import FakeProvider, ProviderReply, ProviderTransientError
from claimlens.llm.types import Message


def test_fake_provider_replays_its_script_and_records_calls() -> None:
    fake = FakeProvider([ProviderTransientError("overloaded"), ProviderReply("hi", 10, 2)])
    with pytest.raises(ProviderTransientError):
        fake.complete("m", "", [Message(role="user", content="x")], 5)
    assert fake.complete("m", "sys", [Message(role="user", content="x")], 5).text == "hi"
    assert [c["model"] for c in fake.calls] == ["m", "m"]
    assert fake.calls[1]["system"] == "sys"


def test_fake_provider_runs_out_loudly() -> None:
    with pytest.raises(AssertionError, match="script"):
        FakeProvider([]).complete("m", "", [Message(role="user", content="x")], 5)
