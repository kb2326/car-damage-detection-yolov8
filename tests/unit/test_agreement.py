import pytest

from claimlens.evals.agreement import compute_agreement


def _labels(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    return (
        {str(i): j for i, (j, _) in enumerate(pairs)},
        {str(i): h for i, (_, h) in enumerate(pairs)},
    )


def test_perfect_agreement() -> None:
    judge, human = _labels([("pass", "pass")] * 6 + [("fail", "fail")] * 4)
    a = compute_agreement(judge, human)
    assert (a.n, a.agreement, a.kappa) == (10, 1.0, 1.0)


def test_known_kappa() -> None:
    # 20 both pass, 5 judge pass / human fail, 10 judge fail / human pass, 15 both fail
    pairs = (
        [("pass", "pass")] * 20
        + [("pass", "fail")] * 5
        + [("fail", "pass")] * 10
        + [("fail", "fail")] * 15
    )
    a = compute_agreement(*_labels(pairs))
    assert a.agreement == pytest.approx(0.70)
    assert a.kappa == pytest.approx(0.40)  # po 0.70, pe 0.5*0.6 + 0.5*0.4 = 0.50
    assert (a.both_pass, a.judge_pass_human_fail, a.judge_fail_human_pass, a.both_fail) == (
        20,
        5,
        10,
        15,
    )


def test_kappa_is_zero_when_one_side_never_varies() -> None:
    a = compute_agreement(*_labels([("pass", "pass")] * 5 + [("pass", "fail")] * 5))
    assert a.kappa == 0.0


def test_mismatched_items_are_refused() -> None:
    with pytest.raises(ValueError, match="missing labels for: b"):
        compute_agreement({"a": "pass", "b": "fail"}, {"a": "pass"})
    with pytest.raises(ValueError, match="labels for items that were not exported: z"):
        compute_agreement({"a": "pass", "b": "fail"}, {"a": "pass", "b": "fail", "z": "pass"})
