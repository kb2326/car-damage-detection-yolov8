"""Agreement between the LLM judge and a human labeller (accuracy and Cohen's kappa)."""

from __future__ import annotations

from collections.abc import Mapping

from claimlens.domain import Frozen


class Agreement(Frozen):
    n: int
    agreement: float
    kappa: float
    both_pass: int
    judge_pass_human_fail: int
    judge_fail_human_pass: int
    both_fail: int


def compute_agreement(judge: Mapping[str, str], human: Mapping[str, str]) -> Agreement:
    missing, extra = sorted(set(judge) - set(human)), sorted(set(human) - set(judge))
    if missing:
        raise ValueError(f"missing labels for: {', '.join(missing)}")
    if extra:
        raise ValueError(f"labels for items that were not exported: {', '.join(extra)}")
    n = len(judge)
    if n < 2:
        raise ValueError("need at least 2 labelled items")
    pp = sum(judge[k] == "pass" and human[k] == "pass" for k in judge)
    pf = sum(judge[k] == "pass" and human[k] == "fail" for k in judge)
    fp = sum(judge[k] == "fail" and human[k] == "pass" for k in judge)
    ff = n - pp - pf - fp
    observed = (pp + ff) / n
    expected = ((pp + pf) / n) * ((pp + fp) / n) + ((fp + ff) / n) * ((pf + ff) / n)
    kappa = 0.0 if expected == 1.0 else (observed - expected) / (1 - expected)
    return Agreement(
        n=n,
        agreement=observed,
        kappa=round(kappa, 4),
        both_pass=pp,
        judge_pass_human_fail=pf,
        judge_fail_human_pass=fp,
        both_fail=ff,
    )
