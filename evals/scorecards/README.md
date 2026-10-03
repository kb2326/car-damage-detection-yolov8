# Evaluation scorecards

CI has no API key and no model weights, so it cannot re-run the evaluation. It checks the
**committed scorecard** instead (`claimlens eval-gate`, ADR 0014).

- `current.json`: written by `claimlens eval-triage … --scorecard evals/scorecards/current.json`.
  It holds the metrics and a sha256 **fingerprint** of every file that can change the result
  (prompts, agent and LLM config, rules, rate card, models registry, taxonomy, policy wordings,
  the golden file). `claimlens judge agreement --scorecard …` adds the judge's pass rate once the
  judge agrees with the owner's labels.
- `baseline.json`: the last accepted scorecard. Accepting a new baseline is a deliberate commit:
  copy `current.json` over it in the same pull request and say why.

The gate fails when a fingerprinted file changed since the evaluation ran (re-run it), when
escalation recall or citation validity is below 1.00, when a metric falls more than its allowed
margin below the baseline, or when the mean cost per claim rises more than 25%. Thresholds:
`config/eval_gate.toml`.
