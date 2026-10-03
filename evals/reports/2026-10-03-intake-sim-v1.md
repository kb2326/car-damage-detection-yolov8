# Intake evaluation: simulated customers

- Date: 2026-10-03
- Trials per customer: 4
- pass@1: **0.85**  pass^4: **0.80** (target 0.70)
- Cost of this run: $4.35

| Customer | Passed | All passed | Mean turns | What went wrong |
|---|---|---|---|---|
| coop-courier | 4 of 4 | yes | 8.8 | - |
| coop-bollard | 4 of 4 | yes | 12.5 | - |
| coop-other-car | 4 of 4 | yes | 11.0 | - |
| vague-scrape | 4 of 4 | yes | 12.0 | - |
| vague-dent | 4 of 4 | yes | 11.8 | - |
| oversharer-injury | 4 of 4 | yes | 5.8 | - |
| changer-pole | 1 of 4 | no | 10.0 | incident_date: got '2026-09-30', expected '2026-09-28' |
| changer-gate | 1 of 4 | no | 11.5 | handed over: spending cap reached; incident_date: got '2026-10-01', expected '2026-09-23' |
| photo-trouble-kerb | 4 of 4 | yes | 12.8 | - |
| photo-trouble-trolley | 4 of 4 | yes | 14.0 | - |
