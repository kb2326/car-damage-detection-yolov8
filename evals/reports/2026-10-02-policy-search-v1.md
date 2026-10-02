# Policy search v1

- Date: 2026-10-02
- Questions: 10
- **recall@5: 1.00** (target ≥ 0.90)

| Question | Expected | Top 5 | Found |
|---|---|---|---|
| My car is in the shop after a crash. Will you pay for a hire car, and for how long? | STD-6.1 | STD-6.1, STD-8.5, STD-7.1, STD-1.1, STD-2.2 | yes |
| Do I get a replacement car while mine is being fixed? | BAS-6.1 | BAS-6.1, BAS-1.1, BAS-4.1, BAS-2.2, BAS-9.2 | yes |
| A stone chipped my windshield. Do I have to pay anything to get it fixed? | PRM-4.1 | PRM-6.1, PRM-4.1, PRM-8.5, PRM-9.2, PRM-3.1 | yes |
| Who pays for the tow truck after an accident, and is there a limit? | STD-7.1 | STD-7.1, STD-2.2, STD-3.1, STD-2.1, STD-9.3 | yes |
| I damaged the car during a track day. Am I covered? | STD-8.1 | STD-8.1, STD-6.1, STD-9.2, STD-8.5, STD-1.1 | yes |
| My car was stolen. Do I need to go to the police? | STD-5.1 | STD-9.3, STD-5.1, STD-6.1, STD-1.1, STD-9.2 | yes |
| How much do I pay myself when my side window is smashed? | BAS-4.1 | BAS-4.1, BAS-2.2, BAS-6.1, BAS-3.1, BAS-7.1 | yes |
| What happens if someone lies on their claim or inflates the damage? | STD-10.1 | STD-5.2, STD-10.1, STD-8.4, STD-8.2, STD-9.1 | yes |
| Is damage to my own car from hitting a pole included in this policy? | BAS-2.1 | BAS-6.1, BAS-2.1, BAS-8.3, BAS-1.1, BAS-4.1 | yes |
| How long do I have to tell you about an accident? | PRM-9.1 | PRM-9.1, PRM-9.3, PRM-9.2, PRM-6.1, PRM-3.1 | yes |

## Notes

- Embedder: `BAAI/bge-small-en-v1.5` (fastembed, ONNX on CPU); index: LanceDB hybrid (BM25 + vector, RRF).
- 56 clauses in 3 wordings; each query is filtered to the policy's wording (17 to 20 clauses).
- The expected clause was ranked first for 6 questions and second for 4.
- With so few clauses per wording, recall@5 is a generous bar. recall@1 is 0.60.
