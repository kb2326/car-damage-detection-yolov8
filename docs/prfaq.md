# PR/FAQ: ClaimLens

*Working Backwards document. The press release describes the product on its intended launch date,
2027-01-15. It is written before building so that every decision can be checked against it. All
companies, people and quotes below are fictional.*

---

## Press release

### ClaimLens fast-tracks simple car-damage claims in minutes and shows adjusters exactly why

**Insurers can settle routine claims the same day, while every denial and every unusual case stays
with a person.**

**January 15, 2027.** Today we are releasing ClaimLens, an AI claims-triage system for vehicle
damage. A driver reports a loss in a short chat and takes the photos ClaimLens asks for.
Within minutes, ClaimLens identifies which parts are damaged and how badly, checks the photos for
signs of fraud, estimates a repair cost range, checks the policy, and routes the claim. Simple,
covered, low-value claims are fast-tracked. Everything else goes to an adjuster or the fraud team,
with the evidence already assembled.

**The problem.** After an accident, drivers wait days for someone to look at their photos. Adjusters
spend much of that time on simple claims that follow the same pattern, which leaves less time for
the complex ones. Decisions also vary from one adjuster to the next, and when AI is involved,
nobody can say exactly why it decided what it did.

**The solution.** ClaimLens separates the work the way a good claims team does. Purpose-built vision
models *measure* the damage. An AI assistant *reasons* about the evidence and the policy and writes
its recommendation with sources. Fixed, auditable business rules *decide* the route. ClaimLens can
fast-track a claim or escalate it, but it can never deny one. Every step is written to a
tamper-evident log, so any decision can be replayed and explained.

> "We didn't want an AI that sounds confident. We wanted one that shows its evidence and knows when
> to hand a claim to a person. ClaimLens tells our adjusters what it saw, which policy clause
> applies, and why it escalated." *(Illustrative quote, fictional head of claims)*

**How it works.**
1. The driver describes what happened in a chat, and ClaimLens guides them to the right photos.
2. Photos are checked for quality, and faces and licence plates are blurred.
3. Vision models find each damaged part, the type of damage, and its size.
4. Integrity checks look for reused, edited or AI-generated photos, and for a story that does not
   match the damage.
5. A cost range is estimated from the parts and the damage.
6. The AI assistant reviews the evidence against the policy and recommends a route, with citations.
7. Business rules make the final routing decision, and people handle every review and denial.

> "I sent four photos from the parking lot, and by the time I got home my claim was approved and a
> repair shop was booked." *(Illustrative quote, fictional driver)*

**Get started.** Try the public demo, read the design and evaluation reports, or run ClaimLens
locally from the open-source repository.

---

## External FAQ (drivers and insurers)

**Can ClaimLens deny my claim?**
No. It can fast-track a claim or send it to a person. Only a person can deny a claim.

**What happens if the AI isn't sure?**
The claim goes to an adjuster. Low model confidence, missing photos, unclear coverage, a high cost
estimate and any fraud signal all lead to human review.

**What happens to my photos and personal data?**
Faces and licence plates are blurred before any AI model reads the photos. Original photos are
stored encrypted with restricted access, and personal data can be deleted on request.

**How accurate is it?**
Accuracy is published per damage type, along with test results for routing, the AI assistant and
security. The targets are a damage-segmentation mAP50 of at least 0.50 on held-out photos, and
escalating at least 95% of the claims that need a person.

**Which damage does it recognise?**
Dents, scratches, cracks, shattered glass, broken lamps and flat tyres, and which car part each one
is on.

---

## Internal FAQ (team and stakeholders)

**Why not just send the photos to a large vision-language model?**
VLMs are weak at precise measurement, their answers are hard to reproduce, and "the model said so"
cannot justify a payout in a regulated industry. We use our own models for measurement and the LLM
for reasoning and language.

**Why is most of the system not an agent?**
Predictable steps are cheaper, faster, testable and auditable as plain code. The LLM is used only
for conversation, judgment and explanation (Anthropic, *Building Effective Agents*).

**What are the biggest risks?**
1. Dataset licensing (CarDD terms are confirmed before any public release of trained weights).
2. Photo manipulation and prompt injection (covered by the OWASP agentic threat model and a
   red-team suite).
3. Over-trust: adjusters accepting recommendations without checking (the UI shows confidence and
   counter-evidence, and decisions are audited by sampling).
4. Synthetic test claims not reflecting real claims (limitations are documented, adversarial cases
   are included, and a human-labelled subset is kept).

**How do we know it works?**
Every component has an evaluation with a target, and CI blocks changes that make the scores worse.
See section 13 of the design spec.

**What does it cost to run?**
The target is at most $0.05 of LLM spend per claim and a p95 latency of at most 15 seconds on CPU,
excluding time spent waiting for people.

**What are we deliberately not building?**
Automatic denials or payments, agent swarms, memory that agents write for themselves, LLM
fine-tuning, voice intake, and heavy infrastructure (Kubernetes, Kafka, microservices).

**How does this meet regulatory expectations?**
It follows the NAIC Model Bulletin on insurers' use of AI (an AIS Program document, human
oversight, an audit trail), and it applies EU AI Act high-risk-style controls voluntarily.

**What does success look like on launch day?**
A public demo; published data, model and system cards; evaluation results that meet the targets
in section 3 of the spec; and a nine-part LinkedIn series documenting the build.
