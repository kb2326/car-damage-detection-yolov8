---
title: ClaimLens
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: other
short_description: Read-only showcase of an auditable AI claims-triage system
---

# ClaimLens: showcase

A **read-only showcase** of ClaimLens (the real web app's pages, rendered once from recorded claims), an AI system that triages car-damage insurance claims:
our own vision models measure the damage, an LLM agent reasons over the evidence and the policy
wording, and rules decide the route. Only a person can deny a claim.

**What you can see here:** five sample claims, recorded on 3 October 2026 with the real models and
agents, and one recorded intake chat. Open a claim to see its stage timeline, the damage the models
found (boxes on the photo), the evidence, the agent's reasoning with cited policy clauses, the rule
that decided the route, a person's review, and the verified, hash-chained audit log.

**What you cannot do:** nothing here can be filed, changed or reviewed. There are no live model or
LLM calls and no visitor data. To run the full system (chat intake, live agents, review), see the
repository.

- **Code, design and evaluations:** [github.com/kb2326/claimlens](https://github.com/kb2326/claimlens)
- **System card:** [docs/system-card.md](https://github.com/kb2326/claimlens/blob/main/docs/system-card.md)
- **Model card:** [docs/model-card.md](https://github.com/kb2326/claimlens/blob/main/docs/model-card.md)
- **Photo credits:** [showcase/CREDITS.md](https://github.com/kb2326/claimlens/blob/main/showcase/CREDITS.md)
  (Wikimedia Commons, CC BY / CC BY-SA)

Policies, insurers and claims are fictional. A non-commercial learning and portfolio project. The
damage model is trained on CarDD (non-commercial research licence); its weights are not published
and are not part of this Space.
