"""Saved agent runs, the owner's labelling page, and the labels it produces."""

from __future__ import annotations

import hashlib
import html
import json
import random
from collections.abc import Collection, Sequence
from pathlib import Path

from claimlens.evals.judge import JudgeItem, Judgement


def save_case(run_dir: Path, item: JudgeItem) -> None:
    """One JSON file per case: evidence, recommendation and cited clause text. No answer key."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{item.case_id}.json").write_text(item.model_dump_json(indent=1), encoding="utf-8")


def load_run(run_dir: Path) -> list[JudgeItem]:
    return [
        JudgeItem.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("*.json"))
        if path.name != "sample.json"
    ]


def _pick(
    pool: Sequence[str], k: int, narrative: Collection[str], want: int, rng: random.Random
) -> list[str]:
    stories = [i for i in pool if i in narrative]
    others = [i for i in pool if i not in narrative]
    rng.shuffle(stories)
    rng.shuffle(others)
    first = stories[: min(want, k)]
    rest = stories[len(first) :] + others
    rng.shuffle(rest)
    return first + rest[: k - len(first)]


def sample_for_labelling(
    judgements: Sequence[Judgement],
    narrative_ids: Collection[str],
    n: int = 50,
    seed: int = 20261002,
) -> list[str]:
    """Up to n case ids: half judge-fail if possible, at least 40% narrative where available."""
    rng = random.Random(seed)
    fails = sorted(j.case_id for j in judgements if j.verdict == "fail")
    passes = sorted(j.case_id for j in judgements if j.verdict == "pass")
    want = (2 * n) // 10  # narrative items per class
    chosen = _pick(fails, min(n // 2, len(fails)), narrative_ids, want, rng)
    chosen += _pick(passes, min(n - len(chosen), len(passes)), narrative_ids, want, rng)
    if len(chosen) < n:  # too few passes: top up with the remaining fails
        spare = [i for i in fails if i not in chosen]
        rng.shuffle(spare)
        chosen += spare[: n - len(chosen)]
    return sorted(chosen)


def load_labels(path: Path) -> dict[str, str]:
    """The owner's labels: case id -> pass (Good) or fail (Bad)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    for case_id, entry in data["labels"].items():
        label = entry["label"]
        if label not in ("pass", "fail"):
            raise ValueError(f"label for {case_id} must be pass or fail, not {label!r}")
        labels[case_id] = label
    return labels


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClaimLens labels</title>
<style>
:root{--paper:#f2f5f7;--surface:#fff;--ink:#15202b;--muted:#5a6875;--rule:#d3dbe1;
--good:#2a7a69;--bad:#b03838}
@media (prefers-color-scheme:dark){:root{--paper:#0e141a;--surface:#151d25;--ink:#e2e8ed;
--muted:#93a1ad;--rule:#2a3540;--good:#5fbfa8;--bad:#e57373}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif;
padding:16px}
main{max-width:860px;margin:auto;display:grid;gap:16px}
header{position:sticky;top:0;background:var(--paper);padding:8px 0;display:flex;flex-wrap:wrap;
gap:12px;align-items:center;border-bottom:1px solid var(--rule)}
.card{background:var(--surface);border:1px solid var(--rule);padding:14px 16px;display:grid;
gap:8px}
.card.good{border-left:4px solid var(--good)}.card.bad{border-left:4px solid var(--bad)}
pre{white-space:pre-wrap;margin:0;font:13px/1.5 ui-monospace,monospace;color:var(--muted)}
h2{font-size:16px;margin:0}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font:inherit;padding:6px 14px;border:1px solid var(--rule);background:var(--surface);
color:var(--ink);cursor:pointer}
button.on.g{background:var(--good);color:#fff}button.on.b{background:var(--bad);color:#fff}
button:disabled{opacity:.5;cursor:default}
input{font:inherit;padding:6px;border:1px solid var(--rule);background:var(--surface);
color:var(--ink);flex:1;min-width:180px}
</style></head><body><main>
<header><strong>Mark each recommendation Good or Bad</strong>
<span id="progress"></span>
<input id="labeller" placeholder="Your name">
<button id="download" disabled>Download labels</button></header>
<p>Good: the reasoning is true to the evidence, cites the right wording (or needs none), and an
adjuster could act on it. Bad: anything else. Judge the reasoning, not the route.</p>
__CARDS__
</main><script>
const KEY = "claimlens-labels-__KEY__";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
const cards = [...document.querySelectorAll(".card")];
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function render() {
  let done = 0;
  for (const card of cards) {
    const s = state[card.dataset.id] || {};
    card.classList.toggle("good", s.label === "pass");
    card.classList.toggle("bad", s.label === "fail");
    card.querySelector(".g").classList.toggle("on", s.label === "pass");
    card.querySelector(".b").classList.toggle("on", s.label === "fail");
    if (s.label) done++;
  }
  document.getElementById("progress").textContent = done + " of " + cards.length + " marked";
  document.getElementById("download").disabled = done < cards.length;
}
for (const card of cards) {
  const id = card.dataset.id;
  const note = card.querySelector("input");
  note.value = (state[id] || {}).note || "";
  const mark = (label) => () => { state[id] = {...state[id], label}; save(); render(); };
  card.querySelector(".g").onclick = mark("pass");
  card.querySelector(".b").onclick = mark("fail");
  note.oninput = () => { state[id] = {...state[id], note: note.value}; save(); };
}
document.getElementById("download").onclick = () => {
  const labels = {};
  for (const card of cards) {
    const s = state[card.dataset.id];
    labels[card.dataset.id] = {label: s.label, note: s.note || ""};
  }
  const name = document.getElementById("labeller").value || "owner";
  const blob = new Blob([JSON.stringify({labeller: name, labels}, null, 1)],
    {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "judge-labels.json";
  a.click();
};
render();
</script></body></html>
"""


def _card(n: int, item: JudgeItem) -> str:
    rec = item.recommendation
    esc = html.escape
    clauses = (
        "\n".join(f"{cid}: {text}" for cid, text in sorted(item.clauses.items())) or "(none cited)"
    )
    questions = "\n".join(f"- {q}" for q in rec.open_questions) or "(none)"
    return (
        f'<section class="card" data-id="{esc(item.case_id)}">'
        f"<h2>{n}. Case {esc(item.case_id)}</h2>"
        f"<pre>{esc(item.evidence)}</pre>"
        f"<div><strong>Cited wording</strong><pre>{esc(clauses)}</pre></div>"
        f"<div><strong>Recommendation:</strong> {esc(rec.route_suggestion.value)}, "
        f"{esc(rec.confidence.value)} confidence</div>"
        f"<div>{esc(rec.rationale)}</div>"
        f"<div><strong>Open questions</strong><pre>{esc(questions)}</pre></div>"
        '<div class="row"><button class="g">Good</button><button class="b">Bad</button>'
        '<input placeholder="Optional note"></div></section>'
    )


def render_label_page(items: Sequence[JudgeItem]) -> str:
    """A self-contained page: no network, no judge output, every text escaped."""
    cards = "\n".join(_card(n, item) for n, item in enumerate(items, start=1))
    # Labels are stored per export, so a new export never shows old labels on new items.
    key = hashlib.sha256("".join(i.model_dump_json() for i in items).encode()).hexdigest()[:12]
    return _PAGE.replace("__CARDS__", cards).replace("__KEY__", key)
