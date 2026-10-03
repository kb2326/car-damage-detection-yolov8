"""Build showcase/: the recorded sample claims for the public read-only demo (M8b).

Runs once on the owner's machine with the real damage and part models and the real LLM agents
(about $0.15). Photos are openly licensed Wikimedia Commons images of damaged cars (CC BY or
CC BY-SA), credited in showcase/CREDITS.md. Never uses CarDD or unknown-source photos.

    uv run python scripts/build_showcase.py --llm-daily-cap 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from uuid import UUID

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "showcase"
AGENT = "ClaimLens-portfolio/1.0 (github.com/kb2326/claimlens)"
PHOTOS = {
    "mirror": "File:2010-03-08 Shattered side mirror on BMW.jpg",
    "altima": "File:Altima-Accident.jpg",
    "front-crash": "File:Autounfall1.JPG",
    "van-1": "File:20171118 damaged Hyundai Grand starex-1.jpg",
    "van-2": "File:20171118 damaged Hyundai Grand starex-2.jpg",
}
# Number plates readable in the photos, blurred before use (left, top, right, bottom) at 1280 px.
PLATES = {"van-1": (40, 560, 112, 636), "van-2": (1032, 576, 1164, 674)}
CUSTOMER = {
    "policy": "P-1001",
    "story": "Bricks from a damaged wall fell onto my parked van and dented the roof and bonnet.",
    "when": "yesterday",
    "where": "outside my office on Station Road",
    "hit": "falling bricks",
}


def _fetch(titles: dict[str, str], folder: Path) -> dict[str, dict[str, str]]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": "|".join(titles.values()),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "1280",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
            "format": "json",
        }
    )
    request = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{query}", headers={"User-Agent": AGENT}
    )
    pages = json.load(urllib.request.urlopen(request))["query"]["pages"].values()
    by_title = {p["title"]: p for p in pages}
    credits: dict[str, dict[str, str]] = {}
    for key, title in titles.items():
        info = by_title[title]["imageinfo"][0]
        meta = info["extmetadata"]
        path = folder / f"{key}.jpg"
        image = urllib.request.Request(info["thumburl"], headers={"User-Agent": AGENT})
        path.write_bytes(urllib.request.urlopen(image).read())
        credits[key] = {
            "title": title.removeprefix("File:"),
            "page": info["descriptionurl"],
            "author": re.sub(r"<[^>]+>", "", meta["Artist"]["value"]).strip(),
            "license": meta["LicenseShortName"]["value"],
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        }
    return credits


def _answer(question: str) -> str:
    q = question.lower()
    rules = [  # the most specific questions first: "when this happened" appears in many
        (("what happened", "own words", "describe", "circumstances"), CUSTOMER["story"]),
        (("policy",), CUSTOMER["policy"]),
        (
            ("driving", "driver", "drive"),
            "nobody was driving, it was parked; I am the policyholder",
        ),
        (("work", "business", "deliver", "paid", "job"), "no, it was parked, not being used"),
        (("injur", "hurt"), "no"),
        (("police",), "none"),
        (("anyone else", "another person", "another vehicle", "other party", "third"), "no"),
        (("where", "location"), CUSTOMER["where"]),
        (("hit", "collide", "object", "cause", "damaged by"), CUSTOMER["hit"]),
        (("when", "date", "what day"), CUSTOMER["when"]),
    ]
    for words, reply in rules:
        if any(w in q for w in words):
            return reply
    return CUSTOMER["story"]


def _args(db: Path, blobs: Path, cap: float) -> argparse.Namespace:
    from claimlens.cli import _fastembedder, _llm_agent

    return argparse.Namespace(
        db=db,
        blobs=blobs,
        config=ROOT / "config",
        detector=None,
        weights=None,
        agent="llm",
        memory=True,
        llm_daily_cap=cap,
        agent_factory=_llm_agent,
        embedder_factory=_fastembedder,
    )


def build(cap: float) -> None:
    from claimlens.blobs import BlobStore
    from claimlens.cli import _default_detector, _make_agent, _make_detector, make_deps
    from claimlens.events.payloads import HumanReviewed, ReviewAction
    from claimlens.events.store import SQLiteEventStore
    from claimlens.intake import submit_claim
    from claimlens.intake_agent.config import load_intake_config
    from claimlens.intake_agent.graph import IntakeState
    from claimlens.intake_agent.session import build_intake, file_intake_claim
    from claimlens.knowledge.fastembedder import FastEmbedder
    from claimlens.memory.index import ClaimMemory
    from claimlens.review_queue import record_review
    from claimlens.workflow import process_claim

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "blobs").mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix="showcase-"))
    photos = work / "photos"
    photos.mkdir()
    credits = _fetch(PHOTOS, photos)
    # A re-saved copy of the mirror photo: what a fraudster sends with a second claim.
    with Image.open(photos / "mirror.jpg") as image:
        image.convert("RGB").save(photos / "mirror-resaved.jpg", "JPEG", quality=60)
    for key, box in PLATES.items():  # privacy: no readable number plate is published
        path = photos / f"{key}.jpg"
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.paste(image.crop(box).filter(ImageFilter.GaussianBlur(14)), box[:2])
        image.save(path, "JPEG", quality=92)
    for file in photos.iterdir():  # blobs are named by content hash: remember which is which
        _HASHES[hashlib.sha256(file.read_bytes()).hexdigest()] = file.stem

    db = OUT / "claims.db"
    blobs = BlobStore(OUT / "blobs")
    args = _args(db, OUT / "blobs", cap)
    store = SQLiteEventStore(db)
    memory = ClaimMemory(work / "memory", FastEmbedder())
    detector = _make_detector(args, _default_detector)
    deps = make_deps(store, blobs, args.config, detector, _make_agent(args, db), memory)

    def claim(policy: str, story: str, photo: str) -> UUID:
        claim_id = submit_claim(
            store, blobs, policy_id=policy, description=story, photo_paths=[photos / photo]
        )
        decision = process_claim(claim_id, deps)
        print(claim_id, decision.route.value, decision.rule_id)
        return claim_id

    claim(
        "P-1002",
        "A van passing in a narrow street clipped my side mirror and shattered it.",
        "mirror.jpg",
    )
    altima = claim(
        "P-1001",
        "Another car ran a red light and hit the front of mine at a junction.",
        "altima.jpg",
    )
    claim(
        "P-1003",
        "I braked too late in traffic and went into the back of a lorry. The front is badly "
        "damaged.",
        "front-crash.jpg",
    )
    fraud = claim(
        "P-1005", "Someone broke my side mirror in the supermarket car park.", "mirror-resaved.jpg"
    )
    record_review(
        store,
        altima,
        HumanReviewed(
            reviewer="Showcase adjuster", action=ReviewAction.APPROVE, note="Repair quote matches."
        ),
        memory=memory,
        photo_path=blobs.path,
    )
    record_review(
        store,
        fraud,
        HumanReviewed(
            reviewer="Showcase investigator",
            action=ReviewAction.DENY,
            note="The photo is a re-saved copy of an earlier claim's photo on another policy.",
        ),
        memory=memory,
        photo_path=blobs.path,
    )

    config = load_intake_config(args.config / "intake.toml")

    def submit(state: IntakeState) -> str:
        return str(file_intake_claim(store, blobs, state, config))

    from claimlens.llm.factory import build_gateway

    gateway = build_gateway(args.config, ROOT, per_day_usd=cap)  # the owner's cap for this build
    sessions = build_intake(
        args.config, ROOT, work / "intake.sqlite", lambda: deps, gateway=gateway, submit=submit
    )
    turn = sessions.start()
    sent = 0
    for _ in range(30):
        if turn.claim_id:
            break
        if turn.photo_kind:
            # Two photos of the van; the plate request gets the first again (a duplicate in one
            # claim is set aside by the pipeline, not treated as fraud).
            photo = photos / f"van-{1 if sent != 1 else 2}.jpg"
            turn = sessions.reply(turn.session_id, "", photo)
            sent += 1
        else:
            turn = sessions.reply(turn.session_id, _answer(turn.message))
    assert turn.claim_id, "the recorded chat did not finish"
    chat_claim = UUID(turn.claim_id)
    process_claim(chat_claim, deps)
    record_review(
        store,
        chat_claim,
        HumanReviewed(
            reviewer="Showcase adjuster",
            action=ReviewAction.REQUEST_INFO,
            note="Please send a photo of the number plate.",
        ),
        memory=memory,
        photo_path=blobs.path,
    )
    saved = sessions.values(turn.session_id)
    transcript: list[dict[str, str]] = []
    for exchange in saved["transcript"]:
        transcript.append({"role": "agent", "text": exchange["agent"]})
        said = exchange["customer"] or ""
        if exchange.get("photo"):
            said = (said + " [photo sent]").strip()
        transcript.append({"role": "customer", "text": said or "[no reply]"})
    transcript.append({"role": "agent", "text": str(saved.get("outgoing", ""))})
    text = json.dumps(transcript, indent=1) + "\n"
    (OUT / "transcript.json").write_text(text, encoding="utf-8")
    store.close()
    sqlite3.connect(db).execute("VACUUM").connection.close()
    _write_credits(credits, blobs)
    shutil.rmtree(work, ignore_errors=True)


def _write_credits(credits: dict[str, Any], blobs: Any) -> None:
    lines = [
        "# Showcase photo credits",
        "",
        "Photos of damaged cars from Wikimedia Commons, used under their licences. Each is",
        "Wikimedia's 1280-px-wide version of the original; changes are noted per photo. The damage",
        "boxes are drawn by the page on top. Each blob is named by the SHA-256 of its bytes.",
        "",
        "| Blob | Photo | Author | Licence | Source |",
        "|---|---|---|---|---|",
    ]
    for blob in sorted(p.name for p in (OUT / "blobs").iterdir()):
        key = _key_for(blob)
        c = credits["mirror"] if key == "mirror-resaved" else credits[key]
        note = " (re-saved as a lower-quality JPEG)" if key == "mirror-resaved" else ""
        if key in PLATES:
            note = " (number plate blurred)"
        lines.append(
            f"| `{blob}` | {c['title']}{note} | {c['author']} | "
            f"[{c['license']}]({c['license_url']}) | {c['page']} |"
        )
    (OUT / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


_HASHES: dict[str, str] = {}


def _key_for(blob: str) -> str:
    return _HASHES[blob.split(".")[0]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-daily-cap", type=float, default=15.0)
    build(parser.parse_args().llm_daily_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
