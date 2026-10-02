"""Pack the inputs of an auto-label job so it can run on a Colab GPU instead of this laptop.

The zip holds the dataset's records and splits plus only the sampled images, at the same paths
the job expects. Run from the repository root:

uv run python scripts/make_autolabel_bundle.py
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from claimlens.autolabel.job import load_autolabel_jobs, select_sample
from claimlens.data.records import read_records

JOB = "fusion-eval-v1"
OUT = Path("var/colab/autolabel-bundle.zip")


def main() -> int:
    job = load_autolabel_jobs(Path("config/autolabel.toml"))[JOB]
    interim = Path("data/interim") / job.dataset
    records = read_records(interim / "records.jsonl")
    splits: dict[str, str] = json.loads((interim / "splits.json").read_text(encoding="utf-8"))
    sample = select_sample(records, splits, split=job.split, size=job.sample_size, seed=job.seed)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (interim / "records.jsonl", interim / "splits.json"):
            archive.write(path, path.as_posix())
        for record in sample:
            archive.write(record.path, record.path)
    size_mb = OUT.stat().st_size / 1_000_000
    print(f"Wrote {OUT} with {len(sample)} images ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
