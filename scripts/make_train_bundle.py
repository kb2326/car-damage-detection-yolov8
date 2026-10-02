"""Pack damage-v1 for the Kaggle training job (private dataset `claimlens-damage-v1`).

The zip holds the YOLO export at the path the job expects, plus `dataset.json` with the md5 DVC
recorded for it, so the job can refuse a stale bundle. Run from the repository root:

uv run python scripts/make_train_bundle.py
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from claimlens.training.config import dvc_out_md5

DATASET = "damage-v1"
SOURCE = Path("data/processed") / DATASET
OUT_DIR = Path("var/kaggle/train-bundle")
KAGGLE_ID = "karthickbalaje/claimlens-damage-v1"


def main() -> int:
    md5 = dvc_out_md5(Path("dvc.lock"), SOURCE.as_posix())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUT_DIR / f"claimlens-{DATASET}.zip"
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file())
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("dataset.json", json.dumps({"dataset": DATASET, "md5": md5}))
        for path in files:
            archive.write(path, path.as_posix())
    metadata = {
        "title": f"claimlens-{DATASET}",
        "id": KAGGLE_ID,
        "licenses": [{"name": "other"}],
    }
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    size_mb = archive_path.stat().st_size / 1_000_000
    print(f"Wrote {archive_path} with {len(files)} files ({size_mb:.0f} MB), md5 {md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
