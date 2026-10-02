"""Pack a built dataset for a Kaggle training job (private dataset `claimlens-<dataset>`).

The zip holds the YOLO export at the path the job expects, plus `dataset.json` with the md5 DVC
recorded for it, so the job can refuse a stale bundle. Run from the repository root:

uv run python scripts/make_train_bundle.py [damage-v1|parts-v1]
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from claimlens.training.config import dvc_out_md5

KAGGLE_USER = "karthickbalaje"


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "damage-v1"
    source = Path("data/processed") / dataset
    out_dir = Path("var/kaggle") / f"train-bundle-{dataset}"
    md5 = dvc_out_md5(Path("dvc.lock"), source.as_posix())
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"claimlens-{dataset}.zip"
    files = sorted(p for p in source.rglob("*") if p.is_file())
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("dataset.json", json.dumps({"dataset": dataset, "md5": md5}))
        for path in files:
            archive.write(path, path.as_posix())
    metadata = {
        "title": f"claimlens-{dataset}",
        "id": f"{KAGGLE_USER}/claimlens-{dataset}",
        "licenses": [{"name": "other"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    size_mb = archive_path.stat().st_size / 1_000_000
    print(f"Wrote {archive_path} with {len(files)} files ({size_mb:.0f} MB), md5 {md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
