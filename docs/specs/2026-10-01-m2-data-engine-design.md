# M2 Data Engine: Design

- **Status:** Accepted (user: "proceed", 2026-10-01)
- **Parent spec:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md), section 12
- **Milestone:** M2, split into **M2a** (pipeline, this plan) and **M2b** (review, auto-labelling,
  golden v1; next plan)

## 1. Goal

One command rebuilds a clean, deduplicated, leakage-free, versioned training set for damage
segmentation (M3) and car-part segmentation (M3 fusion), with a data card that states where every
image came from and under which terms.

## 2. Findings that shape the design

| Finding | Evidence | Consequence |
|---|---|---|
| Roboflow `auto-industry/car-damage-detection-vyhvw` v6 **is CarDD** | 4,000 images; splits 2,816 / 810 / 374 equal CarDD's 70.4 / 20.25 / 9.35 %; CarDD's six classes; file names keep CarDD ids (`000581_jpg.rf...`) | Treat it as CarDD: non-commercial research and education terms apply despite the "CC BY 4.0" label on the re-upload. Do not redistribute images or publish weights for commercial use. Submit the official CarDD licence form. |
| The local 339-image folder is not v6 | 7 classes including `smash`; mixes CarDD ids with other sources (`crashed…`, `glass…`) | Rename it `legacy-course-subset`. Use it only for golden claims, not training. |
| Roboflow COCO export has a placeholder category (id 0, `dent`, supercategory `none`) | Export inspection | Map labels by name through the taxonomy file; categories with no annotations are ignored. |
| Car-parts dataset is CC BY 4.0 (Roboflow `car-seg`), 3,833 images, 23 classes | Ultralytics docs | Parts source for M3. The `object` catch-all class is dropped. |
| Ultralytics is AGPL-3.0 | Ultralytics licence | Affects the public demo (M8). Decide in an M3 ADR. |
| Google Drive DVC remote is blocked by Google policy | DVC forum | Default to a local DVC remote; DagsHub is a one-command switch. |

## 3. Decisions

1. **Damage taxonomy for training:** CarDD's six classes (`crack, dent, glass_shatter,
   lamp_broken, scratch, tire_flat`). `smash` stays in `DamageType` for legacy compatibility but is
   not trained (no source has enough examples).
2. **Splits:** keep CarDD's official test split frozen as the benchmark test set, so results are
   comparable with the CarDD paper. Near-duplicate clusters are assigned to a single split:
   test if any member is in the official test split, else validation if any member is, else train.
3. **Protected images:** any image that is a near-duplicate of a golden-claims image is excluded
   from train and validation, so the triage evaluation never sees training data.
4. **Canonical format:** one JSON Lines file of `ImageRecord`s (normalised polygons) per dataset in
   `data/interim/`; YOLO-seg exports in `data/processed/`.
5. **Near-duplicates:** 64-bit perceptual hash, Hamming distance ≤ 6, clustered with union-find.
6. **Data contract:** hard errors (unreadable image, size mismatch, unknown label, polygon outside
   [0, 1], fewer than 3 points, zero area) fail the build; soft warnings (very small instances,
   images with no annotations) are reported.
7. **Versioning:** raw snapshots tracked with `dvc add`; derived stages in `dvc.yaml`; dataset
   statistics as DVC metrics.
8. **Secrets:** `ROBOFLOW_API_KEY` read from the environment or `.env`; never logged.

## 4. Pipeline

```
fetch (manual, needs API key)      dvc-tracked raw snapshots
  roboflow cardd-v6 (COCO-seg) ─┐
  carparts-seg (YOLO-seg) ──────┼─▶ convert ─▶ validate ─▶ dedupe ─▶ split ─▶ export ─▶ report
  legacy-course-subset ─────────┘   (records)   (contract)  (pHash)   (clusters) (YOLO-seg) (stats, card)
```

| Stage | Input | Output |
|---|---|---|
| convert | raw folders, `config/taxonomy.toml` | `data/interim/<dataset>/records.jsonl` |
| validate | records + images | `reports/data/<dataset>-validation.json` (fails on hard errors) |
| dedupe | records of all damage sources + golden images | `data/interim/<dataset>/clusters.json` |
| split | records + clusters | `data/interim/<dataset>/splits.jsonl` |
| export | splits | `data/processed/<dataset>/` (YOLO-seg + `data.yaml`) |
| report | splits | `reports/data/<dataset>-stats.json` (DVC metrics) + Markdown summary |

Datasets built: **`damage-v1`** (CarDD via Roboflow) and **`parts-v1`** (car-seg).

## 5. M2b (next plan)

FiftyOne visual review; Grounding DINO + SAM 2 part masks on a sample of damage images (Colab);
human correction of 50 images into a fusion evaluation set; golden claims v1 adding minor-damage
cases from the frozen CarDD test split.

## 6. Out of scope

Training (M3), CarDD official-copy reconciliation (when the licence is approved: dedupe the
official copy against the Roboflow copy and record the result), cloud DVC remote setup (one
command, documented).
