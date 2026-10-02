# Damage model v1: YOLO11-seg on damage-v1

- Date: 2026-10-02
- Dataset: `damage-v1` (md5 `cc56ac5162bfc4fdad163862267fedbd.dir`)
- Champion: `damage-yolo11s-v1` (MLflow version 2)
- Rule: chosen on validation mask mAP50; test scores are reported, never used to choose.

| Run | Base model | Status | Epochs (best) | Val mask mAP50 | Test mask mAP50 | Test mask mAP50-95 | Test box mAP50 | CPU ms/image |
|---|---|---|---|---|---|---|---|---|
| damage-yolo11n-v1 | yolo11n-seg.pt | complete | 100 (100) | 0.719 | 0.735 | 0.545 | 0.742 | 132 |
| damage-yolo11s-v1 | yolo11s-seg.pt | complete | 92 (72) | 0.738 | 0.733 | 0.549 | 0.743 | 310 |

Target test mask mAP50 >= 0.50: **met** (0.733).

## Per-class test mask mAP50 (`damage-yolo11s-v1`)

| Class | Mask mAP50 |
|---|---|
| crack | 0.441 |
| dent | 0.643 |
| glass_shatter | 0.971 |
| lamp_broken | 0.883 |
| scratch | 0.566 |
| tire_flat | 0.892 |
