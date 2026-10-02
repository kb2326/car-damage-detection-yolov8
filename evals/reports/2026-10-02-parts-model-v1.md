# Part model v1: YOLO11-seg on parts-v1

- Date: 2026-10-02
- Dataset: `parts-v1` (md5 `9764aa90d16ea83ad39939789180e590.dir`)
- Champion: `parts-yolo11n-v1` (MLflow version 1)
- Rule: chosen on validation mask mAP50; test scores are reported, never used to choose.

| Run | Base model | Status | Epochs (best) | Val mask mAP50 | Test mask mAP50 | Test mask mAP50-95 | Test box mAP50 | CPU ms/image |
|---|---|---|---|---|---|---|---|---|
| parts-yolo11n-v1 | yolo11n-seg.pt | complete | 63 (43) | 0.753 | 0.746 | 0.600 | 0.743 | 193 |

Target test mask mAP50 >= 0.50: **met** (0.746).

## Per-class test mask mAP50 (`parts-yolo11n-v1`)

| Class | Mask mAP50 |
|---|---|
| back_bumper | 0.944 |
| back_door | 0.902 |
| back_glass | 0.926 |
| back_left_door | 0.594 |
| back_left_light | 0.444 |
| back_light | 0.871 |
| back_right_door | 0.485 |
| back_right_light | 0.588 |
| front_bumper | 0.982 |
| front_door | 0.944 |
| front_glass | 0.977 |
| front_left_door | 0.633 |
| front_left_light | 0.590 |
| front_light | 0.901 |
| front_right_door | 0.616 |
| front_right_light | 0.479 |
| hood | 0.967 |
| left_mirror | 0.648 |
| right_mirror | 0.578 |
| tailgate | 0.859 |
| trunk | 0.774 |
| wheel | 0.709 |

## Part agreement on `fusion-eval-v1` (AI-reviewed parts, mask IoU >= 0.5, same group)

| Part group | Found / reviewed |
|---|---|
| door | 4 / 10 |
| glass | 12 / 25 |
| hood | 5 / 11 |
| light | 9 / 35 |
| mirror | 8 / 12 |
| wheel | 23 / 40 |
| **Total** | **61 / 133 (46%)** |

Much lower than the test score: `parts-v1` has whole-car photos, CarDD has close-ups, and the reviewed
outlines include small lights (fog lights, reflectors) that `parts-v1` does not label as lights.
Fusion therefore falls back to image-fraction size whenever no part is found.

## ONNX vs PyTorch on this laptop's CPU

| Format | Median ms/image (20 test photos) |
|---|---|
| `.pt` | 193 |
| `.onnx` | 460 |
