"""Grounding DINO (open-vocabulary boxes) + SAM 2 (masks from boxes), on GPU if present, else CPU.

Requires `uv sync --group autolabel`. Excluded from coverage; see the integration test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from claimlens.autolabel.postprocess import Box, Detection, mask_to_polygon, select_detections
from claimlens.data.records import Annotation


class GroundedSamLabeller:
    def __init__(
        self,
        prompts: Mapping[str, str],
        *,
        box_threshold: float,
        text_threshold: float,
        min_score: float,
        max_per_group: int,
        detector: str,
        segmenter: str,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self._torch: Any = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._prompts = dict(prompts)
        self._text = " ".join(f"{phrase}." for phrase in self._prompts)
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._min_score = min_score
        self._max_per_group = max_per_group
        self._det_processor: Any = AutoProcessor.from_pretrained(detector)
        self._det_model: Any = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(
            self.device
        )
        self._seg_processor: Any = Sam2Processor.from_pretrained(segmenter)
        self._seg_model: Any = Sam2Model.from_pretrained(segmenter).to(self.device)
        self.model_version = f"{detector}+{segmenter}"

    def label(self, image_path: Path) -> tuple[list[Annotation], Counter[str]]:
        torch = self._torch
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self._det_processor(images=image, text=self._text, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self._det_model(**inputs)
        result = self._det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        detections: list[Detection] = []
        for box, score, label in zip(
            result["boxes"].tolist(), result["scores"].tolist(), result["text_labels"], strict=True
        ):
            x1, y1, x2, y2 = (float(v) for v in box)
            coords: Box = (x1, y1, x2, y2)
            detections.append(Detection(phrase=str(label), score=float(score), box=coords))
        kept, dropped = select_detections(
            detections, self._prompts, min_score=self._min_score, max_per_group=self._max_per_group
        )
        if not kept:
            return [], dropped
        seg_inputs = self._seg_processor(
            images=image, input_boxes=[[list(d.box) for _, d in kept]], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            seg_outputs = self._seg_model(**seg_inputs, multimask_output=False)
        masks = self._seg_processor.post_process_masks(
            seg_outputs.pred_masks.cpu(), seg_inputs["original_sizes"].cpu()
        )[0]
        annotations: list[Annotation] = []
        for (group, detection), mask in zip(kept, masks, strict=True):
            polygon = mask_to_polygon(mask[0].numpy())
            if polygon is None:
                dropped["no_mask"] += 1
                continue
            annotations.append(
                Annotation(label=group, polygon=polygon, score=round(detection.score, 4))
            )
        return annotations, dropped
