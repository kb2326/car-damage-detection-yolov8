"""Data source and dataset configuration (config/datasets.toml) and secret lookup."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from claimlens.domain import Frozen


class SplitLayout(Frozen):
    images: str | None = None
    labels: str | None = None
    annotations: str | None = None


class SourceConfig(Frozen):
    id: str
    kind: Literal["roboflow", "url", "local"]
    format: Literal["coco-seg", "yolo-seg"]
    terms: str
    splits: dict[Literal["train", "valid", "test"], SplitLayout]
    class_names_file: str | None = None
    url: str | None = None
    workspace: str | None = None
    project: str | None = None
    version: int | None = None
    export_format: str | None = None

    @model_validator(mode="after")
    def _complete(self) -> Self:
        for name, layout in self.splits.items():
            if self.format == "coco-seg" and not layout.annotations:
                raise ValueError(f"{self.id}/{name}: coco-seg splits need an annotations file")
            if self.format == "yolo-seg" and not (layout.images and layout.labels):
                raise ValueError(f"{self.id}/{name}: yolo-seg splits need images and labels")
        if self.kind == "url" and not self.url:
            raise ValueError(f"{self.id}: url sources need a url")
        if self.kind == "roboflow" and not (
            self.workspace and self.project and self.version and self.export_format
        ):
            raise ValueError(f"{self.id}: roboflow sources need workspace, project, version")
        return self


class DatasetConfig(Frozen):
    id: str
    taxonomy: str
    sources: tuple[str, ...]
    protect_golden: str | None = None
    phash_max_distance: int = 6


class DataConfig(Frozen):
    sources: dict[str, SourceConfig]
    datasets: dict[str, DatasetConfig]


def load_data_config(path: Path) -> DataConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return DataConfig(
        sources={
            key: SourceConfig.model_validate({"id": key, **body})
            for key, body in data.get("source", {}).items()
        },
        datasets={
            key: DatasetConfig.model_validate({"id": key, **body})
            for key, body in data.get("dataset", {}).items()
        },
    )


def read_secret(name: str, env_file: Path = Path(".env")) -> str | None:
    """Environment variable first, then a KEY=VALUE line in the .env file."""
    value = os.environ.get(name)
    if value:
        return value
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, raw = line.strip().partition("=")
            if sep and key.strip() == name and not key.startswith("#"):
                return raw.strip().strip('"').strip("'") or None
    return None
