"""What goes to the Hugging Face Space: the showcase app only, never private data (M8b)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
assert spec is not None
assert spec.loader is not None
deploy_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_space)


def test_the_space_gets_the_app_the_data_and_its_readme(tmp_path: Path) -> None:
    out = tmp_path / "space"
    deploy_space.stage(ROOT, out)
    for needed in ("Dockerfile", "README.md", "pyproject.toml", "uv.lock", "LICENSE"):
        assert (out / needed).is_file(), needed
    for folder in ("src/claimlens/web", "config", "showcase/blobs"):
        assert (out / folder).is_dir(), folder
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\ntitle: ClaimLens")
    assert "sdk: docker" in readme


def test_nothing_private_is_staged(tmp_path: Path) -> None:
    out = tmp_path / "space"
    deploy_space.stage(ROOT, out)
    staged = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    for forbidden in ("data/", "models/", "var/", ".env", "legacy/", "tests/", ".git/"):
        assert not [s for s in staged if s.startswith(forbidden) or s == forbidden.rstrip("/")]
    assert not [s for s in staged if s.endswith((".pt", ".onnx", ".pth", ".safetensors"))]
    assert not [s for s in staged if "__pycache__" in s]
