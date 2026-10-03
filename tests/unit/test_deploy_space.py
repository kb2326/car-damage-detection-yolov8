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


def test_stray_secrets_and_journals_are_never_staged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for name in deploy_space.FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x", encoding="utf-8")
    for folder in deploy_space.FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "keep.txt").write_text("x", encoding="utf-8")
    (root / "config" / ".env").write_text("SECRET=1", encoding="utf-8")
    (root / "showcase" / "claims.db-journal").write_text("x", encoding="utf-8")
    (root / "hf-space").mkdir()
    (root / "hf-space" / "README.md").write_text("---\ntitle: ClaimLens\n---\n", encoding="utf-8")
    out = tmp_path / "space"
    deploy_space.stage(root, out)
    staged = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert "config/keep.txt" in staged
    assert not [s for s in staged if s.endswith(".env") or s.endswith("-journal")]
