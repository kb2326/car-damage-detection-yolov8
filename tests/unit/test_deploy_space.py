"""What goes to the Hugging Face Space: the static showcase only, never private data (M8b)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
assert spec is not None
assert spec.loader is not None
deploy_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_space)


def test_the_space_gets_the_static_showcase_and_its_readme(tmp_path: Path) -> None:
    out = tmp_path / "space"
    deploy_space.stage(ROOT, out)
    for needed in ("index.html", "claims.html", "README.md", "static/app.css"):
        assert (out / needed).is_file(), needed
    assert len(list(out.glob("claim-*.html"))) == 5
    assert len(list((out / "photos").iterdir())) >= 5
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\ntitle: ClaimLens")
    assert "sdk: static" in readme


def test_only_web_files_are_staged(tmp_path: Path) -> None:
    out = tmp_path / "space"
    deploy_space.stage(ROOT, out)
    staged = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    allowed = (".html", ".css", ".js", ".jpg", ".jpeg", ".png", ".webp", ".md")
    assert staged
    assert not [s for s in staged if not s.endswith(allowed)]
    assert not [s for s in staged if s.startswith(("src/", "config/", "data/", "models/", "var/"))]
