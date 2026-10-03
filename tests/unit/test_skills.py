from pathlib import Path

import pytest

from claimlens.skills import load_skills, load_skills_report, parse_skill

ROOT = Path(__file__).resolve().parents[2]
BODY = "1. Check the photos.\n2. Check the wording.\n3. Say what an adjuster should check next.\n"


def write(root: Path, name: str, front: str, body: str = BODY) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    path = folder / "SKILL.md"
    path.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")
    return path


def test_parse_reads_the_front_matter_and_body(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "glass-claims",
        'name: glass-claims\ndescription: "Glass damage."\nversion: 2\n'
        "approved_by: Sam\napproved_on: 2026-10-03",
    )
    skill = parse_skill(path)
    assert (skill.name, skill.description, skill.version) == ("glass-claims", "Glass damage.", 2)
    assert skill.approved_by == "Sam"
    assert skill.body == BODY.strip()


@pytest.mark.parametrize(
    ("front", "error"),
    [
        ("description: x\nversion: 1", "name"),
        ("name: glass-claims\nversion: 1", "description"),
        ("name: other-name\ndescription: x\nversion: 1", "folder"),
        ("name: glass-claims\ndescription: x\nversion: one", "version"),
    ],
)
def test_bad_front_matter_is_refused(tmp_path: Path, front: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_skill(write(tmp_path, "glass-claims", front))


def test_a_skill_needs_a_real_body(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny", "name: tiny\ndescription: x\nversion: 1", body="Do it.")
    with pytest.raises(ValueError, match="too short"):
        parse_skill(path)


def test_a_file_without_front_matter_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "SKILL.md").write_text(BODY, encoding="utf-8")
    with pytest.raises(ValueError, match="front matter"):
        parse_skill(folder / "SKILL.md")


def test_only_approved_skills_are_loaded(tmp_path: Path) -> None:
    write(
        tmp_path,
        "approved",
        "name: approved\ndescription: a\nversion: 1\napproved_by: Sam\napproved_on: 2026-10-03",
    )
    write(tmp_path, "pending", 'name: pending\ndescription: b\nversion: 1\napproved_by: ""')
    write(
        tmp_path,
        "bad-date",
        "name: bad-date\ndescription: c\nversion: 1\napproved_by: Sam\napproved_on: soon",
    )
    loaded, skipped = load_skills_report(tmp_path)
    assert set(loaded) == {"approved"}
    assert skipped == [
        "bad-date: approved_on must be a date (YYYY-MM-DD)",
        "pending: not approved by the owner yet",
    ]
    assert set(load_skills(tmp_path)) == {"approved"}


def test_a_missing_folder_has_no_skills(tmp_path: Path) -> None:
    assert load_skills(tmp_path / "nope") == {}


def test_the_repo_skills_parse() -> None:
    names = {p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md")}
    assert names == {"glass-claims", "flat-tyre-claims", "exclusion-review"}
    for path in (ROOT / "skills").glob("*/SKILL.md"):
        skill = parse_skill(path)
        assert 10 <= len(skill.body.splitlines()) <= 25
