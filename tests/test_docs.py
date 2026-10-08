"""The Markdown documents link only to files and headings that exist (no network needed)."""
import re
import subprocess
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tracked_markdown() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "*.md"], text=True)
        files = [ROOT / p for p in out.split()]
    except (OSError, subprocess.CalledProcessError):     # e.g. a source archive without .git
        files = []
    return files or sorted(p for p in ROOT.rglob("*.md") if ".venv" not in p.parts and "data" not in p.parts)


def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def slug(heading: str) -> str:
    """GitHub's heading anchors: lower case, letters/digits/spaces/hyphens kept, spaces to hyphens."""
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "").strip().lower()
    return "".join(c for c in h if c in " -_" or unicodedata.category(c)[0] in "LN").replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen, out = {}, set()
    for m in re.finditer(r"^#{1,6}\s+(.*?)\s*#*\s*$", strip_code(path.read_text()), flags=re.M):
        s = slug(m.group(1))
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.add(s if n == 0 else f"{s}-{n}")
    return out


@pytest.mark.parametrize("doc", tracked_markdown(), ids=lambda p: str(p.relative_to(ROOT)))
def test_internal_links_resolve(doc):
    broken = []
    for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", strip_code(doc.read_text())):
        if re.match(r"^(https?|mailto):", target):
            continue
        path, _, frag = target.partition("#")
        dest = (doc.parent / path).resolve() if path else doc
        if path and not dest.exists():
            broken.append(f"missing file: {target}")
        elif frag and dest.suffix == ".md" and frag not in anchors(dest):
            broken.append(f"missing heading: {target}")
    assert not broken, "\n".join(broken)


def test_both_readmes_cite_the_same_references():
    def refs(p):
        tail = p.read_text().split("\n## 12.")[1]
        return re.findall(r"^(\d+)\. .*?(doi:[^\]]+|arXiv:[\d.]+|PMLR [\d:–-]+|ANSI/AAMI EC57)", tail, flags=re.M)
    en, ru = refs(ROOT / "README.md"), refs(ROOT / "README.ru.md")
    assert en and en == ru
