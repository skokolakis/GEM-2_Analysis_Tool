"""The About panel and the documentation cite only works listed in docs/references.md."""
import re
from pathlib import Path

import pytest

import RIs_v2 as R

ROOT = Path(__file__).resolve().parent.parent
REFERENCES = (ROOT / "docs" / "references.md").read_text(encoding="utf-8")
REFERENCE_LINES = [line for line in REFERENCES.splitlines() if line.startswith("- ")]

# "Won et al., 1996", "Sheriff & Geldart, 1995", "Akima (1970)", "González Jiménez et al. (2022)"
CITATION = re.compile(
    r"\b([A-Z][\w-]+(?: [A-Z][\w-]+)?)(?:'s)?(?:\s+et\s+al\.|\s+&\s+[A-Z][\w-]+)?,?\s+\(?((?:19|20)\d\d)\b"
)


def _citations(text: str) -> set[tuple[str, str]]:
    return set(CITATION.findall(text))


def _listed(author: str, year: str) -> bool:
    names = {author, author.split()[-1]}         # "González Jiménez", or "After Simon" -> "Simon"
    return any(line.startswith(f"- {name}") and f"({year})" in line
               for line in REFERENCE_LINES for name in names)


APP_TEXTS = {
    "ABOUT_MD": R.ABOUT_MD,
    "PSEUDOSECTION_CAPTION": R.PSEUDOSECTION_CAPTION,
    **{f"SCIENCE_HELP[{k}]": v for k, v in R.SCIENCE_HELP.items()},
}
DOC_TEXTS = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "docs").glob("*.md")
             if p.name != "references.md"}


@pytest.mark.parametrize("name", sorted({**APP_TEXTS, **DOC_TEXTS}))
def test_every_citation_is_in_the_bibliography(name):
    text = {**APP_TEXTS, **DOC_TEXTS}[name]
    missing = sorted(c for c in _citations(text) if not _listed(*c))
    assert not missing, f"{name} cites works missing from docs/references.md: {missing}"


def test_about_cites_the_main_methods():
    cited = {author for author, _ in _citations(R.ABOUT_MD)}
    assert {"Won", "McNeill", "Callegary", "Huang", "Sheriff", "Akima"} <= cited


def test_about_references_come_from_the_docs():
    rendered = R.about_references_markdown().splitlines()
    assert rendered == REFERENCE_LINES
    assert len(rendered) > 40


def test_bibliography_has_unique_dois():
    dois = re.findall(r"https://doi\.org/(\S+)", REFERENCES)
    assert len(dois) == len(set(dois))
