"""Where the judge reads and writes. Standard library only.

These re-derive `out/races` and `out/decisions` from `__file__` rather than
importing them from `simulation/api/main.py`, which already defines them. The
duplication is deliberate: the dependency direction in this repo is
API -> xray, and importing the API from inside `xray/` inverts it, which would
make `uvicorn` and the judge circularly dependent and would drag FastAPI into
the import graph of a package that is meant to need nothing but the standard
library.

Three `Path` joins is the cheaper of the two wrongs, and
`tests/test_judge.py::test_judge_paths_agree_with_the_api` asserts the two
definitions are equal so they cannot drift apart silently.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

RACES = ROOT / "out" / "races"
DECISIONS = ROOT / "out" / "decisions"
DECISIONS_INDEX = DECISIONS / "_index.json"

JUDGE = ROOT / "out" / "judge"
CACHE = JUDGE / "cache"
VERDICTS = JUDGE / "verdicts"
VERDICTS_INDEX = VERDICTS / "_index.json"
SAMPLES = JUDGE / "samples"
REPORTS = JUDGE / "reports"
LATEST_REPORT = REPORTS / "latest.json"
QUOTA = JUDGE / "_quota.json"


def write_atomic(path: Path, text: str) -> None:
    """Write via `.tmp` + `replace`, the convention `_bundle` and `scripts/16`
    already use: a reader never sees a half-written file, and a run killed by a
    quota stop leaves no truncated artefact behind for the next run to parse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
