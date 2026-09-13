"""The read side: what the API imports. Standard library only, always.

`simulation/api/main.py` imports this module and nothing else from the judge
package. If the SDK ever leaked into this import chain, serving a request would
start requiring `google-genai` to be installed and `uvicorn` would fail to start
on a checkout that never intends to run the judge.
`tests/test_judge_api.py::test_the_api_never_imports_the_llm_client` asserts it.
"""
from __future__ import annotations

import json

from . import evidence as ev
from .paths import VERDICTS, VERDICTS_INDEX, write_atomic


def verdict_path(race: str, car: str, rival: str):
    return VERDICTS / f"{race}__{car}__{rival}.json"


def read_pair(race: str, car: str, rival: str) -> dict:
    """Cached verdicts for one pair, as the endpoint serves them.

    `available: false` is a normal answer, not an error. `out/` is gitignored,
    so a fresh clone has no verdicts at all; a 404 here would make the
    Situations tab render as broken for a state that is simply "nobody has run
    the judge yet".
    """
    p = verdict_path(race, car, rival)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"race": race, "car": car, "rival": rival,
                "available": False,
                "reason": ("no verdicts generated for this pair -- run "
                           "scripts/17.judge_situations.py"),
                "verdicts": {}}
    current = ev.artefact_mtime_ns(race)
    d["available"] = True
    d["reason"] = None
    # Stale verdicts are served and FLAGGED, not hidden. Dropping them would
    # make the panel look broken after any re-analysis, which is the moment a
    # reader most needs to be told the commentary predates the data.
    d["stale"] = (current is not None
                  and d.get("artefact_mtime_ns") is not None
                  and d["artefact_mtime_ns"] != current)
    return d


def read_index() -> dict:
    try:
        return json.loads(VERDICTS_INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def materialize(entries) -> dict:
    """Rebuild `verdicts/` and its index from cache entries, with no network.

    The cache is the source of truth; this is a view. `scripts/17 --materialize`
    is the sibling of `scripts/16 --index-only`.
    """
    pairs: dict[tuple[str, str, str], dict] = {}
    for entry in entries:
        v = entry.get("verdict") or {}
        key = entry.get("situation_key")
        if not key:
            continue
        try:
            race, car, rival, t = ev.parse_key(key)
        except (ValueError, KeyError):
            continue
        slot = pairs.setdefault((race, car, rival), {})
        slot[ev.REFUSAL_SENTINEL if t is None else f"{t:.2f}"] = v

    index = {}
    for (race, car, rival), verdicts in sorted(pairs.items()):
        any_v = next(iter(verdicts.values()))
        doc = {
            "race": race, "car": car, "rival": rival,
            "label": any_v.get("label"),
            "is_measurement": False,
            "model_id": any_v.get("model_id"),
            "prompt_version": any_v.get("prompt_version"),
            "rubric_version": any_v.get("rubric_version"),
            "judge_version": any_v.get("judge_version"),
            "stack": any_v.get("stack"),
            "artefact_mtime_ns": any_v.get("artefact_mtime_ns"),
            "generated_at": max((v.get("generated_at") or "")
                                for v in verdicts.values()) or None,
            "verdicts": verdicts,
        }
        write_atomic(verdict_path(race, car, rival),
                     json.dumps(doc, allow_nan=False))
        index[f"{race}__{car}__{rival}"] = {
            "n_verdicts": len(verdicts),
            "model_id": doc["model_id"],
            "prompt_version": doc["prompt_version"],
            "artefact_mtime_ns": doc["artefact_mtime_ns"],
            "generated_at": doc["generated_at"],
        }
    write_atomic(VERDICTS_INDEX, json.dumps(index, indent=1, allow_nan=False))
    return index
