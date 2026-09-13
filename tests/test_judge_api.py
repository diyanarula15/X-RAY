"""The judge endpoints: read-only, and never on the LLM's import path.

The handlers are called directly rather than over HTTP. `starlette.testclient`
needs `httpx`, which this project does not depend on, and adding a transport
dependency to assert the return value of a two-line `def` route buys nothing --
`tests/test_frontend_p2.py` already tests service output by calling it.

Every file read here passes `encoding="utf-8"` (invariant 14).
"""
from __future__ import annotations

import json
import sys

import pytest

from simulation.api.main import RACES, judge_endpoint, judge_report_endpoint
from xray.judge import store


def _a_race() -> str:
    ids = sorted(p.stem for p in RACES.glob("*.json"))
    if not ids:
        pytest.skip("no race artefacts on disk")
    return ids[0]


def test_the_api_never_imports_the_llm_client():
    """An accidental import chain here would make `uvicorn` startup depend on
    `google-genai` being installed, on a checkout that never runs the judge."""
    for mod in ("xray.judge.gemini", "xray.judge.agent"):
        sys.modules.pop(mod, None)
    judge_endpoint(_a_race(), car="X", rival="Y")
    judge_report_endpoint()
    assert "xray.judge.gemini" not in sys.modules
    assert "xray.judge.agent" not in sys.modules


def test_a_pair_with_no_verdicts_is_unavailable_not_a_404():
    body = judge_endpoint(_a_race(), car="NOBODY", rival="NOTHING")
    assert body["available"] is False
    assert body["verdicts"] == {}
    assert "scripts/17" in body["reason"]


def test_the_report_endpoint_degrades_the_same_way():
    body = judge_report_endpoint()
    # Either a real report, or a stated absence. Never an error.
    assert "available" in body or "run_id" in body


def test_a_materialised_pair_is_served_with_its_provenance(tmp_path,
                                                           monkeypatch):
    rid = _a_race()
    entry = {
        "situation_key": f"{rid}__AAA__BBB@123.45",
        "verdict": {
            "verdict": "OVERCLAIMED", "verdict_label": "over-claimed",
            "verdict_tone": "warn", "summary": "s", "judge_confidence": "LOW",
            "abstained": False, "abstain_reason": None, "axes": {},
            "overclaims": [], "label": "GENERATED COMMENTARY -- test",
            "is_measurement": False, "model_id": "m", "prompt_version": "p",
            "rubric_version": "r", "judge_version": "j", "stack": "real",
            "artefact_mtime_ns": (RACES / f"{rid}.json").stat().st_mtime_ns,
            "generated_at": "2026-09-13T00:00:00+00:00",
        },
    }
    monkeypatch.setattr(store, "VERDICTS", tmp_path)
    monkeypatch.setattr(store, "VERDICTS_INDEX", tmp_path / "_index.json")
    store.materialize([entry])

    got = store.read_pair(rid, "AAA", "BBB")
    assert got["available"] is True
    assert got["is_measurement"] is False
    assert "GENERATED COMMENTARY" in got["label"]
    assert got["verdicts"]["123.45"]["verdict"] == "OVERCLAIMED"
    # mtime matches the live artefact, so it is not stale
    assert got["stale"] is False


def test_a_verdict_built_against_an_old_artefact_is_flagged_stale(
        tmp_path, monkeypatch):
    """Served and flagged, never hidden: dropping it would make the panel look
    broken at exactly the moment a reader needs to be told the commentary
    predates the data."""
    rid = _a_race()
    entry = {"situation_key": f"{rid}__AAA__CCC@1.00",
             "verdict": {"verdict": "JUSTIFIED", "artefact_mtime_ns": 1,
                         "label": "L", "is_measurement": False,
                         "generated_at": "2026-01-01T00:00:00+00:00"}}
    monkeypatch.setattr(store, "VERDICTS", tmp_path)
    monkeypatch.setattr(store, "VERDICTS_INDEX", tmp_path / "_index.json")
    store.materialize([entry])
    assert store.read_pair(rid, "AAA", "CCC")["stale"] is True


def test_materialize_indexes_what_it_wrote(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "VERDICTS", tmp_path)
    monkeypatch.setattr(store, "VERDICTS_INDEX", tmp_path / "_index.json")
    rid = _a_race()
    idx = store.materialize([
        {"situation_key": f"{rid}__A__B@1.00",
         "verdict": {"verdict": "JUSTIFIED", "label": "L",
                     "is_measurement": False}},
        {"situation_key": f"{rid}__A__B@2.00",
         "verdict": {"verdict": "OVERCLAIMED", "label": "L",
                     "is_measurement": False}},
    ])
    assert idx[f"{rid}__A__B"]["n_verdicts"] == 2
    assert json.loads((tmp_path / "_index.json").read_text(encoding="utf-8"))


def test_a_refusal_verdict_materialises_under_the_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "VERDICTS", tmp_path)
    monkeypatch.setattr(store, "VERDICTS_INDEX", tmp_path / "_index.json")
    rid = _a_race()
    store.materialize([{"situation_key": f"{rid}__H__L@refusal",
                        "verdict": {"verdict": "CORRECTLY_REFUSED",
                                    "label": "L", "is_measurement": False}}])
    assert "refusal" in store.read_pair(rid, "H", "L")["verdicts"]
