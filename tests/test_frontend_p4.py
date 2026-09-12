"""P4: the productized frontend still computes nothing.

Same failure mode as P2's guard (`test_frontend_p2.py`): a frontend view that
grows its own opinion -- a threshold, a re-ranking, a re-derived status --
until two screens disagree about what the engine said. Cockpit is the new
first-viewport surface and gets the same static discipline as `P2Panel.tsx`;
`ScenarioWalkthrough.tsx` gets the same treatment for the one genuinely new
backend comparison (`matches_recommendation`).
"""
from __future__ import annotations

import pathlib
import re

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "simulation" / "app" / "src"
COCKPIT = APP / "views" / "Cockpit.tsx"
EVIDENCE = APP / "views" / "Evidence.tsx"
FINGERPRINT = APP / "views" / "Fingerprint.tsx"
WALKTHROUGH = APP / "components" / "ScenarioWalkthrough.tsx"
CANDIDATE_TABLE = APP / "components" / "CandidateActionsTable.tsx"
APP_TSX = APP / "App.tsx"
PLAYBACK = APP / "store" / "playback.ts"


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("//"))


def test_cockpit_exists_and_is_wired_into_the_new_view_switch():
    assert COCKPIT.exists()
    src = APP_TSX.read_text(encoding="utf-8")
    assert "Cockpit" in src and "view === 'cockpit'" in src


def test_cockpit_contains_no_decision_mathematics():
    """Reuses P2's exact banned-pattern list so the two views cannot drift apart."""
    from tests.test_frontend_p2 import _ts_sources

    banned = {
        r"Math\.exp\s*\(\s*-": "a logistic / sigmoid",
        r"1\s*/\s*\(\s*1\s*\+\s*Math\.exp": "a logistic",
        r"\bp_pass\b": "a pass-probability formula",
        r"delta_v\s*=": "a delta_v computation",
        r"value_attack\s*=\s*[^=]": "an expected-value computation",
        r"\bbellman\b": "a Bellman step",
        r"expected_regret\s*=\s*[^=]": "a regret computation",
        r"action_consensus\s*=\s*[^=]": "a consensus computation",
        r"decision_margin\s*=\s*[^=]": "a margin computation",
        r"deployment_budget_mj\s*=\s*[^=]": "a budget choice",
    }
    src = _strip_comments(COCKPIT.read_text(encoding="utf-8"))
    for pattern, what in banned.items():
        assert not re.search(pattern, src), f"Cockpit.tsx contains {what}"


def test_cockpit_reads_only_fields_the_service_actually_returns():
    from tests.test_p1_service import _payload
    from xray.decision_service import (evaluate_decision_trace_from_payload,
                                       evaluate_opportunity_decision)

    trace = evaluate_decision_trace_from_payload(_payload(), "OWN", "RIV")
    p2 = evaluate_opportunity_decision(_payload(), "OWN", "RIV")
    row_keys = set(trace["laps"][0]) if trace["laps"] else set()

    src = _strip_comments(COCKPIT.read_text(encoding="utf-8"))
    for name in sorted(set(re.findall(r"\bp2[?.]?\.([a-z_][a-z0-9_]*)", src))):
        assert name in p2, f"Cockpit reads p2.{name}, which the service does not return"
    for name in sorted(set(re.findall(r"\brow[?.]?\.([a-z_][a-z0-9_]*)", src))):
        # nested access like row.own_tyre.compound is covered by the top-level key
        assert name in row_keys, f"Cockpit reads row.{name}, absent from a P1 lap row"


def test_cockpit_does_not_import_the_validation_registry_panels():
    """The first viewport answers 'attack or hold', not 'how validated is this' --
    those panels live on Evidence. Keeping them out of Cockpit is asserted, not
    just intended, so a future edit cannot silently reintroduce the clutter."""
    src = COCKPIT.read_text(encoding="utf-8")
    for name in ("ValidationPanel", "ModelStatusPanel", "DataQualityPanel"):
        assert name not in src, f"Cockpit.tsx imports {name}, which belongs on Evidence"


def test_candidate_actions_table_is_shared_not_duplicated():
    assert CANDIDATE_TABLE.exists()
    cockpit = COCKPIT.read_text(encoding="utf-8")
    p2panel = (APP / "components" / "P2Panel.tsx").read_text(encoding="utf-8")
    assert "CandidateActionsTable" in cockpit
    assert "CandidateActionsTable" in p2panel


def test_evidence_view_hosts_the_registry_panels():
    assert EVIDENCE.exists()
    src = EVIDENCE.read_text(encoding="utf-8")
    for name in ("ValidationPanel", "ModelStatusPanel", "DataQualityPanel"):
        assert name in src


def test_energy_tab_hosts_the_energy_status_panel_and_the_existing_band():
    """The p10/p90 uncertainty band (`Readouts.tsx`) predates P4 and must survive
    the reorg; this only checks the energy-status panel landed beside it, not
    that the band itself changed."""
    src = FINGERPRINT.read_text(encoding="utf-8")
    assert "EnergyStatusPanel" in src


def test_scenario_walkthrough_renders_the_permanent_disclaimer_unconditionally():
    """The banner lives inside the reused `ReplayPanel`, rendered whenever a
    replay result exists -- not behind a hover, not conditional on a flag."""
    src = WALKTHROUGH.read_text(encoding="utf-8")
    assert "ReplayPanel" in src
    panel_src = (APP / "components" / "P3Panel.tsx").read_text(encoding="utf-8")
    assert "replay.label" in panel_src
    assert "title=" not in panel_src.split("{replay.label}")[0][-200:]


def test_scenario_walkthrough_never_recomputes_the_match_client_side():
    """`matches_recommendation` and `actual_action` must be read, never derived,
    in the frontend -- the comparison is the backend's."""
    src = _strip_comments(WALKTHROUGH.read_text(encoding="utf-8"))
    banned = {
        r"actual_action\s*=\s*[^=]": "a client-side actual_action derivation",
        r"matches_recommendation\s*=\s*[^=]": "a client-side match derivation",
        r"position_after\s*<\s*position_before": "a client-side position comparison",
    }
    for pattern, what in banned.items():
        assert not re.search(pattern, src), f"ScenarioWalkthrough.tsx contains {what}"
    assert "r.matches_recommendation === want" in src


def test_scenario_walkthrough_admits_when_no_divergent_instance_exists():
    src = WALKTHROUGH.read_text(encoding="utf-8")
    assert "No opportunity in this race where the driver's action diverged" in src \
        or "No further opportunity where the driver's action matched" in src


def test_the_view_switch_matches_the_new_seven_tab_architecture():
    src = PLAYBACK.read_text(encoding="utf-8")
    i = src.index("export type View =")
    view_type = src[i:src.index(";", i)]
    for name in ("cockpit", "strategy", "energy", "context", "replay", "evidence", "method"):
        assert f"'{name}'" in view_type, f"View union is missing '{name}'"
    for stale in ("'theatre'", "'observability'", "'rdd'", "'decision'", "'fingerprint'"):
        assert stale not in view_type, f"View union still declares the retired id {stale}"
