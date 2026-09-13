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
# The scenario walkthrough became a top-level tab. It used to be a collapsed
# overlay wedged into the 3D scene at a hardcoded offset, and its "find the next
# divergent instance" walk fired up to 40 SERIAL /replay requests with every
# control disabled meanwhile. Same bans, new file.
WALKTHROUGH = APP / "views" / "Situations.tsx"
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


def test_energy_status_panel_survived_the_removal_of_the_energy_tab():
    """The Energy tab was cut: its radar computed its own axes in TypeScript from
    invented constants (`mean(dep)/4`, `reserve_mean/1.4e6`), which is exactly
    what these guards exist to prevent. The one thing on it worth keeping was the
    energy-status panel, which moved to Evidence -- an exported panel nobody
    mounts is still not a surface."""
    src = EVIDENCE.read_text(encoding="utf-8")
    assert "EnergyStatusPanel" in src


def test_no_view_recreates_the_frontend_computed_energy_radar():
    """The specific arithmetic that got the Energy tab cut, banned by pattern so
    it cannot come back in another file."""
    for path in list(APP.rglob("*.tsx")):
        src = _strip_comments(path.read_text(encoding="utf-8"))
        assert "reserve_mean / 1.4e6" not in src.replace(" ", " "), path
        assert "1.4e6" not in src, f"{path} divides by an invented reserve constant"


def test_scenario_walkthrough_renders_the_permanent_disclaimer_unconditionally():
    """The banner lives inside the reused `ReplayPanel`, rendered whenever a
    replay result exists -- not behind a hover, not conditional on a flag."""
    src = WALKTHROUGH.read_text(encoding="utf-8")
    assert "ReplayPanel" in src
    panel_src = (APP / "components" / "P3Panel.tsx").read_text(encoding="utf-8")
    assert "replay.label" in panel_src
    assert "title=" not in panel_src.split("{replay.label}")[0][-200:]


def test_scenario_walkthrough_shows_no_driver_action_verdict_at_all():
    """This test used to pin the opposite assertion, and the assertion was wrong.

    It required `s.matches_recommendation === true/false` to be rendered, i.e. a
    MATCHED / DIVERGED verdict per row. That verdict compared P2's call against a
    "driver action" defined as "position improved over the evaluation window",
    which records a failed attack as a hold and a promotion from the pit stop of
    the car ahead as an attack. 852 rows carried it. The position delta survives
    as an OBSERVED OUTCOME, the driver's action is reported as unobserved, and the
    counterfactual is reported as unresolved -- so what is pinned here now is the
    ABSENCE of the verdict, plus the presence of the honest fields.

    Full coverage of the new semantics is in tests/test_situations_semantics.py.
    """
    src = _strip_comments(WALKTHROUGH.read_text(encoding="utf-8"))
    banned = {
        r"actual_action": "the legacy position-derived driver action",
        r"matches_recommendation": "the legacy MATCHED/DIVERGED verdict",
        r"position_after\s*<\s*position_before": "a client-side position comparison",
    }
    for pattern, what in banned.items():
        assert not re.search(pattern, src), f"Situations.tsx renders {what}"
    # Read, never derived: the outcome string and its delta are the backend's.
    assert "observed_outcome" in src
    assert "observed_position_delta" in src
    # `null` is still a real third state -- the window had no published position
    # at both ends -- and it must not be folded into 0 ("no position change").
    # At the old fixed 30 s horizon, shorter than a lap at every circuit, EVERY
    # point came back null, which is what made the feature dead on arrival.
    assert "observed_position_delta !== null" in src
    assert "observed_position_delta === 0" in src
    # The recommendation filter is canonical P2, not the P1 per-lap flag.
    assert "recommendation === 'ATTACK'" in src


def test_scenario_walkthrough_admits_when_no_divergent_instance_exists():
    """An empty filter result is stated as the real answer for this pair, not
    rendered as a blank panel that reads like a loading state."""
    src = WALKTHROUGH.read_text(encoding="utf-8")
    assert "No situation in this race matches that filter" in src
    assert "not an empty page" in src
    # And the same for a pair the engine never had a call to make about.
    assert "No causal decision points" in src


def test_the_view_switch_matches_the_five_tab_architecture():
    """Seven tabs became five. `energy` invented numbers client-side, `strategy`
    duplicated `cockpit`, `method` duplicated `evidence`, and `context` lost its
    RDD sub-tab (a cutoff slider that refit against 130 MB of re-parsed JSON on
    every drag) so it is now plainly `observability`. `situations` is new."""
    src = PLAYBACK.read_text(encoding="utf-8")
    i = src.index("export type View =")
    view_type = src[i:src.index(";", i)]
    for name in ("cockpit", "situations", "replay", "observability", "evidence"):
        assert f"'{name}'" in view_type, f"View union is missing '{name}'"
    for stale in ("'theatre'", "'rdd'", "'decision'", "'fingerprint'",
                  "'strategy'", "'energy'", "'context'", "'method'"):
        assert stale not in view_type, f"View union still declares the retired id {stale}"


def test_every_view_in_the_union_is_reachable_from_the_nav():
    """A tab in the type but not in the nav is a view nobody can open; a nav
    entry not in the type does not compile. This pins the first direction."""
    view_src = PLAYBACK.read_text(encoding="utf-8")
    i = view_src.index("export type View =")
    ids = re.findall(r"'([a-z]+)'", view_src[i:view_src.index(";", i)])
    app = APP_TSX.read_text(encoding="utf-8")
    nav = app[app.index("const VIEWS"):app.index("export default function App")]
    for name in ids:
        assert f"id: '{name}'" in nav, f"View '{name}' is not in the nav bar"
        assert f"view === '{name}'" in app, f"View '{name}' renders nothing"


def test_the_retired_views_are_gone_not_merely_unmounted():
    """A deleted tab whose file survives is a file the next person re-mounts."""
    for name in ("Fingerprint.tsx", "Strategy.tsx", "RDD.tsx", "Method.tsx"):
        assert not (APP / "views" / name).exists(), f"{name} should have been deleted"
    assert not (APP / "components" / "ScenarioWalkthrough.tsx").exists()
