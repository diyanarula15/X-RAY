"""P2 §25: the frontend displays the canonical result and computes nothing.

The failure this guards against is a frontend that quietly grows its own
opinion -- a threshold here, a re-ranking there -- until two different numbers
are called "the recommendation" and nobody can say which the engine produced.

These are static checks against the TypeScript sources plus a field-by-field
comparison against the real service output. They do not render the page; the
real typecheck and build are run separately (`tsc -b`, `vite build`) and their
result is reported in the audit rather than asserted here, because pytest cannot
install node modules.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "simulation" / "app" / "src"
P2_PANEL = APP / "components" / "P2Panel.tsx"
API_TS = APP / "lib" / "api.ts"
# The P2-bearing view has been renamed twice: Decision.tsx -> Strategy.tsx (P4's
# nav rename) -> Cockpit.tsx, when the Strategy tab was folded into Cockpit.
# Two tabs both answering "attack or hold" off the same two endpoints was a
# split with nothing behind it. Same P2Panel, same `api.p2` wiring, one file.
DECISION = APP / "views" / "Cockpit.tsx"


def _strip_comments(src: str) -> str:
    """Comments are prose. A ban that fires on prose is a ban nobody keeps."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.splitlines()
                      if not l.strip().startswith("//"))


def _ts_sources(skip_animation: bool = True):
    """Decision-bearing TypeScript.

    `src/three/` is excluded from the decision-maths ban and checked separately:
    it is the 3-D scene, and its `Math.exp(-dt * k)` calls are frame-rate
    easing, not a logistic. Banning them outright would be a rule the next
    person deletes rather than obeys.
    """
    out = {}
    for path in list(APP.rglob("*.ts")) + list(APP.rglob("*.tsx")):
        if skip_animation and "three" in path.parts:
            continue
        out[path] = _strip_comments(path.read_text(encoding="utf-8"))
    return out


def test_the_p2_panel_exists_and_is_wired_into_the_decision_view():
    assert P2_PANEL.exists()
    d = DECISION.read_text(encoding="utf-8")
    assert "P2Panel" in d and "api.p2(" in d


def test_the_frontend_contains_no_decision_mathematics():
    """No logistic, no Bellman, no threshold, no hand-rolled consensus."""
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
    for path, src in _ts_sources().items():
        for pattern, what in banned.items():
            assert not re.search(pattern, src), f"{path.name} contains {what}"


def test_the_only_exponentials_in_the_app_are_animation_easing():
    """The exclusion above is asserted, not assumed."""
    for path in (APP / "three").rglob("*.tsx"):
        src = _strip_comments(path.read_text(encoding="utf-8"))
        for m in re.finditer(r"Math\.exp\(([^)]*)\)", src):
            arg = m.group(1)
            assert "dt" in arg or "delta" in arg, (
                f"{path.name} has a non-easing exponential: Math.exp({arg})")


def test_the_panel_only_does_unit_presentation():
    """The only arithmetic allowed is x3.6 (km/h) and x100 (percent)."""
    src = P2_PANEL.read_text(encoding="utf-8")
    body = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith(("*", "//", "/*")))
    mults = set(re.findall(r"\*\s*(\d+(?:\.\d+)?)", body))
    assert mults <= {"3.6", "100"}, f"unexpected arithmetic in the panel: {mults}"
    # and no division except the same two presentational forms
    divs = set(re.findall(r"/\s*(\d+(?:\.\d+)?)", body))
    assert divs <= set(), f"unexpected division in the panel: {divs}"


def test_every_field_the_panel_reads_exists_in_the_service_output():
    """The real contract: TS property names against the real response."""
    from tests.test_p1_service import _payload
    from xray.decision_service import evaluate_opportunity_decision

    r = evaluate_opportunity_decision(_payload(), "OWN", "RIV")
    src = _strip_comments(P2_PANEL.read_text(encoding="utf-8"))
    top = [m for m in re.findall(r"\bp2\.([a-z_][a-z0-9_]*)", src)]
    for name in sorted(set(top)):
        assert name in r, f"P2Panel reads p2.{name}, which the service does not return"

    for key in ("action", "kind", "requested_budget_mj", "actual_deployed_mj",
                "saturated", "own_speed_mps", "rival_speed_mps", "delta_v_mps",
                "pass_probability", "value"):
        assert key in r["candidate_actions"][0], f"candidate_actions lacks {key}"
    for key in ("scenario_index", "rival_usable_energy_mj", "weight",
                "optimal_action", "optimal_value", "chosen_action_value"):
        assert key in r["policy_posterior"][0], f"policy_posterior lacks {key}"


def test_the_typescript_type_matches_the_service_keys():
    """api.ts must not declare a field the backend does not send."""
    from tests.test_p1_service import _payload
    from xray.decision_service import evaluate_opportunity_decision

    r = evaluate_opportunity_decision(_payload(), "OWN", "RIV")
    src = API_TS.read_text(encoding="utf-8")
    block = src[src.index("export type P2Decision = {"):]
    block = block[:block.index("\n};")]
    declared = set(re.findall(r"^\s{2}([a-z_][a-z0-9_]*)\s*:", block, re.M))
    missing = declared - set(r)
    assert not missing, f"api.ts declares fields the service does not return: {missing}"


def test_the_synthetic_pass_model_label_is_visible_and_not_hidden():
    src = P2_PANEL.read_text(encoding="utf-8")
    assert "pass_model_calibration" in src
    assert "NOT EMPIRICALLY CALIBRATED" in src
    assert "PASS MODEL:" in src
    # it must not be conditional on hover/title only
    assert "title=" not in src.split("PASS MODEL:")[0][-400:], (
        "the calibration badge must not be a tooltip")


def test_missing_p2_data_renders_a_fallback_rather_than_crashing():
    src = P2_PANEL.read_text(encoding="utf-8")
    assert "if (!p2)" in src, "no null guard on the P2 payload"
    assert "The P1 decision trace above is" in src or "No P2 recommendation" in src
    d = DECISION.read_text(encoding="utf-8")
    # The property is that a rejected P2 fetch cannot take the view down with
    # it, not that it is spelled with one particular `.catch`. Cockpit now
    # settles all three requests together, which isolates each failure and also
    # lets the view stop showing a spinner once they have all resolved.
    assert (".catch(() => setP2(null))" in d or "Promise.allSettled" in d),         "a failed P2 fetch must not break the view"
    assert "useState<P2Decision | null>(null)" in d


def test_the_candidate_table_sort_is_ui_only_and_documented():
    """P4 extracted the candidate table into its own component so `Cockpit.tsx`
    can reuse it without a second copy; the sort discipline must survive the move."""
    table = APP / "components" / "CandidateActionsTable.tsx"
    src = table.read_text(encoding="utf-8")
    assert "UI-ONLY sort" in src, "an undocumented re-ranking is a second opinion"
    assert ".sort(" in src
    # it must sort BY the backend value, not by a recomputed key
    i = src.index(".sort(")
    assert "value" in src[i:i + 160]
    # and P2Panel must actually use the shared component, not a second copy
    assert "CandidateActionsTable" in P2_PANEL.read_text(encoding="utf-8")


def test_the_api_client_points_at_the_canonical_endpoint():
    src = API_TS.read_text(encoding="utf-8")
    assert "/api/race/${id}/p2?car=${car}&rival=${rival}" in src


def test_the_api_endpoint_returns_exactly_the_service_output():
    """API == canonical service. No reshaping in the web layer."""
    import simulation.api.main as api_main
    from tests.test_p1_service import _payload
    from xray.decision_service import evaluate_opportunity_decision

    payload = _payload()
    canonical = evaluate_opportunity_decision(payload, "OWN", "RIV")
    src = pathlib.Path(api_main.__file__).read_text(encoding="utf-8")
    i = src.index("def p2_decision(")
    body = src[i:i + 1200]
    assert "evaluate_opportunity_decision" in body
    assert "return evaluate_opportunity_decision(" in body, (
        "the endpoint must return the service result unmodified")
    # nothing numeric in the endpoint
    assert not re.search(r"[-+*/]\s*1e6", body), "the endpoint rescales values"
    assert json.dumps(canonical, allow_nan=False)
