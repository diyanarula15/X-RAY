"""P3 Part 3: the evidence layer. Statuses, provenance, service, UI.

What these tests defend is not a number but an honesty property: the system must
be unable to present the energy estimator as working, because it has not been
shown to work on real data. The registry is the single source of every status
word, and the UI is forbidden from deriving one.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

from xray import registry as reg
from xray.decision_service import (historical_replay, p3_race_evidence,
                                   p3_status)

APP = pathlib.Path(__file__).resolve().parent.parent / "simulation" / "app" / "src"
P3_PANEL = APP / "components" / "P3Panel.tsx"
API_TS = APP / "lib" / "api.ts"
DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"


# ---------------------------------------------------------------- the registry
def test_every_entry_has_a_legal_status_and_a_reason_when_it_is_not_live():
    entries = reg.registry()
    assert entries
    for e in entries:
        assert e.status in reg.STATUSES
        if not e.production:
            assert e.reason, f"{e.component} is not production and says no why"


def test_the_registry_covers_every_component_the_spec_names():
    have = {e.component for e in reg.registry()}
    required = {
        "energy_inference", "stage1_estimator", "setmem_rbpf", "pass_model",
        "tyres", "cda", "cla", "wake", "wetness", "wind", "rival_pit",
        "energy_forecast", "regulation_variant", "mass_model",
        "deployment_ceiling",
    }
    assert required <= have, required - have


def test_there_is_no_status_word_that_claims_validation():
    """`VALIDATED` is absent deliberately: no single word is both honest and short."""
    assert "VALIDATED" not in reg.STATUSES
    for e in reg.registry():
        assert e.status != "VALIDATED"


def test_the_noncanonical_stacks_are_research_only_and_not_production():
    by = {e.component: e for e in reg.registry()}
    for c in ("stage1_estimator", "setmem_rbpf"):
        assert by[c].status == reg.RESEARCH_ONLY
        assert by[c].production is False
        assert by[c].reason


def test_energy_inference_states_both_halves_and_never_only_the_good_one():
    s = reg.energy_inference_status()
    assert s["status"] == reg.INFERRED
    assert s["real_validation"]["result"] == reg.NO_ROBUST_IMPROVEMENT
    assert "NOT DEMONSTRATED" in s["headline"]
    assert s["real_ground_truth"]["available"] is False
    assert s["identifiability"]["status"] == reg.IDENT_LIMITED


def test_a_failed_check_and_an_unattempted_check_are_distinguishable():
    """Collapsing these two into one falsy value loses the engineering difference."""
    s = reg.energy_inference_status()
    real = s["real_validation"]
    assert real["available"] is True and real["result"] != reg.NOT_ATTEMPTED
    by = {e.component: e for e in reg.registry()}
    never = by["stage1_estimator"].validation
    assert never.available is False and never.result == reg.NOT_ATTEMPTED


def test_metrics_come_from_an_artifact_and_carry_its_fingerprint():
    s = reg.energy_inference_status()
    for half in ("synthetic_validation", "real_validation"):
        v = s[half]
        assert v["available"], half
        assert v["fingerprint"] and len(v["fingerprint"]) >= 16, half
        assert v["metrics"], half


def test_the_registry_declines_rather_than_remembering_numbers(tmp_path, monkeypatch):
    """No artifact on disk must mean NOT_ATTEMPTED, never a quoted stale number."""
    monkeypatch.setenv("XRAY_P3_ARTIFACTS", str(tmp_path))
    reg._registry_payload.cache_clear()
    try:
        s = reg.energy_inference_status()
        assert s["synthetic_validation"]["result"] == reg.NOT_ATTEMPTED
        assert s["real_validation"]["result"] == reg.NOT_ATTEMPTED
        assert s["synthetic_validation"]["metrics"] == {}
        # the headline never softens just because the evidence went missing
        assert "NOT DEMONSTRATED" in s["headline"]
    finally:
        reg._registry_payload.cache_clear()


def test_the_real_ablation_verdict_matches_the_measured_differences():
    """The verdict must follow the artifact, not a remembered conclusion.

    Lower MAE is better, so a POSITIVE xray_minus_fixed_mae is X-RAY losing. It
    loses on 2 of 3 targets, which is why the verdict is NO_ROBUST_IMPROVEMENT.
    """
    m = reg.energy_inference_status()["real_validation"]["metrics"]
    diffs = m["xray_minus_fixed_mae"]
    losses = [t for t, v in diffs.items() if v > 0]
    assert len(losses) >= 2, diffs
    assert len(losses) < len(diffs), "a clean sweep would need a different verdict"


def test_the_data_quality_limitations_name_the_negative_result():
    text = " ".join(reg.DATA_QUALITY_LIMITATIONS)
    for phrase in ("NO REAL BATTERY GROUND TRUTH", "NO TRACK STATUS",
                   "NO WEATHER TRACE", "NO PIT DATA",
                   "NO GENUINE SYNTHETIC LIFT-AND-COAST",
                   "CDA POORLY IDENTIFIABLE",
                   "DOES NOT SHOW ROBUST REAL PREDICTIVE VALUE"):
        assert phrase in text, phrase


# -------------------------------------------------------------- the service
def test_p3_status_is_json_serialisable_and_carries_no_nans():
    assert json.dumps(p3_status(), allow_nan=False)


def test_the_service_does_not_ship_the_validation_dataset():
    """Fingerprints and summaries only. 48,920 scored examples stay on disk."""
    s = json.dumps(p3_status())
    assert len(s) < 200_000, f"p3_status payload is {len(s)} bytes"
    m = p3_status()["energy_inference"]["real_validation"]["metrics"]
    assert "examples" not in m and "per_example" not in m


def test_the_service_status_words_are_exactly_the_registry_words():
    s = p3_status()
    live = {e["component"]: e["status"] for e in s["registry"]["entries"]}
    canonical = {e.component: e.status for e in reg.registry()}
    assert live == canonical


@pytest.fixture(scope="module")
def real_payload():
    root = pathlib.Path(__file__).resolve().parent.parent / "out" / "races"
    paths = sorted(root.glob("*.json"))
    if not paths:
        pytest.skip("no real race payload in out/races")
    with open(paths[0]) as fh:
        return json.load(fh)


def test_race_evidence_states_the_absence_of_ground_truth_explicitly(real_payload):
    """An omitted field reads as "not looked up"; an explicit False reads as fact."""
    ev = p3_race_evidence(real_payload)
    assert ev["real_ground_truth_available"] is False
    assert ev["real_ground_truth_reason"]
    assert ev["corpus"]["has_track_status"] is False
    assert ev["corpus"]["has_pit_data"] is False


def test_the_replay_is_labelled_off_policy_permanently(real_payload):
    cars = sorted(real_payload["cars"])
    t = real_payload["cars"][cars[0]]["trace"]["t"]
    r = historical_replay(real_payload, cars[0], cars[1], float(t[len(t) // 3]))
    assert r["label"] == "OFF-POLICY HISTORICAL REPLAY — NOT COUNTERFACTUAL PROOF"
    assert "evaluation only" in r["later_telemetry_use"]
    assert "NOT DEMONSTRATED" in r["energy_inference_status"]


def test_the_replay_decision_sees_nothing_after_the_cutoff(real_payload):
    """The causal wall, asserted rather than assumed.

    Moving the cutoff must move the recommendation's inputs; later telemetry must
    reach only `later_observable_outcome`.
    """
    cars = sorted(real_payload["cars"])
    a, b = cars[0], cars[1]
    t = real_payload["cars"][a]["trace"]["t"]
    early = historical_replay(real_payload, a, b, float(t[len(t) // 3]))
    late = historical_replay(real_payload, a, b, float(t[2 * len(t) // 3]))
    n_early = early["information_at_cutoff"]["n_samples"][a]
    n_late = late["information_at_cutoff"]["n_samples"][a]
    assert n_early and n_late and n_late > n_early
    assert early["information_at_cutoff"]["input_fingerprint"] != \
        late["information_at_cutoff"]["input_fingerprint"]
    # and the snapshot really did truncate: no input sample past the cutoff
    assert early["p2_error"] is None, early["p2_error"]
    assert early["inferred_energy_mj"] is not None


def test_the_replay_reports_a_real_observed_future(real_payload):
    """A silently empty future window would make the replay look causal and be empty."""
    cars = sorted(real_payload["cars"])
    t = real_payload["cars"][cars[0]]["trace"]["t"]
    r = historical_replay(real_payload, cars[0], cars[1], float(t[len(t) // 3]))
    out = r["later_observable_outcome"][cars[0]]
    assert out["n_samples"] > 0
    assert out["v_max_mps"] is not None, "future speeds must be read, not dropped"


# ------------------------------------------------------------------- the UI
def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("//"))


def test_the_p3_panel_computes_no_inference_decision_or_validation_logic():
    banned = {
        r"Math\.exp\s*\(\s*-": "a logistic",
        r"\bp_pass\b": "a pass-probability formula",
        r"delta_v\s*=\s*[^=]": "a delta_v computation",
        r"\bbellman\b": "a Bellman step",
        r"usable_energy\s*=\s*[^=]": "an energy computation",
        r"(?<![\w.])status\s*=\s*['\"]": "a hard-coded status word",
        r"NO_ROBUST_IMPROVEMENT\s*=": "a derived verdict",
        r"<\s*0\.0[0-9]": "a numeric validation threshold",
    }
    src = _strip_comments(P3_PANEL.read_text(encoding="utf-8"))
    for pattern, what in banned.items():
        assert not re.search(pattern, src), f"P3Panel contains {what}"


def test_the_panel_renders_the_negative_result_unconditionally():
    """Not a tooltip, not behind a disclosure, not conditional on a flag."""
    src = P3_PANEL.read_text(encoding="utf-8")
    assert "CURRENT INFERRED ENERGY DOES NOT ADD ROBUST HELD-OUT PREDICTIVE VALUE" in src
    assert "NOT AVAILABLE" in src          # true battery energy
    head = src.split("DOES NOT ADD ROBUST")[0][-500:]
    assert "title=" not in head, "the negative result must not be a tooltip"


def test_the_panel_never_shows_an_energy_accuracy_percentage():
    """"Energy accuracy: 94%" is the exact claim the project cannot make."""
    src = _strip_comments(P3_PANEL.read_text(encoding="utf-8"))
    assert not re.search(r"accuracy", src, re.I)


def test_the_panel_explains_the_sign_convention_of_the_ablation():
    """A bare "+0.437" next to "X-RAY" reads as a gain unless the sign is stated."""
    src = P3_PANEL.read_text(encoding="utf-8")
    assert "lower is better" in src
    assert "POSITIVE" in src


def test_the_panel_says_a_status_is_not_a_quality_claim():
    src = P3_PANEL.read_text(encoding="utf-8")
    assert "never how good it is" in src or "not \"verified\"" in src


def test_the_replay_panel_carries_the_off_policy_label_and_the_evaluation_rule():
    src = P3_PANEL.read_text(encoding="utf-8")
    assert "replay.label" in src
    assert "only for evaluation" in src
    assert "did not execute" in src


def test_the_api_client_declares_only_fields_the_service_returns():
    s = p3_status()
    src = API_TS.read_text(encoding="utf-8")
    block = src[src.index("export type P3Status = {"):]
    block = block[:block.index("\n};")]
    declared = set(re.findall(r"^\s{2}([a-z_][a-z0-9_]*)\s*:", block, re.M))
    missing = declared - set(s)
    assert not missing, f"api.ts declares fields the service does not return: {missing}"


def test_the_panels_are_wired_into_the_existing_views_not_a_new_app():
    """P4 redistributed these panels across the new IA -- energy status beside
    the energy numbers (Energy tab), replay inside the interactive scenario
    walkthrough (Replay tab), and validation/registry status onto their own
    Evidence tab -- rather than leaving two P3 panels crammed into Decision/
    Method as before. An exported panel nobody mounts is still not a surface."""
    # The Energy tab was cut (its radar computed its own axes in TypeScript from
    # invented constants) and Method was merged into Evidence, so the energy
    # status, the validation registry and the limits narrative now share one
    # view; the scenario walkthrough became its own top-level Situations tab.
    # The panels still have to be MOUNTED somewhere -- an exported panel nobody
    # mounts is still not a surface, which is what this has always asserted.
    evidence = (APP / "views" / "Evidence.tsx").read_text(encoding="utf-8")
    situations = (APP / "views" / "Situations.tsx").read_text(encoding="utf-8")
    assert "EnergyStatusPanel" in evidence and "api.p3Status()" in evidence
    assert "ReplayPanel" in situations and "api.replay(" in situations
    for name in ("ValidationPanel", "ModelStatusPanel", "DataQualityPanel"):
        assert name in evidence, name


def test_the_api_exposes_the_p3_endpoints_without_reshaping_them():
    main = pathlib.Path(__file__).resolve().parent.parent / "simulation" / "api" / "main.py"
    src = main.read_text(encoding="utf-8")
    for fn, call in (("p3_status_endpoint", "return p3_status()"),
                     ("p3_race_endpoint", "return p3_race_evidence("),
                     ("p3_replay_endpoint", "return historical_replay(")):
        i = src.index(f"def {fn}(")
        body = src[i:i + 400]
        assert call in body, f"{fn} must return the service result unmodified"


# ----------------------------------------------------------- documentation
def test_the_pipeline_doc_shows_the_unified_path_and_the_research_only_stacks():
    src = (DOCS / "pipeline_layers.md").read_text(encoding="utf-8")
    for token in ("grid_lap", "realfit", "decision_service", "RESEARCH_ONLY"):
        assert token in src, token


def test_the_docs_record_the_mass_decision_as_measured_but_not_adopted():
    src = (DOCS / "model_inventory.md").read_text(encoding="utf-8")
    assert "MEASURED — NOT ADOPTED" in src
    assert "790" in src
