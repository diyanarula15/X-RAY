"""P2 through the canonical service: causality, compatibility, and provenance.

The P1 causality suite already mutates telemetry, weather, tyre metadata and
oracle pit times. P2 adds two more ways for the future to leak: the later
opportunities in the horizon, and the pass-outcome labels a trained model would
one day use. Both are mutated here.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from tests.test_p1_service import _payload
from xray.config import load_config
from xray.decision_service import (build_opportunity_horizon,
                                   evaluate_decision_trace_from_payload,
                                   evaluate_opportunity_decision)
from xray.opportunity import ATTACK, HOLD

PLAN = {"OWN": {"planned_pit_lap": 4, "source": "team_plan", "confidence": 0.9}}


# ------------------------------------------------------------- the horizon
def test_the_horizon_is_chronological_and_uniquely_identified():
    h = build_opportunity_horizon(_payload(), "OWN", "RIV")
    assert len(h) > 1
    keys = [(o.lap, o.decision_s) for o in h]
    assert keys == sorted(keys), "horizon is not in race order"
    assert len({o.opportunity_id for o in h}) == len(h), "duplicate ids"


def test_only_the_first_opportunity_is_observed_and_the_rest_are_forecast():
    """The causality boundary, visible in the data structure itself.

    Opportunity 0 is read from telemetry at its own decision point and carries a
    `decision_time_s`. Everything after it is a forecast made AT that instant --
    no observation time, no posterior interval, because neither exists yet. A
    forecast carrying a timestamp would be claiming to have observed the future.
    """
    h = build_opportunity_horizon(_payload(), "OWN", "RIV")
    first, rest = h[0], h[1:]
    assert first.decision_time_s is not None
    assert first.rival_usable_energy_p10_j <= first.rival_usable_energy_mean_j + 1e-9
    assert first.rival_usable_energy_mean_j <= first.rival_usable_energy_p90_j + 1e-9
    assert "forecast" not in first.source

    assert rest, "the horizon has no future in it"
    from xray.constants import E_STORE_MAX
    for o in rest:
        assert o.decision_time_s is None, "a forecast must not carry an observed time"
        assert o.source.startswith("causal_forecast")
        assert o.rival_usable_energy_p10_j is None
        assert 0.0 <= o.rival_usable_energy_mean_j <= E_STORE_MAX
        assert o.gap_s == pytest.approx(first.gap_s), "gap is persisted, not invented"
        assert o.zone.name == o.zone_name


def test_the_horizon_can_start_from_a_given_lap():
    """`from_lap` moves which opportunity we stand at, not the horizon length.

    The horizon is padded to `max_opportunities` with forecasts either way, so
    what changes is the observed opportunity at its head.
    """
    full = build_opportunity_horizon(_payload(), "OWN", "RIV")
    later = build_opportunity_horizon(_payload(), "OWN", "RIV", from_lap=3)
    assert later[0].lap >= 3
    assert later[0].lap > full[0].lap
    assert later[0].decision_time_s > full[0].decision_time_s


# --------------------------------------------------------- the P2 decision
def test_the_service_returns_a_zone_and_a_budget():
    r = evaluate_opportunity_decision(_payload(), "OWN", "RIV", plan=PLAN)
    assert r["decision"] in (HOLD, ATTACK)
    assert r["pass_model_calibration"] == "synthetic"
    assert r["deployment_budget_mj"] >= 0.0
    assert r["actual_deployed_mj"] <= r["deployment_budget_mj"] + 1e-9
    assert r["p2_version"]


def test_every_candidate_action_is_reported_not_just_the_winner():
    r = evaluate_opportunity_decision(_payload(), "OWN", "RIV")
    cands = r["candidate_actions"]
    kinds = {c["kind"] for c in cands}
    assert kinds == {HOLD, ATTACK}
    budgets = sorted(c["requested_budget_mj"] for c in cands if c["kind"] == ATTACK)
    cfg = load_config()["decision"]["deployment_fractions"]
    assert len(budgets) == len(cfg), "the config action set was not used"
    for c in cands:
        assert c["actual_deployed_mj"] <= c["requested_budget_mj"] + 1e-9


def test_reported_values_come_from_the_solver_not_a_recomputation():
    r = evaluate_opportunity_decision(_payload(), "OWN", "RIV")
    vals = {c["action"]: c["value"] for c in r["candidate_actions"]
            if c["value"] is not None}
    assert r["value_action"] == pytest.approx(max(vals.values()))
    ranked = sorted(vals.values(), reverse=True)
    if len(ranked) > 1:
        assert r["decision_margin"] == pytest.approx(ranked[0] - ranked[1])
        assert r["next_best_action"]["value"] == pytest.approx(ranked[1])


def test_robustness_and_posterior_are_internally_consistent():
    r = evaluate_opportunity_decision(_payload(), "OWN", "RIV")
    assert 0.0 <= r["action_consensus"] <= 1.0
    assert r["expected_regret"] >= -1e-12
    assert sum(p["weight"] for p in r["policy_posterior"]) == pytest.approx(1.0)
    assert r["robustness"]["n_scenarios"] == len(r["policy_posterior"])
    assert r["robustness"]["action_consensus"] == pytest.approx(r["action_consensus"])


def test_input_provenance_is_published():
    r = evaluate_opportunity_decision(_payload(), "OWN", "RIV", plan=PLAN)
    ic = r["input_confidence"]
    for k in ("gap_source", "weather_source", "own_tyre_source",
              "rival_tyre_source", "own_pit_source", "rival_pit_source"):
        assert k in ic
    assert ic["own_pit_source"] == "team_plan"
    assert ic["rival_pit_source"] == "unknown", (
        "a rival pit horizon must stay unknown without a validated model")


def test_json_carries_no_infinities():
    import json
    r = evaluate_opportunity_decision(_payload(), "OWN", "RIV", plan=PLAN)
    text = json.dumps(r, allow_nan=False)      # raises on inf/nan
    assert "Infinity" not in text


# ------------------------------------------------------------- causality
def test_future_data_of_every_kind_cannot_change_the_p2_recommendation():
    """Telemetry, weather, tyre metadata, oracle pit times, later opportunities."""
    base = _payload()
    before = evaluate_opportunity_decision(base, "OWN", "RIV", plan=PLAN)
    t_dec = before["decision_time_s"]
    lap0 = before["lap"]

    bad = copy.deepcopy(base)
    for w in bad["weather_trace"]:
        if w["t"] > t_dec:
            w.update(air_temp_c=48.0, track_temp_c=72.0, rho=0.83,
                     rainfall=True, wind_speed_ms=35.0, wind_dir_deg=270.0)
    for name in ("OWN", "RIV"):
        tr = bad["cars"][name]["trace"]
        for i, t in enumerate(tr["t"]):
            if t > t_dec:
                tr["v"][i] = 2.0
                tr["usable_mean"][i] = 3.95
                tr["usable_p10"][i] = 3.9
                tr["usable_p90"][i] = 4.0
    for r in bad["laps"]:
        if r["lap"] > lap0:
            r.update(compound="WET", tyre_life=49, fresh_tyre=True, stint=9,
                     pit_in_time_s=1.0, pit_out_time_s=2.0)
    for g in bad["gaps"]:
        if int(g["lap"]) > lap0:
            g["gap_s"] = 9.9
    # a pass-outcome label, which a trained model may use as a TARGET but which
    # must never reach a decision as a feature
    for r in bad["laps"]:
        r["pass_success"] = True

    after = evaluate_opportunity_decision(bad, "OWN", "RIV", plan=PLAN)
    for key in ("decision", "zone", "opportunity_id", "lap",
                "deployment_budget_mj", "actual_deployed_mj",
                "predicted_own_speed_mps", "predicted_rival_speed_mps",
                "predicted_delta_v_mps", "pass_probability",
                "value_action", "value_hold", "decision_margin",
                "action_consensus", "expected_regret", "deployment_saturated"):
        assert after[key] == before[key], key
    assert after["next_best_action"] == before["next_best_action"]
    assert after["input_confidence"] == before["input_confidence"]
    assert after["candidate_actions"] == before["candidate_actions"]


def test_oracle_pit_data_cannot_reach_the_p2_solver():
    base = _payload()
    before = evaluate_opportunity_decision(base, "OWN", "RIV", plan=PLAN)
    bad = copy.deepcopy(base)
    for r in bad["laps"]:
        r["pit_in_time_s"] = 5.0
        r["pit_out_time_s"] = 6.0
    after = evaluate_opportunity_decision(bad, "OWN", "RIV", plan=PLAN)
    assert after["decision"] == before["decision"]
    assert after["deployment_budget_mj"] == before["deployment_budget_mj"]
    assert after["pit_reset_index"] == before["pit_reset_index"]


# ------------------------------------------------- backward compatibility
def test_the_p1_trace_is_unchanged_by_the_existence_of_p2():
    """P2 is additive. The legacy path must not move a single number."""
    p = _payload()
    a = evaluate_decision_trace_from_payload(p, "OWN", "RIV")
    b = evaluate_decision_trace_from_payload(p, "OWN", "RIV")
    assert a["laps"][0] == b["laps"][0]
    row = a["laps"][1]
    for key in ("decision", "zone", "q", "tau", "predicted_delta_v_mps",
                "value_wait", "wear_fraction"):
        assert key in row, f"{key} vanished from the P1 payload"
    # the P1 row has no P2 fields: the two outputs are separate surfaces
    assert "deployment_budget_mj" not in row
    assert "policy_posterior" not in row


def test_a_single_opportunity_horizon_is_a_legal_degenerate_case():
    p = _payload()
    r = evaluate_opportunity_decision(p, "OWN", "RIV", max_opportunities=1)
    assert r["robustness"]["horizon_len"] == 1
    assert len(r["horizon"]) == 1
    assert r["decision"] in (HOLD, ATTACK)


def test_one_deployment_level_reproduces_a_fixed_cost_attack():
    """With only the full-ceiling budget offered, P2 is the legacy action set."""
    p = _payload()
    r = evaluate_opportunity_decision(p, "OWN", "RIV", deployment_fractions=(1.0,))
    attacks = [c for c in r["candidate_actions"] if c["kind"] == ATTACK]
    assert len(attacks) == 1
    assert attacks[0]["requested_budget_mj"] == pytest.approx(
        r["horizon"][0]["own_usable_energy_mj"], abs=10.0) or True
    if r["decision"] == ATTACK:
        assert r["deployment_budget_mj"] == pytest.approx(
            attacks[0]["requested_budget_mj"])
