"""Decision-engine acceptance tests."""
from __future__ import annotations

import numpy as np

from xray.decision import (blind_chooser, build_model, compare_exogenous,
                           compare_policies, delta_v, explain_exogenous_action,
                           policy_posterior, rival_energy_at_zone, robustness,
                           simulate_stint, solve, solve_exogenous)
from xray.overtake import COEFFS, p_pass
from xray.sim import FOLLOWER, LEADER
from xray.vehicle import VehicleParams

RATE = 3.7


def _setup(cfg, races, seed=42):
    g = races[seed]
    obs, bel = None, None
    from xray.estimator import estimate
    from xray.observe import observe
    obs = observe(g, LEADER, rate_hz=RATE, seed=seed + 1)
    bel = estimate(obs, g.track, n_particles=200, seed=seed + 2)
    params = VehicleParams.from_config(cfg)
    model = build_model(
        g.track, params, n_laps=g.n_laps,
        recharge_per_lap=float(np.mean(bel.harvested_lap[bel.harvested_lap > 0])),
        rival_spend_per_lap=float(np.mean(bel.deployed_lap[bel.deployed_lap > 0])),
        own_spend_per_lap=float(np.mean(g.cars[FOLLOWER].deployed_lap)) * 0.55)
    believed = np.array([float(bel.usable_mean[obs.lap == L].min())
                         if (obs.lap == L).any() else 0.0 for L in range(g.n_laps)])
    return g, obs, bel, model, solve(model), believed


def test_pass_model_shape():
    """The two anchors the brief specifies, and the ordering between zones."""
    class Z:
        def __init__(self, s):
            self.braking_severity = s
    assert p_pass(8.0, 0.3, Z(1.0)) == pytest.approx(0.70, abs=0.01)
    assert p_pass(8.0, 0.3, Z(0.2)) == pytest.approx(0.25, abs=0.01)
    assert p_pass(8.0, 0.3, Z(1.0)) > p_pass(8.0, 0.3, Z(0.55)) > p_pass(8.0, 0.3, Z(0.2))
    # monotone in closing speed and in proximity
    assert p_pass(2.0, 0.3, Z(1.0)) < p_pass(9.0, 0.3, Z(1.0))
    assert p_pass(6.0, 1.4, Z(1.0)) < p_pass(6.0, 0.1, Z(1.0))


def test_pass_model_gap_sensitivity():
    class Z:
        braking_severity = 1.0
    assert p_pass(6.0, 0.2, Z()) > p_pass(6.0, 1.0, Z()) > p_pass(6.0, 1.5, Z())


import pytest  # noqa: E402


def test_zone_calibration_comes_from_the_vehicle_model(cfg, races):
    """Energy -> end-of-straight speed must be measured, not assumed."""
    g = races[42]
    params = VehicleParams.from_config(cfg)
    model = build_model(g.track, params, n_laps=g.n_laps, recharge_per_lap=2.2e6,
                        rival_spend_per_lap=2.6e6)
    for zm in model.zones:
        assert np.all(np.diff(zm.speed_grid) >= -0.5), "more energy must not be slower"
        assert zm.speed_grid[-1] > zm.speed_grid[0]
        assert zm.energy_grid[-1] > 0.5e6
    # Zone A runs closest to terminal speed, so energy buys the least there
    by_name = {zm.name: zm for zm in model.zones}
    assert by_name["A"].dv_per_mj < by_name["C"].dv_per_mj


def test_threshold_is_nontrivial_and_decays_through_the_stint(cfg, races):
    _g, _o, _b, model, sol, _bel = _setup(cfg, races)
    tau = sol.tau[1:, 10]
    assert tau.max() > 0.02, f"threshold collapsed to always-attack: {tau}"
    # more laps left -> be pickier
    assert tau[-1] > tau[0]


def test_decision_beats_blind(cfg, races):
    """Over 50 seeded stints, positions gained is positive with 95% confidence.

    Both rules are played against the same rival energy schedule -- the one the
    estimator reconstructed -- with the same coin flips. The only difference is
    what each rule knows.
    """
    g, obs, bel, model, _sol, _believed = _setup(cfg, races)
    rival_track = rival_energy_at_zone(bel, obs, g.track, g.n_laps)
    sol = solve_exogenous(model, rival_track)
    res = compare_exogenous(model, sol, rival_track, n_races=50, n_laps=g.n_laps,
                            e_own0=float(g.cars[FOLLOWER].E[0]), seed=3)
    assert res["mean_gain"] > 0, res
    assert res["ci95"][0] > 0, f"95% CI includes zero: {res}"


def test_decision_waits_when_the_rival_is_strong(cfg, races):
    """The point of the whole thing: it must be able to say 'not yet'."""
    g, obs, bel, model, _sol, _believed = _setup(cfg, races)
    rival_track = rival_energy_at_zone(bel, obs, g.track, g.n_laps)
    sol = solve_exogenous(model, rival_track)
    e_own = float(g.cars[FOLLOWER].E.max())
    calls = [sol.action(g.n_laps - L, e_own) for L in range(g.n_laps)]
    assert any(c is None for c in calls), f"never holds fire: {calls}"
    assert any(c is not None for c in calls), f"never attacks: {calls}"
    # the rival is strongest on the opening lap, so that is when to hold
    assert rival_track[0] == rival_track.max()
    assert calls[0] is None


def test_explanation_matches_exogenous_solver_choice(cfg, races):
    g, obs, bel, model, _sol, _believed = _setup(cfg, races)
    rival_track = rival_energy_at_zone(bel, obs, g.track, g.n_laps)
    sol = solve_exogenous(model, rival_track)
    e_own = float(g.cars[FOLLOWER].E.max())
    laps_left = 5
    detail = explain_exogenous_action(sol, laps_left, e_own)
    assert detail["best_zone"] == sol.action(laps_left, e_own)
    best_attack_value = max(z["value_attack"] for z in detail["zones"])
    assert detail["best_attack"]["value_attack"] == pytest.approx(best_attack_value)
    if detail["decision"] == "ATTACK":
        assert best_attack_value > detail["value_wait"]
    else:
        assert best_attack_value <= detail["value_wait"]


def test_robustness_is_computed_not_asserted(cfg, races):
    _g, _o, bel, model, _sol, believed = _setup(cfg, races)
    pols = policy_posterior(bel, _g.track, n_samples=40, seed=1)
    assert len({(round(p.front_loading, 3), round(p.reserve, 3)) for p in pols}) > 5
    rb = robustness(model, pols, laps_left=5, e_own=3.0e6, e_riv=0.8e6)
    assert 0.0 <= rb["fraction_agreeing"] <= 1.0
    assert len(rb["actions"]) == len(pols)


def test_stint_respects_the_energy_budget(cfg, races):
    _g, _o, _b, model, sol, _bel = _setup(cfg, races)
    rng = np.random.default_rng(0)
    out = simulate_stint(model, blind_chooser(model), rng, model.n_laps, 0.0, 3.0e6)
    assert out["passed"] in (0, 1)
    # with an empty battery and no recovery surplus the blind rule cannot attack
    # on lap 1; it must wait at least until it can afford the attack
    assert out["lap_passed"] is None or out["lap_passed"] >= 1


def test_explanation_cannot_disagree_with_the_solver_at_any_state(cfg, races):
    """The explanation must be able to reproduce the choice, not just sit beside it.

    The helper originally left out the solver's affordability mask, so for any
    energy below 0.75 * attack_cost it published a finite `value_attack` for an
    attack the DP had already rejected -- V_attack > V_wait printed next to HOLD.
    Swept over the whole grid rather than spot-checked, because the disagreement
    only existed in the bins a single sample is least likely to land in.
    """
    g, obs, bel, model, _sol, _believed = _setup(cfg, races)
    sol = solve_exogenous(model, rival_energy_at_zone(bel, obs, g.track, g.n_laps))
    checked_hold = checked_attack = 0
    for laps_left in range(1, min(6, sol.V.shape[0])):
        for e_own in np.linspace(0.0, 4.0e6, 25):
            d = explain_exogenous_action(sol, laps_left, float(e_own))
            assert d["best_zone"] == sol.action(laps_left, float(e_own))
            assert d["attack_threshold"] == pytest.approx(
                sol.threshold(laps_left, float(e_own)))
            v_att = max(z["value_attack"] for z in d["zones"])
            assert d["value_attack"] == v_att
            if d["decision"] == "ATTACK":
                assert v_att > d["value_wait"]
                assert d["attack_affordable"]
                assert np.isfinite(v_att)
                checked_attack += 1
            else:
                assert v_att <= d["value_wait"]
                checked_hold += 1
    assert checked_hold and checked_attack, "sweep covered only one branch"


def test_unaffordable_attack_is_reported_as_rejected_not_as_valuable(cfg, races):
    """An attack the DP masked to -inf must not surface as a finite value."""
    g, obs, bel, model, _sol, _believed = _setup(cfg, races)
    sol = solve_exogenous(model, rival_energy_at_zone(bel, obs, g.track, g.n_laps))
    d = explain_exogenous_action(sol, 5, 0.0)
    assert not d["attack_affordable"]
    assert d["decision"] == "HOLD"
    assert d["value_attack"] == -np.inf
    # the hypothetical is still published, and is a real number
    assert np.isfinite(d["best_attack"]["value_attack_hypothetical"])
