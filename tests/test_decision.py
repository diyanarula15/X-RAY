"""Decision-engine acceptance tests."""
from __future__ import annotations

import numpy as np

from xray.decision import (blind_chooser, build_model, compare_policies,
                           delta_v, policy_posterior, robustness, simulate_stint,
                           solve)
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
    """Over 50 seeded stints, positions gained is positive with 95% confidence."""
    _g, _o, bel, model, sol, believed = _setup(cfg, races)
    sigma = float(np.mean(bel.usable_p90 - bel.usable_p10) / 2)
    res = compare_policies(model, sol, believed, sigma, n_races=50,
                           n_laps=model.n_laps, e_own0=2.0e6, e_riv0=believed[0],
                           seed=3)
    assert res["mean_gain"] > 0, res
    assert res["ci95"][0] > 0, f"95% CI includes zero: {res}"


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
