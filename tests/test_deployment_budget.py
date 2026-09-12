"""P3 §1-2: the deployment budget means joules the car can actually deploy.

The discrepancy this file closes: P2's audit reported a 1.602 MJ calibrated
budget for Circuit Sigma zone A against roughly 0.449 MJ "executable". Three
separate causes, and the tests below pin each one so it cannot come back.

1. WINDOW. `calibrate_zone` sweeps energy over run-up PLUS straight -- 1700 m
   from the previous corner exit at 200 km/h for zone A -- and the low-speed part
   of that run sits under the full 350 kW. An attack in-race spans the 1100 m
   straight only.
2. ENTRY SPEED. The car arrives at zone A's straight at 314 km/h and crosses
   345 km/h after 387 m, beyond which the normal MGU-K curve is exactly 0 kW. No
   budget buys a joule over the remaining 660 m. 0.448 MJ, not 1.602 MJ.
3. ACCOUNTING. The closed-loop budget was charged against store DRAWDOWN, and a
   deployment straight ends in a braking zone, so the store refills while the
   attack is still running. Drawdown across zone A runs
   0 -> 0.4281 -> 0.4492 -> 0.4492 -> -0.1353 MJ while delivered energy rises
   monotonically to 0.4564 MJ.

The first two are physics and are now exposed to P2 as a measured ceiling. The
third was a bug and is fixed.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from xray.config import load_config
from xray.constants import E_STORE_MAX, P_MGUK_MAX, p_mguk_ceiling
from xray.decision import calibrate_zone, executable_attack_ceiling
from xray.decision_service import (build_opportunity_horizon,
                                   executable_ceiling_at_opportunity,
                                   evaluate_opportunity_decision,
                                   speed_map_cache_key, params_from_payload)
from xray.opportunity import (ATTACK, HOLD, enumerate_actions, evaluate_action)
from xray.sim import circuit_sigma
from xray.vehicle import VehicleParams


@pytest.fixture(scope="module")
def sigma():
    cfg = load_config()
    return circuit_sigma(), VehicleParams.from_config(cfg)


@pytest.fixture(scope="module")
def horizon():
    from tests.test_p1_service import _payload
    pl = _payload()
    return pl, build_opportunity_horizon(pl, "OWN", "RIV", None, None, 12)


# --------------------------------------------------------- the physics itself
def test_the_executable_ceiling_falls_as_entry_speed_rises(sigma):
    """The whole reason a per-zone constant cannot express this."""
    track, params = sigma
    zone = next(z for z in track.zones if z.name == "A")
    ceilings = [executable_attack_ceiling(track, params, zone, v / 3.6
                                          )["executable_ceiling_j"]
                for v in (200.0, 250.0, 314.0)]
    assert ceilings == sorted(ceilings, reverse=True), ceilings
    # and the spread is large enough that treating them as one number is wrong
    assert ceilings[0] > 3.0 * ceilings[-1]


def test_the_taper_zeroes_inside_the_attack_window_at_race_entry_speed(sigma):
    """What actually caps zone A: the rules run out before the straight does."""
    track, params = sigma
    zone = next(z for z in track.zones if z.name == "A")
    r = executable_attack_ceiling(track, params, zone, 314.0 / 3.6)
    assert r["taper_zero_after_m"] is not None
    assert r["taper_zero_after_m"] < 0.5 * r["window_m"], r
    assert p_mguk_ceiling(r["exit_v_mps"]) >= 0.0


def test_the_calibration_axis_is_a_different_run_not_a_bigger_ceiling(sigma):
    """`energy_grid.max()` is not an upper bound on an in-race attack.

    It is larger, but for a reason that has nothing to do with headroom: it
    covers a longer window entered far slower.
    """
    track, params = sigma
    zone = next(z for z in track.zones if z.name == "A")
    axis = float(np.max(calibrate_zone(track, params, zone).energy_grid))
    race = executable_attack_ceiling(track, params, zone,
                                    314.0 / 3.6)["executable_ceiling_j"]
    assert race < 0.5 * axis
    # entered at the speed the CALIBRATION uses, the same window agrees with it
    slow = executable_attack_ceiling(track, params, zone,
                                    200.0 / 3.6)["executable_ceiling_j"]
    assert slow > 2.0 * race


def test_the_ceiling_is_measured_through_the_shared_integrator_not_integrated(sigma):
    """Integrating the taper over the window overstates it; that is why we don't.

    The taper is not exogenous -- deploying harder raises v, which lowers the
    next step's ceiling -- and the integral also counts the braking stretch where
    the car draws nothing.
    """
    track, params = sigma
    zone = next(z for z in track.zones if z.name == "A")
    from xray.vehicle import CarState, step, set_car_mass
    set_car_mass(params.mass_car)
    st = CarState(s=float(zone.s_straight_start), s_total=0.0, v=314.0 / 3.6,
                  E=E_STORE_MAX, fuel=params.fuel_start * 0.5)
    run_m = zone.s_straight_end - zone.s_straight_start
    naive = 0.0
    while st.s_total < run_m:
        naive += float(p_mguk_ceiling(st.v)) * 0.005
        step(track, st, params, P_MGUK_MAX, 0.005)
    measured = executable_attack_ceiling(track, params, zone,
                                        314.0 / 3.6)["executable_ceiling_j"]
    assert naive > 1.5 * measured, (naive, measured)


# ------------------------------------------------- what P2 is allowed to offer
def test_p2_offers_only_budgets_the_car_can_execute(horizon):
    _, h = horizon
    for opp in h:
        assert opp.deploy_ceiling_is_measured, opp.opportunity_id
        for a in enumerate_actions(opp):
            if a.kind == ATTACK:
                assert a.deployment_budget_j <= opp.deploy_ceiling_j + 1e-6


def test_a_larger_budget_never_permits_less_deployment(horizon):
    """Monotonicity. A budget axis that is not monotone is not a budget."""
    _, h = horizon
    for opp in h[:3]:
        dep = [evaluate_action(opp, a).actual_deployed_j
               for a in enumerate_actions(opp) if a.kind == ATTACK]
        assert dep == sorted(dep), (opp.opportunity_id, dep)


def test_deployment_respects_ceiling_store_and_request_together(horizon):
    _, h = horizon
    opp = h[0]
    for a in enumerate_actions(opp):
        out = evaluate_action(opp, a)
        assert out.actual_deployed_j <= opp.deploy_ceiling_j + 1e-6
        assert out.actual_deployed_j <= opp.own_usable_energy_j + 1e-6
        assert out.actual_deployed_j <= out.requested_budget_j + 1e-6
        assert out.actual_deployed_j <= E_STORE_MAX + 1e-6


def test_a_budget_above_the_ceiling_is_saturated_explicitly_not_silently(horizon):
    from xray.opportunity import DecisionAction
    _, h = horizon
    opp = h[0]
    over = DecisionAction(ATTACK, opp.zone_name,
                          opp.deploy_ceiling_j * 2.0)
    out = evaluate_action(opp, over)
    if out.feasible:
        assert out.saturated and out.saturation_reason
        assert out.actual_deployed_j < over.deployment_budget_j


def test_hold_deploys_nothing_and_is_unaffected_by_our_own_ceiling(horizon):
    """The bug this caught: `min(rival_energy, our_ceiling)`.

    Tightening zone A's ceiling from the calibration axis (1.422 MJ) to the
    measured executable value (0.354 MJ) moved HOLD's delta_v from -10.55 to
    -2.31 m/s on this fixture, purely by throttling the rival with OUR number.
    """
    _, h = horizon
    opp = h[0]
    hold = next(a for a in enumerate_actions(opp) if a.kind == HOLD)
    a = evaluate_action(opp, hold)
    b = evaluate_action(dataclasses.replace(opp, executable_ceiling_j=None), hold)
    assert a.actual_deployed_j == 0.0
    assert a.delta_v_mps == pytest.approx(b.delta_v_mps, abs=1e-9)
    assert a.rival_speed_mps == pytest.approx(b.rival_speed_mps, abs=1e-9)


def test_the_rival_ceiling_comes_from_the_rival_entry_speed(horizon):
    _, h = horizon
    opp = h[0]
    assert opp.rival_deploy_ceiling_j is not None
    faster = dataclasses.replace(
        opp, rival_executable_ceiling_j=opp.rival_deploy_ceiling_j * 0.25)
    hold = next(a for a in enumerate_actions(opp) if a.kind == HOLD)
    # a throttled rival is slower; our own speed is untouched
    a, b = evaluate_action(opp, hold), evaluate_action(faster, hold)
    assert b.rival_speed_mps <= a.rival_speed_mps + 1e-9
    assert b.own_speed_mps == pytest.approx(a.own_speed_mps, abs=1e-9)


def test_no_entry_speed_declines_rather_than_guessing(horizon):
    """A guessed entry speed would set the ceiling almost by itself."""
    pl, _ = horizon
    key = speed_map_cache_key(pl, params_from_payload(pl))
    assert executable_ceiling_at_opportunity(key, "A", None) is None
    assert executable_ceiling_at_opportunity(key, "A", float("nan")) is None
    assert executable_ceiling_at_opportunity(key, "A", 0.0) is None
    assert executable_ceiling_at_opportunity(key, "A", 87.0) is not None


def test_without_a_measured_ceiling_the_axis_is_used_and_says_so(horizon):
    _, h = horizon
    bare = dataclasses.replace(h[0], executable_ceiling_j=None)
    assert not bare.deploy_ceiling_is_measured
    assert bare.deploy_ceiling_j == pytest.approx(float(np.max(bare.zone.energy_grid)))


def test_the_service_reports_a_budget_it_can_deploy(horizon):
    pl, _ = horizon
    r = evaluate_opportunity_decision(pl, "OWN", "RIV")
    if r["decision"] == "ATTACK":
        assert r["actual_deployed_mj"] <= r["deployment_budget_mj"] + 1e-9
    for c in r["candidate_actions"]:
        assert c["actual_deployed_mj"] <= c["requested_budget_mj"] + 1e-9


# -------------------------------------------------- the closed-loop accounting
@pytest.fixture(scope="module")
def episode_world():
    from xray.closedloop import world
    return world(load_config(), seed=42)


# One 12-lap simulation per (budget, lap, zone), shared across the tests below.
# Without this the file ran eleven identical races and dominated the suite.
_RUNS: dict = {}


def _run(cfg, budget_mj, lap=5, zone="A"):
    key = (budget_mj, lap, zone)
    if key in _RUNS:
        return _RUNS[key]
    from xray.closedloop import AttackPlan, policy_for
    from xray.policy import get_policy
    from xray.sim import Simulator
    base = get_policy(cfg["sim"]["follower_policy"])
    plan = AttackPlan(source="probe", lap=lap, zone=zone,
                      budget_j=None if budget_mj is None else budget_mj * 1e6)
    pol = policy_for(base, plan)
    gt = Simulator(cfg, policies={"FOLLOWER": pol,
                                  "LEADER": get_policy(cfg["sim"]["leader_policy"])},
                   seed=42).run()
    _RUNS[key] = (gt, pol)
    return gt, pol


def test_store_drawdown_is_not_a_deployment_measurement(episode_world):
    """The bug, stated as a measurement.

    The straight ends in a braking zone, so the store refills while the attack
    is still running: drawdown peaks mid-straight and ends NEGATIVE. A budget
    compared against it stops binding, and the peak of that curve is what the P2
    audit reported as zone A's executable energy.
    """
    gt, _ = _run(episode_world, 0.4)
    tr = gt.cars["FOLLOWER"]
    lap = np.asarray(tr.lap, dtype=int)
    s = np.asarray(tr.s, dtype=float)
    m = np.flatnonzero((lap == 5) & (s >= 0.0) & (s <= 1100.0))
    assert len(m) > 50
    E = np.asarray(tr.E, dtype=float)[m]
    drawdown = E[0] - E
    assert drawdown.min() < -1e4, "expected the store to refill inside the zone"
    assert drawdown[-1] < drawdown.max(), "drawdown must be non-monotone here"


def test_the_budget_is_charged_against_delivered_energy(episode_world):
    """Delivered energy is monotone where drawdown is not, so it can be a budget."""
    _, pol = _run(episode_world, 0.4)
    assert pol.deployed_during_attack() > 0.0
    # the policy's own tally agrees with the simulator's per-lap deployment
    gt, pol2 = _run(episode_world, 1.6)
    assert pol2.deployed_during_attack() == pytest.approx(
        float(gt.cars["FOLLOWER"].deployed_lap[5]), rel=1e-6)


def test_a_larger_budget_never_deploys_less_in_the_closed_loop(episode_world):
    dep = [float(_run(episode_world, b)[0].cars["FOLLOWER"].deployed_lap[5])
           for b in (0.2, 0.8, 1.6, 4.0)]
    assert dep == sorted(dep), dep


def test_an_attack_never_suppresses_normal_deployment(episode_world):
    """A small allocation must not become a LIFT.

    Cutting to zero once made a 0.400 MJ "attack" cap deployment below what the
    driver would have used anyway and dropped peak speed from 365.3 to
    364.9 km/h. The allocation is a floor while it lasts, never a ceiling.
    """
    no_plan, _ = _run(episode_world, None, lap=None, zone=None)
    base_dep = float(no_plan.cars["FOLLOWER"].deployed_lap[5])
    for b in (0.2, 0.4, 0.8):
        gt, _ = _run(episode_world, b)
        assert float(gt.cars["FOLLOWER"].deployed_lap[5]) >= base_dep - 1e3, b


def test_the_delivered_feedback_is_attributed_to_the_lap_that_spent_it(episode_world):
    """`step()` mutates then records, so the POST-step lap is the wrong index."""
    gt, pol = _run(episode_world, 1.6)
    tr = gt.cars["FOLLOWER"]
    assert pol._delivered, "no feedback reached the policy"
    for lap, j in pol._delivered.items():
        if 0 <= lap < len(tr.deployed_lap):
            assert j == pytest.approx(float(tr.deployed_lap[lap]), rel=1e-6), lap


def test_the_base_policy_ignores_the_feedback_hook():
    """`note_deployed` must be inert for every policy that is not budgeted."""
    from xray.policy import get_policy
    p = get_policy("BALANCED")
    before = dataclasses.asdict(p) if dataclasses.is_dataclass(p) else None
    p.note_deployed(3, 1.0e6)
    assert before == (dataclasses.asdict(p) if before is not None else None)
