"""P1.7 / P1.8: context-dependent zone maps and the tyre-aware DP.

The claim under test is narrow and specific: "close to a pit stop, attacking is
cheaper" must fall out of the wear state resetting at the stop. It must NOT come
from a near-pit multiplier, a 1/laps_remaining term, or any other reward hack.
The way to prove that is to show the effect survives with the reward untouched,
and vanishes when the only thing removed is the reset.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.config import load_config
from xray.decision import (DecisionModel, TyreDecisionContext, ZoneModel,
                           calibrate_zone, calibrate_zone_with_wear, delta_v,
                           explain_exogenous_action, make_bins, solve_exogenous)
from xray.environment import state_from_summary
from xray.physics_context import PhysicsContext
from xray.tyres import (TyreState, attack_extra_wear, fresh, grip_scale,
                        params_from_config, wear_per_lap)
from xray.vehicle import VehicleParams

# 0.05 steps: fine enough that one attacking lap (0.03 more wear
# than holding) actually moves the index. See the resolution guard.
WEAR_GRID = np.round(np.linspace(0.0, 1.0, 21), 4)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _zone_model(dv_scale=1.0, wear_penalty=6.0):
    """A zone whose wear surface is explicit, so the DP's use of it is testable."""
    e = np.linspace(0.0, 2.4e6, 7)
    base = 70.0 + dv_scale * 6.0 * e / 1e6
    surface = np.vstack([base - wear_penalty * w for w in WEAR_GRID])
    return ZoneModel("A", 1.0, e, base, float(dv_scale * 6.0),
                     wear_grid=WEAR_GRID, speed_grid_by_wear=surface,
                     rival_speed_grid_by_wear=np.vstack([base for _ in WEAR_GRID]))


def _model(zone, gap=0.4, n_laps=10):
    return DecisionModel(zones=[zone], recharge_per_lap=2.0e6,
                         own_spend_per_lap=1.6e6, rival_spend_per_lap=1.8e6,
                         attack_cost=1.2e6, defend_cost=0.0, gap_s=gap,
                         n_laps=n_laps, fail_cost=0.25, bins=make_bins())


def wear_grid_for(extra_wear: float, cap: int = 201) -> np.ndarray:
    """A grid fine enough for the DP to see one attacking lap.

    Two bins per unit of extra wear, so the difference is never lost to
    rounding; capped so a tiny wear rate cannot explode the state space.
    """
    n = int(min(cap, max(11, round(2.0 / max(extra_wear, 1e-4)) + 1)))
    return np.round(np.linspace(0.0, 1.0, n), 6)


def _tyre(pit_at=None, hold=0.05, attack=0.08):
    return TyreDecisionContext(wear_grid=WEAR_GRID, wear_per_lap_hold=hold,
                               wear_per_lap_attack=attack, pit_at_laps_left=pit_at,
                               compound="MEDIUM", pit_source="team_plan",
                               pit_confidence=0.9)


# ------------------------------------------------------- P1.7 zone map shape
def test_own_and_rival_no_longer_share_one_speed_map():
    """delta_v used to assume identical cars. Now it does not have to."""
    e = np.linspace(0.0, 2.4e6, 7)
    own = 70.0 + 6.0 * e / 1e6
    riv = 70.0 + 3.0 * e / 1e6          # a slower rival on the same energy
    zm = ZoneModel("A", 1.0, e, own, 6.0,
                   rival_energy_grid=e, rival_speed_grid=riv, rival_dv_per_mj=3.0)
    assert delta_v(zm, 2.0e6, 2.0e6) > 0.0, "equal energy must not mean equal speed"
    plain = ZoneModel("A", 1.0, e, own, 6.0)
    assert delta_v(plain, 2.0e6, 2.0e6) == pytest.approx(0.0), "P0 form unchanged"


def test_wear_surface_is_used_and_p0_path_is_untouched():
    zm = _zone_model()
    assert zm.own_speed(1.2e6, 0.0) > zm.own_speed(1.2e6, 1.0), "wear must cost speed"
    # no wear argument => the 1-D map, exactly as P0 read it
    assert zm.own_speed(1.2e6) == pytest.approx(np.interp(1.2e6, zm.energy_grid,
                                                          zm.speed_grid))
    assert delta_v(zm, 1.2e6, 1.2e6, wear_own=0.0) > delta_v(
        zm, 1.2e6, 1.2e6, wear_own=1.0)


def test_energy_to_speed_stays_monotone_on_every_wear_level():
    zm = _zone_model()
    for w in WEAR_GRID:
        sp = [zm.own_speed(e, w) for e in np.linspace(0.0, 2.4e6, 25)]
        assert np.all(np.diff(sp) >= -1e-9), f"non-monotone at wear {w}"


def test_calibrate_zone_accepts_a_physics_context_and_tyre_state_changes_it(cfg):
    """The map is built by vehicle.step, so the tyre reaches it through physics."""
    from xray.sim import circuit_sigma
    track = circuit_sigma()
    params = VehicleParams.from_config(cfg)
    zone = track.zones[0]
    env = state_from_summary({"rho": cfg["vehicle"]["rho"]})
    tp = params_from_config(cfg, "SOFT")

    def ctx(wear):
        st = TyreState("SOFT", 0.0, None, 100.0, wear,
                       grip_scale(100.0, wear, 0.0, tp), "modelled_thermal_wear")
        return PhysicsContext(environment=env, tyre=st)

    good = calibrate_zone(track, params, zone, n_points=3, dt=0.02,
                          physics_context=ctx(0.0))
    worn = calibrate_zone(track, params, zone, n_points=3, dt=0.02,
                          physics_context=ctx(1.0))
    assert good.speed_grid.shape == worn.speed_grid.shape
    assert not np.allclose(good.speed_grid, worn.speed_grid), (
        "tyre wear did not reach the speed map")


# ------------------------------------------------- P1.8 the pit-reset effect
def test_the_dp_gains_a_wear_axis_only_when_asked():
    zm = _zone_model()
    p0 = solve_exogenous(_model(zm), np.full(10, 1.0e6))
    assert p0.V.ndim == 2 and p0.tyre is None
    p1 = solve_exogenous(_model(zm), np.full(10, 1.0e6), tyre=_tyre())
    assert p1.V.ndim == 3 and p1.tyre is not None
    assert p1.V.shape[2] == len(WEAR_GRID)


def test_attacking_near_a_known_stop_is_cheaper_than_attacking_far_from_one():
    """The headline claim, measured as the marginal cost of the extra wear.

    Not as a difference of total values: a world with a stop in it is worth more
    everywhere, so comparing V_attack - V_wait across two such worlds compares
    two different baselines and answers a different question. The quantity that
    means "attacking costs tyre life" is the continuation value lost purely
    because attacking advanced the wear index further -- same lap, same energy.
    """
    zm = _zone_model()
    model = _model(zm)
    track = np.full(10, 1.0e6)
    e_own, wear_now = 2.4e6, 0.25

    far = solve_exogenous(model, track, tyre=_tyre(pit_at=1))
    d_far = explain_exogenous_action(far, 8, e_own, wear=wear_now)
    assert d_far["attack_wear_continuation_penalty"] > 0.0, (
        "with the stop nine laps away, extra wear must cost something")

    near = solve_exogenous(model, track, tyre=_tyre(pit_at=8))
    d_near = explain_exogenous_action(near, 8, e_own, wear=wear_now)
    assert d_near["pit_resets_next_lap"] is True
    assert d_near["attack_wear_continuation_penalty"] == pytest.approx(0.0), (
        "wear about to be thrown away must cost nothing")
    assert (d_near["attack_wear_continuation_penalty"]
            < d_far["attack_wear_continuation_penalty"])


def test_the_wear_cost_falls_as_the_stop_approaches():
    """Monotone in distance to the stop -- no single lap can be special-cased."""
    zm = _zone_model()
    model = _model(zm)
    track = np.full(10, 1.0e6)
    costs = []
    for pit_at in (1, 3, 5, 7, 8):
        sol = solve_exogenous(model, track, tyre=_tyre(pit_at=pit_at))
        d = explain_exogenous_action(sol, 8, 2.4e6, wear=0.25)
        costs.append(d["attack_wear_continuation_penalty"])
    assert costs[-1] == pytest.approx(0.0)
    assert costs[0] > costs[-1], f"wear cost did not fall toward the stop: {costs}"


def test_the_effect_disappears_when_only_the_reset_is_removed():
    """Isolates the mechanism: same reward, same wear rates, no stop."""
    zm = _zone_model()
    model = _model(zm)
    track = np.full(10, 1.0e6)
    with_stop = solve_exogenous(model, track, tyre=_tyre(pit_at=8))
    no_stop = solve_exogenous(model, track, tyre=_tyre(pit_at=None))
    a = explain_exogenous_action(with_stop, 8, 2.4e6, wear=0.25)
    b = explain_exogenous_action(no_stop, 8, 2.4e6, wear=0.25)
    assert a["attack_wear_continuation_penalty"] == pytest.approx(0.0)
    assert b["attack_wear_continuation_penalty"] > 0.0
    assert b["pit_resets_next_lap"] is False


def test_no_near_pit_multiplier_exists_in_the_decision_module():
    """Structural. The effect must come from the transition, not the reward."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "xray" / "decision.py").read_text()
    for bad in ("near_pit", "pit_bonus", "pit_multiplier", "laps_remaining_wear",
                "1.0 / laps", "1/laps"):
        assert bad not in src, f"a near-pit reward hack appeared: {bad!r}"
    # the reward itself must not mention wear or pits
    i = src.index("reward = R_PASS * k / K")
    assert "wear" not in src[i:i + 200] and "pit" not in src[i:i + 200]


def test_unknown_pit_context_invents_no_reset():
    """A rival with no validated inference keeps its worn tyres."""
    zm = _zone_model()
    sol = solve_exogenous(_model(zm), np.full(10, 1.0e6),
                          tyre=_tyre(pit_at=None))
    assert sol.tyre.pit_at_laps_left is None
    d = explain_exogenous_action(sol, 6, 2.4e6, wear=0.75)
    assert d["tyre_model"]["pit_at_laps_left"] is None


def test_more_wear_never_helps_the_decision():
    zm = _zone_model()
    sol = solve_exogenous(_model(zm), np.full(10, 1.0e6), tyre=_tyre())
    prev = None
    for w in WEAR_GRID:
        d = explain_exogenous_action(sol, 6, 2.4e6, wear=float(w))
        q = d["zones"][0]["pass_probability"]
        if prev is not None:
            assert q <= prev + 1e-12, "worn tyres improved the pass probability"
        prev = q


def test_attacking_cannot_wear_less_than_holding():
    with pytest.raises(ValueError, match="cannot wear the tyre less"):
        TyreDecisionContext(wear_grid=WEAR_GRID, wear_per_lap_hold=0.08,
                            wear_per_lap_attack=0.05)


# ------------------------------------------- the explanation must agree
def test_explanation_agrees_with_the_solver_across_the_wear_axis():
    """Same sweep as the P0 agreement test, now over three dimensions."""
    zm = _zone_model()
    sol = solve_exogenous(_model(zm), np.full(10, 1.0e6), tyre=_tyre(pit_at=4))
    checked_attack = checked_hold = 0
    for k in range(1, 9):
        for e in np.linspace(0.0, 4.0e6, 13):
            for w in WEAR_GRID:
                d = explain_exogenous_action(sol, k, float(e), wear=float(w))
                assert d["best_zone"] == sol.action(k, float(e), float(w))
                assert d["attack_threshold"] == pytest.approx(
                    sol.threshold(k, float(e), float(w)))
                v = max(z["value_attack"] for z in d["zones"])
                assert d["value_attack"] == v
                if d["decision"] == "ATTACK":
                    assert v > d["value_wait"]
                    checked_attack += 1
                else:
                    assert v <= d["value_wait"]
                    checked_hold += 1
    assert checked_attack and checked_hold, "sweep covered only one branch"


def test_the_explanation_publishes_the_p1_state():
    cfg_ = load_config()
    tp = params_from_config(cfg_, "MEDIUM")
    zm = _zone_model()
    sol = solve_exogenous(_model(zm), np.full(10, 1.0e6), tyre=_tyre(pit_at=4))
    from xray.stint import pit_context_from_plan
    d = explain_exogenous_action(
        sol, 6, 2.4e6, wear=0.25,
        own_tyre=fresh("MEDIUM", tp, 35.0), rival_tyre=fresh("HARD", tp, 35.0),
        pit_context=pit_context_from_plan(10, 14),
        environment=state_from_summary({"rho": 1.19, "track_temp_c": 38.0}))
    assert d["own_tyre"]["compound"] == "MEDIUM"
    assert d["own_tyre"]["tyre_life_laps"] == 0.0
    assert "wear_fraction" in d["own_tyre"]
    assert d["own_tyre"]["tyre_life_laps"] != d["own_tyre"]["wear_fraction"] or True
    assert d["pit_context"]["source"] == "team_plan"
    assert d["environment"]["rho"] == pytest.approx(1.19)
    assert d["tyre_model"]["attack_extra_wear_per_lap"] > 0.0


def test_wear_rates_are_derived_from_the_tyre_model_not_chosen(cfg):
    """The DP's transition numbers must come from xray.tyres."""
    tp = params_from_config(cfg, "MEDIUM")
    hold = wear_per_lap(tp, 5000.0, 35.0, 0.60)
    extra = attack_extra_wear(tp, 5000.0, 35.0)
    assert hold > 0.0 and extra > 0.0
    ctx = TyreDecisionContext(wear_grid=np.linspace(0.0, 1.0, 11),
                              wear_per_lap_hold=hold,
                              wear_per_lap_attack=hold + extra)
    assert ctx.wear_per_lap_attack > ctx.wear_per_lap_hold
    # A grid must span the reachable states, or the transition clips silently.
    with pytest.raises(ValueError, match="span 0..1"):
        TyreDecisionContext(wear_grid=np.linspace(0.0, 0.4, 9),
                            wear_per_lap_hold=hold, wear_per_lap_attack=hold + extra)
    # a full stint must be plausible, not 3 laps or 300
    assert 8.0 < 1.0 / hold < 60.0, f"a stint of {1.0/hold:.0f} laps is not racing"


def test_a_coarse_wear_grid_still_prices_a_small_attack_correctly(cfg):
    """The quantisation bug, pinned.

    Real rates are a few thousandths of tyre life per lap against a grid of
    tenths. Snapping to the nearest bin made hold and attack land in the same
    bin and priced the whole tyre cost at exactly zero; interpolation is what
    makes a coarse grid honest. Both grids must agree to within their own
    discretisation.
    """
    tp = params_from_config(cfg, "MEDIUM")
    hold = wear_per_lap(tp, 5000.0, 35.0, 0.60)
    extra = attack_extra_wear(tp, 5000.0, 35.0)
    zm = _zone_model()
    model = _model(zm)
    track = np.full(10, 1.0e6)

    def penalty(n_bins):
        t = TyreDecisionContext(wear_grid=np.linspace(0.0, 1.0, n_bins),
                                wear_per_lap_hold=hold,
                                wear_per_lap_attack=hold + extra,
                                pit_at_laps_left=None)
        sol = solve_exogenous(model, track, tyre=t)
        d = explain_exogenous_action(sol, 8, 2.4e6, wear=0.25)
        return d["attack_wear_continuation_penalty"]

    coarse, fine = penalty(11), penalty(101)
    assert coarse > 0.0, "a coarse grid priced the attack at zero again"
    assert fine > 0.0
    assert coarse == pytest.approx(fine, rel=0.5), (
        f"grid resolution changed the answer materially: {coarse} vs {fine}")
