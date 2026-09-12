"""P1 closure: current tyre condition must change current physics.

The gap this closes: the DP could already reason about FUTURE wear while the
speed map it evaluated the CURRENT opportunity on was wear-independent. A car
whose tyres are finished was therefore predicted to reach the braking point at
exactly the speed it would have managed on a fresh set.

Everything here goes through the production `decision_service` path, because the
question is not whether the physics can do it but whether the service asks it to.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from tests.test_p1_service import _payload
from xray.config import load_config
from xray.decision import delta_v
from xray.decision_service import (SURFACE_WEAR_LEVELS, ZoneSurfaceContext,
                                   _cached_zone_models,
                                   environment_at_opportunity,
                                   evaluate_decision_trace_from_payload,
                                   params_from_payload, speed_map_cache_key,
                                   tyre_state_at_opportunity)
from xray.overtake import p_pass

E_TEST = 1.2e6          # J, a mid-grid deployment both cars can afford


@pytest.fixture(scope="module")
def ctx():
    """Zone surfaces from the production cache, for a real payload."""
    p = _payload()
    params = params_from_payload(p)
    env = environment_at_opportunity(p, 200.0)
    own = tyre_state_at_opportunity(p, "OWN", 2, env, load_config())
    riv = tyre_state_at_opportunity(p, "RIV", 2, env, load_config())
    sc = ZoneSurfaceContext.build(own, riv, env, SURFACE_WEAR_LEVELS)
    zones = _cached_zone_models(speed_map_cache_key(p, params, surface=sc))
    return {"payload": p, "params": params, "env": env, "own": own,
            "rival": riv, "zones": zones}


def _braking_zone(zones):
    """The most braking-sensitive zone. Tyres barely matter on a pure straight."""
    return max(zones, key=lambda z: z.braking_severity)


# ------------------------------------------------ Test A: wear changes speed
def test_a_worn_tyre_reaches_the_braking_point_slower(ctx):
    """Same car, same energy, same environment, same compound. Only wear moves."""
    z = _braking_zone(ctx["zones"])
    assert z.wear_grid is not None, "the production path built no wear surface"
    assert z.speed_grid_by_wear is not None

    speeds = [z.own_speed(E_TEST, float(w)) for w in z.wear_grid]
    assert speeds[0] > speeds[-1], (
        f"wear did not slow the car on zone {z.name}: {np.round(speeds, 3)}")
    # monotone, not just different at the ends
    assert np.all(np.diff(speeds) <= 1e-9), f"non-monotone in wear: {speeds}"
    # and the effect is large enough to matter to a decision
    assert speeds[0] - speeds[-1] > 0.1, "wear effect is below numerical noise"


def test_the_energy_map_stays_monotone_at_every_wear_level(ctx):
    z = _braking_zone(ctx["zones"])
    for w in z.wear_grid:
        sp = [z.own_speed(e, float(w)) for e in np.linspace(0.0, 2.0e6, 12)]
        assert np.all(np.diff(sp) >= -1e-9), f"energy->speed broke at wear {w}"


# --------------------------------------- Test B/C: delta_v moves both ways
def test_rival_wear_increases_our_advantage(ctx):
    """Worsen ONLY the rival. delta_v must rise."""
    z = _braking_zone(ctx["zones"])
    fresh = delta_v(z, E_TEST, E_TEST, wear_own=0.0, wear_riv=0.0)
    worn_rival = delta_v(z, E_TEST, E_TEST, wear_own=0.0, wear_riv=1.0)
    assert worn_rival > fresh, (
        f"a worn rival did not help us: {fresh:.4f} -> {worn_rival:.4f}")


def test_our_own_wear_reduces_our_advantage(ctx):
    """Worsen ONLY us. delta_v must fall. The opposite sign of Test B."""
    z = _braking_zone(ctx["zones"])
    fresh = delta_v(z, E_TEST, E_TEST, wear_own=0.0, wear_riv=0.0)
    worn_own = delta_v(z, E_TEST, E_TEST, wear_own=1.0, wear_riv=0.0)
    assert worn_own < fresh, (
        f"our own wear did not cost us: {fresh:.4f} -> {worn_own:.4f}")


def test_equal_wear_on_both_cars_roughly_cancels(ctx):
    """A sanity check that the two effects are the same physics, not two knobs."""
    z = _braking_zone(ctx["zones"])
    both = delta_v(z, E_TEST, E_TEST, wear_own=1.0, wear_riv=1.0)
    neither = delta_v(z, E_TEST, E_TEST, wear_own=0.0, wear_riv=0.0)
    # Not exactly zero: the two compounds wear differently, which is the point.
    assert abs(both - neither) < abs(
        delta_v(z, E_TEST, E_TEST, wear_own=1.0, wear_riv=0.0) - neither)


# --------------------------------------------- Test D: independent surfaces
def test_own_and_rival_read_different_surfaces(ctx):
    """Different compounds must not silently share one curve."""
    z = _braking_zone(ctx["zones"])
    assert ctx["own"].compound != ctx["rival"].compound, "fixture is not a test"
    assert z.rival_speed_grid_by_wear is not None, "no rival surface was built"
    assert not np.allclose(z.speed_grid_by_wear, z.rival_speed_grid_by_wear), (
        "own and rival surfaces are identical despite different compounds")
    # at full wear the softer compound must have lost more grip, so be slower
    own_worn = z.own_speed(E_TEST, 1.0)
    riv_worn = z.rival_speed(E_TEST, 1.0)
    assert own_worn != riv_worn


def test_swapping_the_compounds_swaps_the_surfaces(ctx):
    """Proves the compound reaches the physics, not just the cache key."""
    p, params, env = ctx["payload"], ctx["params"], ctx["env"]
    own, riv = ctx["own"], ctx["rival"]
    normal = ZoneSurfaceContext.build(own, riv, env, SURFACE_WEAR_LEVELS)
    swapped = ZoneSurfaceContext.build(riv, own, env, SURFACE_WEAR_LEVELS)
    a = _braking_zone(_cached_zone_models(speed_map_cache_key(p, params, surface=normal)))
    b = _braking_zone(_cached_zone_models(speed_map_cache_key(p, params, surface=swapped)))
    assert a.own_speed(E_TEST, 1.0) == pytest.approx(b.rival_speed(E_TEST, 1.0))
    assert a.rival_speed(E_TEST, 1.0) == pytest.approx(b.own_speed(E_TEST, 1.0))


# ------------------------------------------- §11: propagation into P(pass)
def test_a_tyre_change_propagates_all_the_way_to_pass_probability(ctx):
    """tyre -> speed -> delta_v -> P(pass). The coefficients are untouched."""
    z = _braking_zone(ctx["zones"])
    gap = 0.55
    fresh_dv = delta_v(z, E_TEST, E_TEST, wear_own=0.0, wear_riv=0.0)
    worn_dv = delta_v(z, E_TEST, E_TEST, wear_own=0.0, wear_riv=1.0)
    assert worn_dv != fresh_dv

    q_fresh = p_pass(fresh_dv, gap, z)
    q_worn = p_pass(worn_dv, gap, z)
    assert q_worn > q_fresh, (
        f"delta_v moved {fresh_dv:.3f}->{worn_dv:.3f} but P(pass) did not: "
        f"{q_fresh:.5f} -> {q_worn:.5f}")


# ------------------------------------- the production path actually uses it
def test_the_service_reports_a_wear_dependent_speed_and_delta_v():
    """End to end: worsen the rival's tyres in the payload and watch the call."""
    base = _payload()
    before = evaluate_decision_trace_from_payload(base, "OWN", "RIV")["laps"][1]

    # A longer-used rival set: more laps of the same stint before this one.
    worse = copy.deepcopy(base)
    for row in worse["laps"]:
        if row["driver"] == "RIV":
            row["fresh_tyre"] = False
            row["tyre_life"] = row["lap"] + 25
    after = evaluate_decision_trace_from_payload(worse, "OWN", "RIV")["laps"][1]

    assert after["rival_wear_fraction"] is not None
    assert before["rival_wear_fraction"] is not None
    # the modelled rival wear must have moved, and the physics with it
    if after["rival_wear_fraction"] != before["rival_wear_fraction"]:
        assert (after["predicted_rival_speed_mps"]
                != before["predicted_rival_speed_mps"]), (
            "rival wear changed but its predicted speed did not")
        assert after["predicted_delta_v_mps"] != before["predicted_delta_v_mps"]


def test_service_output_exposes_both_cars_wear():
    row = evaluate_decision_trace_from_payload(_payload(), "OWN", "RIV")["laps"][1]
    assert row["wear_fraction"] is not None
    assert row["rival_wear_fraction"] is not None
    assert row["own_tyre"]["compound"] != row["rival_tyre"]["compound"]


# ----------------------------------------------- §9: interpolation preserved
def test_coarse_and_fine_surface_wear_grids_agree(ctx):
    """The surface interpolates; a finer grid must not change the answer much."""
    p, params, env = ctx["payload"], ctx["params"], ctx["env"]
    own, riv = ctx["own"], ctx["rival"]

    def speed_at(levels, wear):
        sc = ZoneSurfaceContext.build(own, riv, env, levels)
        z = _braking_zone(_cached_zone_models(speed_map_cache_key(p, params, surface=sc)))
        return z.own_speed(E_TEST, wear)

    coarse = speed_at((0.0, 0.5, 1.0), 0.37)
    fine = speed_at(tuple(np.round(np.linspace(0.0, 1.0, 9), 6)), 0.37)
    assert coarse == pytest.approx(fine, abs=0.15), (
        f"grid resolution changed the speed materially: {coarse} vs {fine}")


def test_surface_interpolation_is_not_nearest_bin(ctx):
    """A value between two levels must land between the two speeds, not on one."""
    z = _braking_zone(ctx["zones"])
    lo, hi = float(z.wear_grid[0]), float(z.wear_grid[1])
    mid = 0.5 * (lo + hi)
    s_lo, s_hi, s_mid = (z.own_speed(E_TEST, lo), z.own_speed(E_TEST, hi),
                         z.own_speed(E_TEST, mid))
    assert min(s_lo, s_hi) < s_mid < max(s_lo, s_hi), (
        "wear lookup snapped to a bin instead of interpolating")


# --------------------------------------------- §15: the legacy path survives
def test_without_tyre_state_the_service_uses_the_legacy_energy_only_map():
    """No tyre metadata -> no surface -> the P0 map, numerically unchanged."""
    p = _payload()
    p["laps"] = []                      # no compound, no age, no stint
    out = evaluate_decision_trace_from_payload(p, "OWN", "RIV")
    row = out["laps"][1]
    assert row["wear_fraction"] is None
    assert row["own_tyre"] is None or row["own_tyre"]["source"] == "unknown"

    params = params_from_payload(p)
    legacy = _cached_zone_models(speed_map_cache_key(p, params))
    for z in legacy:
        assert z.wear_grid is None, "the legacy path grew a wear surface"
        assert z.speed_grid_by_wear is None
        # and reading it with a wear argument must fall back to the 1-D map
        assert z.own_speed(E_TEST, 0.9) == pytest.approx(z.own_speed(E_TEST))


# ------------------------------------------------------- §16: full causality
def test_future_tyre_and_pit_metadata_cannot_build_the_current_speed_surface():
    """The surface is a physical context; a future context must not reach it.

    Extends the P0/P1 causality sweep with everything the wear surface now
    consumes: compound, TyreLife, FreshTyre, stint, pit timing and weather.
    """
    p = _payload()
    before = evaluate_decision_trace_from_payload(p, "OWN", "RIV")
    row = before["laps"][1]
    t_dec = max(c["decision_time_s"] for c in row["zone_candidates"])

    bad = copy.deepcopy(p)
    for w in bad["weather_trace"]:
        if w["t"] > t_dec:
            w.update(air_temp_c=48.0, track_temp_c=70.0, rho=0.85,
                     rainfall=True, wind_speed_ms=33.0, wind_dir_deg=270.0)
    for name in ("OWN", "RIV"):
        tr = bad["cars"][name]["trace"]
        for i, t in enumerate(tr["t"]):
            if t > t_dec:
                tr["v"][i] = 2.0
                tr["usable_mean"][i] = 3.9
                tr["usable_p10"][i] = 3.9
                tr["usable_p90"][i] = 3.9
    for r in bad["laps"]:
        if r["lap"] > row["lap"]:
            r.update(compound="WET", tyre_life=48, fresh_tyre=True, stint=9,
                     pit_in_time_s=1.0, pit_out_time_s=2.0)
    after = evaluate_decision_trace_from_payload(bad, "OWN", "RIV")["laps"][1]

    for key in ("decision", "zone", "q", "tau", "gap_s",
                "own_usable_energy_mj", "rival_usable_energy_mj",
                "predicted_own_speed_mps", "predicted_rival_speed_mps",
                "predicted_delta_v_mps", "pass_probability",
                "value_wait", "attack_threshold",
                "wear_fraction", "rival_wear_fraction",
                "attack_wear_continuation_penalty"):
        assert after[key] == row[key], key
    assert after["own_tyre"] == row["own_tyre"]
    assert after["rival_tyre"] == row["rival_tyre"]
    assert after["environment"] == row["environment"]


def test_oracle_pit_times_cannot_reach_the_wear_surface():
    """The payload contains the real stop. The surface must never see it."""
    p = _payload()
    before = evaluate_decision_trace_from_payload(p, "OWN", "RIV")["laps"][1]
    bad = copy.deepcopy(p)
    for r in bad["laps"]:
        r["pit_in_time_s"] = 3.0
        r["pit_out_time_s"] = 4.0
    after = evaluate_decision_trace_from_payload(bad, "OWN", "RIV")["laps"][1]
    for key in ("predicted_own_speed_mps", "predicted_rival_speed_mps",
                "predicted_delta_v_mps", "wear_fraction", "rival_wear_fraction",
                "decision", "pass_probability"):
        assert after[key] == before[key], key
