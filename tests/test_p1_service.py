"""P1.9: the whole chain, and the causality that has to survive it.

Every P1 input is another way for the future to leak into the past. Weather
samples after the decision, the tyre metadata of later laps, and above all the
lap the car actually pitted -- which is sitting in the same table as the fields
we legitimately read.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from tests.test_decision_service import _payload as _base_payload
from xray.decision_service import (environment_at_opportunity,
                                   evaluate_decision_trace_from_payload,
                                   params_from_payload,
                                   pit_context_at_opportunity,
                                   speed_map_cache_key,
                                   tyre_state_at_opportunity)
from xray.environment import state_from_summary


def _payload():
    """The P0 fixture plus weather, tyre metadata and ORACLE pit times."""
    p = _base_payload()
    p["weather_trace"] = [
        {"t": 0.0, "air_temp_c": 20.0, "track_temp_c": 30.0, "pressure_mbar": 1010.0,
         "humidity_pct": 40.0, "wind_speed_ms": 2.0, "wind_dir_deg": 0.0,
         "rainfall": False, "rho": 1.20},
        {"t": 120.0, "air_temp_c": 24.0, "track_temp_c": 42.0, "pressure_mbar": 1008.0,
         "humidity_pct": 45.0, "wind_speed_ms": 4.0, "wind_dir_deg": 90.0,
         "rainfall": False, "rho": 1.17},
        {"t": 300.0, "air_temp_c": 18.0, "track_temp_c": 24.0, "pressure_mbar": 1012.0,
         "humidity_pct": 90.0, "wind_speed_ms": 9.0, "wind_dir_deg": 200.0,
         "rainfall": True, "rho": 1.22},
    ]
    p["laps"] = []
    for drv, comp in (("OWN", "MEDIUM"), ("RIV", "SOFT")):
        for L in range(1, 5):
            p["laps"].append({
                "driver": drv, "lap": L, "compound": comp, "tyre_life": L,
                "stint": 1, "fresh_tyre": L == 1,
                # ORACLE: the car really pitted on lap 3. Nothing causal may see it.
                "pit_in_time_s": 900.0 if L == 3 else None,
                "pit_out_time_s": 925.0 if L == 3 else None,
            })
    return p


# --------------------------------------------------------------- extraction
def test_environment_is_taken_causally_from_the_trace():
    p = _payload()
    assert environment_at_opportunity(p, 119.0).t == 0.0
    assert environment_at_opportunity(p, 121.0).t == 120.0
    assert environment_at_opportunity(p, 5000.0).t == 300.0
    st = environment_at_opportunity(p, 5000.0)
    assert st.rainfall is True and st.track_wetness_index > 0.0


def test_missing_weather_trace_falls_back_to_the_mean_and_says_so():
    p = _payload()
    p["weather_trace"] = []
    st = environment_at_opportunity(p, 100.0)
    assert "mean" in st.source, "a session mean must be labelled a mean"


def test_tyre_state_is_modelled_and_age_is_not_wear():
    p = _payload()
    from xray.config import load_config
    env = environment_at_opportunity(p, 200.0)
    st = tyre_state_at_opportunity(p, "OWN", 4, env, load_config())
    assert st.compound == "MEDIUM"
    assert st.tyre_life == 4.0, "age comes from the payload untouched"
    assert 0.0 <= st.wear_fraction <= 1.0
    assert st.wear_fraction != st.tyre_life
    assert st.source == "modelled_thermal_wear"
    # a later lap in the same stint must be more worn, never less
    earlier = tyre_state_at_opportunity(p, "OWN", 2, env, load_config())
    assert st.wear_fraction >= earlier.wear_fraction


def test_a_fresh_tyre_lap_starts_the_stint_over():
    p = _payload()
    from xray.config import load_config
    env = environment_at_opportunity(p, 200.0)
    worn = tyre_state_at_opportunity(p, "OWN", 4, env, load_config())
    for row in p["laps"]:
        if row["driver"] == "OWN" and row["lap"] == 4:
            row["fresh_tyre"] = True
    after = tyre_state_at_opportunity(p, "OWN", 4, env, load_config())
    assert after.wear_fraction < worn.wear_fraction


# ------------------------------------------------------- the oracle boundary
def test_rival_pit_context_is_unknown_not_read_from_the_actual_stop():
    """The payload literally contains the answer. It must not be used."""
    p = _payload()
    assert any(r.get("pit_in_time_s") for r in p["laps"]), "fixture has the oracle"
    ctx = pit_context_at_opportunity(p, "RIV", 2)
    assert ctx.source == "unknown"
    assert ctx.laps_to_pit_mean is None
    assert ctx.is_causal


def test_our_own_team_plan_is_allowed_and_is_causal():
    p = _payload()
    ctx = pit_context_at_opportunity(p, "OWN", 2, {"OWN": {"planned_pit_lap": 4}})
    assert ctx.source == "team_plan" and ctx.is_causal
    assert ctx.laps_to_pit_mean == pytest.approx(2.0)


def test_oracle_pit_data_can_never_reach_a_decision():
    """Mutating the actual pit laps must change nothing at all."""
    p = _payload()
    before = evaluate_decision_trace_from_payload(p, "OWN", "RIV")
    changed = copy.deepcopy(p)
    for row in changed["laps"]:
        row["pit_in_time_s"] = 1.0
        row["pit_out_time_s"] = 2.0
    after = evaluate_decision_trace_from_payload(changed, "OWN", "RIV")
    for a, b in zip(after["laps"], before["laps"]):
        for key in ("decision", "zone", "q", "tau", "gap_s",
                    "predicted_delta_v_mps", "wear_fraction"):
            assert a[key] == b[key], key


# ------------------------------------------------------------- full causality
def test_future_weather_tyre_and_pit_data_cannot_change_a_past_decision():
    """The end-to-end gate: telemetry, weather, tyre metadata and oracle pits."""
    p = _payload()
    before = evaluate_decision_trace_from_payload(p, "OWN", "RIV")
    row = before["laps"][0]
    t_dec = max(c["decision_time_s"] for c in row["zone_candidates"])

    changed = copy.deepcopy(p)
    for w in changed["weather_trace"]:
        if w["t"] > t_dec:
            w.update(air_temp_c=45.0, track_temp_c=60.0, rho=0.9,
                     rainfall=True, wind_speed_ms=30.0)
    for name in ("OWN", "RIV"):
        tr = changed["cars"][name]["trace"]
        for i, t in enumerate(tr["t"]):
            if t > t_dec:
                tr["v"][i] = 2.0
                tr["usable_mean"][i] = 3.9
                tr["usable_p10"][i] = 3.9
                tr["usable_p90"][i] = 3.9
    for r in changed["laps"]:
        if r["lap"] > row["lap"]:
            r.update(compound="WET", tyre_life=40, fresh_tyre=True,
                     pit_in_time_s=1.0)

    after = evaluate_decision_trace_from_payload(changed, "OWN", "RIV")["laps"][0]
    for key in ("decision", "zone", "q", "tau", "gap_s", "gap_source",
                "rival_usable_energy_mj", "predicted_delta_v_mps",
                "pass_probability", "wear_fraction", "value_wait"):
        assert after[key] == row[key], key
    assert after["environment"] == row["environment"]
    assert after["own_tyre"] == row["own_tyre"]


# ------------------------------------------------------------------- cache
def test_the_cache_key_describes_the_surface_context_not_the_current_wear():
    """A surface is reusable; a point on it is not a context.

    The first version keyed on `wear_fraction`, which makes 0.183742 and
    0.183891 two different cache entries and forces a full recalibration on
    essentially every sample. Wear is the surface's AXIS. What belongs in the
    key is the grid, the compounds, and the environment the surface was
    integrated in.
    """
    import dataclasses

    from xray.config import load_config
    from xray.decision_service import SURFACE_WEAR_LEVELS, ZoneSurfaceContext

    p = _payload()
    params = params_from_payload(p)
    env = environment_at_opportunity(p, 100.0)
    own = tyre_state_at_opportunity(p, "OWN", 2, env, load_config())
    riv = tyre_state_at_opportunity(p, "RIV", 2, env, load_config())
    base_ctx = ZoneSurfaceContext.build(own, riv, env, SURFACE_WEAR_LEVELS)
    base = speed_map_cache_key(p, params, surface=base_ctx)

    # --- current wear must NOT invalidate: it selects within the surface
    for w in (0.0, 0.37, 0.9):
        moved = dataclasses.replace(own, wear_fraction=w)
        k = speed_map_cache_key(
            p, params,
            surface=ZoneSurfaceContext.build(moved, riv, env, SURFACE_WEAR_LEVELS))
        assert k == base, "instantaneous wear must not force a recalibration"

    # --- everything that genuinely changes the surface MUST invalidate
    def differs(**kw):
        ctx = dataclasses.replace(base_ctx, **kw)
        return speed_map_cache_key(p, params, surface=ctx) != base

    assert differs(own_compound="HARD"), "own compound"
    assert differs(rival_compound="HARD"), "rival compound"
    assert differs(wear_grid=(0.0, 0.5, 1.0)), "wear grid"
    assert differs(rho=base_ctx.rho + 0.05), "air density"
    assert differs(track_temp_c=(base_ctx.track_temp_c or 30.0) + 12.0), "track temp"
    assert differs(wetness=base_ctx.wetness + 0.4), "wetness"

    # vehicle physics
    for field in ("mass_car", "cda_straight", "cla_corner", "crr",
                  "brake_decel_max", "drivetrain_eff"):
        changed = params_from_payload(p)
        object.__setattr__(changed, field, getattr(changed, field) * 1.01 + 0.01)
        assert speed_map_cache_key(p, changed, surface=base_ctx) != base, field

    # track and zone geometry
    for mutate in (lambda q: q["circuit_geometry"].__setitem__("length", 1001.0),
                   lambda q: q["circuit_geometry"]["grade"].__setitem__(1, 0.02),
                   lambda q: q["circuit_geometry"]["zones"][0].__setitem__(
                       "s_straight_end", 250.0)):
        q = copy.deepcopy(p)
        mutate(q)
        assert speed_map_cache_key(q, params_from_payload(q),
                                   surface=base_ctx) != base


def test_a_tyre_coefficient_change_invalidates_the_surface():
    """The compound NAME is not enough: the model behind it is physics too."""
    import xray.decision_service as ds
    from xray.config import load_config
    from xray.decision_service import SURFACE_WEAR_LEVELS, ZoneSurfaceContext

    p = _payload()
    params = params_from_payload(p)
    env = environment_at_opportunity(p, 100.0)
    own = tyre_state_at_opportunity(p, "OWN", 2, env, load_config())
    riv = tyre_state_at_opportunity(p, "RIV", 2, env, load_config())
    ctx = ZoneSurfaceContext.build(own, riv, env, SURFACE_WEAR_LEVELS)
    base = speed_map_cache_key(p, params, surface=ctx)

    cfg = copy.deepcopy(load_config())
    cfg["tyres"]["compounds"]["MEDIUM"]["mu_long_base"] *= 1.1
    ds._config.cache_clear()
    try:
        ds._config.__wrapped__.__globals__  # noqa: B018  (documents the patch target)
        import unittest.mock as mock
        with mock.patch.object(ds, "_config", lambda: cfg):
            changed = speed_map_cache_key(p, params, surface=ctx)
        assert changed != base, "a changed tyre coefficient must rebuild the surface"
    finally:
        ds._config.cache_clear()


def test_ui_metadata_still_does_not_invalidate_the_cache():
    p = _payload()
    params = params_from_payload(p)
    base = speed_map_cache_key(p, params)
    changed = copy.deepcopy(p)
    changed["battles"] = [{"car": "OWN"}]
    changed["cars"]["OWN"]["driver"] = "Display Name"
    changed["laps"][0]["position"] = 9
    assert speed_map_cache_key(changed, params_from_payload(changed)) == base


# ------------------------------------------------------------- provenance
def test_the_backend_publishes_source_and_confidence_for_every_p1_input():
    """The frontend renders these; it must never recompute them."""
    row = evaluate_decision_trace_from_payload(_payload(), "OWN", "RIV")["laps"][0]
    for key in ("weather_source", "weather_age_s", "track_wetness_index",
                "wetness_source", "own_tyre", "rival_tyre", "tyre_calibration",
                "pit_context", "pit_source", "pit_confidence", "wear_fraction"):
        assert key in row, f"{key} missing from the decision payload"
    assert row["tyre_calibration"] == "synthetic"
    assert row["wetness_source"].startswith("inferred")
    assert row["own_tyre"]["tyre_life_laps"] != row["own_tyre"]["wear_fraction"]


def test_metadata_declares_what_is_measured_and_what_is_assumed():
    out = evaluate_decision_trace_from_payload(_payload(), "OWN", "RIV")
    md = out["metadata"]
    assert "AGE IN LAPS" in md["tyre_state"]
    assert "INFERRED" in md["wetness"]
    assert "oracle-only" in md["pit_context"]
    assert md["tyre_calibration"] == "synthetic"
