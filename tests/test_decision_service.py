"""Canonical web/core decision-service regressions."""
from __future__ import annotations

import copy

import numpy as np

from simulation.api.main import _decision_payload
from xray.decision_service import (evaluate_decision_trace_from_payload,
                                   params_from_payload, speed_map_cache_key,
                                   _cached_zone_models, _PayloadTrack,
                                   _speed_map_cache_payload)


def _payload():
    zones = [
        {"name": "A", "s_straight_start": 80.0, "s_straight_end": 240.0,
         "s_end": 285.0, "braking_severity": 1.0, "detection_point": 0.0,
         "length": 160.0},
        {"name": "B", "s_straight_start": 360.0, "s_straight_end": 510.0,
         "s_end": 555.0, "braking_severity": 0.55, "detection_point": 160.0,
         "length": 150.0},
        {"name": "C", "s_straight_start": 690.0, "s_straight_end": 830.0,
         "s_end": 875.0, "braking_severity": 0.2, "detection_point": 490.0,
         "length": 140.0},
    ]

    def trace(offset_s: float, usable_mj: float):
        t, s, lap, v, usable = [], [], [], [], []
        for L in range(1, 5):
            for x in np.linspace(0.0, 950.0, 20):
                in_corner = any(z["s_straight_end"] <= x <= z["s_end"] for z in zones)
                t.append(90.0 * (L - 1) + x / 80.0 + offset_s)
                s.append(float(x))
                lap.append(L)
                v.append(36.0 if in_corner else 78.0)
                usable.append(max(usable_mj - 0.15 * (L - 1), 0.1))
        return {
            "t": t, "s": s, "lap": lap, "v": v,
            "deploy_kw": [0.0] * len(t), "harvest_kw": [0.0] * len(t),
            "usable_mean": usable,
            "usable_p10": [max(x - 0.2, 0.0) for x in usable],
            "usable_p90": [min(x + 0.2, 4.0) for x in usable],
            "dry": [False] * len(t), "coast": [False] * len(t),
        }

    return {
        "id": "test", "circuit": "Unit Test Circuit",
        "weather": {"rho": 1.2},
        "circuit_geometry": {
            "length": 1000.0, "s": [0.0, 250.0, 500.0, 750.0, 1000.0],
            "grade": [0.0, 0.0, 0.0, 0.0, 0.0], "zones": zones,
        },
        "calibration": {"pooled": {"cda_pooled": 0.85}},
        "cars": {
            "OWN": {
                "trace": trace(0.55, 3.2), "reserve_mean": 0.0,
                "deployed_lap": {"1": 0.4, "2": 0.5, "3": 0.6},
                "harvested_lap": {"1": 1.0, "2": 1.0, "3": 1.0},
            },
            "RIV": {
                "trace": trace(0.0, 1.2), "reserve_mean": 0.0,
                "deployed_lap": {"1": 0.7, "2": 0.7, "3": 0.7},
                "harvested_lap": {"1": 1.0, "2": 1.0, "3": 1.0},
            },
        },
        "gaps": [{"lap": L, "car": "OWN", "ahead": "RIV", "gap_s": 0.55,
                  "position": 2} for L in range(1, 5)],
    }


def test_api_uses_the_canonical_decision_service():
    payload = _payload()
    from_api = _decision_payload(payload, "OWN", "RIV")
    from_core = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")
    for key in ("decision", "zone", "q", "tau", "predicted_delta_v_mps"):
        assert from_api["laps"][0][key] == from_core["laps"][0][key]


def test_current_lap_deployed_energy_is_not_current_usable_energy():
    payload = _payload()
    before = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")["laps"][0]
    changed = copy.deepcopy(payload)
    changed["cars"]["RIV"]["deployed_lap"]["1"] = 999.0
    changed["cars"]["OWN"]["deployed_lap"]["1"] = 999.0
    after = evaluate_decision_trace_from_payload(changed, "OWN", "RIV")["laps"][0]
    for key in ("own_usable_energy_mj", "rival_usable_energy_mj",
                "predicted_delta_v_mps", "q", "tau"):
        assert after[key] == before[key]


def test_energy_state_fields_are_not_interchangeable():
    payload = _payload()
    row = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")["laps"][0]
    assert row["own_usable_energy_mj"] != payload["cars"]["OWN"]["deployed_lap"]["1"]
    assert row["rival_usable_energy_mj"] != payload["cars"]["RIV"]["deployed_lap"]["1"]

    changed = copy.deepcopy(payload)
    changed["cars"]["RIV"]["trace"]["usable_mean"] = [0.25] * len(changed["cars"]["RIV"]["trace"]["usable_mean"])
    changed["cars"]["RIV"]["trace"]["usable_p10"] = [0.20] * len(changed["cars"]["RIV"]["trace"]["usable_p10"])
    changed["cars"]["RIV"]["trace"]["usable_p90"] = [0.30] * len(changed["cars"]["RIV"]["trace"]["usable_p90"])
    lower_usable = evaluate_decision_trace_from_payload(changed, "OWN", "RIV")["laps"][0]
    assert lower_usable["rival_usable_energy_mj"] == 0.25
    assert lower_usable["rival_usable_energy_mj"] != row["rival_usable_energy_mj"]


def test_future_trace_changes_do_not_change_a_past_decision():
    payload = _payload()
    before = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")
    row = before["laps"][0]
    cutoff = max(c["rival_sample_index"] for c in row["zone_candidates"])
    changed = copy.deepcopy(payload)
    tr = changed["cars"]["RIV"]["trace"]
    for i in range(cutoff + 1, len(tr["usable_mean"])):
        tr["usable_mean"][i] = 4.0
        tr["usable_p10"][i] = 4.0
        tr["usable_p90"][i] = 4.0
    after = evaluate_decision_trace_from_payload(changed, "OWN", "RIV")
    for key in ("decision", "zone", "q", "tau", "predicted_delta_v_mps",
                "gap_s", "rival_usable_energy_mj", "pass_probability"):
        assert after["laps"][0][key] == row[key]


def test_actual_gap_is_used_and_reported():
    row = evaluate_decision_trace_from_payload(_payload(), "OWN", "RIV")["laps"][0]
    assert row["gap_source"] == "same_time_position_trace"
    assert row["gap_method"] in {"same_time_observation", "causal_constant_velocity_projection"}
    assert row["gap_age_s"] >= 0.0
    assert row["gap_s"] > 0.0
    assert row["gap_confidence"] > 0.5


def test_future_telemetry_changes_do_not_change_a_past_decision_or_gap():
    payload = _payload()
    before = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")["laps"][0]
    chosen_zone = next(z for z in payload["circuit_geometry"]["zones"]
                       if z["name"] == before["requested_zone"])
    own = payload["cars"]["OWN"]["trace"]
    decision_time = max(t for t, lap, s in zip(own["t"], own["lap"], own["s"])
                        if lap == before["lap"] and s <= chosen_zone["s_straight_end"])
    changed = copy.deepcopy(payload)
    for trace in (changed["cars"]["RIV"]["trace"], changed["cars"]["OWN"]["trace"]):
        for i, t in enumerate(trace["t"]):
            if t > decision_time:
                if int(trace["lap"][i]) == int(before["lap"]):
                    trace["s"][i] = 999.0
                trace["v"][i] = 5.0
                trace["t"][i] = t + 500.0
                trace["usable_mean"][i] = 4.0
                trace["usable_p10"][i] = 4.0
                trace["usable_p90"][i] = 4.0
    for gap in changed["gaps"]:
        if int(gap["lap"]) >= int(before["lap"]):
            gap["gap_s"] = 9.9
    after = evaluate_decision_trace_from_payload(changed, "OWN", "RIV")["laps"][0]
    for key in ("gap_s", "gap_source", "gap_method", "gap_age_s",
                "rival_usable_energy_mj",
                "predicted_delta_v_mps", "pass_probability", "decision"):
        assert after[key] == before[key]


def test_decision_energy_sensitivity():
    payload = _payload()
    base = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")["laps"][0]
    weaker = copy.deepcopy(payload)
    tr = weaker["cars"]["RIV"]["trace"]
    tr["usable_mean"] = [max(x - 0.7, 0.0) for x in tr["usable_mean"]]
    tr["usable_p10"] = [max(x - 0.7, 0.0) for x in tr["usable_p10"]]
    tr["usable_p90"] = [max(x - 0.7, 0.0) for x in tr["usable_p90"]]
    low_rival = evaluate_decision_trace_from_payload(weaker, "OWN", "RIV")["laps"][0]
    assert low_rival["predicted_delta_v_mps"] >= base["predicted_delta_v_mps"]


def test_future_energy_forecast_is_bounded_and_labelled_heuristic():
    payload = _payload()
    payload["cars"]["RIV"]["deployed_lap"]["1"] = 99.0
    payload["cars"]["RIV"]["harvested_lap"]["1"] = 99.0
    row = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")["laps"][1]
    forecast = np.asarray(row["future_energy_forecast_mj"])
    assert row["future_energy_forecast_method"] == "bounded_historical_rate_transition"
    assert row["future_energy_model_class"] == "heuristic"
    assert np.all(forecast >= 0.0)
    assert np.all(forecast <= 4.0)


def test_speed_map_cache_key_tracks_physics_inputs():
    payload = _payload()
    params = params_from_payload(payload)
    base = speed_map_cache_key(payload, params)

    payload_changes = [
        ("length", lambda p: p["circuit_geometry"].__setitem__("length", 1001.0)),
        ("s grid", lambda p: p["circuit_geometry"]["s"].__setitem__(1, 251.0)),
        ("grade", lambda p: p["circuit_geometry"]["grade"].__setitem__(1, 0.01)),
        ("zone start", lambda p: p["circuit_geometry"]["zones"][0].__setitem__("s_straight_start", 81.0)),
        ("zone end", lambda p: p["circuit_geometry"]["zones"][0].__setitem__("s_straight_end", 245.0)),
        ("corner end", lambda p: p["circuit_geometry"]["zones"][0].__setitem__("s_end", 290.0)),
        ("severity", lambda p: p["circuit_geometry"]["zones"][0].__setitem__("braking_severity", 0.9)),
        ("apex", lambda p: p["circuit_geometry"]["zones"][0].__setitem__("apex_v", 31.0)),
        ("rho", lambda p: p["weather"].__setitem__("rho", 1.25)),
        ("cda calibration", lambda p: p["calibration"]["pooled"].__setitem__("cda_pooled", 0.9)),
    ]
    for _name, mutate in payload_changes:
        changed_payload = copy.deepcopy(payload)
        mutate(changed_payload)
        changed_params = params_from_payload(changed_payload)
        assert speed_map_cache_key(changed_payload, changed_params) != base

    for field in ("mass_car", "cda_straight", "cda_corner", "crr",
                  "brake_decel_max", "drivetrain_eff"):
        changed_params = params_from_payload(payload)
        object.__setattr__(changed_params, field, getattr(changed_params, field) * 1.01)
        assert speed_map_cache_key(payload, changed_params) != base


def test_speed_map_cache_key_ignores_non_physics_ui_payload():
    payload = _payload()
    params = params_from_payload(payload)
    base = speed_map_cache_key(payload, params)
    changed = copy.deepcopy(payload)
    changed["battles"] = [{"car": "OWN", "ahead": "RIV"}]
    changed["cars"]["OWN"]["driver"] = "Display Name"
    changed["cars"]["OWN"]["trace"]["deploy_kw"] = [999.0] * len(changed["cars"]["OWN"]["trace"]["deploy_kw"])
    assert speed_map_cache_key(changed, params_from_payload(changed)) == base


def test_cache_reconstruction_preserves_grade_alignment_and_physics():
    payload = _payload()
    payload["circuit_geometry"]["zones"][0]["apex_v"] = 30.0
    params = params_from_payload(payload)
    base_key = speed_map_cache_key(payload, params)
    base_maps = _cached_zone_models(base_key)

    uphill = copy.deepcopy(payload)
    uphill["circuit_geometry"]["grade"] = [0.0, 0.04, 0.04, 0.0, 0.0]
    uphill_key = speed_map_cache_key(uphill, params_from_payload(uphill))
    uphill_maps = _cached_zone_models(uphill_key)
    assert uphill_key != base_key
    assert not np.allclose(base_maps[0].speed_grid, uphill_maps[0].speed_grid)

    data = _speed_map_cache_payload(uphill, params_from_payload(uphill))["track"]
    reconstructed = _PayloadTrack.from_cache_payload(data)
    assert reconstructed._grid_s.tolist() == data["s_signature"]
    assert reconstructed._grade.tolist() == data["grade_signature"]


def test_gap_is_read_at_the_decision_point_not_at_the_braking_point():
    """The gap must come from the same instant as the energy belief.

    It used to be sampled at `s_straight_end` -- the braking point, which the ego
    reaches about 160 m after the belief is read. Mutating only the window
    between the decision point and the braking point is the sharp test: under the
    old code that window was an input, so this changed gap_s.
    """
    payload = _payload()
    before = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")["laps"][0]
    cand = before["zone_candidates"][0]
    assert cand["decision_point_s"] == cand["gap_reference_s"]
    zone = next(z for z in payload["circuit_geometry"]["zones"]
                if z["name"] == cand["requested_zone"])
    assert cand["decision_point_s"] == zone["s_straight_start"]

    changed = copy.deepcopy(payload)
    for name in ("OWN", "RIV"):
        tr = changed["cars"][name]["trace"]
        for i, s in enumerate(tr["s"]):
            if int(tr["lap"][i]) == int(before["lap"]) and s > zone["s_straight_start"]:
                tr["s"][i] = 999.0
                tr["v"][i] = 3.0
                tr["usable_mean"][i] = 4.0
    after = evaluate_decision_trace_from_payload(changed, "OWN", "RIV")["laps"][0]
    a = after["zone_candidates"][0]
    for key in ("gap_s", "gap_source", "gap_method", "rival_usable_energy_mj",
                "predicted_delta_v_mps", "pass_probability", "decision"):
        assert a[key] == cand[key], key


def test_every_mutation_after_the_decision_time_is_inert():
    """Causality as a sweep: for each lap, nothing after its decision time moves it."""
    payload = _payload()
    before = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")["laps"]
    for row in before:
        t_dec = max(c["decision_time_s"] for c in row["zone_candidates"])
        changed = copy.deepcopy(payload)
        for name in ("OWN", "RIV"):
            tr = changed["cars"][name]["trace"]
            for i, t in enumerate(tr["t"]):
                if t > t_dec:
                    tr["v"][i] = 2.0
                    tr["usable_mean"][i] = 3.9
                    tr["usable_p10"][i] = 3.9
                    tr["usable_p90"][i] = 3.9
        after = next(r for r in evaluate_decision_trace_from_payload(
            changed, "OWN", "RIV")["laps"] if r["lap"] == row["lap"])
        for key in ("decision", "zone", "q", "tau", "gap_s",
                    "rival_usable_energy_mj", "predicted_delta_v_mps"):
            assert after[key] == row[key], f"lap {row['lap']} key {key}"
