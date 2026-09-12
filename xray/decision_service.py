"""Canonical overtake-decision service.

The API and tests call this module instead of recreating decision math. The
core solver still lives in :mod:`xray.decision`; this layer handles causal
state extraction from analysed telemetry payloads and packages a defensible,
unit-labelled result for the web.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .constants import E_HARVEST_LAP, E_STORE_MAX, POWER_UNIT_2026
from .decision import (DecisionModel, ZoneModel, build_model,
                       explain_exogenous_action, fail_cost_from_geometry,
                       make_bins, solve_exogenous)
from .overtake import COEFFS
from .vehicle import VehicleParams

PHYSICS_CONFIG_VERSION = "decision-speed-map-v2"
ZONE_CALIBRATION_DT = 0.005


@dataclass(frozen=True)
class UsableEnergyAtOpportunity:
    """Current deployable store estimate at one decision location."""
    usable_energy_j: float
    usable_p10_j: float
    usable_p90_j: float
    sample_index: int
    sample_time_s: float
    source: str = "belief.usable_mean_at_or_before_zone"


@dataclass(frozen=True)
class OpportunityGap:
    gap_s: float
    source: str
    confidence: float
    age_s: float = 0.0
    method: str = "unknown"


@dataclass(frozen=True)
class TraceState:
    sample_index: int
    time_s: float
    lap: int
    s_m: float
    v_mps: float
    total_s_m: float


@dataclass(frozen=True)
class OvertakeDecisionInput:
    lap: int
    laps_left: int
    zone_name: str
    own_usable_energy_j: float
    rival_usable_energy_j: float
    rival_usable_p10_j: float
    rival_usable_p90_j: float
    actual_gap_s: float
    gap_source: str
    gap_confidence: float
    gap_age_s: float
    gap_method: str
    model: DecisionModel
    rival_track_j: np.ndarray


def evaluate_overtake_decision(inp: OvertakeDecisionInput) -> dict[str, Any]:
    """Return one structured ATTACK/HOLD decision from the core DP."""
    sol = solve_exogenous(inp.model, inp.rival_track_j)
    detail = explain_exogenous_action(sol, inp.laps_left, inp.own_usable_energy_j)
    zone_rows = detail["zones"]
    selected_zone = detail["best_zone"]
    if selected_zone is None:
        best = detail["best_attack"] or zone_rows[0]
    else:
        best = next(r for r in zone_rows if r["zone"] == selected_zone)
    gap_confidence = float(np.clip(inp.gap_confidence, 0.0, 1.0))
    calibration_confidence = 0.35  # p_pass coefficients are synthetic placeholders.
    confidence = float(np.clip(min(gap_confidence, calibration_confidence), 0.0, 1.0))
    return {
        "decision": detail["decision"],
        "zone": selected_zone or best["zone"],
        "pass_probability": best["pass_probability"],
        "attack_threshold": detail["attack_threshold"],
        "own_usable_energy_mj": inp.own_usable_energy_j / 1e6,
        "rival_usable_energy_mj": inp.rival_usable_energy_j / 1e6,
        "rival_usable_p10_mj": inp.rival_usable_p10_j / 1e6,
        "rival_usable_p90_mj": inp.rival_usable_p90_j / 1e6,
        "predicted_delta_v_mps": best["predicted_delta_v_mps"],
        "predicted_own_speed_mps": best["predicted_own_speed_mps"],
        "predicted_rival_speed_mps": best["predicted_rival_speed_mps"],
        "actual_gap_s": inp.actual_gap_s,
        "gap_s": inp.actual_gap_s,
        "gap_source": inp.gap_source,
        "gap_confidence": gap_confidence,
        "gap_age_s": inp.gap_age_s,
        "gap_method": inp.gap_method,
        # -inf is the DP's honest value for an attack it rejected, and it is not
        # representable in JSON. Publish null plus the flag, and keep the
        # hypothetical separately so the panel can still show what an attack
        # would have been worth.
        "value_attack": (float(best["value_attack"])
                         if np.isfinite(best["value_attack"]) else None),
        "value_attack_ranked": (float(best["value_attack"])
                                if np.isfinite(best["value_attack"]) else -1.0e18),
        "value_attack_hypothetical": float(best["value_attack_hypothetical"]),
        "attack_affordable": bool(detail["attack_affordable"]),
        "value_wait": detail["value_wait"],
        "confidence": confidence,
        "physics_model": "cached longitudinal vehicle.step zone speed map",
        "decision_model": "Core finite-horizon DP / solve_exogenous",
        "opponent_state": "inferred usable energy at current zone",
        "pass_model": {
            "model_type": "logistic",
            "calibration": "placeholder",
            "dataset_version": "synthetic-design-anchors",
            "coefficients": COEFFS.__dict__,
        },
        "model_calibration": "synthetic",
        "forecast_type": "heuristic_current_belief_plus_historical_rates",
        "future_energy_model": "bounded_historical_rate_transition",
        "future_energy_model_class": "heuristic",
        "future_energy_forecast_method": "bounded_historical_rate_transition",
        "forecast_confidence": 0.55,
        "future_energy_forecast_confidence": 0.55,
        "future_energy_forecast_mj": [float(x) / 1e6 for x in inp.rival_track_j],
        "all_zones": zone_rows,
    }


def evaluate_decision_trace_from_payload(payload: dict[str, Any], car: str,
                                         rival: str) -> dict[str, Any]:
    """Evaluate every causal lap/zone opportunity in an analysed race payload."""
    if car not in payload["cars"] or rival not in payload["cars"]:
        raise KeyError("car not analysed")
    track = track_from_payload(payload)
    params = params_from_payload(payload)
    laps = common_trace_laps(payload["cars"][car], payload["cars"][rival])
    if len(laps) < 3:
        raise ValueError("not enough common laps for a decision trace")

    rows = []
    for i, lap in enumerate(laps):
        row = _evaluate_lap(payload, track, params, car, rival, laps, i, lap)
        rows.append(row)
    call = next((r for r in rows if r["attack"]), None)
    return {
        "car": car,
        "rival": rival,
        "laps": rows,
        "call": call,
        "fan": {"curves": [], "consensus_lap": call["lap"] if call else None,
                "consensus_fraction": None, "n_policies": 0},
        "zone_models": [{"name": z.name, "severity": z.braking_severity,
                         "dv_per_mj": round(z.dv_per_mj, 3)}
                        for z in _cached_zone_models(speed_map_cache_key(payload, params))],
        "metadata": {
            "decision_model": "Core finite-horizon DP / solve_exogenous",
            "physics": "cached longitudinal simulation",
            "opponent_state": "inferred usable energy at current zone",
            "pass_model": "synthetic placeholder logistic",
            "forecast": "heuristic current belief plus historical causal rates",
            "deployed_lap": "point-estimate deployed store energy over the lap, MJ",
            "deployed_lap_posterior_mean": "posterior mean deployed store energy over the lap, MJ",
            "usable_mean": "current deployable store energy belief at sample time, MJ",
            "reserve_mean": "inferred held-back buffer, J",
        },
    }


def _evaluate_lap(payload: dict[str, Any], track, params: VehicleParams,
                  car: str, rival: str, laps: list[int], i: int, lap: int) -> dict[str, Any]:
    zones = _cached_zone_models(speed_map_cache_key(payload, params))
    own_trace = payload["cars"][car]["trace"]
    rival_trace = payload["cars"][rival]["trace"]
    candidates = []
    n_laps = len(laps)
    for zi, zone in enumerate(track.zones):
        # One decision point per opportunity, and every input is read at it.
        # The gap used to be taken at `s_straight_end` -- the braking point,
        # which the ego reaches ~160 m AFTER the instant its energy belief is
        # sampled. That is a future observation feeding a past decision, and no
        # amount of "it is only 2 s later" makes it causal. Both the belief and
        # the gap are now read at the zone entry; carrying that gap into
        # `p_pass` as the gap at braking is an approximation, and it is labelled
        # one (`gap_reference_s`) rather than fixed by peeking.
        decision_s = zone.s_straight_start
        own_e = belief_at_position(own_trace, lap, decision_s)
        rival_e = belief_at_position(rival_trace, lap, decision_s)
        gap = gap_at_position(payload, car, rival, lap, decision_s)
        dyn = _historical_dynamics(payload, car, rival, lap)
        laps_left = n_laps - i
        # The DP is given THIS zone only. Handing it every zone made it run its
        # own argmax over zones while the loop below ran a second one, so the
        # reported zone could be a zone whose gap and energy were never the ones
        # evaluated. One selection, here.
        zone_model = zones[zi]
        model = DecisionModel(
            zones=[zone_model], recharge_per_lap=dyn["recharge_per_lap_j"],
            own_spend_per_lap=dyn["own_spend_per_lap_j"],
            rival_spend_per_lap=dyn["rival_spend_per_lap_j"],
            attack_cost=float(zone_model.energy_grid.max()), defend_cost=0.0,
            gap_s=gap.gap_s, n_laps=laps_left, fail_cost=fail_cost_from_geometry(gap.gap_s),
            bins=make_bins())
        rival_track = _rival_energy_forecast(rival_e.usable_energy_j, dyn, laps_left)
        inp = OvertakeDecisionInput(
            lap=lap, laps_left=laps_left, zone_name=zone.name,
            own_usable_energy_j=own_e.usable_energy_j,
            rival_usable_energy_j=rival_e.usable_energy_j,
            rival_usable_p10_j=rival_e.usable_p10_j,
            rival_usable_p90_j=rival_e.usable_p90_j,
            actual_gap_s=gap.gap_s, gap_source=gap.source,
            gap_confidence=gap.confidence, gap_age_s=gap.age_s,
            gap_method=gap.method, model=model, rival_track_j=rival_track)
        res = evaluate_overtake_decision(inp)
        candidate = {**res, "lap": lap, "requested_zone": zone.name,
                     "gap_reference_s": float(decision_s),
                     "decision_point_s": float(decision_s),
                     "decision_time_s": float(own_e.sample_time_s),
                     "own_sample_index": own_e.sample_index,
                     "rival_sample_index": rival_e.sample_index}
        candidates.append(candidate)

    # Attack candidates first, ranked by DP value; if none is affordable the
    # lap is a HOLD and the most valuable hypothetical is shown for context.
    chosen = max(candidates, key=lambda r: (r["decision"] == "ATTACK",
                                            r["value_attack_ranked"]))
    return {
        **chosen,
        "q": round(float(chosen["pass_probability"]), 4),
        "tau": round(float(chosen["attack_threshold"]), 4),
        "attack": chosen["decision"] == "ATTACK",
        "own_mj": round(float(chosen["own_usable_energy_mj"]), 4),
        "rival_mj": round(float(chosen["rival_usable_energy_mj"]), 4),
        "zone_candidates": candidates,
    }


def belief_at_position(trace: dict[str, list], lap: int, s_m: float) -> UsableEnergyAtOpportunity:
    """Causal usable-energy belief at or before ``s_m`` on ``lap``."""
    lap_arr = np.asarray(trace["lap"], dtype=int)
    s_arr = np.asarray(trace["s"], dtype=float)
    m = np.flatnonzero((lap_arr == int(lap)) & (s_arr <= float(s_m)))
    if len(m) == 0:
        raise ValueError(f"no causal trace samples for lap {lap} at s <= {s_m:.1f} m")
    idx = int(m[-1])
    return UsableEnergyAtOpportunity(
        usable_energy_j=float(trace["usable_mean"][idx]) * 1e6,
        usable_p10_j=float(trace["usable_p10"][idx]) * 1e6,
        usable_p90_j=float(trace["usable_p90"][idx]) * 1e6,
        sample_index=idx,
        sample_time_s=float(trace["t"][idx]),
    )


def gap_at_position(payload: dict[str, Any], car: str, rival: str, lap: int,
                    s_m: float) -> OpportunityGap:
    """Actual gap at a decision location, with explicit fallbacks."""
    track_length = float(payload["circuit_geometry"]["length"])
    ego = _state_at_or_before_position(payload["cars"][car]["trace"], lap, s_m,
                                       track_length)
    if ego is not None:
        rival_now = _state_projected_from_last_observation(payload["cars"][rival]["trace"],
                                                           ego.time_s, track_length)
        if rival_now is not None:
            gap_m = rival_now.total_s_m - ego.total_s_m
            if 0.0 < gap_m < 2000.0:
                local_v = max(ego.v_mps, 1.0)
                age_s = max(ego.time_s - rival_now.time_s, 0.0)
                method = ("same_time_observation" if age_s <= 1e-6
                          else "causal_constant_velocity_projection")
                conf = 0.9 if age_s <= 1e-6 else float(np.clip(0.9 * np.exp(-age_s / 1.0),
                                                               0.25, 0.85))
                return OpportunityGap(float(gap_m / local_v),
                                      "same_time_position_trace", conf,
                                      age_s=age_s, method=method)

    tc = _time_at_or_before_position(payload["cars"][car]["trace"], lap, s_m)
    tr = _time_at_or_before_position(payload["cars"][rival]["trace"], lap, s_m)
    if tc is not None and tr is not None:
        gap = tc - tr
        if 0.0 < gap < 30.0:
            return OpportunityGap(float(gap), "historical_position_crossing", 0.75,
                                  age_s=0.0, method="historical_position_crossing")
    for row in payload.get("gaps", []):
        if int(row.get("lap", -1)) == int(lap) and row.get("car") == car and row.get("ahead") == rival:
            return OpportunityGap(float(row["gap_s"]), "lap_timing_fallback", 0.55,
                                  age_s=float("nan"), method="lap_timing_fallback")
    return OpportunityGap(0.45, "explicit_low_confidence_fallback", 0.2,
                          age_s=float("nan"), method="constant_low_confidence_fallback")


def common_trace_laps(own_car: dict[str, Any], rival_car: dict[str, Any]) -> list[int]:
    own = {int(x) for x in own_car["trace"]["lap"]}
    rival = {int(x) for x in rival_car["trace"]["lap"]}
    return sorted(x for x in own & rival if x > 0)


def params_from_payload(payload: dict[str, Any]) -> VehicleParams:
    pooled = payload.get("calibration", {}).get("pooled") or {}
    cda = float(pooled.get("cda_pooled") or 0.9)
    rho = float(payload.get("weather", {}).get("rho") or 1.20)
    return VehicleParams(
        mass_car=768.0, fuel_start=70.0, fuel_burn_per_lap=1.4,
        cda_straight=cda, cda_corner=1.9 * cda, crr=0.012, rho=rho,
        brake_decel_max=45.0, drivetrain_eff=0.95)


def track_from_payload(payload: dict[str, Any]):
    return _PayloadTrack(payload)


def speed_map_cache_key(payload: dict[str, Any], params: VehicleParams) -> str:
    """Canonical cache key for E -> braking-point speed maps."""
    return json.dumps(_speed_map_cache_payload(payload, params), sort_keys=True,
                      separators=(",", ":"))


def _speed_map_cache_payload(payload: dict[str, Any], params: VehicleParams) -> dict:
    track = _PayloadTrack(payload)
    return {
        "version": PHYSICS_CONFIG_VERSION,
        "dt": ZONE_CALIBRATION_DT,
        "vehicle_params": asdict(params),
        "regulation": asdict(POWER_UNIT_2026),
        "track": {
            "length": round(float(track.length), 2),
            "s_signature": [round(float(x), 3) for x in track._grid_s.tolist()],
            "grade_signature": [round(float(x), 6) for x in track._grade.tolist()],
            "zones": [
                {
                    "name": z.name,
                    "s_straight_start": round(z.s_straight_start, 2),
                    "s_straight_end": round(z.s_straight_end, 2),
                    "s_end": round(z.s_end, 2),
                    "braking_severity": round(z.braking_severity, 3),
                    "apex_v": round(z.apex_v, 3),
                }
                for z in track.zones
            ],
        },
    }


@lru_cache(maxsize=32)
def _cached_zone_models(cache_key: str) -> tuple[ZoneModel, ...]:
    data = json.loads(cache_key)
    track = _PayloadTrack.from_cache_payload(data["track"])
    params = VehicleParams(**data["vehicle_params"])
    model = build_model(track, params, n_laps=12, recharge_per_lap=E_HARVEST_LAP * 0.4,
                        rival_spend_per_lap=E_HARVEST_LAP * 0.4)
    return tuple(model.zones)


def _historical_dynamics(payload: dict[str, Any], car: str, rival: str, before_lap: int) -> dict:
    own = payload["cars"][car]
    riv = payload["cars"][rival]
    return {
        "recharge_per_lap_j": _mean_lap_record_j(riv.get("harvested_lap", {}), before_lap,
                                                 default=E_HARVEST_LAP * 0.4),
        "own_spend_per_lap_j": _mean_lap_record_j(own.get("deployed_lap", {}), before_lap,
                                                  default=E_HARVEST_LAP * 0.25),
        "rival_spend_per_lap_j": _mean_lap_record_j(riv.get("deployed_lap", {}), before_lap,
                                                    default=E_HARVEST_LAP * 0.35),
    }


def _mean_lap_record_j(record: dict[str, float], before_lap: int, default: float) -> float:
    vals = [float(v) * 1e6 for k, v in record.items() if int(k) < int(before_lap)
            and np.isfinite(float(v)) and float(v) > 0.0]
    return float(np.mean(vals)) if vals else float(default)


def _rival_energy_forecast(current_usable_energy_j: float, dyn: dict,
                           laps_left: int) -> np.ndarray:
    vals = []
    e = float(np.clip(current_usable_energy_j, 0.0, E_STORE_MAX))
    harvested_per_lap_j = float(np.clip(dyn["recharge_per_lap_j"], 0.0, E_HARVEST_LAP))
    requested_deploy_per_lap_j = float(np.clip(dyn["rival_spend_per_lap_j"],
                                               0.0, E_STORE_MAX + E_HARVEST_LAP))
    for _ in range(max(int(laps_left), 1)):
        vals.append(e)
        deployed_j = min(requested_deploy_per_lap_j, e + harvested_per_lap_j)
        e = float(np.clip(e - deployed_j + harvested_per_lap_j, 0.0, E_STORE_MAX))
    return np.asarray(vals, dtype=float)


def _state_at_or_before_position(trace: dict[str, list], lap: int, s_m: float,
                                 track_length_m: float) -> TraceState | None:
    lap_arr = np.asarray(trace["lap"], dtype=int)
    s_arr = np.asarray(trace["s"], dtype=float)
    m = np.flatnonzero((lap_arr == int(lap)) & (s_arr <= float(s_m)))
    if len(m) == 0:
        return None
    return _state_at_index(trace, int(m[-1]), track_length_m)


def _state_at_or_before_time(trace: dict[str, list], time_s: float,
                             track_length_m: float) -> TraceState | None:
    t_arr = np.asarray(trace["t"], dtype=float)
    m = np.flatnonzero(t_arr <= float(time_s))
    if len(m) == 0:
        return None
    return _state_at_index(trace, int(m[-1]), track_length_m)


def _state_projected_from_last_observation(trace: dict[str, list], time_s: float,
                                           track_length_m: float) -> TraceState | None:
    t_arr = np.asarray(trace["t"], dtype=float)
    before = np.flatnonzero(t_arr <= float(time_s))
    if len(before) == 0:
        return None
    i0 = int(before[-1])
    t0 = float(t_arr[i0])
    lap = int(trace["lap"][i0])
    s0 = float(trace["s"][i0])
    v_mps = float(trace.get("v", [1.0])[i0])
    dt = max(float(time_s) - t0, 0.0)
    projected_total = (lap - 1) * float(track_length_m) + s0 + max(v_mps, 1.0) * dt
    projected_lap = int(projected_total // float(track_length_m)) + 1
    s_m = projected_total % float(track_length_m)
    return TraceState(
        sample_index=i0,
        time_s=t0,
        lap=projected_lap,
        s_m=s_m,
        v_mps=v_mps,
        total_s_m=projected_total,
    )


def _state_at_index(trace: dict[str, list], idx: int, track_length_m: float) -> TraceState:
    lap = int(trace["lap"][idx])
    s_m = float(trace["s"][idx])
    total = (lap - 1) * float(track_length_m) + s_m
    return TraceState(
        sample_index=idx,
        time_s=float(trace["t"][idx]),
        lap=lap,
        s_m=s_m,
        v_mps=float(trace.get("v", [1.0])[idx]),
        total_s_m=total,
    )


def _time_at_or_before_position(trace: dict[str, list], lap: int, s_m: float) -> float | None:
    lap_arr = np.asarray(trace["lap"], dtype=int)
    s_arr = np.asarray(trace["s"], dtype=float)
    t_arr = np.asarray(trace["t"], dtype=float)
    m = np.flatnonzero((lap_arr == int(lap)) & (s_arr <= float(s_m)))
    if len(m) == 0:
        return None
    return float(t_arr[int(m[-1])])


class _PayloadTrack:
    STRAIGHT_V_LIMIT = 400 / 3.6

    def __init__(self, payload: dict[str, Any]):
        geo = payload["circuit_geometry"]
        self.length = float(geo["length"])
        self.name = payload.get("circuit", "analysed circuit")
        self._grid_s = np.asarray(geo.get("s") or [0.0], dtype=float)
        self._grade = np.asarray(geo.get("grade") or np.zeros_like(self._grid_s), dtype=float)
        self.zones = [_PayloadZone(z, self._apex_speed(payload, z)) for z in geo["zones"]]
        self.corners = [(z.s_straight_end, z.s_end, z.apex_v) for z in self.zones]

    @classmethod
    def from_cache_payload(cls, data: dict):
        length = data["length"]
        zones = data["zones"]
        payload = {"circuit_geometry": {
            "length": length,
            "s": data.get("s_signature") or [0.0],
            "grade": data.get("grade_signature") or [0.0],
            "zones": zones}}
        return cls(payload)

    def point(self, s: float) -> tuple:
        s = float(s) % self.length
        is_corner = False
        v_lim = self.STRAIGHT_V_LIMIT
        for z in self.zones:
            if z.s_straight_end <= s < z.s_end:
                is_corner = True
                v_lim = z.apex_v
                break
        ahead = []
        for start, _end, v in self.corners:
            d = (start - s) % self.length
            if d <= 400.0:
                ahead.append((d, v))
        ahead.sort()
        while len(ahead) < 2:
            ahead.append((1e9, 1e9))
        return (float(v_lim), self.grade(s), is_corner,
                float(ahead[0][0]), float(ahead[0][1]),
                float(ahead[1][0]), float(ahead[1][1]))

    def grade(self, s):
        if len(self._grid_s) <= 1:
            return 0.0
        return float(np.interp(float(s) % self.length, self._grid_s, self._grade))

    def zone_by_name(self, name: str):
        return next(z for z in self.zones if z.name == name)

    def _apex_speed(self, payload: dict[str, Any], zone: dict[str, Any]) -> float:
        if "apex_v" in zone:
            return float(zone["apex_v"])
        # Causal API decisions cannot infer a current corner target from future
        # telemetry samples. Payloads produced by real analysis should include
        # zone ``apex_v``; this severity fallback is intentionally conservative.
        severity = float(zone.get("braking_severity", 0.5))
        return float((70.0 + 150.0 * (1.0 - severity)) / 3.6)


class _PayloadZone:
    def __init__(self, data: dict[str, Any], apex_v: float):
        self.name = str(data["name"])
        self.s_straight_start = float(data["s_straight_start"])
        self.s_straight_end = float(data["s_straight_end"])
        self.s_end = float(data.get("s_end", self.s_straight_end + 120.0))
        self.braking_severity = float(data["braking_severity"])
        self.detection_point = float(data.get("detection_point",
                                              self.s_straight_start - 200.0))
        self.energy_cost_hint = float(data.get("energy_cost_hint", 0.0))
        self.apex_v = float(apex_v)
