"""Causal real-race snapshots.

A snapshot is the serialized answer to two different questions:

* what X-RAY knew at ``feature_cutoff_time``;
* what happened during the later evaluation window.

Keeping those dictionaries separate makes future leakage testable. Mutating
future telemetry, gaps, pits or weather may change evaluation fields, but must
not change ``input_fields``.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


TRACE_INPUT_FIELDS = (
    "t", "s", "lap", "v", "deploy_kw", "deploy_lo_kw", "deploy_hi_kw",
    "harvest_kw", "usable_mean", "usable_p10", "usable_p90", "dry", "coast",
    "deployment_zone_eligible",
)


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _select_trace(trace: dict, mask: list[bool]) -> dict:
    out = {}
    for key in TRACE_INPUT_FIELDS:
        if key not in trace:
            continue
        vals = trace[key]
        out[key] = [copy.deepcopy(v) for v, keep in zip(vals, mask) if keep]
    return out


def _trace_until(car: dict, cutoff: float) -> dict:
    trace = car.get("trace", {})
    times = [float(t) for t in trace.get("t", []) if t is not None]
    mask = [(t is not None and float(t) <= cutoff) for t in trace.get("t", [])]
    selected = _select_trace(trace, mask)
    selected["n_samples"] = len(selected.get("t", []))
    selected["last_time"] = max((float(t) for t in selected.get("t", [])), default=None)
    selected["source_samples"] = len(times)
    return selected


def _trace_between(car: dict, start: float, end: float) -> dict:
    trace = car.get("trace", {})
    mask = [(t is not None and start <= float(t) <= end) for t in trace.get("t", [])]
    selected = _select_trace(trace, mask)
    selected["n_samples"] = len(selected.get("t", []))
    return selected


def _weather_at_or_before(payload: dict, cutoff: float) -> dict:
    rows = [r for r in payload.get("weather_trace", [])
            if r.get("t") is not None and float(r["t"]) <= cutoff]
    if not rows:
        return {"source": "session_summary", "row": copy.deepcopy(payload.get("weather", {}))}
    return {"source": "weather_trace", "row": copy.deepcopy(rows[-1])}


def _weather_after(payload: dict, start: float, end: float) -> list[dict]:
    return [copy.deepcopy(r) for r in payload.get("weather_trace", [])
            if r.get("t") is not None and start <= float(r["t"]) <= end]


def _lap_at_cutoff(car: dict, cutoff: float) -> int | None:
    trace = car.get("trace", {})
    laps = trace.get("lap", [])
    times = trace.get("t", [])
    seen = [int(l) for t, l in zip(times, laps) if t is not None and float(t) <= cutoff]
    return max(seen) if seen else None


def _laps_completed(payload: dict, drivers: list[str], cutoff: float) -> list[dict]:
    out = []
    for row in payload.get("laps", []):
        if row.get("driver") not in drivers:
            continue
        t_end = row.get("t_end")
        if t_end is None or float(t_end) > cutoff:
            continue
        keep = {
            "driver": row.get("driver"),
            "lap": row.get("lap"),
            "position": row.get("position"),
            "lap_time": row.get("lap_time"),
            "t_end": t_end,
            "compound": row.get("compound"),
            "tyre_life": row.get("tyre_life"),
            "stint": row.get("stint"),
            "fresh_tyre": row.get("fresh_tyre"),
        }
        out.append(keep)
    return out


def _pit_events_seen(payload: dict, drivers: list[str], cutoff: float) -> list[dict]:
    events = []
    for row in payload.get("laps", []):
        if row.get("driver") not in drivers:
            continue
        for key in ("pit_in_time_s", "pit_out_time_s"):
            t = row.get(key)
            if t is not None and float(t) <= cutoff:
                events.append({
                    "driver": row.get("driver"),
                    "lap": row.get("lap"),
                    "kind": key.removesuffix("_time_s"),
                    "t": float(t),
                })
    return events


def _future_laps(payload: dict, drivers: list[str], start: float, end: float) -> list[dict]:
    rows = []
    for row in payload.get("laps", []):
        if row.get("driver") not in drivers:
            continue
        t_end = row.get("t_end")
        if t_end is not None and start <= float(t_end) <= end:
            rows.append(copy.deepcopy(row))
    return rows


def _gaps_by_lap(payload: dict, drivers: list[str], max_lap: int | None,
                 after: bool = False) -> list[dict]:
    if max_lap is None:
        return []
    out = []
    for g in payload.get("gaps", []):
        if g.get("car") not in drivers and g.get("ahead") not in drivers:
            continue
        lap = g.get("lap")
        if lap is None:
            continue
        if (int(lap) <= max_lap) != after:
            out.append(copy.deepcopy(g))
    return out


def make_snapshot(payload: dict, feature_cutoff_time: float,
                  target_start_time: float | None = None,
                  target_end_time: float | None = None,
                  drivers: list[str] | None = None) -> dict:
    cutoff = float(feature_cutoff_time)
    target_start = cutoff if target_start_time is None else float(target_start_time)
    target_end = (target_start + 30.0 if target_end_time is None
                  else float(target_end_time))
    cars = payload.get("cars", {})
    drivers = list(drivers or sorted(cars)[:2])
    missing = [d for d in drivers if d not in cars]
    exclusion = []
    if missing:
        exclusion.append(f"missing cars: {', '.join(missing)}")
    lap_cutoffs = [_lap_at_cutoff(cars[d], cutoff) for d in drivers if d in cars]
    max_lap = max((l for l in lap_cutoffs if l is not None), default=None)

    input_fields = {
        "session": {k: copy.deepcopy(payload.get(k))
                    for k in ("id", "year", "round", "session", "event", "circuit", "date")},
        "regulation": copy.deepcopy(payload.get("regulation", {})),
        "circuit_geometry": {
            "length": (payload.get("circuit_geometry") or {}).get("length"),
            "has_elevation": (payload.get("circuit_geometry") or {}).get("has_elevation"),
            "zones": copy.deepcopy((payload.get("circuit_geometry") or {}).get("zones", [])),
        },
        "weather": _weather_at_or_before(payload, cutoff),
        "cars": {d: _trace_until(cars[d], cutoff) for d in drivers if d in cars},
        "laps_completed": _laps_completed(payload, drivers, cutoff),
        "pit_events_seen": _pit_events_seen(payload, drivers, cutoff),
        "gaps_observed": _gaps_by_lap(payload, drivers, max_lap, after=False),
    }
    evaluation_fields = {
        "cars": {d: _trace_between(cars[d], target_start, target_end)
                 for d in drivers if d in cars},
        "laps": _future_laps(payload, drivers, target_start, target_end),
        "weather": _weather_after(payload, target_start, target_end),
        "gaps": _gaps_by_lap(payload, drivers, max_lap, after=True),
    }
    snap = {
        "feature_cutoff_time": cutoff,
        "target_start_time": target_start,
        "target_end_time": target_end,
        "input_fields": input_fields,
        "evaluation_fields": evaluation_fields,
        "quality_flags": {
            "has_weather_trace": bool(payload.get("weather_trace")),
            "has_two_cars": len(drivers) >= 2 and not missing,
            "has_future_window": any(v.get("n_samples", 0) > 0
                                     for v in evaluation_fields["cars"].values()),
        },
        "exclusion_reasons": exclusion,
        "provenance": {
            "builder": "xray.snapshots.make_snapshot",
            "causal_rule": "input_fields use samples/events at or before feature_cutoff_time",
            "input_fingerprint": fingerprint(input_fields),
            "evaluation_fingerprint": fingerprint(evaluation_fields),
        },
    }
    return snap
