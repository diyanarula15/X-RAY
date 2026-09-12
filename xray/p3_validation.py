"""P3 real predictive validation.

The inputs are analysed race payloads from the canonical realfit path. The
prediction rollout uses :func:`xray.vehicle.step`; this module only chooses
causal cutoffs, target samples and energy ablations.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .decision_service import _PayloadTrack
from .vehicle import CarState, VehicleParams, set_car_mass, step

SNAPSHOT_SCHEMA_VERSION = "p3-real-validation-v1"
TARGETS = ("future_speed_5s", "straight_speed_3s", "braking_point_speed")


@dataclass(frozen=True)
class ValidationExample:
    race_id: str
    event: str
    circuit: str
    driver: str
    target: str
    cutoff_t: float
    target_t: float
    horizon_s: float
    s0: float
    v0: float
    lap: int
    target_s: float
    target_v: float
    xray_energy_j: float
    cda_straight: float
    cda_lo: float | None
    cda_hi: float | None
    exclusion_reason: str = ""


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _finite_arrays(trace: dict) -> dict[str, np.ndarray]:
    out = {}
    for k in ("t", "s", "v", "lap", "usable_mean"):
        if k not in trace:
            return {}
        out[k] = np.asarray([np.nan if x is None else x for x in trace[k]], dtype=float)
    ok = np.isfinite(out["t"]) & np.isfinite(out["s"]) & np.isfinite(out["v"])
    ok &= np.isfinite(out["lap"]) & np.isfinite(out["usable_mean"])
    return {k: v[ok] for k, v in out.items()}


def _next_zone_distance(track, s: float) -> float | None:
    best = None
    for z in track.zones:
        d = (float(z.s_straight_end) - float(s)) % float(track.length)
        if 20.0 < d <= 900.0:
            best = d if best is None else min(best, d)
    return best


def _zone_straight_end_ahead(track, s: float) -> float | None:
    x = float(s) % float(track.length)
    for z in track.zones:
        start, end = float(z.s_straight_start), float(z.s_straight_end)
        if start <= end and start <= x < end:
            return end
        if start > end and (x >= start or x < end):
            return end
    return None


def _target_index_after(t: np.ndarray, cutoff: float, desired: float,
                        max_late_s: float = 2.0) -> int | None:
    k = int(np.searchsorted(t, desired, side="right"))
    if k >= len(t) or not (t[k] > cutoff):
        return None
    if t[k] - desired > max_late_s:
        return None
    return k


def build_validation_examples(payload: dict, stride: int = 12) -> tuple[list[ValidationExample], dict]:
    track = _PayloadTrack(payload)
    race_id = str(payload.get("id") or payload.get("event") or "unknown")
    event = str(payload.get("event") or race_id)
    circuit = str(payload.get("circuit") or "unknown")
    examples: list[ValidationExample] = []
    exclusions: dict[str, int] = {}

    for driver, car in sorted((payload.get("cars") or {}).items()):
        trace = _finite_arrays(car.get("trace", {}))
        if not trace or len(trace["t"]) < 5:
            exclusions["missing usable trace"] = exclusions.get("missing usable trace", 0) + 1
            continue
        t, s, v, lap, e_mj = (trace["t"], trace["s"], trace["v"],
                              trace["lap"].astype(int), trace["usable_mean"])
        cda = float(car.get("cda", 0.0) or 0.0)
        if not np.isfinite(cda) or cda <= 0.05:
            cda = 0.66
            exclusions["non-positive CdA fell back to config-like value"] = (
                exclusions.get("non-positive CdA fell back to config-like value", 0) + 1)
        cda_lo = car.get("cda_lo")
        cda_hi = car.get("cda_hi")

        for i in range(0, len(t) - 2, stride):
            cutoff = float(t[i])
            base = dict(
                race_id=race_id, event=event, circuit=circuit, driver=str(driver),
                cutoff_t=cutoff, s0=float(s[i]), v0=float(v[i]), lap=int(lap[i]),
                xray_energy_j=float(max(e_mj[i], 0.0) * 1e6), cda_straight=cda,
                cda_lo=None if cda_lo is None else float(cda_lo),
                cda_hi=None if cda_hi is None else float(cda_hi),
                exclusion_reason="",
            )

            k = _target_index_after(t, cutoff, cutoff + 5.0)
            if k is not None:
                examples.append(ValidationExample(
                    **base, target="future_speed_5s", target_t=float(t[k]),
                    horizon_s=float(t[k] - cutoff), target_s=float(s[k]),
                    target_v=float(v[k])))

            end_s = _zone_straight_end_ahead(track, s[i])
            k = _target_index_after(t, cutoff, cutoff + 3.0)
            if k is not None and end_s is not None and int(lap[k]) == int(lap[i]):
                dist = (float(s[k]) - float(s[i])) % float(track.length)
                room = (float(end_s) - float(s[i])) % float(track.length)
                if 0.0 < dist < room:
                    examples.append(ValidationExample(
                        **base, target="straight_speed_3s", target_t=float(t[k]),
                        horizon_s=float(t[k] - cutoff), target_s=float(s[k]),
                        target_v=float(v[k])))

            d_zone = _next_zone_distance(track, s[i])
            if d_zone is not None:
                target_s = (float(s[i]) + d_zone) % float(track.length)
                future = np.flatnonzero((t > cutoff) & (lap == lap[i]))
                if len(future):
                    j = int(future[np.argmin(np.abs(s[future] - target_s))])
                    if t[j] > cutoff and abs(((s[j] - target_s + track.length / 2)
                                              % track.length) - track.length / 2) <= 80.0:
                        examples.append(ValidationExample(
                            **base, target="braking_point_speed", target_t=float(t[j]),
                            horizon_s=float(t[j] - cutoff), target_s=float(s[j]),
                            target_v=float(v[j])))
    return examples, exclusions


def _params_for_example(cfg: dict, ex: ValidationExample,
                        mass_car: float | None = None,
                        cda_straight: float | None = None) -> VehicleParams:
    p = VehicleParams.from_config(cfg)
    cdas = float(ex.cda_straight if cda_straight is None else cda_straight)
    ratio = p.cda_corner / max(p.cda_straight, 1e-9)
    return VehicleParams(
        mass_car=float(p.mass_car if mass_car is None else mass_car),
        fuel_start=p.fuel_start, fuel_burn_per_lap=p.fuel_burn_per_lap,
        cda_straight=cdas, cda_corner=cdas * ratio, crr=p.crr, rho=p.rho,
        brake_decel_max=p.brake_decel_max,
        drivetrain_eff=p.drivetrain_eff, cla_straight=p.cla_straight,
        cla_corner=p.cla_corner,
        brake_calibration_v_ms=p.brake_calibration_v_ms)


def rollout_speed(payload: dict, ex: ValidationExample, energy_j: float,
                  cfg: dict | None = None, *, mass_car: float | None = None,
                  cda_straight: float | None = None) -> float:
    """Predict target speed from the cutoff state with the canonical stepper."""
    cfg = cfg or load_config()
    track = _PayloadTrack(payload)
    params = _params_for_example(cfg, ex, mass_car=mass_car, cda_straight=cda_straight)
    set_car_mass(params.mass_car)
    st = CarState(s=ex.s0, s_total=0.0, v=max(ex.v0, 1.0), E=max(float(energy_j), 0.0),
                  fuel=params.fuel_start * 0.5, lap=ex.lap)
    dt = 0.05
    elapsed = 0.0
    guard = 0
    target_dist = (ex.target_s - ex.s0) % track.length
    by_distance = ex.target == "braking_point_speed"
    while elapsed < ex.horizon_s and guard < 20_000:
        if by_distance and st.s_total >= target_dist:
            break
        step(track, st, params, 350_000.0, dt)
        elapsed += dt
        guard += 1
    return float(st.v)


def leave_one_race_splits(examples: list[ValidationExample]) -> dict[str, dict]:
    races = sorted({e.race_id for e in examples})
    return {race: {
        "train": [e for e in examples if e.race_id != race],
        "test": [e for e in examples if e.race_id == race],
    } for race in races}


def fixed_energy_by_target(train: list[ValidationExample]) -> dict[str, float]:
    out = {}
    for target in TARGETS:
        vals = [e.xray_energy_j for e in train if e.target == target and np.isfinite(e.xray_energy_j)]
        out[target] = float(np.mean(vals)) if vals else 2.0e6
    return out


def _metrics(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0, "mae": None, "rmse": None}
    a = np.asarray(errors, dtype=float)
    return {"n": int(len(a)), "mae": float(np.mean(np.abs(a))),
            "rmse": float(np.sqrt(np.mean(a * a)))}


def score_predictions(payloads: list[dict], examples: list[ValidationExample],
                      cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    by_payload = {str(p.get("id") or p.get("event")): p for p in payloads}
    rows = []
    splits = leave_one_race_splits(examples)
    for race, split in splits.items():
        fixed = fixed_energy_by_target(split["train"])
        for ex in split["test"]:
            payload = by_payload[ex.race_id]
            preds = {
                "xray": rollout_speed(payload, ex, ex.xray_energy_j, cfg),
                "fixed_energy": rollout_speed(payload, ex, fixed[ex.target], cfg),
                "energy_neutral": rollout_speed(payload, ex, 0.0, cfg),
            }
            rows.append({
                **asdict(ex),
                "fixed_energy_j": fixed[ex.target],
                "predictions": preds,
                "errors": {k: float(v - ex.target_v) for k, v in preds.items()},
            })

    per_target = {}
    for target in TARGETS:
        target_rows = [r for r in rows if r["target"] == target]
        per_race = {}
        for race in sorted({r["race_id"] for r in target_rows}):
            rr = [r for r in target_rows if r["race_id"] == race]
            per_race[race] = {model: _metrics([r["errors"][model] for r in rr])
                              for model in ("xray", "fixed_energy", "energy_neutral")}
        aggregate = {model: _metrics([r["errors"][model] for r in target_rows])
                     for model in ("xray", "fixed_energy", "energy_neutral")}
        paired = {
            "xray_minus_fixed_mae": (
                None if not target_rows else float(np.mean(
                    np.abs([r["errors"]["xray"] for r in target_rows])
                    - np.abs([r["errors"]["fixed_energy"] for r in target_rows])))),
            "xray_minus_neutral_mae": (
                None if not target_rows else float(np.mean(
                    np.abs([r["errors"]["xray"] for r in target_rows])
                    - np.abs([r["errors"]["energy_neutral"] for r in target_rows])))),
        }
        per_target[target] = {"per_race": per_race, "aggregate": aggregate,
                              "paired": paired}
    return {"rows": rows, "metrics": per_target}


def nuisance_sensitivity(payloads: list[dict], examples: list[ValidationExample],
                         cfg: dict | None = None, limit: int = 12) -> list[dict]:
    cfg = cfg or load_config()
    by_payload = {str(p.get("id") or p.get("event")): p for p in payloads}
    selected = examples[::max(len(examples) // max(limit, 1), 1)][:limit]
    out = []
    for ex in selected:
        payload = by_payload[ex.race_id]
        variants = {
            "nominal": {},
            "mass_768kg": {"mass_car": 768.0},
            "mass_821kg": {"mass_car": 821.0},
        }
        if ex.cda_lo is not None and ex.cda_lo > 0.05:
            variants["cda_low"] = {"cda_straight": ex.cda_lo}
        if ex.cda_hi is not None and ex.cda_hi > 0.05:
            variants["cda_high"] = {"cda_straight": ex.cda_hi}
        speeds = {
            name: rollout_speed(payload, ex, ex.xray_energy_j, cfg, **kw)
            for name, kw in variants.items()
        }
        out.append({
            "race_id": ex.race_id, "driver": ex.driver, "target": ex.target,
            "cutoff_t": ex.cutoff_t,
            "xray_energy_mj": ex.xray_energy_j / 1e6,
            "predicted_speed_mps": speeds,
            "range_mps": float(max(speeds.values()) - min(speeds.values())),
        })
    return out


def build_real_validation(payloads: list[dict], cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    examples: list[ValidationExample] = []
    exclusions: dict[str, int] = {}
    for p in payloads:
        ex, exc = build_validation_examples(p)
        examples.extend(ex)
        for k, v in exc.items():
            exclusions[k] = exclusions.get(k, 0) + v
    scored = score_predictions(payloads, examples, cfg)
    sens = nuisance_sensitivity(payloads, examples, cfg)
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "targets": list(TARGETS),
        "split": {
            "method": "leave-one-race-out",
            "race_groups": sorted({e.race_id for e in examples}),
            "fixed_energy": "mean inferred energy by target over training races only",
        },
        "dataset": {
            "n_examples": len(examples),
            "n_races": len({e.race_id for e in examples}),
            "by_target": {t: sum(1 for e in examples if e.target == t) for t in TARGETS},
            "exclusions": exclusions,
            "limitations": [
                "track status unavailable; SC/VSC/yellow contamination cannot be removed",
                "weather trace unavailable; only session summary rho is present",
                "pit data unavailable in the corpus manifest",
                "no real battery ground truth; validation is future-observable only",
            ],
        },
        "metrics": scored["metrics"],
        "nuisance_sensitivity": {
            "inferred_energy_rerun": {
                "status": "not_measured_from_current_corpus",
                "reason": ("existing analysed payloads persist inferred traces but not the "
                           "raw canonical gridded FastF1 frames/throttle/brake inputs "
                           "needed to rerun realfit under alternate nuisance assumptions"),
            },
            "downstream_prediction_at_fixed_energy": sens,
        },
        "rows_fingerprint": fingerprint(scored["rows"]),
    }


def load_payload_paths(paths: list[str | Path]) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]
