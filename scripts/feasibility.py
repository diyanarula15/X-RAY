#!/usr/bin/env python3
"""Stage 2, Phase A: the four reality checks, on one real session.

Writes out/feasibility/r<round>.json. `scripts/feasibility_report.py` turns a
set of those into out/feasibility.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fastf1  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "xray" / "data" / "cache"
KMH = 3.6
SPEED_BINS = [340, 330, 320, 310, 300, 290]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--session", default="R")
    ap.add_argument("--max-drivers", type=int, default=20)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    t0 = time.time()
    session = fastf1.get_session(args.year, args.round, args.session)
    session.load(telemetry=True, laps=True, weather=True)
    out = {"year": args.year, "round": args.round, "session": args.session,
           "event": session.event["EventName"], "circuit": session.event["Location"],
           "date": str(session.event["EventDate"].date()),
           "load_seconds": round(time.time() - t0, 1)}
    laps = session.laps
    drivers = list(laps["Driver"].unique())[: args.max_drivers]
    out["n_drivers"] = len(drivers)
    out["n_laps"] = int(len(laps))

    # ---------------------------------------------------------- 1.1 rate
    fastest = laps.pick_fastest()
    car = fastest.get_car_data()
    dt = np.diff(car["SessionTime"].dt.total_seconds().to_numpy())
    dt = dt[np.isfinite(dt) & (dt > 0)]
    hist, edges = np.histogram(dt, bins=[0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0, 2.0, 10.0])
    out["dt_fastest_lap"] = {
        "driver": str(fastest["Driver"]), "n_samples": int(len(car)),
        "median_s": float(np.median(dt)), "mean_s": float(dt.mean()),
        "p10_s": float(np.percentile(dt, 10)), "p90_s": float(np.percentile(dt, 90)),
        "max_s": float(dt.max()), "effective_hz": float(1.0 / np.median(dt)),
        "histogram": {f"{edges[i]:.2f}-{edges[i+1]:.2f}": int(h) for i, h in enumerate(hist)},
    }
    car_d = fastest.get_car_data().add_distance()
    dd = np.diff(car_d["Distance"].to_numpy())
    out["dt_fastest_lap"]["median_dx_m"] = float(np.median(dd[np.isfinite(dd)]))

    # channels populated?
    chan = {}
    for c in ("Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"):
        if c in car.columns:
            v = car[c].to_numpy()
            try:
                v = v.astype(float)
                chan[c] = {"present": True, "frac_nonzero": float(np.mean(v != 0)),
                           "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
            except Exception:
                chan[c] = {"present": True, "frac_nonzero": None}
        else:
            chan[c] = {"present": False}
    out["channels"] = chan

    # ---------------------------------------------------------- 1.3 Z
    # Position data is not always attached to the fastest lap; fall back to the
    # session-wide stream for this driver before giving up on elevation.
    out["position"] = {"n_samples": 0, "has_Z": False, "source": None}
    for label, getter in (("lap", lambda: fastest.get_pos_data()),
                          ("session", lambda: session.pos_data[str(fastest["DriverNumber"])])):
        try:
            pos = getter()
            if pos is None or len(pos) == 0 or "X" not in pos.columns:
                continue
            z = pos["Z"].to_numpy().astype(float) if "Z" in pos.columns else np.array([])
            z = z[np.isfinite(z)]
            out["position"] = {
                "n_samples": int(len(pos)), "source": label,
                "has_Z": bool(len(z)) and float(np.nanstd(z)) > 0,
                "z_range_m": [float(np.nanmin(z)) / 10, float(np.nanmax(z)) / 10] if len(z) else None,
                "z_std_m": float(np.nanstd(z)) / 10 if len(z) else None,
                "z_frac_zero": float(np.mean(z == 0)) if len(z) else None,
                "xy_range_m": [float(np.ptp(pos["X"])) / 10, float(np.ptp(pos["Y"])) / 10],
            }
            break
        except Exception as exc:  # noqa: BLE001
            out["position"]["error"] = f"{label}: {exc}"
    

    # ---------------------------------------------------------- 1.2 window
    # Across every driver's laps: what fraction of the samples sit above each
    # speed? This is the calibration window the Stage 1 estimator depends on.
    counts = {b: 0 for b in SPEED_BINS}
    total = 0
    vmax_by_driver = {}
    dt_gaps_over_1s = 0
    lap_count = 0
    for drv in drivers:
        dl = laps.pick_drivers(drv)
        vmax = 0.0
        for _, lap in dl.iterlaps():
            try:
                cd = lap.get_car_data()
            except Exception:
                continue
            if len(cd) < 10:
                continue
            v = cd["Speed"].to_numpy().astype(float)
            total += len(v)
            for b in SPEED_BINS:
                counts[b] += int(np.sum(v > b))
            vmax = max(vmax, float(np.nanmax(v)))
            g = np.diff(cd["SessionTime"].dt.total_seconds().to_numpy())
            dt_gaps_over_1s += int(np.sum(g > 1.0))
            lap_count += 1
        vmax_by_driver[str(drv)] = vmax
    out["speed_window"] = {
        "total_samples": total, "laps_scanned": lap_count,
        "frac_above": {str(b): (counts[b] / total if total else 0.0) for b in SPEED_BINS},
        "vmax_session_kmh": max(vmax_by_driver.values()) if vmax_by_driver else 0.0,
        "vmax_by_driver": vmax_by_driver,
        "gaps_over_1s": dt_gaps_over_1s,
        "gaps_per_lap": dt_gaps_over_1s / max(lap_count, 1),
    }
    out["weather"] = {
        "available": session.weather_data is not None and len(session.weather_data) > 0,
        "air_temp_c": float(session.weather_data["AirTemp"].mean()) if session.weather_data is not None else None,
        "pressure_mbar": float(session.weather_data["Pressure"].mean()) if session.weather_data is not None and "Pressure" in session.weather_data else None,
        "wind_speed_ms": float(session.weather_data["WindSpeed"].mean()) if session.weather_data is not None and "WindSpeed" in session.weather_data else None,
    }
    out["total_seconds"] = round(time.time() - t0, 1)

    dst = Path("out/feasibility") / f"r{args.round}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "speed_window"}, indent=1))
    print("speed window:", {k: round(v, 4) for k, v in out["speed_window"]["frac_above"].items()},
          "vmax", round(out["speed_window"]["vmax_session_kmh"], 1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
