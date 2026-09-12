"""RaceAnalysis: run the whole pipeline over a real session, emit one JSON.

The browser never runs a particle filter and the backend never computes on
request. Everything the six views need is precomputed here.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .constants import E_STORE_MAX, MOM_GAP_S, P_MGUK_MAX
from .data.circuits import RealTrack, geometry_from_session
from .data.ingest import ingest_session
from .realfit import (belief_from_deployment, build_kin, common_mode,
                      coast_phases, deployment_trace, fit_nuisance_real,
                      observability, pool_field)

OUT = Path(__file__).resolve().parent.parent / "out" / "races"
MIN_IDENT = 0.15


def _f(a, nd=4):
    """Compact float list for JSON — the browser does not need 15 digits."""
    a = np.asarray(a, dtype=float)
    a = np.where(np.isfinite(a), a, None)
    return [None if v is None else round(float(v), nd) for v in a]


def _downsample(a, step):
    return np.asarray(a)[::step]


def _opt_int(lap, key):
    if key not in lap:
        return None
    v = lap[key]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if np.isfinite(f) else None


def _opt_bool(lap, key):
    if key not in lap:
        return None
    v = lap[key]
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
    except (TypeError, ValueError):
        return None
    return bool(v)


def _opt_seconds(lap, key):
    """A pandas Timedelta column to seconds, or None. ORACLE fields only."""
    if key not in lap:
        return None
    v = lap[key]
    try:
        sec = float(v.total_seconds())
    except (AttributeError, TypeError, ValueError):
        return None
    return None if not np.isfinite(sec) else round(sec, 3)


def analyse(year: int, rnd: int, session_name: str = "R",
            drivers: list[str] | None = None, max_laps: int | None = None,
            mass_kg: float = 790.0, n_particles: int = 400,
            verbose: bool = True) -> dict:
    import fastf1
    fastf1.Cache.enable_cache(str(Path(__file__).resolve().parent / "data" / "cache"))
    ses = fastf1.get_session(year, rnd, session_name)
    ses.load(telemetry=True, laps=True, weather=True)

    geo = geometry_from_session(ses)
    track = RealTrack(geo)
    sd = ingest_session(year, rnd, session_name, drivers=drivers,
                        max_laps=max_laps, verbose=False)
    rho = sd.weather.get("rho", 1.20)

    fits, kins, traces, beliefs, refusals = {}, {}, {}, {}, {}
    for drv, df in sd.frames.items():
        d = df[df["usable"]] if "usable" in df else df
        if len(d) < 500:
            refusals[drv] = {"kind": "insufficient_data",
                             "message": f"only {len(d)} usable samples after the "
                                        f"gap filter — not enough clean running to estimate"}
            continue
        kin = build_kin(d, track, mass_kg, rho)
        try:
            fit = fit_nuisance_real(kin, rho)
        except ValueError as exc:
            refusals[drv] = {"kind": "calibration_failed", "message": str(exc)}
            continue
        fits[drv], kins[drv] = fit, kin
        if fit.identifiability < MIN_IDENT:
            refusals[drv] = {
                "kind": "low_identifiability",
                "message": (f"drag area is only identified to "
                            f"[{fit.cda_lo:.2f}, {fit.cda_hi:.2f}] m² at this circuit "
                            f"— the speed range this car ran does not pin it down"),
                "identifiability": fit.identifiability}

    pooled = pool_field(fits, MIN_IDENT)
    cm = common_mode(fits, kins)

    # Cars that could not calibrate alone inherit the pooled field constraint —
    # that is the whole point of §2.3. They are still flagged as inherited.
    for drv, fit in fits.items():
        use = fit
        if fit.identifiability < MIN_IDENT and pooled.get("available"):
            object.__setattr__(use, "cda_hat", pooled["cda_pooled"])
            object.__setattr__(use, "cda_lo", pooled["cda_lo"])
            object.__setattr__(use, "cda_hi", pooled["cda_hi"])
        tr = deployment_trace(kins[drv], use)
        traces[drv] = tr
        beliefs[drv] = belief_from_deployment(kins[drv], tr, use,
                                              n_particles=n_particles)
        # the UI shows the split-corrected deployment, matching the belief
        tr["deploy"] = beliefs[drv].get("deploy_star", tr["deploy"])
        if verbose:
            dl = beliefs[drv]["deployed_lap"]
            med = np.median(list(dl.values())) if dl else 0.0
            print(f"  {drv}: CdA {use.cda_hat:.3f} ident {fit.identifiability:.2f} "
                  f"deploy {med/1e6:.2f} MJ/lap")

    # -------------------------------------------------------------- laps/gaps
    laps = ses.laps
    lap_rows = []
    for _, lap in laps.iterlaps():
        try:
            lt = float(lap["LapTime"].total_seconds())
        except Exception:
            lt = float("nan")
        try:
            # session time at which this car crossed the line ending this lap
            tend = float(lap["Time"].total_seconds())
        except Exception:
            tend = float("nan")
        lap_rows.append({
            "driver": str(lap["Driver"]), "lap": int(lap["LapNumber"]),
            "position": None if np.isnan(lap["Position"]) else int(lap["Position"]),
            "lap_time": None if not np.isfinite(lt) else round(lt, 3),
            "t_end": None if not np.isfinite(tend) else round(tend, 3),
            "compound": str(lap["Compound"]) if "Compound" in lap else None,
            # AGE IN LAPS. Not wear. xray.tyres models wear separately and
            # test_tyres asserts the two can never be the same number.
            "tyre_life": None if not np.isfinite(lap.get("TyreLife", np.nan))
            else int(lap["TyreLife"]),
            # Verified present in the installed FastF1 (3.8.3) Laps columns:
            # Stint, FreshTyre, PitInTime, PitOutTime. Serialised as None when
            # absent rather than defaulted, so a missing field stays missing.
            "stint": _opt_int(lap, "Stint"),
            "fresh_tyre": _opt_bool(lap, "FreshTyre"),
            # ORACLE. These are in the future relative to any decision being
            # replayed. xray.stint only reads them through oracle_pit_context,
            # whose output decision_service refuses. They are here to SCORE a
            # pit inference after the fact, never to make one.
            "pit_in_time_s": _opt_seconds(lap, "PitInTime"),
            "pit_out_time_s": _opt_seconds(lap, "PitOutTime"),
        })

    # gap to the car ahead at each lap, from race position and lap time
    by_lap = {}
    for r in lap_rows:
        by_lap.setdefault(r["lap"], []).append(r)
    # The real gap: the difference in session time at which two cars crossed the
    # same line on the same lap. An earlier version differenced their lap TIMES,
    # which is a completely different quantity -- two cars a pit stop apart can
    # post identical lap times -- and it made the Manual Override boundary
    # unfindable because the running variable was not the gap at all.
    gaps = []
    for lp, rows in sorted(by_lap.items()):
        rows = [r for r in rows if r["position"] and r["t_end"] is not None]
        rows.sort(key=lambda r: r["position"])
        for i in range(1, len(rows)):
            a, b = rows[i - 1], rows[i]
            g = b["t_end"] - a["t_end"]
            if not (0.0 < g < 30.0):     # lapped or stale timing
                continue
            gaps.append({"lap": lp, "car": b["driver"], "ahead": a["driver"],
                         "gap_s": round(g, 3), "position": b["position"]})

    # ------------------------------------------------------------- battles
    # Pairs that actually raced each other: laps spent within 2 s. The app
    # defaults to the best of these so the theatre opens on a fight, not on
    # two cars half a lap apart chosen alphabetically.
    close = {}
    for g in gaps:
        if g["gap_s"] < 2.0 and g["car"] in fits and g["ahead"] in fits:
            k = (g["car"], g["ahead"])
            close.setdefault(k, []).append(g)
    battles = sorted(
        [{"car": k[0], "ahead": k[1], "laps_close": len(v),
          "median_gap": float(np.median([x["gap_s"] for x in v])),
          "first_lap": int(min(x["lap"] for x in v))} for k, v in close.items()],
        key=lambda b: (-b["laps_close"], b["median_gap"]))[:8]

    # ------------------------------------------------------------------ RDD
    rdd = _rdd_rows(beliefs, gaps)

    # ------------------------------------------------------- observability
    obs_ref = None
    for drv in fits:
        if fits[drv].identifiability >= MIN_IDENT:
            obs_ref = observability(kins[drv], fits[drv], track)
            break
    if obs_ref is None and fits:
        drv = next(iter(fits))
        obs_ref = observability(kins[drv], fits[drv], track)

    step = max(len(geo.s) // 600, 1)
    payload = {
        "id": f"{year}_r{rnd}_{session_name}",
        "year": year, "round": rnd, "session": session_name,
        "event": sd.event, "circuit": sd.circuit, "date": sd.date,
        "weather": sd.weather,
        # The session MEAN stays exactly where it was for the UI header. The
        # time-resolved trace is added beside it, because a decision on lap 3
        # must not be made with lap 50's air. Causal selection happens in
        # decision_service.environment_at_opportunity.
        "weather_trace": getattr(sd, "weather_trace", []) or [],
        "telemetry": {
            "median_hz": round(float(1.0 / np.nanmedian(
                [q.median_dt for q in sd.quality if np.isfinite(q.median_dt)])), 2),
            "laps_total": len(sd.quality),
            "laps_usable": sum(1 for q in sd.quality if q.usable),
            "gap_limit_s": 1.0,
        },
        "circuit_geometry": {
            "length": round(geo.length, 1),
            "has_elevation": bool(geo.has_elevation),
            "s": _f(_downsample(geo.s, step), 1),
            "x": _f(_downsample(geo.xy[:, 0], step), 2),
            "y": _f(_downsample(geo.xy[:, 1], step), 2),
            "z": _f(_downsample(geo.z, step), 2),
            "grade": _f(_downsample(geo.grade, step), 5),
            "curvature": _f(_downsample(geo.curvature, step), 6),
            "is_corner": [bool(b) for b in _downsample(geo.is_corner, step)],
            "zones": track.zones,
        },
        "calibration": {
            "pooled": pooled,
            "common_mode": {k: v for k, v in cm.items() if k != "common_mode_w"},
            "per_car": {d: {**{k: v for k, v in asdict(f).items() if k != "notes"},
                            "notes": f.notes} for d, f in fits.items()},
        },
        "refusals": refusals,
        "battles": battles,
        "cars": {},
        "laps": lap_rows,
        "gaps": gaps,
        "rdd": rdd,
        "observability": {
            "s": _f(obs_ref["s"], 1) if obs_ref else [],
            "deployment_info": _f(obs_ref["deployment_info"], 4) if obs_ref else [],
            "nuisance_info": _f(obs_ref["nuisance_info"], 4) if obs_ref else [],
            "speed": _f(obs_ref["speed"], 2) if obs_ref else [],
        },
    }

    for drv in fits:
        kin, tr, bel = kins[drv], traces[drv], beliefs[drv]
        keep = np.isfinite(kin.v) & kin.valid
        st = max(int(keep.sum()) // 3500, 1)
        sel = np.flatnonzero(keep)[::st]
        payload["cars"][drv] = {
            "driver": drv,
            "identifiability": round(fits[drv].identifiability, 3),
            "cda": round(fits[drv].cda_hat, 4),
            "cda_lo": round(fits[drv].cda_lo, 4),
            "cda_hi": round(fits[drv].cda_hi, 4),
            "inherited_pooled": bool(fits[drv].identifiability < MIN_IDENT),
            "n_coast_samples": int(fits[drv].n_coast),
            "reserve_mean": round(bel["reserve_mean"], 1),
            "recovery_balance": round(float(bel.get("balance", 1.0)), 3),
            "deployed_lap": {str(k): round(v / 1e6, 4) for k, v in bel["deployed_lap"].items()},
            "harvested_lap": {str(k): round(v / 1e6, 4) for k, v in bel["harvested_lap"].items()},
            "trace": {
                "t": _f(kin.t[sel], 2),      # session time: the shared clock
                "s": _f(kin.s[sel], 1),
                "lap": [int(x) for x in kin.lap[sel]],
                "v": _f(kin.v[sel], 2),
                "deploy_kw": _f(tr["deploy"][sel] / 1e3, 1),
                "deploy_lo_kw": _f(tr["deploy_lo"][sel] / 1e3, 1),
                "deploy_hi_kw": _f(tr["deploy_hi"][sel] / 1e3, 1),
                "harvest_kw": _f(tr["harvest"][sel] / 1e3, 1),
                "usable_mean": _f(bel["usable_mean"][sel] / 1e6, 4),
                "usable_p10": _f(bel["usable_p10"][sel] / 1e6, 4),
                "usable_p90": _f(bel["usable_p90"][sel] / 1e6, 4),
                "dry": [bool(b) for b in bel["dry"][sel]],
                "coast": [bool(b) for b in coast_phases(kin)[sel]],
            },
            # the particle cloud drives the 3D signature visual; 120 particles
            # at every 6th frame is plenty to read as a cloud and keeps the
            # payload small enough to load instantly
            "cloud_stride": 5,
            "cloud": [[round(float(x), 2) for x in row[::4]]
                      for row in (bel["cloud"][sel][::5] / 1e6)],
        }
    return payload


def _rdd_rows(beliefs: dict, gaps: list) -> dict:
    """Regression discontinuity at the Manual Override eligibility boundary.

    The one validation available on real data. There is no public ground-truth
    energy channel, so the claim cannot be checked directly. What can be checked
    is a prediction the *regulation* makes: a car within 1.000 s of the car ahead
    at the detection point becomes eligible for extra deployment, and a car at
    1.001 s does not. If our estimates are measuring deployment, they should jump
    at that boundary and be smooth everywhere else.
    """
    rows = []
    for g in gaps:
        car = g["car"]
        b = beliefs.get(car)
        if not b:
            continue
        nxt = b["deployed_lap"].get(g["lap"] + 1)
        if nxt is None or not np.isfinite(nxt):
            continue
        rows.append({"driver": car, "lap": g["lap"], "gap_s": g["gap_s"],
                     "next_lap_mj": round(nxt / 1e6, 4), "position": g["position"]})
    return {"cutoff": MOM_GAP_S, "n": len(rows), "rows": rows}


def write(payload: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{payload['id']}.json"
    p.write_text(json.dumps(payload, separators=(",", ":")))
    return p
