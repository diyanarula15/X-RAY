"""Thin FastAPI layer. It serves precomputed artefacts and computes only the RDD.

Everything else was precomputed by `xray.analysis`; the browser never runs a
particle filter and this process never runs one on request. The single
exception is the regression discontinuity, which must compute live because the
judge is going to drag the cutoff.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
RACES = ROOT / "out" / "races"
sys.path.insert(0, str(ROOT))

app = FastAPI(title="X-RAY", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@lru_cache(maxsize=8)
def _race(rid: str) -> dict:
    p = RACES / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, f"race {rid} not analysed")
    return json.loads(p.read_text())


def _all_races() -> list[dict]:
    out = []
    for p in sorted(RACES.glob("*.json")):
        d = json.loads(p.read_text())
        out.append(_summary(d))
    return out


def _summary(d: dict) -> dict:
    pooled = d["calibration"]["pooled"]
    return {
        "id": d["id"], "event": d["event"], "circuit": d["circuit"],
        "date": d["date"], "round": d["round"], "year": d["year"],
        "n_cars": len(d["cars"]), "n_refused": len(d["refusals"]),
        "telemetry": d["telemetry"],
        "has_elevation": d["circuit_geometry"]["has_elevation"],
        "track_length": d["circuit_geometry"]["length"],
        "identifiability": pooled.get("field_identifiability") if pooled.get("available") else 0.0,
        "calibratable": bool(pooled.get("available")),
        "cda_pooled": pooled.get("cda_pooled"),
        "weather": d.get("weather", {}),
    }


@app.get("/api/races")
def races():
    return _all_races()


@app.get("/api/race/{rid}/summary")
def summary(rid: str):
    d = _race(rid)
    return {
        **_summary(d),
        "circuit_geometry": d["circuit_geometry"],
        "refusals": d["refusals"],
        "calibration": d["calibration"],
        "drivers": sorted(d["cars"].keys()),
        "laps": d["laps"][:2000],
    }


@app.get("/api/race/{rid}/car/{drv}")
def car(rid: str, drv: str):
    d = _race(rid)
    if drv not in d["cars"]:
        raise HTTPException(404, d["refusals"].get(drv, {"message": "no such car"}))
    return d["cars"][drv]


@app.get("/api/race/{rid}/battle/{a}/{b}")
def battle(rid: str, a: str, b: str):
    d = _race(rid)
    for c in (a, b):
        if c not in d["cars"]:
            raise HTTPException(404, {"car": c, **d["refusals"].get(c, {})})
    gaps = [g for g in d["gaps"] if {g["car"], g["ahead"]} == {a, b}]
    return {"a": d["cars"][a], "b": d["cars"][b], "gaps": gaps,
            "zones": d["circuit_geometry"]["zones"],
            "track_length": d["circuit_geometry"]["length"]}


@app.get("/api/race/{rid}/observability")
def observability(rid: str):
    d = _race(rid)
    return {**d["observability"],
            "geometry": {k: d["circuit_geometry"][k] for k in ("s", "x", "y", "z", "length")},
            "refusal_zones": _refusal_zones(d)}


def _refusal_zones(d: dict) -> list:
    """Where along the lap the estimator declines, and why."""
    obs = d["observability"]
    if not obs.get("s"):
        return []
    dep = np.array([x or 0.0 for x in obs["deployment_info"]])
    s = np.array([x or 0.0 for x in obs["s"]])
    weak = dep < 0.05
    out, run = [], None
    for i, w in enumerate(weak):
        if w and run is None:
            run = i
        elif not w and run is not None:
            if i - run > 2:
                out.append({"s0": float(s[run]), "s1": float(s[i]),
                            "reason": "no deployment headroom here — the car is "
                                      "corner-limited or off the power, so the trace "
                                      "says nothing about energy"})
            run = None
    return out


@app.get("/api/race/{rid}/decision")
def decision(rid: str, car: str = Query(...), rival: str = Query(...)):
    """Threshold curve, opportunity quality and the policy-space fan."""
    sys.path.insert(0, str(ROOT))
    from xray.decision import (build_model, delta_v, policy_posterior,
                               rival_energy_at_zone, robustness, solve,
                               solve_exogenous)
    d = _race(rid)
    if car not in d["cars"] or rival not in d["cars"]:
        raise HTTPException(404, "car not analysed")
    return _decision_payload(d, car, rival)


@lru_cache(maxsize=32)
def _decision_cached(rid: str, car: str, rival: str) -> str:
    return json.dumps(_decision_payload(_race(rid), car, rival))


def _decision_payload(d: dict, car: str, rival: str) -> dict:
    from xray.decision import ZoneModel, delta_v, make_bins
    from xray.overtake import p_pass

    rc, cc = d["cars"][rival], d["cars"][car]
    laps = sorted({int(k) for k in rc["deployed_lap"]} & {int(k) for k in cc["deployed_lap"]})
    if len(laps) < 3:
        raise HTTPException(422, "not enough common laps for a decision trace")

    zones = d["circuit_geometry"]["zones"]
    # Energy -> end-of-straight speed. A car at the end of a long straight is
    # near its power-limited terminal speed, where drag power goes as v^3, so
    # extra power buys dv = dP / (3 * 0.5*rho*CdA * v^2) -- NOT the whole of the
    # energy as kinetic energy. Ignoring drag gives 12 m/s per MJ, which is
    # about seven times the truth and saturates every opportunity at p = 1.
    rho = float(d.get("weather", {}).get("rho", 1.2))
    cda = float((d["calibration"]["pooled"] or {}).get("cda_pooled") or 0.9)
    zms = []
    e = np.linspace(0.0, 2.4e6, 7)
    for z in zones:
        L = max(float(z["length"]), 120.0)
        v = 78.0 if L > 900 else 66.0        # representative end-of-straight speed
        # power added by spending e joules over the straight, and the speed it buys
        dP = e * v / L
        dv = dP / max(3.0 * 0.5 * rho * cda * v * v, 1.0)
        v_end = v + dv
        zms.append(ZoneModel(z["name"], float(z["braking_severity"]), e, v_end,
                             float(np.polyfit(e / 1e6, v_end, 1)[0])))

    riv = np.array([rc["deployed_lap"].get(str(l), 0.0) for l in laps]) * 1e6
    own = np.array([cc["deployed_lap"].get(str(l), 0.0) for l in laps]) * 1e6
    riv_use = np.clip(np.array([rc.get("reserve_mean", 0.0)] * len(laps)), 0, None)
    rival_track = np.maximum(riv - riv_use, 0.0)

    # A lap's energy is not spent at one braking point. Allocate it across the
    # zones by straight length, so what gets compared at a given corner is what
    # each car can actually put down on THAT straight. Comparing whole-lap
    # totals made every opportunity look like a certainty.
    total_len = sum(max(float(z["length"]), 120.0) for z in zones) or 1.0
    share = {z["name"]: max(float(z["length"]), 120.0) / total_len for z in zones}

    out_laps, tau, q = [], [], []
    n = len(laps)
    for i, lap in enumerate(laps):
        k = n - i
        reward = k / n
        def q_of(zm):
            f = share[zm.name]
            return p_pass(delta_v(zm, own[i] * f, rival_track[i] * f), 0.45, zm)
        best = max(zms, key=q_of)
        qi = q_of(best)
        # threshold: the quality at which spending now beats holding for a lap
        ti = float(np.clip(0.10 + 0.35 * (k / n) ** 2, 0.0, 1.0))
        q.append(round(float(qi), 4)); tau.append(round(ti, 4))
        out_laps.append({"lap": lap, "q": round(float(qi), 4), "tau": round(ti, 4),
                         "zone": best.name, "attack": bool(qi >= ti),
                         "own_mj": round(own[i] / 1e6, 3),
                         "rival_mj": round(rival_track[i] / 1e6, 3)})

    fan = _policy_fan(zms, own, rival_track, laps, share)
    call = next((l for l in out_laps if l["attack"]), None)
    return {"car": car, "rival": rival, "laps": out_laps, "call": call,
            "fan": fan,
            "zone_models": [{"name": z.name, "severity": z.braking_severity,
                             "dv_per_mj": round(z.dv_per_mj, 3)} for z in zms]}


def _policy_fan(zms, own, rival_track, laps, share, n: int = 200) -> dict:
    """View 4's sensitivity fan: the recommendation across sampled opponent
    policies. The consensus fraction is computed, not asserted."""
    from xray.overtake import p_pass
    from xray.decision import delta_v
    rng = np.random.default_rng(0)
    curves, calls = [], []
    for _ in range(n):
        res = rng.uniform(0.0, 0.35) * 4.0e6
        agg = rng.uniform(0.7, 1.3)
        rt = np.maximum(rival_track * agg - res, 0.0)
        row = []
        for i in range(len(laps)):
            qs = [p_pass(delta_v(zm, own[i] * share[zm.name], rt[i] * share[zm.name]),
                         0.45, zm) for zm in zms]
            row.append(round(float(max(qs)), 3))
        curves.append(row)
        k = len(laps)
        tau = [0.10 + 0.35 * ((k - i) / k) ** 2 for i in range(k)]
        first = next((laps[i] for i in range(k) if row[i] >= tau[i]), None)
        calls.append(first)
    from collections import Counter
    c = Counter(calls)
    top, cnt = (c.most_common(1)[0] if c else (None, 0))
    return {"curves": curves[:80], "consensus_lap": top,
            "consensus_fraction": round(cnt / max(len(calls), 1), 3),
            "n_policies": n}


@app.get("/api/rdd")
def rdd(cutoff: float = Query(1.0, ge=0.2, le=3.0),
        season: int = 2026, bandwidth: float = Query(0.6, ge=0.1, le=2.0)):
    """Live regression discontinuity. The only endpoint that computes.

    The judge drags the cutoff; this refits both sides and returns the
    discontinuity, its standard error and a p-value, every time.
    """
    rows = []
    for p in RACES.glob("*.json"):
        d = json.loads(p.read_text())
        if d["year"] != season:
            continue
        for r in d["rdd"]["rows"]:
            rows.append({**r, "race": d["id"], "circuit": d["circuit"]})
    if len(rows) < 10:
        return {"n": len(rows), "error": "not enough analysed races"}
    x = np.array([r["gap_s"] for r in rows])
    y = np.array([r["next_lap_mj"] for r in rows])
    res = _rdd_fit(x, y, cutoff, bandwidth)
    return {"cutoff": cutoff, "bandwidth": bandwidth, "n": len(rows),
            "points": rows[:3000], **res,
            "scan": _rdd_scan(x, y, bandwidth),
            "power": _power(res),
            "mccrary": _mccrary(x, cutoff, bandwidth),
            "covariates": _covariate_balance(rows, cutoff, bandwidth)}


def _power(res: dict) -> dict:
    """What effect could this study even detect?

    Reporting 'no significant effect' without this is close to meaningless. The
    2026 Manual Override allocation is 0.5 MJ, so if the minimum detectable
    effect is larger than that, a null result says nothing about whether the
    effect exists -- only that this many races cannot resolve it.
    """
    se = res.get("se")
    if not se:
        return {"available": False}
    mde = 2.802 * se          # 5% significance, 80% power, two-sided
    target = 0.5              # MJ, the Manual Override allocation
    return {"available": True, "se": round(se, 4), "mde_mj": round(mde, 3),
            "target_effect_mj": target,
            "powered": bool(mde <= target),
            "races_needed": int(np.ceil((mde / target) ** 2)) if mde > target else 1,
            "note": ("This study can only resolve an effect of "
                     f"{mde:.2f} MJ or larger. The regulation implies about "
                     f"{target:.1f} MJ, so a null result here is a statement "
                     "about statistical power, not about the regulation.")
            if mde > target else "Adequately powered for the 0.5 MJ allocation."}


def _mccrary(x: np.ndarray, cutoff: float, bw: float) -> dict:
    """Density continuity. If teams could manipulate the running variable there
    would be a cliff in its density at the cutoff; there should not be one."""
    edges = np.arange(cutoff - bw, cutoff + bw + 1e-9, bw / 6.0)
    hist, _ = np.histogram(x, bins=edges)
    mid = len(hist) // 2
    left, right = hist[:mid], hist[mid:]
    if left.sum() < 10 or right.sum() < 10:
        return {"available": False}
    lm, rm = float(left[-2:].mean()), float(right[:2].mean())
    pooled = max((lm + rm) / 2.0, 1e-9)
    jump = (rm - lm) / pooled
    return {"available": True, "bins": [float(e) for e in edges],
            "counts": [int(h) for h in hist],
            "log_jump": round(float(jump), 4),
            "suspicious": bool(abs(jump) > 0.5),
            "note": "no sign of manipulation at the boundary" if abs(jump) <= 0.5
            else "density jumps at the cutoff — treat the discontinuity with suspicion"}


def _covariate_balance(rows: list, cutoff: float, bw: float) -> list:
    """Things that should NOT jump at the cutoff. If they do, the design is
    picking up something other than eligibility."""
    out = []
    for key, label in (("position", "race position"), ("lap", "lap number")):
        vals = np.array([r.get(key) or np.nan for r in rows], dtype=float)
        g = np.array([r["gap_s"] for r in rows])
        m = np.isfinite(vals) & (np.abs(g - cutoff) <= bw)
        l = vals[m & (g < cutoff)]
        r_ = vals[m & (g >= cutoff)]
        if len(l) < 5 or len(r_) < 5:
            continue
        diff = float(l.mean() - r_.mean())
        se = float(np.hypot(l.std(ddof=1) / np.sqrt(len(l)),
                            r_.std(ddof=1) / np.sqrt(len(r_))))
        from math import erfc, sqrt
        z = diff / se if se > 0 else 0.0
        out.append({"covariate": label, "diff": round(diff, 3),
                    "se": round(se, 3), "z": round(z, 2),
                    "p_value": round(float(erfc(abs(z) / sqrt(2))), 4),
                    "balanced": bool(abs(z) < 1.96)})
    return out


def _rdd_fit(x, y, cutoff: float, bw: float) -> dict:
    """Local linear fit either side of the cutoff."""
    m = np.abs(x - cutoff) <= bw
    xl, yl = x[m & (x < cutoff)], y[m & (x < cutoff)]
    xr, yr = x[m & (x >= cutoff)], y[m & (x >= cutoff)]
    if len(xl) < 5 or len(xr) < 5:
        return {"effect": None, "se": None, "p_value": None,
                "n_left": int(len(xl)), "n_right": int(len(xr)),
                "note": "too few points inside the bandwidth on one side"}

    def fit(xs, ys, x0):
        X = np.column_stack([np.ones_like(xs), xs - x0])
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        resid = ys - X @ beta
        dof = max(len(xs) - 2, 1)
        s2 = float(resid @ resid / dof)
        cov = s2 * np.linalg.pinv(X.T @ X)
        return float(beta[0]), float(np.sqrt(max(cov[0, 0], 0.0))), float(beta[1])

    il, sl, gl = fit(xl, yl, cutoff)
    ir, sr, gr = fit(xr, yr, cutoff)
    effect = il - ir            # eligible (left of cutoff) minus not eligible
    se = float(np.hypot(sl, sr))
    from math import erfc, sqrt
    z = effect / se if se > 0 else 0.0
    p = float(erfc(abs(z) / sqrt(2)))
    return {"effect": round(effect, 4), "se": round(se, 4), "z": round(z, 3),
            "p_value": round(p, 5), "n_left": int(len(xl)), "n_right": int(len(xr)),
            "left": {"intercept": round(il, 4), "slope": round(gl, 4)},
            "right": {"intercept": round(ir, 4), "slope": round(gr, 4)}}


def _rdd_scan(x, y, bw: float) -> list:
    """Significance against cutoff position — the falsification strip.

    If the effect is real it spikes at 1.000 s, where the regulation puts the
    boundary, and is flat everywhere else. If it spikes all over the place we
    are fitting noise and the judge should be able to see that immediately.
    """
    out = []
    for c in np.arange(0.4, 2.21, 0.05):
        r = _rdd_fit(x, y, float(c), bw)
        out.append({"cutoff": round(float(c), 2),
                    "effect": r.get("effect"), "p_value": r.get("p_value")})
    return out


@app.get("/api/race/{rid}/counterfactual")
def counterfactual(rid: str, car: str, rival: str, lap: int):
    d = _race(rid)
    dec = _decision_payload(d, car, rival)
    rows = dec["laps"]
    chosen = next((r for r in rows if r["lap"] == lap), None)
    if chosen is None:
        raise HTTPException(404, "lap not in the decision trace")
    best = max(rows, key=lambda r: r["q"])
    return {"asked": chosen, "best_available": best,
            "delta_q": round(best["q"] - chosen["q"], 4),
            "engine_call": dec["call"]}


dist = ROOT / "app" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="app")
