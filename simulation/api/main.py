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

ROOT = Path(__file__).resolve().parent.parent        # simulation/  -- for app/dist
REPO_ROOT = ROOT.parent                               # repo root   -- for out/, xray/
RACES = REPO_ROOT / "out" / "races"
sys.path.insert(0, str(REPO_ROOT))

app = FastAPI(title="X-RAY", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@lru_cache(maxsize=8)
def _race(rid: str) -> dict:
    p = RACES / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, f"race {rid} not analysed")
    return json.loads(p.read_text())


@lru_cache(maxsize=1)
def _all_races_cached(stamp: str) -> str:
    """Summaries only. Re-parsing every race payload on every /api/races call
    meant reading ~160 MB of JSON to answer a request that returns 3 kB."""
    return json.dumps([_summary(json.loads(p.read_text()))
                       for p in sorted(RACES.glob("*.json"))])


def _all_races() -> list[dict]:
    stamp = ",".join(f"{p.name}:{p.stat().st_mtime_ns}"
                     for p in sorted(RACES.glob("*.json")))
    return json.loads(_all_races_cached(stamp))


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


@app.on_event("startup")
def _warm() -> None:
    """Preload so the first request is not the slow one."""
    try:
        _all_races()
        for p in sorted(RACES.glob("*.json"))[:1]:
            _race(p.stem)
    except Exception:
        pass


@app.get("/api/race/{rid}/summary")
def summary(rid: str):
    d = _race(rid)
    return {
        **_summary(d),
        "circuit_geometry": d["circuit_geometry"],
        "refusals": d["refusals"],
        "calibration": d["calibration"],
        "drivers": sorted(d["cars"].keys()),
        "battles": d.get("battles", []),
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
    d = _race(rid)
    if car not in d["cars"] or rival not in d["cars"]:
        raise HTTPException(404, "car not analysed")
    return _decision_payload(d, car, rival)


@app.get("/api/race/{rid}/p2")
def p2_decision(rid: str, car: str = Query(...), rival: str = Query(...),
                from_lap: int | None = Query(None)):
    """The P2 recommendation: which opportunity, which zone, how many joules.

    Serialisation only. Every number comes from
    `decision_service.evaluate_opportunity_decision`, which is the canonical
    orchestrator over the canonical solver -- this endpoint adds no arithmetic.
    """
    from xray.decision_service import evaluate_opportunity_decision

    d = _race(rid)
    if car not in d["cars"] or rival not in d["cars"]:
        raise HTTPException(404, "car not analysed")
    try:
        return evaluate_opportunity_decision(d, car, rival, from_lap=from_lap)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@lru_cache(maxsize=32)
def _decision_cached(rid: str, car: str, rival: str) -> str:
    return json.dumps(_decision_payload(_race(rid), car, rival))


def _decision_payload(d: dict, car: str, rival: str) -> dict:
    from xray.decision_service import evaluate_decision_trace_from_payload

    try:
        return evaluate_decision_trace_from_payload(d, car, rival)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


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


# ------------------------------------------------------------------ P3 evidence
# Every one of these returns the canonical service result unmodified. No
# thresholds, no rounding, no reshaping: a status word that differs between the
# API and the registry is a second opinion about what the model is.


@app.get("/api/p3/status")
def p3_status_endpoint():
    from xray.decision_service import p3_status
    return p3_status()


@app.get("/api/race/{rid}/p3")
def p3_race_endpoint(rid: str):
    from xray.decision_service import p3_race_evidence
    return p3_race_evidence(_race(rid))


@app.get("/api/race/{rid}/replay")
def p3_replay_endpoint(rid: str, car: str, rival: str, cutoff: float,
                       horizon: float = 30.0):
    from xray.decision_service import historical_replay
    return historical_replay(_race(rid), car, rival, cutoff, horizon_s=horizon)


dist = ROOT / "app" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="app")
