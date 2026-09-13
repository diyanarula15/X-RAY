"""Thin FastAPI layer. It serves precomputed artefacts and computes only the RDD.

Everything else was precomputed by `xray.analysis`; the browser never runs a
particle filter and this process never runs one on request. The single
exception is the regression discontinuity, which must compute live because the
judge is going to drag the cutoff.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache, partial
from math import isfinite
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent        # simulation/  -- for app/dist
REPO_ROOT = ROOT.parent                               # repo root   -- for out/, xray/
RACES = REPO_ROOT / "out" / "races"
OUT = REPO_ROOT / "out"
DECISIONS = REPO_ROOT / "out" / "decisions"
sys.path.insert(0, str(REPO_ROOT))

# `build_bundle` costs ~54 s of pure-Python `vehicle.step` calls for a pair
# nobody has precomputed (see `build_bundle`'s docstring). Running that inline
# on a `def` route still starves every other request in the process even
# though FastAPI puts sync routes on a threadpool: a tight pure-Python loop
# holds the GIL almost continuously, so `GET /api/race/{rid}` -- what the
# RacePicker calls to switch track -- sat queued behind it, and switching races
# looked broken for as long as Cockpit or Situations was mid-solve. A thread
# doesn't fix that (still one GIL); it has to run in a separate process, the
# way `scripts/16.precompute_decisions.py` already does for the same workload.
_BUNDLE_POOL: ProcessPoolExecutor | None = None
# One solve per pair, keyed by `bundle_path(...).stem`. Cockpit fires
# `/decision` and `/p2` for the same pair on one mount and Situations adds a
# third, so this is de-dupe as much as it is progress reporting: without it the
# same bundle would be solved three times concurrently.
_BUNDLE_JOBS: dict[str, dict] = {}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Preload so the first request is not the slow one."""
    global _BUNDLE_POOL

    # Vercel serverless functions should not create our local
    # multiprocessing pool during application startup.
    if os.getenv("VERCEL"):
        _BUNDLE_POOL = None
        print("X-RAY: running on Vercel, process pool disabled", flush=True)
    else:
        _BUNDLE_POOL = ProcessPoolExecutor(max_workers=2)

    try:
        print("X-RAY: preload starting", flush=True)

        _all_races()

        for p in sorted(RACES.glob("*.json"))[:1]:
            _race(p.stem)

        print("X-RAY: preload complete", flush=True)
    except Exception as exc:
        print(
            f"X-RAY preload warning: {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()

    try:
        yield
    finally:
        if _BUNDLE_POOL is not None:
            _BUNDLE_POOL.shutdown(wait=False, cancel_futures=True)

        _BUNDLE_POOL = None


app = FastAPI(title="X-RAY", version="2.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# One slot per race artefact on disk, with a floor so an empty `out/races/` still
# caches. At the old fixed 8 every request evicted a neighbour and re-parsed
# ~13 MB of JSON to answer it -- and the comment claimed "12 against 10 races"
# while `out/races/` holds 5, so the margin it described did not exist. Counted
# at import instead of written down: a hardcoded count goes stale the first time
# a round is analysed.
_RACE_CACHE_SLOTS = max(8, len(list(RACES.glob("*.json"))) * 2)


@lru_cache(maxsize=_RACE_CACHE_SLOTS)
def _race_at(rid: str, mtime_ns: int) -> dict:
    p = RACES / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, f"race {rid} not analysed")
    return json.loads(p.read_text())


def _race(rid: str) -> dict:
    """Keyed on mtime, not just id. Cached on id alone, re-analysing a round
    through POST /api/races/analyze wrote a new artefact that this process then
    refused to read for the rest of its life -- the button appeared to do
    nothing for any round already loaded."""
    p = RACES / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, f"race {rid} not analysed")
    return _race_at(rid, p.stat().st_mtime_ns)


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


# In-memory job registry for on-demand Stage 2 analysis. This process is a
# single dev-server instance (the same one `docs/STARTUP.md` tells you to run
# with `uvicorn ... --port 8011`), so a module-level dict is the whole state
# store; it does not survive a restart, same as the CLI script it wraps.
_JOBS: dict[str, dict] = {}


def _run_analysis_job(job_id: str, year: int, round_: int, session: str,
                       drivers: list[str] | None, max_laps: int | None,
                       particles: int) -> None:
    from xray.analysis import analyse, write

    try:
        payload = analyse(year, round_, session, drivers=drivers,
                          max_laps=max_laps, n_particles=particles)
        path = write(payload)
        _JOBS[job_id] = {"status": "done", "race_id": payload["id"],
                         "path": str(path)}
    except Exception as exc:  # noqa: BLE001 -- surfaced to the poller, not swallowed
        _JOBS[job_id] = {"status": "error", "message": str(exc),
                         "traceback": traceback.format_exc()}


@app.post("/api/races/analyze")
def analyze_race(body: dict = Body(...)):
    """Run the same pipeline `scripts/08.b_analyse_race.py --round N` runs,
    on demand, from the frontend. FastF1 download + particle filter is
    network-bound and can take minutes, so this returns a job id immediately
    and the real work happens on a background thread -- never on the request
    thread, which would block every other endpoint until it finished."""
    try:
        round_ = int(body["round"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, "round is required and must be an integer") from exc
    year = int(body.get("year", 2026))
    session = str(body.get("session", "R"))
    drivers = body.get("drivers")
    max_laps = body.get("max_laps")
    particles = int(body.get("particles", 400))

    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"status": "running"}
    threading.Thread(target=_run_analysis_job,
                     args=(job_id, year, round_, session, drivers, max_laps, particles),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/races/analyze/{job_id}")
def analyze_race_status(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.get("/api/race/{rid}/summary")
def summary(rid: str):
    d = _race(rid)
    return {
        **_summary(d),
        "circuit_geometry": d["circuit_geometry"],
        # `.get`, not `[...]`. Every race artefact currently on disk predates
        # `analysis.py` stamping the regulation variant (it writes the block at
        # analysis.py:232), so a hard index raised KeyError and this endpoint
        # returned 500 for ALL FIVE races -- and since the frontend loads a race
        # through `/summary`, nothing in the app could open at all. The real fix
        # is re-analysing the artefacts; until then absence is reported as
        # absence rather than filled in with a guessed variant, which is the one
        # thing a regulation field must never do.
        "regulation": d.get("regulation"),
        "regulation_available": "regulation" in d,
        "refusals": d["refusals"],
        "calibration": d["calibration"],
        "drivers": sorted(d["cars"].keys()),
        # Each battle carries what the precomputed bundle knows about it, so the
        # pairing picker can mark a pair the engine declines instead of letting
        # you choose it and land on an explanation. `null` means "not solved
        # yet", which is a third state and is shown as such.
        "battles": _annotate_battles(rid, d.get("battles", [])),
        "laps": d["laps"][:2000],
    }


def _annotate_battles(rid: str, battles: list[dict]) -> list[dict]:
    idx = read_index()
    out = []
    for b in battles:
        e = idx.get(f"{rid}__{b['car']}__{b['ahead']}")
        # `e is not None` was the old test, which counted a stale entry as
        # solved -- see `bundle_is_current`. A stale entry still carries a
        # usable refusal and situation count from the last build, so those are
        # kept; only the "this is instant" claim is withdrawn.
        solved = e is not None and bundle_is_current(rid, e)
        out.append({**b,
                    "solved": solved,
                    "refusal": (e or {}).get("refusal"),
                    "n_situations": (e or {}).get("n_situations")})
    return out


@app.get("/api/race/{rid}/car/{drv}")
def car(rid: str, drv: str):
    d = _race(rid)
    if drv not in d["cars"]:
        raise HTTPException(404, d["refusals"].get(drv, {"message": "no such car"}))
    return d["cars"][drv]


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
async def decision(rid: str, car: str = Query(...), rival: str = Query(...)):
    """Threshold curve, opportunity quality and the policy-space fan."""
    d = _race(rid)
    if car not in d["cars"] or rival not in d["cars"]:
        raise HTTPException(404, "car not analysed")
    b = _bundle_cached(rid, car, rival)
    if b is None:
        return _building(rid, car, rival)
    if b.get("refusal"):
        raise HTTPException(422, b["refusal"])
    return b["decision"]


@app.get("/api/race/{rid}/p2")
async def p2_decision(rid: str, car: str = Query(...), rival: str = Query(...),
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
    if from_lap is None:
        # The default view is precomputed with everything else for this pair.
        b = _bundle_cached(rid, car, rival)
        if b is None:
            return _building(rid, car, rival)
        p2 = b["p2"]
        if "error" in p2:
            raise HTTPException(422, p2["error"])
        return p2
    # `from_lap` is not bundled -- no view requests it today, so it has no
    # cache. It still has to leave this process: this is an `async def` route
    # now, so a solve called inline here would block the event loop outright,
    # which is worse than the threadpool it used to run on.
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            _BUNDLE_POOL, partial(evaluate_opportunity_decision,
                                  d, car, rival, from_lap=from_lap))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _decision_payload(d: dict, car: str, rival: str) -> dict:
    from xray.decision_service import evaluate_decision_trace_from_payload

    try:
        return evaluate_decision_trace_from_payload(d, car, rival)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _default_horizon_s(d: dict) -> float:
    """The evaluation window has to be long enough to contain a lap completion.

    Position is only published per lap, so `historical_replay` can only say what
    the driver actually did once both cars have crossed the line inside the
    window. At the old 30 s default, `evaluation_fields["laps"]` came back empty
    at every one of Melbourne's 22 decision points (median lap 85.2 s), so
    `actual_action` was None, `matches_recommendation` was None, and the
    follow/disobey walk searched the whole race and never once found a match --
    the feature could not fire on any circuit. 1.75 median laps clears one lap
    boundary from any starting phase; at Melbourne it resolves to True/held.
    """
    times = [r["lap_time"] for r in d.get("laps", []) if r.get("lap_time")]
    if not times:
        return 150.0
    times.sort()
    return round(1.75 * times[len(times) // 2], 1)


# Bumped whenever the SHAPE of a situation row changes. The bundle stamp used to
# track only `artefact_mtime_ns`, i.e. the inputs -- so when the row schema changed
# (P1 `attack` -> canonical P2 `recommendation`) every bundle built by the previous
# code stayed "valid" and would have been served with the recommendation field the
# frontend now reads simply absent: a silently blank column, on 60 of 74 bundles.
# A cache key has to cover the producing code's output contract, not just its input.
#
# 2 -> 3: the position-derived "driver action" and its MATCHED/DIVERGED verdict
# stopped being user-facing and the observed-outcome fields
# (`observed_position_delta`, `observed_outcome`, `counterfactual_status`, ...)
# were added. A schema-2 bundle has none of them, so served against schema-3
# semantics it would render an empty outcome column next to a recommendation --
# `_bundle` compares this number exactly, so such a bundle is rebuilt, not served.
SITUATIONS_SCHEMA = 3


def _situations_rows(d: dict, car: str, rival: str, trace: dict,
                     horizon: float) -> list[dict]:
    """Every causal decision point in one race, each already replayed off-policy.

    The frontend used to build this by firing one `/replay` per point, serially,
    up to 40 of them, with every control disabled for the duration -- which is
    how you lose a live demo. One request instead.

    ONE RECOMMENDATION PER ROW, and it is P2's. Every displayed recommendation
    field is lifted from `rep["p2_recommendation"]`, the canonical solver result.

    THREE SEPARATED CONCEPTS, none of which is allowed to impersonate another:
    the X-RAY recommendation (canonical P2), the observed historical outcome (a
    position delta, `observed_*`), and the driver's action (not observed -- there
    is no public channel for it). The row no longer carries a user-facing
    matched/diverged verdict, because the one it used to carry was derived from
    the position delta and so asserted intent from an outcome.

    They used to. The row's `attack` came from the P1 per-lap trace while the
    verdict came from P2 inside `historical_replay`, and the two only coincided
    when P2 happened to call HOLD. 137 of 852 rows (16.1%) rendered a
    contradiction -- Monaco showed 40 rows reading `HOLD | held | diverged`,
    because the P1 column said HOLD while P2 had called ATTACK. The P1 trace is
    still used here to ENUMERATE decision points, which is all it is good for;
    its own call is carried as `legacy_p1_attack` and is not displayed.

    Nothing here computes a decision. No P2 equation is restated.
    """
    from xray.decision_service import historical_replay

    rows = []
    for r in trace.get("laps", []):
        t = r.get("decision_time_s")
        if t is None:
            continue
        row = {
            "lap": r.get("lap"), "decision_time_s": t, "gap_s": r.get("gap_s"),
            # --- canonical P2 recommendation, all from one solver result -------
            "recommendation": None, "recommended_zone": None,
            "deployment_budget_mj": None, "actual_deployed_mj": None,
            "pass_probability": None, "value_action": None, "value_hold": None,
            "decision_margin": None, "next_best_action": None,
            "pass_model_calibration": None,
            "recommendation_source": "p2/evaluate_opportunity_decision",
            # --- OBSERVED HISTORICAL OUTCOME (what physically happened) -------
            # A position delta, from the public `laps[].position` field. It is
            # not a driver action: pit stops, retirements ahead, penalties,
            # incidents, traffic and safety cars all move it, and an attack that
            # failed moves it not at all. `driver_action_observed` is therefore
            # False on every row -- no public channel carries the driver's choice
            # -- and the counterfactual is unresolved because the race never
            # branched onto P2's recommendation.
            "observed_position_before": None,
            "observed_position_after": None,
            "observed_position_delta": None,
            "observed_outcome": None,
            "observed_outcome_basis": "public laps[].position at cutoff vs at window end",
            "driver_action_observed": False,
            "counterfactual_status": "unresolved_from_historical_telemetry",
            # --- LEGACY, INTERNAL, NEVER DISPLAYED ---------------------------
            # These are the old position-derived "driver action" and the
            # MATCHED/DIVERGED verdict built on it. 852 rows rendered that
            # verdict, and every one was a claim about intent read off a
            # quantity that carries none. Retained only so an older reader does
            # not KeyError and so the bundle index can keep counting resolved
            # windows. No frontend field reads them.
            "matches_recommendation": None,
            "matches_recommendation_is_legacy": True,
            "actual_action": None,
            "actual_action_is_legacy": True,
            "inferred_action_from_position": None,
            "actual_action_basis": "legacy: track_position_change_over_evaluation_window",
            # --- legacy, internal, never displayed ---------------------------
            "legacy_p1_attack": bool(r.get("attack")),
            "legacy_p1_requested_zone": r.get("requested_zone"),
            "replay_error": None,
        }
        try:
            rep = historical_replay(d, car, rival, float(t), horizon_s=horizon)
        except (ValueError, KeyError) as exc:
            # A point the snapshot cannot serve is reported as such, not
            # dropped: a filter over a silently shortened list is a filter that
            # lies about how many situations the race contained.
            row["replay_error"] = f"{type(exc).__name__}: {exc}"
        else:
            p2 = rep.get("p2_recommendation") or {}
            row["recommendation"] = p2.get("decision")
            row["recommended_zone"] = p2.get("zone")
            for k in ("deployment_budget_mj", "actual_deployed_mj",
                      "pass_probability", "value_action", "value_hold",
                      "decision_margin", "next_best_action",
                      "pass_model_calibration"):
                row[k] = p2.get(k)
            for k in ("observed_position_before", "observed_position_after",
                      "observed_position_delta", "observed_outcome",
                      "driver_action_observed", "counterfactual_status"):
                row[k] = rep.get(k)
            row["matches_recommendation"] = rep.get("matches_recommendation")
            row["actual_action"] = rep.get("actual_action")
            row["inferred_action_from_position"] = rep.get("actual_action")
            row["replay_error"] = rep.get("p2_error")
        rows.append(row)
    return rows


def _json_safe(o):
    """Map non-finite floats to null at the serialisation boundary.

    The DP uses `-inf` to mean "this action is not available at any price", and
    196 of them reach one Melbourne-round decision payload as `value_attack`.
    JSON has no infinity, so FastAPI's encoder raises and the endpoint returns
    500 -- which it has always done: `/api/race/2026_r10_R/decision?car=HUL&
    rival=LAW` was a hard failure, so Cockpit and the old Strategy tab rendered
    nothing at all for those pairs. `null` is the right image because it is the
    value the frontend already treats as "not affordable"; the solver keeps its
    `-inf`, which is meaningful arithmetic and is not touched.
    """
    if isinstance(o, float):
        return o if isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


def bundle_path(rid: str, car: str, rival: str) -> Path:
    return DECISIONS / f"{rid}__{car}__{rival}.json"


INDEX = DECISIONS / "_index.json"


def read_index() -> dict:
    """Tiny sidecar: bundle key -> {refusal, n_situations, n_resolved}.

    The summary endpoint needs to know which battles are actually viable so the
    pairing picker can mark the dead ones, and some races open on a battle the
    engine declines. Answering that by reading the bundles themselves would be
    8 x ~600 kB of JSON per summary request, to extract one string each. `rebuild_index` regenerates it from whatever is on disk.
    """
    try:
        return json.loads(INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_index_entry(key: str, entry: dict) -> None:
    idx = read_index()
    idx[key] = entry
    DECISIONS.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, allow_nan=False), encoding="utf-8")
    tmp.replace(tmp.with_suffix(""))


def bundle_is_current(rid: str, stamp: dict) -> bool:
    """Would `_bundle` serve this, or rebuild it?

    One predicate, because there used to be two. `_bundle` required both the
    artefact mtime AND `situations_schema` to match, while the sidecar recorded
    only the mtime -- so after `SITUATIONS_SCHEMA` went 2 -> 3 the pairing
    picker advertised all 89 indexed battles as "solved · N situations" while
    every one of them was in fact a full ~112 s rebuild. The picker was lying
    about the only thing it exists to say. `stamp` is either a bundle or an
    index entry; both carry the two fields this reads.
    """
    try:
        mtime = (RACES / f"{rid}.json").stat().st_mtime_ns
    except OSError:
        return False
    return (stamp.get("artefact_mtime_ns") == mtime
            and stamp.get("situations_schema") == SITUATIONS_SCHEMA)


def index_entry(bundle: dict) -> dict:
    sits = bundle.get("situations") or []
    return {"refusal": bundle.get("refusal"),
            "n_situations": len(sits),
            # Rows whose evaluation window actually has a published position at
            # both ends. This used to count `matches_recommendation is not None`,
            # i.e. rows with a MATCHED/DIVERGED verdict -- a count of rows on
            # which an invalid claim could be made. Same arithmetic, honest
            # subject: the window resolved, not the driver obeyed.
            "n_resolved": sum(1 for x in sits
                              if x.get("observed_position_delta") is not None),
            "artefact_mtime_ns": bundle.get("artefact_mtime_ns"),
            # Without this the sidecar cannot answer `bundle_is_current`, which
            # is the whole point of the entry.
            "situations_schema": bundle.get("situations_schema")}


def rebuild_index() -> dict:
    idx = {}
    for p in sorted(DECISIONS.glob("*__*__*.json")):
        try:
            idx[p.stem] = index_entry(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    DECISIONS.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, allow_nan=False), encoding="utf-8")
    tmp.replace(INDEX)
    return idx


def build_bundle(rid: str, car: str, rival: str, d: dict | None = None) -> dict:
    """Everything the Cockpit and Situations tabs need for one pair, in one object.

    `evaluate_decision_trace_from_payload` costs 54 s for a 22-lap pair -- 6.2M
    Python-level `vehicle.step` calls inside `calibrate_zone`. It sits on the
    path of every driver swap, which is what made choosing drivers feel broken
    rather than merely slow. It is also fully determined by the race artefact,
    so it is computed once and written to `out/decisions/`; the API only ever
    recomputes for a pair nobody has asked for before, and writes that one
    through too. `scripts/09.precompute_decisions.py` does the batch.
    """
    from xray.decision_service import evaluate_opportunity_decision

    d = d if d is not None else _race(rid)
    horizon = _default_horizon_s(d)

    # A pair the engine legitimately declines is CARRIED, not raised past the
    # cache: a refusal is a correct output and has to be reported as one -- but
    # it also has to be cached, or every visit to that pair pays the full solve
    # to be told no. The refusal this used to describe ("no causal trace samples
    # for lap 1 at s <= 0.0 m", all 8 Zandvoort battles) is gone: that was one
    # opportunity at the start/finish line aborting a whole race, and
    # `decision_service` now skips the opportunity instead.
    refusal = None
    try:
        trace = _decision_payload(d, car, rival)
    except HTTPException as exc:
        trace, refusal = None, str(exc.detail)
    if refusal is None:
        try:
            p2 = evaluate_opportunity_decision(d, car, rival)
        except (KeyError, ValueError) as exc:
            p2 = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        p2 = {"error": refusal}
    return _json_safe({
        "race": rid, "car": car, "rival": rival, "horizon_s": horizon,
        # Stamped so a re-analysed race invalidates its bundles instead of
        # serving a decision trace built from telemetry that no longer exists.
        "artefact_mtime_ns": (RACES / f"{rid}.json").stat().st_mtime_ns,
        "situations_schema": SITUATIONS_SCHEMA,
        "refusal": refusal,
        "decision": trace,
        "p2": p2,
        "situations": ([] if trace is None
                       else _situations_rows(d, car, rival, trace, horizon)),
    })


def _bundle_cached(rid: str, car: str, rival: str) -> dict | None:
    """The bundle on disk, or None if there isn't a usable one.

    `None` means "a solve is needed", never "an error". Same validity test the
    pairing picker uses, so the two cannot disagree about what is instant.
    """
    p = bundle_path(rid, car, rival)
    if not p.exists():
        return None
    try:
        cached = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None       # a truncated bundle is rebuilt, never served
    return _json_safe(cached) if bundle_is_current(rid, cached) else None


async def _solve_bundle(rid: str, car: str, rival: str) -> dict:
    """One solve, off-process, written through by the parent.

    `build_bundle` is a tight pure-Python loop -- measured at 112 s for one
    22-lap pair on the development machine, not the 54 s its docstring quotes.
    Run in this process, even on a worker thread, it holds the GIL almost
    continuously and starves every other request: `GET /api/race/{rid}`, the
    call the RacePicker makes to switch track, sat behind it, which is what
    "can't change track" was. A separate process has its own GIL. The file I/O
    stays here so the atomic write and the sidecar update happen exactly once.
    """
    loop = asyncio.get_running_loop()
    d = _race(rid)
    out = await loop.run_in_executor(_BUNDLE_POOL, build_bundle, rid, car, rival, d)
    p = bundle_path(rid, car, rival)
    DECISIONS.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    tmp.replace(p)        # atomic: a reader never sees a half-written bundle
    write_index_entry(p.stem, index_entry(out))
    return out


def _job_public(job: dict) -> dict:
    """The client-facing view.

    `task` is an asyncio object and is not serialisable at all; `_t0` is a
    `time.monotonic()` reading, which is meaningless outside this process and
    was briefly served to the browser as if it were a timestamp. Anything
    private stays private: underscore-prefixed keys are internal by convention
    and dropped by that rule rather than by a growing deny-list.
    """
    return {k: v for k, v in job.items()
            if k != "task" and not k.startswith("_")}


def _bundle_job(rid: str, car: str, rival: str) -> dict:
    """The running solve for this pair, started if there isn't one.

    One job per pair, not per request: Cockpit opens `/decision` and `/p2`
    simultaneously and Situations adds `/situations`, so three routes miss on
    the same pair within a few milliseconds of each other. Without this they
    would each start their own 112 s solve of the identical bundle.
    """
    key = bundle_path(rid, car, rival).stem
    job = _BUNDLE_JOBS.get(key)
    if job is not None:
        if job["status"] == "building":
            job["elapsed_s"] = round(time.monotonic() - job["_t0"], 1)
            return job
        # A finished record is not reusable: "ready" means the caller should
        # have found it on disk, and "error" is reported once and then dropped
        # so a transient failure does not pin the pair as broken for the rest
        # of the process's life.
        del _BUNDLE_JOBS[key]
        if job["status"] == "error":
            return job

    job = {"job_id": uuid.uuid4().hex, "status": "building", "elapsed_s": 0.0,
           "race": rid, "car": car, "rival": rival, "message": None,
           "_t0": time.monotonic()}
    task = asyncio.ensure_future(_solve_bundle(rid, car, rival))

    def _finished(t: asyncio.Future, job: dict = job) -> None:
        job["elapsed_s"] = round(time.monotonic() - job["_t0"], 1)
        if t.cancelled():
            job["status"], job["message"] = "error", "cancelled"
            return
        exc = t.exception()
        if exc is None:
            job["status"] = "ready"
        else:
            job["status"] = "error"
            job["message"] = f"{type(exc).__name__}: {exc}"

    task.add_done_callback(_finished)
    job["task"] = task
    _BUNDLE_JOBS[key] = job
    return job


def _building(rid: str, car: str, rival: str) -> JSONResponse:
    """202 with the job state, rather than blocking the request for two minutes.

    202 and not 200: the body is a progress report, not the payload the caller
    asked for, and `lib/api.ts` branches on the status code. `r.ok` is true for
    both, so a 200 here would be parsed as a decision payload and render as an
    empty tab.
    """
    return JSONResponse(status_code=202,
                        content=_job_public(_bundle_job(rid, car, rival)))


@app.get("/api/race/{rid}/situations")
async def situations(rid: str, car: str = Query(...), rival: str = Query(...)):
    d = _race(rid)
    if car not in d["cars"] or rival not in d["cars"]:
        raise HTTPException(404, "car not analysed")
    b = _bundle_cached(rid, car, rival)
    if b is None:
        return _building(rid, car, rival)
    return {"race": rid, "car": car, "rival": rival,
            "horizon_s": b["horizon_s"], "situations": b["situations"],
            "refusal": b.get("refusal")}


@app.get("/api/race/{rid}/bundle")
async def bundle_status(rid: str, car: str = Query(...), rival: str = Query(...)):
    """Is this pair's decision trace ready, and if not, how long has it been?

    Polled by the frontend so a pair nobody has solved before reports progress
    instead of holding a tab in an indefinite spinner. Asking starts the solve,
    which is deliberate: this is the endpoint a view calls when it wants the
    bundle, not a passive probe.
    """
    d = _race(rid)
    if car not in d["cars"] or rival not in d["cars"]:
        raise HTTPException(404, "car not analysed")
    if _bundle_cached(rid, car, rival) is not None:
        return {"status": "ready", "race": rid, "car": car, "rival": rival}
    return _job_public(_bundle_job(rid, car, rival))


@lru_cache(maxsize=4)
def _rdd_rows_cached(season: int, stamp: str) -> list[dict]:
    """Pooled RDD rows across every analysed race of a season.

    This used to run inline in the route: glob and `json.loads` every artefact in
    `out/races/` -- measured at ten of them, ~130 MB and 2.9 s of GIL-bound
    parsing (5 are on disk now, so re-measure before quoting the MB) -- on every
    single request, to produce
    a few thousand rows totalling well under a megabyte. These are `def` routes
    on the threadpool, so a handful of concurrent RDD requests stalled every
    other endpoint. Keyed on mtime like `_all_races_cached`, so a freshly
    analysed round still invalidates it.
    """
    rows = []
    for p in sorted(RACES.glob("*.json")):
        d = json.loads(p.read_text())
        if d["year"] != season:
            continue
        for r in d["rdd"]["rows"]:
            rows.append({**r, "race": d["id"], "circuit": d["circuit"]})
    return rows


def _rdd_rows(season: int) -> list[dict]:
    stamp = ",".join(f"{p.name}:{p.stat().st_mtime_ns}"
                     for p in sorted(RACES.glob("*.json")))
    return _rdd_rows_cached(season, stamp)


@app.get("/api/rdd")
def rdd(cutoff: float = Query(1.0, ge=0.2, le=3.0),
        season: int = 2026, bandwidth: float = Query(0.6, ge=0.1, le=2.0)):
    """Regression discontinuity at the Manual Override eligibility boundary.

    The fit is live -- both sides refit at whatever cutoff is asked for, with
    the falsification scan across cutoff positions beside it -- but the pooled
    rows it fits are cached, because re-reading the race artefacts was three
    orders of magnitude more expensive than the regression.
    """
    rows = _rdd_rows(season)
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


@app.get("/api/ablation")
def ablation():
    """Stage 1's measured sample-rate ablation, from `out/ablation.json`.

    Served rather than retyped: the same eight (Hz, MAPE) pairs were literals in
    `Method.tsx`, so the chart and the artefact could disagree without anything
    failing. These are SIMULATOR numbers from the Stage 1 estimator, not the
    real-data stack, and the view labels them that way.
    """
    p = OUT / "ablation.json"
    if not p.exists():
        raise HTTPException(404, "out/ablation.json not generated")
    return json.loads(p.read_text())


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
                       horizon: float | None = None):
    # Horizon defaults per-race, not to 30 s -- see `_default_horizon_s`.
    from xray.decision_service import historical_replay
    d = _race(rid)
    h = float(horizon) if horizon else _default_horizon_s(d)
    return historical_replay(d, car, rival, cutoff, horizon_s=h)


# ------------------------------------------------------------------ LLM judge
# Read-only. This process never calls a language model, on this path or any
# other: `scripts/16` exists because a 54 s solve on the driver-swap path made
# choosing a driver look broken, and a network-bound LLM call on a request path
# is the same mistake with worse tails. Verdicts are produced offline by
# `scripts/17.judge_situations.py` and served from disk here.
#
# Only `xray.judge.store` is imported, and it is standard-library-only. If the
# SDK ever leaked into this import chain, serving a request would start
# requiring `google-genai` to be installed --
# `tests/test_judge_api.py::test_the_api_never_imports_the_llm_client` asserts
# it has not.


@app.get("/api/race/{rid}/judge")
def judge_endpoint(rid: str, car: str = Query(...), rival: str = Query(...)):
    """Cached LLM-judge verdicts for one pair, keyed by decision time.

    `available: false` is a normal answer, not an error. `out/` is gitignored,
    so a fresh clone has no verdicts at all, and a 404 here would render the
    Situations tab as broken for a state that just means nobody has run the
    judge yet.
    """
    from xray.judge import store as judge_store
    return _json_safe(judge_store.read_pair(rid, car, rival))


@app.get("/api/judge/report")
def judge_report_endpoint():
    from xray.judge.paths import LATEST_REPORT
    try:
        return _json_safe(json.loads(LATEST_REPORT.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {"available": False,
                "reason": "no judge report yet -- run scripts/17.judge_situations.py"}


dist = ROOT / "app" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="app")
