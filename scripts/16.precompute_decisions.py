#!/usr/bin/env python3
"""Precompute the per-pair decision bundle the app serves, into `out/decisions/`.

Why this exists: `evaluate_decision_trace_from_payload` costs ~54 s for a
22-lap pair (6.2M Python-level `vehicle.step` calls inside `calibrate_zone`),
and it sits on the path of *every driver swap* in the frontend -- Cockpit and
Situations both open on it. Computed on request, choosing a driver looked
broken rather than merely slow. The trace is fully determined by the race
artefact, so it is computed once here and written to disk; the API serves the
file and only falls back to computing for a pair nobody has asked for before
(and writes that one through, so it is paid at most once ever).

Default target is each race's `battles` -- the pairs that actually raced each
other, which is what the frontend's pairing picker leads with. `--all-pairs`
does the full ordered cross product, which is ~380 pairs per race and hours of
CPU; only worth it if the demo needs arbitrary pairs to be instant.

    python scripts/16.precompute_decisions.py                 # every race's battles
    python scripts/16.precompute_decisions.py --race 2026_r1_R
    python scripts/16.precompute_decisions.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.api.main import (DECISIONS, RACES, build_bundle,  # noqa: E402
                                 SITUATIONS_SCHEMA, bundle_path, rebuild_index)


def pairs_for(d: dict, all_pairs: bool) -> list[tuple[str, str]]:
    cars = sorted(d["cars"])
    if all_pairs:
        return [(a, b) for a in cars for b in cars if a != b]
    # Both orderings of each battle: the picker lets you attack from either
    # side, and the decision trace is not symmetric in (car, rival).
    out = []
    for b in d.get("battles", []):
        for pair in ((b["car"], b["ahead"]), (b["ahead"], b["car"])):
            if pair[0] in d["cars"] and pair[1] in d["cars"] and pair not in out:
                out.append(pair)
    return out


def build_one(rid: str, car: str, rival: str, force: bool) -> tuple[str, float, str]:
    p = bundle_path(rid, car, rival)
    mtime = (RACES / f"{rid}.json").stat().st_mtime_ns
    if not force and p.exists():
        try:
            cached = json.loads(p.read_text(encoding="utf-8"))
            if (cached.get("artefact_mtime_ns") == mtime
                    and cached.get("situations_schema") == SITUATIONS_SCHEMA):
                return (f"{rid} {car}->{rival}", 0.0, "cached")
        except (ValueError, OSError):
            pass
    t0 = time.time()
    try:
        out = build_bundle(rid, car, rival)
    except Exception as exc:  # noqa: BLE001 -- one bad pair must not kill the batch
        return (f"{rid} {car}->{rival}", time.time() - t0, f"FAILED {type(exc).__name__}: {exc}")
    DECISIONS.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(p)
    n_sit = len(out["situations"])
    # "Resolved" means the evaluation window had a published classification at BOTH
    # ends, so an observed position delta exists. It used to count rows where
    # `matches_recommendation` was non-null -- the driver-action verdict that has
    # since been retired as invalid (position change also moves for pit stops,
    # retirements ahead, penalties, incidents and safety cars). Counting the
    # retired field still "worked" because it is retained internally, so the log
    # would have kept reporting a quantity no view shows any more.
    n_res = sum(1 for s in out["situations"]
                if s.get("observed_position_delta") is not None)
    return (f"{rid} {car}->{rival}", time.time() - t0,
            f"{n_sit} situations, {n_res} with an observed position delta")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", action="append", default=None,
                    help="race id, repeatable; default is every race on disk")
    ap.add_argument("--all-pairs", action="store_true",
                    help="every ordered pair, not just the observed battles")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rebuild even if current")
    ap.add_argument("--index-only", action="store_true",
                    help="just regenerate out/decisions/_index.json from what "
                         "is already on disk, without solving anything")
    a = ap.parse_args()

    if a.index_only:
        idx = rebuild_index()
        refused = sum(1 for v in idx.values() if v["refusal"])
        print(f"indexed {len(idx)} bundles ({refused} refused)")
        return 0

    ids = a.race or [p.stem for p in sorted(RACES.glob("*.json"))]
    jobs: list[tuple[str, str, str]] = []
    for rid in ids:
        p = RACES / f"{rid}.json"
        if not p.exists():
            print(f"  ! no artefact for {rid}, skipped")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        jobs += [(rid, c, r) for c, r in pairs_for(d, a.all_pairs)]

    print(f"{len(jobs)} pair bundles over {len(ids)} races, {a.workers} workers")
    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, *j, a.force): j for j in jobs}
        for f in as_completed(futs):
            name, dt, note = f.result()
            done += 1
            failed += note.startswith("FAILED")
            print(f"  [{done}/{len(jobs)}] {name:<28} {dt:6.1f}s  {note}")
    print(f"\nwrote to {DECISIONS} in {time.time() - t0:.0f}s "
          f"({done - failed} ok, {failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
