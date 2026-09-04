#!/usr/bin/env python3
"""Precompute one race into a single JSON for the app."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xray.analysis import analyse, write

ap = argparse.ArgumentParser()
ap.add_argument("--round", type=int, required=True)
ap.add_argument("--year", type=int, default=2026)
ap.add_argument("--session", default="R")
ap.add_argument("--max-laps", type=int, default=None)
ap.add_argument("--drivers", nargs="+", default=None)
ap.add_argument("--particles", type=int, default=400)
a = ap.parse_args()
t0 = time.time()
p = analyse(a.year, a.round, a.session, drivers=a.drivers, max_laps=a.max_laps,
            n_particles=a.particles)
path = write(p)
mb = path.stat().st_size / 1e6
print(f"\n{p['event']}: {len(p['cars'])} cars analysed, {len(p['refusals'])} refused")
print(f"  telemetry {p['telemetry']['median_hz']} Hz, "
      f"{p['telemetry']['laps_usable']}/{p['telemetry']['laps_total']} laps usable")
print(f"  pooled CdA: {p['calibration']['pooled']}")
print(f"  RDD rows: {p['rdd']['n']}")
print(f"wrote {path} ({mb:.1f} MB) in {time.time()-t0:.0f}s")
