#!/usr/bin/env python3
"""Run the Stage 1 core over a real session (Phase B/C end-to-end)."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from xray.data.circuits import RealTrack, geometry_from_session
from xray.data.ingest import ingest_session, write_parquet
from xray.realfit import (build_kin, common_mode, deployment_trace,
                          fit_nuisance_real, pool_field)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=10)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--session", default="R")
    ap.add_argument("--drivers", nargs="+", default=None)
    ap.add_argument("--max-laps", type=int, default=None)
    ap.add_argument("--mass", type=float, default=790.0)
    args = ap.parse_args()

    t0 = time.time()
    import fastf1
    fastf1.Cache.enable_cache("xray/data/cache")
    ses = fastf1.get_session(args.year, args.round, args.session)
    ses.load(telemetry=True, laps=True, weather=True)
    geo = geometry_from_session(ses)
    track = RealTrack(geo)
    print(f"\n{ses.event['EventName']} — {geo.name}: {geo.length:.0f} m, "
          f"elevation {'yes' if geo.has_elevation else 'NO'}, "
          f"{len(track._corner_seg)} corners, zones "
          f"{[(z['name'], round(z['length'])) for z in track.zones]}")

    sd = ingest_session(args.year, args.round, args.session,
                        drivers=args.drivers, max_laps=args.max_laps, verbose=False)
    rho = sd.weather.get("rho", 1.20)
    usable = sum(1 for q in sd.quality if q.usable)
    print(f"ingest: {len(sd.frames)} cars, {len(sd.quality)} laps, "
          f"{usable} usable ({usable/max(len(sd.quality),1)*100:.0f}%), rho={rho:.4f} kg/m^3")

    fits, kins, rows = {}, {}, []
    for drv, df in sd.frames.items():
        d = df[df["usable"]] if "usable" in df else df
        if len(d) < 500:
            continue
        kin = build_kin(d, track, args.mass, rho)
        try:
            fit = fit_nuisance_real(kin, rho)
        except ValueError as exc:
            print(f"  {drv}: REFUSED — {exc}")
            continue
        fits[drv], kins[drv] = fit, kin
        rows.append((drv, fit))

    print(f"\n{'car':5s} {'CdA':>6s} {'identified set':>16s} {'ident':>6s} "
          f"{'binding':>8s} {'coast':>6s} {'wind':>6s} {'resid kW':>9s}")
    for drv, f in sorted(rows, key=lambda r: -r[1].identifiability):
        print(f"{drv:5s} {f.cda_hat:6.3f} [{f.cda_lo:6.3f},{f.cda_hi:6.3f}] "
              f"{f.identifiability:6.2f} {f.n_binding:8d} {f.n_coast:6d} "
              f"{f.v_wind_hat:+6.1f} {f.residual_rms/1e3:9.0f}")

    cda = np.array([f.cda_hat for _, f in rows])
    print(f"\nfield CdA: median {np.median(cda):.3f}, spread {cda.min():.3f}-{cda.max():.3f}, "
          f"iqr {np.percentile(cda,75)-np.percentile(cda,25):.3f}")
    ident = np.array([f.identifiability for _, f in rows])
    print(f"identifiability: median {np.median(ident):.2f}  "
          f"({int((ident>=0.25).sum())}/{len(ident)} cars usable)")

    pf = pool_field(fits)
    if pf.get("available"):
        print(f"field pooling: {pf['n_cars_pooled']} identifiable cars -> CdA "
              f"{pf['cda_pooled']:.3f} [{pf['cda_lo']:.3f}, {pf['cda_hi']:.3f}], "
              f"field identifiability {pf['field_identifiability']:.2f}")
    else:
        print(f"field pooling: unavailable — {pf['reason']}")

    cm = common_mode(fits, kins)
    print(f"cross-car common mode: {cm}" if not cm.get("available")
          else f"cross-car common mode: {cm['n_cars']} cars, "
               f"shared rms {cm['common_mode_rms']/1e3:.0f} kW, "
               f"per-car residual {cm['per_car_rms']/1e3:.0f} kW, "
               f"variance explained {cm['variance_explained']*100:.0f}%")

    if rows:
        drv, f = rows[0]
        tr = deployment_trace(kins[drv], f)
        d = tr["deploy"]
        laps = kins[drv].lap
        dtv = kins[drv].dt
        per_lap = [np.nansum(d[laps == L] * dtv[laps == L]) for L in np.unique(laps)]
        per_lap = [x for x in per_lap if np.isfinite(x) and x > 0]
        h = tr["harvest"]
        per_lap_h = [np.nansum(h[laps == L] * dtv[laps == L]) for L in np.unique(laps)]
        per_lap_h = [x for x in per_lap_h if np.isfinite(x) and x > 0]
        print(f"\n{drv} deployment: peak {np.nanmax(d)/1e3:.0f} kW, "
              f"mean-when-on {np.nanmean(d[d>1e4])/1e3:.0f} kW, "
              f"per-lap {np.median(per_lap)/1e6:.2f} MJ deployed / "
              f"{np.median(per_lap_h)/1e6:.2f} MJ recovered "
              f"(regulation: 4.0 MJ store, 7.0 MJ harvest/lap)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
