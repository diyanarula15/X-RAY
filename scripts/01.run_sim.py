#!/usr/bin/env python3
"""Run the two-car simulation and report the ground truth."""
from __future__ import annotations

import numpy as np

from _common import base_parser, config_from
from xray.constants import E_STORE_MAX
from xray.sim import FOLLOWER, LEADER, run_sim


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--save", action="store_true", help="write out/ground_truth.npz")
    args = ap.parse_args()
    cfg = config_from(args)
    gt = run_sim(cfg, seed=args.seed)

    print(f"Circuit {gt.track.name}: {gt.track.length:.0f} m, "
          f"{len(gt.track.corners)} corners, {len(gt.track.zones)} overtake zones")
    print(f"{gt.n_laps} laps, dt={gt.dt} s, seed={gt.seed}, {len(gt.t)} steps, "
          f"{gt.t[-1]:.1f} s of racing\n")
    for car in (LEADER, FOLLOWER):
        tr = gt.cars[car]
        bal = (np.diff(tr.e_lap_open)[:gt.n_laps - 1]
               - (tr.harvested_lap - tr.deployed_lap)[:gt.n_laps - 1]
               - tr.mom_lap[1:gt.n_laps])
        print(f"{car} ({gt.policies[car]}):")
        print(f"  lap times   {np.round(tr.lap_time, 2)}")
        print(f"  top speed   {tr.v.max() * 3.6:.1f} km/h")
        print(f"  deployed MJ {np.round(tr.deployed_lap / 1e6, 2)}")
        print(f"  harvested MJ{np.round(tr.harvested_lap / 1e6, 2)}")
        print(f"  MOM MJ      {np.round(tr.mom_lap / 1e6, 2)}")
        print(f"  store MJ    {tr.E.min() / 1e6:.2f} .. {tr.E.max() / 1e6:.2f} "
              f"(cap {E_STORE_MAX / 1e6:.1f})")
        print(f"  energy balance closes to {np.abs(bal).max():.3f} J\n")

    ot = [e for e in gt.events if e.kind == "overtake"]
    mom = [e for e in gt.events if e.kind == "mom_grant"]
    print(f"gap: {gt.gap_s.min():.2f}..{gt.gap_s.max():.2f} s (mean {gt.gap_s.mean():.2f})")
    print(f"{len(mom)} Manual Override grants, {len(ot)} overtake attempts, "
          f"{sum(e.detail['success'] for e in ot)} completed")
    for e in ot:
        if e.detail["success"]:
            print(f"  lap {e.lap} t={e.t:.1f}s  {e.car} passes in Zone {e.detail['zone']} "
                  f"(dv {e.detail['delta_v']:.2f} m/s, gap {e.detail['gap_s']:.2f} s, "
                  f"p={e.detail['p']:.3f})")
    print(f"final order: {'LEADER' if gt.order[-1] == 0 else 'FOLLOWER'} ahead")

    if args.save:
        out = args.out or "out/ground_truth.npz"
        np.savez_compressed(out, t=gt.t, gap=gt.gap_s,
                            **{f"{c}_{k}": getattr(gt.cars[c], k)
                               for c in (LEADER, FOLLOWER)
                               for k in ("s", "v", "E", "P_mguk")})
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
