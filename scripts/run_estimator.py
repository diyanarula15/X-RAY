#!/usr/bin/env python3
"""Simulate, blindfold, estimate, and score."""
from __future__ import annotations

import json

import numpy as np

from _common import base_parser, config_from
from xray.estimator import estimate
from xray.metrics import aggregate, band_width_by_regime, format_table, score_estimate
from xray.observe import observe
from xray.sim import FOLLOWER, LEADER, run_sim


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--rate", type=float, nargs="+", default=[100.0, 3.7])
    ap.add_argument("--car", nargs="+", default=[LEADER, FOLLOWER])
    ap.add_argument("--particles", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from(args)
    npart = args.particles or cfg["estimator"]["n_particles"]
    gt = run_sim(cfg, seed=args.seed)
    cda_true = cfg["vehicle"]["cda_straight"]

    rows = []
    for car in args.car:
        for rate in args.rate:
            obs = observe(gt, car, rate_hz=rate,
                          speed_noise_ms=cfg["observe"]["speed_noise_ms"],
                          seed=args.seed + 1)
            bel = estimate(obs, gt.track, n_particles=npart, seed=args.seed + 2)
            rows.append(score_estimate(gt, car, obs, bel, cda_true))
            print(f"\n{car} @ {rate} Hz")
            print(f"  Stage A: CdA {bel.nuisance.cda_hat:.4f} +- {bel.nuisance.cda_sigma:.4f} "
                  f"(true {cda_true:.3f}), wind {bel.nuisance.v_wind_hat:+.2f} m/s, "
                  f"{bel.nuisance.n_samples} samples in the calibration band")
            print(f"  Stage C: {bel.dry_events.sum()} deployment cut-outs seen, "
                  f"inferred driver buffer {bel.reserve_mean / 1e6:.2f} MJ, "
                  f"band width sigma {bel.deploy_scale_sigma:.3f}")
            print(f"  band width by regime (MJ): "
                  f"{ {k: round(v, 3) for k, v in band_width_by_regime(gt, car, obs, bel).items()} }")

    print("\n" + format_table(rows))
    print("\n" + json.dumps(aggregate(rows), indent=2))


if __name__ == "__main__":
    main()
