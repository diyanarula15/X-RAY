#!/usr/bin/env python3
"""Blind versus X-RAY, evaluated over seeded races.

Two levels of fidelity:
  --mode fast  (default) lap-level Monte Carlo over the calibrated zone models
  --mode full  re-runs the full 200 Hz two-car simulation per seed, estimates
               the rival's energy from its speed trace, and replays both
               decision rules against that race's real opportunity sequence
"""
from __future__ import annotations

import numpy as np

from _common import base_parser, config_from
from xray.decision import (blind_chooser, build_model, compare_policies,
                           policy_posterior, robustness, solve, xray_chooser,
                           simulate_stint)
from xray.estimator import estimate
from xray.observe import observe
from xray.sim import FOLLOWER, LEADER, run_sim
from xray.vehicle import VehicleParams


def _model_for(cfg, gt, belief, ours):
    params = VehicleParams.from_config(cfg)
    recharge = float(np.mean(belief.harvested_lap[belief.harvested_lap > 0]))
    spend = float(np.mean(belief.deployed_lap[belief.deployed_lap > 0]))
    own = float(np.mean(gt.cars[ours].deployed_lap)) * 0.55
    return build_model(gt.track, params, n_laps=gt.n_laps, recharge_per_lap=recharge,
                       rival_spend_per_lap=spend, own_spend_per_lap=own)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--races", type=int, default=50)
    args = ap.parse_args()
    cfg = config_from(args)

    if args.mode == "fast":
        gt = run_sim(cfg, seed=args.seed)
        obs = observe(gt, LEADER, rate_hz=cfg["observe"]["rate_hz"], seed=args.seed + 1)
        bel = estimate(obs, gt.track, n_particles=cfg["estimator"]["n_particles"],
                       seed=args.seed + 2)
        model = _model_for(cfg, gt, bel, FOLLOWER)
        sol = solve(model)
        believed = np.array([float(bel.usable_mean[obs.lap == L].min())
                             if (obs.lap == L).any() else 0.0
                             for L in range(gt.n_laps)])
        sigma = float(np.mean(bel.usable_p90 - bel.usable_p10) / 2)
        res = compare_policies(model, sol, believed, sigma, n_races=args.races,
                               n_laps=gt.n_laps, e_own0=float(gt.cars[FOLLOWER].E[0]),
                               e_riv0=believed[0], seed=args.seed)
        print(f"lap-level Monte Carlo, {args.races} seeded stints")
    else:
        xr, bl = [], []
        for i in range(args.races):
            seed = args.seed + i
            gt = run_sim(cfg, seed=seed)
            obs = observe(gt, LEADER, rate_hz=cfg["observe"]["rate_hz"], seed=seed + 1)
            bel = estimate(obs, gt.track, n_particles=cfg["estimator"]["n_particles"],
                           seed=seed + 2)
            model = _model_for(cfg, gt, bel, FOLLOWER)
            sol = solve(model)
            believed = np.array([float(bel.usable_mean[obs.lap == L].min())
                                 if (obs.lap == L).any() else 0.0
                                 for L in range(gt.n_laps)])
            e_own0 = float(gt.cars[FOLLOWER].E[0])
            chooser = xray_chooser(sol, believed, 0.0, np.random.default_rng(seed))
            xr.append(simulate_stint(model, chooser, np.random.default_rng(seed),
                                     gt.n_laps, e_own0, believed[0])["passed"])
            bl.append(simulate_stint(model, blind_chooser(model),
                                     np.random.default_rng(seed), gt.n_laps,
                                     e_own0, believed[0])["passed"])
            print(f"  seed {seed}: xray {xr[-1]}  blind {bl[-1]}", flush=True)
        xr, bl = np.array(xr, float), np.array(bl, float)
        d = xr - bl
        se = d.std(ddof=1) / np.sqrt(len(d))
        res = {"xray_pass_rate": xr.mean(), "blind_pass_rate": bl.mean(),
               "mean_gain": d.mean(),
               "ci95": (d.mean() - 1.96 * se, d.mean() + 1.96 * se),
               "n_races": args.races}
        print(f"full-fidelity: one 200 Hz two-car simulation and one estimator "
              f"run per race, {args.races} races")

    print(f"  X-RAY pass rate : {res['xray_pass_rate']:.2f}")
    print(f"  blind pass rate : {res['blind_pass_rate']:.2f}")
    print(f"  positions gained: {res['mean_gain']:+.3f}  "
          f"95% CI [{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}]")
    print(f"  {'PASS' if res['ci95'][0] > 0 else 'FAIL'}: "
          f"lower CI bound {'>' if res['ci95'][0] > 0 else '<='} 0")


if __name__ == "__main__":
    main()
