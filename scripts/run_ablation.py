#!/usr/bin/env python3
"""Sweep the telemetry sample rate and plot accuracy against it.

This is the honest answer to "does this survive real data rates". The spec's
target rate, 3.7 Hz, is what a public feed actually publishes.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import base_parser, config_from
from xray.estimator import EstimatorError, estimate
from xray.metrics import score_estimate
from xray.observe import observe
from xray.render.style import AMBER, BG, CARD, DIM, GRAY, GREEN, RED, WHITE, apply_style
from xray.sim import FOLLOWER, LEADER, run_sim

RATES = [100.0, 50.0, 20.0, 10.0, 5.0, 3.7, 2.0, 1.0]


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--rates", type=float, nargs="+", default=RATES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--cars", nargs="+", default=[LEADER])
    args = ap.parse_args()
    cfg = config_from(args)
    cda_true = cfg["vehicle"]["cda_straight"]
    npart = cfg["estimator"]["n_particles"]

    results = {r: {"mape": [], "cov": [], "cda": []} for r in args.rates}
    failures = {}
    for seed in args.seeds:
        gt = run_sim(cfg, seed=seed)
        for car in args.cars:
            for rate in args.rates:
                obs = observe(gt, car, rate_hz=rate,
                              speed_noise_ms=cfg["observe"]["speed_noise_ms"],
                              seed=seed + 1)
                try:
                    bel = estimate(obs, gt.track, n_particles=npart, seed=seed + 2)
                except EstimatorError as exc:
                    failures.setdefault(rate, []).append(f"{car}/{seed}: {exc}")
                    continue
                sc = score_estimate(gt, car, obs, bel, cda_true)
                results[rate]["mape"].append(sc.deployed_mape)
                results[rate]["cov"].append(sc.usable_coverage)
                results[rate]["cda"].append(abs(sc.cda_error_pct))
                print(f"  seed {seed} {car:8s} {rate:6.1f} Hz -> MAPE {sc.deployed_mape:5.1f}%  "
                      f"coverage {sc.usable_coverage:.3f}  |CdA err| {abs(sc.cda_error_pct):.2f}%")

    rates = [r for r in args.rates if results[r]["mape"]]
    mape = np.array([np.mean(results[r]["mape"]) for r in rates])
    mape_sd = np.array([np.std(results[r]["mape"]) for r in rates])
    cov = np.array([np.mean(results[r]["cov"]) for r in rates])
    cda = np.array([np.mean(results[r]["cda"]) for r in rates])

    print("\n rate Hz | dep MAPE | band coverage | |CdA err|")
    for r, m, c, d in zip(rates, mape, cov, cda):
        print(f"  {r:6.1f} | {m:7.1f}% | {c:12.3f}  | {d:7.2f}%")
    for rate, msgs in failures.items():
        print(f"  {rate:6.1f} | estimator refused: {msgs[0]}")

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_facecolor(CARD)
    ax.errorbar(rates, mape, yerr=mape_sd, color=AMBER, marker="o", lw=2.2,
                capsize=4, elinewidth=1.2, label="per-lap deployed energy error")
    ax.axhline(8, color=GREEN, ls="--", lw=1.2)
    ax.axhline(15, color=RED, ls="--", lw=1.2)
    ax.text(1.05, 8.4, "8% target at 100 Hz", color=GREEN, fontsize=10)
    ax.text(1.05, 15.4, "15% target at 3.7 Hz", color=RED, fontsize=10)
    ax.axvline(3.7, color=WHITE, ls=":", lw=1.4, alpha=0.7)
    ax.text(3.9, max(mape) * 0.92, "3.7 Hz\npublic feed", color=WHITE, fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(rates)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("telemetry sample rate (Hz)")
    ax.set_ylabel("per-lap deployed energy MAPE (%)")
    ax.set_title("X-RAY accuracy versus telemetry sample rate",
                 color=WHITE, fontsize=15, pad=14, loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=GRAY, loc="upper right")
    fig.text(0.5, 0.015, f"{len(args.seeds)} seeds x {len(args.cars)} cars per rate; "
                         f"bars are 1 s.d. across runs", color=DIM, fontsize=9, ha="center")
    out = args.out or "out/ablation_mape_vs_rate.png"
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=BG)
    print(f"\nwrote {out}")

    with open("out/ablation.json", "w") as fh:
        json.dump({"rates": rates, "mape": mape.tolist(), "coverage": cov.tolist(),
                   "cda_abs_err_pct": cda.tolist(),
                   "failures": {str(k): v for k, v in failures.items()}}, fh, indent=2)


if __name__ == "__main__":
    main()
