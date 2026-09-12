#!/usr/bin/env python3
"""Closed-loop P2 strategy evaluation in the Stage 1 simulator (P2 step 9).

Orchestration only: every policy, metric and world lives in `xray.closedloop`.
Historical replay cannot answer the counterfactual, so the strategy comparison
happens here, in paired synthetic worlds under known ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xray.closedloop import (WORLD_VARIATION, evaluate_worlds,  # noqa: E402
                             paired, summarise)
from xray.config import load_config  # noqa: E402

DEFAULT_SEEDS = [42, 7, 13, 101, 202, 303, 404, 505]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--oracle", action="store_true",
                    help="add the labelled oracle upper bound (evaluation only)")
    ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    cfg = load_config()
    t0 = time.time()
    out = evaluate_worlds(cfg, a.seeds, include_oracle=a.oracle,
                          n_particles=a.particles)
    el = time.time() - t0
    agg = summarise(out)

    print("=" * 96)
    print("CLOSED-LOOP STRATEGY EVALUATION -- paired synthetic worlds")
    print("=" * 96)
    print("  varied per world (all existing config keys):")
    for k, v in WORLD_VARIATION.items():
        print(f"    {k:28s} {v}")
    print()
    for s, w in out["worlds"].items():
        print(f"    seed {s:<5} gap0={w['start_gap_s']:.3f}s  E0={w['e_start_frac']:.3f}"
              f"  leader={w['leader_policy']:<11s} noise={w['speed_noise_ms']:.3f}")
    print()
    hdr = (f"  {'policy':12s} {'att':>4s} {'succ':>5s} {'fail':>5s} {'ahead':>8s} "
           f"{'finA':>5s} {'dep MJ':>7s} {'unused':>7s} {'atk':>4s} {'interior':>8s} "
           f"{'meanP':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k, v in agg.items():
        print(f"  {k:12s} {v['attempts']:4d} {v['successes']:5d} {v['failures']:5d} "
              f"{v['mean_ahead_fraction']:8.5f} {v['finished_ahead']:5d} "
              f"{v['mean_energy_deployed_mj']:7.2f} {v['mean_unused_mj']:7.3f} "
              f"{v['attacked']:4d} {v['interior_budget']:8d} {v['mean_p_pass']:7.4f}")
    print()
    print("  PAIRED (same world, same dice, different decision rule)")
    pairs = [("p2", "hold"), ("p2", "fixed_cost"), ("p2", "max_deploy")]
    if a.oracle:
        pairs.append(("oracle_p2", "p2"))
    for x, y in pairs:
        if x in agg and y in agg:
            r = paired(out, x, y)
            print(f"    {x:10s} vs {y:12s} n={r['n']:2d} "
                  f"W/L/T={r['wins']}/{r['losses']}/{r['ties']} "
                  f"mean ahead delta={r['mean_ahead_delta']:+.5f}")
    print()
    print(f"  {len([r for r in out['rows'] if not isinstance(r, dict)])} episodes "
          f"in {el:.0f} s")
    print("  Pass model: SYNTHETIC. Passes are rare by construction; a null "
          "separation is a result, not a tuning target.")

    if a.out:
        rows = [{"seed": r.seed, "policy": r.policy, "lap": r.plan.lap,
                 "zone": r.plan.zone,
                 "budget_mj": None if r.plan.budget_j is None else r.plan.budget_j / 1e6,
                 "attempts": r.attempts, "successes": r.successes,
                 "ahead": r.ahead_fraction, "finished_ahead": r.finished_ahead,
                 "deployed_mj": r.energy_deployed_j / 1e6,
                 "unused_mj": r.energy_unused_j / 1e6}
                for r in out["rows"] if not isinstance(r, dict)]
        Path(a.out).write_text(json.dumps(
            {"worlds": out["worlds"], "aggregate": agg, "rows": rows,
             "seconds": el}, indent=1, default=str))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
