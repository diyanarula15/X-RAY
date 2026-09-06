#!/usr/bin/env python3
"""Regenerate the golden estimator outputs in tests/golden/.

Run this ONLY when a numerics change is intended, and say in the commit message
which headline metric moved and why. `tests/test_golden.py` compares against
these files, so regenerating them silently converts a regression into a new
baseline -- which is the one failure mode this harness exists to prevent.

The configuration here is pinned to match `tests/conftest.py` exactly (seed 42,
LEADER, 3.7 Hz, obs seed 43, estimate seed 44), so the test reuses the
session-scoped fixtures and costs no extra simulation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import base_parser, config_from
from tests.golden_io import SCALAR_NAMES, belief_arrays
from xray.estimator import estimate
from xray.observe import observe
from xray.sim import LEADER, run_sim

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden"
RATE = 3.7
CAR = LEADER


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = config_from(args)
    seed = args.seed
    gt = run_sim(cfg, seed=seed)
    obs = observe(gt, CAR, rate_hz=RATE,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=seed + 1)
    bel = estimate(obs, gt.track, n_particles=cfg["estimator"]["n_particles"],
                   seed=seed + 2)
    arrays = belief_arrays(bel)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    out = GOLDEN / f"belief_seed{seed}_{RATE:g}hz.npz"
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
    for name, v in zip(SCALAR_NAMES, arrays["scalars"]):
        print(f"  {name:20s} {v:.10g}")


if __name__ == "__main__":
    main()
