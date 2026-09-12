#!/usr/bin/env python3
"""Build the 60-second demo video, end to end, from a clean checkout."""
from __future__ import annotations

import time
from pathlib import Path

from _common import base_parser, config_from
from xray.render.video import FPS, build_scene, render


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--rate", type=float, default=None,
                    help="telemetry sample rate the estimator is given")
    args = ap.parse_args()
    cfg = config_from(args)
    out = args.out or "out/xray_demo.mp4"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"building scene (seed {args.seed}) ...")
    sc = build_scene(cfg, seed=args.seed, rate_hz=args.rate)
    print(f"rendering {out} ...")
    render(sc, out, fps=args.fps)
    size = Path(out).stat().st_size / 1e6
    print(f"\nwrote {out}  ({size:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
