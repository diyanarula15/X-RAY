"""Live-readiness: kernel purity, and how far the estimator peeks ahead.

Live operation will stream samples and will likely run compiled kernels. Nothing
is ported yet; these tests pin the two properties that would make a port either
possible or impossible, so that a change which quietly breaks them fails here
rather than in a race.

The architecture is a *batch calibration* (Stage A) feeding a *causal filter*
(Stages B and C). That split is what makes live operation tractable: Stage A
becomes a rolling re-fit, Stage C runs sample by sample. These tests assert the
second half really is causal, and measure the latency of the first.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from xray.estimator import (estimate, fit_nuisance, kinematics, smooth_speed,
                            smooth_window)
from xray.observe import Observation, observe
from xray.sim import LEADER

XRAY = Path(__file__).resolve().parent.parent / "xray"

# Modules that must stay pure numerics: contiguous float64 arrays in, arrays and
# scalars out. FastF1 and pandas live in the ingest/analysis boundary layer, and
# a port that had to drag a DataFrame into a kernel would not be a port.
KERNELS = ("estimator.py", "realfit.py", "overtake.py", "decision.py",
           "metrics.py", "observe.py", "constants.py", "track.py",
           "regs.py", "balance.py", "setmem.py", "modes.py", "pipeline.py")
IMPURE = {"pandas", "fastf1", "matplotlib", "pyplot", "json", "requests"}

# Longest braking event on seed 42, measured at both rates: 1.730 s at 100 Hz,
# 1.892 s at 3.7 Hz. `brake_mask` thresholds each event at half of that event's
# own peak deceleration, so it cannot classify the event's first sample until
# the event has ended -- and that dominates the smoother's half-window. This is
# the recovery channel's live latency, and it is load-bearing: the peak-relative
# threshold is what took recovered-energy error from +45% to about 5%. Raising
# this number means recovery gets later in live mode, which is a decision to
# make deliberately, not to discover.
BRAKE_LATENCY_S = 2.0


def _truncate(obs: Observation, k: int) -> Observation:
    return Observation(t=obs.t[:k], s=obs.s[:k], v=obs.v[:k], lap=obs.lap[:k],
                       gap_to_leader=obs.gap_to_leader[:k], car_id=obs.car_id,
                       sample_rate_hz=obs.sample_rate_hz)


def test_kernels_are_pure():
    """No I/O, no DataFrames, no plotting inside the numerical modules.

    Enforced the same way as the blindfold, and for the same reason: a helpful
    `import pandas` in a kernel is invisible in review and fatal to a port.
    """
    for name in KERNELS:
        path = XRAY / name
        tree = ast.parse(path.read_text())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        leaked = found & IMPURE
        assert not leaked, f"{name} imports {sorted(leaked)} inside a kernel"


def test_smoother_is_causal_beyond_half_a_window(cfg, races):
    """Savitzky-Golay is centred, so it sees half a window into the future.

    Bounded, and exactly bounded: truncating the trace changes the last
    `window // 2` samples and *nothing* before them, bit for bit. That is what
    makes a fixed-latency streaming implementation possible at all. If this ever
    fails, some filter has started reaching further ahead than its window.
    """
    g = races[42]
    for rate in (3.7, 100.0):
        obs = observe(g, LEADER, rate_hz=rate,
                      speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
        half = smooth_window(rate) // 2
        k = len(obs.v) // 2
        _vf, af = smooth_speed(obs)
        _vt, at = smooth_speed(_truncate(obs, k))
        assert np.array_equal(af[:k - half], at[:k - half]), (
            f"{rate} Hz: smoothed acceleration differs before k-{half}")
        assert not np.array_equal(af[:k], at[:k]), (
            f"{rate} Hz: truncation changed nothing at all -- is the fixture real?")


def test_brake_classification_latency_is_bounded(cfg, races):
    """Recovery cannot be reported until the braking event has finished.

    Measured rather than assumed, because it sets the live latency of the
    recovery channel -- see BRAKE_LATENCY_S.
    """
    g = races[42]
    for rate in (3.7, 100.0):
        obs = observe(g, LEADER, rate_hz=rate,
                      speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
        kin = kinematics(obs, g.track)
        b = kin.brake.astype(np.int8)
        edges = np.diff(np.concatenate([[0], b, [0]]))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        assert len(starts) > 0, "no braking events in the fixture"
        worst = float((ends - starts).max()) / rate
        assert worst <= BRAKE_LATENCY_S, (
            f"{rate} Hz: longest braking event {worst:.3f} s exceeds the "
            f"declared live latency {BRAKE_LATENCY_S} s")


def test_stage_bc_is_causal_given_its_batch_inputs(cfg, races):
    """Stages B and C are bit-for-bit causal, to within the smoother's half-window.

    Fix the three quantities Stage A and the stint supply -- the nuisance fit,
    the band scale, and the stint length -- then truncate the trace a lap early.
    Every belief before `k - window // 2` is identical to the full-trace run, bit
    for bit, which is the property a streaming implementation needs.

    All three are known up front in live operation: the race distance is
    published, and the nuisance fit is a rolling estimate. Leaving stint length
    to `laps.max()` instead moves the buffer-release ramp as the window grows and
    shifts usable_mean by up to 0.19 MJ over the final RESERVE_RELEASE_LAPS laps
    -- which is why `estimate` takes `n_laps`.
    """
    g = races[42]
    obs = observe(g, LEADER, rate_hz=3.7,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
    nuisance = fit_nuisance(obs, g.track)
    kw = dict(n_particles=200, seed=5, nuisance=nuisance,
              deploy_scale_sigma=0.12, n_laps=int(obs.lap.max()) + 1)
    half = smooth_window(3.7) // 2
    k = int(np.flatnonzero(obs.lap == obs.lap.max())[0])   # start of the final lap

    full = estimate(obs, g.track, **kw)
    part = estimate(_truncate(obs, k), g.track, **kw)

    for name in ("soc_mean", "soc_p10", "soc_p90", "usable_mean", "usable_p10",
                 "usable_p90", "p_mguk_mean", "harvest_mean"):
        a = getattr(full, name)[:k - half]
        b = getattr(part, name)[:k - half]
        assert np.array_equal(a, b), (
            f"{name}: streaming would not reproduce batch -- "
            f"max |d| {np.abs(a - b).max():.3e}")
    assert np.array_equal(full.dry_events[:k - half], part.dry_events[:k - half])
