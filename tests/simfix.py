"""Bridging the Stage 1 simulator onto the interval-balance kernels.

The simulator is the only place a *true* theta exists, so it is the only place
set-membership containment can be checked. Two alignment details matter and
both were found by measurement, not by reading:

1. `sim.step()` mutates the car state and *then* records, so `trace.v[k]` is
   the speed *after* step k while `trace.P_mguk[k]` is the power applied
   *during* it. The power that acts across the interval [v[k], v[k+1]] is
   therefore index k+1. Verified: the store balance
   E[k+1] - E[k] = (-P_mguk[k+1] + harvest[k+1]) * dt closes to 0.000 J rms,
   against 168.9 J rms for index k.
2. Circuit Sigma publishes gradient, not elevation, and the simulator applies
   it as a force m*g*grade over each step. Elevation is therefore the running
   integral of gradient along the lap, which is what makes dE_pot agree with
   the work the simulator actually did.
"""
from __future__ import annotations

import numpy as np

from xray.balance import M_REF_KG
from xray.constants import G

N_GRID = 20_000


def elevation_fn(track):
    """z(s) from the track's gradient, matching the simulator's own bookkeeping."""
    s = np.linspace(0.0, track.length, N_GRID, endpoint=False)
    z = np.cumsum(track.grade(s)) * (track.length / N_GRID)
    return lambda q: np.interp(np.asarray(q) % track.length, s, z)


def true_theta(cfg) -> np.ndarray:
    """(CdA_X, CdA_Z, F_rr, dm) as the simulator actually ran it.

    F_rr is rolling resistance expressed as a force at M_REF_KG, so it is
    crr * M_REF * g rather than crr itself; dm is dry mass minus the published
    weight, and the fixture publishes its own mass, so dm is exactly zero.
    """
    v = cfg["vehicle"]
    return np.array([v["cda_straight"], v["cda_corner"],
                     v["crr"] * M_REF_KG * G, 0.0])


def observed_sigma(cfg) -> float:
    """Speed error the feed actually delivers: sensor noise and quantisation."""
    from xray.balance import QUANT_SIGMA_MS
    return float(np.hypot(cfg["observe"]["speed_noise_ms"], QUANT_SIGMA_MS))
