"""Composing steps 1-3: channels -> modes -> constraints -> identified set.

A thin outer layer. It holds no numerics of its own -- every computation lives
in `modes`, `balance` or `setmem` -- so that the port to a compiled kernel later
moves those three and leaves this behind.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .balance import build_window_constraints
from .modes import ModePath, infer
from .regs import RegSet, regs_for
from .setmem import IdentifiedSet, identify


@dataclass(frozen=True)
class Feed:
    """One car's public telemetry, on a common time base.

    Everything here is published. `x_allowed` and `in_zone` come from the
    circuit's aero and deployment maps, which are hand-entered per circuit and
    flagged as such -- see docs/feasibility.md 1.4.
    """
    t: np.ndarray
    v: np.ndarray
    s: np.ndarray
    z: np.ndarray
    lap_frac: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    rpm: np.ndarray
    gap_s: np.ndarray
    x_allowed: np.ndarray
    in_zone: np.ndarray
    manual_override: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.t)


@dataclass(frozen=True)
class Identification:
    modes: ModePath
    identified: IdentifiedSet
    regs: RegSet
    window_s: float
    n_constraints: int

    def describe(self) -> str:
        return f"{self.regs.variant} | {self.identified.describe()}"


def accel_from_speed(v, t):
    """Acceleration, for the HMM's benefit only.

    A derivative, and the one place one survives -- but it feeds a *classifier*,
    where an error costs a label, not an energy. The energy path never
    differentiates: see `balance.py`.
    """
    return np.gradient(np.asarray(v, float), np.asarray(t, float))


def identify_car(feed: Feed, rho: float, race_date=None, window_s: float = 8.0,
                 eta_d: float = 0.95, m_published: float = 790.0,
                 speed_sigma_ms: float | None = None, causal: bool = False,
                 fuel_start: float = 70.0, fuel_burn_per_lap: float = 1.15,
                 **kw) -> Identification:
    regs = regs_for(race_date)
    modes = infer(feed.v, feed.brake, feed.throttle,
                  accel_from_speed(feed.v, feed.t), feed.in_zone,
                  feed.x_allowed, causal=causal)
    cons = build_window_constraints(
        feed.v, feed.t, feed.z, feed.lap_frac, modes.is_x, feed.in_zone,
        feed.rpm, feed.throttle, feed.brake, feed.gap_s, regs, rho,
        window_s=window_s, eta_d=eta_d, manual_override=feed.manual_override,
        m_published=m_published, fuel_start=fuel_start,
        fuel_burn_per_lap=fuel_burn_per_lap,
        **({} if speed_sigma_ms is None else {"speed_sigma_ms": speed_sigma_ms}),
        **kw)
    return Identification(modes=modes, identified=identify(cons), regs=regs,
                          window_s=window_s, n_constraints=len(cons))
