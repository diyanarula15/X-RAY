"""The blindfold.

`observe` is the only door between the simulator and the estimator, and
`Observation` is the only thing that fits through it. There is no energy,
throttle, brake, gear or deployment field on it, and there never will be.

`public_channels` sits beside it and is deliberately NOT part of that struct.
Stage 1 solves the speed-only problem and must keep solving it, so nothing
reached through `Observation` ever gains a throttle or brake field. The real
inference core is a different case: FastF1 publishes those channels, so feeding
it a trace without them tests a code path no real race takes. The two callers
are kept apart rather than the struct widened -- see `public_channels`.

The energy state -- `E`, `P_mguk`, `harvest` -- is hidden from both, always.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import P_ICE_MAX

KMH = 1 / 3.6


@dataclass(frozen=True)
class Observation:
    t: np.ndarray             # timestamps, seconds
    s: np.ndarray             # distance along lap, metres (from timing loops)
    v: np.ndarray             # speed, m/s  <-- THE ONLY SIGNAL
    lap: np.ndarray           # integer lap index per sample
    gap_to_leader: np.ndarray  # seconds, or NaN when this car is leading
    car_id: str
    sample_rate_hz: float

    def __len__(self) -> int:
        return len(self.t)


def observe(gt, car_id: str, rate_hz: float = 3.7, speed_noise_ms: float = 0.35,
            seed: int = 0, quantize_kmh: float = 1.0) -> Observation:
    """Ground truth -> what a public timing feed would actually publish.

    Nearest-sample resampling (a feed reports the most recent sample, it does
    not interpolate), additive Gaussian speed noise, and 1 km/h quantisation.
    """
    trace = gt.cars[car_id]
    t_src = gt.t
    n_out = int(np.floor((t_src[-1] - t_src[0]) * rate_hz)) + 1
    t_out = t_src[0] + np.arange(n_out) / rate_hz
    idx = np.clip(np.searchsorted(t_src, t_out), 0, len(t_src) - 1)

    rng = np.random.default_rng(seed)
    v = trace.v[idx] + rng.normal(0.0, speed_noise_ms, size=n_out)
    if quantize_kmh:
        step = quantize_kmh * KMH
        v = np.round(v / step) * step
    v = np.maximum(v, 0.0)

    # gap to the car ahead; NaN whenever this car is the one being chased
    leading = (gt.order[idx] == (0 if car_id == "LEADER" else 1))
    gap = np.where(leading, np.nan, gt.gap_s[idx])

    return Observation(
        t=t_out, s=trace.s[idx].copy(), v=v, lap=trace.lap[idx].astype(np.int64),
        gap_to_leader=gap, car_id=car_id, sample_rate_hz=float(rate_hz))


def public_channels(gt, car_id: str, obs: Observation) -> dict:
    """The published non-speed channels, at the observation's own sample times.

    The counterpart of FastF1's `lap.get_car_data()`: source-specific extraction,
    kept here because this module is already the one place that reads the
    simulator. What it returns goes straight into `data.ingest.grid_lap`, the
    same gridding real telemetry uses.

    This is not a hole in the blindfold. The hidden state is the ELECTRICAL one
    -- `E`, `P_mguk`, `harvest` -- and none of it is read here. Throttle and
    brake are published FastF1 channels, so a synthetic feed without them is not
    a harder honest problem, it is a feed the real inference core never sees:
    measured, omitting them drives `fit_nuisance_real` to n_binding = 0 and
    identifiability 0.000 at every rate from 3.7 to 100 Hz, because
    `realfit.coast_phases` falls back to a bare deceleration test that cannot
    tell coasting from braking. `Observation` omits them because Stage 1 chose
    the harder speed-only problem, not because they are secret.

    Throttle is derived as `P_ice / P_ICE_MAX`, which makes `realfit.ice_power`'s
    pedal-to-power model exact by construction -- OPTIMISTIC, and stated rather
    than hidden. The alternative of a binary on-power flag is worse rather than
    more honest: it reports wide-open throttle through corners where the
    simulator is actually at `resist * v / eta`, a few tens of kW.
    """
    idx = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)
    trace = gt.cars[car_id]
    return {
        "throttle": 100.0 * np.clip(trace.P_ice[idx] / P_ICE_MAX, 0.0, 1.0),
        "brake": (trace.regime[idx] == "brake").astype(float),
    }
