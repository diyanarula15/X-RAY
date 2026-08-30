"""Scoring. This module is allowed to look at the ground truth -- it is the
only thing in the pipeline that is, and it does so strictly after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import E_STORE_MAX
from .estimator import RESERVE_RELEASE_LAPS, BeliefTrace
from .observe import Observation
from .policy import get_policy


def true_reserve_floor(gt, car_id: str, lap: np.ndarray) -> np.ndarray:
    """The buffer the rival's driver was actually holding back, per sample.

    Ground truth, used only for scoring the deployable-energy band.
    """
    reserve = get_policy(gt.policies[car_id]).reserve * E_STORE_MAX
    laps_left = np.maximum(gt.n_laps - np.asarray(lap), 0)
    return reserve * np.minimum(1.0, laps_left / RESERVE_RELEASE_LAPS)


def _at_obs_times(gt, obs: Observation) -> np.ndarray:
    return np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)


def complete_laps(obs: Observation, belief: BeliefTrace, n_laps: int) -> np.ndarray:
    """Laps that are both fully sampled and inside the simulator's lap table."""
    laps = belief.lap_index
    ok = (laps >= 0) & (laps < n_laps)
    if not ok.any():
        return ok
    counts = np.array([np.sum(obs.lap == L) for L in laps])
    return ok & (counts >= 0.9 * np.median(counts[ok]))


@dataclass(frozen=True)
class Score:
    car_id: str
    rate_hz: float
    deployed_mape: float
    harvested_mape: float
    usable_coverage: float
    soc_coverage: float
    usable_band_mj: float
    usable_rmse_mj: float
    soc_rmse_mj: float
    cda_error_pct: float
    n_laps_scored: int
    extra: dict = field(default_factory=dict)


def score_estimate(gt, car_id: str, obs: Observation, belief: BeliefTrace,
                   cda_true: float) -> Score:
    trace = gt.cars[car_id]
    ok = complete_laps(obs, belief, gt.n_laps)
    laps = belief.lap_index[ok]
    true_dep = trace.deployed_lap[laps]
    true_har = trace.harvested_lap[laps]

    def mape(est, true):
        good = true > 1.0e4
        if not good.any():
            return float("nan")
        return float(100.0 * np.mean(np.abs(est[good] - true[good]) / true[good]))

    idx = _at_obs_times(gt, obs)
    true_soc = trace.E[idx]
    true_usable = np.maximum(true_soc - true_reserve_floor(gt, car_id, obs.lap), 0.0)

    return Score(
        car_id=car_id, rate_hz=obs.sample_rate_hz,
        deployed_mape=mape(belief.deployed_lap[ok], true_dep),
        harvested_mape=mape(belief.harvested_lap[ok], true_har),
        usable_coverage=float(np.mean((true_usable >= belief.usable_p10)
                                      & (true_usable <= belief.usable_p90))),
        soc_coverage=float(np.mean((true_soc >= belief.soc_p10)
                                   & (true_soc <= belief.soc_p90))),
        usable_band_mj=float(np.mean(belief.usable_p90 - belief.usable_p10) / 1e6),
        usable_rmse_mj=float(np.sqrt(np.mean((belief.usable_mean - true_usable) ** 2)) / 1e6),
        soc_rmse_mj=float(np.sqrt(np.mean((belief.soc_mean - true_soc) ** 2)) / 1e6),
        cda_error_pct=float(100.0 * (belief.nuisance.cda_hat - cda_true) / cda_true),
        n_laps_scored=int(ok.sum()),
        extra={"deploy_scale_sigma": belief.deploy_scale_sigma,
               "n_taper_samples": belief.nuisance.n_samples,
               "dry_events": int(belief.dry_events.sum())})


def band_width_by_regime(gt, car_id: str, obs: Observation, belief: BeliefTrace) -> dict:
    """Where does the belief tighten, and where does it go slack?

    The claim the demo makes is that the band is not decorative: it narrows
    when the trace is informative and widens when it is not.
    """
    trace = gt.cars[car_id]
    idx = _at_obs_times(gt, obs)
    regime = trace.regime[idx]
    width = belief.usable_p90 - belief.usable_p10
    out = {}
    for name in ("accel", "brake", "corner"):
        m = regime == name
        if m.any():
            out[name] = float(np.mean(width[m]) / 1e6)
    zone = np.array([gt.track.zone_at(s) is not None for s in obs.s])
    out["zone_straight"] = float(np.mean(width[zone]) / 1e6) if zone.any() else float("nan")
    out["elsewhere"] = float(np.mean(width[~zone]) / 1e6) if (~zone).any() else float("nan")
    return out


TARGETS = {
    "deployed_mape_100hz": ("Per-lap deployed energy error (MAPE) @ 100 Hz", "<= 8%"),
    "deployed_mape_3p7hz": ("Per-lap deployed energy error (MAPE) @ 3.7 Hz", "<= 15%"),
    "usable_coverage": ("Deployable-energy band coverage (p10-p90)", "0.75 - 0.85"),
    "cda_error": ("Nuisance fit: CdA_hat vs true", "within 6%"),
    "positions_gained": ("Positions gained vs blind baseline", "> 0, 95% CI"),
}


def format_table(rows: list[Score]) -> str:
    head = (f"{'car':9s} {'rate':>7s} {'dep MAPE':>9s} {'har MAPE':>9s} "
            f"{'usable cov':>11s} {'band MJ':>8s} {'usable RMSE':>12s} "
            f"{'CdA err':>8s} {'SoC cov':>8s}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.car_id:9s} {r.rate_hz:6.1f}H {r.deployed_mape:8.1f}% "
                     f"{r.harvested_mape:8.1f}% {r.usable_coverage:11.3f} "
                     f"{r.usable_band_mj:8.2f} {r.usable_rmse_mj:11.2f}M "
                     f"{r.cda_error_pct:+7.2f}% {r.soc_coverage:8.3f}")
    return "\n".join(lines)


def aggregate(rows: list[Score]) -> dict:
    def agg(f):
        vals = np.array([f(r) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return dict(mean=float(vals.mean()), min=float(vals.min()), max=float(vals.max()))
    return {
        "deployed_mape": agg(lambda r: r.deployed_mape),
        "harvested_mape": agg(lambda r: r.harvested_mape),
        "usable_coverage": agg(lambda r: r.usable_coverage),
        "soc_coverage": agg(lambda r: r.soc_coverage),
        "cda_abs_error_pct": agg(lambda r: abs(r.cda_error_pct)),
    }
