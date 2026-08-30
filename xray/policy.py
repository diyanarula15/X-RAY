"""Deployment policy -- the "driver".

Four interpretable parameters. This is what the estimator is ultimately trying
to see through, and what `decision.py` samples a posterior over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .constants import E_STORE_MAX, P_MGUK_MAX, p_mguk_ceiling

GAP_WINDOW_S = 1.5  # gaps closer than this start to modulate deployment


@dataclass(frozen=True)
class DeploymentPolicy:
    name: str
    front_loading: float          # 0..1  0 = save for the last sector, 1 = dump early
    reserve: float                # 0..1  fraction of the store held back as buffer
    gap_sensitivity: float        # 0..2  extra deployment per unit closing gap
    zone_preference: dict         # {"A": 1.0, "B": 0.6, "C": 0.3} relative priority

    # ---------------------------------------------------------------- weights
    def zone_weight(self, track, zone) -> float:
        """Zone priority, skewed early or late by ``front_loading``."""
        if zone is None:
            # baseline lap deployment: half the least-favoured zone
            return 0.5 * min(self.zone_preference.values())
        names = [z.name for z in track.zones]
        i = names.index(zone.name)
        span = max(len(names) - 1, 1)
        skew = 1.0 + self.front_loading * (1.0 - 2.0 * i / span)
        return self.zone_preference[zone.name] * skew

    def effective_reserve(self, laps_left: int) -> float:
        """The buffer is released over the last three laps of the stint."""
        return self.reserve * min(1.0, max(laps_left, 0) / 3.0)

    # ----------------------------------------------------------------- demand
    def demand(self, track, s: float, v: float, E: float, gap_ahead: float,
               gap_behind: float, laps_left: int, is_corner: bool = False) -> float:
        """Requested store-side MGU-K power, in W."""
        if is_corner:
            return 0.0
        zone = track.zone_at(s)
        if zone is not None and s >= zone.s_straight_end:
            return 0.0  # inside the zone's braking area / apex

        usable = E - self.effective_reserve(laps_left) * E_STORE_MAX * (1.0 - 0.0)
        if usable <= 0.0:
            return 0.0
        # soft roll-off over the last 10% of the usable store, so the trace
        # shows a real "running dry" signature rather than a numerical cliff
        avail_frac = min(1.0, usable / (0.10 * E_STORE_MAX))

        w = self.zone_weight(track, zone)

        closing = 0.0
        for gap in (gap_ahead, gap_behind):
            if gap is not None and 0.0 < gap < GAP_WINDOW_S:
                closing = max(closing, (GAP_WINDOW_S - gap) / GAP_WINDOW_S)
        w *= 1.0 + self.gap_sensitivity * closing

        return min(w * avail_frac * P_MGUK_MAX, p_mguk_ceiling(v))


def policy_demand(policy: DeploymentPolicy, track, track_pos: float, v: float,
                  E: float, gap_ahead: float, gap_behind: float,
                  laps_left: int) -> float:
    """Module-level form of :meth:`DeploymentPolicy.demand`."""
    return policy.demand(track, track_pos, v, E, gap_ahead, gap_behind, laps_left,
                         is_corner=track.is_corner(track_pos))


AGGRESSIVE = DeploymentPolicy(
    name="AGGRESSIVE", front_loading=0.80, reserve=0.00, gap_sensitivity=1.20,
    zone_preference={"A": 1.00, "B": 0.70, "C": 0.40})

BALANCED = DeploymentPolicy(
    name="BALANCED", front_loading=0.45, reserve=0.15, gap_sensitivity=0.60,
    zone_preference={"A": 1.00, "B": 0.60, "C": 0.30})

CONSERVATIVE = DeploymentPolicy(
    name="CONSERVATIVE", front_loading=0.15, reserve=0.35, gap_sensitivity=0.25,
    zone_preference={"A": 0.80, "B": 0.50, "C": 0.20})

MOM_ATTACK = DeploymentPolicy(
    name="MOM_ATTACK", front_loading=0.20, reserve=0.05, gap_sensitivity=1.60,
    zone_preference={"A": 1.00, "B": 0.20, "C": 0.10})

PRESETS = {p.name: p for p in (AGGRESSIVE, BALANCED, CONSERVATIVE, MOM_ATTACK)}


def get_policy(name: str) -> DeploymentPolicy:
    try:
        return PRESETS[name.upper()]
    except KeyError:
        raise KeyError(f"unknown policy {name!r}; have {sorted(PRESETS)}") from None
