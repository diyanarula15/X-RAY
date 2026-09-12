"""Deployment policy -- the "driver".

Four interpretable parameters. This is what the estimator is ultimately trying
to see through, and what `decision.py` samples a posterior over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .constants import E_STORE_MAX, P_MGUK_MAX, p_mguk_ceiling

GAP_WINDOW_S = 1.5  # gaps closer than this start to modulate deployment
_WEIGHTS: dict = {}   # (track, policy) -> {zone name: weight}


@dataclass(frozen=True)
class DeploymentPolicy:
    name: str
    front_loading: float          # 0..1  0 = save for the last sector, 1 = dump early
    reserve: float                # 0..1  fraction of the store held back as buffer
    gap_sensitivity: float        # 0..2  extra deployment per unit closing gap
    zone_preference: dict         # {"A": 1.0, "B": 0.6, "C": 0.3} relative priority
    # An attack the decision engine has called: on this lap, in this zone, spend
    # everything and ignore the buffer. This is what closes the loop -- the
    # recommendation changes the physics, and the overtake then either happens
    # or it does not.
    attack_lap: int | None = None
    attack_zone: str | None = None
    save_until: int | None = None   # hold the store back until this lap

    # ---------------------------------------------------------------- weights
    def zone_weight(self, track, zone) -> float:
        """Zone priority, skewed early or late by ``front_loading``.

        Cached per track: this is called twice per simulator timestep, and
        rebuilding the name list to find an index every time was pure waste.
        """
        if zone is None:
            return 0.5 * min(self.zone_preference.values())
        cache = _WEIGHTS.get((id(track), id(self)))
        if cache is None:
            names = [z.name for z in track.zones]
            span = max(len(names) - 1, 1)
            cache = {n: self.zone_preference.get(n, 0.0)
                     * (1.0 + self.front_loading * (1.0 - 2.0 * i / span))
                     for i, n in enumerate(names)}
            _WEIGHTS[(id(track), id(self))] = cache
        return cache.get(zone.name, 0.0)

    def effective_reserve(self, laps_left: int) -> float:
        """The buffer is released over the last three laps of the stint."""
        return self.reserve * min(1.0, max(laps_left, 0) / 3.0)

    # --------------------------------------------------------------- feedback
    def note_deployed(self, lap: int, energy_j: float) -> None:
        """Told how much MGU-K energy the integrator ACTUALLY delivered.

        A no-op here. It exists because a budgeted policy cannot measure its own
        spend from the store level: a deployment straight ends in a braking zone,
        so the store is refilling while the attack is still running. Measured on
        Circuit Sigma zone A, drawdown across the straight runs
        0 -> 0.4281 -> 0.4492 -> 0.4492 -> -0.1353 MJ -- non-monotone, and
        negative by the end -- while deployed energy rises monotonically to
        0.4564 MJ. A budget compared against drawdown therefore never trips, and
        the 0.449 MJ once reported as zone A's "executable" energy was the peak
        of that curve, not a deployment measurement.
        """

    # ----------------------------------------------------------------- demand
    def demand(self, track, s: float, v: float, E: float, gap_ahead: float,
               gap_behind: float, laps_left: int, is_corner: bool = False,
               lap: int = -1) -> float:
        """Requested store-side MGU-K power, in W."""
        if is_corner:
            return 0.0
        zone = track.zone_at(s)
        if zone is not None and s >= zone.s_straight_end:
            return 0.0  # inside the zone's braking area / apex

        # The called attack: everything, here, now. This is what closes the loop
        # -- the recommendation changes the physics, and the pass then either
        # happens or it does not.
        if (self.attack_lap is not None and lap == self.attack_lap
                and zone is not None
                and (self.attack_zone is None or zone.name == self.attack_zone)):
            return min(P_MGUK_MAX, p_mguk_ceiling(v))

        # Saving for a called attack: bank anything above a working level.
        if self.save_until is not None and 0 <= lap < self.save_until:
            if E < 0.55 * E_STORE_MAX:
                return 0.0

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
