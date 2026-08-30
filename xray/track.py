"""Circuit Sigma: a synthetic circuit.

The track is *public*. Its geometry, its corner speed limits, its aero-mode
schedule and its Manual-Override detection points are all known a priori to
everyone, including the estimator. Nothing here is inferred from the cars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

KMH = 1 / 3.6
STRAIGHT_V_LIMIT = 400 * KMH  # a limit high enough never to bind


@dataclass(frozen=True)
class Segment:
    kind: str  # "straight" | "corner"
    length: float  # m
    v_limit: float  # m/s (corner apex speed; high value for straights)
    aero_mode: str  # "straight" | "corner" -- position-gated, known a priori
    is_overtake_zone: bool
    grade: float = 0.0  # rad, positive = uphill
    name: str = ""


@dataclass(frozen=True)
class Zone:
    name: str
    s_straight_start: float
    s_straight_end: float  # == corner entry
    s_end: float  # end of the apex
    apex_v: float  # m/s
    braking_severity: float  # [0, 1]
    detection_point: float  # m, 200 m before the straight starts
    energy_cost_hint: float  # J, full-deployment cost of the straight (filled in later)


def _sigma_segments() -> list[Segment]:
    """Circuit Sigma: 5,200 m, three overtake zones of deliberately different
    character, one straight long enough to push a fully deployed car past
    340 km/h so that the regulatory taper is genuinely exercised."""
    S, C = "straight", "corner"
    return [
        Segment(S, 1100.0, STRAIGHT_V_LIMIT, S, True, 0.0, "S1 (Zone A straight)"),
        Segment(C, 170.0, 65 * KMH, C, True, 0.0, "T1 hairpin"),
        Segment(S, 850.0, STRAIGHT_V_LIMIT, S, True, 0.0, "S2 (Zone B straight)"),
        Segment(C, 90.0, 140 * KMH, C, True, 0.0, "T2"),
        Segment(S, 580.0, STRAIGHT_V_LIMIT, S, True, 0.0, "S3 (Zone C straight)"),
        Segment(C, 80.0, 240 * KMH, C, True, 0.0, "T3 kink"),
        Segment(S, 380.0, STRAIGHT_V_LIMIT, S, False, 0.010, "S4"),
        Segment(C, 100.0, 110 * KMH, C, False, 0.0, "T4"),
        Segment(S, 330.0, STRAIGHT_V_LIMIT, S, False, 0.0, "S5"),
        Segment(C, 80.0, 175 * KMH, C, False, 0.0, "T5"),
        Segment(S, 260.0, STRAIGHT_V_LIMIT, S, False, 0.0, "S6"),
        Segment(C, 150.0, 95 * KMH, C, False, 0.0, "T6"),
        Segment(S, 430.0, STRAIGHT_V_LIMIT, S, False, -0.00884, "S7"),
        Segment(C, 90.0, 200 * KMH, C, False, 0.0, "T7"),
        Segment(S, 600.0, STRAIGHT_V_LIMIT, S, False, 0.0, "S8 (pit straight)"),
    ]


class Track:
    """Ordered segments plus O(1) grid lookups.

    ``aero_cda_ratio`` is the declared high-downforce / low-drag CdA ratio.  It
    is part of the public aero-mode schedule: the estimator is told *where* the
    car switches modes and by what ratio, and fits only the absolute scale.
    """

    GRID_DS = 0.5  # m

    def __init__(self, segments: list[Segment] | None = None,
                 aero_cda_ratio: float = 1.42 / 0.66, name: str = "Circuit Sigma"):
        self.name = name
        self.segments = segments if segments is not None else _sigma_segments()
        self.aero_cda_ratio = aero_cda_ratio

        bounds = np.cumsum([0.0] + [seg.length for seg in self.segments])
        self._starts = bounds[:-1]
        self._ends = bounds[1:]
        self.length = float(bounds[-1])

        n = int(round(self.length / self.GRID_DS))
        self._grid_s = (np.arange(n) + 0.5) * self.GRID_DS
        idx = np.searchsorted(self._ends, self._grid_s, side="right")
        idx = np.clip(idx, 0, len(self.segments) - 1)
        self._grid_seg = idx
        self._grid_vlim = np.array([s.v_limit for s in self.segments])[idx]
        self._grid_grade = np.array([s.grade for s in self.segments])[idx]
        self._grid_iscorner = np.array([s.kind == "corner" for s in self.segments])[idx]

        self.corners = [
            (float(self._starts[i]), float(self._ends[i]), float(seg.v_limit))
            for i, seg in enumerate(self.segments) if seg.kind == "corner"
        ]
        self.zones = self._build_zones()
        self.detection_points = [z.detection_point for z in self.zones]
        self._xy = None

    # ------------------------------------------------------------------ zones
    def _build_zones(self) -> list[Zone]:
        severity = {"A": 1.0, "B": 0.55, "C": 0.2}
        zones, letter = [], iter("ABC")
        for i, seg in enumerate(self.segments):
            if not (seg.kind == "straight" and seg.is_overtake_zone):
                continue
            corner = self.segments[i + 1]
            name = next(letter)
            start = float(self._starts[i])
            zones.append(Zone(
                name=name,
                s_straight_start=start,
                s_straight_end=float(self._ends[i]),
                s_end=float(self._ends[i + 1]),
                apex_v=float(corner.v_limit),
                braking_severity=severity[name],
                detection_point=float((start - 200.0) % self.length),
                energy_cost_hint=0.0,
            ))
        return zones

    # ---------------------------------------------------------------- lookups
    def _gi(self, s):
        return (np.asarray(s, dtype=float) % self.length / self.GRID_DS).astype(np.int64) \
            .clip(0, len(self._grid_s) - 1)

    def v_limit(self, s):
        out = self._grid_vlim[self._gi(s)]
        return float(out) if np.ndim(out) == 0 else out

    def grade(self, s):
        out = self._grid_grade[self._gi(s)]
        return float(out) if np.ndim(out) == 0 else out

    def is_corner(self, s):
        out = self._grid_iscorner[self._gi(s)]
        return bool(out) if np.ndim(out) == 0 else out

    def aero_mode(self, s):
        if np.ndim(s) == 0:
            return "corner" if self.is_corner(s) else "straight"
        return np.where(self.is_corner(s), "corner", "straight")

    def cda_scale(self, s):
        """Position-gated multiplier on the low-drag reference CdA."""
        return np.where(self.is_corner(s), self.aero_cda_ratio, 1.0)

    def zone_at(self, s) -> Optional[Zone]:
        s = float(s) % self.length
        for z in self.zones:
            if z.s_straight_start <= s < z.s_end:
                return z
        return None

    def zone_by_name(self, name: str) -> Zone:
        return next(z for z in self.zones if z.name == name)

    def next_corner(self, s, horizon=1800.0):
        """(distance_ahead, v_target) for every corner within the horizon."""
        s = float(s) % self.length
        out = []
        for start, _end, vlim in self.corners:
            d = (start - s) % self.length
            if d <= horizon:
                out.append((d, vlim))
        return out

    # --------------------------------------------------------- schematic map
    def xy(self) -> np.ndarray:
        """Schematic 2-D centreline for the track-map panel.

        Corner turn angles are normalised so the heading closes over a lap, and
        a small linear drift correction closes the position. Purely cosmetic --
        no physics reads this.
        """
        if self._xy is not None:
            return self._xy
        turn = {}
        for i, seg in enumerate(self.segments):
            if seg.kind == "corner":
                # tighter corner (lower apex speed) -> larger turn angle
                turn[i] = 1.0 / max(seg.v_limit, 8.0) ** 0.9
        scale = 2 * np.pi / sum(turn.values())
        pts, heading, pos = [], 0.0, np.zeros(2)
        for i, seg in enumerate(self.segments):
            n = max(2, int(seg.length / 10))
            dtheta = turn.get(i, 0.0) * scale / n
            for _ in range(n):
                heading += dtheta
                pos = pos + (seg.length / n) * np.array([np.cos(heading), np.sin(heading)])
                pts.append(pos.copy())
        pts = np.array(pts)
        drift = pts[-1] - pts[0]
        pts -= np.outer(np.linspace(0, 1, len(pts)), drift)
        self._xy = pts
        return pts

    def xy_at(self, s) -> np.ndarray:
        pts = self.xy()
        frac = (np.asarray(s, dtype=float) % self.length) / self.length
        idx = (frac * (len(pts) - 1)).astype(np.int64)
        return pts[idx]


def circuit_sigma() -> Track:
    return Track()
