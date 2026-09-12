"""Circuit geometry from real telemetry: centreline, elevation, curvature.

The Stage 1 `Track` was a synthetic list of segments. This builds the same
interface from a real circuit's position stream, so the Stage 1 core runs
unchanged on real data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CIRCUIT_DIR = Path(__file__).resolve().parent / "circuits"
GRID_DS = 10.0
CORNER_KAPPA = 1.0 / 180.0   # 1/m; tighter than a 180 m radius counts as a corner


@dataclass(frozen=True)
class CircuitGeometry:
    name: str
    length: float
    s: np.ndarray            # m along the lap
    xy: np.ndarray           # (n, 2) metres, centred
    z: np.ndarray            # m, elevation
    grade: np.ndarray        # rad, +uphill
    curvature: np.ndarray    # 1/m
    is_corner: np.ndarray    # bool
    has_elevation: bool
    source: str

    def at(self, key: str, s):
        arr = getattr(self, key)
        idx = np.clip((np.asarray(s) % self.length / self.length * (len(self.s) - 1)),
                      0, len(self.s) - 1).astype(int)
        return arr[idx]


def _smooth_closed(x: np.ndarray, win: int) -> np.ndarray:
    """Moving average on a closed loop, so the start/finish line is not a seam."""
    if win < 3:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ext = np.concatenate([x[-pad:], x, x[:pad]])
    k = np.ones(win) / win
    return np.convolve(ext, k, mode="same")[pad:-pad]


def geometry_from_session(session, driver: str | None = None,
                          smooth_m: float = 60.0) -> CircuitGeometry:
    """Build the centreline from a representative lap's position stream."""
    laps = session.laps
    lap = laps.pick_drivers(driver).pick_fastest() if driver else laps.pick_fastest()
    pos = lap.get_pos_data()
    if pos is None or len(pos) == 0 or "X" not in pos.columns:
        pos = session.pos_data[str(lap["DriverNumber"])]
    pos = pos[(pos["X"] != 0) | (pos["Y"] != 0)]

    x = pos["X"].to_numpy().astype(float) / 10.0   # FastF1 position is in 1/10 m
    y = pos["Y"].to_numpy().astype(float) / 10.0
    z = pos["Z"].to_numpy().astype(float) / 10.0 if "Z" in pos.columns else np.zeros_like(x)

    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    length = float(d[-1])
    s = np.arange(0.0, length, GRID_DS)
    xg = np.interp(s, d, x)
    yg = np.interp(s, d, y)
    zg = np.interp(s, d, z)

    win = max(int(smooth_m / GRID_DS), 3)
    xs, ys = _smooth_closed(xg, win), _smooth_closed(yg, win)
    zs = _smooth_closed(zg, max(win * 2, 5))

    dx, dy = np.gradient(xs, GRID_DS), np.gradient(ys, GRID_DS)
    ddx, ddy = np.gradient(dx, GRID_DS), np.gradient(dy, GRID_DS)
    denom = np.power(dx * dx + dy * dy, 1.5)
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(denom, 1e-9)
    kappa = _smooth_closed(kappa, max(win // 2, 3))

    dz = np.gradient(zs, GRID_DS)
    grade = np.arctan(_smooth_closed(dz, max(win, 5)))

    has_elev = bool(np.nanstd(z) > 1.0)
    return CircuitGeometry(
        name=str(session.event["Location"]), length=length, s=s,
        xy=np.column_stack([xs - xs.mean(), ys - ys.mean()]),
        z=zs - zs.mean() if has_elev else np.zeros_like(zs),
        grade=grade if has_elev else np.zeros_like(grade),
        curvature=kappa, is_corner=kappa > CORNER_KAPPA,
        has_elevation=has_elev,
        source="fastf1-position")


class RealTrack:
    """Stage 1 `Track` interface, backed by real geometry.

    The estimator is written against this interface and does not know or care
    whether the circuit came from a synthetic segment list or a live position
    stream.
    """

    def __init__(self, geo: CircuitGeometry, config: dict | None = None):
        self.geo = geo
        self.name = geo.name
        self.length = geo.length
        self.cfg = config or {}
        self.aero_cda_ratio = float(self.cfg.get("aero_cda_ratio", 1.9))
        self._corner_seg = self._segment_corners()
        self.zones = self._build_zones()
        self.detection_points = [z["detection_point"] for z in self.zones]
        self.corners = [(a, b, v) for a, b, v in self._corner_speeds]
        self._tangent = None   # built lazily by tangent()

    # ------------------------------------------------------------- segments

    def _segment_corners(self):
        c = self.geo.is_corner.astype(np.int8)
        edges = np.diff(np.concatenate([[0], c, [0]]))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        segs = [(float(self.geo.s[a]), float(self.geo.s[min(b, len(self.geo.s) - 1)]))
                for a, b in zip(starts, ends) if (b - a) * GRID_DS > 25.0]
        self._corner_speeds = [(a, b, np.nan) for a, b in segs]
        return segs

    def set_corner_speeds(self, speeds: list[float]) -> None:
        """Apex speeds measured from the field, not assumed."""
        self._corner_speeds = [(a, b, v) for (a, b, _), v in
                               zip(self._corner_speeds, speeds)]
        self.corners = list(self._corner_speeds)

    def _build_zones(self):
        """Overtaking zones: the longest straights, each into its next corner."""
        straights = []
        prev_end = 0.0
        for a, b in self._corner_seg:
            if a - prev_end > 200.0:
                straights.append((prev_end, a))
            prev_end = b
        if self.length - prev_end > 200.0:
            straights.append((prev_end, self.length))
        straights.sort(key=lambda p: p[1] - p[0], reverse=True)
        zones = []
        sev = [1.0, 0.55, 0.2]
        for i, (a, b) in enumerate(straights[:3]):
            zones.append({
                "name": "ABC"[i], "s_straight_start": a, "s_straight_end": b,
                "s_end": min(b + 120.0, self.length),
                "braking_severity": sev[i],
                "detection_point": (a - 200.0) % self.length,
                "length": b - a,
            })
        return zones

    # -------------------------------------------------------------- lookups
    def _idx(self, s):
        n = len(self.geo.s)
        return np.clip((np.asarray(s, dtype=float) % self.length) / self.length * (n - 1),
                       0, n - 1).astype(int)

    def grade(self, s):
        out = self.geo.grade[self._idx(s)]
        return float(out) if np.ndim(out) == 0 else out

    def is_corner(self, s):
        out = self.geo.is_corner[self._idx(s)]
        return bool(out) if np.ndim(out) == 0 else out

    def curvature(self, s):
        out = self.geo.curvature[self._idx(s)]
        return float(out) if np.ndim(out) == 0 else out

    def cda_scale(self, s):
        return np.where(self.is_corner(s), self.aero_cda_ratio, 1.0)

    def v_limit(self, s):
        """Apex speed limit, measured from the field's own cornering speeds."""
        return self._v_limit[self._idx(s)]

    def set_v_limit_profile(self, profile: np.ndarray) -> None:
        self._v_limit = profile

    def zone_at(self, s):
        s = float(s) % self.length
        for z in self.zones:
            if z["s_straight_start"] <= s < z["s_end"]:
                return _Zone(z)
        return None

    def zone_by_name(self, name):
        return _Zone(next(z for z in self.zones if z["name"] == name))

    def xy(self):
        return self.geo.xy

    def xy_at(self, s):
        return self.geo.xy[self._idx(s)]

    def tangent(self, s):
        """Unit direction of travel at `s`, from the real centreline.

        Same `_idx` lookup as grade() and curvature(), so all three describe the
        same point of the lap. Derived from `geo.xy`, which is real position
        data in metres -- unlike Circuit Sigma's `xy()`, which is documented
        cosmetic and therefore has no tangent at all.

        The frame is whatever FastF1's position data uses. It is NOT oriented to
        true north, and nothing here pretends otherwise: projecting a
        meteorological wind bearing onto this needs an explicit
        `north_offset_deg`, which `environment.wind_parallel` demands and
        refuses to invent.
        """
        if self._tangent is None:
            xy = np.asarray(self.geo.xy, dtype=float)
            # Closed loop, so the seam is a wrap rather than an endpoint.
            d = np.gradient(np.vstack([xy, xy[:1]]), axis=0)[:-1]
            norm = np.linalg.norm(d, axis=1, keepdims=True)
            self._tangent = np.divide(d, norm, out=np.zeros_like(d),
                                      where=norm > 1e-12)
        out = self._tangent[self._idx(s)]
        return out


class _Zone:
    def __init__(self, d):
        self.__dict__.update(d)


def load_circuit_config(circuit: str) -> dict:
    p = CIRCUIT_DIR / f"{circuit.lower().replace(' ', '_')}.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text())
    return {}


def save_circuit_config(circuit: str, cfg: dict) -> Path:
    CIRCUIT_DIR.mkdir(parents=True, exist_ok=True)
    p = CIRCUIT_DIR / f"{circuit.lower().replace(' ', '_')}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p
