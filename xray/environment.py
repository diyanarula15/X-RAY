"""Causal weather state, and the wind projection that needs a frame it may not have.

Two things live here so that ingestion, the real-data fit, the simulator and the
zone calibrator cannot each grow their own copy:

1. `EnvironmentalState` / `EnvironmentTimeline` -- the weather at one instant and
   the time-resolved record it is read from. The read is causal by construction:
   `environment_at_time` takes the sample at or before the timestamp and never
   interpolates through a later one. A session mean is not a substitute; the
   FastF1 weather feed updates once per minute, so a race has ~90 samples and a
   mean silently imports the last lap's conditions into the first.

2. The wind convention, stated once. Every other module projects through
   `wind_parallel` rather than deciding for itself which way 90 degrees points.

Air density is NOT computed here. `xray.data.ingest.air_density` is the single
implementation and this module imports it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------- convention
# FastF1's `WindDirection` is the meteorological convention: the bearing the
# wind blows FROM, in degrees clockwise from north (0 = from the north, 90 =
# from the east). The velocity vector therefore points at bearing + 180.
#
# Track xy is expressed east/north: +x east, +y north. So for a wind from
# bearing b, the velocity components are
#
#     w_east  = -speed * sin(b)
#     w_north = -speed * cos(b)
#
# and the along-track component is the dot product with the unit tangent.
# Relative airspeed is then
#
#     v_air = v_car - w_parallel
#
# which gives the sign the physics needs: a headwind opposes travel, so
# w_parallel < 0 and |v_air| > |v_car|. Tested in test_environment.py rather
# than argued here.
WIND_CONVENTION = "meteorological_from_bearing_deg_cw_from_north"

# A wetness index is not a FastF1 channel. `Rainfall` is a bool; how wet the
# track actually is depends on how long it has been raining and how fast it
# dries, neither of which is published. These are the two coefficients of a
# deliberately crude memory, and they are ASSUMED -- see model_inventory.md.
ASSUMED_RAIN_GAIN_PER_S = 1.0 / 120.0     # full wet after ~2 min of rain
ASSUMED_DRYING_RATE_PER_S = 1.0 / 600.0   # dry after ~10 min at reference temp
ASSUMED_DRYING_REF_TRACK_TEMP_C = 30.0


@dataclass(frozen=True)
class EnvironmentalState:
    """Weather at one instant, with where it came from attached."""
    t: float
    air_temp_c: float | None
    track_temp_c: float | None
    pressure_mbar: float | None
    humidity_pct: float | None
    wind_speed_ms: float
    wind_dir_deg: float
    rainfall: bool | None
    rho: float
    track_wetness_index: float
    source: str
    age_s: float = 0.0


@dataclass(frozen=True)
class EnvironmentTimeline:
    """Time-resolved weather. Arrays are parallel and sorted by `t`."""
    t: np.ndarray
    air_temp_c: np.ndarray
    track_temp_c: np.ndarray
    pressure_mbar: np.ndarray
    humidity_pct: np.ndarray
    wind_speed_ms: np.ndarray
    wind_dir_deg: np.ndarray
    rainfall: np.ndarray
    rho: np.ndarray
    track_wetness_index: np.ndarray | None = None
    source: str = "session_weather_trace"

    def __len__(self) -> int:
        return int(len(self.t))


def _f(x, default=None):
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def timeline_from_samples(samples: list[dict], source: str = "session_weather_trace"
                          ) -> EnvironmentTimeline:
    """Build a timeline from already-parsed weather rows.

    Rows use this project's lower-case field names, not FastF1's column names;
    translation happens once, in `xray.data.ingest`.
    """
    rows = sorted(samples, key=lambda r: float(r["t"]))
    if not rows:
        raise ValueError("no weather samples")

    def col(key, default=np.nan):
        return np.array([_f(r.get(key), default) for r in rows], dtype=float)

    rain = np.array([bool(r.get("rainfall")) if r.get("rainfall") is not None
                     else False for r in rows], dtype=bool)
    return EnvironmentTimeline(
        t=np.array([float(r["t"]) for r in rows], dtype=float),
        air_temp_c=col("air_temp_c"), track_temp_c=col("track_temp_c"),
        pressure_mbar=col("pressure_mbar"), humidity_pct=col("humidity_pct"),
        wind_speed_ms=col("wind_speed_ms", 0.0), wind_dir_deg=col("wind_dir_deg", 0.0),
        rainfall=rain, rho=col("rho"), source=source)


def with_wetness(timeline: EnvironmentTimeline,
                 rain_gain_per_s: float = ASSUMED_RAIN_GAIN_PER_S,
                 drying_rate_per_s: float = ASSUMED_DRYING_RATE_PER_S,
                 ref_track_temp_c: float = ASSUMED_DRYING_REF_TRACK_TEMP_C
                 ) -> EnvironmentTimeline:
    """Integrate the assumed wetness memory along an existing timeline.

    Wetness is INFERRED. The only measured input is the `Rainfall` bool; the
    rest is this model. Anything reading `track_wetness_index` is reading a
    modelling assumption, which is why the state carries `source`.
    """
    n = len(timeline)
    wet = np.zeros(n, dtype=float)
    for i in range(1, n):
        dt = float(timeline.t[i] - timeline.t[i - 1])
        prev = float(wet[i - 1])
        if bool(timeline.rainfall[i]):
            wet[i] = min(1.0, prev + rain_gain_per_s * dt)
        else:
            # Warmer track dries faster. Bounded so a cold track still dries and
            # a hot one cannot dry in negative time.
            temp = timeline.track_temp_c[i]
            scale = 1.0 if not np.isfinite(temp) else float(
                np.clip(temp / max(ref_track_temp_c, 1.0), 0.25, 2.0))
            wet[i] = max(0.0, prev - drying_rate_per_s * scale * dt)
    return replace(timeline, track_wetness_index=wet)


def environment_at_time(timeline: EnvironmentTimeline, t: float,
                        max_age_s: float | None = None) -> EnvironmentalState:
    """The weather sample at or before `t`. Never a later one.

    Before the first sample there is nothing causal to return, so the first
    sample is used and `source` says so -- a caller that cares can read `age_s`,
    which is negative in exactly that case.
    """
    t = float(t)
    idx = int(np.searchsorted(timeline.t, t, side="right") - 1)
    source = timeline.source
    if idx < 0:
        idx = 0
        source = f"{timeline.source}:before_first_sample"
    age = t - float(timeline.t[idx])
    if max_age_s is not None and age > max_age_s:
        source = f"{timeline.source}:stale"
    wet = (0.0 if timeline.track_wetness_index is None
           else float(timeline.track_wetness_index[idx]))
    return EnvironmentalState(
        t=float(timeline.t[idx]),
        air_temp_c=_f(timeline.air_temp_c[idx]),
        track_temp_c=_f(timeline.track_temp_c[idx]),
        pressure_mbar=_f(timeline.pressure_mbar[idx]),
        humidity_pct=_f(timeline.humidity_pct[idx]),
        wind_speed_ms=_f(timeline.wind_speed_ms[idx], 0.0) or 0.0,
        wind_dir_deg=_f(timeline.wind_dir_deg[idx], 0.0) or 0.0,
        rainfall=bool(timeline.rainfall[idx]),
        rho=_f(timeline.rho[idx], 1.20) or 1.20,
        track_wetness_index=wet,
        source=source, age_s=age)


def state_from_summary(summary: dict, t: float = 0.0,
                       source: str = "session_summary_mean") -> EnvironmentalState:
    """Fallback state from the session-average weather dict.

    Explicitly labelled a mean. It is the right answer for a UI header and the
    wrong one for a decision, and the label is what lets a consumer tell.
    """
    return EnvironmentalState(
        t=float(t), air_temp_c=_f(summary.get("air_temp_c")),
        track_temp_c=_f(summary.get("track_temp_c")),
        pressure_mbar=_f(summary.get("pressure_mbar")),
        humidity_pct=_f(summary.get("humidity_pct")),
        wind_speed_ms=_f(summary.get("wind_speed_ms"), 0.0) or 0.0,
        wind_dir_deg=_f(summary.get("wind_dir_deg"), 0.0) or 0.0,
        rainfall=summary.get("rainfall"),
        rho=_f(summary.get("rho"), 1.20) or 1.20,
        track_wetness_index=0.0, source=source, age_s=float("nan"))


# ------------------------------------------------------------------- wind
#
# Two frames, and the whole point of this section is that they are not the same
# frame and nothing is allowed to pretend otherwise.
#
#   ENU      east/north, metres. True compass bearings live here.
#   local    CircuitGeometry.xy, metres. This is whatever frame the FastF1
#            position feed happened to use. Its axes are NOT east and north and
#            F1 publishes no rotation between the two.
#
# `TrackFrame.heading_deg_true_north` is the one number that connects them: the
# true compass bearing of the LOCAL +X AXIS. Without it a meteorological bearing
# cannot be projected onto a track tangent at all -- not approximately, not with
# a caveat. Guessing 0 would silently assert that the position feed happens to
# be north-aligned, and the failure mode is a sign flip: every headwind becomes
# a tailwind and the car is wrong in a way no test of magnitude would catch.


@dataclass(frozen=True)
class TrackFrame:
    """Orientation of the track-local Cartesian frame, and where it came from.

    `heading_deg_true_north` is the true compass bearing (0 = north, 90 = east,
    clockwise) of the local +X axis. `source` must say how it was established --
    a survey, a circuit map, a manual alignment -- because an undocumented
    orientation is indistinguishable from a guess.
    """
    heading_deg_true_north: float | None = None
    source: str = "unknown"

    @property
    def oriented(self) -> bool:
        return (self.heading_deg_true_north is not None
                and np.isfinite(self.heading_deg_true_north)
                and self.source not in ("", "unknown", "assumed"))


UNORIENTED = TrackFrame(None, "unknown")


@dataclass(frozen=True)
class WindProjection:
    """Along-track wind, or an explicit statement that there isn't one.

    `available` is the field to branch on. When it is False,
    `wind_along_track_mps` is None -- not 0.0 -- so that "we do not know" cannot
    be read as "there is no wind", and so a caller cannot accidentally infer a
    headwind or a tailwind from a missing orientation.
    """
    available: bool
    wind_along_track_mps: float | None
    source: str
    confidence: float

    @property
    def value_or_zero(self) -> float:
        """0.0 when unavailable -- for force models that need a number.

        Physically this is 'model no wind', which is the correct fallback, and
        it is deliberately a separate accessor so that choosing it is visible at
        the call site rather than implied by a default.
        """
        return 0.0 if not self.available else float(self.wind_along_track_mps)


UNAVAILABLE = WindProjection(False, None, "unavailable_no_physical_tangent", 0.0)
UNAVAILABLE_UNORIENTED = WindProjection(
    False, None, "unavailable_track_frame_heading_unknown", 0.0)


def wind_vector_enu(state: EnvironmentalState) -> np.ndarray:
    """Airflow velocity in ENU, m/s, as (east, north).

    `wind_dir_deg` is meteorological: the bearing the wind blows FROM. The air
    therefore travels toward bearing + 180, and a unit vector at bearing c is
    (sin c, cos c) in ENU, so the airflow is -(sin b, cos b) * speed.

        from north (0)   -> (0, -s)   blowing south
        from east  (90)  -> (-s, 0)   blowing west
        from south (180) -> (0, +s)   blowing north
        from west  (270) -> (+s, 0)   blowing east
    """
    b = np.deg2rad(float(state.wind_dir_deg))
    s = float(state.wind_speed_ms)
    return np.array([-s * np.sin(b), -s * np.cos(b)], dtype=float)


# Backwards-compatible alias: the older name said nothing about the frame.
wind_vector = wind_vector_enu


def tangent_to_enu(tangent_xy, frame: TrackFrame) -> np.ndarray:
    """Rotate a local-frame tangent into ENU using the frame heading.

    With theta the true bearing of local +X, local +X maps to ENU
    (sin t, cos t) and local +Y -- 90 degrees counter-clockwise from +X, which
    is 90 degrees ANTI-clockwise in bearing terms -- maps to (-cos t, sin t):

        E = x sin(t) - y cos(t)
        N = x cos(t) + y sin(t)
    """
    if not frame.oriented:
        raise ValueError("track frame heading is unknown; cannot rotate to ENU")
    t = np.deg2rad(float(frame.heading_deg_true_north))
    xy = np.asarray(tangent_xy, dtype=float)
    x, y = xy[..., 0], xy[..., 1]
    ct, st = np.cos(t), np.sin(t)
    return np.stack([x * st - y * ct, x * ct + y * st], axis=-1)


def project_wind_along_track(track, s, state: EnvironmentalState,
                             frame: TrackFrame = UNORIENTED) -> WindProjection:
    """wind_along_track = dot(wind_enu, tangent_enu), or an explicit refusal.

    Positive is a TAILWIND -- the air moves with the car -- because both vectors
    point in the direction of travel. Relative airspeed is then

        v_air = v_car - wind_along_track

    so a headwind (negative) raises it. Two independent things must hold before
    this is a physical number:

    1. the track must expose a real tangent. Circuit Sigma's `xy()` is
       documented cosmetic -- the heading it draws is an artist's impression of
       the lap -- so it has no `tangent` and lands on UNAVAILABLE.
    2. the local frame must be oriented, via `TrackFrame`. FastF1 position data
       is track-local with no published rotation to true north.

    Neither is inferred. A missing orientation returns available=False with a
    value of None, never a plausible-looking zero.
    """
    tangent = getattr(track, "tangent", None)
    if tangent is None:
        return UNAVAILABLE
    try:
        tan = np.asarray(tangent(s), dtype=float)
    except (NotImplementedError, AttributeError, ValueError):
        return UNAVAILABLE
    if tan.size == 0 or not np.all(np.isfinite(tan)):
        return UNAVAILABLE
    if not frame.oriented:
        return UNAVAILABLE_UNORIENTED

    tan_enu = tangent_to_enu(tan, frame)
    w = wind_vector_enu(state)
    along = np.sum(tan_enu * w, axis=-1)
    value = float(along) if np.ndim(along) == 0 else along
    conf = 0.6 if state.source.endswith("stale") else 0.8
    return WindProjection(True, value, "track_tangent_enu_projection", conf)


def relative_airspeed(v_car, wind_along_track_mps: float = 0.0):
    """v_air = v_car - wind_along_track. One definition, imported everywhere.

    NOTE the sign relative to `realfit`, which historically writes the same
    physics as (v + v_wind_hat) and therefore means the OPPOSITE sign by
    `v_wind_hat`. See realfit._terms; the two are reconciled there explicitly
    rather than by whichever module is read first.
    """
    return np.asarray(v_car, dtype=float) - float(wind_along_track_mps)
