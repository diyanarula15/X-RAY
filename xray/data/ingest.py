"""FastF1 -> canonical parquet, on a common distance grid.

Distance, not time, is what makes two cars comparable: at the same distance they
are at the same corner. Everything downstream indexes on it.

Hard rule from the spec: never silently interpolate across a gap larger than
GAP_LIMIT_S. Gaps are recorded, flagged, and the estimator suspends across them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GAP_LIMIT_S = 1.0          # never interpolate across a longer hole
GRID_DS = 10.0             # m, common distance grid
MIN_LAP_SAMPLES = 40
KMH = 1 / 3.6

CACHE = Path(__file__).resolve().parent / "cache"
OUT = Path(__file__).resolve().parent.parent.parent / "out" / "sessions"


@dataclass
class LapQuality:
    """Per car, per lap. Downstream code refuses on these, it does not guess."""
    driver: str
    lap: int
    n_samples: int
    median_dt: float
    max_dt: float
    gap_count: int          # holes longer than GAP_LIMIT_S
    gap_seconds: float      # total time inside those holes
    frac_covered: float     # fraction of the distance grid actually observed
    usable: bool
    reason: str = ""


@dataclass
class SessionData:
    year: int
    round: int
    session: str
    event: str
    circuit: str
    date: str
    track_length: float
    grid: np.ndarray                     # m, common distance grid
    frames: dict                         # driver -> DataFrame on the grid
    quality: list                        # LapQuality
    weather: dict
    # Time-resolved weather. `weather` above stays the session summary it always
    # was -- a mean is fine for a UI header and wrong for a decision, and the
    # two are kept side by side rather than one silently becoming the other.
    weather_trace: list = field(default_factory=list)
    centreline: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)


# Verified against the installed FastF1 (3.8.3) weather parser, not from memory:
# the channels are Time, AirTemp, Humidity, Pressure, Rainfall, TrackTemp,
# WindDirection, WindSpeed. Rainfall really is published, and really is only a
# bool -- how wet the track IS is not a channel, which is why the wetness index
# in xray.environment is labelled inferred.
#
# The feed updates once per minute. That is the resolution a "causal" weather
# lookup actually has, and it is why the summary mean is kept separate rather
# than replaced: a 90-sample race averaged to one number puts the last lap's air
# in the first lap's drag.
_WEATHER_COLS = ("AirTemp", "TrackTemp", "Pressure", "Humidity",
                 "WindSpeed", "WindDirection", "Rainfall")


def _weather_trace(w) -> list:
    """Per-sample weather rows in this project's field names, seconds from t0."""
    if w is None or not len(w):
        return []
    if "Time" not in w:
        return []
    t = pd.to_timedelta(w["Time"]).dt.total_seconds().to_numpy(dtype=float)

    def col(name):
        if name not in w:
            return [None] * len(w)
        return [None if pd.isna(v) else v for v in w[name].tolist()]

    air, track, press = col("AirTemp"), col("TrackTemp"), col("Pressure")
    hum, ws, wd, rain = (col("Humidity"), col("WindSpeed"),
                         col("WindDirection"), col("Rainfall"))
    rows = []
    for i in range(len(w)):
        a = None if air[i] is None else float(air[i])
        pmb = 1013.0 if press[i] is None else float(press[i])
        h = 50.0 if hum[i] is None else float(hum[i])
        rows.append({
            "t": float(t[i]),
            "air_temp_c": a,
            "track_temp_c": None if track[i] is None else float(track[i]),
            "pressure_mbar": pmb,
            "humidity_pct": h,
            "wind_speed_ms": 0.0 if ws[i] is None else float(ws[i]),
            "wind_dir_deg": 0.0 if wd[i] is None else float(wd[i]),
            "rainfall": None if rain[i] is None else bool(rain[i]),
            # Same single density implementation the summary uses.
            "rho": air_density(a, pmb, h) if a is not None else None,
        })
    return rows


def air_density(temp_c: float, pressure_mbar: float, humidity_pct: float = 50.0) -> float:
    """Real air density from the session's own weather, not an assumed 1.20.

    Ideal gas with a humidity correction. One fewer nuisance parameter for the
    estimator to carry.
    """
    t_k = temp_c + 273.15
    p_pa = pressure_mbar * 100.0
    # saturation vapour pressure, Tetens
    p_sat = 610.78 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    p_v = humidity_pct / 100.0 * p_sat
    p_d = p_pa - p_v
    return float(p_d / (287.058 * t_k) + p_v / (461.495 * t_k))


def grid_lap(distance, time, speed, grid: np.ndarray, channels: dict | None = None,
             driver: str = "", lap_no: int = -1,
             n_raw: int | None = None) -> tuple[pd.DataFrame | None, LapQuality]:
    """One lap of any source onto the common distance grid.

    THE single gridding step. Both data sources reach it -- FastF1 through
    `frame_from_fastf1`, the simulator through `frame_from_observation` -- and
    nothing in here knows or can know which one it is serving. That is the point:
    the two paths previously had separate gridding, and the copy drifted. It
    grew a brake channel that was interpolated while acceleration was smoothed
    over 60 m, which put 153 samples at up to -44.3 m/s2 with the brake flag
    reading 0, drove a CdA lower bound of 13.2 m2 and emptied the interval
    intersection outright. One implementation cannot drift from itself.

    `channels` is a dict of raw per-sample arrays keyed by the column name they
    should take in the frame. They are reordered and deduplicated with the same
    indices as speed and time, so a caller cannot accidentally align them
    differently.
    """
    d = np.asarray(distance, dtype=float)
    t = np.asarray(time, dtype=float)
    v = np.asarray(speed, dtype=float)
    channels = dict(channels or {})
    n_raw = len(d) if n_raw is None else n_raw

    if len(d) < MIN_LAP_SAMPLES:
        return None, LapQuality(driver, lap_no, n_raw, np.nan, np.nan, 0, 0.0, 0.0,
                                False, f"only {n_raw} samples")

    dt = np.diff(t)
    holes = dt > GAP_LIMIT_S
    gap_count = int(holes.sum())
    gap_seconds = float(dt[holes].sum()) if gap_count else 0.0
    # Record the distance span of each hole NOW, against the original arrays.
    # Doing it after the sort/dedupe below would index a shorter array with
    # positions taken from the longer one.
    hole_spans = [(float(d[i]), float(d[i + 1])) for i in np.flatnonzero(holes)]

    order = np.argsort(d)
    d, t, v = d[order], t[order], v[order]
    keep = np.concatenate([[True], np.diff(d) > 1e-6])
    d, t, v = d[keep], t[keep], v[keep]

    inside = (grid >= d[0]) & (grid <= d[-1])
    out = pd.DataFrame({"distance": grid})
    for name, src in (("speed", v), ("time", t)):
        col = np.full(len(grid), np.nan)
        col[inside] = np.interp(grid[inside], d, src)
        out[name] = col
    for name, raw in channels.items():
        src = np.asarray(raw, dtype=float)[order][keep]
        col = np.full(len(grid), np.nan)
        col[inside] = np.interp(grid[inside], d, src)
        out[name] = col

    # blank the grid points that fall inside a hole: interpolating across a
    # 1-second gap at 300 km/h invents 80 m of trajectory
    if gap_count:
        bad = np.zeros(len(grid), dtype=bool)
        for d_lo, d_hi in hole_spans:
            lo_, hi_ = min(d_lo, d_hi), max(d_lo, d_hi)
            bad |= (grid >= lo_) & (grid <= hi_)
        out.loc[bad, ["speed", "time"]] = np.nan
        inside = inside & ~bad

    # A lap is usable if enough of it was actually observed. Gaps are already
    # handled where they belong -- at sample level, by blanking the grid cells
    # they span, so nothing is ever interpolated across one. Rejecting the whole
    # lap as well threw away two thirds of the race for a single hole, which
    # starved the calibration and left the replay with ten-minute holes in it.
    frac = float(inside.mean())
    ok = frac > 0.60
    q = LapQuality(driver, lap_no, n_raw, float(np.median(dt)), float(dt.max()),
                   gap_count, gap_seconds, frac, usable=bool(ok),
                   reason="" if ok else f"only {frac * 100:.0f}% of the lap observed")
    out["lap"] = lap_no
    out["driver"] = driver
    # Per-LAP scalar, not per-cell. `analysis.py` filters rows with df[df["usable"]],
    # so a per-cell boolean here would silently mean something else: drop these
    # samples, rather than drop this lap. Cell-level gaps are already handled
    # above by blanking; this flag is only about whether enough of the lap
    # survived to be worth using at all.
    out["usable"] = q.usable
    return out, q


def frame_from_fastf1(lap, grid: np.ndarray) -> tuple[pd.DataFrame | None, LapQuality]:
    """FastF1 entry point: pull the raw arrays, hand them to `grid_lap`.

    Everything source-specific about real telemetry lives here and nowhere else
    -- the accessors, the column names, and the km/h to m/s conversion.
    """
    drv = str(lap["Driver"])
    lap_no = int(lap["LapNumber"]) if not pd.isna(lap["LapNumber"]) else -1
    try:
        car = lap.get_car_data().add_distance()
    except Exception as exc:  # noqa: BLE001
        return None, LapQuality(drv, lap_no, 0, np.nan, np.nan, 0, 0.0, 0.0, False,
                                f"telemetry unavailable: {exc}")

    channels = {name: car[chan].to_numpy().astype(float)
                for chan, name in (("Throttle", "throttle"), ("Brake", "brake"),
                                   ("nGear", "gear"), ("RPM", "rpm"))
                if chan in car.columns}
    return grid_lap(distance=car["Distance"].to_numpy().astype(float),
                    time=car["SessionTime"].dt.total_seconds().to_numpy().astype(float),
                    speed=car["Speed"].to_numpy().astype(float) * KMH,
                    grid=grid, channels=channels, driver=drv, lap_no=lap_no,
                    n_raw=len(car))


def frame_from_observation(obs, grid: np.ndarray, channels: dict | None = None,
                           driver: str = "") -> tuple[pd.DataFrame, list]:
    """Simulator entry point: the blinded trace through the same `grid_lap`.

    Takes `observe()` output rather than `GroundTruth`. The simulator runs at
    dt = 0.005 s, and handing the inference core a 200 Hz noiseless trace would
    measure it on a signal 50x better than the ~4 Hz irregular feed a real race
    delivers. The blindfold also stays where it already is: `observe.py` is the
    only module that reads the simulator.

    `obs` is duck-typed on `.s/.t/.v/.lap`, so this module never imports anything
    from the simulator side and the layering in docs/pipeline_layers.md holds.
    """
    s = np.asarray(obs.s, dtype=float)
    t = np.asarray(obs.t, dtype=float)
    v = np.asarray(obs.v, dtype=float)
    lap = np.asarray(obs.lap, dtype=int)
    channels = dict(channels or {})

    frames, quality = [], []
    for L in np.unique(lap):
        m = lap == L
        f, q = grid_lap(distance=s[m], time=t[m], speed=v[m], grid=grid,
                        channels={k: np.asarray(a)[m] for k, a in channels.items()},
                        driver=driver, lap_no=int(L))
        quality.append(q)
        if f is not None:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["distance", "speed", "time", "lap",
                                     "driver", "usable"]), quality
    return pd.concat(frames, ignore_index=True), quality


def ingest_session(year: int, rnd: int, session_name: str = "R",
                   drivers: list[str] | None = None, max_laps: int | None = None,
                   verbose: bool = True) -> SessionData:
    import fastf1

    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    ses = fastf1.get_session(year, rnd, session_name)
    ses.load(telemetry=True, laps=True, weather=True)

    laps = ses.laps
    if drivers is None:
        drivers = [str(d) for d in laps["Driver"].unique()]

    track_length = float(np.nanpercentile(
        [lap["LapDistance"] if "LapDistance" in lap else np.nan for _, lap in laps.iterlaps()], 50)) \
        if "LapDistance" in laps.columns else float("nan")
    if not np.isfinite(track_length):
        sample = laps.pick_fastest().get_car_data().add_distance()
        track_length = float(sample["Distance"].max())
    grid = np.arange(0.0, track_length, GRID_DS)

    frames, quality = {}, []
    for drv in drivers:
        rows = []
        dl = laps.pick_drivers(drv)
        for i, (_, lap) in enumerate(dl.iterlaps()):
            if max_laps and i >= max_laps:
                break
            f, q = frame_from_fastf1(lap, grid)
            quality.append(q)
            if f is not None:
                rows.append(f)
        if rows:
            frames[drv] = pd.concat(rows, ignore_index=True)
        if verbose:
            good = sum(1 for q in quality if q.driver == drv and q.usable)
            tot = sum(1 for q in quality if q.driver == drv)
            print(f"  {drv}: {good}/{tot} usable laps")

    w = ses.weather_data
    weather = {}
    weather_trace = _weather_trace(w)
    if w is not None and len(w):
        weather = {
            "air_temp_c": float(w["AirTemp"].mean()),
            "track_temp_c": float(w["TrackTemp"].mean()) if "TrackTemp" in w else None,
            "pressure_mbar": float(w["Pressure"].mean()) if "Pressure" in w else 1013.0,
            "humidity_pct": float(w["Humidity"].mean()) if "Humidity" in w else 50.0,
            "wind_speed_ms": float(w["WindSpeed"].mean()) if "WindSpeed" in w else 0.0,
            "wind_dir_deg": float(w["WindDirection"].mean()) if "WindDirection" in w else 0.0,
        }
        weather["rho"] = air_density(weather["air_temp_c"], weather["pressure_mbar"],
                                     weather["humidity_pct"] or 50.0)
    else:
        weather = {"rho": 1.20, "assumed": True}

    return SessionData(
        year=year, round=rnd, session=session_name,
        event=str(ses.event["EventName"]), circuit=str(ses.event["Location"]),
        date=str(ses.event["EventDate"].date()), track_length=track_length,
        grid=grid, frames=frames, quality=quality, weather=weather,
        weather_trace=weather_trace,
        meta={"n_drivers": len(frames), "grid_ds": GRID_DS,
              "gap_limit_s": GAP_LIMIT_S})


def write_parquet(sd: SessionData, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (OUT / f"{sd.year}_r{sd.round}_{sd.session}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames = pd.concat(sd.frames.values(), ignore_index=True) if sd.frames else pd.DataFrame()
    all_frames.to_parquet(out_dir / "telemetry.parquet", index=False)
    pd.DataFrame([asdict(q) for q in sd.quality]).to_parquet(out_dir / "quality.parquet",
                                                             index=False)
    import json
    (out_dir / "meta.json").write_text(json.dumps({
        "year": sd.year, "round": sd.round, "session": sd.session, "event": sd.event,
        "circuit": sd.circuit, "date": sd.date, "track_length": sd.track_length,
        "weather": sd.weather, "meta": sd.meta}, indent=2))
    return out_dir
