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
    centreline: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)


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


def _lap_frame(lap, grid: np.ndarray) -> tuple[pd.DataFrame | None, LapQuality]:
    drv = str(lap["Driver"])
    lap_no = int(lap["LapNumber"]) if not pd.isna(lap["LapNumber"]) else -1
    try:
        car = lap.get_car_data().add_distance()
    except Exception as exc:  # noqa: BLE001
        return None, LapQuality(drv, lap_no, 0, np.nan, np.nan, 0, 0.0, 0.0, False,
                                f"telemetry unavailable: {exc}")
    if len(car) < MIN_LAP_SAMPLES:
        return None, LapQuality(drv, lap_no, len(car), np.nan, np.nan, 0, 0.0, 0.0,
                                False, f"only {len(car)} samples")

    t = car["SessionTime"].dt.total_seconds().to_numpy().astype(float)
    d = car["Distance"].to_numpy().astype(float)
    v = car["Speed"].to_numpy().astype(float) * KMH

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
    for chan, name in (("Throttle", "throttle"), ("Brake", "brake"),
                       ("nGear", "gear"), ("RPM", "rpm")):
        if chan in car.columns:
            src = car[chan].to_numpy().astype(float)[order][keep]
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
    q = LapQuality(drv, lap_no, len(car), float(np.median(dt)), float(dt.max()),
                   gap_count, gap_seconds, frac, usable=bool(ok),
                   reason="" if ok else f"only {frac * 100:.0f}% of the lap observed")
    out["lap"] = lap_no
    out["driver"] = drv
    out["usable"] = q.usable
    return out, q


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
            f, q = _lap_frame(lap, grid)
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
