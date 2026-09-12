"""Exact re-inference of the canonical estimator at a historical cutoff.

P3 could not measure nuisance → energy sensitivity because the persisted race
payloads keep only estimator *outputs* — `usable_mean`, `deploy_kw`, the particle
cloud — and not the raw throttle/brake the canonical front-end consumes. So the
strongest thing P3 could say was how a *downstream prediction* moved while the
inferred energy was held fixed. That measures the wrong derivative: it answers
"if the energy were this, what would we predict", not "if the physics were this,
what energy would we infer".

This module closes that gap. An `EstimatorInputWindow` holds exactly the inputs
`build_kin` and the canonical estimator read, truncated at a causal cutoff, and
`reinfer()` runs THE SAME functions over it — `build_kin`, `fit_nuisance_real`,
`deployment_trace`, `belief_from_deployment`, imported from `realfit`, never
reimplemented. Nuisance overrides are arguments to those calls.

Two boundaries are enforced rather than documented:

* **No future data in the estimator namespace.** A window stores samples at or
  before `cutoff_time_s` and nothing else. Evaluation targets live in a separate
  `EvaluationTargets` object written to a separate file, because a target sitting
  in the same dict as the inputs is one refactor away from being read as one.
* **No second loader.** Windows are built from `data/ingest.ingest_session` —
  the existing path, served from the existing FastF1 cache — and the track
  quantities are replayed through `_WindowTrack`, which returns the numbers the
  live `RealTrack` produced at build time and computes nothing of its own. The
  deployment-zone mask is deliberately NOT stored: the window keeps the zone
  geometry and `realfit.deployment_zone_mask` recomputes it, so the canonical
  code stays the only definition.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .realfit import (ASSUMED_RACE_FUEL_START_KG, BRAKE_DECEL, COAST_MAX_DECEL,
                      COAST_MIN_DECEL, COAST_THROTTLE, RESERVE_SIGMA_REAL,
                      belief_from_deployment, build_kin, deployment_trace,
                      fit_nuisance_real)
from .regs import RegSet, regs_for

SCHEMA_VERSION = "p35-estimator-input-window-v1"

# The channels `build_kin` actually reads off the gridded frame. Storing more
# would invite a future reader to use a channel the estimator does not consume
# and call the result a re-inference.
FRAME_CHANNELS = ("distance", "speed", "time", "throttle", "brake", "lap")

# Per-sample track quantities `build_kin` asks the track for. `grade` is stored
# in RADIANS, as the interface returns it -- `build_kin` takes the sine itself,
# and storing the sine would silently change what an override of grade means.
TRACK_CHANNELS = ("cda_scale", "grade")


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


def _fp_arrays(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[k], dtype=np.float64))
        h.update(k.encode())
        h.update(np.nan_to_num(a, nan=-9.87e18).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class EstimatorAssumptions:
    """Every scalar the canonical estimator takes as given rather than inferring.

    These are the override surface for sensitivity work. They are stored with the
    window so a rerun a year from now uses the assumptions the original run used,
    not whatever the module defaults have drifted to.
    """
    crr: float = 0.012
    eta: float = 0.95
    smooth_m: float = 60.0
    mass_kg: float = 790.0
    rho: float = 1.20
    n_particles: int = 400
    particle_seed: int = 0
    reserve_max_frac: float = 0.35
    robust_q: float = 0.02
    wind_grid_lo: float = -4.0
    wind_grid_hi: float = 4.0
    wind_grid_step: float = 0.5
    # Read from `realfit` rather than restated, so a change there cannot leave a
    # window describing thresholds the estimator no longer uses.
    coast_throttle: float = COAST_THROTTLE
    coast_min_decel: float = COAST_MIN_DECEL
    coast_max_decel: float = COAST_MAX_DECEL
    brake_decel: float = BRAKE_DECEL
    reserve_sigma_j: float = RESERVE_SIGMA_REAL
    assumed_race_fuel_start_kg: float = ASSUMED_RACE_FUEL_START_KG
    # P3.5 Part 2 hardening switches. Defaults reproduce the estimator that
    # produced the P3 artifacts exactly, so those numbers stay reproducible while
    # the hardened path is evaluated; the activation gate decides the default.
    centre: str = "midpoint"        # "midpoint" | "evidence"
    boundary: str = "clip"          # "clip" | "truncated" | "conditional"
    reserve_obs: str = "point"      # "point" | "one_sided"

    def wind_grid(self) -> np.ndarray:
        return np.arange(self.wind_grid_lo, self.wind_grid_hi + 1e-9,
                         self.wind_grid_step)

    def replace(self, **kw) -> "EstimatorAssumptions":
        import dataclasses
        return dataclasses.replace(self, **kw)


class _WindowTrack:
    """Replays the track quantities the live `RealTrack` produced. No model.

    `cda_scale` and `grade` are served by nearest-sample lookup on the stored
    distance axis, which is exact for the samples in the window because those are
    the distances they were sampled at. `zones` and `length` are passed straight
    through so `realfit.deployment_zone_mask` -- the canonical definition --
    recomputes eligibility instead of trusting a stored mask.

    A test asserts a Kin built through this view is bit-identical to one built
    through the live track, which is what makes "replay" a measurement rather
    than a claim.
    """

    def __init__(self, s: np.ndarray, cda_scale: np.ndarray, grade: np.ndarray,
                 zones: list, length: float):
        self._s = np.asarray(s, dtype=float)
        self._cda = np.asarray(cda_scale, dtype=float)
        self._grade = np.asarray(grade, dtype=float)
        self.zones = zones
        self.length = float(length)
        order = np.argsort(self._s)
        self._os = self._s[order]
        self._oc = self._cda[order]
        self._og = self._grade[order]

    def _lookup(self, s, table):
        q = np.asarray(s, dtype=float)
        i = np.clip(np.searchsorted(self._os, q), 0, len(self._os) - 1)
        j = np.clip(i - 1, 0, len(self._os) - 1)
        pick = np.where(np.abs(self._os[i] - q) <= np.abs(self._os[j] - q), i, j)
        return table[pick]

    def cda_scale(self, s):
        return self._lookup(s, self._oc)

    def grade(self, s):
        return self._lookup(s, self._og)


@dataclass(frozen=True)
class EvaluationTargets:
    """Later-than-cutoff observables. Deliberately NOT part of the window.

    Kept in its own object and its own file so that no code path can reach a
    future sample while holding only estimator inputs. The label is carried in
    the data, not only in a comment.
    """
    window_id: str
    cutoff_time_s: float
    t: np.ndarray
    v: np.ndarray
    s: np.ndarray
    lap: np.ndarray
    label: str = "EVALUATION ONLY — NEVER AN ESTIMATOR INPUT"


@dataclass(frozen=True)
class EstimatorInputWindow:
    """Everything needed to rerun the canonical estimator, and nothing more."""
    window_id: str
    race_id: str
    year: int
    round: int
    session: str
    driver: str
    cutoff_time_s: float

    frame: dict                      # FRAME_CHANNELS -> arrays
    track: dict                      # TRACK_CHANNELS -> arrays
    zones: list
    track_length: float

    regulation_variant: str
    regulation: dict                 # the RegSet fields actually used
    assumptions: EstimatorAssumptions

    rho_series: np.ndarray | None = None
    rho_source: str = "scalar"
    w_along_mps: np.ndarray | None = None
    wind_source: str = "unavailable"

    schema_version: str = SCHEMA_VERSION
    source: str = "xray.data.ingest.ingest_session"
    source_fingerprint: str = ""
    notes: tuple[str, ...] = ()

    @property
    def n_samples(self) -> int:
        return int(len(self.frame["time"]))

    def regs(self) -> RegSet:
        return RegSet(**self.regulation)

    def frame_df(self) -> pd.DataFrame:
        return pd.DataFrame({k: np.asarray(v) for k, v in self.frame.items()})

    def window_track(self) -> _WindowTrack:
        return _WindowTrack(self.frame["distance"], self.track["cda_scale"],
                            self.track["grade"], self.zones, self.track_length)

    def provenance(self) -> dict:
        return {"schema_version": self.schema_version, "window_id": self.window_id,
                "race_id": self.race_id, "driver": self.driver,
                "cutoff_time_s": self.cutoff_time_s, "n_samples": self.n_samples,
                "regulation_variant": self.regulation_variant,
                "rho_source": self.rho_source, "wind_source": self.wind_source,
                "source": self.source,
                "source_fingerprint": self.source_fingerprint,
                "assumptions": asdict(self.assumptions),
                "notes": list(self.notes)}


# ------------------------------------------------------------------- building
def build_windows(sd, track, driver: str, cutoffs: list[float],
                  assumptions: EstimatorAssumptions | None = None,
                  target_horizon_s: float = 5.0
                  ) -> list[tuple[EstimatorInputWindow, EvaluationTargets]]:
    """Cut one ingested session into causal estimator-input windows.

    `sd` is a `SessionData` from the existing `ingest_session`; `track` the live
    `RealTrack`. Nothing is re-downloaded and nothing is re-gridded: the window is
    a causal slice of the frame the canonical path already built.
    """
    a = assumptions or EstimatorAssumptions()
    df = sd.frames[driver]
    df = df[df["usable"]] if "usable" in df else df
    rho = float(sd.weather.get("rho", a.rho))
    a = a.replace(rho=rho)
    regs = regs_for(sd.date)
    race_id = f"{sd.year}_r{sd.round}_{sd.session}"

    t_all = df["time"].to_numpy(dtype=float)
    out = []
    for cutoff in cutoffs:
        pre = df[t_all <= float(cutoff)]
        post = df[(t_all > float(cutoff))
                  & (t_all <= float(cutoff) + float(target_horizon_s))]
        if len(pre) < 500:
            continue
        s_pre = pre["distance"].to_numpy(dtype=float)
        frame = {c: (pre[c].to_numpy(dtype=float) if c in pre
                     else np.full(len(pre), np.nan)) for c in FRAME_CHANNELS}
        tr = {"cda_scale": np.asarray(track.cda_scale(s_pre), dtype=float),
              "grade": np.asarray(track.grade(s_pre), dtype=float)}
        # Zones arrive as dicts from RealTrack and as objects from the synthetic
        # Track. `deployment_zone_mask` already accepts both, so the window
        # normalises to dicts rather than assuming either shape.
        zones = []
        for z in (getattr(track, "zones", None) or []):
            g = (lambda k: z[k]) if isinstance(z, dict) else (lambda k: getattr(z, k))
            zones.append({"name": (z.get("name") if isinstance(z, dict)
                                   else getattr(z, "name", None)),
                          "s_straight_start": float(g("s_straight_start")),
                          "s_straight_end": float(g("s_straight_end"))})
        wid = f"{race_id}:{driver}:{float(cutoff):.3f}"
        src_fp = _fp_arrays({**frame, **tr})
        w = EstimatorInputWindow(
            window_id=wid, race_id=race_id, year=int(sd.year),
            round=int(sd.round), session=str(sd.session), driver=str(driver),
            cutoff_time_s=float(cutoff), frame=frame, track=tr, zones=zones,
            track_length=float(getattr(track, "length", np.nan)),
            regulation_variant=regs.variant, regulation=asdict(regs),
            assumptions=a, rho_source="scalar",
            source_fingerprint=src_fp,
            notes=("estimator inputs only; every sample has time <= cutoff",
                   "deployment-zone eligibility is NOT stored: zone geometry is, "
                   "and realfit.deployment_zone_mask recomputes the mask"))
        tgt = EvaluationTargets(
            window_id=wid, cutoff_time_s=float(cutoff),
            t=post["time"].to_numpy(dtype=float),
            v=post["speed"].to_numpy(dtype=float),
            s=post["distance"].to_numpy(dtype=float),
            lap=post["lap"].to_numpy(dtype=float))
        out.append((w, tgt))
    return out


# ------------------------------------------------------------ re-inference
@dataclass(frozen=True)
class ReinferenceResult:
    """One run of the canonical estimator. Semantics named, not blurred.

    `cda_lo/hi` is an identified SET, `usable_*_p10/p90` are particle QUANTILES.
    Neither is a credible interval, and the field names and this docstring are
    the only places that can say so before a caller plots them as one.
    """
    window_id: str
    overrides: dict
    cda_hat: float
    cda_lo: float
    cda_hi: float
    cda_set_width: float
    identifiability: float
    n_binding: int
    n_coast: int
    v_wind_hat: float
    residual_rms: float
    balance: float
    usable_mean_j: float             # belief at the LAST sample in the window
    usable_p10_j: float
    usable_p90_j: float
    usable_particle_width_j: float
    deployed_last_lap_j: float
    deploy_mean_w: float
    # Whole-window floor behaviour, not just the last sample. The last-sample
    # reading alone made a window look fully degenerate when 20.5% of its samples
    # were.
    frac_reported_empty: float = float("nan")
    frac_band_degenerate: float = float("nan")
    mean_band_width_j: float = float("nan")
    notes: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return np.isfinite(self.cda_hat)


def _floor_stats(um, p10, p90) -> dict:
    """Window-level floor statistics, computed once so every result carries them."""
    from .estimator_diag import DEGENERATE_WIDTH_J, EMPTY_ENERGY_J
    m = np.isfinite(um) & np.isfinite(p10) & np.isfinite(p90)
    if not m.any():
        return {"frac_reported_empty": float("nan"),
                "frac_band_degenerate": float("nan"),
                "mean_band_width_j": float("nan")}
    w = (p90 - p10)[m]
    return {"frac_reported_empty": float(np.mean(um[m] < EMPTY_ENERGY_J)),
            "frac_band_degenerate": float(np.mean(w < DEGENERATE_WIDTH_J)),
            "mean_band_width_j": float(np.mean(w))}


def reinfer(window: EstimatorInputWindow, **overrides) -> ReinferenceResult:
    """Rerun THE canonical estimator over a window, with nuisance overrides.

    Overrides name fields of `EstimatorAssumptions`. Everything numerical below
    happens inside `realfit`; this function only routes arguments, which is what
    keeps it a re-inference rather than a second estimator.
    """
    a = window.assumptions.replace(**overrides) if overrides else window.assumptions
    regs = window.regs()
    kin = build_kin(window.frame_df(), window.window_track(), a.mass_kg, a.rho,
                    smooth_m=a.smooth_m,
                    w_along_mps=window.w_along_mps, wind_source=window.wind_source,
                    rho_series=window.rho_series, rho_source=window.rho_source,
                    regs=regs)
    notes = []
    try:
        fit = fit_nuisance_real(kin, a.rho, crr=a.crr, eta=a.eta,
                                wind_grid=a.wind_grid(), robust_q=a.robust_q,
                                regs=regs)
    except ValueError as exc:
        return ReinferenceResult(
            window.window_id, dict(overrides), *([float("nan")] * 3),
            float("nan"), 0.0, 0, 0, float("nan"), float("nan"), float("nan"),
            *([float("nan")] * 5), float("nan"), float("nan"), float("nan"),
            notes=(f"calibration refused: {exc}",))

    tr = deployment_trace(kin, fit, crr=a.crr, eta=a.eta, regs=regs,
                          centre=a.centre)
    bel = belief_from_deployment(kin, tr, fit, n_particles=a.n_particles,
                                 seed=a.particle_seed,
                                 reserve_max_frac=a.reserve_max_frac,
                                 boundary=a.boundary, reserve_obs=a.reserve_obs,
                                 reserve_sigma_j=a.reserve_sigma_j)

    um = np.asarray(bel["usable_mean"], dtype=float)
    p10 = np.asarray(bel["usable_p10"], dtype=float)
    p90 = np.asarray(bel["usable_p90"], dtype=float)
    fin = np.flatnonzero(np.isfinite(um))
    k = int(fin[-1]) if len(fin) else -1
    dl = bel.get("deployed_lap") or {}
    last_lap = max(dl) if dl else None
    dep = np.asarray(tr["deploy"], dtype=float)

    return ReinferenceResult(
        window_id=window.window_id, overrides=dict(overrides),
        cda_hat=float(fit.cda_hat), cda_lo=float(fit.cda_lo),
        cda_hi=float(fit.cda_hi),
        cda_set_width=float(fit.cda_hi - fit.cda_lo),
        identifiability=float(fit.identifiability),
        n_binding=int(fit.n_binding), n_coast=int(fit.n_coast),
        v_wind_hat=float(fit.v_wind_hat),
        residual_rms=float(fit.residual_rms),
        balance=float(bel.get("balance", float("nan"))),
        usable_mean_j=float(um[k]) if k >= 0 else float("nan"),
        usable_p10_j=float(p10[k]) if k >= 0 else float("nan"),
        usable_p90_j=float(p90[k]) if k >= 0 else float("nan"),
        usable_particle_width_j=(float(p90[k] - p10[k]) if k >= 0 else float("nan")),
        deployed_last_lap_j=float(dl.get(last_lap, float("nan")))
        if last_lap is not None else float("nan"),
        **_floor_stats(um, p10, p90),
        deploy_mean_w=float(np.nanmean(dep)) if np.isfinite(dep).any() else float("nan"),
        notes=tuple(notes))


# ----------------------------------------------------------------- storage
def save_window(window: EstimatorInputWindow, targets: EvaluationTargets | None,
                out_dir: Path) -> dict:
    """Inputs and targets to SEPARATE files, by design.

    One file with an `inputs` key and a `targets` key would be smaller and would
    also be the kind of structure where a future loader reads both and nobody
    notices. The separation is the safety property.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = window.window_id.replace(":", "_").replace(".", "p")
    npz = out_dir / f"{stem}.inputs.npz"
    arrays = {f"frame__{k}": np.asarray(v) for k, v in window.frame.items()}
    arrays.update({f"track__{k}": np.asarray(v) for k, v in window.track.items()})
    if window.rho_series is not None:
        arrays["rho_series"] = np.asarray(window.rho_series)
    if window.w_along_mps is not None:
        arrays["w_along_mps"] = np.asarray(window.w_along_mps)
    np.savez_compressed(npz, **arrays)

    meta = {k: v for k, v in asdict(window).items()
            if k not in ("frame", "track", "rho_series", "w_along_mps")}
    meta["assumptions"] = asdict(window.assumptions)
    meta["arrays_file"] = npz.name
    meta["arrays_fingerprint"] = _fp_arrays(arrays)
    meta_path = out_dir / f"{stem}.inputs.json"
    meta_path.write_text(json.dumps(meta, indent=1, default=str))

    written = {"inputs_meta": str(meta_path), "inputs_arrays": str(npz),
               "arrays_fingerprint": meta["arrays_fingerprint"]}
    if targets is not None:
        tp = out_dir / f"{stem}.targets.npz"
        np.savez_compressed(tp, t=targets.t, v=targets.v, s=targets.s,
                            lap=targets.lap)
        (out_dir / f"{stem}.targets.json").write_text(json.dumps(
            {"window_id": targets.window_id, "cutoff_time_s": targets.cutoff_time_s,
             "label": targets.label, "arrays_file": tp.name,
             "n_samples": int(len(targets.t))}, indent=1))
        written["targets"] = str(tp)
    return written


def load_window(meta_path: str | Path) -> EstimatorInputWindow:
    """Rebuild a window from disk. Targets are NOT loaded here, on purpose."""
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    z = np.load(meta_path.parent / meta["arrays_file"], allow_pickle=False)
    frame = {k[len("frame__"):]: z[k] for k in z.files if k.startswith("frame__")}
    track = {k[len("track__"):]: z[k] for k in z.files if k.startswith("track__")}
    return EstimatorInputWindow(
        window_id=meta["window_id"], race_id=meta["race_id"],
        year=int(meta["year"]), round=int(meta["round"]),
        session=meta["session"], driver=meta["driver"],
        cutoff_time_s=float(meta["cutoff_time_s"]),
        frame=frame, track=track, zones=meta["zones"],
        track_length=float(meta["track_length"]),
        regulation_variant=meta["regulation_variant"],
        regulation={k: (float(v) if k != "variant" else v)
                    for k, v in meta["regulation"].items()},
        assumptions=EstimatorAssumptions(**meta["assumptions"]),
        rho_series=(z["rho_series"] if "rho_series" in z.files else None),
        rho_source=meta.get("rho_source", "scalar"),
        w_along_mps=(z["w_along_mps"] if "w_along_mps" in z.files else None),
        wind_source=meta.get("wind_source", "unavailable"),
        schema_version=meta.get("schema_version", SCHEMA_VERSION),
        source=meta.get("source", "xray.data.ingest.ingest_session"),
        source_fingerprint=meta.get("source_fingerprint", ""),
        notes=tuple(meta.get("notes", ())))
