#!/usr/bin/env python3
"""P3 Part 1 targeted measurements.

This script intentionally does not run the full test suite and does not create
a second real-data loader. Synthetic scoring uses simulator truth only after the
canonical public-observation -> realfit path has produced its estimates.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xray.config import load_config  # noqa: E402
from xray.corpus import build_corpus, fingerprint_json  # noqa: E402
from xray.data.ingest import frame_from_observation  # noqa: E402
from xray.observe import observe, public_channels  # noqa: E402
from xray.realfit import (belief_from_deployment, build_kin, coast_phases,
                          deployment_trace, fit_nuisance_real, interval_bounds)  # noqa: E402
from xray.regs import POST_MIAMI, PRE_MIAMI, rules_for_simulator  # noqa: E402
from xray.sim import LEADER, run_sim  # noqa: E402
from xray.snapshots import make_snapshot  # noqa: E402


def _pct(err):
    a = np.asarray(err, dtype=float)
    return float(100.0 * np.nanmean(a))


def _nearest_idx(t_src, t):
    return np.clip(np.searchsorted(t_src, t), 0, len(t_src) - 1)


def _dilate_binary(x, radius: int):
    if radius <= 0:
        return x
    b = np.asarray(x, dtype=float) > 0.5
    out = b.copy()
    for k in range(1, radius + 1):
        out[:-k] |= b[k:]
        out[k:] |= b[:-k]
    return out.astype(float)


def shipping_realfit_on_sim(cfg, seed: int, *, grid_ds: float = 10.0,
                            mass_kg: float = 790.0, regs=None,
                            smooth_m: float = 60.0, brake_dilate_cells: int = 0,
                            n_particles: int = 160, gt=None,
                            centre: str = "midpoint",
                            boundary: str = "clip",
                            reserve_obs: str = "point") -> dict:
    regs = regs or rules_for_simulator()
    gt = gt or run_sim(cfg, seed=seed)
    obs = observe(gt, LEADER, rate_hz=cfg["observe"]["rate_hz"],
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"],
                  seed=seed + 1, quantize_kmh=cfg["observe"]["quantize_kmh"])
    channels = public_channels(gt, LEADER, obs)
    channels["brake"] = _dilate_binary(channels["brake"], brake_dilate_cells)
    grid = np.arange(0.0, gt.track.length, grid_ds)
    df, quality = frame_from_observation(obs, grid, channels=channels, driver=LEADER)
    usable = df[df["usable"]] if "usable" in df else df
    rho = float(cfg["vehicle"]["rho"])
    kin = build_kin(usable, gt.track, mass_kg, rho, smooth_m=smooth_m, regs=regs)
    fit = fit_nuisance_real(kin, rho, regs=regs)
    trace = deployment_trace(kin, fit, regs=regs, centre=centre)
    belief = belief_from_deployment(kin, trace, fit, n_particles=n_particles,
                                    seed=seed + 2, boundary=boundary,
                                    reserve_obs=reserve_obs)

    true = gt.cars[LEADER]
    idx = _nearest_idx(gt.t, kin.t)
    ok = kin.valid & np.isfinite(kin.t)
    true_e = true.E[idx]
    true_dep = true.P_mguk[idx]
    cda_true = float(cfg["vehicle"]["cda_straight"])

    dep_lo = np.asarray(trace["deploy_lo"], dtype=float)
    dep_hi = np.asarray(trace["deploy_hi"], dtype=float)
    dep_star = np.asarray(belief["deploy_star"], dtype=float)
    dep_band = ok & np.isfinite(dep_lo) & np.isfinite(dep_hi)
    e_band = ok & np.isfinite(belief["soc_p10"]) & np.isfinite(belief["soc_p90"])

    lap_err = []
    lap_bias = []
    for lap, est_j in belief["deployed_lap"].items():
        if 0 <= int(lap) < len(true.deployed_lap) and true.deployed_lap[int(lap)] > 1.0:
            err = (est_j - true.deployed_lap[int(lap)]) / true.deployed_lap[int(lap)]
            lap_err.append(abs(err))
            lap_bias.append(err)

    lo, hi, bounds_ok = interval_bounds(kin, fit.v_wind_hat, 0.012, fit.rho, regs=regs)
    finite_hi = np.isfinite(hi) & bounds_ok
    finite_lo = np.isfinite(lo) & bounds_ok
    braking = (np.nan_to_num(kin.brake) > 0.5
               if kin.brake is not None else kin.a < -6.0)
    coast = coast_phases(kin)
    regime = true.regime[idx[ok]]
    regime_props = {name: float(np.mean(regime == name)) for name in ("brake", "corner", "accel")}
    in_zone_ok = ok & kin.in_deployment_zone
    out_zone_ok = ok & ~kin.in_deployment_zone

    return {
        "seed": seed,
        "estimator": {"centre": centre, "boundary": boundary,
                      "reserve_obs": reserve_obs},
        "grid_ds_m": grid_ds,
        "mass_kg": mass_kg,
        "regulation_variant": regs.variant,
        "brake_dilate_cells": brake_dilate_cells,
        "quality": {
            "usable_laps": sum(1 for q in quality if q.usable),
            "total_laps": len(quality),
            "usable_samples": int(ok.sum()),
        },
        "cda": {
            "truth": cda_true,
            "hat": fit.cda_hat,
            "lo": fit.cda_lo,
            "hi": fit.cda_hi,
            "truth_contained": bool(fit.cda_lo <= cda_true <= fit.cda_hi),
            "centre_error_pct": 100.0 * (fit.cda_hat - cda_true) / cda_true,
            "width_pct_of_truth": 100.0 * (fit.cda_hi - fit.cda_lo) / cda_true,
            "identifiability": fit.identifiability,
            "n_binding": fit.n_binding,
            "n_coast": fit.n_coast,
            "notes": list(fit.notes),
        },
        "deployment": {
            "sample_band_containment": float(np.mean(
                (true_dep[dep_band] >= dep_lo[dep_band] - 1.0)
                & (true_dep[dep_band] <= dep_hi[dep_band] + 1.0))) if dep_band.any() else None,
            "sample_centre_bias_kw": float(np.nanmean(
                (dep_star[dep_band] - true_dep[dep_band]) / 1e3)) if dep_band.any() else None,
            "lap_mape_pct": _pct(lap_err) if lap_err else None,
            "lap_bias_pct": _pct(lap_bias) if lap_bias else None,
        },
        "energy_belief": {
            "soc_particle_band_containment": float(np.mean(
                (true_e[e_band] >= np.asarray(belief["soc_p10"])[e_band])
                & (true_e[e_band] <= np.asarray(belief["soc_p90"])[e_band]))) if e_band.any() else None,
            "soc_mean_mae_mj": float(np.nanmean(
                np.abs(np.asarray(belief["soc_mean"])[ok] - true_e[ok])) / 1e6),
            "soc_mean_bias_mj": float(np.nanmean(
                np.asarray(belief["soc_mean"])[ok] - true_e[ok]) / 1e6),
            # P3.5 addition, purely additive: the ADVERTISED uncertainty. Bias
            # and containment alone cannot distinguish "wrong and saying so"
            # from "wrong while reporting a narrow band", and the second is the
            # dangerous failure mode under model mismatch.
            "soc_particle_width_mj": float(np.nanmean(
                np.asarray(belief["soc_p90"])[ok]
                - np.asarray(belief["soc_p10"])[ok]) / 1e6),
            "usable_particle_width_mj": float(np.nanmean(
                np.asarray(belief["usable_p90"])[ok]
                - np.asarray(belief["usable_p10"])[ok]) / 1e6),
            # P3.5 Part 2: the degeneracy metrics, so synthetic and real are
            # scored on the same quantity.
            "frac_band_degenerate": float(np.mean(
                (np.asarray(belief["usable_p90"])[ok]
                 - np.asarray(belief["usable_p10"])[ok]) < 1.0e4)),
            "frac_reported_empty": float(np.mean(
                np.asarray(belief["usable_mean"])[ok] < 5.0e4)),
            "soc_corr_with_truth": float(np.corrcoef(
                np.asarray(belief["soc_mean"])[ok], true_e[ok])[0, 1]),
        },
        "coverage_counts": {
            "valid_samples": int(ok.sum()),
            "upper_ceiling_bound_samples": int(finite_hi.sum()),
            "lower_harvest_bound_samples": int(finite_lo.sum()),
            "braking_exclusion_samples": int((ok & braking).sum()),
            "coast_samples": int(coast.sum()),
            "in_zone_samples": int((ok & kin.in_deployment_zone).sum()),
            "out_of_zone_samples": int((ok & ~kin.in_deployment_zone).sum()),
            "zone_mask_source": kin.zone_source,
            "regime_proportions": regime_props,
        },
        "zone_ceiling": {
            "regulation_variant": regs.variant,
            "zone_mask_source": kin.zone_source,
            "regulation_variant_separate": True,
            "in_zone_samples": int(in_zone_ok.sum()),
            "out_of_zone_samples": int(out_zone_ok.sum()),
            "in_zone_ceiling_max_kw": (
                float(np.nanmax(kin.ceiling[in_zone_ok]) / 1e3) if in_zone_ok.any() else None),
            "out_of_zone_ceiling_max_kw": (
                float(np.nanmax(kin.ceiling[out_zone_ok]) / 1e3) if out_zone_ok.any() else None),
            "reg_static_cap_zone_kw": float(regs.p_dep_max_zone / 1e3),
            "reg_static_cap_elsewhere_kw": float(regs.p_dep_max_elsewhere / 1e3),
        },
    }


def coverage_map(row: dict) -> list[dict]:
    c = row["coverage_counts"]
    reg = row["regulation_variant"]
    return [
        {"channel": "acceleration/power balance",
         "synthetic_coverage": "YES",
         "evidence": f"{c['valid_samples']} valid canonical samples"},
        {"channel": "deployment upper ceiling from speed taper",
         "synthetic_coverage": "YES",
         "evidence": f"{c['upper_ceiling_bound_samples']} upper-bound samples"},
        {"channel": "deployment-zone eligibility mask construction",
         "synthetic_coverage": "YES",
         "evidence": f"{c['in_zone_samples']} in-zone and {c['out_of_zone_samples']} out-of-zone samples"},
        {"channel": "post-Miami reduced outside-zone deployment cap",
         "synthetic_coverage": "NO" if reg == "stage1-simulator" else "PARTIAL",
         "evidence": f"validation regulations: {reg}"},
        {"channel": "braking exclusion",
         "synthetic_coverage": "YES",
         "evidence": f"{c['braking_exclusion_samples']} brake-flagged samples"},
        {"channel": "harvest lower bound away from brakes",
         "synthetic_coverage": "YES",
         "evidence": f"{c['lower_harvest_bound_samples']} finite lower-bound samples"},
        {"channel": "lift/coast drag identification",
         "synthetic_coverage": "NO",
         "evidence": (f"{c['coast_samples']} detector-positive samples, but the simulator "
                      "has no genuine lift-and-coast regime")},
        {"channel": "weather/wind causal projection",
         "synthetic_coverage": "NO",
         "evidence": "Circuit Sigma has no physical heading; rho is synthetic constant"},
    ]


def bias_sweep(cfg, seeds: list[int], worlds: dict[int, object]) -> dict:
    variants = [
        ("baseline", {}),
        ("grid_20m", {"grid_ds": 20.0}),
        ("grid_5m", {"grid_ds": 5.0}),
        ("mass_768kg", {"mass_kg": 768.0}),
        ("mass_821kg", {"mass_kg": 821.0}),
        ("mass_838kg", {"mass_kg": 838.0}),
        ("pre_miami_bounds", {"regs": PRE_MIAMI}),
        ("post_miami_bounds", {"regs": POST_MIAMI}),
        ("brake_dilation_1cell", {"brake_dilate_cells": 1}),
    ]
    rows = {}
    for name, kw in variants:
        got = []
        for seed in seeds:
            try:
                got.append(shipping_realfit_on_sim(
                    cfg, seed, n_particles=80, gt=worlds[seed], **kw))
            except Exception as exc:  # noqa: BLE001
                got.append({"seed": seed, "error": str(exc)})
        errs = [r["cda"]["centre_error_pct"] for r in got if "cda" in r]
        rows[name] = {
            "runs": got,
            "n_ok": len(errs),
            "mean_cda_centre_error_pct": float(np.mean(errs)) if errs else None,
            "mean_abs_cda_centre_error_pct": float(np.mean(np.abs(errs))) if errs else None,
            "bias_range_pct": [float(np.min(errs)), float(np.max(errs))] if errs else None,
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,13")
    ap.add_argument("--out-root", default=str(ROOT / "out" / "p3" / "part1"))
    args = ap.parse_args()
    cfg = load_config()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=False)

    worlds = {s: run_sim(cfg, seed=s) for s in seeds}
    validation = [shipping_realfit_on_sim(cfg, s, gt=worlds[s]) for s in seeds]
    bias = bias_sweep(cfg, seeds, worlds)

    race_payloads = sorted((ROOT / "out" / "races").glob("2026_r*_R.json"))
    corpus_dir = None
    if race_payloads:
        corpus_dir = build_corpus(
            race_payloads, ROOT / "out" / "p3" / "corpus",
            config_path=ROOT / "config" / "default.yaml",
        )

    snapshot = None
    if race_payloads:
        payload = json.loads(race_payloads[0].read_text())
        first_car = next(iter(payload.get("cars", {})), None)
        if first_car:
            times = payload["cars"][first_car]["trace"].get("t", [])
            finite = [float(t) for t in times if t is not None]
            if finite:
                cutoff = finite[len(finite) // 3]
                snapshot = make_snapshot(payload, cutoff, cutoff, cutoff + 45.0)
                (out_dir / "snapshot_example.json").write_text(
                    json.dumps(snapshot, indent=2, sort_keys=True))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zone_mask_result": {
            "implemented": True,
            "semantics": "current-sample distance inside deployment-zone straight",
            "source": "track.zones geometry shared by synthetic Track and real RealTrack",
            "causal": True,
            "regulation_variant_separate": True,
        },
        "realfit_output_semantics": {
            "cda": "identified interval/set with midpoint centre",
            "deployment": "per-sample identified band plus balance-selected centre used by particles",
            "energy": "particle belief summaries; p10/p90 are particle quantiles, not credible-interval claims",
        },
        "synthetic_direct_validation": validation,
        "coverage_map": coverage_map(validation[0]),
        "cda_bias_investigation": bias,
        "real_corpus": {
            "source_payloads": [str(p) for p in race_payloads],
            "manifest_dir": None if corpus_dir is None else str(corpus_dir),
        },
        "causal_snapshot": {
            "definition": "input_fields are at or before feature_cutoff_time; evaluation_fields are after target_start_time",
            "example_fingerprint": None if snapshot is None else snapshot["provenance"]["input_fingerprint"],
        },
    }
    summary["summary_fingerprint"] = fingerprint_json(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(out_dir / "summary.json")
    print(json.dumps({
        "summary_fingerprint": summary["summary_fingerprint"],
        "direct_validation_seeds": seeds,
        "corpus_dir": summary["real_corpus"]["manifest_dir"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
