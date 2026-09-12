"""Synthetic compensating-error harness: does wrong physics give confidently wrong energy?

P3 measured the estimator against a simulator whose physics MATCHED the
estimator's assumptions. That answers "is the inference self-consistent", not "what
happens when the world differs from what we assumed" — and the second question is
the one a real race asks, because the real world certainly differs.

The failure mode being hunted is specific, and it is worse than simple error:

    wrong nuisance physics  ->  wrong energy  ->  NARROW particle range

Bias and containment cannot separate that from honest error. A biased estimate
that widens its band is behaving correctly under mismatch; one that keeps a narrow
band while drifting is asserting confidence it has not earned, and every consumer
downstream — the DP, the budget optimiser, the UI — will treat that narrow band as
information. So width is measured alongside bias, and the diagnostic verdict is a
function of both.

Truth is read only for SCORING, through the existing P3 harness. Nothing here
gives the estimator access to it.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np

MISMATCH_VERSION = "p35-synthetic-mismatch-v1"

_HARNESS = Path(__file__).resolve().parent.parent / "scripts" / "11.p3_part1.py"


def _harness():
    """Load P3 Part 1's synthetic harness by path.

    It lives in a numerically-named script so it cannot be imported normally.
    Loading it is deliberate: reimplementing `shipping_realfit_on_sim` would
    create the second synthetic estimator the P3.5 brief forbids, and a copy
    would drift from the one that produced the authoritative P3 numbers.
    """
    spec = importlib.util.spec_from_file_location("_p3_part1", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Each case perturbs the SIMULATOR's truth while the estimator keeps its shipping
# assumptions. `estimator_mass_kg` stays 790 throughout, because that is what
# `analysis.py` passes on the real path -- the point is the gap, not a sweep.
MISMATCH_CASES = (
    {"case": "matched", "cfg": {}, "note": "control: simulator physics equals the "
                                           "estimator's assumptions"},
    {"case": "cda_low", "cfg": {("vehicle", "cda_straight"): 0.55},
     "note": "truth drag 17% below the configured value"},
    {"case": "cda_high", "cfg": {("vehicle", "cda_straight"): 0.79},
     "note": "truth drag 20% above the configured value"},
    {"case": "mass_high", "cfg": {("vehicle", "mass_car"): 845.0},
     "note": "truth mass 845 kg against the estimator's assumed 790"},
    {"case": "mass_low", "cfg": {("vehicle", "mass_car"): 760.0},
     "note": "truth mass 760 kg against the estimator's assumed 790"},
    {"case": "noise_high", "cfg": {("observe", "speed_noise_ms"): 0.90},
     "note": "feed noise 0.90 m/s against the 0.35 the estimator was tuned at"},
)


def _apply(cfg: dict, patch: dict) -> dict:
    out = copy.deepcopy(cfg)
    for (section, key), val in patch.items():
        out[section][key] = val
    return out


def run_mismatch(cfg: dict, seeds=(42, 7), estimator_mass_kg: float = 790.0,
                 n_particles: int = 160, cases=MISMATCH_CASES,
                 estimator: dict | None = None) -> dict:
    """Score the canonical estimator on worlds whose physics it got wrong.

    `estimator` selects the configuration under test, so Part 2 can run the
    hardened path against the SAME case definitions Part 1 recorded without
    regenerating the old numbers.
    """
    est = dict(estimator or {})
    h = _harness()
    rows = []
    for case in cases:
        patch = {k: v for k, v in case["cfg"].items()}
        wcfg = _apply(cfg, patch)
        per_seed = []
        for seed in seeds:
            try:
                r = h.shipping_realfit_on_sim(
                    wcfg, seed=seed, mass_kg=estimator_mass_kg,
                    n_particles=n_particles, **est)
            except Exception as exc:            # a refusal is a result here
                per_seed.append({"seed": seed, "failed": f"{type(exc).__name__}: {exc}"})
                continue
            eb, cd = r["energy_belief"], r["cda"]
            per_seed.append({
                "seed": seed,
                "cda_truth": cd["truth"], "cda_hat": cd["hat"],
                "cda_contained": cd["truth_contained"],
                "cda_width_pct_of_truth": cd["width_pct_of_truth"],
                "identifiability": cd["identifiability"],
                "soc_mae_mj": eb["soc_mean_mae_mj"],
                "soc_bias_mj": eb["soc_mean_bias_mj"],
                "soc_containment": eb["soc_particle_band_containment"],
                "soc_width_mj": eb.get("soc_particle_width_mj"),
                "usable_width_mj": eb.get("usable_particle_width_mj"),
                "frac_band_degenerate": eb.get("frac_band_degenerate"),
                "frac_reported_empty": eb.get("frac_reported_empty"),
                "soc_corr_with_truth": eb.get("soc_corr_with_truth"),
                "dep_lap_mape_pct": r["deployment"]["lap_mape_pct"],
                "dep_lap_bias_pct": r["deployment"]["lap_bias_pct"],
            })
        ok = [x for x in per_seed if "failed" not in x]
        agg = {}
        if ok:
            def m(k):
                v = [x[k] for x in ok if x.get(k) is not None]
                return float(np.mean(v)) if v else None
            agg = {"soc_mae_mj": m("soc_mae_mj"), "soc_bias_mj": m("soc_bias_mj"),
                   "soc_containment": m("soc_containment"),
                   "soc_width_mj": m("soc_width_mj"),
                   "usable_width_mj": m("usable_width_mj"),
                   "frac_band_degenerate": m("frac_band_degenerate"),
                   "frac_reported_empty": m("frac_reported_empty"),
                   "soc_corr_with_truth": m("soc_corr_with_truth"),
                   "cda_width_pct_of_truth": m("cda_width_pct_of_truth"),
                   "identifiability": m("identifiability"),
                   "cda_contained": float(np.mean([bool(x["cda_contained"])
                                                   for x in ok])),
                   "dep_lap_mape_pct": m("dep_lap_mape_pct"),
                   "dep_lap_bias_pct": m("dep_lap_bias_pct")}
        rows.append({"case": case["case"], "note": case["note"],
                     "patch": {f"{a}.{b}": v for (a, b), v in patch.items()},
                     "per_seed": per_seed, "aggregate": agg,
                     "n_failed": len(per_seed) - len(ok)})
    return {"mismatch_version": MISMATCH_VERSION,
            "estimator": est or {"centre": "midpoint", "boundary": "clip",
                                 "reserve_obs": "point"},
            "estimator_mass_kg": estimator_mass_kg,
            "seeds": list(seeds), "cases": rows,
            "verdict": verdict(rows)}


def verdict(rows: list[dict]) -> dict:
    """Did mismatch widen the band, or did it stay narrow while the error grew?

    A band that does not widen when the error grows is the dangerous case, and
    `compensating` names it: the estimator absorbed the wrong physics into other
    terms and carried on reporting the same confidence.
    """
    base = next((r for r in rows if r["case"] == "matched"), None)
    if base is None or not base["aggregate"]:
        return {"available": False, "reason": "control case did not produce a result"}
    b = base["aggregate"]
    out = []
    for r in rows:
        if r["case"] == "matched" or not r["aggregate"]:
            continue
        a = r["aggregate"]
        d_err = ((a["soc_mae_mj"] or 0.0) - (b["soc_mae_mj"] or 0.0))
        d_width = ((a["soc_width_mj"] or 0.0) - (b["soc_width_mj"] or 0.0))
        grew = d_err > 0.02                      # 20 kJ: above run-to-run noise
        widened = d_width > 0.02
        out.append({
            "case": r["case"], "d_soc_mae_mj": d_err, "d_soc_width_mj": d_width,
            "containment": a["soc_containment"],
            "error_grew": grew, "band_widened": widened,
            "compensating_error": bool(grew and not widened),
            "reading": ("error grew while the band did NOT widen — confidently "
                        "wrong" if (grew and not widened) else
                        "error grew and the band widened with it" if (grew and widened)
                        else "no material error increase")})
    n_bad = sum(1 for x in out if x["compensating_error"])
    return {"available": True, "cases": out,
            "n_compensating": n_bad, "n_cases": len(out),
            "headline": (f"{n_bad} of {len(out)} mismatch cases produced a LARGER "
                         f"energy error WITHOUT a wider band")}
