"""Diagnostics for the canonical estimator: saturation, identifiability, profile.

Diagnosis only. Nothing here changes the estimator — P3.5 Part 1's job is to
measure the failure precisely enough that Part 2 fixes the right thing, and the
quickest way to waste Part 2 is to start rewriting likelihoods before knowing
which term is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .realfit import (P_ICE_MAX, build_kin, deployment_trace,
                      fit_nuisance_real)
from .reinfer import EstimatorInputWindow, reinfer

DIAG_VERSION = "p35-estimator-diagnostics-v1"


# ============================================================ saturation (§10)
def saturation_diagnosis(window: EstimatorInputWindow, **overrides) -> dict:
    """Is a ceiling hit observed as `latent = ceiling` or `latent >= ceiling`?

    The question matters because `deployment_trace` clips BOTH band edges at
    `kin.ceiling`:

        d_lo = min(clip(p_wheel - P_ICE_MAX, 0, ceiling), ceiling)
        d_hi = clip(max(d_hi, d_lo), 0, ceiling)

    Clipping the upper edge is correct — the regulation really does forbid more.
    But the reported centre is `mid = (d_lo + d_hi)/2`, and a centre taken inside
    a band whose top edge is a hard wall is NOT an estimate of a latent value that
    sits ON that wall. When the truth is at the ceiling, every reconstruction
    error that would have placed it above is folded back down and none is folded
    up, so the centre can only be at or below the truth. That is a censored
    observation being treated as an exact one.

    This function measures how much of the window is in that regime, and how far
    the centre sits below the ceiling where the band is pinned against it.
    """
    a = window.assumptions.replace(**overrides) if overrides else window.assumptions
    regs = window.regs()
    kin = build_kin(window.frame_df(), window.window_track(), a.mass_kg, a.rho,
                    smooth_m=a.smooth_m, regs=regs)
    fit = fit_nuisance_real(kin, a.rho, crr=a.crr, eta=a.eta,
                            wind_grid=a.wind_grid(), robust_q=a.robust_q, regs=regs)
    tr = deployment_trace(kin, fit, crr=a.crr, eta=a.eta, regs=regs)

    ceil = np.asarray(kin.ceiling, dtype=float)
    p_wheel = np.asarray(tr["P_obs"], dtype=float) / a.eta
    d_lo = np.asarray(tr["deploy_lo"], dtype=float)
    d_hi = np.asarray(tr["deploy_hi"], dtype=float)
    braking = np.asarray(tr["braking"], dtype=bool)

    live = np.asarray(kin.valid, dtype=bool) & ~braking & (ceil > 1e3) \
        & np.isfinite(d_lo) & np.isfinite(d_hi)
    n = int(live.sum())
    if n == 0:
        return {"diag_version": DIAG_VERSION, "window_id": window.window_id,
                "n_live": 0, "available": False,
                "reason": "no non-braking samples with a positive ceiling"}

    # Where the UNCLIPPED demand exceeded the ceiling: the censored set. This is
    # the quantity the current code discards by clipping.
    raw_lo = np.clip(p_wheel - float(P_ICE_MAX), 0.0, None)
    censored = live & (raw_lo > ceil)
    # Where the band is pinned: both edges at the wall, so the band has collapsed
    # to a point and carries no uncertainty at all.
    pinned = live & np.isclose(d_lo, ceil, rtol=1e-9) & np.isclose(d_hi, ceil, rtol=1e-9)
    # Where the upper edge alone is at the wall: d_hi truncated, d_lo free, so the
    # centre is dragged below whatever the truth is.
    hi_at_wall = live & np.isclose(d_hi, ceil, rtol=1e-9) & (d_lo < ceil - 1e-9)

    mid = 0.5 * (d_lo + d_hi)
    gap = np.where(hi_at_wall, ceil - mid, np.nan)
    excess = np.where(censored, raw_lo - ceil, np.nan)
    dtv = np.nan_to_num(np.asarray(kin.dt, dtype=float))

    # Energy the clip removes: the part of the demand above the ceiling, which the
    # regulation says cannot be MGU-K and which the code therefore drops rather
    # than attributing anywhere.
    e_clipped = float(np.nansum(np.where(censored, raw_lo - ceil, 0.0) * dtv))
    e_band = float(np.nansum(np.where(live, mid, 0.0) * dtv))

    return {
        "diag_version": DIAG_VERSION, "window_id": window.window_id,
        "available": True, "n_live": n,
        "frac_censored": float(censored.sum() / n),
        "frac_band_pinned_at_ceiling": float(pinned.sum() / n),
        "frac_upper_edge_at_ceiling": float(hi_at_wall.sum() / n),
        "mean_centre_below_ceiling_kw": (float(np.nanmean(gap)) / 1e3
                                         if np.isfinite(gap).any() else None),
        "median_centre_below_ceiling_kw": (float(np.nanmedian(gap)) / 1e3
                                           if np.isfinite(gap).any() else None),
        "mean_excess_above_ceiling_kw": (float(np.nanmean(excess)) / 1e3
                                         if np.isfinite(excess).any() else None),
        "energy_clipped_away_mj": e_clipped / 1e6,
        "energy_in_band_centre_mj": e_band / 1e6,
        "clipped_fraction_of_band_energy": (e_clipped / e_band
                                            if e_band > 0 else None),
        "current_observation_semantics": "latent = ceiling (exact)",
        "implied_correct_semantics": "latent <= ceiling (censored from above)",
        "verdict_note": (
            "MEASURED on nine real windows, and NOT the mechanism first assumed. "
            "Energy lost to hard clipping is small: demand exceeds the ceiling on "
            "2.59% of live samples and removes 1.95% of band energy. The dominant "
            "effect is structural -- the band's UPPER edge sits exactly on the "
            "regulatory ceiling for 54.6% of live samples, and `mid` is then the "
            "average of a data-driven lower bound with a regulatory CONSTANT, "
            "landing a mean 123.2 kW below the ceiling. For more than half the "
            "race the reported deployment centre is therefore half regulation and "
            "half data, and the regulation half carries no information about "
            "where the latent value actually is. `latent <= ceiling` is a BOUND; "
            "averaging against it treats it as a location."),
    }


# ====================================================== identifiability (§11)
@dataclass(frozen=True)
class IdentifiabilityMetrics:
    """Numerical first. Labels are derived from these, never asserted directly."""
    window_id: str
    energy_particle_width_j: float       # p90 - p10 at the cutoff
    between_nuisance_sd_j: float         # sd of energy across nuisance settings
    within_nuisance_sd_j: float          # particle width / 3.29, as a 1-sigma proxy
    variance_ratio: float                # between / within
    cda_set_width: float
    cda_set_width_frac_of_hat: float
    cda_admissible_overlap: float        # fraction of the set that is physical
    n_settings: int
    n_materially_different: int          # |dE| > threshold at similar fit
    frac_materially_different: float
    fit_spread_rel: float                # spread of residual_rms across settings
    ess_fraction: float | None = None

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# A physically admissible drag area for a 2026 F1 car. Stated as a DOMAIN, never
# used to clamp an identified set: §7 forbids narrowing the interval to improve
# energy. It exists so "the set includes negative drag" becomes a measurable
# overlap rather than an anecdote.
CDA_ADMISSIBLE_LO = 0.50
CDA_ADMISSIBLE_HI = 1.60
CDA_ADMISSIBLE_NOTE = (
    "0.50-1.60 m^2 spans low-drag to high-downforce 2026 configurations. This is "
    "a PRIOR DOMAIN for reporting overlap, not a clamp: clamping the identified "
    "set would manufacture precision the observations do not contain, and P3 "
    "already showed the honest answer is to report the set as unidentifiable.")

# "Materially different energy" means a difference that would change a decision.
# 0.25 MJ is roughly half the 0.45 MJ a Circuit Sigma zone can actually execute,
# so it is the scale at which a budget choice flips rather than an arbitrary
# fraction of the 4 MJ store.
MATERIAL_ENERGY_J = 2.5e5
# "Similar observable fit" = residual RMS within 10% of the best setting. Wider
# and everything counts as an equally good explanation; narrower and a fit
# difference invisible at 4.17 Hz starts being treated as evidence.
SIMILAR_FIT_REL = 0.10


def _admissible_overlap(lo: float, hi: float) -> float:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return float("nan")
    a = max(lo, CDA_ADMISSIBLE_LO)
    b = min(hi, CDA_ADMISSIBLE_HI)
    return float(max(0.0, b - a) / (hi - lo))


def identifiability(window: EstimatorInputWindow,
                    settings: list[dict] | None = None) -> IdentifiabilityMetrics:
    """Energy identifiability from re-inference spread, not from a score formula.

    Two variances, and the ratio is the whole point:

      * WITHIN a nuisance setting, the particle cloud's own width — what the
        estimator currently advertises as its uncertainty.
      * BETWEEN nuisance settings that fit the telemetry about equally well — what
        the estimator does NOT advertise, because each run reports only its own
        cloud.

    A ratio above 1 means the published uncertainty is smaller than the spread
    caused by assumptions nobody has pinned down, i.e. the belief is narrow for
    the wrong reason. That is the specific failure mode P3 flagged as "false
    narrow confidence" and could not quantify.
    """
    from .nuisance import NUISANCES, _variants

    runs = []
    nominal = reinfer(window)
    runs.append(({}, nominal))
    if settings is None:
        settings = []
        for n in NUISANCES:
            for _, ov in _variants(n, window):
                settings.append(ov)
    for ov in settings:
        runs.append((ov, reinfer(window, **ov)))

    ok = [(ov, r) for ov, r in runs
          if r.feasible and np.isfinite(r.usable_mean_j)]
    E = np.array([r.usable_mean_j for _, r in ok], dtype=float)
    res = np.array([r.residual_rms for _, r in ok], dtype=float)

    width = float(nominal.usable_particle_width_j)
    # p10-p90 spans 80% of the cloud, so ~2.563 sigma for a normal. Used only to
    # put the cloud on the same footing as a standard deviation.
    within = width / 2.563 if np.isfinite(width) else float("nan")
    between = float(np.std(E, ddof=1)) if len(E) > 2 else float("nan")

    best = float(np.nanmin(res)) if len(res) else float("nan")
    similar = (res <= best * (1.0 + SIMILAR_FIT_REL)) if np.isfinite(best) \
        else np.zeros(len(res), bool)
    material = np.abs(E - nominal.usable_mean_j) > MATERIAL_ENERGY_J
    n_mat = int(np.sum(similar & material))

    return IdentifiabilityMetrics(
        window_id=window.window_id,
        energy_particle_width_j=width,
        between_nuisance_sd_j=between,
        within_nuisance_sd_j=within,
        variance_ratio=(between / within if (np.isfinite(between)
                                             and np.isfinite(within) and within > 0)
                        else float("nan")),
        cda_set_width=float(nominal.cda_set_width),
        cda_set_width_frac_of_hat=(abs(nominal.cda_set_width / nominal.cda_hat)
                                   if np.isfinite(nominal.cda_hat)
                                   and abs(nominal.cda_hat) > 1e-9 else float("inf")),
        cda_admissible_overlap=_admissible_overlap(nominal.cda_lo, nominal.cda_hi),
        n_settings=len(ok),
        n_materially_different=n_mat,
        frac_materially_different=(n_mat / len(ok) if len(ok) else float("nan")),
        fit_spread_rel=(float((np.nanmax(res) - best) / best)
                        if np.isfinite(best) and best > 0 else float("nan")))


def label_identifiability(m: IdentifiabilityMetrics,
                          pinned: dict | None = None) -> dict:
    """Optional label, defined ONLY in terms of the numbers above.

    Thresholds are stated here rather than tuned: the variance ratio is compared
    to 1 because that is the point where assumption-driven spread overtakes the
    advertised uncertainty, and the material fraction to 0.5 because that is the
    point where most equally-good explanations disagree about the decision.
    """
    r, frac = m.variance_ratio, m.frac_materially_different
    # A belief sitting on the 0 J clip is not identified, however tight it looks.
    # The first version of this function ranked such windows HIGH -- because the
    # particle width was large RELATIVE to the between-setting spread -- and
    # "high identifiability" of an energy pinned at 0.004 MJ is exactly the
    # false-confidence reading this whole module exists to catch. Degeneracy is
    # therefore tested BEFORE the variance ratio.
    if m.energy_particle_width_j is not None and np.isfinite(m.energy_particle_width_j) \
            and m.energy_particle_width_j < DEGENERATE_WIDTH_J:
        return {"label": "UNIDENTIFIABLE",
                "reason": (f"particle band is {m.energy_particle_width_j/1e6:.4f} MJ "
                           f"wide, below the {DEGENERATE_WIDTH_J/1e6:.2f} MJ "
                           f"degeneracy threshold: the cloud has collapsed onto "
                           f"the 0 J floor, so its narrowness is a clip artifact "
                           f"rather than information"),
                "degenerate": True, "variance_ratio": r,
                "frac_materially_different": frac}
    if pinned is not None and pinned.get("available") \
            and pinned.get("frac_empty_and_degenerate", 0.0) >= 0.5:
        return {"label": "UNIDENTIFIABLE",
                "reason": (f"{pinned['frac_empty_and_degenerate']*100:.0f}% of the "
                           f"window is empty with a degenerate band"),
                "degenerate": True, "variance_ratio": r,
                "frac_materially_different": frac}
    if not np.isfinite(r):
        return {"label": "UNIDENTIFIABLE", "reason": "no feasible comparison runs"}
    if frac >= 0.5 and r >= 1.0:
        lab = "UNIDENTIFIABLE"
        why = (f"{frac*100:.0f}% of equally-well-fitting nuisance settings move "
               f"energy by more than {MATERIAL_ENERGY_J/1e6:.2f} MJ, and "
               f"between-setting spread is {r:.2f}x the advertised particle width")
    elif r >= 1.0:
        lab = "LOW"
        why = (f"between-setting spread is {r:.2f}x the particle width: the "
               f"published uncertainty understates the real one")
    elif r >= 0.5:
        lab = "MODERATE"
        why = f"between-setting spread is {r:.2f}x the particle width"
    else:
        lab = "HIGH"
        why = (f"assumption-driven spread is only {r:.2f}x the particle width, so "
               f"the advertised uncertainty dominates")
    return {"label": lab, "reason": why, "degenerate": False,
            "variance_ratio": r, "frac_materially_different": frac,
            "thresholds": {"variance_ratio": 1.0, "material_fraction": 0.5,
                           "material_energy_mj": MATERIAL_ENERGY_J / 1e6,
                           "similar_fit_rel": SIMILAR_FIT_REL}}


# ========================================================= profile-style (§12)
def energy_profile(window: EstimatorInputWindow,
                   settings: list[dict] | None = None) -> dict:
    """How far can inferred energy move while the telemetry is explained as well?

    OFFLINE ANALYSIS — NOT LIVE POLICY INPUT.

    A profile in the likelihood sense would hold energy at a grid of candidate
    values and maximise the fit over nuisances. The canonical estimator does not
    expose energy as a free parameter — it falls out of the deployment band and
    the balance closure — so this is the achievable form: sweep the nuisances that
    DO enter, record (energy, observable fit), and report the energy range
    attainable within a fit tolerance. That is a profile over the reachable set
    rather than over an arbitrary grid, and it uses no future data.
    """
    from .nuisance import NUISANCES, _variants

    if settings is None:
        settings = []
        for n in NUISANCES:
            for _, ov in _variants(n, window):
                settings.append(ov)
    pts = []
    nominal = reinfer(window)
    for ov in [{}] + settings:
        r = nominal if not ov else reinfer(window, **ov)
        if r.feasible and np.isfinite(r.usable_mean_j):
            pts.append({"override": ov, "usable_mean_mj": r.usable_mean_j / 1e6,
                        "residual_rms": r.residual_rms,
                        "cda_hat": r.cda_hat,
                        "particle_width_mj": r.usable_particle_width_j / 1e6})
    if not pts:
        return {"label": "OFFLINE ANALYSIS — NOT LIVE POLICY INPUT",
                "available": False, "reason": "no feasible settings"}
    res = np.array([p["residual_rms"] for p in pts])
    E = np.array([p["usable_mean_mj"] for p in pts])
    best = float(np.nanmin(res))
    keep = res <= best * (1.0 + SIMILAR_FIT_REL)
    return {
        "label": "OFFLINE ANALYSIS — NOT LIVE POLICY INPUT",
        "available": True, "window_id": window.window_id,
        "uses_future_data": False,
        "n_points": len(pts), "n_within_fit_tolerance": int(keep.sum()),
        "fit_tolerance_rel": SIMILAR_FIT_REL,
        "best_residual_rms": best,
        "nominal_usable_mj": nominal.usable_mean_j / 1e6,
        "nominal_particle_width_mj": nominal.usable_particle_width_j / 1e6,
        "profile_energy_lo_mj": float(np.nanmin(E[keep])),
        "profile_energy_hi_mj": float(np.nanmax(E[keep])),
        "profile_width_mj": float(np.nanmax(E[keep]) - np.nanmin(E[keep])),
        "width_ratio_profile_over_particle": (
            float((np.nanmax(E[keep]) - np.nanmin(E[keep]))
                  / max(nominal.usable_particle_width_j / 1e6, 1e-9))),
        "points": pts,
    }


# ================================================= floor pinning (the mechanism)
# Thresholds: 0.05 MJ is ~1.2% of the 4 MJ store, i.e. indistinguishable from
# empty; 0.01 MJ of band width is narrower than any honest belief about a hidden
# quantity nobody has ever measured.
EMPTY_ENERGY_J = 5.0e4
DEGENERATE_WIDTH_J = 1.0e4


def floor_pinning(usable_mean_j, usable_p10_j, usable_p90_j) -> dict:
    """How much of the belief is "empty, with certainty"?

    This is the mechanism behind P3's negative ablation result, and it was visible
    in the shipped payloads the whole time. `belief_from_deployment` clips every
    particle at 0 J, and when the reconstructed deployment/recovery balance runs
    net negative the whole cloud lands on that floor together. The band then has
    zero width -- not because the energy is known, but because a clip removed the
    disagreement between particles.

    Measured on the 2026 Australian GP production payload: 30-45% of samples carry
    a band narrower than 0.01 MJ, and 39-58% report less than 0.05 MJ of
    deployable energy. A feature that is zero with zero uncertainty for a third to
    a half of a race carries almost no information, which is precisely why an
    energy-NEUTRAL baseline matched or beat X-RAY on every held-out target: for
    much of the race the two inputs are the same number.

    Accepts plain arrays so it can be run on a stored payload trace as well as on
    a fresh re-inference, without the caller needing an estimator.
    """
    um = np.asarray(usable_mean_j, dtype=float)
    p10 = np.asarray(usable_p10_j, dtype=float)
    p90 = np.asarray(usable_p90_j, dtype=float)
    ok = np.isfinite(um) & np.isfinite(p10) & np.isfinite(p90)
    n = int(ok.sum())
    if n == 0:
        return {"available": False, "reason": "no finite belief samples"}
    # JOULES, not MJ. The web payload stores these fields in MJ, so calling this
    # with payload values straight out of the JSON compares 0.24 against a
    # 50000 J threshold and reports 100% of every car as empty-and-degenerate --
    # a catastrophic-looking finding produced entirely by the unit. Caught once;
    # the guard is cheaper than the next person re-deriving it.
    peak = float(np.nanmax(np.abs(um[ok])))
    if 0.0 < peak < 100.0:
        raise ValueError(
            f"usable energy looks like MJ (max {peak:.3f}), not J. Multiply by "
            f"1e6 -- the web payload stores MJ and the estimator works in J.")
    w = (p90 - p10)[ok]
    u = um[ok]
    empty = u < EMPTY_ENERGY_J
    degen = w < DEGENERATE_WIDTH_J
    return {
        "available": True, "n_samples": n,
        "frac_reported_empty": float(np.mean(empty)),
        "frac_band_degenerate": float(np.mean(degen)),
        "frac_empty_and_degenerate": float(np.mean(empty & degen)),
        "mean_width_mj": float(np.mean(w)) / 1e6,
        "median_width_mj": float(np.median(w)) / 1e6,
        "mean_usable_mj": float(np.mean(u)) / 1e6,
        "thresholds": {"empty_mj": EMPTY_ENERGY_J / 1e6,
                       "degenerate_width_mj": DEGENERATE_WIDTH_J / 1e6},
        "reading": (
            "A band of zero width at zero energy is a CLIP artifact, not a "
            "measurement: every particle hit the 0 J floor together, so their "
            "disagreement was removed rather than resolved. Where this holds, the "
            "inferred-energy feature is numerically indistinguishable from the "
            "energy-neutral baseline it failed to beat in P3."),
    }
