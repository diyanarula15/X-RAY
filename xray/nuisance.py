"""Which physical assumptions actually reach the canonical energy inference.

The test for inclusion here is mechanical: the quantity must appear in the
equations `realfit` evaluates on the production path. That rules out most of the
repository's physics. `tyres`, `ClA`, wake and wetness are real, tested models
that the canonical estimator never reads — they live in the simulator and the
decision layer — so listing them as nuisances of the estimator would invite
someone to "calibrate" them expecting the energy belief to move, and it would not.

Each entry records whether current data can constrain it, because that is the
field that decides what Part 2 may propagate. A nuisance that matters and cannot
be constrained is a research problem; one that matters and can be is a fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .reinfer import EstimatorInputWindow, ReinferenceResult, reinfer

# Status vocabulary reused from the P3 registry so one word cannot mean two
# things in two places.
from .registry import (HEURISTIC, INFERRED, PHYSICS, REGULATION, SYNTHETIC,
                       UNAVAILABLE)

REGISTRY_VERSION = "p35-nuisance-registry-v1"


@dataclass(frozen=True)
class Nuisance:
    """One quantity the canonical inference takes as given.

    `override` is the `EstimatorAssumptions` field name, or None when the
    quantity is not currently overridable through that surface — which is itself
    a finding, not an omission.
    """
    name: str
    symbol: str
    canonical_source: str          # where the production value comes from
    enters: str                    # the equation it appears in
    value: Any
    low: Any                       # defensible low end, with a reason
    high: Any
    reason_for_range: str
    p3_status: str
    static: bool                   # constant over a race, or time-varying
    currently_inferred: bool       # fitted by realfit, or fixed by assumption
    affects_energy_directly: bool
    constrainable_now: bool
    constrainable_note: str
    override: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# Nominal mass is `analysis.py`'s constant 790 kg. The range is the one
# `realfit.mass_series` already measured against simulator truth: 768 kg is the
# dry minimum and 838 kg is dry plus a full fuel load, so the true race mass is
# inside it by construction and 790 sits below the physical floor plus any fuel.
_MASS_NOTE = ("768 kg is the published dry minimum and 838 kg is dry plus a full "
              "fuel load, so the truth lies inside. 790 kg is BELOW dry+fuel, "
              "i.e. an end-of-race number applied to a whole race; "
              "realfit.mass_series measured that as ~3 points of MAPE and most "
              "of an 8% low bias on synthetic truth")

NUISANCES: tuple[Nuisance, ...] = (
    Nuisance(
        "Drag area", "CdA", "fit_nuisance_real interval intersection",
        "_terms: B = 0.5*rho*cda_scale*(v-w+v_wind)^2*v",
        value="identified set per car", low=None, high=None,
        reason_for_range="bracketed from both sides by the regulation; the set "
                         "width IS the identifiability score and is not a range "
                         "to be narrowed by hand",
        p3_status=INFERRED, static=True, currently_inferred=True,
        affects_energy_directly=True, constrainable_now=True,
        constrainable_note="constrained by the coast-down channel, which exists "
                           "at every circuit; width is the open problem, not "
                           "absence of a channel",
        override=None),
    Nuisance(
        "Vehicle mass", "m", "analysis.py constant 790 kg",
        "_terms: A = m*a*v + crr*m*G*v + m*G*sin(grade)*v — all three terms",
        value=790.0, low=768.0, high=838.0, reason_for_range=_MASS_NOTE,
        p3_status=PHYSICS, static=True, currently_inferred=False,
        affects_energy_directly=True, constrainable_now=False,
        constrainable_note="no public channel publishes a fuel load; mass is "
                           "degenerate with CdA in the accel term and cannot be "
                           "separated from it by a speed trace alone",
        override="mass_kg"),
    Nuisance(
        "Rolling resistance", "crr", "fit_nuisance_real / deployment_trace default",
        "_terms: crr*m*G*v",
        value=0.012, low=0.008, high=0.016,
        reason_for_range="slick-on-asphalt literature spread; no measurement for "
                         "2026 compounds exists in this corpus",
        p3_status=SYNTHETIC, static=True, currently_inferred=False,
        affects_energy_directly=True, constrainable_now=False,
        constrainable_note="enters linearly in v like part of the drag term; not "
                           "separable from CdA without a load sweep",
        override="crr"),
    Nuisance(
        "Air density", "rho", "session weather summary (sd.weather['rho'])",
        "_terms: B scales linearly with rho",
        value="session mean", low=-0.03, high=+0.03,
        reason_for_range="+/-3% covers a cold first lap to a hot last one; the "
                         "corpus has no weather TRACE, only a session summary, so "
                         "the within-race variation is unobserved",
        p3_status=HEURISTIC, static=False, currently_inferred=False,
        affects_energy_directly=True, constrainable_now=False,
        constrainable_note="a per-sample rho channel exists in the code "
                           "(rho_series) but the corpus cannot fill it",
        override="rho"),
    Nuisance(
        "Drivetrain / ICE efficiency", "eta", "fit_nuisance_real default 0.95",
        "interval_bounds upper bound (P_ICE_MAX + ceiling)*eta; "
        "deployment_trace p_wheel = P_obs/eta",
        value=0.95, low=0.90, high=0.98,
        reason_for_range="mechanical efficiency of a modern F1 drivetrain; no "
                         "published figure for 2026",
        p3_status=SYNTHETIC, static=True, currently_inferred=False,
        affects_energy_directly=True, constrainable_now=False,
        constrainable_note="scales the whole power budget, so it trades off "
                           "against CdA and against the deployment band together",
        override="eta"),
    Nuisance(
        "Derivative smoothing length", "smooth_m", "build_kin default 60 m",
        "a = v*dv/ds via Savitzky-Golay; a enters _terms through m*a*v",
        value=60.0, low=40.0, high=100.0,
        reason_for_range="real telemetry is 4.17 Hz median and irregular; too "
                         "short amplifies noise into a, too long smears real "
                         "acceleration across corner exits",
        p3_status=HEURISTIC, static=True, currently_inferred=False,
        affects_energy_directly=True, constrainable_now=True,
        constrainable_note="observable: the residual RMS of the power balance "
                           "responds to it, so it can be chosen rather than assumed",
        override="smooth_m"),
    Nuisance(
        "Along-track wind", "v_wind / w_along", "fitted nuisance; measured "
                                                "projection only if oriented",
        "_terms: (v - w_along + v_wind)^2",
        value="fitted over the wind grid", low=-4.0, high=4.0,
        reason_for_range="the grid fit_nuisance_real already searches",
        p3_status=UNAVAILABLE, static=True, currently_inferred=True,
        affects_energy_directly=True, constrainable_now=False,
        constrainable_note="no verified circuit orientation, so the MEASURED "
                           "projection is unavailable and the fitted term is the "
                           "whole wind rather than a residual",
        override=None),
    Nuisance(
        "Regulation variant", "RegSet", "regs_for(session date)",
        "interval_bounds harvest floor; build_kin deployment ceiling via p_dep_max",
        value="date-dispatched", low=None, high=None,
        reason_for_range="not a nuisance to vary: the rule in force is a fact "
                         "about the date, and varying it would be analysing the "
                         "wrong race",
        p3_status=REGULATION, static=True, currently_inferred=False,
        affects_energy_directly=True, constrainable_now=True,
        constrainable_note="known exactly from the calendar",
        override=None),
    Nuisance(
        "Particle reserve prior", "reserve_max_frac", "belief_from_deployment 0.35",
        "sets how much store the driver may be holding back, which is what makes "
        "DEPLOYABLE energy differ from stored energy",
        value=0.35, low=0.20, high=0.50,
        reason_for_range="a buffer below 20% leaves no room for the cut-out "
                         "observation to be informative; above 50% the reported "
                         "deployable energy becomes mostly prior",
        p3_status=SYNTHETIC, static=True, currently_inferred=False,
        affects_energy_directly=True, constrainable_now=True,
        constrainable_note="cut-out events are observable and do constrain it, "
                           "but only on laps where one occurs",
        override="reserve_max_frac"),
    Nuisance(
        "Cut-out locating sigma", "RESERVE_SIGMA_REAL",
        "realfit module constant 0.5 MJ",
        "the particle likelihood charged at a detected deployment cut-out",
        value=5.0e5, low=2.5e5, high=1.0e6,
        reason_for_range="halving and doubling the only observation-noise knob in "
                         "the energy likelihood",
        p3_status=SYNTHETIC, static=True, currently_inferred=False,
        affects_energy_directly=True, constrainable_now=False,
        constrainable_note="a synthetic calibration knob, not a battery "
                           "parameter; nothing in the corpus measures it",
        override="reserve_sigma_j"),
)

# Deliberately NOT nuisances of this estimator. Recorded so the exclusion is a
# decision on the page rather than an oversight a reader has to detect.
NOT_REACHED = {
    "tyres / grip": "TyreState and the wear model reach the simulator and the "
                    "decision layer. `realfit` never reads them, so calibrating "
                    "tyres cannot move the energy belief.",
    "ClA / downforce": "DISABLED in P3 and absent from _terms, which has no "
                       "normal-load term at all.",
    "wake / dirty air": "an exponential surrogate inside sim/policy; no wake term "
                        "appears in the canonical inference equations.",
    "wetness": "reaches grip in the decision layer; not a term in the power "
               "balance the estimator solves.",
    "p_pass coefficients": "decision layer only; downstream of the belief.",
}


def nuisance_registry() -> dict:
    """JSON-ready registry, plus the exclusions."""
    return {
        "registry_version": REGISTRY_VERSION,
        "inclusion_rule": "appears in the equations realfit evaluates on the "
                          "production path",
        "nuisances": [n.as_dict() for n in NUISANCES],
        "affecting_energy_directly": [n.symbol for n in NUISANCES
                                      if n.affects_energy_directly],
        "fixed_by_assumption": [n.symbol for n in NUISANCES
                                if not n.currently_inferred],
        "constrainable_now": [n.symbol for n in NUISANCES if n.constrainable_now],
        "not_constrainable_now": [n.symbol for n in NUISANCES
                                  if not n.constrainable_now],
        "overridable": {n.symbol: n.override for n in NUISANCES if n.override},
        "not_reached_by_this_estimator": NOT_REACHED,
    }


# ------------------------------------------------- exact sensitivity runner
def _variants(n: Nuisance, window: EstimatorInputWindow) -> list[tuple[str, dict]]:
    """low/high override dicts for one nuisance, or [] if not overridable.

    `rho` is expressed as a FRACTIONAL perturbation because its nominal value is
    the session's own measurement, not a constant — a fixed 1.17/1.23 pair would
    silently be a different perturbation at every circuit.
    """
    if n.override is None:
        return []
    if n.override == "rho":
        base = float(window.assumptions.rho)
        return [("low", {"rho": base * (1.0 + float(n.low))}),
                ("high", {"rho": base * (1.0 + float(n.high))})]
    return [("low", {n.override: float(n.low)}),
            ("high", {n.override: float(n.high)})]


def sensitivity(window: EstimatorInputWindow,
                nuisances: tuple[Nuisance, ...] = NUISANCES,
                base: dict | None = None) -> dict:
    """Rerun the SAME estimator at nominal, low and high for each nuisance.

    This is the measurement P3 could not make. P3 perturbed physics and watched a
    downstream prediction while holding the inferred energy FIXED; here the
    perturbation goes through the estimator, so the reported deltas are
    `d(inferred energy)/d(assumption)` rather than `d(prediction)/d(assumption)`.
    """
    # `base` selects the estimator configuration the perturbations are measured
    # AROUND. Part 1 measured around the clipped estimator, where the belief was
    # often pinned and therefore could not move; Part 2 remeasures around the
    # hardened one.
    base = dict(base or {})
    nominal = reinfer(window, **base)
    rows = []
    for n in nuisances:
        for label, ov in _variants(n, window):
            r = reinfer(window, **{**base, **ov})
            rows.append({
                "nuisance": n.symbol, "name": n.name, "arm": label,
                "override": ov,
                "feasible": bool(r.feasible),
                "cda_hat": r.cda_hat, "cda_set_width": r.cda_set_width,
                "identifiability": r.identifiability,
                "usable_mean_j": r.usable_mean_j,
                "usable_particle_width_j": r.usable_particle_width_j,
                "deploy_mean_w": r.deploy_mean_w,
                "balance": r.balance,
                "d_usable_mean_j": r.usable_mean_j - nominal.usable_mean_j,
                "d_usable_width_j": (r.usable_particle_width_j
                                     - nominal.usable_particle_width_j),
                "d_deploy_mean_w": r.deploy_mean_w - nominal.deploy_mean_w,
                "d_cda_hat": r.cda_hat - nominal.cda_hat,
            })
    return {"window_id": window.window_id,
            "cutoff_time_s": window.cutoff_time_s,
            "nominal": {
                "cda_hat": nominal.cda_hat, "cda_set_width": nominal.cda_set_width,
                "identifiability": nominal.identifiability,
                "usable_mean_j": nominal.usable_mean_j,
                "usable_particle_width_j": nominal.usable_particle_width_j,
                "deploy_mean_w": nominal.deploy_mean_w,
                "balance": nominal.balance,
                "n_binding": nominal.n_binding, "n_coast": nominal.n_coast},
            "arms": rows}


def rank_nuisances(sens: list[dict]) -> list[dict]:
    """Rank by the largest |Δ inferred energy| a defensible perturbation causes.

    Ranked on the SPAN between the low and high arms rather than on either arm
    alone: a nuisance that moves energy 0.4 MJ down and 0.4 MJ up matters more
    than one that moves it 0.4 MJ down and not at all up, and a one-sided
    summary would rank them equal.
    """
    by: dict[str, dict] = {}
    for s in sens:
        nom = s["nominal"]["usable_mean_j"]
        for a in s["arms"]:
            e = by.setdefault(a["nuisance"], {
                "nuisance": a["nuisance"], "name": a["name"],
                "n_windows": 0, "abs_d_energy_j": [], "d_low_j": [],
                "d_high_j": [], "abs_d_width_j": [], "abs_d_deploy_w": [],
                "infeasible_arms": 0})
            if not a["feasible"] or not np.isfinite(a["usable_mean_j"]):
                e["infeasible_arms"] += 1
                continue
            d = a["usable_mean_j"] - nom
            e["abs_d_energy_j"].append(abs(d))
            (e["d_low_j"] if a["arm"] == "low" else e["d_high_j"]).append(d)
            e["abs_d_width_j"].append(abs(a["d_usable_width_j"]))
            e["abs_d_deploy_w"].append(abs(a["d_deploy_mean_w"]))
    out = []
    for e in by.values():
        lo = float(np.mean(e["d_low_j"])) if e["d_low_j"] else float("nan")
        hi = float(np.mean(e["d_high_j"])) if e["d_high_j"] else float("nan")
        out.append({
            "nuisance": e["nuisance"], "name": e["name"],
            "mean_abs_d_energy_mj": (float(np.mean(e["abs_d_energy_j"])) / 1e6
                                     if e["abs_d_energy_j"] else float("nan")),
            "max_abs_d_energy_mj": (float(np.max(e["abs_d_energy_j"])) / 1e6
                                    if e["abs_d_energy_j"] else float("nan")),
            "mean_d_low_mj": lo / 1e6, "mean_d_high_mj": hi / 1e6,
            "span_mj": (abs(hi - lo) / 1e6 if np.isfinite(lo) and np.isfinite(hi)
                        else float("nan")),
            "mean_abs_d_width_mj": (float(np.mean(e["abs_d_width_j"])) / 1e6
                                    if e["abs_d_width_j"] else float("nan")),
            "mean_abs_d_deploy_kw": (float(np.mean(e["abs_d_deploy_w"])) / 1e3
                                     if e["abs_d_deploy_w"] else float("nan")),
            "infeasible_arms": e["infeasible_arms"]})
    out.sort(key=lambda r: (-(r["span_mj"] if np.isfinite(r["span_mj"]) else -1.0),
                            -(r["mean_abs_d_energy_mj"]
                              if np.isfinite(r["mean_abs_d_energy_mj"]) else -1.0)))
    return out
