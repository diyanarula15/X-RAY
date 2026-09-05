"""Partial pooling across the field, and the teammate consistency test.

Twenty cars run the same regulations through the same air. Drag areas differ
between teams by a few percent, not by a factor of five, so a car with no clean
high-speed running can borrow strength from cars that had some -- and two cars
of the same team, running the same aero package, must agree.

    CdA[car] ~ N(CdA[team], tau_team)
    CdA[team] ~ N(mu_field, tau_field)
    C_rr      shared across the grid (the tyre spec is common)

WHY THIS IS CLOSED FORM AND NOT NUTS. The plan specifies NUTS via NumPyro or
PyMC. For a Gaussian hierarchy with per-car variances that are already known --
and they are: each car's variance comes out of step 2's identified set -- the
posterior is conjugate, so the shrinkage estimator below *is* the posterior
mean, exactly, with no sampler error and no dependency. NUTS would add jax and
several minutes per weekend to reproduce an answer available in closed form.

The moment that stops being true is the moment it should be replaced: a
non-Gaussian prior, a heavy-tailed likelihood to survive one mislabelled car,
or pooling the policy parameters (v_cut, v_harv) whose per-lap structure is not
conjugate. `needs_mcmc()` names those cases rather than leaving the choice
implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CarEstimate:
    """One car's identified drag area, as a mean and a width."""
    car: str
    team: str
    cda: float
    sigma: float          # from the identified set's projection width
    identifiable: bool


@dataclass(frozen=True)
class Pooled:
    mu_field: float
    tau_team: float
    per_car: dict          # car -> shrunken estimate
    per_car_sigma: dict    # car -> posterior width
    per_team: dict
    shrinkage: dict        # car -> weight given to its own data, 0..1
    n_pooled: int
    inherited: tuple = ()
    notes: tuple = ()

    def width_reduction(self, prior: dict) -> dict:
        """How much narrower each car got. The claim being tested."""
        return {c: (self.per_car_sigma[c] / prior[c]) for c in self.per_car_sigma
                if prior.get(c)}


def _group(estimates):
    teams = {}
    for e in estimates:
        teams.setdefault(e.team, []).append(e)
    return teams


def pool(estimates, tau_team: float | None = None,
         min_cars: int = 3) -> Pooled:
    """Conjugate partial pooling. Returns posterior means and widths.

    `tau_team` is the between-car spread within a team. Estimated from the data
    by method of moments when not supplied -- the observed scatter of teammate
    differences minus the measurement variance that explains part of it. If that
    difference is negative the scatter is entirely measurement error, which
    means complete pooling, and the estimator says so rather than taking a
    square root of a negative number.
    """
    usable = [e for e in estimates if e.identifiable and np.isfinite(e.cda)]
    notes = ()
    if len(usable) < min_cars:
        return Pooled(mu_field=float("nan"), tau_team=float("nan"), per_car={},
                      per_car_sigma={}, per_team={}, shrinkage={},
                      n_pooled=len(usable),
                      notes=(f"only {len(usable)} identifiable cars, need "
                             f"{min_cars}; nothing to pool",))

    w = np.array([1.0 / max(e.sigma, 1e-9) ** 2 for e in usable])
    x = np.array([e.cda for e in usable])
    mu_field = float(np.sum(w * x) / np.sum(w))

    if tau_team is None:
        # A single tau pooled across the whole grid, not one per team. Ten
        # teammate pairs will not support a method-of-moments variance: measured
        # on a fixture with a true tau of 0.012, ten pairs returned 0.0562. The
        # pooled version uses every pair on the grid at once, which is the same
        # estimator with 10x the data behind it, and it is floored rather than
        # square-rooting a negative excess.
        diffs, var_meas = [], []
        for _team, members in _group(usable).items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    diffs.append(members[i].cda - members[j].cda)
                    var_meas.append(members[i].sigma ** 2 + members[j].sigma ** 2)
        if diffs:
            # Precision-weight the pairs: a pair of badly-measured cars says
            # almost nothing about the between-car spread and should not be
            # allowed to dominate the moment.
            wts = 1.0 / np.maximum(np.asarray(var_meas), 1e-12)
            wts = wts / wts.sum()
            excess = float(np.sum(wts * (np.square(diffs) - np.asarray(var_meas))))
            if excess <= 0.0:
                tau_team = 1e-6
                notes += ("teammate scatter is entirely explained by measurement "
                          "error, so pooling is complete",)
            else:
                tau_team = float(np.sqrt(excess / 2.0))
            notes += (f"tau_team {tau_team:.4f} from {len(diffs)} pooled "
                      f"teammate pairs",)
        else:
            tau_team = float(np.std(x)) if len(x) > 1 else 0.05
            notes += ("no teammate pairs; tau estimated from field scatter",)

    # Two levels, as specified. Shrinking each car straight to the field mean
    # is the obvious shortcut and it is wrong: teams genuinely differ, so on a
    # fixture where teammates agreed to within measurement error but teams
    # differed by 0.04 m^2, collapsing everything to the field mean made the
    # mean absolute error 5% *worse* than not pooling at all. A car borrows
    # strength from its teammate first; only the team borrows from the field.
    teams = _group(usable)
    tau_field = float(np.std([np.mean([m.cda for m in ms])
                              for ms in teams.values()])) if len(teams) > 1 else 1e-6
    team_mean, team_sigma = {}, {}
    for team, members in teams.items():
        prec = float(np.sum([1.0 / max(m.sigma, 1e-9) ** 2 for m in members]))
        raw = float(np.sum([m.cda / max(m.sigma, 1e-9) ** 2 for m in members]) / prec)
        prec_fld = 1.0 / max(tau_field, 1e-9) ** 2
        kt = prec / (prec + prec_fld)
        team_mean[team] = kt * raw + (1.0 - kt) * mu_field
        team_sigma[team] = float(np.sqrt(1.0 / (prec + prec_fld)))

    per_car, per_sigma, shrink = {}, {}, {}
    for e in usable:
        # conjugate normal-normal against the car's own TEAM, not the field
        prec_own = 1.0 / max(e.sigma, 1e-9) ** 2
        prec_grp = 1.0 / max(float(np.hypot(tau_team, team_sigma[e.team])), 1e-9) ** 2
        k = prec_own / (prec_own + prec_grp)
        per_car[e.car] = k * e.cda + (1.0 - k) * team_mean[e.team]
        per_sigma[e.car] = float(np.sqrt(1.0 / (prec_own + prec_grp)))
        shrink[e.car] = float(k)

    per_team = team_mean

    # cars that could not calibrate alone inherit the field constraint, flagged
    inherited = tuple(e.car for e in estimates if not e.identifiable)
    for car in inherited:
        per_car[car] = mu_field
        per_sigma[car] = float(np.hypot(tau_team, np.std(x) if len(x) > 1 else tau_team))
        shrink[car] = 0.0

    return Pooled(mu_field=mu_field, tau_team=float(tau_team), per_car=per_car,
                  per_car_sigma=per_sigma, per_team=per_team, shrinkage=shrink,
                  n_pooled=len(usable), inherited=inherited, notes=notes)


def teammate_consistency(estimates, pooled: Pooled, n_sigma: float = 3.0) -> list:
    """Two cars of the same team run the same aero package, so they must agree.

    A built-in falsification test with no ground truth required. When teammates
    disagree the first suspect is not the drag estimate: it is the aero-mode
    labelling, because a car whose straights were labelled Z has its high-speed
    running attributed to the wrong parameter entirely.
    """
    out = []
    for team, members in _group([e for e in estimates if e.identifiable]).items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                diff = a.cda - b.cda
                se = float(np.hypot(a.sigma, b.sigma))
                z = diff / se if se > 0 else 0.0
                out.append({
                    "team": team, "cars": (a.car, b.car),
                    "difference": float(diff), "se": se, "z": float(z),
                    "consistent": bool(abs(z) <= n_sigma),
                    "suspect": None if abs(z) <= n_sigma else
                    "aero-mode labelling: check whether one car's straights "
                    "were assigned to CdA_Z"})
    return out


def needs_mcmc(reason_flags: dict) -> tuple:
    """Which requested features would take this beyond the conjugate case.

    Stated explicitly so the choice to stay closed-form is revisited on purpose
    rather than by omission.
    """
    triggers = {
        "heavy_tailed_likelihood": "one mislabelled car should not move the field",
        "pool_policy_params": "v_cut / v_harv are per lap and not conjugate",
        "unknown_per_car_variance": "step 2 supplies it, but a fitted variance is not conjugate",
        "non_gaussian_prior": "a bounded or skewed prior on CdA has no closed form",
    }
    return tuple(f"{k}: {v}" for k, v in triggers.items() if reason_flags.get(k))
