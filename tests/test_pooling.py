"""Step 7: partial pooling across the field, in closed form."""
from __future__ import annotations

import numpy as np
import pytest

from xray.pooling import CarEstimate, needs_mcmc, pool, teammate_consistency

TAU_TRUE = 0.012      # teammates share an aero package, not a setup
TEAM_SPREAD = 0.04    # teams genuinely differ


def _field(seed=0, n_teams=10):
    rng = np.random.default_rng(seed)
    true_team = {f"T{i}": 0.90 + TEAM_SPREAD * rng.standard_normal()
                 for i in range(n_teams)}
    est, truth = [], {}
    for t, mu in true_team.items():
        for c in (0, 1):
            sigma = float(10 ** rng.uniform(-2.3, -1.0))   # 0.005 .. 0.10 m^2
            car_true = mu + TAU_TRUE * rng.standard_normal()
            truth[f"{t}_{c}"] = car_true
            est.append(CarEstimate(f"{t}_{c}", t,
                                   car_true + sigma * rng.standard_normal(),
                                   sigma, True))
    return est, truth


def _mae(est, truth, values=None):
    if values is None:
        return float(np.mean([abs(e.cda - truth[e.car]) for e in est]))
    return float(np.mean([abs(values[e.car] - truth[e.car]) for e in est]))


def test_pooling_halves_the_error(cfg):
    """The claim. Measured 48% on a 20-car field."""
    est, truth = _field()
    p = pool(est)
    own = _mae(est, truth)
    pooled = _mae(est, truth, p.per_car)
    assert pooled < 0.6 * own, f"pooled MAE {pooled:.4f} against own {own:.4f}"


def test_shrinking_to_the_field_mean_is_worse_than_two_levels(cfg):
    """Why the hierarchy has two levels and not one.

    Collapsing every car to the field mean ignores that teams genuinely differ.
    Measured on a fixture where teammates agreed to within measurement error but
    teams differed by 0.04 m^2, one-level pooling made the error 5% *worse* than
    not pooling at all.
    """
    est, truth = _field()
    p = pool(est)
    two_level = _mae(est, truth, p.per_car)
    one_level = _mae(est, truth, {e.car: p.mu_field for e in est})
    assert two_level < one_level, (
        f"two-level {two_level:.4f} is no better than shrinking to the field "
        f"mean {one_level:.4f}")


def test_only_the_badly_measured_cars_move(cfg):
    """Borrowing strength must be proportionate. A car with a tight identified
    set keeps its own answer; a car with a loose one leans on its teammate."""
    est, truth = _field()
    p = pool(est)
    loose = [e for e in est if e.sigma > 0.03]
    tight = [e for e in est if e.sigma < 0.01]
    assert loose and tight
    k_loose = np.mean([p.shrinkage[e.car] for e in loose])
    k_tight = np.mean([p.shrinkage[e.car] for e in tight])
    assert k_tight > 0.9, f"well-measured cars gave up their own data ({k_tight:.2f})"
    assert k_loose < 0.7, f"badly-measured cars ignored the field ({k_loose:.2f})"
    gain_loose = np.mean([abs(e.cda - truth[e.car])
                          - abs(p.per_car[e.car] - truth[e.car]) for e in loose])
    gain_tight = np.mean([abs(e.cda - truth[e.car])
                          - abs(p.per_car[e.car] - truth[e.car]) for e in tight])
    assert gain_loose > 10 * abs(gain_tight)


def test_teammates_must_agree_and_disagreement_names_its_suspect(cfg):
    """A falsification test that needs no ground truth: two cars of one team run
    the same aero package. When they disagree the first suspect is not the drag
    estimate, it is the aero-mode labelling -- a car whose straights were
    assigned to CdA_Z has its high-speed running on the wrong parameter."""
    est, _truth = _field()
    clean = teammate_consistency(est, pool(est))
    assert len(clean) == 10
    assert all(c["consistent"] for c in clean)

    bad = list(est)
    bad[0] = CarEstimate(bad[0].car, bad[0].team, bad[0].cda * 2, bad[0].sigma, True)
    flagged = [c for c in teammate_consistency(bad, pool(bad)) if not c["consistent"]]
    assert len(flagged) == 1
    assert "aero-mode labelling" in flagged[0]["suspect"]


def test_a_car_with_no_high_speed_running_inherits_and_is_flagged(cfg):
    """Monaco's problem, solved by the field rather than by a guess -- and the
    car is marked as having borrowed, so nobody quotes it as its own result."""
    est, _truth = _field()
    est = est + [CarEstimate("X_0", "X", float("nan"), float("inf"), False)]
    p = pool(est)
    assert "X_0" in p.inherited
    assert p.per_car["X_0"] == pytest.approx(p.mu_field)
    assert p.shrinkage["X_0"] == 0.0
    assert p.per_car_sigma["X_0"] > max(
        v for k, v in p.per_car_sigma.items() if k != "X_0")


def test_complete_pooling_is_recognised_not_crashed_into(cfg):
    """If teammate scatter is entirely measurement error, tau_team is zero and
    pooling is complete. The method-of-moments estimate goes negative there, so
    the estimator must say so rather than take a square root of it."""
    rng = np.random.default_rng(1)
    est = []
    for i in range(4):
        mu = 0.9 + 0.03 * rng.standard_normal()
        for c in (0, 1):
            est.append(CarEstimate(f"T{i}_{c}", f"T{i}",
                                   mu + 0.02 * rng.standard_normal(), 0.02, True))
    p = pool(est, tau_team=None)
    assert np.isfinite(p.tau_team)
    assert p.tau_team >= 0.0


def test_refuses_a_field_too_small_to_pool(cfg):
    """Two cars are not a field."""
    est = [CarEstimate("a", "T", 0.9, 0.01, True), CarEstimate("b", "T", 0.91, 0.01, True)]
    p = pool(est)
    assert p.per_car == {}
    assert any("nothing to pool" in n for n in p.notes)


def test_the_closed_form_choice_is_revisitable(cfg):
    """NUTS is specified in the plan and deliberately not used, because for a
    Gaussian hierarchy with known per-car variances the posterior is conjugate
    and a sampler would reproduce it with added error and a jax dependency.
    The cases that break conjugacy are named rather than left implicit."""
    assert needs_mcmc({}) == ()
    triggers = needs_mcmc({"pool_policy_params": True, "non_gaussian_prior": True})
    assert len(triggers) == 2
    assert any("not conjugate" in t for t in triggers)
