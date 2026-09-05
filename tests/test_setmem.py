"""Step 2: the feasible set as a polytope, and what it can and cannot pin down.

Two properties are asserted here and they pull in opposite directions:

  containment -- the true theta must lie inside the reported set, always. A
    set-membership estimate that excludes the truth is not conservative, it is
    wrong, and every downstream number inherits the error.

  informativeness -- the set must be narrower than the prior box, or the method
    has told us nothing.

The tests also pin the *shape* of what is identifiable, because that shape is a
result rather than an implementation detail: drag area is bounded above and not
below. See `test_drag_has_no_lower_bound_from_a_speed_trace`.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.balance import build_window_constraints
from xray.observe import observe
from xray.regs import N_RPM_MAX, PRE_MIAMI
from xray.setmem import (DEFAULT_BOX, IDENT_REF, chebyshev_centre, identify,
                         intersect, prune_redundant)
from xray.sim import LEADER

from tests.simfix import elevation_fn, observed_sigma, true_theta

WINDOW_S = 8.0    # measured sweet spot on this fixture: the CdA_X upper bound
                  # falls from 1.057 m^2 at 0.6 s to 0.744 m^2 at 8 s, then
                  # loosens again past 15 s as whole-lap windows average the
                  # informative high-speed running away.


def _cons(cfg, gt, rate=3.7, window_s=WINDOW_S, eta=None, seed=43):
    track = gt.track
    z = elevation_fn(track)
    reg = gt.cars[LEADER].regime
    brake = (reg == "brake").astype(float)
    # A real feed publishes the pedal. Off the brakes this overstates it to 1.0,
    # which can only loosen an upper bound, so containment is preserved.
    throttle = np.where(reg == "brake", 0.0, 1.0)
    obs = observe(gt, LEADER, rate_hz=rate,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=seed)
    idx = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)
    n = len(obs.v)
    v = cfg["vehicle"]
    return build_window_constraints(
        obs.v, obs.t, z(obs.s), obs.lap + obs.s / track.length,
        ~np.asarray(track.is_corner(obs.s)), np.ones(n, bool),
        np.full(n, N_RPM_MAX), throttle[idx], brake[idx], np.full(n, np.nan),
        PRE_MIAMI, v["rho"], window_s=window_s,
        eta_d=eta or v["drivetrain_eff"], m_published=v["mass_car"],
        speed_sigma_ms=observed_sigma(cfg), fuel_start=v["fuel_start"],
        fuel_burn_per_lap=v["fuel_burn_per_lap"])


def test_identified_set_contains_the_truth(cfg, races):
    """The property everything else rests on, at three sample rates."""
    th = true_theta(cfg)
    for rate in (3.7, 20.0, 100.0):
        S = identify(_cons(cfg, races[42], rate=rate))
        assert not S.empty, f"{rate} Hz: set empty on data with a known theta"
        inside = (th >= S.lo - 1e-9) & (th <= S.hi + 1e-9)
        assert inside.all(), (
            f"{rate} Hz: identified set excludes the truth on "
            f"{[n for n, ok in zip(('CdA_X', 'CdA_Z', 'F_rr', 'dm'), inside) if not ok]}")


def test_the_set_is_narrower_than_its_prior(cfg, races):
    """Containment is free if you report the prior box. This is the other half."""
    S = identify(_cons(cfg, races[42]))
    prior_width = DEFAULT_BOX[0, 1] - DEFAULT_BOX[0, 0]
    assert S.width[0] < 0.75 * prior_width, (
        f"CdA_X range {S.width[0]:.3f} m^2 is no better than the "
        f"{prior_width:.3f} m^2 prior")


def test_drag_has_no_lower_bound_from_a_speed_trace(cfg, races):
    """A structural result, asserted so it cannot be quietly "fixed".

    Nothing in a speed trace bounds drag area from below, because recovery can
    always account for a deceleration. Measured: a 3 s coast shedding 5 m/s
    dissipates 0.267 MJ, while the regulation permits 0.75 MJ of harvesting over
    the same window -- and post-Miami super-clipping raises that to 1.05 MJ. The
    integral caps do not rescue it either: over a closed lap the kinetic term
    vanishes, so the lap balance reads E_drag + E_rr = eta(E_ice + E_K) with
    E_K in [-4, +4] MJ, which bounds drag from above and leaves the lower side
    trivially satisfied at CdA = 0.

    The lower bound therefore requires the store dynamics -- a car cannot
    harvest into a full battery -- which is step 5, not step 2. The polytope and
    the filter are not as separable as the build order assumes.
    """
    S = identify(_cons(cfg, races[42]))
    assert S.at_box_edge[0], (
        "CdA_X now has a data-driven lower bound. That would be a real result, "
        "but it contradicts the energy argument above -- check whether a bound "
        "has been tightened past what the regulation actually says.")
    assert S.lo[0] == pytest.approx(DEFAULT_BOX[0, 0])
    # and the upper bound is the informative side
    assert S.hi[0] < DEFAULT_BOX[0, 1]


def test_empty_set_is_an_alarm_not_a_fit(cfg, races):
    """Claim the wrong drivetrain efficiency and the data must refuse it.

    This is "decline rather than guess" made exact. A thresholded estimator
    cannot tell "no informative data" from "the data contradict the rulebook";
    a polytope reports the first as a wide projection and the second as an
    empty set with the size of the contradiction attached.
    """
    S = identify(_cons(cfg, races[42], eta=0.40))
    assert S.empty, "a 0.40 drivetrain efficiency should contradict the data"
    assert S.min_violation > 0.0
    assert any("cannot be satisfied together" in n for n in S.notes)


def test_outlier_budget_survives_a_single_bad_sample(cfg, races):
    """One corrupted sample must not take the whole race down.

    Hard intersection has no answer to this: measured, the true theta sat
    outside 1.6% of the per-interval lower bounds at 3.7 Hz -- all at braking
    onsets -- and that was enough to report INFEASIBLE with a 228.2 kJ minimum
    violation on data whose parameters are known exactly.
    """
    cons = _cons(cfg, races[42])
    A = cons.A.copy()
    b = cons.b.copy()
    # one window whose budget no admissible theta can meet -- 50 MJ against a
    # 4 MJ store. Chosen well past any real slack so the test exercises the
    # trimming path rather than the slack the constraint already had.
    b[len(b) // 2] -= 5.0e7
    from xray.balance import Constraints
    poisoned = Constraints(A=A, b=b, n_intervals=cons.n_intervals,
                           n_upper=cons.n_upper, n_lower=cons.n_lower,
                           n_dropped=cons.n_dropped, drop_reasons=cons.drop_reasons)
    S = identify(poisoned)
    assert not S.empty, "one bad sample emptied the set"
    assert any("trimmed" in n for n in S.notes)
    th = true_theta(cfg)
    assert ((th >= S.lo - 1e-9) & (th <= S.hi + 1e-9)).all()


def test_chebyshev_centre_is_inside_the_set(cfg, races):
    """A point estimate outside the feasible set violates the regulation, which
    is the one thing this construction exists to prevent. The midpoint of the
    projections does not have that guarantee; the Chebyshev centre does."""
    cons = _cons(cfg, races[42])
    centre, radius = chebyshev_centre(cons)
    assert radius > 0.0
    assert (cons.A @ centre <= cons.b + 1e-6).all()


def test_intersection_is_the_streaming_update(cfg, races):
    """Two half-races intersected must equal the whole race, and adding data can
    only ever narrow the set -- which is what makes a running intersection a
    legitimate streaming estimator."""
    gt = races[42]
    whole = _cons(cfg, gt)
    half = whole.A.shape[0] // 2
    from xray.balance import Constraints
    a = Constraints(A=whole.A[:half], b=whole.b[:half], n_intervals=0,
                    n_upper=half, n_lower=0, n_dropped=0, drop_reasons={})
    joined = intersect(a, Constraints(
        A=whole.A[half:], b=whole.b[half:], n_intervals=0,
        n_upper=0, n_lower=whole.A.shape[0] - half, n_dropped=0, drop_reasons={}))
    assert joined.A.shape == whole.A.shape
    np.testing.assert_array_equal(joined.A, whole.A)
    S_part = identify(a)
    S_full = identify(joined)
    assert S_full.width[0] <= S_part.width[0] + 1e-9, "more data widened the set"


def test_pruning_cannot_tighten_the_set(cfg, races):
    """Bounded memory for a streaming implementation. Dropping slack constraints
    can only widen what is reported, so it can never cause a false refusal."""
    cons = _cons(cfg, races[42])
    S = identify(cons)
    pruned = identify(prune_redundant(cons, S.centre, keep=len(cons) // 4))
    assert len(pruned.notes) >= 0
    assert pruned.width[0] >= S.width[0] - 1e-9
    th = true_theta(cfg)
    assert ((th >= pruned.lo - 1e-9) & (th <= pruned.hi + 1e-9)).all()


def test_identifiability_score_is_scale_free(cfg):
    """dm has a legitimate midpoint of zero, so a width normalised by its own
    midpoint is a division by zero rather than a score."""
    assert (IDENT_REF > 0).all()
    assert len(IDENT_REF) == DEFAULT_BOX.shape[0]
