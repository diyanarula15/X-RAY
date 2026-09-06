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

from xray.balance import build_window_constraints, fuel_closure_constraint
from xray.observe import observe
from xray.constants import P_ICE_MAX
from xray.regs import N_RPM_MAX, POST_MIAMI, PRE_MIAMI
from xray.setmem import (DEFAULT_BOX, IDENT_REF, chebyshev_centre, identify,
                         intersect, prune_redundant)
from xray.sim import LEADER

from tests.simfix import elevation_fn, observed_sigma, true_theta

WINDOW_S = 8.0    # measured sweet spot on this fixture: the CdA_X upper bound
                  # falls from 1.057 m^2 at 0.6 s to 0.744 m^2 at 8 s, then
                  # loosens again past 15 s as whole-lap windows average the
                  # informative high-speed running away.


def _cons(cfg, gt, rate=3.7, window_s=WINDOW_S, eta=None, seed=43,
          regs=PRE_MIAMI, ice_floor_delta=None, fuel_burned_kg=None):
    track = gt.track
    z = elevation_fn(track)
    reg = gt.cars[LEADER].regime
    brake = (reg == "brake").astype(float)
    obs = observe(gt, LEADER, rate_hz=rate,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=seed)
    idx = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)
    n = len(obs.v)
    v = cfg["vehicle"]
    # A faithful throttle channel. The naive proxy -- 1.0 whenever off the
    # brakes -- is safe for an upper bound, because overstating the pedal only
    # loosens it, but it is NOT safe for an ICE floor: it would claim 360 kW in
    # a corner where the simulator ran 100 kW and exclude the truth. Stage 1
    # publishes no throttle, so this stands in for the pedal a real feed does
    # publish. Used only as a gate, never as an energy.
    throttle = np.clip(gt.cars[LEADER].P_ice[idx] / P_ICE_MAX, 0.0, 1.0)
    lap_frac = obs.lap + obs.s / track.length
    is_x = ~np.asarray(track.is_corner(obs.s))
    cons = build_window_constraints(
        obs.v, obs.t, z(obs.s), lap_frac, is_x, np.ones(n, bool),
        np.full(n, N_RPM_MAX), throttle, brake[idx], np.full(n, np.nan),
        regs, v["rho"], window_s=window_s,
        eta_d=eta or v["drivetrain_eff"], m_published=v["mass_car"],
        speed_sigma_ms=observed_sigma(cfg), fuel_start=v["fuel_start"],
        fuel_burn_per_lap=v["fuel_burn_per_lap"],
        ice_floor_delta=ice_floor_delta)
    if fuel_burned_kg is not None:
        cons = intersect(cons, fuel_closure_constraint(
            obs.v, obs.t, z(obs.s), lap_frac, is_x, None, v["rho"],
            fuel_burned_kg, brake=brake[idx], eta_d=eta or v["drivetrain_eff"],
            m_published=v["mass_car"], fuel_start=v["fuel_start"],
            fuel_burn_per_lap=v["fuel_burn_per_lap"]))
    return cons


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


def test_a_speed_trace_alone_gives_no_lower_bound_on_drag(cfg, races):
    """The structural result, isolated: with P_ice >= 0 there is no floor.

    "Engine at idle, motor harvesting" explains every straight, so CdA = 0 is
    feasible. Measured: a 3 s coast shedding 5 m/s dissipates 0.267 MJ where the
    rules permit 0.75 MJ of recovery, and post-Miami super-clipping raises that
    to 1.05 MJ. The integral caps do not rescue it either -- over a closed lap
    the kinetic term vanishes, so the balance bounds drag above and is trivially
    satisfied at zero below.

    delta = 1.0 switches the ICE floor off, which is exactly the P_ice >= 0
    assumption. The bound must then sit on the prior edge.
    """
    S = identify(_cons(cfg, races[42], ice_floor_delta=1.0))
    assert S.at_box_edge[0], (
        "CdA_X has a lower bound with no claim about ICE output, which "
        "contradicts the energy argument above")
    assert S.lo[0] == pytest.approx(DEFAULT_BOX[0, 0])
    assert S.hi[0] < DEFAULT_BOX[0, 1]


def test_the_local_ice_floor_works_pre_miami_and_dies_post_miami(cfg, races):
    """The lower bound has to come from a claim about ICE output. ASSUMED.

    At full throttle P_ice >= (1 - delta) P_ice_max. Pre-Miami the harvest cap
    is 250 kW, so 400(1-delta) - 250 must go somewhere; post-Miami the cap is
    350 kW and the same arithmetic leaves 10 kW at delta = 0.10. Measured
    identified floors on CdA_X, true value 0.660:

        delta   pre-Miami    post-Miami
        0.00      0.134      prior edge
        0.05      0.095      prior edge
        0.10      0.057      prior edge
        0.20    prior edge   prior edge

    So the April rule change made drag area lower-unidentifiable from power
    bounds alone. That is a fact about the regulation, not about the estimator.
    """
    gt = races[42]
    tight = identify(_cons(cfg, gt, regs=PRE_MIAMI, ice_floor_delta=0.0))
    loose = identify(_cons(cfg, gt, regs=PRE_MIAMI, ice_floor_delta=0.20))
    post = identify(_cons(cfg, gt, regs=POST_MIAMI, ice_floor_delta=0.0))
    assert not tight.at_box_edge[0], "delta = 0 gives no floor pre-Miami"
    assert tight.lo[0] > DEFAULT_BOX[0, 0] + 1e-6
    assert loose.at_box_edge[0], "delta = 0.20 should leave nothing to stand on"
    assert post.at_box_edge[0], (
        "post-Miami super-clipping should leave the local floor useless; if this "
        "now passes, check the harvest cap")


def test_fuel_closure_gives_a_floor_no_harvest_rule_can_evade(cfg, races):
    """The global bound, and the one that survives the rule change.

    Over the stint the ICE does a known amount of work and it has nowhere to go
    but drag, rolling resistance, the friction brakes and the store -- and the
    store cannot absorb more than 4 MJ net. Measured on seed 42: E_ice_min
    296 MJ against E_friction_max 166 MJ, giving CdA_X >= 0.264 with the HMM's
    own mode labels and 0.396 with the circuit's aero gate. True value 0.660.

    Two inputs make or break it, and both were wrong first time round:
    the fuel-mass uncertainty must be fractional (a +/-5 kg race figure is 60%
    of a 12-lap stint and killed the constraint by itself, taking E_ice_min from
    296 MJ to 127 MJ), and the friction bound must be restricted to intervals
    where the brakes are actually on (including coast-downs inflated it from
    100 MJ to 177 MJ, which alone made the constraint vacuous).
    """
    gt = races[42]
    v = cfg["vehicle"]
    without = identify(_cons(cfg, gt, ice_floor_delta=1.0))
    with_fuel = identify(_cons(cfg, gt, ice_floor_delta=1.0,
                               fuel_burned_kg=v["fuel_burn_per_lap"] * 12))
    assert without.at_box_edge[0]
    assert not with_fuel.at_box_edge[0], "fuel closure produced no floor"
    assert with_fuel.lo[0] > 0.2, f"floor only {with_fuel.lo[0]:.3f} m^2"
    assert with_fuel.lo[0] < 0.660, "the floor exceeds the truth"
    th = true_theta(cfg)
    assert ((th >= with_fuel.lo - 1e-9) & (th <= with_fuel.hi + 1e-9)).all()


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
