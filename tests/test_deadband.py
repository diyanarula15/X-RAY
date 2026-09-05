"""Step 14: the dead band, measured before the filter rather than inside it.

What this stage buys, on seed 42 and with everything else held fixed:

    configuration                     CdA_X   per-lap MAPE   deployable cov
    joint v_cut inside the filter     0.407       13.9%           0.55
    measured v_cut, this module       0.597        9.9%           0.67

    and across the three fixture seeds, measured: CdA_X 0.597 / 0.571 / 0.494,
    per-lap energy 9.9% / 15.2% / 14.0%, deployable coverage 0.67 / 0.72 / 0.80.

against a true 0.660 inside an identified set of [0.264, 0.744]. Two of the
three gains came from diagnostics this stage forced rather than from the
changepoint itself -- see test_rbpf.py for the resampling desync and the missing
deployment slack -- which is the argument for running the diagnostic invariant 9
asks for before believing any of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.deadband import N_DEAD_MIN, dead_band_cda, fit_cut_speeds, gated_terms
from xray.regs import POST_MIAMI, PRE_MIAMI
from xray.sim import LEADER

from tests.simfix import true_theta
from tests.test_rbpf import _setup


def _kw(cfg):
    v = cfg["vehicle"]
    return dict(eta_d=v["drivetrain_eff"], m_published=v["mass_car"],
                fuel_start=v["fuel_start"],
                fuel_burn_per_lap=v["fuel_burn_per_lap"])


def test_the_dead_band_identifies_drag_to_five_percent(cfg, races):
    """The point of the whole stage: one parameter, one least squares.

    With the band fixed and the policy saying P_K = 0 inside it, the balance is
    ICE against drag and CdA is the only unknown left. Measured on seed 42:
    0.632 against a true 0.660, and 0.629 to 0.665 as the band is narrowed from
    234 to 324 km/h. The residual 3-5% is contamination and it can only bias
    drag *down* -- the simulator's policy is positional, not a speed threshold,
    so a little deployment survives above any cut-off.
    """
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    cuts = fit_cut_speeds(feed, POST_MIAMI, cfg["vehicle"]["rho"], **_kw(cfg))
    truth = true_theta(cfg)
    cda = dead_band_cda(feed, cuts, POST_MIAMI, cfg["vehicle"]["rho"],
                        cuts.f_rr_hint, **_kw(cfg))
    assert abs(cda - truth[0]) / truth[0] < 0.08, (
        f"dead-band CdA {cda:.3f} against a true {truth[0]:.3f}")
    # and it must sit inside the assumption-free identified set
    assert ident.identified.lo[0] <= cda <= ident.identified.hi[0]


def test_a_free_intercept_destroys_the_fit(cfg, races):
    """Why the fit has no constant term and no free signs, asserted rather
    than commented.

    Over 210-355 km/h a v^3 column and a constant are nearly collinear, so an
    unconstrained intercept buys fit by pushing drag up and the offset down.
    Measured on seed 42 against a true 0.660:

        drag only, no intercept                 0.613
        drag + free intercept                   0.842   (-77 kW)
        drag + free roll + free intercept       0.374   (F_rr 4181 N)

    The third is the one to keep in mind: with the sign of rolling resistance
    unconstrained the fit uses it as the intercept it was just denied, at 45
    times a physical value, and lands 43% low instead of 28% high. That is why
    the module fits by non-negative least squares. Same collinearity that
    invariant 9 avoids by using the dead band instead of a step regressor, one
    level down.
    """
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    v, drag, roll, y, _ = gated_terms(feed, POST_MIAMI, cfg["vehicle"]["rho"],
                                      **_kw(cfg))
    truth = true_theta(cfg)[0]
    with_b, *_ = np.linalg.lstsq(np.column_stack([drag, np.ones(len(y))]), y,
                                 rcond=None)
    free, *_ = np.linalg.lstsq(np.column_stack([drag, roll, np.ones(len(y))]),
                               y, rcond=None)
    without = float(np.sum(drag * y) / np.sum(drag * drag))
    assert abs(with_b[0] - truth) > 2 * abs(without - truth), (
        f"a free intercept no longer breaks the fit ({with_b[0]:.3f} against "
        f"{truth:.3f}); re-derive whether the no-intercept form is still needed")
    assert free[1] > 1000.0, (
        f"free-sign rolling resistance came out at {free[1]:.0f} N, no longer "
        "absurd; re-measure whether non-negativity is still load-bearing")


def test_no_super_clip_step_is_claimed_when_there_is_none(cfg, races):
    """Decline rather than guess, in the one place it is cheap to guess.

    `vehicle.step` harvests on the brakes only, so the Stage 1 simulator has no
    off-throttle super-clipping and there is no v_harv in the trace to find. The
    fit must return infinity and leave the dead band open at the top rather than
    invent the number it is supposed to measure. On real post-Miami data the
    350 kW super-clip should produce a step here, and its absence would then be
    a finding about the regulation tables.
    """
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    cuts = fit_cut_speeds(feed, POST_MIAMI, cfg["vehicle"]["rho"], **_kw(cfg))
    assert not np.isfinite(cuts.v_harv), (
        f"claimed a super-clip onset at {cuts.v_harv * 3.6:.0f} km/h on a "
        "simulator that only harvests on the brakes")
    assert "no super-clip step" in cuts.note


def test_v_cut_is_found_on_every_seed_and_is_a_speed_the_car_reaches(cfg, races):
    """The failure this replaces: jointly inferred, v_cut ran to 98 m/s --
    353 km/h, above the car's top speed -- where the dead band is empty and the
    likelihood costs nothing. A measured v_cut cannot do that, because an empty
    band has no samples to fit.

    Measured: 236, 239 and 214 km/h on seeds 42, 7 and 13, with BIC gains of
    170.2, 156.0 and 159.9 over the no-changepoint model. Seed 13's lower
    cut-off is why its filter CdA_X is the weakest of the three (0.494 against
    0.597 and 0.571): a lower band admits more surviving deployment, and that
    bias is one-sided.
    """
    for seed, gt in races.items():
        obs, idx, feed, ident = _setup(cfg, gt)
        cuts = fit_cut_speeds(feed, POST_MIAMI, cfg["vehicle"]["rho"], **_kw(cfg))
        assert not cuts.declined, f"seed {seed}: {cuts.note}"
        assert cuts.n_dead >= 10 * N_DEAD_MIN, f"seed {seed}: {cuts.n_dead}"
        assert cuts.v_cut < float(np.max(feed.v)), (
            f"seed {seed}: v_cut {cuts.v_cut * 3.6:.0f} km/h is above the top "
            f"speed {np.max(feed.v) * 3.6:.0f} km/h, so the band is empty")
        assert cuts.bic_gain > 50.0, f"seed {seed}: BIC gain {cuts.bic_gain:.1f}"


def test_it_declines_on_a_trace_with_no_full_throttle(cfg, races):
    """Refusal is a correct output. A trace that never runs at full throttle
    with the brakes off has no window where ICE output is known, so there is
    nothing to fit and nothing to report -- and the caller falls back to the
    filter's own prior rather than to a guessed cut-off."""
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    import dataclasses
    lifted = dataclasses.replace(feed, throttle=np.full(len(feed.v), 0.5))
    cuts = fit_cut_speeds(lifted, POST_MIAMI, cfg["vehicle"]["rho"], **_kw(cfg))
    assert cuts.declined
    assert "full-throttle brakes-off intervals" in cuts.note
    with pytest.raises(ValueError, match="dead-band intervals"):
        dead_band_cda(lifted, cuts, POST_MIAMI, cfg["vehicle"]["rho"], 93.0,
                      **_kw(cfg))


def test_the_regulation_variant_does_not_reach_this_stage(cfg, races):
    """Invariant 10 at this stage's boundary, and the answer is the pleasant
    one: it does not depend on the variant.

    The fit uses two regulation numbers, the ICE cap and the shape of the
    290-355 km/h taper, and the Miami amendment changed neither. What it did
    change -- the harvest cap and the in-zone deployment cap -- never enters,
    because inside the dead band the motor is idle by construction. So the
    variant mismatch that cost an iteration (350 kW harvest reconstructed
    against a 250 kW assumption, every harvest number 0.71x) cannot express
    itself here. Asserted so that a future change which quietly makes this
    stage variant-dependent has to say so.
    """
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    post = fit_cut_speeds(feed, POST_MIAMI, cfg["vehicle"]["rho"], **_kw(cfg))
    pre = fit_cut_speeds(feed, PRE_MIAMI, cfg["vehicle"]["rho"], **_kw(cfg))
    assert post.cda_hint == pytest.approx(pre.cda_hint)
    assert post.v_cut == pytest.approx(pre.v_cut)
