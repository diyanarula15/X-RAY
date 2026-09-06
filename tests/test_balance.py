"""Step 1: the interval energy balance must CONTAIN the truth.

Set-membership identification only means anything if the constraint set is
guaranteed to hold the true parameters. A set that excludes the truth is not
conservative, it is wrong -- and it fails loudly later, because the polytope in
`setmem.py` goes empty and reports an alarm.

So these tests do not check accuracy. They check containment, on the only data
where a true theta exists.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.balance import (M_REF_KG, QUANT_SIGMA_MS, build_constraints,
                          energy_slack, fuel_mass, tow_f_max)
from xray.observe import observe
from xray.regs import N_RPM_MAX, POST_MIAMI, PRE_MIAMI
from xray.sim import LEADER

from tests.simfix import elevation_fn, observed_sigma, true_theta

# The simulator integrates with semi-implicit Euler at dt = 5 ms and has no
# traction limit, so out of the hairpin it accelerates at 44 m/s^2 on 750 kW at
# the wheels. The Euler error 1/2 m a^2 dt^2 is then 20 J, and those intervals
# sit at *exactly* the regulatory ceiling (400 kW ICE + 350 kW MGU-K = 3562.5 J
# per step), so there is no headroom to absorb it. Measured worst case 24.2 J,
# which is 0.68% of the interval. This is the simulator's integration error, not
# the balance's: the balance is an identity.
SIM_EULER_SLACK_J = 50.0


def _constraints(cfg, gt, v, t, s, lap, brake, sigma, n_sigma=3.0,
                 regs=PRE_MIAMI):
    track = gt.track
    z = elevation_fn(track)
    n = len(v)
    return build_constraints(
        v=v, t=t, z=z(s), lap_frac=lap + s / track.length,
        is_x_mode=~np.asarray(track.is_corner(s)), in_zone=np.ones(n, bool),
        rpm=np.full(n, N_RPM_MAX), throttle=np.ones(n), brake=brake,
        gap_s=np.full(n, np.nan), regs=regs, rho=cfg["vehicle"]["rho"],
        eta_d=cfg["vehicle"]["drivetrain_eff"],
        m_published=cfg["vehicle"]["mass_car"], speed_sigma_ms=sigma,
        n_sigma=n_sigma, fuel_start=cfg["vehicle"]["fuel_start"],
        fuel_burn_per_lap=cfg["vehicle"]["fuel_burn_per_lap"])


def test_truth_is_contained_on_the_public_feed(cfg, races):
    """The operating case: 3.7 Hz, quantised, noisy.

    No upper bound may exclude the truth at all. A few lower bounds may, because
    the brake channel is resampled onto the feed's grid and a braking onset can
    land on the wrong side of an interval -- that is a labelling error, not a
    physics error, and it is what the HMM in step 3 is for.
    """
    gt = races[42]
    obs = observe(gt, LEADER, rate_hz=3.7,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
    brake = (gt.cars[LEADER].regime == "brake").astype(float)
    idx = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)
    c = _constraints(cfg, gt, obs.v, obs.t, obs.s, obs.lap, brake[idx],
                     observed_sigma(cfg))
    r = c.A @ true_theta(cfg) - c.b
    upper_viol = int((r[:c.n_upper] > 0).sum())
    lower_frac = float((r[c.n_upper:] > 0).mean())
    assert upper_viol == 0, f"{upper_viol} upper bounds exclude the truth"
    assert lower_frac <= 0.03, f"{100 * lower_frac:.1f}% of lower bounds exclude the truth"


def test_truth_is_contained_on_clean_truth_to_within_euler_error(cfg, races):
    """Same check on the noiseless 200 Hz trace, with NO statistical slack.

    Any violation here is a modelling error, so the assertion is on magnitude
    rather than count: the balance may disagree with the simulator only by the
    simulator's own integration error.
    """
    gt = races[42]
    tr = gt.cars[LEADER]
    c = _constraints(cfg, gt, tr.v, gt.t, tr.s, tr.lap,
                     (tr.regime == "brake").astype(float), 1e-9, n_sigma=0.0)
    r = c.A @ true_theta(cfg) - c.b
    worst = float(r.max())
    assert worst <= SIM_EULER_SLACK_J, (
        f"balance disagrees with the simulator by {worst:.1f} J, beyond the "
        f"{SIM_EULER_SLACK_J} J Euler allowance")
    # and the disagreement must not be one-sided, which would bias CdA
    assert float(np.median(r)) < 0.0, "constraints are not slack on average"



def test_braking_drops_only_the_lower_bound(cfg, races):
    """Brake force is not in the model, so a braking interval can no longer say
    the wheels delivered *at least* so much. It can still say at most."""
    gt = races[42]
    tr = gt.cars[LEADER]
    n = len(tr.v)
    free = _constraints(cfg, gt, tr.v, gt.t, tr.s, tr.lap, np.zeros(n), 1e-9)
    braked = _constraints(cfg, gt, tr.v, gt.t, tr.s, tr.lap, np.ones(n), 1e-9)
    assert braked.n_lower == 0
    assert braked.n_upper == free.n_upper
    assert free.n_lower == free.n_intervals


def test_no_derivative_is_taken(cfg, races):
    """Halving the sample rate must not double the noise in the constraints.

    A derivative-based formulation gets *worse* as the feed slows: dividing a
    fixed quantisation error by a smaller dt inflates it. The interval balance
    does the opposite, because the energy crossing the interval grows with dt
    while the endpoint error does not. This asserts the direction of that
    trend, which is the whole reason for the rewrite.
    """
    gt = races[42]
    brake = (gt.cars[LEADER].regime == "brake").astype(float)
    noise = []
    for rate in (20.0, 3.7):
        obs = observe(gt, LEADER, rate_hz=rate,
                      speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
        idx = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)
        c = _constraints(cfg, gt, obs.v, obs.t, obs.s, obs.lap, brake[idx],
                         observed_sigma(cfg), n_sigma=0.0)
        v0 = obs.v[:-1][:c.n_intervals]
        m = cfg["vehicle"]["mass_car"] + fuel_mass(0.0)
        # noise-to-signal on the interval: fixed endpoint error over the energy
        # the interval carries
        noise.append(float(np.median(
            energy_slack(v0, v0, m, observed_sigma(cfg)) / np.abs(c.b[:c.n_intervals]))))
    assert noise[1] < noise[0], (
        f"noise-to-signal rose as the feed slowed ({noise[0]:.3f} -> {noise[1]:.3f}); "
        "that is derivative behaviour and the point of the interval form is to avoid it")


def test_post_miami_allows_recovery_off_the_brakes(cfg, races):
    """Super-clipping. Pre-Miami the lower bound on P_K is -250 kW; post-Miami
    it is -350 kW at any throttle, so the post-Miami constraint set is strictly
    the looser of the two and cannot exclude a car the earlier rules allowed."""
    gt = races[42]
    tr = gt.cars[LEADER]
    brake = (tr.regime == "brake").astype(float)
    pre = _constraints(cfg, gt, tr.v, gt.t, tr.s, tr.lap, brake, 1e-9, regs=PRE_MIAMI)
    post = _constraints(cfg, gt, tr.v, gt.t, tr.s, tr.lap, brake, 1e-9, regs=POST_MIAMI)
    # same geometry, so the rows line up and only the offsets differ
    assert pre.A.shape == post.A.shape
    lower = slice(pre.n_upper, None)
    assert (post.b[lower] >= pre.b[lower] - 1e-9).all(), (
        "post-Miami lower bounds are tighter than pre-Miami; the harvest cap went the wrong way")


def test_tow_is_a_bound_not_a_model(cfg):
    """A car in clear air gets no tow allowance; a car one second back gets the
    widest. The estimator never claims to know the tow strength."""
    assert tow_f_max(np.nan) == 0.0
    assert tow_f_max(0.0) > tow_f_max(1.0) > tow_f_max(5.0)
    assert tow_f_max(5.0) < 0.01


def test_fuel_burn_moves_mass_by_the_declared_load(cfg):
    """70 kg on 790 is 9% of mass, and mass multiplies the two largest terms."""
    assert fuel_mass(0.0, 70.0, 1.15) == 70.0
    assert fuel_mass(30.0, 70.0, 1.15) == pytest.approx(70.0 - 34.5)
    # never negative: a stint longer than the fuel load must not imply
    # anti-mass, which would flip the sign of the largest term in the balance
    assert fuel_mass(200.0, 70.0, 1.15) == 0.0
