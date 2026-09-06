"""Step 5: the store, the reserve, and an honest account of which one works.

The store belief works: 0.72 coverage at 0.46 MJ width, 5.1% per-lap energy
error, and it degrades in the way a particle filter should when starved of
particles rather than silently.

The reserve does not, and `test_reserve_is_not_identified_yet` records that as a
measured failure rather than a tuned pass. Chasing it with wider jitter, a
hysteretic cut-out detector, an accelerating gate and the buffer-release ramp
each moved the number and none fixed it, which is the signature of a structural
problem rather than a parameter one.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.constants import E_STORE_MAX
from xray.metrics import true_reserve_floor
from xray.observe import observe
from xray.pipeline import Feed, identify_car
from xray.regs import N_RPM_MAX, PRE_MIAMI
from xray.rbpf import run as rbpf_run
from xray.sim import LEADER

from tests.simfix import elevation_fn, observed_sigma


def _setup(cfg, gt, car=LEADER, rate=3.7):
    track = gt.track
    z = elevation_fn(track)
    reg = gt.cars[car].regime
    obs = observe(gt, car, rate_hz=rate,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
    idx = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)
    n = len(obs.v)
    feed = Feed(t=obs.t, v=obs.v, s=obs.s, z=z(obs.s),
                lap_frac=obs.lap + obs.s / track.length,
                throttle=np.where(reg[idx] == "brake", 0.0, 1.0),
                brake=(reg[idx] == "brake").astype(float),
                rpm=np.full(n, N_RPM_MAX), gap_s=np.full(n, np.nan),
                x_allowed=~np.asarray(track.is_corner(obs.s)),
                in_zone=np.ones(n, bool))
    v = cfg["vehicle"]
    kw = dict(rho=v["rho"], eta_d=v["drivetrain_eff"], m_published=v["mass_car"],
              speed_sigma_ms=observed_sigma(cfg), fuel_start=v["fuel_start"],
              fuel_burn_per_lap=v["fuel_burn_per_lap"])
    ident = identify_car(feed, **kw)
    return obs, idx, feed, ident


def _belief(cfg, gt, ident, feed, **over):
    v = cfg["vehicle"]
    args = dict(n_particles=400, seed=7, eta_d=v["drivetrain_eff"],
                m_published=v["mass_car"], fuel_start=v["fuel_start"],
                fuel_burn_per_lap=v["fuel_burn_per_lap"],
                is_x=ident.modes.is_x, regime=ident.modes.regime)
    args.update(over)
    return rbpf_run(feed, ident.identified, PRE_MIAMI, v["rho"], **args)


def test_store_band_covers_the_truth(cfg, races):
    """The raw store, scored against the simulator's hidden state.

    Measured 0.72 at 0.46 MJ width. Not the headline quantity -- the store is
    identified only up to the driver's buffer -- but it is the quantity this
    stage actually reconstructs, so it is the one to check.
    """
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    b = _belief(cfg, gt, ident, feed)
    e_true = gt.cars[LEADER].E[idx]
    cov = float(np.mean((e_true >= b.soc_p10) & (e_true <= b.soc_p90)))
    assert cov > 0.60, f"store band coverage {cov:.2f}"
    width = float(np.mean(b.soc_p90 - b.soc_p10))
    assert 0.1e6 < width < 1.5e6, f"store band width {width / 1e6:.2f} MJ"


def test_per_lap_deployed_energy(cfg, races):
    """Energy flows are what a speed trace measures well. Measured 5.1%."""
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    b = _belief(cfg, gt, ident, feed)
    truth = gt.cars[LEADER].deployed_lap
    laps = sorted(k for k in b.deployed_lap if k < len(truth))
    est = np.array([b.deployed_lap[k] for k in laps])
    tru = np.array([truth[k] for k in laps])
    ok = tru > 1e4
    mape = 100.0 * np.mean(np.abs(est[ok] - tru[ok]) / tru[ok])
    assert mape <= 10.0, f"per-lap deployed MAPE {mape:.1f}%"


def test_the_store_never_leaves_its_bounds(cfg, races):
    """0 to 4 MJ is a regulation, not a preference. A particle outside it is
    describing a car that cannot exist."""
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    b = _belief(cfg, gt, ident, feed)
    assert (b.soc_p10 >= -1e-6).all()
    assert (b.soc_p90 <= E_STORE_MAX + 1e-6).all()
    assert (b.usable_p10 >= -1e-6).all()


def test_every_particle_stays_inside_the_identified_set(cfg, races):
    """The filter may not rescue itself with a theta the regulation forbids.
    This is the join between steps 2 and 5, and it is the reason the filter can
    be trusted to report a store at all."""
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    b = _belief(cfg, gt, ident, feed)
    S = ident.identified
    assert (b.theta_post_lo >= S.lo - 1e-9).all()
    assert (b.theta_post_hi <= S.hi + 1e-9).all()


def test_four_hundred_particles_is_the_right_order(cfg, races):
    """The plan claims 300-500 is plenty. Measured, and the failure mode at 100
    is the informative part: ESS 1 of 100, per-lap error 39.6%. It does not
    degrade gracefully, it collapses -- so the particle count is a correctness
    parameter, not a speed/accuracy dial. 1,600 buys nothing (ESS 128, same
    5.1% error), so 400 is the knee.
    """
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    truth = gt.cars[LEADER].deployed_lap

    def mape(b):
        laps = sorted(k for k in b.deployed_lap if k < len(truth))
        est = np.array([b.deployed_lap[k] for k in laps])
        tru = np.array([truth[k] for k in laps])
        ok = tru > 1e4
        return 100.0 * np.mean(np.abs(est[ok] - tru[ok]) / tru[ok])

    poor = _belief(cfg, gt, ident, feed, n_particles=100)
    good = _belief(cfg, gt, ident, feed, n_particles=400)
    assert np.median(good.ess) > 20, f"ESS {np.median(good.ess):.0f} of 400"
    assert mape(good) < mape(poor) / 2.0, (
        f"100 particles gave {mape(poor):.1f}% and 400 gave {mape(good):.1f}%; "
        "the collapse at low particle counts has stopped showing up, which "
        "means the filter is no longer weight-driven")


def test_rao_blackwellisation_is_wired_into_the_likelihood(cfg, races):
    """The store's propagated variance must actually enter the weights.

    Carrying E_var and never using it would make the label decoration, which is
    exactly what the first version of this module did.
    """
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    on = _belief(cfg, gt, ident, feed, rao_blackwell=True)
    off = _belief(cfg, gt, ident, feed, rao_blackwell=False)
    assert not np.allclose(on.soc_mean, off.soc_mean), (
        "toggling Rao-Blackwellisation changed nothing, so it is not connected")


@pytest.mark.xfail(reason=(
    "The reserve is not separately identified by this filter, so the "
    "deployable-energy band is unusable. Measured on seed 42: the inferred "
    "buffer is 0.97 +/- 0.32 MJ for a LEADER whose policy holds 0.00, and "
    "because U = max(E - R, 0) is then exactly zero for every particle the "
    "band collapses to 0.06 MJ and covers the truth 4% of the time. The store "
    "itself is fine at 0.72 coverage, so this is the E/R split, not the "
    "filter. Diagnosis: the cut-out likelihood selects *consistent pairs* of "
    "(E, R) rather than pinning R, so R stays at its prior mean. A hysteretic "
    "detector, an accelerating gate, wider reserve jitter and the "
    "buffer-release ramp each moved the number without fixing it. The Stage 1 "
    "estimator reaches 0.94 on this same quantity, so the capability exists in "
    "the repo and this rewrite has lost it -- that comparison is the next "
    "thing to chase."), strict=True)
def test_reserve_is_not_identified_yet(cfg, races):
    gt = races[42]
    obs, idx, feed, ident = _setup(cfg, gt)
    b = _belief(cfg, gt, ident, feed)
    reserve_true = float(true_reserve_floor(gt, LEADER, obs.lap).max())
    e_true = gt.cars[LEADER].E[idx]
    u_true = np.clip(e_true - true_reserve_floor(gt, LEADER, obs.lap), 0.0, None)
    cov = float(np.mean((u_true >= b.usable_p10) & (u_true <= b.usable_p90)))
    assert abs(b.reserve_mean - reserve_true) < 0.4e6
    assert cov > 0.60, f"deployable-energy coverage {cov:.2f}"
