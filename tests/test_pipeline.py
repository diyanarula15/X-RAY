"""Step 4: mass drift, the tow bound, and the wiring that composes steps 1-3."""
from __future__ import annotations

import numpy as np

from xray.balance import fuel_mass
from xray.constants import G
from xray.observe import observe
from xray.pipeline import Feed, identify_car
from xray.regs import N_RPM_MAX
from xray.sim import LEADER

from tests.simfix import elevation_fn, observed_sigma, true_theta


def _feed(cfg, gt, gap_s=None, rate=3.7):
    track = gt.track
    z = elevation_fn(track)
    obs = observe(gt, LEADER, rate_hz=rate,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
    idx = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)
    reg = gt.cars[LEADER].regime[idx]
    n = len(obs.v)
    return obs, Feed(
        t=obs.t, v=obs.v, s=obs.s, z=z(obs.s),
        lap_frac=obs.lap + obs.s / track.length,
        throttle=np.where(reg == "brake", 0.0, 1.0),
        brake=(reg == "brake").astype(float), rpm=np.full(n, N_RPM_MAX),
        gap_s=np.full(n, np.nan) if gap_s is None else np.full(n, gap_s),
        x_allowed=~np.asarray(track.is_corner(obs.s)), in_zone=np.ones(n, bool))


def _kw(cfg):
    v = cfg["vehicle"]
    return dict(rho=v["rho"], eta_d=v["drivetrain_eff"],
                m_published=v["mass_car"], speed_sigma_ms=observed_sigma(cfg),
                fuel_start=v["fuel_start"], fuel_burn_per_lap=v["fuel_burn_per_lap"])


def test_pipeline_composes_and_contains_the_truth(cfg, races):
    _obs, feed = _feed(cfg, races[42])
    out = identify_car(feed, **_kw(cfg))
    th = true_theta(cfg)
    S = out.identified
    assert not S.empty
    assert ((th >= S.lo - 1e-9) & (th <= S.hi + 1e-9)).all(), out.describe()
    assert out.regs.variant == "pre-Miami-2026"


def test_causal_pipeline_runs_and_still_contains_the_truth(cfg, races):
    """Live operation uses the forward-only mode filter. Containment is not
    allowed to depend on the backward pass."""
    _obs, feed = _feed(cfg, races[42])
    out = identify_car(feed, causal=True, **_kw(cfg))
    th = true_theta(cfg)
    S = out.identified
    assert not S.empty
    assert ((th >= S.lo - 1e-9) & (th <= S.hi + 1e-9)).all()


def test_a_car_in_traffic_is_told_less(cfg, races):
    """The tow is a bound, so running in a wake must *widen* the identified set.

    Measured: 0.6 s behind another car for a whole race loosens the CdA_X upper
    bound by 10.4%. The estimator does not know the tow strength and does not
    pretend to -- it reports a worse answer, which is the honest outcome and the
    opposite of Stage 1, where contaminated samples produced a drag area that
    was quietly 5% low.
    """
    _o1, clear = _feed(cfg, races[42])
    _o2, towed = _feed(cfg, races[42], gap_s=0.6)
    a = identify_car(clear, **_kw(cfg)).identified
    b = identify_car(towed, **_kw(cfg)).identified
    assert b.hi[0] > a.hi[0], "a tow tightened the drag bound"
    assert (b.hi[0] - a.hi[0]) / a.hi[0] > 0.02
    th = true_theta(cfg)
    assert ((th >= b.lo - 1e-9) & (th <= b.hi + 1e-9)).all()


def test_mass_drift_does_not_move_the_drag_bound_and_that_is_expected(cfg, races):
    """Fuel burn is 17 kg over the stint and changes the CdA bound by nothing.

    Not a bug, and worth pinning so nobody "fixes" it: the window that binds the
    drag ceiling sits at near-steady speed, where the net kinetic change is
    1.1 kJ. Mass multiplies *that*, so 17 kg is worth 18.5 kJ against a
    3.93 MJ per m^2 drag column -- 0.64% of the bound, below its resolution.

    Where mass drift does bite is the energy reconstruction, which is the next
    test.
    """
    _obs, feed = _feed(cfg, races[42])
    kw = _kw(cfg)
    with_drift = identify_car(feed, **kw).identified
    without = identify_car(feed, **{**kw, "fuel_burn_per_lap": 0.0}).identified
    rel = abs(without.hi[0] - with_drift.hi[0]) / with_drift.hi[0]
    assert rel < 0.02, (
        f"mass drift now moves the drag bound by {100 * rel:.2f}%; the binding "
        "window must have shifted off steady speed, so re-derive the estimate")


def test_mass_drift_biases_lap_energy_and_grows_across_the_stint(cfg, races):
    """The reason mass drift is modelled at all.

    Holding mass at its start value overstates the positive wheel energy of
    every lap, and the error grows as the fuel goes: +0.09% on lap 0 to +2.05%
    on the last. A drift that grows monotonically across a stint is exactly the
    shape that corrupts a store reconstruction, because the store integrates it.
    """
    obs, feed = _feed(cfg, races[42])
    v = cfg["vehicle"]

    def per_lap(burn):
        m = v["mass_car"] + fuel_mass(feed.lap_frac, v["fuel_start"], burn)
        e = (0.5 * np.asarray(m)[:-1] * np.diff(obs.v ** 2)
             + np.asarray(m)[:-1] * G * np.diff(feed.z))
        return np.array([np.sum(np.clip(e[obs.lap[:-1] == L], 0, None))
                         for L in np.unique(obs.lap[:-1])])

    true_m = per_lap(v["fuel_burn_per_lap"])
    flat_m = per_lap(0.0)
    bias = 100.0 * (flat_m - true_m) / true_m
    assert bias[0] < bias[-1], "the bias does not grow with fuel burned"
    assert bias[-1] > 1.0, f"final-lap bias only {bias[-1]:.2f}%"
    assert np.all(np.diff(bias[:-1]) > -0.05), "bias is not monotone across the stint"
