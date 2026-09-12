"""P3.5 Part 2: boundary behaviour, ceiling semantics, reserve observability.

New surfaces only. Part 1's 28 tests are authoritative and are not rerun.

Every test here is about WHERE uncertainty comes from. The estimator is allowed to
be confident when observations identify a value, and must not be confident merely
because a projection mapped a region onto a point.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.constants import E_STORE_MAX
from xray.realfit import (Kin, RealNuisanceFit, belief_from_deployment,
                          deployment_trace)


def _kin(n=600, v=60.0, ceiling=3.0e5, throttle=90.0, brake=0.0, laps=3):
    """A flat synthetic Kin: no circuit, no physics, just the filter's inputs."""
    lap = np.repeat(np.arange(laps), n // laps)[:n]
    return Kin(s=np.linspace(0, 5000, n), v=np.full(n, v), a=np.zeros(n),
               dt=np.full(n, 0.1), t=np.linspace(0, 60, n),
               mass=np.full(n, 800.0), cda_scale=np.ones(n),
               sin_grade=np.zeros(n), in_deployment_zone=np.ones(n, bool),
               ceiling=np.full(n, ceiling),
               throttle=np.full(n, throttle), brake=np.full(n, brake),
               valid=np.ones(n, bool), lap=lap)


def _fit(cda=0.8, sigma=0.08, resid=2.0e4):
    return RealNuisanceFit(
        cda_hat=cda, cda_lo=cda - 0.2, cda_hi=cda + 0.2, cda_sigma=sigma,
        v_wind_hat=0.0, rho=1.2, identifiability=0.5, n_samples=600,
        n_binding=300, n_coast=50, coast_cda=cda, residual_rms=resid,
        systematic_rms=resid)


def _trace(kin, deploy_w, harvest_w=0.0, lo=None, hi=None):
    n = len(kin.v)
    d = np.full(n, float(deploy_w))
    return {"deploy": d,
            "deploy_lo": np.full(n, float(lo if lo is not None else deploy_w)),
            "deploy_hi": np.full(n, float(hi if hi is not None else deploy_w)),
            "harvest": np.full(n, float(harvest_w)),
            "braking": np.zeros(n, bool),
            "P_obs": d, "P_ice": np.zeros(n)}


def _width(bel, frac=None):
    u = np.asarray(bel["usable_mean"], float)
    lo = np.asarray(bel["usable_p10"], float)
    hi = np.asarray(bel["usable_p90"], float)
    m = np.isfinite(u) & np.isfinite(lo) & np.isfinite(hi)
    w = (hi - lo)[m]
    return (float(np.mean(w)), float(np.mean(w < 1.0e4))) if frac is None \
        else float(np.mean(w))


# ====================================================== §2 Case A / B / C
def test_case_a_genuine_zero_information_may_legitimately_be_narrow():
    """When the evidence really does identify an empty store, narrow is correct.

    Sustained deployment far above any harvest drains the store under either
    boundary rule, and a narrow band there is earned rather than manufactured.
    """
    kin = _kin()
    tr = _trace(kin, deploy_w=3.0e5, harvest_w=0.0)
    bel = belief_from_deployment(kin, tr, _fit(), n_particles=200, seed=1,
                                 boundary="truncated")
    u = np.asarray(bel["usable_mean"], float)
    assert np.nanmean(u[-50:]) < 5.0e4, "a fully drained store should read empty"


def test_case_b_boundary_collision_must_not_manufacture_certainty():
    """Particles reaching 0 only because of the physical bound stay distinguishable.

    This is the measured production pathology: hard clipping mapped 71.5% of
    particles onto exactly 0 J while the cloud still held 1.63 MJ of real spread
    in E, and the 10/90 weighted quantiles then coincided.
    """
    kin = _kin()
    # Deployment and harvest nearly balance, so particles drift across the floor
    # without the evidence ever identifying an empty store.
    tr = _trace(kin, deploy_w=1.2e5, harvest_w=1.1e5, lo=0.2e5, hi=2.4e5)
    clip = belief_from_deployment(kin, tr, _fit(), n_particles=300, seed=2,
                                 boundary="clip")
    trunc = belief_from_deployment(kin, tr, _fit(), n_particles=300, seed=2,
                                   boundary="truncated")
    w_clip, degen_clip = _width(clip)
    w_tr, degen_tr = _width(trunc)
    assert degen_tr <= degen_clip, (degen_clip, degen_tr)
    assert w_tr >= w_clip * 0.9, (w_clip, w_tr)


def test_case_c_repeated_boundary_contact_does_not_ratchet_up_certainty():
    """More laps of boundary contact must not keep shrinking the band.

    A filter that narrows with every boundary collision is accumulating
    information it never received.
    """
    short = _kin(n=600, laps=3)
    long = _kin(n=1800, laps=9)
    args = dict(n_particles=300, seed=3, boundary="truncated")
    w_short = _width(belief_from_deployment(
        short, _trace(short, 1.2e5, 1.1e5, lo=0.2e5, hi=2.4e5), _fit(), **args), 1)
    w_long = _width(belief_from_deployment(
        long, _trace(long, 1.2e5, 1.1e5, lo=0.2e5, hi=2.4e5), _fit(), **args), 1)
    assert w_long > 0.2 * w_short, (w_short, w_long)


def test_the_truncated_path_never_produces_a_negative_or_over_full_store():
    kin = _kin()
    tr = _trace(kin, deploy_w=2.0e5, harvest_w=5.0e4, lo=0.0, hi=3.0e5)
    bel = belief_from_deployment(kin, tr, _fit(), n_particles=200, seed=4,
                                 boundary="truncated")
    for k in ("soc_mean", "soc_p10", "soc_p90"):
        a = np.asarray(bel[k], float)
        a = a[np.isfinite(a)]
        assert a.min() >= -1e-6 and a.max() <= E_STORE_MAX + 1e-6, k


def test_the_default_boundary_is_still_the_clip_that_produced_the_p3_artifacts():
    """P3's numbers must stay reproducible until the activation gate says otherwise."""
    kin = _kin()
    tr = _trace(kin, 1.2e5, 1.1e5, lo=0.2e5, hi=2.4e5)
    a = belief_from_deployment(kin, tr, _fit(), n_particles=120, seed=5)
    b = belief_from_deployment(kin, tr, _fit(), n_particles=120, seed=5,
                              boundary="clip")
    assert np.allclose(np.nan_to_num(a["usable_mean"]),
                       np.nan_to_num(b["usable_mean"]))


# ============================================ §3/§4 ceiling as a bound
def test_the_ceiling_is_flagged_where_it_is_the_only_upper_information():
    # The ceiling must sit BELOW the observed wheel power for its edge to be the
    # regulatory one. At v=60 with CdA=0.8 the wheel takes about 71 kW, so a
    # 150 kW ceiling is not binding and the upper edge is legitimately
    # observational -- the first version of this fixture tested nothing.
    kin = _kin(v=60.0, ceiling=5.0e4, throttle=90.0)
    fit = _fit(cda=1.2)
    tr = deployment_trace(kin, fit, regs=None)
    assert "upper_is_regulation" in tr
    # P_obs well above the ceiling -> the upper edge is regulatory
    assert tr["upper_is_regulation"].any()


def test_the_evidence_centre_does_not_average_against_a_regulatory_bound():
    """Measured on a real window: the midpoint sat 127.0 kW above the evidence."""
    kin = _kin(v=60.0, ceiling=5.0e4, throttle=90.0)
    fit = _fit(cda=1.2)
    mid = deployment_trace(kin, fit, regs=None, centre="midpoint")
    ev = deployment_trace(kin, fit, regs=None, centre="evidence")
    m = mid["upper_is_regulation"]
    assert m.any()
    assert np.nanmean(ev["deploy"][m]) < np.nanmean(mid["deploy"][m])
    # the BAND is untouched: only which quantity may locate the latent changed
    assert np.allclose(np.nan_to_num(mid["deploy_lo"]), np.nan_to_num(ev["deploy_lo"]))
    assert np.allclose(np.nan_to_num(mid["deploy_hi"]), np.nan_to_num(ev["deploy_hi"]))


def test_an_unknown_centre_mode_is_refused_rather_than_silently_defaulted():
    kin = _kin()
    with pytest.raises(ValueError, match="centre"):
        deployment_trace(kin, _fit(), regs=None, centre="whatever")


def test_the_cut_out_detector_keeps_its_own_signal():
    """Correcting the reported centre must not silently recalibrate the detector.

    Its 150 kW / 25 kW thresholds were chosen against the midpoint; scoring them
    against the evidence centre raised degeneracy from 22.6% to 43.8%.
    """
    kin = _kin(v=60.0, ceiling=5.0e4, throttle=90.0)
    fit = _fit(cda=1.2)
    ev = deployment_trace(kin, fit, regs=None, centre="evidence")
    mid = deployment_trace(kin, fit, regs=None, centre="midpoint")
    assert "detect_signal" in ev
    assert np.allclose(np.nan_to_num(ev["detect_signal"]),
                       np.nan_to_num(mid["detect_signal"]))
    assert not np.allclose(np.nan_to_num(ev["deploy"]),
                           np.nan_to_num(mid["deploy"]))


def test_the_default_centre_is_still_the_midpoint():
    kin = _kin(v=60.0, ceiling=5.0e4)
    fit = _fit(cda=1.2)
    a = deployment_trace(kin, fit, regs=None)
    b = deployment_trace(kin, fit, regs=None, centre="midpoint")
    assert np.allclose(np.nan_to_num(a["deploy"]), np.nan_to_num(b["deploy"]))
    assert a["centre_semantics"] == "midpoint"


# ================================================ §5 reserve observability
def test_the_reserve_sigma_is_reachable_from_the_caller():
    """Part 1 called this observation inert. The override had nowhere to land.

    `RESERVE_SIGMA_REAL` was read directly inside the loop, so a harness passing
    a different value changed nothing and the measured sensitivity was 0.000 MJ
    for that reason alone -- a dead path in the harness, not a property of the
    estimator.
    """
    import inspect
    sig = inspect.signature(belief_from_deployment)
    assert "reserve_sigma_j" in sig.parameters


def test_a_different_reserve_sigma_changes_the_belief():
    kin = _kin(n=900, v=60.0, laps=3, throttle=90.0)
    n = len(kin.v)
    # a cut-out: deployment above the arm threshold, then below the fire
    # threshold, while below the taper and on power
    d = np.full(n, 2.0e5)
    d[300:360] = 1.0e4
    d[600:660] = 1.0e4
    tr = {"deploy": d, "deploy_lo": d * 0.5, "deploy_hi": d * 1.5,
          "harvest": np.full(n, 1.0e5), "braking": np.zeros(n, bool),
          "P_obs": d, "P_ice": np.zeros(n), "detect_signal": d}
    a = belief_from_deployment(kin, tr, _fit(), n_particles=200, seed=6,
                              reserve_sigma_j=2.5e5)
    b = belief_from_deployment(kin, tr, _fit(), n_particles=200, seed=6,
                              reserve_sigma_j=1.0e6)
    ua = np.asarray(a["usable_mean"], float); ub = np.asarray(b["usable_mean"], float)
    m = np.isfinite(ua) & np.isfinite(ub)
    assert m.any()
    assert not np.allclose(ua[m], ub[m]), "reserve sigma still has no effect"


def test_the_one_sided_reserve_likelihood_only_penalises_particles_above_the_floor():
    """A cut-out says `E <= reserve`; it does not locate E at the reserve."""
    kin = _kin(n=900, v=60.0, laps=3, throttle=90.0)
    n = len(kin.v)
    d = np.full(n, 2.0e5); d[300:360] = 1.0e4
    tr = {"deploy": d, "deploy_lo": d * 0.5, "deploy_hi": d * 1.5,
          "harvest": np.full(n, 1.0e5), "braking": np.zeros(n, bool),
          "P_obs": d, "P_ice": np.zeros(n), "detect_signal": d}
    pt = belief_from_deployment(kin, tr, _fit(), n_particles=200, seed=7,
                               reserve_obs="point")
    os = belief_from_deployment(kin, tr, _fit(), n_particles=200, seed=7,
                               reserve_obs="one_sided")
    a = np.asarray(pt["usable_mean"], float); b = np.asarray(os["usable_mean"], float)
    m = np.isfinite(a) & np.isfinite(b)
    assert not np.allclose(a[m], b[m])


def test_the_default_reserve_observation_is_still_the_point_form():
    kin = _kin(n=900, laps=3)
    tr = _trace(kin, 2.0e5, 1.0e5, lo=1.0e5, hi=3.0e5)
    a = belief_from_deployment(kin, tr, _fit(), n_particles=120, seed=8)
    b = belief_from_deployment(kin, tr, _fit(), n_particles=120, seed=8,
                              reserve_obs="point")
    assert np.allclose(np.nan_to_num(a["usable_mean"]),
                       np.nan_to_num(b["usable_mean"]))


# ================================================ §6 temporal consistency
def test_energy_remains_a_bounded_accumulation_of_deploy_and_harvest():
    """E[t+1] = E[t] - deployed + harvested, inside [0, E_STORE_MAX].

    Checked on the reported mean rather than on internals: with zero harvest and
    a steady draw the store must fall monotonically until it reaches the floor,
    and with zero draw and steady harvest it must rise and then stop at the cap.
    """
    kin = _kin(n=900, laps=3)
    drain = belief_from_deployment(
        kin, _trace(kin, 2.5e5, 0.0, lo=2.5e5, hi=2.5e5), _fit(),
        n_particles=200, seed=9, boundary="truncated")
    fill = belief_from_deployment(
        kin, _trace(kin, 0.0, 2.5e5, lo=0.0, hi=0.0), _fit(),
        n_particles=200, seed=9, boundary="truncated")
    sd = np.asarray(drain["soc_mean"], float); sd = sd[np.isfinite(sd)]
    sf = np.asarray(fill["soc_mean"], float); sf = sf[np.isfinite(sf)]
    assert sd[-1] < sd[0], "a pure drain must lower the store"
    assert sf[-1] > sf[0], "a pure harvest must raise the store"
    assert sd.min() >= -1e-6 and sf.max() <= E_STORE_MAX + 1e-6


def test_harvest_scaling_moves_the_store_in_the_expected_direction():
    kin = _kin(n=900, laps=3)
    lo = belief_from_deployment(kin, _trace(kin, 2.0e5, 0.5e5, lo=2.0e5, hi=2.0e5),
                               _fit(), n_particles=200, seed=10,
                               boundary="truncated")
    hi = belief_from_deployment(kin, _trace(kin, 2.0e5, 1.9e5, lo=2.0e5, hi=2.0e5),
                               _fit(), n_particles=200, seed=10,
                               boundary="truncated")
    a = np.asarray(lo["soc_mean"], float); b = np.asarray(hi["soc_mean"], float)
    assert np.nanmean(b[-100:]) > np.nanmean(a[-100:])
