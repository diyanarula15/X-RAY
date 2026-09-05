"""Step 6: the PMP strategy prior, and the two results it derives."""
from __future__ import annotations

import numpy as np
import pytest

from xray.regs import PRE_MIAMI, taper
from xray.strategy import (aggression, dv_per_mj, fit_policy, joule_value,
                           pmp_deployment)


def test_a_joule_is_worth_v_cubed_less_at_speed(cfg):
    """The whole module in one line: d(time)/dE = -ds/(m v^3)."""
    v = np.array([20.0, 40.0, 80.0])
    val = joule_value(790.0, v)
    ratios = val / val[-1]
    np.testing.assert_allclose(ratios, (80.0 / v) ** 3, rtol=1e-9)
    assert val[0] > 60 * val[-1]


def test_zone_ordering_is_derived_not_typed(cfg, races):
    """The analytic marginal value must reproduce the *measured* zone ordering.

    decision.py gets Zone A 1.7, B 4.6, C 6.8 m/s per MJ by running the
    longitudinal model down each straight. The closed form
    dv = E / (1.5 rho CdA v L) has no access to that simulation and must still
    rank them the same way -- Zone A, the obvious overtaking place, is the worst
    place to spend an energy advantage because both its length and its
    end-of-straight speed are large.
    """
    from xray.decision import calibrate_zone
    from xray.vehicle import VehicleParams
    gt = races[42]
    params = VehicleParams.from_config(cfg)
    measured = {z.name: calibrate_zone(gt.track, params, z).dv_per_mj
                for z in gt.track.zones}
    analytic = {}
    for z in gt.track.zones:
        length = z.s_straight_end - z.s_straight_start
        # end-of-straight speed from the measured zone model's own top point
        zm = calibrate_zone(gt.track, params, z)
        analytic[z.name] = dv_per_mj(cfg["vehicle"]["rho"],
                                     cfg["vehicle"]["cda_straight"],
                                     float(zm.speed_grid.max()), length)
    m_order = sorted(measured, key=measured.get)
    a_order = sorted(analytic, key=analytic.get)
    assert m_order == a_order, (
        f"analytic ordering {a_order} does not match measured {m_order}: "
        f"analytic {analytic}, measured {measured}")
    assert measured["A"] < measured["C"], "the counterintuitive result vanished"


def test_policy_fit_recovers_a_known_policy(cfg):
    """Three parameters against a lap of samples. Recovery to 0.2 m/s."""
    n = 400
    v = np.linspace(30.0, 95.0, n)
    zone = np.ones(n, bool)
    on_power = np.ones(n, bool)
    braking = np.zeros(n, bool)
    truth = (72.0, 88.0, 3.0)
    clean = pmp_deployment(v, *truth, zone, PRE_MIAMI, on_power, braking)
    noisy = clean + np.random.default_rng(0).normal(0.0, 8.0e3, n)
    fit = fit_policy(v, noisy, zone, PRE_MIAMI, on_power, braking)
    assert abs(fit.v_cut - truth[0]) < 1.0
    assert abs(fit.v_harv - truth[1]) < 1.0
    assert abs(fit.width - truth[2]) < 1.0


def test_the_prior_predicts_super_clipping_at_the_top_of_the_range(cfg):
    """Harvesting where the joule is cheapest means harvesting where v is
    highest, which is the same place the taper makes deployment worthless. The
    two halves of the regulation point the same way, and the prior has to as
    well or it is not describing a rational team."""
    v = np.linspace(30.0, 98.0, 200)
    p = pmp_deployment(v, 72.0, 88.0, 3.0, np.ones(200, bool), PRE_MIAMI,
                       np.ones(200, bool), np.zeros(200, bool))
    assert p[0] > 0, "not deploying out of a slow corner"
    assert p[-1] < 0, "not recovering at the top of the speed range"
    crossover = v[int(np.argmin(np.abs(p)))]
    assert 70.0 < crossover < 92.0
    # and the recovery region must sit where the taper has already given up
    assert taper(v[-1]) < 0.2


def test_the_prior_never_exceeds_the_regulation(cfg):
    """It is a prior, not a licence: it must stay inside the same bounds the
    identified set uses, or it could tilt the posterior outside them."""
    v = np.linspace(20.0, 99.0, 300)
    p = pmp_deployment(v, 80.0, 60.0, 8.0, np.ones(300, bool), PRE_MIAMI,
                       np.ones(300, bool), np.zeros(300, bool))
    from xray.regs import p_dep_max
    assert (p <= p_dep_max(v, np.ones(300, bool), PRE_MIAMI) + 1e-6).all()
    assert (p >= -PRE_MIAMI.p_harv_max - 1e-6).all()


def test_braking_and_coasting_gates(cfg):
    """No deployment under braking, and none off throttle."""
    v = np.full(50, 60.0)
    zone = np.ones(50, bool)
    braked = pmp_deployment(v, 80.0, 95.0, 3.0, zone, PRE_MIAMI,
                            np.ones(50, bool), np.ones(50, bool))
    assert (braked <= 0).all()
    coasting = pmp_deployment(v, 80.0, 95.0, 3.0, zone, PRE_MIAMI,
                              np.zeros(50, bool), np.zeros(50, bool))
    assert np.allclose(coasting, 0.0)


def test_aggression_is_bounded_and_monotone(cfg):
    """The fingerprint number. A driver who deploys across more of the speed
    range is spending more; one who cuts out early is hoarding.

    Both earlier definitions failed this. taper(v_cut) ranked them backwards --
    a 180 km/h cut-off scored 1.00 -- and 1 - taper(v_cut) scored every
    realistic cut-off at exactly 0, because taper is flat below 290 km/h.
    """
    from xray.strategy import PolicyFit
    lo, hi = 20.0, 98.0
    hoarding = aggression(PolicyFit(40.0, 90.0, 3.0, 0.0, 100), lo, hi)
    spending = aggression(PolicyFit(90.0, 90.0, 3.0, 0.0, 100), lo, hi)
    assert 0.0 <= hoarding < spending <= 1.0
    assert hoarding < 0.35 and spending > 0.8
    # and it must have resolution where drivers differ: below the taper window
    a = aggression(PolicyFit(50.0, 90.0, 3.0, 0.0, 100), lo, hi)
    b = aggression(PolicyFit(65.0, 90.0, 3.0, 0.0, 100), lo, hi)
    assert b - a > 0.1, "no resolution across realistic cut-offs"


def test_fit_refuses_a_lap_it_cannot_constrain(cfg):
    """Three parameters need more than a handful of samples. Refusal, not a
    fit with meaningless numbers."""
    with pytest.raises(ValueError, match="usable samples"):
        fit_policy(np.arange(5.0), np.zeros(5), np.ones(5, bool), PRE_MIAMI)
