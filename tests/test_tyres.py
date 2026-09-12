"""P1.5: the tyre state, and the one confusion it exists to prevent.

FastF1's TyreLife is age in laps. Every shortcut in this area starts by treating
it as wear because both are "how used up is the tyre" and both go up. They are
not the same quantity and the first test here is the one that matters.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.config import load_config
from xray.tyres import (MAX_THERMAL_WEAR_MULTIPLIER, MIN_THERMAL_GRIP,
                        TyreState, advance, force_limits, fresh, grip_scale,
                        mu_effective, params_from_config,
                        thermal_grip_factor, thermal_wear_multiplier,
                        unknown, utilisation, wear_grip_factor, wet_grip_factor)
from xray.vehicle import G


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def soft(cfg):
    return params_from_config(cfg, "SOFT")


# ---------------------------------------------------------- the semantic rule
def test_tyre_life_is_age_and_is_never_wear(soft):
    """A set fitted under a safety car has age without wear."""
    st = fresh("SOFT", soft, track_temp_c=35.0, tyre_life=6.0)
    assert st.tyre_life == 6.0, "age must be carried through untouched"
    assert st.wear_fraction == 0.0, "a new tyre has no wear whatever its age"
    assert st.tyre_life != st.wear_fraction
    assert st.fresh_tyre is True


def test_wear_is_never_assigned_from_age_anywhere_in_the_module():
    """Structural, not behavioural: the shortcut must not exist in the source."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "xray" / "tyres.py").read_text()
    for bad in ("wear_fraction=tyre_life", "wear_fraction = tyre_life",
                "wear_fraction=state.tyre_life", "tyre_life /", "tyre_life/"):
        assert bad not in src, f"wear derived from age: {bad!r}"


def test_a_fresh_set_resets_wear_independently_of_age(soft):
    worn = TyreState("SOFT", tyre_life=28.0, fresh_tyre=False,
                     estimated_temp_c=105.0, wear_fraction=0.8,
                     grip_scale=0.8, source="modelled_thermal_wear")
    new = fresh("SOFT", soft, track_temp_c=35.0, tyre_life=0.0)
    assert worn.wear_fraction == 0.8 and new.wear_fraction == 0.0

    # Compared at the SAME temperature, the fresh set grips more. Compared at
    # their own temperatures it does not, and that is not a bug: this fresh set
    # is at 35 C against a 100 C optimum, and a cold new tyre really is worse
    # than a warm worn one. Isolating wear is the only way to assert wear.
    assert (grip_scale(105.0, 0.0, 0.0, soft)
            > grip_scale(105.0, 0.8, 0.0, soft))
    assert new.grip_scale < worn.grip_scale, (
        "a brand-new tyre 65 C below its window should NOT out-grip a warm one")


def test_unknown_state_is_neutral_and_says_so(soft):
    st = unknown("SOFT", 12.0)
    assert st.source == "unknown"
    assert st.grip_scale == 1.0, "absence of information must not penalise"
    assert st.wear_fraction == 0.0
    assert st.tyre_life == 12.0


def test_an_unrecognised_compound_is_preserved_and_neutral(cfg):
    p = params_from_config(cfg, "C7")
    assert p.compound == "C7", "unknown compound strings must survive"
    assert p.wet_grip_loss == 0.0 and p.dry_grip_loss == 0.0, (
        "an unparameterised compound must make no wet/dry claim")


# ---------------------------------------------------------------- grip shape
def test_temperature_has_an_optimum_not_a_direction(soft):
    """Cold and hot must BOTH be worse than the window."""
    best = thermal_grip_factor(soft.optimal_temp_c, soft)
    cold = thermal_grip_factor(soft.optimal_temp_c - 2 * soft.temp_window_c, soft)
    hot = thermal_grip_factor(soft.optimal_temp_c + 2 * soft.temp_window_c, soft)
    assert best == pytest.approx(1.0)
    assert cold < best and hot < best
    # and it is not a single linear rule in either direction
    warmer_from_cold = thermal_grip_factor(soft.optimal_temp_c - 5.0, soft)
    assert warmer_from_cold > cold, "warming a cold tyre must help"
    warmer_from_hot = thermal_grip_factor(soft.optimal_temp_c + 5.0, soft)
    assert warmer_from_hot > hot, "the same rule cannot hold on both sides"


def test_thermal_factor_is_bounded(soft):
    for t in (-50.0, 0.0, 95.0, 300.0, 1000.0):
        f = thermal_grip_factor(t, soft)
        assert MIN_THERMAL_GRIP <= f <= 1.0


def test_grip_loss_is_monotone_in_wear(soft):
    prev = 2.0
    for w in np.linspace(0.0, 1.0, 11):
        f = wear_grip_factor(w, soft)
        assert f <= prev, "more wear can never increase grip"
        prev = f
    assert wear_grip_factor(0.0, soft) == pytest.approx(1.0)
    assert wear_grip_factor(1.0, soft) == pytest.approx(1.0 - soft.max_wear_grip_loss)


def test_wet_behaviour_is_compound_specific_and_not_a_universal_mu(cfg):
    """A slick gets worse as it wets; a wet tyre gets worse as it dries."""
    slick = params_from_config(cfg, "SOFT")
    wet = params_from_config(cfg, "WET")
    assert wet_grip_factor(0.0, slick) > wet_grip_factor(1.0, slick)
    assert wet_grip_factor(1.0, wet) > wet_grip_factor(0.0, wet)
    # and in a downpour the wet compound must beat the slick outright
    assert (mu_effective(60.0, 0.0, 1.0, wet)[0]
            > mu_effective(100.0, 0.0, 1.0, slick)[0])
    # in the dry the slick must win, or the compounds are interchangeable
    assert (mu_effective(100.0, 0.0, 0.0, slick)[0]
            > mu_effective(100.0, 0.0, 0.0, wet)[0])


def test_no_magic_universal_mu(cfg):
    """Compounds must actually differ, and all mu must be tyre-like."""
    mus = {c: params_from_config(cfg, c).mu_long_base
           for c in ("SOFT", "MEDIUM", "HARD")}
    assert mus["SOFT"] > mus["MEDIUM"] > mus["HARD"], "compounds are not ordered"
    for c, mu in mus.items():
        assert 1.0 < mu < 2.0, f"{c} mu {mu} is not a tyre friction coefficient"


# -------------------------------------------------------------- the envelope
def test_available_force_depends_on_aerodynamic_load(soft):
    """The join with P1.2, and the reason downforce was worth adding."""
    m = 800.0
    n0, long0, lat0 = force_limits(m, 0.0, 100.0, 0.0, 0.0, soft)
    n1, long1, lat1 = force_limits(m, 20000.0, 100.0, 0.0, 0.0, soft)
    assert n0 == pytest.approx(m * G)
    assert n1 == pytest.approx(m * G + 20000.0)
    assert long1 > long0 and lat1 > lat0, "force must scale with normal load"
    assert long0 == pytest.approx(soft.mu_long_base * n0)


def test_utilisation_is_the_friction_ellipse(soft):
    assert utilisation(0.0, 0.0, 1000.0, 1000.0) == pytest.approx(0.0)
    assert utilisation(1000.0, 0.0, 1000.0, 1000.0) == pytest.approx(1.0)
    assert utilisation(0.0, 1000.0, 1000.0, 1000.0) == pytest.approx(1.0)
    # combined use exceeds either alone
    assert utilisation(1000.0, 1000.0, 1000.0, 1000.0) == pytest.approx(np.sqrt(2))
    # and past the limit is reported, not clipped away
    assert utilisation(2000.0, 0.0, 1000.0, 1000.0) > 1.0


# ------------------------------------------------------------ state advance
def test_temperature_relaxes_towards_a_target_and_stays_finite(soft):
    st = fresh("SOFT", soft, track_temp_c=30.0)
    hot = st
    for _ in range(400):
        hot = advance(hot, soft, dt_s=0.5, ds_m=40.0, track_temp_c=30.0,
                      util=1.0, wetness=0.0)
    assert np.isfinite(hot.estimated_temp_c)
    target = 30.0 + soft.heating_gain * 1.0
    assert hot.estimated_temp_c == pytest.approx(target, abs=1.0)
    # cooling works too: drop the load and it comes back down
    cool = hot
    for _ in range(400):
        cool = advance(cool, soft, dt_s=0.5, ds_m=0.0, track_temp_c=30.0, util=0.0)
    assert cool.estimated_temp_c < hot.estimated_temp_c
    assert cool.estimated_temp_c == pytest.approx(30.0, abs=1.0)


def test_wear_is_monotone_bounded_and_only_reset_by_a_tyre_change(soft):
    st = fresh("SOFT", soft, track_temp_c=95.0)
    prev = st.wear_fraction
    for _ in range(2000):
        st = advance(st, soft, dt_s=0.1, ds_m=80.0, track_temp_c=95.0, util=1.0)
        assert st.wear_fraction >= prev, "wear ran backwards"
        prev = st.wear_fraction
    assert 0.0 <= st.wear_fraction <= 1.0
    assert st.wear_fraction > 0.0, "a stint of hard running must wear the tyre"
    # time alone does not reset it; only a change does
    assert advance(st, soft, dt_s=0.1, ds_m=0.0, track_temp_c=95.0,
                   util=0.0).wear_fraction >= st.wear_fraction
    assert fresh("SOFT", soft, track_temp_c=95.0).wear_fraction == 0.0


def test_harder_use_wears_faster(soft):
    def stint(util):
        st = fresh("SOFT", soft, track_temp_c=95.0)
        for _ in range(500):
            st = advance(st, soft, dt_s=0.1, ds_m=80.0, track_temp_c=95.0, util=util)
        return st.wear_fraction

    assert stint(1.0) > stint(0.5) > stint(0.0)


def test_thermal_wear_multiplier_rises_both_sides_and_is_capped(soft):
    best = thermal_wear_multiplier(soft.optimal_temp_c, soft)
    assert best == pytest.approx(1.0)
    assert thermal_wear_multiplier(soft.optimal_temp_c - 40.0, soft) > best
    assert thermal_wear_multiplier(soft.optimal_temp_c + 40.0, soft) > best
    for t in (-100.0, 500.0):
        assert 1.0 <= thermal_wear_multiplier(t, soft) <= MAX_THERMAL_WEAR_MULTIPLIER


def test_grip_scale_is_bounded_over_the_whole_state_space(soft):
    for t in (-20.0, 40.0, 95.0, 200.0):
        for w in (0.0, 0.5, 1.0):
            for wet in (0.0, 0.5, 1.0):
                k = grip_scale(t, w, wet, soft)
                assert 0.0 < k <= 1.0, f"grip_scale {k} out of bounds"


def test_all_tyre_coefficients_live_in_config_not_in_code(cfg):
    """Every compound is configured; the module holds only bounds."""
    table = cfg["tyres"]["compounds"]
    for c in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "_NEUTRAL"):
        assert c in table, f"{c} missing from config"
    p = params_from_config(cfg, "MEDIUM")
    assert p.mu_long_base == table["MEDIUM"]["mu_long_base"]
