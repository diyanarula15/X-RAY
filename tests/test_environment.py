"""P1.1: causal weather, one density formula, and a wind sign that is tested.

The wind convention is the part worth testing rather than reading. A sign error
here is invisible -- the car simply goes slightly wrong-ish -- so headwind and
tailwind are asserted against a fixture whose tangent is known exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.data.circuits import CircuitGeometry, RealTrack
from xray.data.ingest import air_density
from xray.environment import (UNORIENTED, EnvironmentalState, TrackFrame,
                              environment_at_time, project_wind_along_track,
                              relative_airspeed, state_from_summary,
                              tangent_to_enu, timeline_from_samples,
                              wind_vector_enu, with_wetness)

# A documented orientation. In a test the provenance is "the fixture was built
# this way"; in production it has to be a survey or a circuit map, and
# TrackFrame.oriented rejects "unknown"/"assumed" so an undocumented value
# cannot quietly enable the projection.
NORTH_ALIGNED = TrackFrame(0.0, "test fixture: +X built pointing true north")


def _samples():
    return [
        {"t": 0.0, "air_temp_c": 20.0, "track_temp_c": 30.0, "pressure_mbar": 1010.0,
         "humidity_pct": 40.0, "wind_speed_ms": 2.0, "wind_dir_deg": 0.0,
         "rainfall": False, "rho": air_density(20.0, 1010.0, 40.0)},
        {"t": 60.0, "air_temp_c": 22.0, "track_temp_c": 34.0, "pressure_mbar": 1009.0,
         "humidity_pct": 45.0, "wind_speed_ms": 5.0, "wind_dir_deg": 90.0,
         "rainfall": False, "rho": air_density(22.0, 1009.0, 45.0)},
        {"t": 120.0, "air_temp_c": 19.0, "track_temp_c": 25.0, "pressure_mbar": 1012.0,
         "humidity_pct": 80.0, "wind_speed_ms": 8.0, "wind_dir_deg": 180.0,
         "rainfall": True, "rho": air_density(19.0, 1012.0, 80.0)},
    ]


def _straight_track(heading_xy=(0.0, 1.0), n=400, length=1000.0):
    """A fixture whose tangent is known: a circle is not, a line is."""
    h = np.asarray(heading_xy, dtype=float)
    h = h / np.linalg.norm(h)
    s = np.linspace(0.0, length, n, endpoint=False)
    xy = np.outer(s, h)
    return RealTrack(CircuitGeometry(
        name="straight", length=length, s=s, xy=xy, z=np.zeros(n),
        grade=np.zeros(n), curvature=np.zeros(n),
        is_corner=np.zeros(n, bool), has_elevation=False, source="test"))


# ------------------------------------------------------------------ causality
def test_environment_lookup_is_causal():
    tl = timeline_from_samples(_samples())
    assert environment_at_time(tl, 59.9).t == 0.0
    assert environment_at_time(tl, 60.0).t == 60.0
    assert environment_at_time(tl, 119.9).t == 60.0
    assert environment_at_time(tl, 1e6).t == 120.0
    # age is the staleness of the sample actually used
    assert environment_at_time(tl, 90.0).age_s == pytest.approx(30.0)


def test_future_weather_cannot_change_a_past_lookup():
    """The mutation test, on the weather feed itself."""
    rows = _samples()
    before = environment_at_time(with_wetness(timeline_from_samples(rows)), 90.0)
    for r in rows:
        if r["t"] > 90.0:
            r.update(air_temp_c=99.0, pressure_mbar=800.0, wind_speed_ms=40.0,
                     wind_dir_deg=270.0, rainfall=True, rho=0.5)
    after = environment_at_time(with_wetness(timeline_from_samples(rows)), 90.0)
    assert after == before


def test_before_the_first_sample_is_labelled_not_silently_extrapolated():
    tl = timeline_from_samples(_samples())
    st = environment_at_time(tl, -10.0)
    assert st.source.endswith("before_first_sample")
    assert st.age_s < 0.0


# --------------------------------------------------------------- air density
def test_air_density_is_the_sole_formula_and_is_monotone():
    """Pressure up => denser; temperature up => thinner."""
    assert air_density(20.0, 1020.0, 50.0) > air_density(20.0, 1000.0, 50.0)
    assert air_density(35.0, 1010.0, 50.0) < air_density(5.0, 1010.0, 50.0)
    # humid air is lighter than dry air at the same temperature and pressure
    assert air_density(25.0, 1010.0, 90.0) < air_density(25.0, 1010.0, 10.0)
    # the timeline carries that same function's output, not a second formula
    tl = timeline_from_samples(_samples())
    assert tl.rho[0] == pytest.approx(air_density(20.0, 1010.0, 40.0))


# ---------------------------------------------------------------- wind sign
def test_wind_vector_enu_for_all_four_cardinal_winds():
    """WindDirection is the bearing the wind comes FROM, so the air goes the other way."""
    def enu(bearing):
        return wind_vector_enu(state_from_summary(
            {"wind_speed_ms": 10.0, "wind_dir_deg": bearing}))

    assert enu(0.0) == pytest.approx([0.0, -10.0], abs=1e-9)    # from N -> blows S
    assert enu(90.0) == pytest.approx([-10.0, 0.0], abs=1e-9)   # from E -> blows W
    assert enu(180.0) == pytest.approx([0.0, 10.0], abs=1e-9)   # from S -> blows N
    assert enu(270.0) == pytest.approx([10.0, 0.0], abs=1e-9)   # from W -> blows E


def test_tangent_rotates_into_enu_by_known_headings():
    """heading is the true bearing of the LOCAL +X axis."""
    # +X north: local +X -> north, local +Y (90 deg ccw of +X) -> west
    assert tangent_to_enu([1.0, 0.0], TrackFrame(0.0, "fix")) == pytest.approx([0.0, 1.0], abs=1e-9)
    assert tangent_to_enu([0.0, 1.0], TrackFrame(0.0, "fix")) == pytest.approx([-1.0, 0.0], abs=1e-9)
    # +X east
    assert tangent_to_enu([1.0, 0.0], TrackFrame(90.0, "fix")) == pytest.approx([1.0, 0.0], abs=1e-9)
    assert tangent_to_enu([0.0, 1.0], TrackFrame(90.0, "fix")) == pytest.approx([0.0, 1.0], abs=1e-9)
    # +X south and west
    assert tangent_to_enu([1.0, 0.0], TrackFrame(180.0, "fix")) == pytest.approx([0.0, -1.0], abs=1e-9)
    assert tangent_to_enu([1.0, 0.0], TrackFrame(270.0, "fix")) == pytest.approx([-1.0, 0.0], abs=1e-9)
    # rotation preserves length
    out = tangent_to_enu([0.6, 0.8], TrackFrame(37.0, "fix"))
    assert float(np.linalg.norm(out)) == pytest.approx(1.0)


def test_rotation_by_a_known_heading_changes_the_answer():
    """The same local tangent and the same wind give opposite results at 180 deg."""
    track = _straight_track(heading_xy=(1.0, 0.0))    # local +X is the direction of travel
    st = state_from_summary({"wind_speed_ms": 10.0, "wind_dir_deg": 0.0})   # from north
    north = project_wind_along_track(track, 10.0, st, TrackFrame(0.0, "fix"))
    south = project_wind_along_track(track, 10.0, st, TrackFrame(180.0, "fix"))
    assert north.wind_along_track_mps == pytest.approx(-10.0)   # driving north into it
    assert south.wind_along_track_mps == pytest.approx(10.0)    # driving south with it
    east = project_wind_along_track(track, 10.0, st, TrackFrame(90.0, "fix"))
    assert east.wind_along_track_mps == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bearing,expect_sign", [(0.0, -1), (180.0, +1)])
def test_headwind_and_tailwind_signs(bearing, expect_sign):
    """Positive along-track is a tailwind; v_air = v - w_along."""
    track = _straight_track(heading_xy=(1.0, 0.0))
    st = state_from_summary({"wind_speed_ms": 10.0, "wind_dir_deg": bearing})
    wp = project_wind_along_track(track, 10.0, st, NORTH_ALIGNED)
    assert wp.available
    assert np.sign(wp.wind_along_track_mps) == expect_sign
    v = 80.0
    v_air = relative_airspeed(v, wp.wind_along_track_mps)
    assert v_air == pytest.approx(v + 10.0 if expect_sign < 0 else v - 10.0)


def test_crosswind_has_no_along_track_component():
    track = _straight_track(heading_xy=(1.0, 0.0))
    st = state_from_summary({"wind_speed_ms": 12.0, "wind_dir_deg": 90.0})
    wp = project_wind_along_track(track, 10.0, st, NORTH_ALIGNED)
    assert wp.wind_along_track_mps == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------- refusals, not guesses
def test_projection_is_unavailable_without_a_documented_orientation():
    """Missing orientation must not become a zero, and must not imply no wind."""
    track = _straight_track()
    st = state_from_summary({"wind_speed_ms": 9.0, "wind_dir_deg": 45.0})

    wp = project_wind_along_track(track, 10.0, st)              # default UNORIENTED
    assert wp.available is False
    assert wp.wind_along_track_mps is None, "None, not 0.0 -- unknown is not calm"
    assert wp.confidence == 0.0
    assert "heading_unknown" in wp.source
    # a force model still needs a number, and choosing it is explicit
    assert wp.value_or_zero == 0.0

    # an undocumented heading is a guess, so it does not count as oriented
    for bad in (TrackFrame(0.0, "unknown"), TrackFrame(0.0, "assumed"),
                TrackFrame(None, "survey")):
        assert not bad.oriented
        assert not project_wind_along_track(track, 10.0, st, bad).available


def test_unoriented_frame_cannot_be_rotated_at_all():
    with pytest.raises(ValueError, match="heading is unknown"):
        tangent_to_enu([1.0, 0.0], UNORIENTED)


def test_circuit_sigma_has_no_physical_tangent_so_wind_is_unavailable():
    """Sigma's xy() is documented cosmetic. It must not become a physics input."""
    from xray.track import circuit_sigma
    track = circuit_sigma()
    assert not hasattr(track, "tangent"), (
        "Circuit Sigma grew a tangent(); its xy() is cosmetic, so promoting it "
        "to physical geometry needs an explicit contract change and a test")
    st = state_from_summary({"wind_speed_ms": 9.0, "wind_dir_deg": 45.0})
    wp = project_wind_along_track(track, 10.0, st, NORTH_ALIGNED)
    assert wp.available is False
    assert wp.wind_along_track_mps is None


# -------------------------------------------------------------------- wetness
def test_wetness_is_inferred_bounded_and_labelled():
    tl = with_wetness(timeline_from_samples(_samples()))
    wet = tl.track_wetness_index
    assert wet is not None
    assert np.all((wet >= 0.0) & (wet <= 1.0))
    assert wet[0] == 0.0
    assert wet[2] > wet[1], "rain must wet the track"
    st = environment_at_time(tl, 130.0)
    assert st.rainfall is True
    assert st.track_wetness_index > 0.0


def test_wetness_dries_and_a_warmer_track_dries_faster():
    rows = _samples() + [
        {"t": 180.0, "air_temp_c": 19.0, "track_temp_c": 20.0, "pressure_mbar": 1012.0,
         "humidity_pct": 70.0, "wind_speed_ms": 3.0, "wind_dir_deg": 0.0,
         "rainfall": False, "rho": 1.2}]
    cool = with_wetness(timeline_from_samples(rows)).track_wetness_index
    rows[-1]["track_temp_c"] = 50.0
    warm = with_wetness(timeline_from_samples(rows)).track_wetness_index
    assert cool[-1] < cool[-2], "a dry sample must dry the track"
    assert warm[-1] < cool[-1], "a warmer track must dry faster"


def test_summary_state_is_labelled_a_mean():
    st = state_from_summary({"rho": 1.18, "wind_speed_ms": 3.0, "wind_dir_deg": 12.0})
    assert isinstance(st, EnvironmentalState)
    assert "mean" in st.source


# --------------------------------------------- no double-counted wind
def _kin(w_along=None, source="unavailable"):
    """Minimal Kin for the wind-term algebra. Values are arbitrary but fixed."""
    from xray.realfit import Kin
    n = 64
    v = np.linspace(50.0, 90.0, n)
    return Kin(s=np.linspace(0, 1000, n), v=v, a=np.zeros(n),
               dt=np.full(n, 0.1), t=np.linspace(0, 6.3, n),
               mass=np.full(n, 800.0), cda_scale=np.ones(n),
               sin_grade=np.zeros(n), ceiling=np.zeros(n),
               throttle=None, brake=None, valid=np.ones(n, bool),
               lap=np.zeros(n, int),
               w_along_mps=(None if w_along is None else np.full(n, float(w_along))),
               wind_source=source)


def test_without_orientation_the_wind_term_is_exactly_the_legacy_expression():
    """The fallback must be bit-identical, or 'preserved pathway' is a claim."""
    from xray.realfit import _terms
    kin = _kin(None)
    _, b = _terms(kin, 3.0, 0.012, 1.2)
    expected = 0.5 * 1.2 * kin.cda_scale * (kin.v + 3.0) ** 2 * kin.v
    assert np.allclose(b, expected), "legacy (v + v_wind) form was not preserved"


def test_measured_wind_is_subtracted_and_the_fitted_term_is_only_a_residual():
    """Measured + zero residual must equal measured-only physics.

    This is the no-double-count gate. The two conventions are opposed -- w_along
    is positive for a tailwind, v_wind_hat positive for a headwind -- so a sign
    slip here would show up as a wind of twice the size rather than as an error.
    """
    from xray.realfit import _terms
    v = _kin().v
    # 6 m/s tailwind measured, nothing left over for the fit to explain
    _, b_measured = _terms(_kin(6.0), 0.0, 0.012, 1.2)
    expected = 0.5 * 1.2 * (v - 6.0) ** 2 * v
    assert np.allclose(b_measured, expected)

    # the same 6 m/s expressed ONLY as a fitted headwind must differ in sign
    _, b_fitted_only = _terms(_kin(None), -6.0, 0.012, 1.2)
    assert np.allclose(b_fitted_only, expected), (
        "the two conventions are not reconciled: w_along=+6 must equal v_wind=-6")

    # and stacking both must NOT reproduce it -- that is the double count
    _, b_double = _terms(_kin(6.0), -6.0, 0.012, 1.2)
    assert not np.allclose(b_double, expected)
    assert np.allclose(b_double, 0.5 * 1.2 * (v - 12.0) ** 2 * v)


def test_the_fit_publishes_which_meaning_its_wind_term_carries():
    """v_wind_hat keeps its name, so the meaning has to be readable."""
    from xray.realfit import RealNuisanceFit
    f = RealNuisanceFit.__dataclass_fields__
    assert "v_wind_is_residual" in f and "wind_source" in f
    assert f["v_wind_is_residual"].default is False, (
        "the safe default is 'this is the whole wind', matching the legacy path")
