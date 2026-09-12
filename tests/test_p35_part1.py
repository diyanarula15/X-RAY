"""P3.5 Part 1: exact re-inference, nuisance registry, identifiability diagnostics.

New surfaces only. P3's suite is authoritative and is not rerun here.

The property most of these defend is that a re-inference really is the SAME
estimator: if the window path and the live path can disagree, then every
sensitivity number measured through the window is a property of the artifact
rather than of the estimator, and the whole exercise is void.
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import pytest

from xray import estimator_diag as diag
from xray import nuisance as nz
from xray import reinfer as ri

WINDOWS = sorted(glob.glob("out/p35/windows/*.inputs.json"))
RACES = sorted(glob.glob("out/races/*.json"))


def _window():
    if not WINDOWS:
        pytest.skip("no P3.5 windows on disk")
    return ri.load_window(WINDOWS[0])


# ------------------------------------------------------ the artifact itself
def test_a_window_holds_only_estimator_inputs():
    w = _window()
    assert set(w.frame) == set(ri.FRAME_CHANNELS)
    assert set(w.track) == set(ri.TRACK_CHANNELS)
    # nothing resembling an outcome, a belief or a target
    banned = ("usable", "soc", "deploy", "harvest", "target", "future", "truth")
    for key in list(w.frame) + list(w.track):
        assert not any(b in key for b in banned), key


def test_every_input_sample_is_at_or_before_the_cutoff():
    """The causal wall. Asserted on the stored arrays, not on the builder."""
    for mp in WINDOWS:
        w = ri.load_window(mp)
        t = np.asarray(w.frame["time"], dtype=float)
        assert np.nanmax(t) <= w.cutoff_time_s + 1e-9, w.window_id


def test_targets_are_strictly_after_the_cutoff_and_stored_separately():
    if not WINDOWS:
        pytest.skip("no P3.5 windows on disk")
    mp = pathlib.Path(WINDOWS[0])
    tmeta = mp.parent / mp.name.replace(".inputs.json", ".targets.json")
    assert tmeta.exists(), "targets must be their own file, not a key in the inputs"
    meta = json.loads(tmeta.read_text())
    assert "EVALUATION ONLY" in meta["label"]
    z = np.load(mp.parent / meta["arrays_file"], allow_pickle=False)
    assert float(np.nanmin(z["t"])) > float(meta["cutoff_time_s"])


def test_the_input_metadata_never_mentions_a_target_file():
    """A loader that can see both namespaces will eventually read both."""
    meta = json.loads(pathlib.Path(WINDOWS[0]).read_text()) if WINDOWS else None
    if meta is None:
        pytest.skip("no P3.5 windows on disk")
    blob = json.dumps(meta).lower()
    assert "target" not in blob
    assert meta["arrays_file"].endswith(".inputs.npz")


def test_load_window_does_not_return_targets():
    w = _window()
    assert not hasattr(w, "targets")
    assert "targets" not in json.dumps(w.provenance()).lower()


def test_a_window_carries_the_assumptions_it_was_built_with():
    w = _window()
    a = w.assumptions
    # read from realfit rather than restated, so a drift there is visible here
    from xray.realfit import COAST_THROTTLE, RESERVE_SIGMA_REAL
    assert a.coast_throttle == COAST_THROTTLE
    assert a.reserve_sigma_j == RESERVE_SIGMA_REAL
    assert w.regulation_variant and w.source_fingerprint
    assert w.schema_version == ri.SCHEMA_VERSION


def test_the_window_stores_zone_geometry_and_not_a_precomputed_mask():
    """The canonical mask function must remain the only definition."""
    w = _window()
    assert w.zones and all("s_straight_start" in z for z in w.zones)
    assert "in_deployment_zone" not in w.track
    assert "in_deployment_zone" not in w.frame
    from xray.realfit import deployment_zone_mask
    mask, src = deployment_zone_mask(w.window_track(), w.frame["distance"])
    assert src == "circuit_zone_geometry_current_distance"
    assert mask.any(), "no sample fell inside a deployment zone"


def test_a_window_round_trips_through_disk_unchanged(tmp_path):
    w = _window()
    ri.save_window(w, None, tmp_path)
    back = ri.load_window(sorted(tmp_path.glob("*.inputs.json"))[0])
    for k, v in w.frame.items():
        assert np.allclose(np.nan_to_num(v, nan=-9e18),
                           np.nan_to_num(back.frame[k], nan=-9e18))
    assert back.assumptions == w.assumptions
    assert back.regulation_variant == w.regulation_variant


# ------------------------------------------- the re-inference is THE estimator
def test_the_frozen_track_reproduces_the_live_track_exactly():
    """If these ever diverge, every sensitivity number is an artifact property.

    Measured on the 2026 Australian GP: all twelve Kin fields bit-identical.
    """
    w = _window()
    t = w.window_track()
    s = np.asarray(w.frame["distance"], dtype=float)
    assert np.array_equal(np.asarray(t.cda_scale(s), dtype=float),
                          np.asarray(w.track["cda_scale"], dtype=float))
    assert np.array_equal(np.asarray(t.grade(s), dtype=float),
                          np.asarray(w.track["grade"], dtype=float))


def test_reinference_is_deterministic():
    w = _window()
    a, b = ri.reinfer(w), ri.reinfer(w)
    assert a.usable_mean_j == b.usable_mean_j
    assert a.cda_hat == b.cda_hat


def test_an_override_actually_reaches_the_estimator():
    """A no-op override surface would make every sensitivity reading zero."""
    w = _window()
    base = ri.reinfer(w)
    moved = ri.reinfer(w, mass_kg=860.0)
    assert moved.feasible
    assert moved.cda_hat != base.cda_hat, "mass did not reach the power balance"


def test_reinference_reports_set_and_quantile_semantics_separately():
    w = _window()
    r = ri.reinfer(w)
    assert r.cda_set_width == pytest.approx(r.cda_hi - r.cda_lo)
    # the contract names them; nothing here may call a set a posterior
    src = pathlib.Path(ri.__file__).read_text()
    assert "identified SET" in src and "particle QUANTILES" in src


# ------------------------------------------------------- the nuisance registry
def test_the_registry_lists_only_quantities_that_reach_the_equations():
    r = nz.nuisance_registry()
    names = {n["symbol"] for n in r["nuisances"]}
    for must in ("CdA", "m", "crr", "rho", "eta", "smooth_m"):
        assert must in names, must
    # things that exist in the repo but never reach realfit must be excluded
    blob = json.dumps(r["nuisances"]).lower()
    for absent in ("tyre", "wear", "cla", "downforce", "wake", "wetness", "p_pass"):
        assert absent not in blob, f"{absent} is not a nuisance of this estimator"
    assert set(r["not_reached_by_this_estimator"])


def test_every_nuisance_records_whether_data_can_constrain_it():
    for n in nz.NUISANCES:
        assert isinstance(n.constrainable_now, bool)
        assert n.constrainable_note
        assert n.p3_status
        assert n.enters, f"{n.symbol} does not say which equation it enters"


def test_overridable_nuisances_map_to_real_assumption_fields():
    fields = set(ri.EstimatorAssumptions().__dataclass_fields__)
    for n in nz.NUISANCES:
        if n.override is not None:
            assert n.override in fields, n.override


def test_the_rho_perturbation_is_fractional_not_absolute():
    """A fixed 1.17/1.23 pair would be a different perturbation per circuit."""
    w = _window()
    rho_n = next(n for n in nz.NUISANCES if n.override == "rho")
    got = dict(nz._variants(rho_n, w))
    base = w.assumptions.rho
    assert got["low"]["rho"] == pytest.approx(base * (1.0 + rho_n.low))
    assert got["high"]["rho"] == pytest.approx(base * (1.0 + rho_n.high))


def test_ranking_uses_the_low_to_high_span_not_one_arm():
    """One-sided ranking would tie a two-sided mover with a one-sided one."""
    two = {"window_id": "w", "nominal": {"usable_mean_j": 1.0e6},
           "arms": [
               {"nuisance": "two", "name": "two", "arm": "low", "feasible": True,
                "usable_mean_j": 0.6e6, "d_usable_width_j": 0.0,
                "d_deploy_mean_w": 0.0},
               {"nuisance": "two", "name": "two", "arm": "high", "feasible": True,
                "usable_mean_j": 1.4e6, "d_usable_width_j": 0.0,
                "d_deploy_mean_w": 0.0},
               {"nuisance": "one", "name": "one", "arm": "low", "feasible": True,
                "usable_mean_j": 0.6e6, "d_usable_width_j": 0.0,
                "d_deploy_mean_w": 0.0},
               {"nuisance": "one", "name": "one", "arm": "high", "feasible": True,
                "usable_mean_j": 1.0e6, "d_usable_width_j": 0.0,
                "d_deploy_mean_w": 0.0}]}
    rank = nz.rank_nuisances([two])
    assert rank[0]["nuisance"] == "two"
    assert rank[0]["span_mj"] > rank[1]["span_mj"]


# --------------------------------------------------------------- diagnostics
def test_floor_pinning_refuses_megajoule_input():
    """The unit slip that reported 100% of every car as degenerate."""
    with pytest.raises(ValueError, match="MJ"):
        diag.floor_pinning([0.2, 0.3], [0.1, 0.2], [0.3, 0.4])


def test_floor_pinning_detects_a_collapsed_cloud():
    n = 100
    um = np.concatenate([np.zeros(50), np.full(50, 1.5e6)])
    p10 = um.copy()
    p90 = np.concatenate([np.zeros(50), np.full(50, 2.0e6)])
    r = diag.floor_pinning(um, p10, p90)
    assert r["frac_reported_empty"] == pytest.approx(0.5)
    assert r["frac_band_degenerate"] == pytest.approx(0.5)
    assert r["frac_empty_and_degenerate"] == pytest.approx(0.5)


def test_the_shipped_payloads_show_the_floor_pinning_pathology():
    """The mechanism behind P3's negative ablation, on production output.

    Measured across 5 races / 105 car-races: 41.9% of samples report under
    0.05 MJ and 31.8% carry a band under 0.01 MJ wide.
    """
    if not RACES:
        pytest.skip("no race payloads on disk")
    with open(RACES[0]) as fh:
        d = json.load(fh)
    toJ = lambda a: [None if x is None else x * 1e6 for x in a]
    fr = []
    for car in d["cars"].values():
        tr = car["trace"]
        r = diag.floor_pinning(toJ(tr["usable_mean"]), toJ(tr["usable_p10"]),
                               toJ(tr["usable_p90"]))
        if r.get("available"):
            fr.append(r["frac_empty_and_degenerate"])
    assert fr
    assert float(np.mean(fr)) > 0.05, (
        "expected a material degenerate fraction; if this drops, the estimator "
        "changed and the P3.5 diagnosis needs redoing")


def test_a_degenerate_band_is_never_labelled_identifiable():
    """It was, and 'HIGH identifiability' of 0.004 MJ is the false-confidence bug."""
    m = diag.IdentifiabilityMetrics("w", 0.0, 5e3, 0.0, float("nan"), 0.3, 1.0,
                                    1.0, 15, 0, 0.0, 0.1)
    lab = diag.label_identifiability(m)
    assert lab["label"] == "UNIDENTIFIABLE" and lab["degenerate"]


def test_identifiability_labels_follow_the_variance_ratio():
    def mk(ratio, width=5.0e5):
        return diag.IdentifiabilityMetrics(
            "w", width, ratio * width / 2.563, width / 2.563, ratio,
            0.3, 1.0, 1.0, 15, 0, 0.0, 0.1)
    assert diag.label_identifiability(mk(0.1))["label"] == "HIGH"
    assert diag.label_identifiability(mk(0.7))["label"] == "MODERATE"
    assert diag.label_identifiability(mk(1.5))["label"] == "LOW"


def test_the_admissible_cda_domain_is_reporting_only_never_a_clamp():
    src = pathlib.Path(diag.__file__).read_text()
    assert "not a clamp" in src
    # an identified set spanning negative drag must report low overlap, not be cut
    assert diag._admissible_overlap(-2.0, 2.0) < 0.35
    assert diag._admissible_overlap(0.6, 0.8) == pytest.approx(1.0)


def test_the_saturation_diagnostic_measures_the_ceiling_regime():
    w = _window()
    s = diag.saturation_diagnosis(w)
    assert s["available"] and s["n_live"] > 100
    assert 0.0 <= s["frac_upper_edge_at_ceiling"] <= 1.0
    assert s["current_observation_semantics"] == "latent = ceiling (exact)"
    assert s["implied_correct_semantics"] == "latent <= ceiling (censored from above)"
    # the measured regime is large; this is the Part 2 target
    assert s["frac_upper_edge_at_ceiling"] > 0.2, s


def test_the_profile_is_labelled_offline_and_uses_no_future_data():
    w = _window()
    p = diag.energy_profile(w)
    assert p["label"] == "OFFLINE ANALYSIS — NOT LIVE POLICY INPUT"
    if p["available"]:
        assert p["uses_future_data"] is False
        assert p["profile_energy_hi_mj"] >= p["profile_energy_lo_mj"]


# ------------------------------------------------------------ mismatch harness
def test_the_mismatch_harness_reuses_the_p3_synthetic_path():
    from xray import mismatch
    assert mismatch._HARNESS.exists()
    h = mismatch._harness()
    assert hasattr(h, "shipping_realfit_on_sim")


def test_the_mismatch_cases_keep_the_estimator_assumption_fixed():
    from xray import mismatch
    cases = {c["case"] for c in mismatch.MISMATCH_CASES}
    assert "matched" in cases, "a control case is required"
    for c in mismatch.MISMATCH_CASES:
        for (section, _key) in c["cfg"]:
            assert section in ("vehicle", "observe"), section


def test_the_mismatch_verdict_flags_error_without_widening():
    from xray.mismatch import verdict
    rows = [
        {"case": "matched", "aggregate": {"soc_mae_mj": 0.20, "soc_width_mj": 0.50,
                                          "soc_containment": 0.5}},
        {"case": "silent", "aggregate": {"soc_mae_mj": 0.60, "soc_width_mj": 0.50,
                                         "soc_containment": 0.2}},
        {"case": "honest", "aggregate": {"soc_mae_mj": 0.60, "soc_width_mj": 1.20,
                                         "soc_containment": 0.5}},
    ]
    v = verdict(rows)
    by = {c["case"]: c for c in v["cases"]}
    assert by["silent"]["compensating_error"] is True
    assert by["honest"]["compensating_error"] is False
    assert v["n_compensating"] == 1
