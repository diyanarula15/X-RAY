from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from xray.config import load_config, merge
from xray.corpus import build_corpus
from xray.data.ingest import frame_from_fastf1, frame_from_observation
from xray.observe import observe, public_channels
from xray.realfit import (build_kin, deployment_trace, fit_nuisance_real,
                          deployment_zone_mask)
from xray.regs import POST_MIAMI, p_dep_max, rules_for_simulator
from xray.sim import LEADER, run_sim
from xray.snapshots import make_snapshot
from xray.track import circuit_sigma


class _CarData(pd.DataFrame):
    @property
    def _constructor(self):
        return _CarData

    def add_distance(self):
        return self


class _Lap(dict):
    def __init__(self, car):
        super().__init__(Driver="TST", LapNumber=1)
        self._car = car

    def get_car_data(self):
        return self._car


def _regular_df(track, n=520):
    s = np.linspace(0.0, track.length - 10.0, n)
    v = 58.0 + 12.0 * np.sin(2.0 * np.pi * s / track.length) ** 2
    t = np.cumsum(np.r_[0.0, np.diff(s) / v[:-1]])
    return pd.DataFrame({
        "distance": s,
        "speed": v,
        "time": t,
        "lap": np.zeros(n, dtype=int),
        "driver": ["TST"] * n,
        "usable": [True] * n,
        "throttle": np.full(n, 100.0),
        "brake": np.zeros(n),
    })


def test_canonical_zone_mask_controls_realfit_ceiling():
    track = circuit_sigma()
    df = _regular_df(track)
    kin = build_kin(df, track, 790.0, 1.20, regs=POST_MIAMI)
    mask, source = deployment_zone_mask(track, kin.s)

    assert source == "circuit_zone_geometry_current_distance"
    assert np.array_equal(kin.in_deployment_zone, mask)
    assert mask.any()
    assert (~mask).any()
    np.testing.assert_allclose(kin.ceiling, p_dep_max(kin.v, mask, POST_MIAMI))
    assert np.nanmax(kin.ceiling[~mask]) <= POST_MIAMI.p_dep_max_elsewhere + 1.0


def test_sim_and_fastf1_entry_points_share_grid_channel_semantics():
    grid = np.arange(0.0, 300.0, 10.0)
    t = np.linspace(0.0, 5.0, 80)
    s = np.linspace(0.0, 290.0, 80)
    v = np.linspace(40.0, 70.0, 80)
    lap = np.ones(80, dtype=int)
    obs = type("Obs", (), {"s": s, "t": t, "v": v, "lap": lap})()
    sim_frame, _ = frame_from_observation(
        obs, grid, channels={"throttle": np.full(80, 80.0),
                             "brake": np.zeros(80)}, driver="TST")

    car = _CarData({
        "Distance": s,
        "SessionTime": pd.to_timedelta(t, unit="s"),
        "Speed": v * 3.6,
        "Throttle": np.full(80, 80.0),
        "Brake": np.zeros(80),
    })
    real_frame, _ = frame_from_fastf1(_Lap(car), grid)

    assert {"distance", "speed", "time", "lap", "driver", "usable",
            "throttle", "brake"} <= set(sim_frame.columns)
    assert {"distance", "speed", "time", "lap", "driver", "usable",
            "throttle", "brake"} <= set(real_frame.columns)
    np.testing.assert_allclose(sim_frame["speed"], real_frame["speed"])
    np.testing.assert_allclose(sim_frame["throttle"], real_frame["throttle"])


def test_synthetic_public_observation_reaches_realfit_without_truth():
    cfg = merge(load_config(), sim={"n_laps": 3})
    gt = run_sim(cfg, seed=42)
    obs = observe(gt, LEADER, rate_hz=3.7,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
    df, _ = frame_from_observation(
        obs, np.arange(0.0, gt.track.length, 10.0),
        channels=public_channels(gt, LEADER, obs), driver=LEADER)

    assert "E" not in df.columns
    assert "P_mguk" not in df.columns
    kin = build_kin(df[df["usable"]], gt.track, 790.0, cfg["vehicle"]["rho"],
                    regs=rules_for_simulator())
    fit = fit_nuisance_real(kin, cfg["vehicle"]["rho"], regs=rules_for_simulator())
    tr = deployment_trace(kin, fit, regs=rules_for_simulator())

    assert np.isfinite(fit.cda_hat)
    assert fit.cda_hi >= fit.cda_lo
    assert np.isfinite(np.nanmean(tr["deploy_lo"]))
    assert np.isfinite(np.nanmean(tr["deploy_hi"]))


def test_corpus_manifest_is_versioned_and_fingerprinted(tmp_path):
    payload = {
        "id": "2026_r1_R", "year": 2026, "event": "Synthetic GP", "session": "R",
        "regulation": {"variant": "pre-Miami-2026"},
        "weather": {"rho": 1.2}, "weather_trace": [{"t": 0.0, "rho": 1.2}],
        "circuit_geometry": {"length": 5200.0, "has_elevation": False, "zones": []},
        "cars": {"VER": {"trace": {"t": [0.0]}}},
        "laps": [{"driver": "VER", "lap": 1, "compound": "MEDIUM",
                  "tyre_life": 1, "stint": 1, "fresh_tyre": True,
                  "pit_in_time_s": None, "pit_out_time_s": None}],
    }
    p = tmp_path / "race.json"
    p.write_text(json.dumps(payload))
    cfg = tmp_path / "default.yaml"
    cfg.write_text("x: 1\n")
    when = datetime(2026, 9, 12, tzinfo=timezone.utc)

    out = build_corpus([p], tmp_path / "corpus", config_path=cfg, generated_at=when)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_sessions"] == 1
    assert manifest["sessions"][0]["artifact_fingerprint"]
    with pytest.raises(FileExistsError):
        build_corpus([p], tmp_path / "corpus", config_path=cfg, generated_at=when)


def test_snapshot_inputs_are_stable_under_future_mutation():
    payload = {
        "id": "race", "year": 2026, "round": 1, "session": "R",
        "event": "Test GP", "circuit": "Test", "date": "2026-03-08",
        "regulation": {"variant": "pre-Miami-2026"},
        "weather": {"rho": 1.2},
        "weather_trace": [{"t": 5.0, "rho": 1.2}, {"t": 25.0, "rho": 1.1}],
        "circuit_geometry": {"length": 1000.0, "has_elevation": False,
                             "zones": [{"name": "A", "s_straight_start": 0.0,
                                        "s_straight_end": 100.0}]},
        "cars": {
            "A": {"trace": {"t": [0.0, 10.0, 20.0, 30.0], "s": [0, 1, 2, 3],
                            "lap": [1, 1, 2, 2], "v": [50, 51, 52, 53],
                            "usable_mean": [1, 1, 1, 1]}},
            "B": {"trace": {"t": [0.0, 10.0, 20.0, 30.0], "s": [0, 1, 2, 3],
                            "lap": [1, 1, 2, 2], "v": [49, 50, 51, 52],
                            "usable_mean": [1, 1, 1, 1]}},
        },
        "laps": [
            {"driver": "A", "lap": 1, "t_end": 12.0, "pit_in_time_s": None,
             "pit_out_time_s": None},
            {"driver": "A", "lap": 2, "t_end": 40.0, "pit_in_time_s": 32.0,
             "pit_out_time_s": 45.0},
        ],
        "gaps": [{"lap": 1, "car": "B", "ahead": "A", "gap_s": 0.8},
                 {"lap": 3, "car": "B", "ahead": "A", "gap_s": 4.0}],
    }
    before = make_snapshot(payload, 15.0, 20.0, 35.0, drivers=["A", "B"])
    mutated = json.loads(json.dumps(payload))
    mutated["cars"]["A"]["trace"]["v"][3] = 999.0
    mutated["weather_trace"][1]["rho"] = 0.5
    mutated["laps"][1]["pit_in_time_s"] = 22.0
    mutated["gaps"][1]["gap_s"] = 99.0
    after = make_snapshot(mutated, 15.0, 20.0, 35.0, drivers=["A", "B"])

    assert before["provenance"]["input_fingerprint"] == after["provenance"]["input_fingerprint"]
    assert before["provenance"]["evaluation_fingerprint"] != after["provenance"]["evaluation_fingerprint"]
