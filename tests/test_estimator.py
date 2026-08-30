"""Estimator acceptance tests -- including the one that keeps it honest."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from xray.estimator import EstimatorError, estimate, fit_nuisance
from xray.metrics import score_estimate
from xray.observe import Observation, observe
from xray.sim import FOLLOWER, LEADER

CARS = (LEADER, FOLLOWER)
FORBIDDEN = {"sim", "policy", "vehicle"}
ESTIMATOR = Path(__file__).resolve().parent.parent / "xray" / "estimator.py"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[-1])
            for a in node.names:
                names.add(a.name.split(".")[-1])
    return names


def test_estimator_is_blind():
    """The blindfold, enforced. estimator.py may not reach the simulator."""
    leaked = _imported_modules(ESTIMATOR) & FORBIDDEN
    assert not leaked, f"estimator.py imports the simulator: {sorted(leaked)}"


def test_observation_carries_no_hidden_state():
    fields = set(Observation.__dataclass_fields__)
    assert fields == {"t", "s", "v", "lap", "gap_to_leader", "car_id", "sample_rate_hz"}
    for banned in ("E", "soc", "energy", "throttle", "brake", "gear", "P_mguk",
                   "deployment", "harvest"):
        assert banned not in fields


def test_nuisance_recovery(cfg, races, beliefs):
    """CdA_hat within 6% of the truth -- everything downstream rests on this."""
    truth = cfg["vehicle"]["cda_straight"]
    errs = []
    for (seed, car, rate), (_obs, bel) in beliefs.items():
        err = 100.0 * (bel.nuisance.cda_hat - truth) / truth
        errs.append(err)
        assert abs(err) <= 6.0, f"{car}/{seed}/{rate}Hz: CdA off by {err:+.2f}%"
    assert abs(np.mean(errs)) <= 3.0, f"systematic CdA bias {np.mean(errs):+.2f}%"


def test_energy_mape_at_3p7hz(cfg, races, beliefs):
    """Per-lap deployed energy within 15% at public-feed sample rate."""
    truth = cfg["vehicle"]["cda_straight"]
    for (seed, car, rate), (obs, bel) in beliefs.items():
        sc = score_estimate(races[seed], car, obs, bel, truth)
        assert sc.deployed_mape <= 15.0, f"{car}/{seed}: MAPE {sc.deployed_mape:.1f}%"


def test_energy_mape_at_100hz(cfg, races):
    """Per-lap deployed energy within 8% at full telemetry rate."""
    truth = cfg["vehicle"]["cda_straight"]
    seed = 42
    g = races[seed]
    for car in CARS:
        obs = observe(g, car, rate_hz=100.0, seed=seed + 1)
        bel = estimate(obs, g.track, n_particles=cfg["estimator"]["n_particles"],
                       seed=seed + 2)
        sc = score_estimate(g, car, obs, bel, truth)
        assert sc.deployed_mape <= 8.0, f"{car}: MAPE {sc.deployed_mape:.1f}%"


def test_band_coverage(cfg, races, beliefs):
    """True deployable energy inside p10-p90 between 75% and 85% of the time.

    Both bounds: an over-wide band is as wrong as an over-narrow one. The
    quantity scored is deployable energy, which is what the estimator claims to
    identify -- the raw store is identifiable only up to the driver's unspent
    buffer (see the README).
    """
    truth = cfg["vehicle"]["cda_straight"]
    cov = [score_estimate(races[seed], car, obs, bel, truth).usable_coverage
           for (seed, car, _rate), (obs, bel) in beliefs.items()]
    mean = float(np.mean(cov))
    assert 0.75 <= mean <= 0.85, f"band coverage {mean:.3f}, samples {np.round(cov, 3)}"


def test_estimator_refuses_when_the_window_closes(cfg, races):
    """It must fail loudly, not guess, when the calibration band is empty."""
    g = races[42]
    obs = observe(g, LEADER, rate_hz=3.7, seed=1)
    slow = Observation(t=obs.t, s=obs.s, v=np.minimum(obs.v, 250 / 3.6), lap=obs.lap,
                       gap_to_leader=obs.gap_to_leader, car_id=obs.car_id,
                       sample_rate_hz=obs.sample_rate_hz)
    with pytest.raises(EstimatorError, match="samples above"):
        fit_nuisance(slow, g.track)


def test_band_reacts_to_the_trace(cfg, races, beliefs):
    """The band must be a belief, not decoration: it should be measurably
    tighter where the trace is informative than where it is not."""
    from xray.metrics import band_width_by_regime
    for (seed, car, _rate), (obs, bel) in beliefs.items():
        w = band_width_by_regime(races[seed], car, obs, bel)
        assert w["accel"] < w["corner"], (
            f"{car}/{seed}: band no tighter under acceleration ({w})")


def test_reconstruction_is_reproducible(cfg, races):
    g = races[42]
    obs = observe(g, LEADER, rate_hz=3.7, seed=1)
    a = estimate(obs, g.track, n_particles=200, seed=5)
    b = estimate(obs, g.track, n_particles=200, seed=5)
    assert np.array_equal(a.soc_mean, b.soc_mean)
    assert np.array_equal(a.usable_p10, b.usable_p10)
