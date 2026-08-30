"""Shared fixtures. The simulation is expensive; build it once per session."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.config import load_config          # noqa: E402
from xray.estimator import estimate          # noqa: E402
from xray.observe import observe             # noqa: E402
from xray.sim import FOLLOWER, LEADER, run_sim  # noqa: E402

CARS = (LEADER, FOLLOWER)
SEEDS = (42, 7)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def races(cfg):
    return {s: run_sim(cfg, seed=s) for s in SEEDS}


@pytest.fixture(scope="session")
def gt(races):
    return races[SEEDS[0]]


@pytest.fixture(scope="session")
def beliefs(cfg, races):
    """(seed, car, rate) -> (observation, belief), estimated once."""
    out = {}
    for seed, g in races.items():
        for car in CARS:
            for rate in (3.7,):
                obs = observe(g, car, rate_hz=rate,
                              speed_noise_ms=cfg["observe"]["speed_noise_ms"],
                              seed=seed + 1)
                out[(seed, car, rate)] = (
                    obs, estimate(obs, g.track,
                                  n_particles=cfg["estimator"]["n_particles"],
                                  seed=seed + 2))
    return out
