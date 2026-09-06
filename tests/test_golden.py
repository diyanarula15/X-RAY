"""Golden estimator outputs: the baseline every optimisation must reproduce.

An optimisation is allowed to be faster. It is not allowed to change the answer.
Vectorising a loop, adding `numba`, or porting a kernel to C++ all reassociate
floating-point sums, so the tolerance below is loose enough to survive that and
tight enough that a real numerics change cannot hide in it: 1e-9 relative is
about 4 mJ on a 4 MJ store, against a per-lap energy the estimator only claims
to 3.5%.

Regenerate with `python scripts/make_golden.py --seed 42` ONLY when a numerics
change is intended, and say which headline metric moved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.golden_io import SCALAR_NAMES, belief_arrays

GOLDEN = Path(__file__).resolve().parent / "golden" / "belief_seed42_3.7hz.npz"
RTOL = 1e-9
ATOL = 1e-6   # J. Absolute floor so a zero-valued sample cannot fail on rtol.


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"no golden file at {GOLDEN}; run scripts/make_golden.py")
    return np.load(GOLDEN)


def test_belief_matches_golden(golden, beliefs):
    """Per-sample beliefs and per-lap flows, against the committed baseline."""
    _obs, bel = beliefs[(42, "LEADER", 3.7)]
    got = belief_arrays(bel)
    for key in golden.files:
        if key in ("scalars", "dry_events"):
            continue
        np.testing.assert_allclose(
            got[key], golden[key], rtol=RTOL, atol=ATOL,
            err_msg=f"{key} drifted from golden -- intended, or a regression?")


def test_regime_classification_matches_golden(golden, beliefs):
    """Deployment cut-outs are a discrete decision, so they must match exactly.

    A single flipped cut-out moves the store by up to the buffer width (0.52 MJ
    on this seed), which no energy tolerance would catch.
    """
    _obs, bel = beliefs[(42, "LEADER", 3.7)]
    assert np.array_equal(bel.dry_events, golden["dry_events"])


def test_stage_a_scalars_match_golden(golden, beliefs):
    """The nuisance fit the beliefs were derived from.

    Scored separately because a port could match the beliefs while drifting on
    CdA and be reproducing the answer by luck.
    """
    _obs, bel = beliefs[(42, "LEADER", 3.7)]
    got = belief_arrays(bel)["scalars"]
    for name, a, b in zip(SCALAR_NAMES, got, golden["scalars"]):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL,
                                   err_msg=f"Stage A scalar {name} drifted")
