"""What a golden estimator baseline contains.

Shared by `scripts/99.make_golden.py` (which writes it) and `tests/test_golden.py`
(which checks against it), so the two cannot drift into disagreeing about which
fields define correctness.
"""
from __future__ import annotations

import numpy as np

SCALAR_NAMES = ("cda_hat", "cda_sigma", "v_wind_hat", "v_wind_sigma",
                "residual_rms", "systematic_rms", "v3_ref", "deploy_frac_hat",
                "deploy_scale_sigma", "reserve_mean", "reserve_sigma")


def belief_arrays(bel) -> dict:
    """The fields a compiled kernel would have to reproduce.

    Per-sample beliefs and per-lap flows, plus the Stage A scalars they were
    derived from -- a port that matched the beliefs while drifting on the
    nuisance fit would be reproducing the answer by luck.
    """
    n = bel.nuisance
    return {
        "t": bel.t, "soc_mean": bel.soc_mean, "soc_p10": bel.soc_p10,
        "soc_p90": bel.soc_p90, "usable_mean": bel.usable_mean,
        "usable_p10": bel.usable_p10, "usable_p90": bel.usable_p90,
        "deployed_lap": bel.deployed_lap, "harvested_lap": bel.harvested_lap,
        "lap_index": bel.lap_index, "p_mguk_mean": bel.p_mguk_mean,
        "harvest_mean": bel.harvest_mean, "dry_events": bel.dry_events,
        "scalars": np.array([
            n.cda_hat, n.cda_sigma, n.v_wind_hat, n.v_wind_sigma,
            n.residual_rms, n.systematic_rms, n.v3_ref, n.deploy_frac_hat,
            bel.deploy_scale_sigma, bel.reserve_mean, bel.reserve_sigma,
        ], dtype=np.float64),
    }
