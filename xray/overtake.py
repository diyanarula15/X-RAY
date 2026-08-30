"""Overtake probability model.

PLACEHOLDER COEFFICIENTS. Every number in ``PassCoeffs`` is a prototype value,
NOT a fit to data. b1 and b2 are taken from the design brief; b0 and b3 are
solved from the two shape anchors the brief specifies (dv = 8 m/s at a 0.3 s
gap gives p ~ 0.70 in Zone A and p ~ 0.25 in Zone C). They must be refitted on
real 2026 race data before any of this means anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAP_SCALE_S = 1.5


@dataclass(frozen=True)
class PassCoeffs:
    b0: float = -7.6650  # solved from the Zone A / Zone C anchors -- PLACEHOLDER
    b1: float = 0.55     # per m/s of closing speed at the end of the straight
    b2: float = 2.10     # proximity at the braking point
    b3: float = 2.4324   # solved from the anchors -- PLACEHOLDER


COEFFS = PassCoeffs()


def p_pass(delta_v_end, gap_at_braking, zone, coeffs: PassCoeffs = COEFFS):
    """Probability that a pass completes.

    delta_v_end : m/s, follower minus leader at the end of the straight
    gap_at_braking : s
    zone : anything with a ``braking_severity`` in [0, 1]
    """
    severity = zone.braking_severity if hasattr(zone, "braking_severity") else float(zone)
    proximity = 1.0 - np.minimum(np.asarray(gap_at_braking, dtype=float),
                                 GAP_SCALE_S) / GAP_SCALE_S
    z = (coeffs.b0 + coeffs.b1 * np.asarray(delta_v_end, dtype=float)
         + coeffs.b2 * proximity + coeffs.b3 * severity)
    p = 1.0 / (1.0 + np.exp(-z))
    return float(p) if np.ndim(p) == 0 else p
