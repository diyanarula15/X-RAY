"""Overtake probability model.

PLACEHOLDER COEFFICIENTS. Every number below is a prototype value, NOT a fit to
data. ``b1`` and ``b2`` are taken from the design brief; ``b0`` and ``b3`` are
solved here from two shape anchors rather than pasted in, so the derivation is
executable instead of a comment that can drift from the constants under it.
They must be refitted on real 2026 race data before any of this means anything,
and `xray.passmodel` still refuses that fit -- see its `audit_dataset`.

TWO ANCHOR SETS, because the two call sites feed this function two different
physical quantities and one set cannot serve both:

  ASSUMED_ENERGY_DV_COEFFS (the default) -- for `decision.delta_v`, which is a
      speed delta IMPLIED BY AN ENERGY DIFFERENCE: `own_speed(e_own) -
      rival_speed(e_riv)`, at roughly 1.3-2.3 m/s per MJ.
  ASSUMED_BRIEF_COEFFS -- for `sim.py`, which feeds the OBSERVED peak-speed
      difference between two cars at a braking zone. That is a different
      measurement with a different range, and it is also what every golden
      trace was generated against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GAP_SCALE_S = 1.5

# The brief's two shape anchors: at this gap, a follower this much faster passes
# with p_A in Zone A (severity 1.0) and p_C in Zone C (severity 0.2).
ASSUMED_ANCHOR_GAP_S = 0.3
ASSUMED_ANCHOR_P_ZONE_A = 0.70
ASSUMED_ANCHOR_P_ZONE_C = 0.25
_SEVERITY_ZONE_A = 1.0
_SEVERITY_ZONE_C = 0.2

# The brief's closing-speed anchor.
ASSUMED_BRIEF_ANCHOR_DV_MPS = 8.0

# The same anchor, re-solved for the energy-implied delta_v the decision path
# actually produces. 8 m/s is not reachable by that quantity: the zone models
# measure 1.291-2.286 m/s per MJ against a 4 MJ store, so 8 m/s requires one car
# holding the entire store while the other holds none. Between two cars actually
# racing each other the measured spread is p50 -0.013, p90 0.081, max 1.357 m/s
# over 515 candidate actions in out/decisions/.
#
# Evaluated 5x outside its anchored range, the logistic returned p50 0.000 / p75
# 0.000 / max 0.017 on every real row, so attacking carried no expected value
# and `value_action == value_hold` to the digit on 1423 of 1469 rows. P2 was not
# choosing HOLD on 97% of rows, it was reporting a tie and taking the tiebreak.
# This does NOT make the model calibrated -- it moves it from degenerate to
# discriminating, which is a weaker claim and the UI must keep saying so.
ASSUMED_ENERGY_DV_ANCHOR_MPS = 1.3


@dataclass(frozen=True)
class PassCoeffs:
    b0: float
    b1: float = 0.55     # per m/s of closing speed at the end of the straight
    b2: float = 2.10     # proximity at the braking point
    b3: float = 2.4324   # solved from the anchors -- PLACEHOLDER


def _solve_anchors(anchor_dv_mps: float, b1: float = 0.55, b2: float = 2.10
                   ) -> PassCoeffs:
    """Solve b0 and b3 from the two shape anchors at a given closing speed.

    b3 falls out of the Zone A / Zone C spread and so does not depend on the
    closing-speed anchor at all; only b0 moves when the anchor moves. Solving
    rather than hardcoding is what makes the two coefficient sets below provably
    the same model at two scales.
    """
    logit = lambda p: math.log(p / (1.0 - p))
    proximity = 1.0 - min(ASSUMED_ANCHOR_GAP_S, GAP_SCALE_S) / GAP_SCALE_S
    b3 = ((logit(ASSUMED_ANCHOR_P_ZONE_A) - logit(ASSUMED_ANCHOR_P_ZONE_C))
          / (_SEVERITY_ZONE_A - _SEVERITY_ZONE_C))
    b0 = (logit(ASSUMED_ANCHOR_P_ZONE_A) - b1 * anchor_dv_mps
          - b2 * proximity - b3 * _SEVERITY_ZONE_A)
    return PassCoeffs(b0=b0, b1=b1, b2=b2, b3=b3)


# b0 = -7.6651, b3 = 2.4324 -- reproduces the constants this module carried
# before the anchors were made executable.
ASSUMED_BRIEF_COEFFS = _solve_anchors(ASSUMED_BRIEF_ANCHOR_DV_MPS)
# b0 = -3.9801, b3 = 2.4324 (b3 is unchanged by construction).
ASSUMED_ENERGY_DV_COEFFS = _solve_anchors(ASSUMED_ENERGY_DV_ANCHOR_MPS)

COEFFS = ASSUMED_ENERGY_DV_COEFFS


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
