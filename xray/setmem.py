"""Set-membership identification: the feasible set as a polytope.

The 1-D CdA bracket generalises. Every unknown in `balance.py` enters the
interval balance linearly, so each interval is a pair of half-spaces in
theta = (CdA_X, CdA_Z, F_rr, dm) and the feasible set is their intersection:

    P = { theta : A theta <= b }

Then every question is a linear program:

  - the identified range of any parameter is its min and max over P (2 LPs);
  - the identifiability score is the normalised width of that projection;
  - a point estimate is the Chebyshev centre, the deepest interior point;
  - P = empty is an *alarm*, not a fit. It means a bound or a mode label is
    wrong, and the minimum violation says by how much.

That last property is what "decline rather than guess" becomes when it is
exact. A thresholded estimator cannot tell "no data up here" from "the data
disagree with the rulebook". A polytope can: the first is a wide projection,
the second is an empty set.

Cost is a few LPs per lap, not per sample, so the whole thing is milliseconds.
Streaming is a running intersection -- see `prune_redundant` for why memory
stays bounded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .balance import N_PARAMS, PARAM_NAMES, Constraints

# Physical box on the parameters. Needed because the polytope can be unbounded
# in a direction no interval constrains -- a car that never ran in X mode says
# nothing about CdA_X, and an LP over an unbounded set has no answer to report.
# The box IS the prior, and it has to be an honest one. Given a +/-40 kg mass
# range and a free rolling-resistance coefficient the LP simply trades them
# against drag: measured on 20 Hz simulator data, that put the *true* mass
# offset outside the identified set (dm projected to [-40.0, -8.0] when the
# truth is 0.0) and drove every identifiability score to 0.00. Mass and rolling
# resistance are close to public -- the spec gives m_0 a N(published, 2 kg)
# prior and treats C_rr as common across the grid -- so pinning them to that is
# using information the estimator legitimately has, not smuggling in an answer.
# Drag area, the quantity actually being identified, keeps a range no prior
# could be accused of setting.
#
# A projection that lands *on* a box edge is reported in `at_box_edge`, because
# that is the estimator hitting its own prior rather than the data.
DEFAULT_BOX = np.array([
    [0.30, 3.00],    # CdA_X, m^2 -- an F1 car in its lowest-drag state
    [0.30, 3.00],    # CdA_Z, m^2 -- and its highest-downforce state
    [62.0, 124.0],   # F_rr, N at M_REF_KG: crr in [0.008, 0.016] on 790 kg
    [-6.0, 6.0],     # dm, kg -- 3 sigma of the spec's N(published, 2 kg)
])

# Reference scale per parameter, used to normalise a projection width into an
# identifiability score. Dividing by the projection's own midpoint is the
# obvious choice and it is wrong for dm, whose midpoint is legitimately zero.
IDENT_REF = np.array([1.0, 1.0, 93.0, 4.0])

IDENT_FULL_WIDTH = 0.5   # relative projection width scored as zero information.
                         # Matches the Stage 2 convention so the UI's
                         # identifiability numbers stay comparable.

# Fraction of half-spaces allowed to be wrong before the set is called empty.
#
# Hard set-membership has one fatal weakness: the intersection of a million
# half-spaces is destroyed by one bad one. Measured here -- at 3.7 Hz the true
# theta sits outside 1.6% of the lower bounds, entirely at braking onsets where
# the resampled brake channel mislabels the interval, and that was enough to
# report INFEASIBLE with a 228.2 kJ minimum violation on data whose parameters
# are known exactly.
#
# So a budget, and it is not a fudge: at 3 sigma a Gaussian tail puts 0.27% of
# constraints outside two-sided, and the brake-labelling error contributes the
# rest. Exceeding the budget is still an alarm -- that is the case where the
# regulation variant or the mode labelling is wrong rather than noisy.
OUTLIER_FRAC = 0.03
VIOL_TOL_J = 1.0


@dataclass(frozen=True)
class IdentifiedSet:
    """The projection of P onto each parameter, plus where its centre is."""
    lo: np.ndarray
    hi: np.ndarray
    centre: np.ndarray
    radius: float               # Chebyshev radius, in scaled units
    identifiability: np.ndarray  # per parameter, 0 (nothing) .. 1 (pinned)
    n_constraints: int
    empty: bool
    min_violation: float        # 0 when feasible; else the smallest possible
                                # max-violation, in joules
    at_box_edge: np.ndarray     # per parameter: the box, not the data, is binding
    notes: tuple = ()

    @property
    def width(self) -> np.ndarray:
        return self.hi - self.lo

    def usable(self, i: int = 0, min_ident: float = 0.25) -> bool:
        return (not self.empty) and bool(self.identifiability[i] >= min_ident)

    def describe(self) -> str:
        if self.empty:
            return (f"INFEASIBLE: no theta satisfies the regulation on this "
                    f"data (min violation {self.min_violation / 1e3:.1f} kJ)")
        return "  ".join(
            f"{n} [{lo:.3f}, {hi:.3f}] id {q:.2f}"
            for n, lo, hi, q in zip(PARAM_NAMES, self.lo, self.hi,
                                    self.identifiability))


def _scaled(A: np.ndarray, b: np.ndarray):
    """Column-scale the constraint matrix.

    The columns differ by four orders of magnitude -- the drag coefficient is
    ~5e4 J per m^2 while the mass column is ~1 J per kg -- and HiGHS reports
    spurious infeasibility on a matrix that badly conditioned. Scaling is
    undone before anything is reported.
    """
    scale = np.median(np.abs(A), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return A / scale, b, scale


def _bounds(box: np.ndarray, scale: np.ndarray):
    return [(lo * s, hi * s) for (lo, hi), s in zip(box, scale)]


def inconsistent_constraints(cons: Constraints, box: np.ndarray = DEFAULT_BOX,
                             tol: float = VIOL_TOL_J):
    """Which half-spaces cannot be satisfied together, and by how much.

    Solves the L1 relaxation

        min sum(s)   s.t.   A theta - b <= s,  s >= 0,  theta in box

    An L1 penalty is what makes this useful rather than merely feasible: it
    drives the solution to violate a *few* constraints a lot instead of all of
    them a little, so `s > 0` is a short list of named samples to go and look
    at. A min-max (Chebyshev) relaxation spreads the blame evenly and diagnoses
    nothing.

    The s-block is held sparse. Dense it would be N x N -- 41,099 constraints
    is a 1.7e9-entry matrix, which is not a rounding error in memory.
    """
    A, b, scale = _scaled(cons.A, cons.b)
    n_c, n_p = A.shape
    A_ub = sparse.hstack([sparse.csr_matrix(A),
                          -sparse.identity(n_c, format="csr")], format="csr")
    c = np.concatenate([np.zeros(n_p), np.ones(n_c)])
    res = linprog(c, A_ub=A_ub, b_ub=b,
                  bounds=_bounds(box, scale) + [(0.0, None)] * n_c,
                  method="highs")
    if not res.success:
        return (np.full(n_p, np.nan), np.zeros(n_c, bool), float("inf"),
                f"L1 relaxation failed: {res.message}")
    theta = res.x[:n_p] / scale
    slack = res.x[n_p:]
    # tol is in scaled units of b, which are joules -- b is an energy
    return theta, slack > tol, float(slack.max()), ""


def feasibility(cons: Constraints, box: np.ndarray = DEFAULT_BOX,
                outlier_frac: float = OUTLIER_FRAC) -> tuple:
    """Is the data consistent with the rulebook, allowing a small outlier budget?

    Returns (theta_l1, violated_mask, worst_violation_j, note). The caller
    decides emptiness from the *fraction* violated, so that a noisy sample and a
    wrong regulation variant produce different outcomes rather than both
    producing "infeasible".
    """
    theta, viol, worst, note = inconsistent_constraints(cons, box)
    return theta, viol, worst, note


def _project(A, b, bounds, i: int, sign: float):
    c = np.zeros(A.shape[1])
    c[i] = sign
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return (float(sign * res.fun) if res.success else np.nan), res.success


def chebyshev_centre(cons: Constraints, box: np.ndarray = DEFAULT_BOX):
    """The deepest interior point of P, and its depth.

    Preferred over the midpoint of the projections because the midpoint of a
    box that contains a thin slanted polytope need not be inside it at all --
    and a point estimate outside the feasible set violates the regulation,
    which is the one thing this whole construction exists to prevent.
    """
    A, b, scale = _scaled(cons.A, cons.b)
    norms = np.linalg.norm(A, axis=1)
    A_ub = np.hstack([A, norms[:, None]])
    c = np.zeros(A.shape[1] + 1)
    c[-1] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=b,
                  bounds=_bounds(box, scale) + [(0.0, None)], method="highs")
    if not res.success:
        return np.full(A.shape[1], np.nan), 0.0
    return res.x[:-1] / scale, float(res.x[-1])


def identify(cons: Constraints, box: np.ndarray = DEFAULT_BOX,
             outlier_frac: float = OUTLIER_FRAC) -> IdentifiedSet:
    """Project P onto every parameter. Two LPs each, after trimming outliers."""
    theta_l1, viol, worst, note = feasibility(cons, box, outlier_frac)
    notes = (note,) if note else ()
    n_viol = int(viol.sum())
    frac = n_viol / max(len(cons), 1)
    if not np.isfinite(worst) or frac > outlier_frac:
        return IdentifiedSet(
            lo=np.full(N_PARAMS, np.nan), hi=np.full(N_PARAMS, np.nan),
            centre=theta_l1, radius=0.0,
            identifiability=np.zeros(N_PARAMS), n_constraints=len(cons),
            empty=True, min_violation=worst, at_box_edge=np.zeros(N_PARAMS, bool),
            notes=notes + (
                f"{n_viol} of {len(cons)} half-spaces ({100 * frac:.1f}%) cannot "
                f"be satisfied together, over the {100 * outlier_frac:.0f}% budget; "
                f"worst {worst / 1e3:.1f} kJ. Check the regulation variant and the "
                f"aero-mode labelling before the data.",))

    kept = Constraints(
        A=cons.A[~viol], b=cons.b[~viol], n_intervals=cons.n_intervals,
        n_upper=int((~viol)[:cons.n_upper].sum()),
        n_lower=int((~viol)[cons.n_upper:].sum()),
        n_dropped=cons.n_dropped + n_viol,
        drop_reasons={**cons.drop_reasons, "inconsistent": n_viol})

    A, b, scale = _scaled(kept.A, kept.b)
    bnds = _bounds(box, scale)
    lo = np.empty(N_PARAMS)
    hi = np.empty(N_PARAMS)
    for i in range(N_PARAMS):
        lo[i], ok_lo = _project(A, b, bnds, i, +1.0)
        hi[i], ok_hi = _project(A, b, bnds, i, -1.0)
        lo[i] /= scale[i]
        hi[i] /= scale[i]
        if not (ok_lo and ok_hi):
            notes += (f"projection LP failed for {PARAM_NAMES[i]}",)
    centre, radius = chebyshev_centre(kept, box)

    ident = np.clip(1.0 - (hi - lo) / (IDENT_FULL_WIDTH * IDENT_REF), 0.0, 1.0)
    tol = 1e-6 * np.maximum(np.abs(box[:, 1] - box[:, 0]), 1.0)
    at_edge = (lo <= box[:, 0] + tol) | (hi >= box[:, 1] - tol)
    if n_viol:
        notes += (f"trimmed {n_viol} inconsistent half-spaces "
                  f"({100 * frac:.2f}%), worst {worst / 1e3:.1f} kJ",)
    return IdentifiedSet(
        lo=lo, hi=hi, centre=centre, radius=radius, identifiability=ident,
        n_constraints=len(kept), empty=False, min_violation=0.0,
        at_box_edge=at_edge, notes=notes)


def intersect(*cons: Constraints) -> Constraints:
    """Running intersection. This is the streaming update: a new lap is a new
    block of half-spaces, and nothing already accepted is revisited."""
    A = np.vstack([c.A for c in cons])
    b = np.concatenate([c.b for c in cons])
    reasons = {}
    for c in cons:
        for k, v in c.drop_reasons.items():
            reasons[k] = reasons.get(k, 0) + v
    return Constraints(
        A=A, b=b, n_intervals=sum(c.n_intervals for c in cons),
        n_upper=sum(c.n_upper for c in cons), n_lower=sum(c.n_lower for c in cons),
        n_dropped=sum(c.n_dropped for c in cons), drop_reasons=reasons)


def prune_redundant(cons: Constraints, theta: np.ndarray, keep: int = 4000
                    ) -> Constraints:
    """Keep the most binding constraints and drop the rest.

    A race is ~1.9 million half-spaces and almost all of them are slack by
    hundreds of kJ: a car trundling out of a hairpin constrains nothing. Sorting
    by slack at a known-interior point and keeping the tightest `keep` bounds
    the memory a streaming implementation needs, which is the difference between
    a live estimator and a batch one.

    Pruning by slack at an interior point can only *widen* the reported set --
    it never invents information -- so it cannot cause a false refusal.
    """
    slack = cons.b - cons.A @ theta
    order = np.argsort(slack)[:keep]
    order.sort()
    up = order < cons.n_upper
    return Constraints(
        A=cons.A[order], b=cons.b[order], n_intervals=cons.n_intervals,
        n_upper=int(up.sum()), n_lower=int((~up).sum()),
        n_dropped=cons.n_dropped, drop_reasons=cons.drop_reasons)
