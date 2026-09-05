"""Aero mode and powertrain regime as a hidden Markov model.

Two discrete latents drive the bounds in `balance.py`, and getting either wrong
does not degrade the estimate gracefully:

  mu in {X, Z}   the aero state. X is low drag, Z is high downforce. Mislabel a
                 straight as Z and its drag constraint lands on the wrong
                 parameter, which is why `build_window_constraints` splits the
                 drag sum across both columns rather than guessing.

  r in {deploy, ice_only, super_clip, lift_coast, brake_harvest}
                 the powertrain regime. It decides which side of the P_K bound
                 is active: a super-clipping car is recovering at full throttle,
                 which a braking-only harvest model cannot represent at all.

Both are sequences with strong persistence -- a car does not alternate aero
modes every 240 ms -- so the right model is a chain, not a per-sample
classifier. Transitions carry the track: X is permitted only where the circuit
designates it, and brakes force Z.

Where a hard rule exists it is used as a rule, not as evidence. Brakes on means
harvesting or braking, full stop; the HMM is for the cases the feed leaves
ambiguous, which is most of them.

PELT change-point detection is deliberately absent. The plan lists it as an
optional confirmation of segment boundaries, and the Viterbi path already
produces boundaries; adding a second segmenter would mean reconciling two
answers with no way to tell which is right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regs import COAST_THROTTLE_FRAC, taper

X, Z = 0, 1
AERO_NAMES = ("X", "Z")

DEPLOY, ICE_ONLY, SUPER_CLIP, LIFT_COAST, BRAKE_HARVEST = range(5)
REGIME_NAMES = ("deploy", "ice_only", "super_clip", "lift_coast", "brake_harvest")

# Persistence, as a probability of staying put per sample. A 240 ms feed sample
# against aero-mode segments that last seconds and regime segments that last
# roughly a second: 0.98 and 0.90 are those durations expressed as a geometric
# hold, not tuned numbers.
AERO_STAY = 0.98
REGIME_STAY = 0.90
LOG0 = -1.0e6   # a forbidden transition, finite so the LP-free arithmetic stays
                # well behaved and an all-forbidden column is still recoverable


def _log(x):
    return np.log(np.maximum(np.asarray(x, float), 1e-300))


def viterbi(log_emit: np.ndarray, log_trans: np.ndarray,
            log_start: np.ndarray | None = None) -> np.ndarray:
    """MAP state path. log_emit is (n, k), log_trans is (k, k)."""
    n, k = log_emit.shape
    if log_start is None:
        log_start = np.full(k, -np.log(k))
    delta = log_start + log_emit[0]
    psi = np.zeros((n, k), dtype=np.int8)
    for i in range(1, n):
        cand = delta[:, None] + log_trans          # (from, to)
        psi[i] = np.argmax(cand, axis=0)
        delta = cand[psi[i], np.arange(k)] + log_emit[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = psi[i, path[i]]
    return path


def forward_backward(log_emit: np.ndarray, log_trans: np.ndarray,
                     log_start: np.ndarray | None = None) -> np.ndarray:
    """Posterior over states per sample, (n, k).

    Batch only. Live operation uses `forward_only`, which is the same recursion
    without the backward pass and is what the measured 0.5 s latency refers to.
    """
    n, k = log_emit.shape
    if log_start is None:
        log_start = np.full(k, -np.log(k))
    fwd = np.empty((n, k))
    fwd[0] = log_start + log_emit[0]
    for i in range(1, n):
        fwd[i] = log_emit[i] + _logsumexp(fwd[i - 1][:, None] + log_trans, axis=0)
    bwd = np.zeros((n, k))
    for i in range(n - 2, -1, -1):
        bwd[i] = _logsumexp(log_trans + (log_emit[i + 1] + bwd[i + 1])[None, :], axis=1)
    post = fwd + bwd
    return np.exp(post - _logsumexp(post, axis=1, keepdims=True))


def forward_only(log_emit: np.ndarray, log_trans: np.ndarray,
                 log_start: np.ndarray | None = None) -> np.ndarray:
    """Causal filtered posterior: uses samples up to and including each one."""
    n, k = log_emit.shape
    if log_start is None:
        log_start = np.full(k, -np.log(k))
    out = np.empty((n, k))
    f = log_start + log_emit[0]
    out[0] = f - _logsumexp(f)
    for i in range(1, n):
        f = log_emit[i] + _logsumexp(f[:, None] + log_trans, axis=0)
        out[i] = f - _logsumexp(f)
    return np.exp(out)


def _logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


# ------------------------------------------------------------------ aero mode
def aero_log_emission(v, brake, throttle, x_allowed) -> np.ndarray:
    """Evidence for X versus Z, per sample.

    Three signals, all public:
      - the circuit permits X only in designated segments -- a hard gate;
      - brakes on means Z, because a car does not brake in its low-drag state;
      - off throttle means the driver has given the mode up.

    Speed is deliberately *not* used. The obvious feature is "this sample is
    fast, so it is probably X", but `taper()` is flat at 1 below 290 km/h, so on
    a circuit where 95% of running happens below that the feature is identically
    zero and its only effect is a constant tilt towards Z -- which is what it
    did: every sample of a full-throttle accelerating fixture came back Z.

    The genuinely informative signal is the drag-implied residual, since X mode
    shows up as a lower one. That needs theta, which needs the mode, so it
    belongs in an outer iteration (step 5) rather than here.
    """
    v = np.asarray(v, float)
    brake = np.asarray(brake, float) > 0.5
    throttle = np.asarray(throttle, float)
    x_allowed = np.asarray(x_allowed, bool)
    on_power = throttle >= COAST_THROTTLE_FRAC
    # P(X | observables), before the chain applies persistence
    p_x = np.where(on_power, 0.90, 0.30)
    p_x = np.where(x_allowed & ~brake, p_x, 0.0)
    return np.column_stack([_log(p_x), _log(1.0 - p_x)])


def aero_log_transition(x_stay: float = AERO_STAY) -> np.ndarray:
    p = np.array([[x_stay, 1 - x_stay], [1 - x_stay, x_stay]])
    return _log(p)


def infer_aero(v, brake, throttle, x_allowed, causal: bool = False):
    """(path, posterior) over {X, Z}."""
    e = aero_log_emission(v, brake, throttle, x_allowed)
    tr = aero_log_transition()
    post = (forward_only if causal else forward_backward)(e, tr)
    return viterbi(e, tr), post


# --------------------------------------------------------------------- regime
def regime_log_emission(v, brake, throttle, accel, in_zone) -> np.ndarray:
    """Evidence for the five powertrain regimes.

    The discriminating signature is super-clip, and it is the reason this
    module exists: full throttle while *losing* speed on a straight. Nothing
    else in the feed looks like that, and a braking-only harvest model has no
    state for it -- which is how a top-speed drop at full throttle gets
    attributed to high drag instead of to recovery.
    """
    v = np.asarray(v, float)
    brake = np.asarray(brake, float) > 0.5
    thr = np.asarray(throttle, float)
    accel = np.asarray(accel, float)
    in_zone = np.asarray(in_zone, bool)
    n = len(v)
    e = np.zeros((n, 5))

    on_power = thr >= COAST_THROTTLE_FRAC
    room = taper(v)          # 1 where the regulation still allows full deployment

    # brakes on is a rule, not evidence
    e[brake, :] = LOG0
    e[brake, BRAKE_HARVEST] = 0.0
    e[~brake, BRAKE_HARVEST] = LOG0

    free = ~brake
    # off throttle and not braking: lift and coast
    e[free & ~on_power, DEPLOY] = LOG0
    e[free & ~on_power, ICE_ONLY] = LOG0
    e[free & ~on_power, SUPER_CLIP] = LOG0
    e[free & on_power, LIFT_COAST] = LOG0

    # on power: deploy needs headroom under the taper and, post-Miami, a zone
    e[free & on_power, DEPLOY] += _log(0.05 + 0.95 * room[free & on_power])
    e[free & on_power, DEPLOY] += np.where(in_zone[free & on_power], 0.0, _log(0.4))
    # super-clip: on power and decelerating
    slowing = np.clip(-accel, 0.0, None)
    sc = free & on_power
    e[sc, SUPER_CLIP] += _log(0.02 + 0.98 * np.tanh(slowing[sc] / 1.5))
    e[sc, DEPLOY] += _log(0.02 + 0.98 * np.tanh(np.clip(accel[sc], 0, None) / 1.5 + 0.2))
    e[sc, ICE_ONLY] += _log(0.5)
    return e


def regime_log_transition(stay: float = REGIME_STAY) -> np.ndarray:
    k = 5
    p = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(p, stay)
    return _log(p)


def infer_regime(v, brake, throttle, accel, in_zone, causal: bool = False):
    e = regime_log_emission(v, brake, throttle, accel, in_zone)
    tr = regime_log_transition()
    post = (forward_only if causal else forward_backward)(e, tr)
    return viterbi(e, tr), post


@dataclass(frozen=True)
class ModePath:
    aero: np.ndarray          # 0 = X, 1 = Z
    aero_post: np.ndarray     # (n, 2)
    regime: np.ndarray        # 0..4
    regime_post: np.ndarray   # (n, 5)

    @property
    def is_x(self) -> np.ndarray:
        return self.aero == X

    def ambiguous(self, p: float = 0.8) -> np.ndarray:
        """Samples the HMM will not commit on. They are not dropped: they go
        into `build_window_constraints` as windows spanning both drag columns,
        which is a wide constraint rather than a wrong one."""
        return self.aero_post.max(axis=1) < p


def infer(v, brake, throttle, accel, in_zone, x_allowed,
          causal: bool = False) -> ModePath:
    a, ap = infer_aero(v, brake, throttle, x_allowed, causal)
    r, rp = infer_regime(v, brake, throttle, accel, in_zone, causal)
    return ModePath(aero=a, aero_post=ap, regime=r, regime_post=rp)
