"""Step 3: aero mode and powertrain regime as a chain, not a classifier."""
from __future__ import annotations

import numpy as np

from xray.modes import (AERO_STAY, BRAKE_HARVEST, DEPLOY, ICE_ONLY, LIFT_COAST,
                        SUPER_CLIP, X, Z, forward_backward, forward_only,
                        infer, regime_log_transition, viterbi)
from xray.observe import observe
from xray.sim import LEADER


def _channels(cfg, gt, rate=3.7):
    """The public feed, plus the two channels a real one publishes."""
    obs = observe(gt, LEADER, rate_hz=rate,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=43)
    idx = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)
    reg = gt.cars[LEADER].regime[idx]
    brake = (reg == "brake").astype(float)
    throttle = np.where(reg == "brake", 0.0, 1.0)
    accel = np.gradient(obs.v, obs.t)
    x_allowed = ~np.asarray(gt.track.is_corner(obs.s))
    return obs, reg, brake, throttle, accel, x_allowed


def test_braking_is_a_rule_not_a_guess(cfg, races):
    """Brakes on means harvesting or braking. There is nothing to infer, so the
    HMM must not be able to disagree with the channel."""
    obs, reg, brake, throttle, accel, x_allowed = _channels(cfg, races[42])
    mp = infer(obs.v, brake, throttle, accel, np.ones(len(obs.v), bool), x_allowed)
    assert ((mp.regime == BRAKE_HARVEST) == (brake > 0.5)).all()


def test_aero_mode_tracks_the_circuit_gate(cfg, races):
    """Agreement with the simulator's position-gated aero mode, and the
    disagreement attributed rather than tolerated.

    The simulator switches aero mode on position alone, so it calls a *braking*
    car on a straight X. The regulation model says brakes force Z. Measured
    93.1% agreement, and this asserts the residual is concentrated where the
    two models genuinely differ -- not spread over the lap, which would mean the
    chain is drifting.
    """
    obs, reg, brake, throttle, accel, x_allowed = _channels(cfg, races[42])
    mp = infer(obs.v, brake, throttle, accel, np.ones(len(obs.v), bool), x_allowed)
    truth_x = x_allowed
    agree = mp.is_x == truth_x
    assert agree.mean() > 0.90, f"aero agreement {100 * agree.mean():.1f}%"
    disagreements = ~agree
    on_brakes = brake > 0.5
    explained = (disagreements & on_brakes).sum() / max(disagreements.sum(), 1)
    assert explained > 0.95, (
        f"only {100 * explained:.0f}% of aero disagreements are braking samples; "
        "the rest are the chain drifting, which is a different problem")


def test_x_mode_never_claimed_where_the_circuit_forbids_it(cfg, races):
    """A hard gate must be hard: no amount of persistence may carry X into a
    segment where the regulation does not allow it."""
    obs, reg, brake, throttle, accel, x_allowed = _channels(cfg, races[42])
    mp = infer(obs.v, brake, throttle, accel, np.ones(len(obs.v), bool), x_allowed)
    assert not mp.is_x[~x_allowed].any()


def test_super_clip_is_detected_and_is_the_reason_this_exists(cfg):
    """Full throttle while losing speed on a straight.

    Nothing else in the feed looks like that, and a braking-only harvest model
    has no state for it -- which is how a top-speed drop at full throttle gets
    attributed to high drag instead of to recovery. Synthetic, because the
    Stage 1 simulator predates the rule and never does it (4 samples out of
    3,972 on seed 42, all noise).
    """
    n = 60
    v = np.concatenate([np.linspace(70, 88, 30), np.linspace(88, 80, 30)])
    brake = np.zeros(n)
    throttle = np.ones(n)
    accel = np.gradient(v, np.arange(n) * 0.24)
    mp = infer(v, brake, throttle, accel, np.ones(n, bool), np.ones(n, bool))
    accelerating, slowing = mp.regime[:28], mp.regime[32:]
    assert (accelerating != SUPER_CLIP).all(), "called super-clip while accelerating"
    assert (slowing == SUPER_CLIP).mean() > 0.8, (
        f"full throttle while decelerating was classified "
        f"{np.bincount(slowing, minlength=5)} instead of super-clip")


def test_lift_and_coast_is_distinct_from_braking(cfg):
    """Off throttle, off the brakes. The coast-down channel depends on telling
    these apart: coasting bounds drag, braking says nothing about it."""
    n = 40
    v = np.linspace(80, 70, n)
    accel = np.gradient(v, np.arange(n) * 0.24)
    mp = infer(v, np.zeros(n), np.zeros(n), accel, np.ones(n, bool), np.ones(n, bool))
    assert (mp.regime == LIFT_COAST).mean() > 0.9


def test_causal_inference_matches_batch_on_the_rules(cfg, races):
    """Forward-only is what live operation runs. It cannot match the smoothed
    posterior everywhere -- that is the point of a backward pass -- but it must
    agree wherever a hard rule decides the answer, and it must not be wildly
    worse elsewhere."""
    obs, reg, brake, throttle, accel, x_allowed = _channels(cfg, races[42])
    args = (obs.v, brake, throttle, accel, np.ones(len(obs.v), bool), x_allowed)
    batch = infer(*args, causal=False)
    live = infer(*args, causal=True)
    assert ((live.regime == BRAKE_HARVEST) == (batch.regime == BRAKE_HARVEST)).all()
    assert not live.is_x[~x_allowed].any()
    agreement = (live.aero == batch.aero).mean()
    assert agreement > 0.95, f"causal aero path agrees only {100 * agreement:.1f}%"


def test_posteriors_are_distributions(cfg, races):
    obs, reg, brake, throttle, accel, x_allowed = _channels(cfg, races[42])
    mp = infer(obs.v, brake, throttle, accel, np.ones(len(obs.v), bool), x_allowed)
    for post in (mp.aero_post, mp.regime_post):
        assert np.isfinite(post).all()
        np.testing.assert_allclose(post.sum(axis=1), 1.0, rtol=1e-9)
        assert (post >= 0).all()


def test_viterbi_beats_per_sample_argmax_on_persistence(cfg):
    """A chain exists to stop the label flickering. This is the property that
    justifies it over a per-sample classifier: given evidence that alternates,
    the MAP path must not."""
    n = 200
    rng = np.random.default_rng(0)
    # true state is one long block; evidence is noisy and flips constantly
    truth = np.zeros(n, dtype=int)
    truth[n // 2:] = 1
    noisy = np.where(rng.random(n) < 0.30, 1 - truth, truth)
    log_emit = np.column_stack([np.where(noisy == 0, np.log(0.7), np.log(0.3)),
                                np.where(noisy == 0, np.log(0.3), np.log(0.7))])
    trans = np.log(np.array([[0.99, 0.01], [0.01, 0.99]]))
    path = viterbi(log_emit, trans)
    flips_raw = int(np.abs(np.diff(noisy)).sum())
    flips_path = int(np.abs(np.diff(path)).sum())
    assert flips_path < flips_raw / 10, (
        f"Viterbi path flips {flips_path} times against {flips_raw} in the evidence")
    assert (path == truth).mean() > 0.95


def test_smoothing_only_ever_adds_confidence(cfg):
    """Forward-only is what live operation runs; forward-backward is the batch
    luxury. On unambiguous evidence the two must agree in the interior, and the
    smoothed pass must be at least as confident everywhere.

    They do *not* agree at the first sample, and that is the whole value of the
    backward pass rather than a defect: forward-only has seen one observation
    there and reports 0.990, while the smoother uses the other 49 and reports
    0.9997. The gap closes within two samples.
    """
    n = 50
    log_emit = np.column_stack([np.full(n, np.log(0.99)), np.full(n, np.log(0.01))])
    trans = regime_log_transition()[:2, :2]
    fb = forward_backward(log_emit, trans)
    fo = forward_only(log_emit, trans)
    assert fb[0, 0] > fo[0, 0], "the backward pass added nothing at the boundary"
    np.testing.assert_allclose(fb[2:, 0], fo[2:, 0], atol=1e-3)
    assert (fb[:, 0] >= fo[:, 0] - 1e-12).all(), (
        "smoothing reduced confidence on evidence that never conflicts")
