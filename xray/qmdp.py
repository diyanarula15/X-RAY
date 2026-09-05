"""Decision under a belief: QMDP over the particle cloud.

The rival's deployable energy is not a number, it is the cloud from `rbpf`. The
exact treatment is a POMDP over that belief; QMDP is the standard approximation
and the right one here -- solve the fully-observed DP for each particle's state,
then average the values under the belief and act on the average.

    Q(a) = sum_i  w_i * Q_MDP(a | e_riv = particle_i)

That is Monte Carlo over the reserve posterior and the drag uncertainty in one
pass, and it costs one DP solve per particle bin rather than per particle.

What QMDP gives up is the value of *information*: it assumes the uncertainty
resolves immediately after the current action, so it will never choose an
action in order to learn something. For this problem that is the right trade --
there is no probing action available, since the rival's energy is revealed by
their driving and not by ours -- but a two-step lookahead is included for the
one case where our action does change what we learn, below.

Two additions the research forces.

The defender's joule is worth 1/(m v^3) too. A defender forced to deploy out of
a slow corner is spending at maximum value, so provoking deployment one corner
*before* the real attack is often better than attacking directly. That is a
two-step calculation, not a one-step one.

Manual Override eligibility is a known rule on a known state: within 1.000 s at
the detection point, and the rival's deployment ceiling changes next lap. It is
a state variable, not noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import E_STORE_MAX
from .decision import DecisionModel, ZoneModel, delta_v
from .overtake import p_pass
from .regs import RegSet

# Minimum expected-value gain before a feint is worth showing the rival that an
# attack is coming. Below this the two-step is inside its own noise.
PROVOKE_MIN_GAIN = 0.01


@dataclass(frozen=True)
class BeliefDecision:
    lap: int
    action: str | None          # zone name, or None to hold
    q_values: dict              # action -> expected value under the belief
    q_best: float
    q_hold: float
    p_pass_mean: float
    consensus: float            # fraction of belief mass agreeing with the call
    sensitivity: float          # d(expected value) / d(reserve width), per MJ
    mom_eligible: bool
    provoke: bool               # attack the previous zone first to drain them
    notes: tuple = ()


def _q_attack(zm: ZoneModel, e_own: float, e_riv: float, model: DecisionModel,
              reward: float, v_fail: float) -> float:
    p = p_pass(delta_v(zm, min(e_own, model.attack_cost),
                       min(e_riv, model.attack_cost)), model.gap_s, zm)
    return p * reward + (1.0 - p) * (1.0 - model.fail_cost) * v_fail


def decide(model: DecisionModel, belief_particles: np.ndarray,
           weights: np.ndarray | None, laps_left: int, e_own: float,
           reward: float = 1.0, v_fail: float = 0.0,
           gap_s: float | None = None, regs: RegSet | None = None,
           prev_zone: ZoneModel | None = None) -> BeliefDecision:
    """One decision, averaged over the belief rather than over a point estimate.

    `belief_particles` is the rival's deployable energy, one entry per particle.
    Acting on the mean of that cloud is the thing this replaces: p_pass is
    convex in the energy difference over the relevant range, so the mean of the
    probability is not the probability of the mean, and a point estimate
    systematically misprices a wide belief.
    """
    e_riv = np.asarray(belief_particles, float)
    w = (np.full(len(e_riv), 1.0 / len(e_riv)) if weights is None
         else np.asarray(weights, float) / np.sum(weights))

    q = {}
    per_particle = {}
    for zm in model.zones:
        vals = np.array([_q_attack(zm, e_own, float(er), model, reward, v_fail)
                         for er in e_riv])
        per_particle[zm.name] = vals
        q[zm.name] = float(np.sum(w * vals))
    q_hold = float(v_fail)

    best_name = max(q, key=q.get)
    q_best = q[best_name]
    action = best_name if q_best > q_hold else None

    # How much of the belief agrees? A call that is optimal on average but only
    # for 51% of the cloud is a different animal from one that holds for 95%,
    # and the difference is invisible in the expected value.
    if action is None:
        agree = float(np.sum(w * (np.max(np.array(list(per_particle.values())),
                                         axis=0) <= q_hold)))
    else:
        stacked = np.array([per_particle[k] for k in q])
        chosen = list(q).index(best_name)
        agree = float(np.sum(w * (np.argmax(stacked, axis=0) == chosen)))

    # Sensitivity to the reserve degeneracy, reported with the call. Where this
    # is small the reserve being weakly identified does not matter -- and that
    # is the common case, which is worth saying out loud rather than leaving a
    # known limitation looking fatal.
    spread = float(np.sqrt(np.sum(w * (e_riv - np.sum(w * e_riv)) ** 2)))
    if spread > 1e3:
        shifted = np.clip(e_riv + 0.5e6, 0.0, E_STORE_MAX)
        q_shift = max(float(np.sum(w * np.array(
            [_q_attack(zm, e_own, float(er), model, reward, v_fail)
             for er in shifted]))) for zm in model.zones)
        sensitivity = float((q_shift - q_best) / 0.5)   # per MJ
    else:
        sensitivity = 0.0

    p_mean = float(np.sum(w * np.array([
        p_pass(delta_v(model.zones[list(q).index(best_name)],
                       min(e_own, model.attack_cost),
                       min(float(er), model.attack_cost)), model.gap_s,
               model.zones[list(q).index(best_name)]) for er in e_riv])))

    mom = bool(gap_s is not None and regs is not None and gap_s <= regs.mom_gap_s)
    provoke, notes = False, ()
    if action is not None and prev_zone is not None:
        provoke, why = _worth_provoking(model, prev_zone,
                                        model.zones[list(q).index(best_name)],
                                        e_own, e_riv, w, reward, v_fail)
        if why:
            notes += (why,)
    if mom:
        notes += (f"rival within {regs.mom_gap_s:.3f} s at the detection point: "
                  "their deployment ceiling rises next lap, so waiting costs more "
                  "than the energy arithmetic alone says",)

    return BeliefDecision(
        lap=laps_left, action=action, q_values=q, q_best=q_best, q_hold=q_hold,
        p_pass_mean=p_mean, consensus=agree, sensitivity=sensitivity,
        mom_eligible=mom, provoke=provoke, notes=notes)


def _worth_provoking(model: DecisionModel, prev: ZoneModel, target: ZoneModel,
                     e_own: float, e_riv: np.ndarray, w: np.ndarray,
                     reward: float, v_fail: float):
    """Two-step: feint at `prev` to make them spend, then attack `target`.

    The defender's joule obeys the same 1/(m v^3), so forcing a defence out of
    the slowest corner available is where their energy buys them the most and
    therefore where draining them costs them the most. Worth it only when the
    energy they must spend to hold `prev` exceeds what we spend provoking it.
    """
    # A car with nothing left cannot mount a defence, so there is nothing to
    # provoke -- it simply gets passed. Without this guard the two-step fired on
    # a rival believed to hold 0.02 MJ, for +0.005 expected value: arithmetically
    # consistent (they drop 0.02 MJ, we spend 0.02 MJ, and their loss is worth
    # marginally more) and strategically meaningless.
    e_mean = float(np.mean(e_riv))
    if e_mean < 0.25 * model.attack_cost:
        return False, ()
    # what a defence of `prev` costs them: enough to match our speed advantage
    defend_cost = min(model.attack_cost, e_mean)
    drained = np.clip(e_riv - defend_cost, 0.0, E_STORE_MAX)
    direct = max(float(np.sum(w * np.array(
        [_q_attack(target, e_own, float(er), model, reward, v_fail)
         for er in e_riv]))), 0.0)
    after = float(np.sum(w * np.array(
        [_q_attack(target, e_own - defend_cost, float(er), model, reward, v_fail)
         for er in drained])))
    # and the gain has to be worth the risk of showing them the attack early
    if after > direct + PROVOKE_MIN_GAIN:
        return True, (f"provoking a defence at {prev.name} first is worth "
                      f"{after - direct:+.3f} expected value: their joule is worth "
                      f"more out of a slow corner than ours is")
    return False, ()
