# The mathematical model as built

Branch `math-model`. Steps 1-8 of the proposed build order, one commit each,
additive: the Stage 1 estimator and its golden file are untouched and still
pass, so nothing here has yet displaced a working number.

The design principle held throughout: physics and the regulation give hard
bounds, and probability is used only inside them. Every learned quantity is
traceable to a constraint it cannot violate.

---

## What is where

| Module | Step | What it owns |
|---|---|---|
| `regs.py` | 1 | The rulebook as bounds. Pre/post-Miami variants by date. |
| `balance.py` | 1 | Interval energy balance -> linear constraints on theta. |
| `setmem.py` | 2 | The feasible set as a polytope; projections by LP. |
| `modes.py` | 3 | Aero mode and powertrain regime as an HMM. |
| `pipeline.py` | 4 | Composes the above. No numerics of its own. |
| `rbpf.py` | 5 | Store, reserve and theta by particle filter. |
| `strategy.py` | 6 | The PMP prior: three parameters per lap. |
| `pooling.py` | 7 | Partial pooling across the field, closed form. |
| `qmdp.py` | 8 | Deciding under the belief, with sensitivity. |

`theta = (CdA_X, CdA_Z, F_rr, dm)`: the two aero states, rolling resistance as
a force at a reference mass, and dry mass minus the published weight.

---

## Why integrate instead of differentiate

Speed arrives quantised to 1 km/h at about 4 Hz. A single difference is 0.28 m/s
over 0.24 s, which is 1.16 m/s^2 of pure quantisation noise -- 64 kW on 790 kg
at 70 m/s, against a 350 kW signal.

Over an interval the work-energy theorem is an identity, so `dE_kin` is a
difference of squares of *measured* speeds and the quantisation enters linearly.
The resulting error is 6.3 kJ at 70 m/s, fixed by the quantiser. What is not
fixed is the signal: the energy crossing the interval grows with `dt`.

| feed rate | energy per interval | noise / signal |
|---|---|---|
| 200 Hz | 2.3 kJ | 274% |
| 100 Hz | 77 kJ | 8% |
| 3.7 Hz | 215 kJ | 3% |

**The formulation improves as the feed slows.** That is the exact opposite of a
derivative, and it matters because real 2026 telemetry arrives at 4.17 Hz.

---

## The three results that changed the plan

### 1. Drag area has no lower bound from a speed trace

Recovery can always account for a deceleration. A 3 s coast shedding 5 m/s
dissipates 0.267 MJ; the rules permit 0.75 MJ of harvesting over the same
window, and post-Miami super-clipping raises that to 1.05 MJ.

The integral caps do not rescue it. Over a closed lap the kinetic term vanishes,
so the balance reads `E_drag + E_rr = eta(E_ice + E_K)` with `E_K` in
[-4, +4] MJ. That bounds drag from above and is trivially satisfied at `CdA = 0`
below. Adding the per-lap harvest cap and the store bound moved the projection
by nothing.

So `CdA_X` reports `[0.300, 0.744]` on the Stage 1 fixture -- upper bound from
the data, lower bound sitting on its prior edge and flagged in `at_box_edge`.
Step 5 was expected to close it, on the argument that a bounded store cannot
absorb unlimited recovery. It does not: the floor penalty pushes the other way,
because low drag means less deployment, a fuller store and fewer floor
violations. Before theta was constrained to the polytope this drove `CdA_X` to
0.335 against a true 0.660.

**The polytope and the filter are not separable the way steps 2-5 assume.**

### 2. Interval intersection is robust but statistically inefficient

An intersection keeps the single *tightest* bound, so noise never averages down.
With slack honestly set at 5 sigma, one interval carries 31 kJ of slack against
36.7 kJ of drag signal, and the identified range for `CdA_X` came out as the
entire prior box.

Window aggregation fixes it because the kinetic and potential terms telescope:
`sum 1/2 m (v_{k+1}^2 - v_k^2)` is `1/2 m (v_end^2 - v_start^2)`. A window's
measurement error is the same 6.3 kJ as a single interval's, while drag, rolling
and the energy budget all grow linearly with it.

| window | CdA_X upper bound |
|---|---|
| 0.6 s | 1.057 m^2 |
| 2 s | 0.853 |
| 4 s | 0.771 |
| **8 s** | **0.744** |
| 15 s | 1.448 |
| 89 s (a lap) | 1.606 |

Past about 8 s whole-lap windows average the informative high-speed running
away. Truth is 0.660.

### 3. Hard set-membership is destroyed by one bad half-space

At 3.7 Hz the true theta sat outside 1.6% of the per-interval lower bounds --
all at braking onsets -- and the set reported INFEASIBLE with a 228.2 kJ minimum
violation on data whose parameters are known exactly.

An L1 relaxation `min sum(s) s.t. A theta - b <= s` now names the inconsistent
samples and trims them under a 3% budget; over budget is still an alarm. L1
rather than min-max because L1 violates a few constraints a lot, so `s > 0` is a
short list to go and look at, where a Chebyshev relaxation spreads the blame and
diagnoses nothing.

---

## What works, measured against the simulator's hidden state

| Quantity | Result |
|---|---|
| Constraint containment | 0 upper-bound violations at 3.7 / 20 / 100 Hz |
| Braking classification | exact (it is a rule, not an inference) |
| Aero mode agreement | 93.1%, with >95% of the residual at braking samples |
| Raw store band coverage | 0.72 at 0.46 MJ width |
| Per-lap deployed energy | 5.1% MAPE |
| Policy recovery (v_cut, v_harv, w) | 71.9 / 88.0 / 3.0 against 72 / 88 / 3 |
| Field pooling | 48% mean-absolute-error reduction over 20 cars |
| LP cost | 11-18 ms for a 12-lap race |

The particle count is a **correctness** parameter, not a speed dial: 100
particles collapses (ESS 1 of 100, 39.6% error), 400 works, 1,600 buys nothing.

---

## What does not work

**The reserve is not separately identified, so the deployable-energy band is
unusable.** Inferred buffer 0.97 +/- 0.32 MJ for a driver whose policy holds
0.00. Because `U = max(E - R, 0)` is then exactly zero for every particle, the
band collapses to 0.06 MJ wide and covers the truth 4% of the time.

The store underneath it is fine, so this is the E/R split and not the filter.
The cut-out likelihood selects *consistent pairs* of `(E, R)` rather than
pinning `R`, so `R` sits at its prior mean. Four fixes were tried and each moved
the number without repairing it: an accelerating gate on the detector,
hysteresis on it, 10x wider reserve jitter, and the buffer-release ramp.

Recorded as a strict `xfail` with the evidence rather than a passing test with a
relaxed threshold. **The Stage 1 estimator reaches 0.94 on this same quantity**,
so the capability exists in this repo and the rewrite has lost it. That
comparison is the next thing to chase.

Also short of the plan: pooling was expected to halve the posterior *width*, and
the width falls only to 0.94x while the error halves. The method-of-moments
`tau_team` reads 0.0562 against a true 0.012 on ten noisy teammate pairs.
Overestimating tau makes the pooling too timid, which is the safe direction, but
the claim is not met.

---

## Deviations from the proposed design, and why

**NUTS replaced by closed-form conjugate pooling.** For a Gaussian hierarchy
whose per-car variances are already known -- and step 2 supplies them as
projection widths -- the posterior *is* the shrinkage estimator, exactly. NUTS
would reproduce it with sampler error plus a jax dependency. `needs_mcmc()`
names the four cases that break conjugacy so the choice is revisited on purpose.

**PELT omitted.** The Viterbi path already produces segment boundaries, and a
second segmenter would need reconciling with no way to tell which is right.

**Regulation numbers are spec-sourced, not FIA-sourced.** There is no
machine-readable source. Anything the spec did not pin exactly is named
`ASSUMED_*` with its derivation -- notably `ASSUMED_MIAMI_2026`, the changeover
date, which is the single input most worth checking before quoting a result.

The fuel-flow schedule is self-consistent, which is worth noting as a check
rather than an assumption: `EF(n) = 0.27n + 165 MJ/h` at the 10,500 rpm limit is
833 kW of fuel, and the 400 kW ICE cap implies exactly 0.48 thermal efficiency.
Cap and schedule agree at maximum revs and the schedule binds below it -- 274 kW
at 7,000 rpm.

---

## Bugs found by measuring rather than reading

- Bounds evaluated at the interval midpoint excluded the truth on 3,884 upper
  bounds. The deployment cap falls 19.4 kW per m/s and a car through 338 km/h
  deploys at the cap for its *entry* speed. Each bound is now evaluated at
  whichever endpoint makes it loosest.
- The brake flag was read at the interval's opening sample, keeping the lower
  bound across a braking onset: 371 violations, up to 17.3 kJ, every one at an
  onset.
- The cut-out detector had no accelerating gate, so a corner-limited car
  qualified -- on full throttle, not deploying because it does not need to. 36
  detections a race at a true store of 0.01 MJ median but 2.86 MJ maximum.
- It had no hysteresis either, so it triggered on every dip and saturated its
  per-lap budget every lap.
- `E_var` was computed and never used, making "Rao-Blackwellised" a label rather
  than a method.
- `aggression` was `taper(v_cut)`, which ranks drivers backwards, and
  `1 - taper(v_cut)` scores every realistic cut-off at exactly 0 because taper
  is flat below 290 km/h.
- The pooling hierarchy was one level, which made the error 5% *worse* than not
  pooling. Its first fixture could not have caught it: teammates were generated
  with identical true drag, so complete pooling was correct there.
- The two-step feint fired against a rival believed to hold 0.02 MJ. A car with
  nothing left cannot mount a defence, so there is nothing to provoke.

And one in the simulator itself, verified by the store balance: `sim.step()`
mutates state then records, so `trace.v[k]` is post-step while
`trace.P_mguk[k]` is the power applied during it. `E[k+1] - E[k]` closes to
0.000 J rms against index `k+1`, and 168.9 J rms against `k`.

---

## Not done

Step 9 as scoped: the periodic GP residual over track position, NPE proposals,
and the live port of the kernels.

Four of the seven validation items in section 8 need real telemetry, which needs
`fastf1` and a session download: polytope non-emptiness per real lap, the
pit-stop mass step, the RDD recomputed against the corrected `P_dep_max`, and
the pre/post-Miami split. The other three are done -- teammate consistency,
super-clip recovery, and batch == streaming.
