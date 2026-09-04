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
| `deadband.py` | 14 | v_cut and v_harv by changepoint, before the filter. |
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

### 1. Drag area is bounded below only by a claim about ICE output

A speed trace alone gives no floor: with `P_ice >= 0`, "engine at idle, motor
harvesting" explains every straight and `CdA = 0` is feasible. A 3 s coast
shedding 5 m/s dissipates 0.267 MJ where the rules permit 0.75 MJ of recovery,
1.05 MJ post-Miami. Over a closed lap the kinetic term vanishes, so the balance
bounds drag above and is trivially satisfied at zero below.

The floor has to come from an assumption about the engine, and there are two.

**Local** -- `P_ice >= (1 - delta) P_ice_max` at full throttle. Measured floors
on `CdA_X`, true value 0.660:

| delta | pre-Miami | post-Miami |
|---|---|---|
| 0.00 | 0.134 | prior edge |
| 0.05 | 0.095 | prior edge |
| 0.10 | 0.057 | prior edge |
| 0.20 | prior edge | prior edge |

The post-Miami column is the result: `400(1-delta) - 350` leaves 10 kW at
`delta = 0.10`, so **super-clipping made drag area lower-unidentifiable from
power bounds alone.** That is a fact about the April change, not the estimator.

**Global -- fuel closure**, and this is the one that works. Over the stint the
ICE does a known amount of work with nowhere to go but drag, rolling
resistance, the friction brakes and a store that cannot absorb more than 4 MJ
net. `CdA_X >= 0.264` with the HMM's mode labels, 0.396 with the circuit's aero
gate, independent of `delta` and of the harvest rule. The set is now two-sided:
**[0.264, 0.744] bracketing 0.660.**

Two inputs each killed this constraint on their own before being fixed: the
fuel-mass uncertainty has to be fractional (a race-level +/-5 kg figure is 60%
of a 12-lap stint and took `E_ice_min` from 296 MJ to 127 MJ), and the friction
bound has to be restricted to intervals where the brakes are actually on
(counting coast-downs inflated it from 100 MJ to 177 MJ).

The prior box was also masking the answer. With the lower edge at 0.30 the
0.264 floor was invisible. The drag edges are now 0.05, deliberately below
anything physical, because a prior edge should only be reached when the data
genuinely has no opinion.

**The step-5 low-drag bias is not fixed by the floor.** With the set two-sided
the filter still sits at 0.277. Sweeping the floor penalty is decisive:

| FLOOR_PENALTY | CdA_X posterior | store coverage | MAPE | ESS |
|---|---|---|---|---|
| 0.0 | 0.505 | 0.32 | 344.7% | 400 |
| 0.5 | 0.340 | 0.38 | 67.7% | 270 |
| 2.0 | 0.382 | 0.52 | 16.4% | 43 |
| 6.0 | 0.277 | 0.69 | 10.9% | 12 |

At zero the posterior is exactly the uniform draw mean and the energy estimate
is useless. **The floor penalty is the only informative likelihood term**, so it
does double duty: it is what makes the energy estimate work at all, and it
biases drag down and collapses the ensemble while doing it. The missing term is
store closure over the race -- deployed must equal harvested to within one
store -- which constrains the deployment level without punishing particles.
That is the fix and it is not yet implemented.

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
| Deployable-energy band coverage | 0.67 / 0.72 / 0.80 on seeds 42 / 7 / 13 |
| Raw store bracket coverage | 0.74 at 3.50 MJ wide (a bracket, not an estimate) |
| Per-lap deployed energy | 9.9% / 15.2% / 14.0% MAPE |
| Drag area CdA_X | 0.597 / 0.571 / 0.494 against a true 0.660 |
| Policy recovery (v_cut, v_harv, w) | 71.9 / 88.0 / 3.0 against 72 / 88 / 3 |
| Field pooling | 48% mean-absolute-error reduction over 20 cars |
| LP cost | 11-18 ms for a 12-lap race |

Particle count, re-measured a third time (invariant 7 exists because this table
keeps moving). Deployable coverage now has a real knee -- 0.42 / 0.73 / 0.67 /
0.70 at 100 / 200 / 400 / 800 -- where every earlier version read 0.57-0.63 flat
and the discriminator did not discriminate. ESS also stopped being a collapse
indicator: it sits at 0.91-0.94 of the count everywhere, against 0.17-0.18
before, because there is now one well-scaled likelihood term instead of several
fighting over the same particles.

---

## Deployable energy: fixed, and how

The band now covers **0.62 at 0.17 MJ width**, against 0.04 before. Three
changes did it, and their ablations say which mattered.

**Inside the polytope the trace is uninformative by construction.** Every theta
in P explains the data exactly -- that is what set membership *means* -- so
something has to decide where inside the deployment band P_K sits. Previously
nothing did (a uniform position), which left the store-floor penalty as the only
selection pressure. That penalty was simultaneously the only thing making the
energy estimate work and the thing biasing drag low, on one knob.

| variant | deployable coverage | per-lap MAPE |
|---|---|---|
| policy prior + box rejection + solved closure | **0.62** | 12.9% |
| uniform position in the band | 0.14 | 43.6% |
| store box as a soft penalty, not a rejection | 0.44 | 33.6% |
| soft closure weight removed | 0.64 | 13.0% |

So the strategy prior is what makes it work, the box rejection matters, and the
soft closure *weight* is redundant -- because closure is **solved**, not
weighted. The prior gives the shape of P_K but its amplitude saturates at the
regulation ceiling, so at face value it pins deployment to the top of the band:
3.82 MJ/lap against a true 2.41, net flow -1.71 MJ/lap against -0.25, and a
store drifting 20 MJ out of a 4 MJ box. Bisecting an amplitude lambda per lap so
the lap's flows close brings deployment to 2.51 against 2.41.

**No cut-out detector.** The store is known up to a constant, E_k = c + F_k, so
if the driver has touched the reserve at least once then E_min = R and
deployable energy is D_k = F_k - min_{j<=k} F_j exactly, with c and R both
cancelling. Before the first touch it is a lower bound. The detector is gone,
and with it the whole class of failure it caused -- its false positives at a
nearly-full store were what inferred a 0.97 MJ buffer for a driver holding zero.

One fixture bug found on the way, and it had been poisoning everything: the
Stage 1 simulator harvests at `P_MGUK_MAX` = 350 kW, which is the *post*-Miami
super-clip cap, but the tests ran it against pre-Miami's 250 kW. Reconstructed
harvest came out at 0.71x the truth, which made per-lap net flow -2.22 MJ
against a true -0.25 and drifted the store -26.6 MJ over twelve laps. No
particle could satisfy store closure, so the running-minimum deployable
collapsed to zero width -- a correct formulation reporting nothing because the
regulation variant underneath it was wrong.

Raw-store coverage fell 0.69 -> 0.50 in the process, and that is the right
trade: it is not the headline, and the ablation shows why chasing it is a trap.
Turning the box rejection off takes raw-store coverage from 0.50 to 0.80 while
tripling per-lap energy error -- a wider band covers more truth without being
more informative.

## Drag: the diagnostic found a bug, the likelihood found signal

Invariant 9 prescribes a diagnostic before touching any prior: histogram the
survivors against the uniform draw. It found the real problem immediately.

**Sampling an initial store was rejecting, not discriminating.** Survivors
collapsed to zero above CdA_X = 0.444 -- five of eight bins empty, the true
0.660 among them -- and the posterior of 0.303 looked like inference. The cause
is that `E_k = c + F_k` with `c` unidentified, so rejecting on a sampled `c`'s
walk conflates "wrong theta" with "wrong c". The box's only c-free statement
about theta is that a feasible `c` exists at all: `max(F) - min(F) <= 4 MJ`.

**The dead band is where drag is actually identified, and the statistic is
exact.** Between the deployment cut-off v_cut and the super-clip onset v_harv
the policy says P_K = 0, so the balance there is pure ICE against drag and the
residual is `delta_CdA * 0.5 rho v^3` with no other regressor -- Stage 1's
high-speed window relocated to a speed that exists. Swept against the *true*
dead band on seed 42 (446 samples, 283-347 km/h) the rms residual minimises at
**CdA_X = 0.66, exactly the truth**, with the mean residual crossing zero there
(+0.4 kW).

That the simulator has a speed-separable dead band at all is worth recording:
the fraction of full-throttle brakes-off samples with true P_K = 0 rises
monotonically from 0.00 below 279 km/h to 0.83 at 336-355 km/h.

**v_cut is not a particle dimension, and inferring it jointly was the whole
problem.** Every weighting scheme tried either let particles escape the test or
over-constrained:

| scheme | CdA_X | what happened |
|---|---|---|
| sum of squared residuals | 0.336 | an empty dead band costs nothing, so escaping is free |
| mean, with a population-mean fallback | 0.325 | escaping still cheaper than being tested and wrong |
| mean, untestable particles rejected | **0.407** | v_cut posterior 243 km/h against a true 300-330 |

At the middle setting the diagnostic was unambiguous: v_cut collapsed to 98 m/s
(353 km/h, above the car's top speed) where the dead band is empty, and
`corr(CdA, log weight) = -0.00` -- the likelihood was exerting no pressure on
drag at all while appearing wired in.

### Two-stage v_cut: the fix, and the two bugs it exposed

v_cut is now estimated *before* the filter, in `xray/deadband.py`, and handed in
as an input. On full-throttle brakes-off intervals the model is

    y(v) = 0.5 rho CdA v^3 + F_rr v m/M_ref - A taper(v) 1[v < v_cut] + H 1[v > v_harv]

with every coefficient non-negative, scanned over v_cut with a BIC penalty so an
empty band cannot win. Measured: **236 / 239 / 214 km/h** on seeds 42 / 7 / 13,
BIC gains of 170.2 / 156.0 / 159.9 over the no-changepoint model. The v_harv
scan **declines on all three**, correctly: `vehicle.step` harvests on the brakes
only, so the Stage 1 simulator has no off-throttle super-clipping to find, and
guessing one would invent the number the fit exists to measure.

Two constraints in that fit are load-bearing and both were found by their
absence, on seed 42 against a true 0.660:

| fit | CdA_X |
|---|---|
| drag only, no intercept | 0.613 |
| drag + free intercept | 0.842 (intercept -77 kW) |
| drag + free roll + free intercept | 0.374 (F_rr 4181 N, 45x physical) |

Over 210-355 km/h a v^3 column and a constant are nearly collinear, so a free
intercept buys fit by pushing drag up; and with the *sign* of rolling resistance
free the fit simply uses it as the intercept it was denied. Hence non-negative
least squares. Inside the band the one-parameter estimate is then **0.632**
against 0.660, stable at 0.629-0.665 as the band narrows from 234 to 324 km/h;
the residual 3-5% is contamination that can only bias drag down, because this
simulator's policy is positional rather than a speed threshold.

Wiring that into the filter moved CdA_X from 0.407 to 0.597 -- but only after
two bugs the change forced into the open.

**The deployment floor carried no measurement slack.** `max(0, e_wheel -
e_ice_max)` was taken per interval with no 5-sigma term, so every positive noise
excursion accumulated and none cancelled. At the *true* theta on seed 42 the
summed floor claimed 4.2-4.4 MJ of deployment a lap against a true 2.15, while
the lap-aggregate floor `max(0, sum x)` was 0.00 -- so the floor was not a bound
on the lap at all. That forced the closure solve to deploy twice the truth,
drove range(F) to **28.7 MJ against a 4 MJ store** (true swing 3.00 MJ), and made
the store box reject the true parameters. Every box-rejection count reported
before the fix was measuring the bug, including the ablation that made the box
look load-bearing (coverage 0.62 -> 0.44) -- it was throwing away the particles
nearest the truth. With the slack in place range(F) at the truth is 2.8 MJ, the
box rejects 37 of 1.6 million particle-samples, and its ablation changes nothing.
This is invariant 3 in the one place it had not been applied, and invariant 9's
own precondition -- *compute range(F) at true theta before trusting a binding
box* -- is what caught it.

**Resampling reindexed theta but not the flow bands.** `e_wheel`, `d_lo`, `d_hi`
and `harvest` were precomputed once for the whole trace from the initial draw
and never reindexed by the resampling permutation, so from the first resample
onward each particle's bands belonged to a different particle's theta -- the
dead-band residual was scored against another particle's drag. The symptom was
precise and misleading: the per-lap weighted CdA_X was **0.60, correct**, and the
reported posterior was **0.42**, because the correct answer was shuffled away
immediately. The bands are now recomputed per lap from the current theta, which
also removes the module's largest allocation and makes theta rejuvenation
possible at all.

Resample-move jitter on theta, once possible, turns out not to matter much:
0.00 / 0.02 / 0.05 / 0.10 of each projection's width give CdA_X 0.583 / 0.589 /
0.597 / 0.585. It is kept at 0.05 because a cloud that can only shrink cannot
recover a direction the draw under-samples, but what was blamed on a collapsed
cloud was the desync above.

**Store closure had to stop being exact.** With the flows corrected, forcing
per-lap net flow to zero left range(F) at 1.37 MJ against a true 3.00 and
deployable coverage at **0.01**: lap 0 genuinely dumps 5.04 MJ out of a full
store, and a store that may never drain has no deployable energy to report. Each
particle now carries a per-lap drift nuisance bounded by `E_STORE_MAX/n_laps` --
derived, not invented: a trajectory that has to fit in a 4 MJ store over n laps
cannot average more drift than that -- and the box does the rest of the
rejecting. Coverage 0.01 -> 0.67.

**And the raw store is now reported as a bracket.** `E_k = c + F_k` with c
unidentified, so the store at sample k is the whole interval
`[F_k - min F, E_max - (max F - F_k)]`. The old code placed c at that interval's
centre and reported percentiles across particles, which read as an estimate;
once the flows were fixed that convention sat 2 MJ from the truth and coverage
went to **0.00**. The earlier 0.53 was an artefact of the over-counted floor
dragging F down until the centre happened to land near the truth. Reported as
the interval the trace admits: coverage 0.74 at 3.50 MJ wide, almost the whole
box -- which is exactly why invariant 4 reports deployable energy instead.

**The earlier all-samples shape likelihood also found signal.** For each theta the band
implies a P_K(t), and implied P_K is linear in CdA through the v^3 drag term, so
a wrong CdA leaves a residual proportional to `delta_CdA * v^3` once the policy
step is fitted out. Regressing implied P_K on `[step(v), 1, v^3]` and penalising
the v^3 coefficient moves CdA_X from 0.271 to **0.408**, with weighted mass in
three of eight bins instead of one.

The wrong statistic here is instructive: the clipping distortion
`sum(clip(lam*want) - lam*want)^2` measures how often the band binds, and the
band is *widest* at low CdA, so it rewards low drag systematically. It collapsed
the cloud to one bin at ESS 2 and pushed the posterior down to 0.297.

| variant (seed 42) | CdA_X | deployable coverage | MAPE | ESS |
|---|---|---|---|---|
| measured v_cut, full | 0.597 | 0.67 | 9.9% | 364 |
| no shape likelihood | 0.502 | 0.79 | 9.1% | 400 |
| v_cut inferred jointly | 0.610 | 0.62 | 9.2% | 351 |
| no box rejection | 0.502 | 0.67 | 10.1% | 364 |
| no closure weight | 0.597 | 0.67 | 9.9% | 364 |

Read the second row carefully: 0.502 is the uniform draw's own mean, i.e. with
the shape likelihood off nothing moves drag at all, and coverage is *better*
without it (0.79 against 0.67). A band that carries no drag information is
wider and covers more truth while saying less -- the same trap as raw-store
coverage. The third row is the one that stings least and matters most: jointly
inferred v_cut reaches a comparable CdA_X on this seed once the two bugs above
are fixed, but it empties the ensemble at the end of the trace on two of three
seeds and it does so for the particles' convenience rather than the trace's.

Two earlier claims do not survive this table. The policy *proposal* was
credited with making everything work (coverage 0.62 against 0.14) -- that 0.14
was measured while the box was rejecting on a sampled `c`, and with the box
fixed the proposal is worth nothing on coverage (0.60 against 0.63). What the
prior buys is drag, through the likelihood, not coverage. And the soft closure
weight slightly *hurts* energy error (15.3% against 13.8%), on top of being
redundant with the solved amplitude.

The identified set travels with the tilted posterior, because the shape
likelihood is a behavioural assumption that cannot be checked on real data.
`Belief.theta_polytope` carries the assumption-free interval and a test asserts
the posterior lies inside it.

## What does not work

**Drag area is still 10-25% low** -- 0.597 / 0.571 / 0.494 against a true 0.660,
where the direct dead-band least squares on the same traces gives 0.632. So the
filter recovers most but not all of what the statistic contains, and the gap is
not the statistic: it is that the band still admits some surviving deployment
(the simulator's policy is positional, so no speed cut-off is clean) and that
bias is one-sided by construction. Seed 13 is the weakest of the three because
its fitted cut-off is the lowest (214 km/h against 236 and 239), which admits
more of it.

**Deployable coverage is 0.67-0.80 where it should be 0.85.** The missing
variance is named rather than tuned away: the CdA posterior is not propagated
into the flows, and the per-lap drift nuisance is uniform where the real thing is
a strategy. Narrowing the band to improve a MAPE number would be the wrong
trade and is explicitly not done.

The store box is now the binding constraint rather than a formality: 3,146
rejections against 643 from untestability, and the ensemble empties at the very
last sample. That is a real tension -- the dead-band likelihood pushes drag up,
which raises implied deployment, which widens range(F) towards the 4 MJ the
store allows -- and `first_empty_sample` plus the two rejection counts are
reported so it cannot be mistaken for either a clean run or a crash.

Also short of the plan: pooling reduces error 66% but width only to 0.87x. Six fixes have now been tried and measured: an accelerating gate on
the cut-out detector, hysteresis on it, 10x wider reserve jitter, the
buffer-release ramp, a `v_cut` policy gate, and detecting on the deployment
lower bound instead of the band midpoint. The `k/k+1` convention on the
likelihood *was* wrong and is fixed. Deployable energy is now also reported as
`since_reserve`, the net flow integral since the last reserve hit, which is
identified without separating E from R -- but it inherits the detector's
reliability, resets about twice a lap, and so correlates only +0.18 with the
truth.

The detector fires in mostly the right places (true store 0.02 MJ median at
hits) with a few false positives at a nearly full store (2.81 MJ max), and those
are enough to hold R at its prior. Inferred buffer 0.97 +/- 0.32 MJ for a driver whose policy holds
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
