# X-RAY: status, September 2026

Branch `math-model`, 17 commits, one per build step. **Additive**: the Stage 1
estimator (`xray/estimator.py`) and its golden file are untouched and still
pass, so nothing here has displaced a working number. `pytest -q` is 111 passed
in about 52 s.

This file is the map. `docs/model.md` is the detail: every number below has its
derivation and its failed alternatives there.

---

## The claim being defended

A rival car's hidden electrical energy state can be reconstructed from its
**public speed trace alone** and used to time an overtake. Public F1 telemetry
carries speed, throttle, brake, gear, RPM and position. It carries no energy
channel. That absence is the premise, not an inconvenience.

Two quantities matter and they are not equally identifiable:

- **Deployable energy** -- how much the driver can still spend. A flow integral,
  measured well.
- **Drag area (CdA)** -- needed to convert speed into power at all. Bounded by
  the regulation from both sides, never thresholded.

The reporting discipline that falls out of this is the project's spine: the
**assumption-free identified set** and the **policy-tilted posterior** are two
numbers and travel together. On real data the second rests on a behavioural
assumption about how drivers deploy, and that cannot be checked.

---

## What is built

| Step | Module | What it owns |
|---|---|---|
| 1 | `regs.py` | The 2026 rulebook as bounds. Pre/post-Miami variants by date. |
| 1 | `balance.py` | Interval energy balance -> linear constraints on theta. |
| 2 | `setmem.py` | The feasible set as a polytope; projections by LP; L1 alarm. |
| 3 | `modes.py` | Aero mode and powertrain regime as an HMM. |
| 4 | `pipeline.py` | Composes the above. No numerics of its own. |
| 5 | `rbpf.py` | Store and theta by Rao-Blackwellised particle filter. |
| 6 | `strategy.py` | The PMP prior: three interpretable parameters per lap. |
| 7 | `pooling.py` | Partial pooling car -> team -> field, closed form. |
| 8 | `qmdp.py` | Deciding under the belief, with sensitivity reported. |
| 14 | `deadband.py` | v_cut and v_harv by changepoint, *before* the filter. |

`theta = (CdA_X, CdA_Z, F_rr, dm)`: the two aero states, rolling resistance as a
force at a reference mass, and dry mass minus the published weight.

### Measured against the simulator's hidden state

Three fixture seeds, 42 / 7 / 13, at a 3.7 Hz feed:

| Quantity | Result |
|---|---|
| Deployable-energy band coverage | 0.67 / 0.72 / 0.80 |
| Per-lap deployed energy | 9.9% / 15.2% / 14.0% MAPE |
| Drag area CdA_X | 0.597 / 0.571 / 0.494 against a true 0.660 |
| Identified set for CdA_X (seed 42) | [0.264, 0.744], two-sided |
| Raw store | a *bracket*, 0.74 coverage at 3.50 MJ wide |
| Constraint containment | 0 upper-bound violations at 3.7 / 20 / 100 Hz |
| Aero mode agreement | 93.1%, >95% of the residual at braking samples |
| Field pooling | 48% mean-absolute-error reduction over 20 cars |
| LP cost | 11-18 ms for a 12-lap race |

### The five results that shaped the design

1. **Integrate, don't differentiate.** Over an interval the work-energy theorem
   is an identity, so quantisation enters linearly and the *formulation improves
   as the feed slows*: noise-to-signal 274% at 200 Hz, 8% at 100 Hz, 3% at
   3.7 Hz. Real 2026 telemetry arrives at 4.17 Hz.
2. **A speed trace has no lower bound on drag.** Harvesting explains any
   deceleration and the kinetic term vanishes over a closed lap. The floor must
   come from a claim about ICE output, and post-Miami super-clipping (350 kW)
   kills the local one at every delta. Fuel closure supplies it instead:
   `CdA_X >= 0.264`, independent of the harvest rule.
3. **Windows, not intervals.** An intersection keeps the single tightest bound
   and never averages noise down. Kinetic terms telescope over a window, so
   measurement error stays at one interval's while signal grows: the CdA_X upper
   bound falls 1.057 -> 0.744 from 0.6 s to 8 s windows, then degrades as
   whole-lap windows average the informative high-speed running away.
4. **Deployable energy needs no cut-out detector.** `E_k = c + F_k` with c
   unidentified, so `D_k = F_k - min_j F_j` exactly once the reserve has been
   touched. A detector cannot tell a reserve hit from a policy cut-off without a
   policy, and its false positives at a full store are what broke the band
   (0.04 coverage) before the running-minimum form replaced it.
5. **Inside the polytope the trace is uninformative by construction.** Every
   theta in the set explains the data exactly. So drag is identified only in the
   **dead band** `v_cut < v < v_harv`, where the policy says P_K = 0 and the
   balance is pure ICE against drag: the residual there is
   `delta_CdA * 0.5 rho v^3` with no other regressor. Swept against the true
   band the statistic minimises at **CdA_X = 0.66, exactly the truth**.

### Step 14, the most recent: v_cut is measured, not inferred

Jointly inferred, particles *escape* the dead-band test rather than pass it --
v_cut ran to 98 m/s, 353 km/h, above the car's top speed, where the band is
empty. `deadband.py` fits

    y(v) = 0.5 rho CdA v^3 + F_rr v m/M_ref - A taper(v) 1[v<v_cut] + H 1[v>v_harv]

by **non-negative** least squares, scanning v_cut with a BIC penalty so an empty
band cannot win: 236 / 239 / 214 km/h on the three seeds. The v_harv scan
**declines on all three**, correctly -- the Stage 1 simulator harvests on the
brakes only, so there is no off-throttle super-clipping to find.

The changepoint was not what moved the numbers. It forced two diagnostics:

- **The deployment floor carried no measurement slack**, so per-interval
  `max(0, .)` rectified noise. At the *true* theta the summed floor claimed
  4.2-4.4 MJ of deployment a lap against a true 2.15 and drove `range(F)` to
  28.7 MJ against a 4 MJ store -- the box was rejecting the truth. Every
  box-rejection count reported before the fix was measuring the bug.
- **Resampling reindexed theta but not the flow bands**, so from the first
  resample each particle's bands belonged to another particle's theta. Per-lap
  weighted CdA_X was 0.60, correct; the reported posterior was 0.42.

---

## What is pending

In order. Each gates the next, and the order is deliberate: nothing after item 3
is worth doing on simulated data.

1. **Band calibration.** Deployable coverage is 0.67-0.80 where it should be
   >= 0.85. The missing variance is named rather than tuned away: the CdA
   posterior is not propagated into the flows, and the per-lap drift nuisance is
   uniform where the real thing is a strategy. **A band is never narrowed to hit
   a MAPE number.**
2. **A drifting-v_cut driver.** A simulator variant whose cut-off wanders
   +/- 10 km/h lap to lap with a varying reserve. Step 14's gain has to survive
   it, or the shape likelihood gets a wider sigma and the polytope becomes the
   headline. The current fixture has a *fixed* policy, which makes the shape
   likelihood trivially well specified -- that is the largest unfalsified
   assumption in the stack.
3. **One pre-Miami race through `analyse_race.py`** (Bahrain or Jeddah: long
   dead band, 250 kW harvest cap so the local ICE floor also works). Checks:
   polytope non-empty per real lap, teammate CdA agreement, pit-stop mass slope.
   Empty polytopes mean the regulation tables are wrong, and that is a finding
   to report, not a bug to patch.
4. **`tau` gets a half-normal prior** (invariant 8). Pooled precision-weighted
   method-of-moments gives 0.0318 against a true 0.012; ten team pairs do not
   support per-team MoM.
5. **Step 9 of the original plan**, explicitly scoped as later: the periodic GP
   residual over track position, NPE proposals, and the live port of the
   kernels.
6. **`overtake.py` is untouched.** Its `b0`, `b1`, `b2`, `b3` are the one
   invented number in the codebase and they propagate into every decision claim.
   They cannot be fitted until the estimates feeding them are validated on real
   data, which is item 3.
7. **`scripts/run_estimator.py --seed 42` still raises an unhandled traceback**
   on a correct refusal (the tow-bound FOLLOWER). The fix is to catch and print
   the refusal, not to make the estimator return a number.

Four of the seven validation items in the original section 8 need real telemetry
and so sit behind item 3: polytope non-emptiness per real lap, the pit-stop mass
step, the RDD recomputed against the corrected `P_dep_max`, and the
pre/post-Miami split. The other three are done -- teammate consistency,
super-clip recovery, and batch == streaming.

---

## What we are aiming towards

**Near term.** A drag estimate whose error is dominated by physics rather than
by inference bookkeeping, and a deployable-energy band that is calibrated rather
than merely narrow -- then the same two numbers on one real race, with every
refusal reported as a refusal.

**Medium term.** Live operation on streaming telemetry with the hot kernels
ported to C++/Rust. Nothing is ported now, and the code is shaped for it rather
than built for it: kernels are pure `float64` arrays in, arrays out; the
estimator exposes `update(state, sample)` alongside `batch(trace)` and
`batch == reduce(update, .)` is asserted bit-for-bit on fixed seeds, which also
catches any hidden look-ahead. There are no threads, no ring buffers and no
sockets, and a change that is only useful for live waits.

**The end state.** A decision engine that names a lap and a zone to attack, with
the number it is least sure about stated next to the recommendation. That
requires the estimate to be trustworthy on real data first, which is why
`overtake.py` sits last rather than first.

### What would falsify the claim

Stated in advance, so the answer counts either way:

- Polytopes that come out empty on real laps -- the regulation tables or the
  labelling are wrong.
- Teammates whose identified CdA sets do not overlap.
- A dead band that does not exist at a real circuit's speeds, at which point
  drag is identified only from fuel closure and the band widens accordingly.
- Coverage that holds on the fixed-policy simulator and collapses on the
  drifting-v_cut variant -- the shape likelihood would then be reporting the
  simulator's policy back to itself.

### Standing invariants

Enforced by tests, not by convention. The estimator never imports the simulator
(`test_estimator_is_blind` parses the file with `ast`). CdA is bracketed, never
thresholded -- no speed threshold, no "clean sample" filter, no fallback guess.
Refusal is a correct output: Silverstone and Monaco legitimately return 0%, and
a car under tow is legitimately unidentifiable. Deployable energy is the
headline, not the raw store. Nulls are reported as nulls with an MDE beside any
effect estimate. New constants are named `ASSUMED_*` with their derivation, or
they do not go in. And if a change moves a headline metric outside noise, that
is a finding to report, not a test to relax.
