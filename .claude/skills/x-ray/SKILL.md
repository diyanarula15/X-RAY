---
name: x-ray
description: Working rules for the X-RAY codebase — reconstructing a rival F1 car's hidden electrical energy state (deployable energy, CdA) from public speed telemetry alone, in a Stage 1 simulator (`xray/`, `tests/`) and a Stage 2 real-data pipeline (FastF1 ingest, `xray/regs.py`/`setmem.py`/`rbpf.py`/`pooling.py`/`qmdp.py`, `simulation/api/`, `simulation/app/`). Use this whenever touching any `xray/*.py` module, the simulator, `overtake.py`, `08.b_analyse_race.py`, `07.a_feasibility.py`, the tests, the API, or the frontend; whenever asked to make the simulation or filter faster; and whenever writing or editing comments in this repo. Consult it even for small edits — the repo has hard invariants (blindfold, bracket-not-threshold, decline-not-guess, regulation-variant agreement) that a well-meaning change can silently break.
---

# X-RAY

One claim: a rival's hidden electrical energy state can be reconstructed from its
public speed trace alone and used to time an overtake. Public telemetry has
speed/throttle/brake/gear/RPM/position and no energy channel. The absence is the
premise; everything below defends it.

Two quantities matter and they are not equally identifiable:
- **Deployable energy** — how much the driver can still spend. A flow integral,
  measured well.
- **Drag area (CdA)** — needed to convert speed into power at all. Bounded by the
  regulation from both sides, never thresholded.

The reporting discipline this implies is the project's spine: the
**assumption-free identified set** and the **policy-tilted posterior** are always
reported as two numbers together. On real data the tilted posterior rests on a
behavioural assumption about how drivers deploy, and that cannot be checked —
never collapse it to one number.

## Architecture — Stage 1 (simulator) and Stage 2 (real data), same estimator core

| Module | What it owns |
|---|---|
| `xray/regs.py` | The 2026 rulebook as bounds. Pre/post-Miami variants by date. |
| `xray/balance.py` | Interval energy balance → linear constraints on `theta`. |
| `xray/setmem.py` | The feasible set as a polytope; projections by LP; L1 alarm for infeasibility. |
| `xray/modes.py` | Aero mode and powertrain regime as an HMM. |
| `xray/pipeline.py` | Composes the above. No numerics of its own. |
| `xray/rbpf.py` | Store, reserve and `theta` by Rao-Blackwellised particle filter. |
| `xray/deadband.py` | `v_cut`/`v_harv` by changepoint (NNLS + BIC scan), fitted *before* the filter, handed in as an input. |
| `xray/strategy.py` | The PMP policy prior: `v_cut`, `v_harv`, aggression `w` per lap. |
| `xray/pooling.py` | Partial pooling of `tau` across the field, closed form (no MCMC). |
| `xray/qmdp.py` | Deciding under the belief, with sensitivity reported. |
| `xray/estimator.py` | Stage 1's older, simulator-facing estimator. Must never import the simulator — see invariant 1. |
| `xray/vehicle.py`, `xray/sim.py`, `xray/track.py`, `xray/policy.py` | The simulator: physics ground truth the estimator is blind to. |
| `xray/decision.py`, `xray/decision_service.py`, `xray/overtake.py` | The decision layer consuming the belief. |
| `xray/realfit.py`, `xray/analysis.py` | Real-telemetry fitting and race analysis, driven by `scripts/08.b_analyse_race.py`. |
| `simulation/api/main.py`, `simulation/app/` | FastAPI service + React/Three.js frontend that serve the six-view instrument (see Stage 2 section). |

`theta = (CdA_X, CdA_Z, F_rr, dm)`: the two aero states, rolling resistance as a
force at a reference mass, and dry mass minus published weight.

## Invariants — do not break these, and add a test if you touch them

1. **The estimator is blind.** `xray/estimator.py` never imports the simulator.
   Enforced by `test_estimator_is_blind`, which parses the file with `ast` and
   fails the build on any simulator import. Do not "helpfully" share a constants
   module between them either — if a number lives in the simulator, the
   estimator must derive it or take it as a public-regulation input.
2. **CdA is bracketed, never thresholded.** Wheel power is linear in CdA and the
   2026 regulation bounds it from both sides, so *every* sample brackets CdA.
   Real telemetry across six 2026 races clears 340 km/h on 0.00–0.11% of samples
   (Monaco: none, tops out at 292 km/h) — a speed-threshold "clean sample"
   estimator returns nothing on five of six real circuits. Never reintroduce a
   speed threshold or a fallback guess. Interval width is the identifiability
   score; drag is otherwise identified only in the dead band `v_cut < v < v_harv`
   (`xray/deadband.py`), where the policy sets P_K = 0 and the residual is pure
   `delta_CdA * 0.5 * rho * v^3`.
3. **Decline rather than guess.** Silverstone and Monaco legitimately return 0%
   identifiability on real 2026 data. A `FOLLOWER` under tow, or a car with no
   dead-band samples, is legitimately unidentifiable. Refusal is a correct
   output, not an error path to paper over — but it must be *caught and
   reported*, never an unhandled traceback (see Gotchas).
4. **Report deployable energy, not raw store.** A speed trace measures flows
   exactly and absolute level only up to a constant: `E_k = c + F_k` with `c`
   unidentified. If the reserve has been touched at least once,
   `D_k = F_k - min_{j<=k} F_j` exactly; before that it's a lower bound. Raw
   store is reported as the *bracket* `c` admits (coverage 0.74 at ~3.5 MJ
   wide), never as a point estimate with a chosen `c` — an earlier version
   centred `c` in the interval and it read as a real number. Do not add a
   raw-store headline or try to "fix" its coverage; that width is correct.
5. **No cut-out detector for the reserve.** A detector ("deployment stops while
   throttle is full") cannot distinguish a genuine reserve hit from a policy
   cut-off without a policy, and its false positives at a near-full store
   collapsed the deployable band to 0.04 coverage before the running-minimum
   form replaced it. If a detector is reintroduced, gate it on a *sustained*
   deviation from the policy prior, not a single sample.
6. **Store closure is solved, not guessed, and per-lap drift is bounded, not
   forced to zero.** Deployed ≈ harvested within one store over the *race*, but
   forcing exact per-lap closure is wrong when a lap genuinely starts on a full
   store and dumps energy (coverage collapses to 0.01 if you do). Each particle
   carries a per-lap drift nuisance bounded by `E_STORE_MAX / n_laps` — derived
   from the store size, not invented — and the store box does the rejecting.
7. **Nulls are reported as nulls, with power.** The real-data RDD at the 1.000 s
   Manual Override boundary is −0.20 MJ, p = 0.41, MDE 0.68 MJ vs a 0.5 MJ
   allocation, over 5 races / 1,851 car-laps. Any code path that prints an
   effect estimate prints the MDE next to it. Never reframe a null as evidence
   for the regulation working or not working.
8. **One `Rules(D)` object; simulator and fixtures must agree on it.** The
   simulator once ran 350 kW (post-Miami) harvest while fixtures assumed 250 kW
   for a full iteration; every measurement came out 0.71× on harvest. The
   blindfold forbids the estimator *importing* the simulator; it does not
   forbid a test asserting `sim.rules == fixture.rules`. A regulation-variant
   mismatch must fail loudly, not silently scale results.
9. **Assumption-free set and policy-tilted posterior are always two numbers.**
   `Belief.theta_polytope` carries the assumption-free interval; the RBPF
   posterior is tilted by the strategy-shape likelihood, which is a behavioural
   assumption unverifiable on real data. A test asserts the posterior lies
   inside the polytope. Never present only the tilted number.
10. **Don't treat a heuristic or a calibration placeholder as physical ground
    truth.** See "Model inventory" below before adding presentation code that
    reads a value as if it were measured physics.

## Headline numbers (regression baseline — reproduce before and after any change)

Simulator / fixture seeds 42, 7, 13 at 3.7 Hz (`docs/model.md`, `docs/status.md`
were the source of these before deletion — reproduce via the scripts named
there, e.g. `scripts/05.run_ablation.py`, `scripts/99.make_golden.py`):

| Metric | Value |
|---|---|
| `pytest -q` | 111 passed (~52 s) on the full math-model stack |
| Deployable-energy band coverage | 0.67 / 0.72 / 0.80 (target ≥0.85 — a known open item, do not narrow the band to hit it) |
| Per-lap deployed energy MAPE | 9.9% / 15.2% / 14.0% |
| Drag area CdA_X | 0.597 / 0.571 / 0.494 vs true 0.660 (still 10–25% low; dead-band NNLS alone gets 0.632) |
| Identified set for CdA_X (seed 42) | [0.264, 0.744], two-sided, bracketing 0.660 |
| Raw store | bracket only, 0.74 coverage at 3.50 MJ wide |
| Constraint containment | 0 upper-bound violations at 3.7 / 20 / 100 Hz |
| Aero mode (HMM) agreement | 93.1%, >95% of residual at braking samples |
| Field pooling | 48% mean-absolute-error reduction over 20 cars |
| LP cost | 11–18 ms for a 12-lap race |

Real 2026 telemetry (`scripts/07.a_feasibility.py`, `scripts/08.b_analyse_race.py`):

| Metric | Value |
|---|---|
| Real telemetry rate | 4.17 Hz median, irregular (p90 gap 0.36–0.40 s) — on the accuracy cliff Stage 1's ablation found near 4 Hz, no margin |
| Samples >340 km/h across 6 races | 0.00–0.11% (Monaco: 0%, max 292 km/h) |
| Identifiability by circuit | Zandvoort 97%, Spa 78%, Melbourne 72%, Silverstone 0%, Monaco 0% |
| Per-lap energy, real races | 3.5–4.7 MJ deployed vs 3.3–4.6 MJ recovered (energy-neutral inside a 4 MJ store) |
| RDD at 1.000 s boundary | −0.20 MJ, SE 0.242, p = 0.41, MDE 0.68 MJ vs 0.5 MJ allocation (5 races, 1,851 car-laps) — a null, not a finding either way |
| Decision quality (`scripts/03.a_validate_decision.py`) | at full rate: oracle's exact lap 100% of races; at real 4.17 Hz: exact 40%, within one lap 100%, EV cost of the miss ≈ nil |

If a change moves any of these outside noise, that is a finding to report, not a
test to relax.

## Gotchas (found, not read — verify they are still true before relying on them)

- `python scripts/02.run_estimator.py --seed 42` crashes with an unhandled
  traceback. Cause: correct refusal on the tow-bound `FOLLOWER`, uncaught by the
  script. Use `--car LEADER`. The right fix is to catch the refusal and print
  it, not to make the estimator return a number. (Still open per `docs/status.md`
  before deletion — check before assuming it's fixed.)
- Two bugs that produced *plausible-looking* wrong numbers before being caught,
  worth re-checking for the pattern whenever a metric looks slightly-off-but-
  reasonable: (a) `sim.step()` mutates state then records, so `trace.v[k]` is
  post-step while `trace.P_mguk[k]` is the power applied during it — always
  close energy against index `k+1`, not `k`; (b) after a particle-filter
  resample, per-particle flow bands (`e_wheel`, `d_lo`, `d_hi`, `harvest`) must
  be recomputed from the *current* theta, not carried over from the initial
  draw — otherwise a resample silently scores each particle's dead-band
  residual against a different particle's drag.
- Deployment-floor terms need 5-sigma measurement slack when accumulated over an
  interval; a bare `max(0, e_wheel - e_ice_max)` with no slack rectifies noise
  one-sidedly and can inflate a lap's claimed deployment several-fold, driving
  `range(F)` far past the store size and making the store-box reject the truth.
- `overtake.py` contains the one *invented* constants (`b0`, `b1`, `b2`, `b3`) in
  the codebase, and they propagate into every decision claim. Fitting them on
  real data is gated behind validating the estimator on real data first — do
  not fit them prematurely. Do not add more invented constants; if you must,
  name them `ASSUMED_*` and comment the derivation (e.g. `ASSUMED_MIAMI_2026`
  for the pre/post-Miami changeover date — the single most worth re-checking).
- Stage 2 requires `pip install -r requirements.txt` (adds `fastf1`, `fastapi`,
  `uvicorn`), `python scripts/08.b_analyse_race.py --round N` (downloads real
  telemetry via FastF1 — slow, network-bound, cache it), and
  `cd simulation/app && npm install && npm run build` before the API/frontend serve
  anything real. Check `.venv/` contents and whether `out/`/`simulation/app/dist` exist
  before assuming Stage 2 has been run locally.
- An earlier version rejected a whole lap on a single telemetry gap, discarding
  ~⅔ of every real race. Gaps are handled at sample level (grid cells spanning a
  gap are blanked, never interpolated across); don't reintroduce lap-level
  rejection on top of that.

## Model inventory — don't treat these as more certain than they are

When writing presentation code (UI, reports, decision explanations), match the
claim strength to the category:

- **Physics** (ground truth): `xray.vehicle.drag_force`, `xray.vehicle.step`
  (rolling resistance, grade force, `P = Fv`, battery integration), longitudinal
  timestep sim, `xray.decision.calibrate_zone`.
- **Regulation** (rulebook, not measured): `xray.constants.mguk_power_limit_normal`
  / `_overtake`, `xray.constants.POWER_UNIT_2026`.
- **Reduced-order physics**: track reduced to straights/corner entries/apex
  speeds/grade/zone severity; braking is a reduced force envelope, not a
  brake/tyre thermal model.
- **Heuristics** (not fitted, not physical): `Track._build_zones` synthetic
  braking severity; `RealTrack` zone extraction from longest straights;
  `decision_service.gap_at_position` low-confidence fallback and its causal
  constant-velocity projection when traces don't overlap in time; the
  exponential dirty-air surrogate in `xray.sim`/`xray.policy`;
  `xray.estimator.RESERVE_SIGMA` (a synthetic observation-noise calibration
  knob, not a battery parameter); the bounded historical lap-rate transition
  used for future rival-energy forecasts in `decision_service`.
- **Statistical / learned, calibration is placeholder**: `xray.overtake.p_pass`
  — coefficients are synthetic design anchors, *not* fit to real race data yet
  (see Gotchas). `xray.estimator` / `xray.realfit` hidden-energy belief fields:
  `usable_mean`, `soc_mean`, `reserve_mean`, `deployed_lap` (point estimate),
  `deployed_lap_posterior_mean`, `harvested_lap` — J internally, MJ in web JSON.
- **Optimization**: `xray.decision.solve` (finite-horizon DP),
  `xray.decision.solve_exogenous` (exogenous rival-energy DP used by the web
  service), `xray.decision_service.evaluate_overtake_decision` (canonical
  web/core entry point).

## The generalized calibration (Stage 2's replacement for the 340 km/h window)

Every sample brackets CdA — no threshold anywhere:

```
(-P_MGUK_MAX - A)/B  <=  CdA  <=  ((P_ICE_MAX + ceiling(v))*eta - A)/B
```

Both bounds come from the rulebook, not an engine model — an earlier version
inferred ICE output from throttle and got `CdA > 13` lower bounds, because a
real car at 30 m/s is traction-limited, not power-limited, and no
throttle-to-power curve survives that. Sharpen with the feed: brakes-on drops
the lower bound (brake force isn't modelled); throttle-off drops the 400 kW ICE
term, tightening the upper bound severalfold (the coast-down channel — exists
at every circuit, unlike a high-speed window). Pool identified sets across the
field (20 cars, same regs, same air); a car with no clean running inherits the
field constraint and must be flagged as such in the UI.

Deployment is a *band*, not a point: lower bound = whatever exceeds the 400 kW
ICE cap must be electrical; upper bound = the MGU-K ceiling at that speed. Where
in the band the truth sits is not visible in the trace — it's pinned by store
closure (deployment ≈ recovery within one 4 MJ store over the race). A single
global split spreads deployment evenly around the lap where a real car bursts it
out of slow corners — visible in the app as the reconstructed store reading
empty 14–45% of the race with an over-narrow (0.04–0.26 MJ) belief band. That's
a known, stated gap, not a bug to quietly patch by widening the band to look
better.

## Efficiency rules for the simulator and filter

1. **Measure first.** `python -m cProfile -o prof.out <entry>` then
   `python -c "import pstats; pstats.Stats('prof.out').sort_stats('cumtime').print_stats(25)"`.
   For a single hot function, `line_profiler` (`kernprof -l -v`). Don't optimise
   anything outside the top of the profile.
2. **Vectorise the time axis.** Fixed-rate traces: replace
   `for i in range(len(speed))` with numpy array ops (`np.diff`, `np.cumsum`,
   boolean masks, `np.where`). Preallocate outputs; never append inside the
   sample loop.
3. **Keep the integrator's order of operations.** Check an explicit time-step
   recurrence is actually vectorisable (no dependence on its own previous
   output) before vectorising it. If not, keep the loop and JIT it
   (`numba.njit`, optional behind an import guard) rather than rewriting the
   physics.
4. **Pandas out of hot paths.** FastF1 returns DataFrames; convert to numpy once
   at the boundary. No per-row `.iloc`, no `.apply` with a Python lambda.
5. **Cache expensive inputs.** FastF1 sessions, generated reference laps, and
   per-track regulation tables should be cached (FastF1's own cache dir,
   `functools.lru_cache`, or `.npz` for generated traces). Seed all randomness
   explicitly and store the seed with the cache key.
6. **Keep the test suite cheap.** Move generated traces into session-scoped
   pytest fixtures; mark multi-seed coverage tests `@pytest.mark.slow`; keep a
   `-m "not slow"` path fast. `test_estimator_is_blind` must stay in the fast
   path.
7. **Prove it.** Every optimisation PR reports before/after wall time *and* the
   headline-metrics table above, unchanged to stated precision. A speedup that
   shifts MAPE by 0.1 pp is a bug until explained.

## Design for live (a later phase — shape the code for it now, don't build it)

1. **Kernels are pure.** Every numerical step (wheel power, CdA bracket update,
   deployable-energy accumulation, overtake decision) is a function of
   contiguous `float64` numpy arrays → arrays/scalars. No FastF1 objects,
   pandas, I/O, or plotting inside a kernel. The eventual C++/Rust port moves
   kernels only.
2. **Batch == streaming, tested.** Expose `state = update(state, sample)`
   alongside `batch(trace)`, and assert `batch(trace) == reduce(update, trace)`
   bit-for-bit on fixed seeds. This also catches hidden look-ahead (a filter
   peeking at future samples) — treat it as seriously as
   `test_estimator_is_blind`.
3. **Golden outputs define correctness.** Keep a fixed-seed `.npz` of estimator
   outputs per test trace (`scripts/99.make_golden.py`, `tests/test_golden.py`).
   Every optimisation must match golden to a stated tolerance before it merges.
4. **No live-only code paths yet.** No threads, no ring buffers, no sockets. A
   change only useful for live operation waits.

## House style (the codebase has a voice — keep it)

Comments explain *why the wrong version was wrong*, with the measured
consequence attached. Not what the code does.

**Bad:**
```python
# Clip deployment above 340 km/h
```

**Good:**
```python
# No speed threshold here. The 340 km/h "pure engine" window looked clean in the
# simulator but 0.00-0.11% of real samples reach it (Monaco: none, max 292
# km/h), so a thresholded estimator returned nothing on five of six real
# races. Every sample brackets CdA instead; interval width is the
# identifiability score.
```

Rules:
- Cite the number that killed the alternative (percent of samples, MJ,
  p-value).
- Refusals get a comment saying what would have to be true for the estimate to
  exist.
- No hedging adjectives ("robust", "careful") — state the measurement.
- Keep docstrings short; put the argument in the comment nearest the decision.

## Before you finish any task

- `pytest -q` passes, including `test_estimator_is_blind`.
- Headline metrics reproduced and pasted into the summary (Stage 1 fixture
  numbers and, if touched, Stage 2 real-data numbers).
- No new speed thresholds, no new fallback guesses, no new invented constants
  outside `ASSUMED_*` with a stated derivation.
- The assumption-free polytope and the policy-tilted posterior are both
  reported if either changed — never just the tilted number.
- If you touched `overtake.py`, say explicitly which numbers are still
  invented/assumed.
- If you touched anything in the "Model inventory" heuristic/statistical rows,
  say so — those are the values a reviewer should not mistake for physics.
