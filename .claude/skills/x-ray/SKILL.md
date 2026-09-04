---
name: x-ray
description: Working rules for the X-RAY codebase — reconstructing a rival F1 car's hidden electrical energy state (deployable energy, CdA) from public speed telemetry, in a Stage 1 simulator and Stage 2 real-data pipeline. Use this whenever touching estimator.py, the simulator, overtake.py, analyse_race.py, the tests, or the app; whenever asked to make the simulation faster; and whenever writing or editing comments in this repo. Consult it even for small edits — the repo has hard invariants (blindfold, bracket-not-threshold, decline-not-guess) that a well-meaning change can silently break.
---

# X-RAY

One claim: a rival's hidden energy state can be reconstructed from its public speed trace
alone and used to time an overtake. Public telemetry has speed/throttle/brake/gear/RPM/
position and no energy channel. The absence is the premise; everything below defends it.

## Invariants — do not break these, and add a test if you touch them

1. **The estimator is blind.** `estimator.py` never imports the simulator. This is
   enforced by `test_estimator_is_blind`, which parses the file with `ast` and fails the
   build on any simulator import. Do not "helpfully" share a constants module between
   them either — if a number lives in the simulator, the estimator must derive it or take
   it as a public-regulation input.
2. **CdA is bracketed, never thresholded.** Wheel power is linear in CdA and the 2026
   regulation bounds deployment from both sides, so *every* sample brackets CdA. The old
   ">340 km/h window" approach is dead (0.00–0.11% of real samples clear it; Monaco tops
   out at 292). Never reintroduce a speed threshold, a "clean sample" filter, or a
   fallback guess. The width of the surviving interval *is* the identifiability score.
3. **Decline rather than guess.** Silverstone and Monaco legitimately return 0%. A
   `FOLLOWER` under tow is legitimately unidentifiable. Refusal is a correct output, not
   an error path to paper over — but it should be *caught and reported*, never an
   unhandled traceback (see Gotchas).
4. **Report deployable energy, not raw store.** A speed trace measures flows exactly and
   absolute level only up to a constant, because a deployment cut-out means "hit the
   driver's buffer", not "empty". Raw-store coverage is 0.09–0.42 and that is expected.
   Do not add a raw-store headline or try to "fix" its coverage.
5. **Nulls are reported as nulls, with power.** The real-data RDD at the 1.000 s Manual
   Override boundary is −0.20 MJ, p = 0.41, MDE 0.68 MJ vs a 0.5 MJ allocation. Any code
   path that prints an effect estimate prints the MDE next to it. Never reframe a null
   as evidence for the regulation.

## Headline numbers (regression baseline — reproduce these before and after any change)

| Metric | Value |
|---|---|
| `pytest -q` | 26 passed (~195 s) |
| Deployed-energy MAPE | 4.6% @ 100 Hz, 3.5% @ 3.7 Hz |
| CdA error | 0.99% mean |
| Recovered energy | 6.3% @ 100 Hz vs 22.5% @ 3.7 Hz (braking-duration measurement → rate-sensitive; deployment is not) |
| Band width | accel 0.665 < braking 0.753 < corner 0.915 MJ |
| Band coverage (stated) | 0.86–0.90, test ceiling 0.92, averaged over three seeds |

If a change moves any of these outside noise, that is a finding to report, not a test
to relax.

## Gotchas (found, not read — verify they are still true before relying on them)

- `python scripts/run_estimator.py --seed 42` crashes with an unhandled traceback.
  Cause: correct refusal on the tow-bound `FOLLOWER`, uncaught by the script. Use
  `--car LEADER`. The right fix is to catch the refusal and print it, not to make the
  estimator return a number.
- Seed 42 band coverage runs 0.94–0.97 — above the stated 0.86–0.90 *and* the test's
  own 0.92 ceiling. It passes only because the test averages three seeds. Don't
  "fix" by widening the ceiling; understand why seed 42 is over-covered.
- Local state: `.venv/` (gitignored) has Stage 1 deps only. Stage 2 has never been run
  here — no `out/`, no `app/node_modules`, no `app/dist`. Nothing touching the six views
  or the API works until `pip install -r requirements.txt`, `python analyse_race.py`
  (downloads real telemetry via FastF1 — slow, network-bound, cache it) and
  `npm run build`. `PROJECT_BRIEF.md` is untracked.
- `overtake.py` contains the one *invented* number in the codebase, and it propagates
  into every decision claim. Fitting it on real data is the #1 open item. Do not add
  more invented constants; if you must, name them `ASSUMED_*` and comment the source.

## Efficiency rules for the simulator

The goal is a faster loop with *identical* numerics. Order of operations:

1. **Measure first.** `python -m cProfile -o prof.out <entry>` then
   `python -c "import pstats; pstats.Stats('prof.out').sort_stats('cumtime').print_stats(25)"`.
   For a single hot function, `line_profiler` (`kernprof -l -v`). Do not optimise
   anything that isn't in the top of the profile.
2. **Vectorise the time axis.** The simulator and estimator operate on fixed-rate
   traces. Any `for i in range(len(speed))` over samples should become numpy array ops
   (`np.diff`, `np.cumsum`, boolean masks, `np.where`). Preallocate output arrays; never
   append to lists inside the sample loop.
3. **Keep the integrator's order of operations.** When vectorising an explicit
   time-step, check the recurrence is actually vectorisable (no dependence on the previous
   output). If it isn't, leave the loop and JIT it (`numba.njit`, kept optional behind
   an import guard) rather than rewriting the physics.
4. **Pandas out of hot paths.** FastF1 returns DataFrames; convert to numpy once at the
   boundary. No per-row `.iloc`, no `.apply` with a Python lambda.
5. **Cache expensive inputs.** FastF1 sessions, generated reference laps, and per-track
   regulation tables should be cached (FastF1's own cache dir; `functools.lru_cache` or
   `.npz` for generated traces). Seed all randomness explicitly and store the seed with
   the cache key.
6. **Make the test suite cheap.** 195 s for 26 tests is mostly simulation warm-up.
   Move generated traces into session-scoped `pytest` fixtures, mark the multi-seed
   coverage tests `@pytest.mark.slow`, and keep a `-m "not slow"` path under ~30 s.
   The blindfold test must stay in the fast path.
7. **Prove it.** Every optimisation PR reports before/after wall time *and* the
   headline-metrics table above, unchanged to stated precision. A speedup that shifts
   MAPE by 0.1 pp is a bug until explained.

## Design for live (a later phase — shape the code for it now, don't build it yet)

Live operation will eventually run the estimator on streaming telemetry, likely with
the hot kernels ported to C++/Rust. Nothing is ported now. What is done now:

1. **Kernels are pure.** Every numerical step (wheel power, CdA bracket update,
   deployable-energy accumulation, overtake decision) is a function of contiguous
   `float64` numpy arrays → arrays/scalars. No FastF1 objects, pandas, I/O, or plotting
   inside a kernel. I/O lives in a thin outer layer. The port later moves kernels only.
2. **The estimator is incremental, and batch == streaming is tested.** Expose
   `state = update(state, sample)` alongside `batch(trace)`, and assert
   `batch(trace) == reduce(update, trace)` bit-for-bit on fixed seeds. The CdA bracket
   is a running interval intersection, deployable energy a running sum, identifiability
   the current interval width — so this costs nothing. The test also catches any hidden
   look-ahead (a filter peeking at future samples), which would be a correctness bug
   in live mode. Treat it with the same seriousness as `test_estimator_is_blind`.
3. **Golden outputs define correctness.** Keep a fixed-seed `.npz` of estimator
   outputs per test trace. Every optimisation — vectorising, `numba`, a future
   compiled kernel — must match golden to a stated tolerance before it merges.
4. **No live-only code paths yet.** No threads, no ring buffers, no sockets. If a
   change is only useful for live, it waits.

## House style (the codebase has a voice — keep it)

Comments explain *why the wrong version was wrong*, with the measured consequence
attached. Not what the code does.

**Bad:**
```python
# Clip deployment above 340 km/h
```

**Good:**
```python
# No speed threshold here. The 340 km/h "pure engine" window looked clean in the
# simulator but 0.00–0.11% of real samples reach it (Monaco: none, max 292 km/h),
# so a thresholded estimator returned nothing on five of six races. Every sample
# brackets CdA instead; the interval width is the identifiability score.
```

Rules:
- Cite the number that killed the alternative (percent of samples, MJ, p-value).
- Refusals get a comment saying what would have to be true for the estimate to exist.
- No hedging adjectives ("robust", "careful") — state the measurement.
- Keep docstrings short; put the argument in the comment nearest the decision.

## Before you finish any task

- `pytest -q` passes, including `test_estimator_is_blind`.
- Headline metrics reproduced and pasted into the summary.
- No new speed thresholds, no new fallback guesses, no new invented constants.
- If you touched `overtake.py`, say explicitly which number is still assumed.
