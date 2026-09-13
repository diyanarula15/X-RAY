# X-RAY — startup guide

Copy-pasteable commands, grouped by setup. Jump to the section matching what
you need; each one lists exactly which prerequisites it needs, so you don't
have to read the ones before it. For *why* each stage exists, what its
inputs/outputs are, and the pipeline order between scripts, see
[`dev_readme.md`](dev_readme.md) §2, §2a, and §3 — this file only sequences
commands.

All commands below are PowerShell-flavored (`.\.venv\Scripts\python`); swap
in `./.venv/bin/python` on macOS/Linux.

---

## 0. Prerequisites (shared)

- **Python 3.11+** is a hard floor (`numpy`/`pandas` publish no wheels for
  3.10 or earlier). Confirm with `python --version` — on Windows, `python` on
  bare PATH may be an older interpreter; always go through the venv below.
- **Node 20+ and npm** — only needed for Setup E (the frontend).
- **`ffmpeg` on PATH** — only needed for `06.b_make_video.py` (Setup B).

```powershell
py -3.12 -m venv .venv                 # or any 3.11+
```

Every setup below installs into this same `.venv` from the root
`requirements.txt`, which is the committed manifest. It is one file in two
commented blocks:

- **Light** (Stage 1 only): `numpy scipy matplotlib PyYAML pytest`
- **Full** (adds Stage 2 real-data + API): light tier +
  `fastf1 pandas==2.3.3 fastapi uvicorn pyarrow`

`-r requirements.txt` installs both blocks, which is what every setup below
does and what Setups C–E require. For a Stage-1-only machine (Setup A or B)
you may instead install just the first block by hand.

`pandas==2.3.3`, not a newer 3.x, is required — fastf1 pins
`pandas<3.0.0,>=2.1.1` in its own metadata, so pip enforces it either way.

---

## Setup A — Stage 1 core only (no network, no Node)

The minimal "does it work" path: simulate a race, then reconstruct the
hidden energy state from public telemetry alone.

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python scripts\01.run_sim.py --seed 42
.\.venv\Scripts\python scripts\02.run_estimator.py --seed 42 --car LEADER
```

`--car LEADER` is required — `FOLLOWER` is tow-bound and raises by design
(`dev_readme.md` §7 item 6).

---

## Setup B — Stage 1 full artifact pipeline

Everything in Setup A, plus ablation, decision evaluation, attack
simulation, and the summary figure / demo video.

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python scripts\01.run_sim.py --seed 42
.\.venv\Scripts\python scripts\02.run_estimator.py --seed 42 --car LEADER
.\.venv\Scripts\python scripts\05.run_ablation.py                  # -> out/ablation.json
.\.venv\Scripts\python scripts\03.a_validate_decision.py           # oracle vs X-RAY vs blind
.\.venv\Scripts\python scripts\03.b_run_decision_eval.py --mode fast
.\.venv\Scripts\python scripts\04.simulate_attack.py
.\.venv\Scripts\python scripts\06.a_make_summary.py --seed 42      # needs out/ablation.json
.\.venv\Scripts\python scripts\06.b_make_video.py --seed 42        # needs ffmpeg on PATH
```

`06.a_make_summary.py` exits with `SystemExit` if `out/ablation.json` is
missing — run `05.run_ablation.py` first. `--mode fast` is shown explicitly
above even though it's already the default; `--mode full` on
`03.b_run_decision_eval.py` reruns a full 200 Hz sim per race (minutes, not
seconds).

---

## Setup C — Stage 2 real-data only, no server

Fits and feasibility checks against real FastF1 telemetry, no API or
frontend involved. Needs network on first run per round (FastF1 caches to
`xray/data/cache/` after that).

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python scripts\07.a_feasibility.py --round 10     # -> out/feasibility/r10.json
.\.venv\Scripts\python scripts\08.a_feasibility_report.py         # -> out/feasibility.md
.\.venv\Scripts\python scripts\07.b_run_real.py --round 10        # stdout: CdA fit, identified set
```

`08.a_feasibility_report.py` reads every `out/feasibility/r*.json` and exits
with `SystemExit` if none exist — run `07.a_feasibility.py` at least once
first. Pick rounds 1–3 (Melbourne/Shanghai/Suzuka) for a pre-Miami check —
see `dev_readme.md` §7 item 10 for why later "pre-Miami" round numbers are
wrong in the 2026 schedule.

---

## Setup D — Stage 2 + API, no frontend build

Adds the precomputed race JSON and a running API, without building the
React app — useful for hitting `GET /api/*` directly (curl, browser,
Postman) while iterating on the backend.

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python scripts\08.b_analyse_race.py --round 10    # -> out/races/<id>.json
.\.venv\Scripts\python -m uvicorn simulation.api.main:app --port 8011
```

Every route 404s until `08.b_analyse_race.py` has written a race JSON.
`GET /api/races` globs `out/races/*.json` directly (`simulation/api/main.py`
`_all_races()`) — it always reflects exactly whichever rounds have been
analysed, nothing more. To populate multiple races, loop the analysis step
over several round numbers before starting the server:

```powershell
foreach ($n in 1..10) {
    .\.venv\Scripts\python scripts\07.a_feasibility.py --round $n
    .\.venv\Scripts\python scripts\07.b_run_real.py --round $n
    .\.venv\Scripts\python scripts\08.b_analyse_race.py --round $n
}
.\.venv\Scripts\python scripts\08.a_feasibility_report.py   # optional, once at the end
```

`07.a`/`07.b` are optional per-round diagnostics (feasibility check / CdA
fit printout); only `08.b_analyse_race.py` writes the `out/races/<id>.json`
that the API serves. Each round needs a network fetch on first run only —
FastF1 caches to `xray/data/cache/` after that. Don't assume a fixed round
count: not every round number is guaranteed to exist for a given season, so
expect some rounds in a large range to fail — the loop above doesn't stop on
a single failure, so just check `out/races/` afterward to see which rounds
actually produced JSON. There is no round/calendar list checked into the
repo; round numbers are always picked by hand.

Routes:

```
GET  /api/races
POST /api/races/analyze
GET  /api/races/analyze/{job_id}
GET  /api/race/{rid}/summary
GET  /api/race/{rid}/car/{drv}
GET  /api/race/{rid}/battle/{a}/{b}
GET  /api/race/{rid}/observability
GET  /api/race/{rid}/decision
GET  /api/race/{rid}/p2
GET  /api/race/{rid}/p3
GET  /api/race/{rid}/counterfactual
GET  /api/race/{rid}/replay
GET  /api/p3/status
GET  /api/rdd
```

---

## Setup E — Full stack (Stage 2 + API + frontend)

Everything in Setup D, plus the built React/three.js viewer served at `/`.

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python scripts\08.b_analyse_race.py --round 10

cd simulation/app
npm install
npm run build
cd ../..

.\.venv\Scripts\python -m uvicorn simulation.api.main:app --port 8011
```

Open <http://127.0.0.1:8011/> (append `?lite=1` on a weak GPU).
`simulation/api/main.py`'s `ROOT` resolves to `simulation/`, so it serves
`simulation/app/dist/` once built.

**Frontend-dev variant** — iterating on `simulation/app/` itself without
rebuilding through the API on every change:

```powershell
# API still needs to be running (Setup D) for data requests
cd simulation/app
npm install
npm run dev              # Vite dev server with hot reload
```

---

## Setup F — Tests only

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt    # full manifest: some tests exercise Stage 2 modules

.\.venv\Scripts\python -m pytest -q
```

The light tier (no `fastf1`/`pandas`/`fastapi`/`pyarrow`) is enough for the
Stage 1 suite — every `tests/test_*.py` file except the ones that import
`fastf1`, `pandas`, `fastapi`, or `simulation.api.main`/`xray.data.ingest`/
`xray.realfit`. Currently that's: `test_frontend_p2.py`, `test_p3_part1.py`,
`test_p35_part1.py`, `test_p35_part2.py`, `test_p35_part3.py`,
`test_environment.py`, `test_decision_service.py`, `test_live_readiness.py`
— those need the full tier.

**Do not** run `scripts\99.make_golden.py` to silence a golden-baseline
failure — see `dev_readme.md` §7 item 2 for the known cross-platform
tolerance issue and why regenerating the baseline defeats the point of the
test.

---

## Setup G — Maintenance / audit scripts

Not part of the normal pipeline; run these deliberately, not as a fix for a
failing test.

```powershell
.\.venv\Scripts\python scripts\99.make_golden.py --seed 42   # only when a numerics change is intended
.\.venv\Scripts\python scripts\99.worked_trace.py            # prints one worked decision trace + causality proof
.\.venv\Scripts\python scripts\99.release_manifest.py        # -> out/release_manifest.json
```

`99.make_golden.py` regenerates `tests/golden/belief_seed<seed>_3.7hz.npz`,
pinned to `tests/conftest.py`'s exact configuration. If you run it, say in
the commit message which headline metric moved and why (`dev_readme.md` §5).

`99.release_manifest.py` pins a claim to an exact commit, estimator state and
model fingerprint — reads existing sources of truth (git HEAD, the registry,
`overtake.COEFFS`, the tyre-compound table) rather than recomputing anything.
Run it before handing off a build for review; it needs the full manifest
installed (imports `xray.registry`/`xray.overtake` and reads
`simulation/api/main.py`'s FastAPI version string, but not a running server).
