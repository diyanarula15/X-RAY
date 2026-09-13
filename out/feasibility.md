# Stage 2 feasibility — the four reality checks

Measured on real 2026 sessions via FastF1, not assumed. Every number here is
produced by `scripts/07.a_feasibility.py`; rerun it to reproduce.

Sessions probed: 6 — Zandvoort (R12), Montréal (R5), Silverstone (R9), Spa-Francorchamps (R10), Melbourne (R1), Monte Carlo (R6)


## 1.1 Does 2026 telemetry exist, and at what rate?

Yes. The `Speed` channel is populated for every session probed. It is **irregularly
sampled**, so the nominal rate is meaningless — the distribution of Δt is what matters.


| circuit | median Δt | effective Hz | p10 | p90 | max Δt | median Δx |
|---|---|---|---|---|---|---|
| Zandvoort | 0.240 s | **4.17 Hz** | 0.16 | 0.36 | 0.84 s | 13.0 m |
| Montréal | 0.240 s | **4.17 Hz** | 0.16 | 0.40 | 1.04 s | 14.4 m |
| Silverstone | 0.240 s | **4.17 Hz** | 0.16 | 0.40 | 1.00 s | 16.3 m |
| Spa-Francorchamps | 0.240 s | **4.17 Hz** | 0.16 | 0.40 | 1.12 s | 15.0 m |
| Melbourne | 0.240 s | **4.17 Hz** | 0.16 | 0.36 | 1.12 s | 15.4 m |
| Monte Carlo | 0.240 s | **4.17 Hz** | 0.16 | 0.40 | 1.16 s | 10.5 m |

Effective rate across sessions: **4.17–4.17 Hz**.

> **This is the headline risk.** The Stage 1 ablation put the accuracy cliff at about
> 4 Hz: 3.9% per-lap energy error at 3.7 Hz, 19.4% at 2 Hz, 41.7% at 1 Hz.
> Real 2026 telemetry lands at 4.2–4.2 Hz — **on the edge of that cliff,
> with no margin.** It is not below it, but nothing about the rate is comfortable, and
> the p90 Δt of 0.40 s means a meaningful
> tail of the trace is sampled at 2.5 Hz or worse. This must be stated in the UI and
> the pitch, not buried.


### Which channels are populated

| circuit | Speed | Throttle | Brake | nGear | RPM | DRS |
|---|---|---|---|---|---|---|
| Zandvoort | 100% nonzero | 99% nonzero | 21% nonzero | 100% nonzero | 100% nonzero | **all zero** |
| Montréal | 100% nonzero | 83% nonzero | 19% nonzero | 100% nonzero | 100% nonzero | **all zero** |
| Silverstone | 100% nonzero | 86% nonzero | 13% nonzero | 100% nonzero | 100% nonzero | **all zero** |
| Spa-Francorchamps | 100% nonzero | 84% nonzero | 13% nonzero | 100% nonzero | 100% nonzero | **all zero** |
| Melbourne | 100% nonzero | 79% nonzero | 13% nonzero | 100% nonzero | 100% nonzero | **all zero** |
| Monte Carlo | 100% nonzero | 69% nonzero | 29% nonzero | 100% nonzero | 100% nonzero | **all zero** |

**DRS is dead in 2026, as expected** — the channel exists and is identically zero.
Throttle, brake, gear and RPM *are* populated. That is more than Stage 1 assumed: the
simulator's estimator deliberately saw speed only. It changes nothing about the core
claim — **there is still no energy, deployment or state-of-charge channel, which is the
entire point** — but it does mean regime classification (accelerating / braking /
corner-limited) can be read off the feed instead of inferred from the speed trace.
Stage 2 uses them for regime only, and the estimator still infers energy from dynamics.


## 1.2 Does the high-speed calibration window exist at real circuits?

**No. This is the finding that forces the Stage 2 methodology change.**


Stage 1 calibrated drag above 340 km/h, where the regulatory taper has squeezed
deployment to near zero. Fraction of all race samples above each threshold:


| circuit | v_max | >340 | >330 | >320 | >310 | >300 | >290 | samples |
|---|---|---|---|---|---|---|---|---|
| Zandvoort | 353 km/h | 0.11% | 0.36% | 1.29% | 3.48% | 5.10% | 6.67% | 420,417 |
| Montréal | 350 km/h | 0.04% | 0.22% | 1.04% | 3.89% | 7.76% | 9.97% | 342,127 |
| Silverstone | 350 km/h | 0.01% | 0.05% | 0.21% | 0.80% | 4.15% | 11.69% | 377,727 |
| Spa-Francorchamps | 349 km/h | 0.08% | 0.48% | 3.17% | 9.67% | 14.35% | 18.84% | 376,141 |
| Melbourne | 343 km/h | 0.00% | 0.14% | 1.53% | 5.78% | 10.23% | 14.84% | 337,175 |
| Monte Carlo | 292 km/h | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 538,218 |

Even at **Zandvoort**, the fastest circuit probed, only **0.11%** of samples clear 340 km/h — a few hundred samples across an entire race, and they are spread across 20 cars.


> **Verdict: §2 generalized calibration is mandatory, not optional.** A binary
> `v > 340` gate does not survive contact with real telemetry at any circuit. The
> continuous formulation (the taper ceiling as a smooth speed-varying constraint) and
> the coast-down channel are the only way this works outside a simulator.


## 1.3 Can we get track gradient?

**Yes, from FastF1 `Position` data — 6 of 6 sessions carry a live Z channel.**


| circuit | pos samples | source | Z range | Z std | frac zero |
|---|---|---|---|---|---|
| Zandvoort | 275 | lap | 9.0 m | 2.0 m | 0% |
| Montréal | 293 | lap | 5.2 m | 1.6 m | 0% |
| Silverstone | 350 | lap | 11.3 m | 3.2 m | 0% |
| Spa-Francorchamps | 435 | lap | 102.4 m | 30.1 m | 0% |
| Melbourne | 307 | lap | 2.5 m | 0.5 m | 0% |
| Monte Carlo | 8087 | session | 89.6 m | 27.9 m | 40% |

Zandvoort reports 9 m of elevation change, which matches the real circuit. The channel is usable: smoothed along the centreline it gives `sin(grade)` directly, and it is also what makes the 3D track ribbon real geometry rather than a drawing.


## 1.4 Are active-aero zones published anywhere?

Not in machine-readable form. FIA race director event notes are PDFs, and FastF1 exposes
no aero-mode channel. Per the spec, aero zones are hand-built in
`xray/data/circuits/<circuit>.yaml` for a small number of circuits only, from the
published straight/corner geometry, and the file records that they are hand-entered.


## Data quality: gaps


| circuit | laps scanned | Δt gaps > 1.0 s | per lap |
|---|---|---|---|
| Zandvoort | 1294 | 864 | 0.67 |
| Montréal | 1144 | 857 | 0.75 |
| Silverstone | 1009 | 995 | 0.99 |
| Spa-Francorchamps | 858 | 872 | 1.02 |
| Melbourne | 1006 | 1004 | 1.00 |
| Monte Carlo | 1346 | 1712 | 1.27 |

Gaps above 1.0 s are common — roughly one per lap. The pipeline never interpolates
across them; it marks them and the estimator suspends across the break (spec §8).


## Weather: real air density


| circuit | air temp | pressure | wind |
|---|---|---|---|
| Zandvoort | 18.5 °C | 1025 mbar | 1.5 m/s |
| Montréal | 12.8 °C | 1024 mbar | 3.0 m/s |
| Silverstone | 24.8 °C | 1006 mbar | 1.8 m/s |
| Spa-Francorchamps | 18.0 °C | 973 mbar | 2.0 m/s |
| Melbourne | 24.2 °C | 1013 mbar | 2.6 m/s |
| Monte Carlo | 23.7 °C | 1021 mbar | 0.8 m/s |

Stage 1 assumed ρ = 1.20 kg/m³. Real sessions publish air temperature and pressure, so
ρ is computed per session from the ideal gas law instead of assumed — one fewer nuisance.


## Verdict

| check | result | consequence |
|---|---|---|
| 1.1 telemetry rate | 4.2–4.2 Hz, irregular | **On the cliff edge.** Stated in the UI; no margin claimed. |
| 1.2 calibration window | Essentially absent everywhere | **§2 is mandatory.** Continuous taper constraint + coast-down. |
| 1.3 gradient | Z channel live at 6/6 | Usable. Circuits without it are excluded, not fudged. |
| 1.4 aero zones | Not machine-readable | Hand-built for 3 circuits, flagged as such. |

Proceed to Phase B, with §2 promoted ahead of the UI as the spec requires.
