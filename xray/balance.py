"""Interval energy balance: one linear constraint per pair of samples.

Stage 1 differentiated the speed trace to get acceleration, then formed a power
balance. That is the dominant noise source in the whole pipeline: speed arrives
quantised to 1 km/h at ~4 Hz, so a single-sample difference is 0.28 m/s over
0.24 s = 1.16 m/s^2 of pure quantisation noise, which on 790 kg at 70 m/s is
64 kW -- against a 350 kW signal. Savitzky-Golay smoothing suppresses it at the
cost of a half-window of look-ahead (0.4-0.54 s, measured).

Integrating instead of differentiating removes the problem rather than filtering
it. Over an interval the work-energy theorem is exact:

    dE_kin + dE_pot = INT (P_wheel - P_drag - P_rr) dt - dE_brake

Nothing is differentiated: dE_kin is a difference of *squares of measured
speeds*, and the quantisation enters linearly instead of being divided by dt.

Every unknown enters linearly, which is what makes the feasible set a polytope
(see `setmem.py`):

    theta = (CdA_X, CdA_Z, F_rr, dm)

CdA_X / CdA_Z are the two aero states, F_rr is rolling resistance as a force at
a reference mass, and dm is the car's dry mass minus its published weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import E_HARVEST_LAP, E_STORE_MAX, G
from .regs import RegSet, p_ice_max, p_k_bounds

PARAM_NAMES = ("CdA_X", "CdA_Z", "F_rr", "dm")
N_PARAMS = len(PARAM_NAMES)

M_REF_KG = 790.0        # mass the rolling-resistance force is referenced to
GAP_LIMIT_S = 1.0       # never form an interval across a telemetry hole
ASSUMED_TOW_F_MAX = 0.20  # peak fractional drag reduction one car length back.
                          # CFD literature, not measured here. Used only as the
                          # *width* of a bound, so an error costs interval width
                          # rather than biasing the answer.
ASSUMED_TOW_TAU_S = 0.8   # decay of the tow with gap


@dataclass(frozen=True)
class Constraints:
    """A x <= b, in the parameters of PARAM_NAMES."""
    A: np.ndarray
    b: np.ndarray
    n_intervals: int
    n_upper: int
    n_lower: int
    n_dropped: int
    drop_reasons: dict

    def __len__(self) -> int:
        return len(self.b)


QUANT_SIGMA_MS = (1.0 / 3.6) / np.sqrt(12.0)  # 1 km/h uniform quantisation

# How many sigma of slack each bound carries. Calibrated, not chosen: swept
# against the *known* true theta on simulator data at three sample rates, and
# set at the point where the truth stops being excluded. Violations of the true
# theta, out of the constraints at that rate:
#
#   n_sigma   3.7 Hz        20 Hz         100 Hz
#      3      1 / 7,325     4 / 41,099    190 / 206,817
#      4      1 / 7,325     0 / 41,099      2 / 206,817
#      5      1 / 7,325     0 / 41,099      0 / 206,817
#
# Five is where 100 Hz reaches zero. The single stubborn 3.7 Hz constraint is a
# braking-onset label, which is the outlier budget's job (see setmem).
# The error is not Gaussian -- uniform quantisation plus sensor noise plus the
# simulator's own Euler error -- so the multiplier is larger than a normal tail
# would suggest, and that is why it is measured rather than derived.
N_SIGMA_DEFAULT = 5.0


def energy_slack(v0, v1, m, speed_sigma_ms: float = QUANT_SIGMA_MS):
    """One sigma of measurement error on the interval's kinetic-energy term, J.

    Speed arrives quantised to 1 km/h, so d(1/2 m v^2) inherits m*v*sigma_v at
    each endpoint. At 70 m/s on 790 kg that is 4.4 kJ per endpoint, 6.3 kJ for
    the pair.

    This is why integrating beats differentiating, and the reason is worth
    stating precisely: the *error* here is fixed by the quantiser, while the
    *signal* -- the energy crossing the interval -- grows with dt. At 200 Hz the
    interval carries 2.3 kJ and the noise is 6.3 kJ, so the constraint says
    nothing. At the 3.7 Hz the public feed actually delivers, the interval
    carries ~94 kJ and the same noise is 7% of it. The formulation gets *better*
    as the feed gets slower, which is the exact opposite of a derivative.
    """
    v0 = np.asarray(v0, float); v1 = np.asarray(v1, float)
    return np.asarray(m, float) * speed_sigma_ms * np.hypot(v0, v1)


def tow_f_max(gap_s, f_max: float = ASSUMED_TOW_F_MAX,
              tau: float = ASSUMED_TOW_TAU_S):
    """Largest plausible drag reduction from running in a wake.

    Bounded, not modelled. The gap is published, so we know *when* a tow can be
    acting even though we do not know how strong it is; carrying it as an
    interval means a car in traffic yields wider bounds instead of a drag area
    that is quietly 5% low.
    """
    g = np.asarray(gap_s, dtype=float)
    out = f_max * np.exp(-np.maximum(g, 0.0) / tau)
    return np.where(np.isfinite(g), out, 0.0)


def fuel_mass(lap_frac, fuel_start: float = 70.0, burn_per_lap: float = 1.15):
    """Fuel aboard, in kg. Linear burn from the declared start load.

    Not a nuisance parameter: the start load is declared and the lap count is
    known, so this is public. It matters because 70 kg on 790 is 9% of mass,
    and mass multiplies the two largest terms in the balance. Holding it
    constant biases early-lap energy by roughly that 9%.
    """
    return np.maximum(fuel_start - burn_per_lap * np.asarray(lap_frac, float), 0.0)


def build_constraints(v, t, z, lap_frac, is_x_mode, in_zone, rpm, throttle,
                      brake, gap_s, regs: RegSet, rho: float,
                      cda_scale=None, eta_d: float = 0.95,
                      manual_override=None, m_published: float = M_REF_KG,
                      gap_limit_s: float = GAP_LIMIT_S,
                      speed_sigma_ms: float = QUANT_SIGMA_MS,
                      n_sigma: float = N_SIGMA_DEFAULT,
                      fuel_start: float = 70.0,
                      fuel_burn_per_lap: float = 1.15) -> Constraints:
    """Turn one car's telemetry into a set of half-spaces on theta.

    Two half-spaces per interval where the car is not braking, one where it is.
    Every bound is widened by `n_sigma` times the measurement error it inherits
    (see `energy_slack`), because the regulation bound is *exactly tight* when a
    car is flat out: the simulator sits on P_ice = 400 kW and P_K = ceiling for
    whole straights, so a bound with no slack is violated by quantisation alone.
    Measured on clean 200 Hz truth with the true theta: 1.8% of upper bounds
    violated with no slack, 0.00% at 3 sigma.

    Braking drops the *lower* bound only: brake force is absent from the model,
    so a decelerating car could be shedding any amount of energy and the
    balance can no longer say the wheels delivered at least so much. The upper
    bound survives untouched, because unmodelled braking can only make the
    left-hand side larger.
    """
    v = np.asarray(v, float); t = np.asarray(t, float)
    z = np.asarray(z, float); lap_frac = np.asarray(lap_frac, float)
    is_x = np.asarray(is_x_mode, bool)
    brake = np.asarray(brake, float) > 0.5   # per sample, not per interval
    scale = np.ones_like(v) if cda_scale is None else np.asarray(cda_scale, float)
    mo = np.zeros_like(v, bool) if manual_override is None else np.asarray(manual_override, bool)

    k0, k1 = np.arange(len(v) - 1), np.arange(1, len(v))
    dt = t[k1] - t[k0]

    reasons = {}
    ok = np.isfinite(dt) & (dt > 1e-6) & (dt <= gap_limit_s)
    reasons["gap_or_bad_dt"] = int((~ok).sum())
    finite = np.isfinite(v[k0]) & np.isfinite(v[k1]) & np.isfinite(z[k0]) & np.isfinite(z[k1])
    reasons["non_finite"] = int((~finite).sum())
    ok &= finite
    # An interval that straddles an aero-mode change has two different drag
    # states inside it and cannot be attributed to either. Dropping them costs
    # a few percent of samples; keeping them puts the X/Z difference -- the
    # thing being identified -- into the residual.
    same_mode = is_x[k0] == is_x[k1]
    reasons["mode_change"] = int((~same_mode).sum())
    ok &= same_mode

    k0, k1 = k0[ok], k1[ok]
    dt = dt[ok]
    v0, v1 = v[k0], v[k1]
    vbar = 0.5 * (v0 + v1)
    # INT v^3 dt by trapezoid. The drag term is the one place a nonlinearity in
    # v survives, and v^3 over a 0.24 s interval at 5 m/s^2 changes by 20%, so
    # the endpoint average is worth having over a midpoint value.
    v3bar = 0.5 * (v0 ** 3 + v1 ** 3)
    m_nom = m_published + fuel_mass(lap_frac[k0], fuel_start, fuel_burn_per_lap)

    d_ekin = 0.5 * (v1 ** 2 - v0 ** 2)          # x m
    d_epot = G * (z[k1] - z[k0])                # x m
    known = (d_ekin + d_epot) * m_nom
    coef_dm = d_ekin + d_epot                   # dm multiplies the same bracket

    scale_i = 0.5 * (scale[k0] + scale[k1])
    drag_full = dt * 0.5 * rho * scale_i * v3bar
    f_max = tow_f_max(gap_s[k0] if gap_s is not None else np.full(len(k0), np.nan))
    drag_lo = drag_full * (1.0 - f_max)         # strongest permissible tow
    drag_hi = drag_full                         # clear air

    rr = dt * vbar * (m_nom / M_REF_KG)         # x F_rr

    # Regulation bounds on what the wheels can have received. Each bound is
    # evaluated at whichever endpoint makes it *loosest*, so the half-space is
    # guaranteed to contain the truth for any trajectory inside the interval.
    # This is not fussiness: the deployment cap falls 19.4 kW per m/s through
    # the taper, and a car accelerating through 338 km/h deploys at the cap for
    # its entry speed. Using the midpoint instead put the true theta outside
    # 3,884 upper bounds on clean 200 Hz truth -- by only 24 J at worst, but a
    # set-membership method that excludes the truth is not conservative, it is
    # wrong. Lowest speed gives the highest taper; highest revs and throttle
    # give the largest ICE allowance.
    v_cap = np.minimum(v0, v1)
    rpm_hi = np.maximum(rpm[k0], rpm[k1])
    thr_hi = np.maximum(throttle[k0], throttle[k1])
    in_zone_any = np.asarray(in_zone, bool)[k0] | np.asarray(in_zone, bool)[k1]
    pk_lo, pk_hi = p_k_bounds(v_cap, in_zone_any, rpm_hi, thr_hi, regs,
                              mo[k0] | mo[k1])
    pice_hi = p_ice_max(rpm_hi, thr_hi, regs)
    rhs_hi = dt * eta_d * (pice_hi + pk_hi)
    rhs_lo = dt * eta_d * (0.0 + pk_lo)
    slack = n_sigma * energy_slack(v0, v1, m_nom, speed_sigma_ms)

    zeros = np.zeros(len(k0))
    # UPPER: known + drag*CdA + rr*F_rr + coef_dm*dm <= rhs_hi.
    # Use the *smallest* admissible drag coefficient so the constraint cannot
    # exclude the true theta for any tow strength in range.
    A_up = np.column_stack([
        np.where(is_x[k0], drag_lo, zeros), np.where(is_x[k0], zeros, drag_lo),
        rr, coef_dm])
    b_up = rhs_hi - known + slack

    # LOWER: -(...) <= -(rhs_lo), and only where the brakes are off.
    # An interval is braking if *either* endpoint is. Taking only the opening
    # sample keeps the lower bound across a braking onset, where the brakes came
    # on partway through: measured on clean 200 Hz truth, that put the true
    # theta outside 371 lower bounds by up to 17.3 kJ, all of them at an onset.
    # Either-endpoint is the conservative reading, and conservative is the only
    # safe direction for a set that must contain the truth.
    free = ~(brake[k0] | brake[k1])
    A_dn = -np.column_stack([
        np.where(is_x[k0], drag_hi, zeros), np.where(is_x[k0], zeros, drag_hi),
        rr, coef_dm])[free]
    b_dn = -(rhs_lo - known - slack)[free]
    reasons["braking_lower_dropped"] = int((~free).sum())

    return Constraints(
        A=np.vstack([A_up, A_dn]), b=np.concatenate([b_up, b_dn]),
        n_intervals=len(k0), n_upper=len(b_up), n_lower=len(b_dn),
        n_dropped=int((~ok).sum()), drop_reasons=reasons)


def residual_power(v, t, z, lap_frac, theta, is_x_mode, rho, cda_scale=None,
                   m_published: float = M_REF_KG):
    """Interval-averaged wheel power implied by theta, in W.

    The quantity a regime classifier and a residual GP both work on. Returned
    per interval, so it lines up with `build_constraints`.
    """
    v = np.asarray(v, float); t = np.asarray(t, float); z = np.asarray(z, float)
    is_x = np.asarray(is_x_mode, bool)
    scale = np.ones_like(v) if cda_scale is None else np.asarray(cda_scale, float)
    cda_x, cda_z, f_rr, dm = theta
    dt = np.diff(t)
    v0, v1 = v[:-1], v[1:]
    vbar, v3bar = 0.5 * (v0 + v1), 0.5 * (v0 ** 3 + v1 ** 3)
    m = m_published + dm + fuel_mass(np.asarray(lap_frac, float)[:-1])
    cda = np.where(is_x[:-1], cda_x, cda_z) * 0.5 * (scale[:-1] + scale[1:])
    d_e = m * (0.5 * (v1 ** 2 - v0 ** 2) + G * (z[1:] - z[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dt > 1e-9,
                        d_e / dt + 0.5 * rho * cda * v3bar
                        + f_rr * vbar * (m / M_REF_KG), np.nan)


# --------------------------------------------------------- window aggregation
def build_window_constraints(v, t, z, lap_frac, is_x_mode, in_zone, rpm,
                             throttle, brake, gap_s, regs: RegSet, rho: float,
                             window_s: float = 4.0, cda_scale=None,
                             eta_d: float = 0.95, manual_override=None,
                             m_published: float = M_REF_KG,
                             gap_limit_s: float = GAP_LIMIT_S,
                             speed_sigma_ms: float = QUANT_SIGMA_MS,
                             n_sigma: float = N_SIGMA_DEFAULT,
                             fuel_start: float = 70.0,
                             fuel_burn_per_lap: float = 1.15,
                             e_harvest_lap: float = E_HARVEST_LAP) -> Constraints:
    """The same balance, summed over a window. This is what makes it identify.

    Per-interval constraints are robust and nearly useless. The reason is that
    an intersection keeps the single *tightest* bound rather than averaging, so
    noise never cancels: with the slack honestly set at 5 sigma, one interval's
    31 kJ of slack sits against 36.7 kJ of drag signal, and the identified range
    for CdA_X came out as the entire prior box [0.30, 3.00].

    Summing consecutive intervals fixes it, and the reason is worth stating
    because it is the whole trick: the kinetic and potential terms *telescope*.

        sum_k [ 1/2 m (v_{k+1}^2 - v_k^2) ] = 1/2 m (v_end^2 - v_start^2)

    Only the two endpoints survive, so the measurement error of a window is the
    same 6.3 kJ as that of a single interval -- while the drag term, the rolling
    term and the regulation's energy budget all grow linearly with the window.
    Signal grows, noise does not.

    A window spanning an aero-mode change is not dropped: the drag sum splits
    into an X part and a Z part, and both are parameters, so a mixed window
    simply constrains a combination of them.

    Windows are non-overlapping, which keeps their errors independent and keeps
    the LP small enough to re-solve per lap.
    """
    v = np.asarray(v, float); t = np.asarray(t, float); z = np.asarray(z, float)
    lap_frac = np.asarray(lap_frac, float)
    is_x = np.asarray(is_x_mode, bool)
    brake = np.asarray(brake, float) > 0.5
    scale = np.ones_like(v) if cda_scale is None else np.asarray(cda_scale, float)
    mo = np.zeros_like(v, bool) if manual_override is None else np.asarray(manual_override, bool)
    in_zone = np.asarray(in_zone, bool)

    dt = np.diff(t)
    v0, v1 = v[:-1], v[1:]
    vbar = 0.5 * (v0 + v1)
    v3bar = 0.5 * (v0 ** 3 + v1 ** 3)
    scale_i = 0.5 * (scale[:-1] + scale[1:])
    good = (np.isfinite(dt) & (dt > 1e-6) & (dt <= gap_limit_s)
            & np.isfinite(v0) & np.isfinite(v1) & np.isfinite(z[:-1]) & np.isfinite(z[1:]))

    v_cap = np.minimum(v0, v1)
    rpm_hi = np.maximum(rpm[:-1], rpm[1:])
    thr_hi = np.maximum(throttle[:-1], throttle[1:])
    zone_any = in_zone[:-1] | in_zone[1:]
    mo_any = mo[:-1] | mo[1:]
    pk_lo, pk_hi = p_k_bounds(v_cap, zone_any, rpm_hi, thr_hi, regs, mo_any)
    pice_hi = p_ice_max(rpm_hi, thr_hi, regs)

    # per-interval contributions, zeroed where the interval is unusable so a
    # window containing a hole contributes nothing rather than nonsense
    g = good.astype(float)
    drag_x = g * dt * 0.5 * rho * scale_i * v3bar * is_x[:-1]
    drag_z = g * dt * 0.5 * rho * scale_i * v3bar * (~is_x[:-1])
    rr_i = g * dt * vbar
    bud_hi = g * dt * eta_d * (pice_hi + pk_hi)
    bud_lo = g * dt * eta_d * pk_lo
    pice_sum = g * dt * pice_hi

    rows_A, rows_b, n_up, n_dn = [], [], 0, 0
    reasons = {"window_had_hole": 0, "braking_lower_dropped": 0}
    n = len(v)
    starts = []
    i = 0
    while i < n - 1:
        j = i
        t_end = t[i] + window_s
        while j < n - 1 and t[j + 1] <= t_end:
            j += 1
        if j <= i:
            j = i + 1
        starts.append((i, j))
        i = j
    for i, j in starts:
        sl = slice(i, j)
        if not good[sl].all():
            reasons["window_had_hole"] += 1
            continue
        m_nom = m_published + fuel_mass(0.5 * (lap_frac[i] + lap_frac[j]),
                                        fuel_start, fuel_burn_per_lap)
        d_e = 0.5 * (v[j] ** 2 - v[i] ** 2) + G * (z[j] - z[i])
        known = d_e * m_nom
        f_max = float(np.max(tow_f_max(np.asarray(gap_s, float)[sl]))) if gap_s is not None else 0.0
        dx, dz = drag_x[sl].sum(), drag_z[sl].sum()
        row = np.array([dx, dz, rr_i[sl].sum() * (m_nom / M_REF_KG), d_e])
        slack = n_sigma * float(energy_slack(v[i], v[j], m_nom, speed_sigma_ms))

        # The integral caps, and they are what give the lower bound any teeth
        # at all. Instantaneously the MGU-K may recover 250 kW, so over an 8 s
        # window the rules permit -2 MJ of store-side flow against 1.09 MJ of
        # drag: the constraint is then satisfied by CdA = 0 and the lower bound
        # sits on its prior edge, which is exactly what was measured before this
        # block existed. But a car cannot harvest 2 MJ every 8 s for a race --
        # the store holds 4 MJ and the per-lap harvest allowance is 7 MJ. Taking
        # whichever cap binds turns "a car could always be recovering" into a
        # quantity with a budget.
        laps_spanned = max(lap_frac[j] - lap_frac[i], 1e-9)
        harv_cap = e_harvest_lap * max(laps_spanned, 1.0)
        dep_cap = E_STORE_MAX + e_harvest_lap * laps_spanned
        pk_lo_tot = max(bud_lo[sl].sum() / eta_d, -harv_cap) * eta_d
        pk_hi_tot = min((bud_hi[sl].sum() / eta_d) - pice_sum[sl].sum(),
                        dep_cap) * eta_d

        # upper: loosest drag (strongest admissible tow)
        rows_A.append(np.array([dx * (1 - f_max), dz * (1 - f_max), row[2], row[3]]))
        rows_b.append(eta_d * pice_sum[sl].sum() + pk_hi_tot - known + slack)
        n_up += 1
        if not brake[i:j + 1].any():
            rows_A.append(-row)
            rows_b.append(-(pk_lo_tot - known - slack))
            n_dn += 1
        else:
            reasons["braking_lower_dropped"] += 1

    if not rows_A:
        return Constraints(A=np.zeros((0, N_PARAMS)), b=np.zeros(0), n_intervals=0,
                           n_upper=0, n_lower=0, n_dropped=len(starts),
                           drop_reasons=reasons)
    order = np.argsort([0] * n_up + [1] * n_dn, kind="stable")
    A = np.array(rows_A)[order]
    b = np.array(rows_b)[order]
    return Constraints(A=A, b=b, n_intervals=len(starts), n_upper=n_up,
                       n_lower=n_dn, n_dropped=reasons["window_had_hole"],
                       drop_reasons=reasons)
