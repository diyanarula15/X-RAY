"""Two-car race simulator. Produces the ground truth -- and nothing else may
read its internals except the scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import E_STORE_MAX, MOM_DEPLOYMENT_ALLOWANCE_J, MOM_GAP_S
from .overtake import ASSUMED_BRIEF_COEFFS, p_pass
from .environment import state_from_summary
from .physics_context import PhysicsContext
from .policy import DeploymentPolicy, get_policy
from .track import Track, circuit_sigma
from .vehicle import (CarState, VehicleParams, cornering_grip_from_downforce,
                      set_car_mass, step)

LEADER, FOLLOWER = "LEADER", "FOLLOWER"


@dataclass(frozen=True)
class Event:
    t: float
    lap: int
    kind: str  # "overtake" | "mom_grant" | "lap"
    car: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CarTrace:
    s: np.ndarray          # m, distance along the current lap
    s_total: np.ndarray    # m, cumulative distance (defines race order)
    v: np.ndarray          # m/s
    E: np.ndarray          # J, energy store  <-- the hidden state
    P_mguk: np.ndarray     # W, store-side deployment
    P_ice: np.ndarray      # W
    harvest: np.ndarray    # W, store-side recovery
    fuel: np.ndarray       # kg
    lap: np.ndarray        # int
    aero_mode: np.ndarray  # "straight" | "corner"
    regime: np.ndarray     # "accel" | "brake" | "corner"
    deployed_lap: np.ndarray   # J, per-lap totals indexed by lap
    harvested_lap: np.ndarray
    mom_lap: np.ndarray        # J, Manual Override energy credited that lap
    e_lap_open: np.ndarray     # J, store at each lap boundary
    lap_time: np.ndarray       # s


@dataclass(frozen=True)
class GroundTruth:
    t: np.ndarray
    cars: dict[str, CarTrace]
    gap_s: np.ndarray          # s, gap from the trailing car to the leading car
    order: np.ndarray          # car id that is ahead at each sample
    events: list[Event]
    track: Track
    dt: float
    n_laps: int
    policies: dict[str, str]
    seed: int


class Simulator:
    def __init__(self, cfg: dict, track: Track | None = None,
                 policies: dict[str, DeploymentPolicy] | None = None,
                 seed: int = 42, n_laps: int | None = None):
        self.cfg = cfg
        self.track = track if track is not None else circuit_sigma()
        self.params = VehicleParams.from_config(cfg)
        set_car_mass(self.params.mass_car)
        self.dt = float(cfg["sim"]["dt"])
        self.n_laps = int(n_laps if n_laps is not None else cfg["sim"]["n_laps"])
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.tow = cfg["tow"]
        # Config migration, resolved once. `dirty_air_downforce_loss` is the P1
        # key; `dirty_air_grip_loss` is the P0 one and is still honoured so an
        # older config does not silently lose its wake entirely. Exactly one of
        # them is in force and `_cla_active` says which physics is running.
        self._legacy_grip_loss = float(self.tow.get("dirty_air_grip_loss", 0.0))
        self._dirty_air_loss = float(self.tow.get("dirty_air_downforce_loss",
                                                  self._legacy_grip_loss))
        params0 = VehicleParams.from_config(cfg)
        self._cla_active = params0.cla_corner > 0.0
        self._rho = float(params0.rho)
        # Circuit Sigma has no physical heading, so no wind can be projected on
        # to it. Said once, here, rather than implied by a zero.
        self._wind_source = "unavailable_synthetic_track_has_no_heading"
        self._env = state_from_summary(
            {"rho": self._rho, "wind_speed_ms": 0.0, "wind_dir_deg": 0.0},
            source="config_synthetic_constant")
        if policies is None:
            policies = {LEADER: get_policy(cfg["sim"]["leader_policy"]),
                        FOLLOWER: get_policy(cfg["sim"]["follower_policy"])}
        self.policies = policies

    # ------------------------------------------------------------------ setup
    def _initial_states(self) -> dict[str, CarState]:
        L = self.track.length
        v0 = 85.0  # m/s, a representative speed on the pit straight
        gap0 = float(self.cfg["sim"]["start_gap_s"])
        e0 = float(self.cfg["sim"]["e_start_frac"]) * E_STORE_MAX
        lead = CarState(s=0.0, s_total=0.0, v=v0, E=e0, fuel=self.params.fuel_start, lap=0)
        back = gap0 * v0
        foll = CarState(s=L - back, s_total=-back, v=v0, E=e0,
                        fuel=self.params.fuel_start, lap=-1)
        return {LEADER: lead, FOLLOWER: foll}

    # -------------------------------------------------------------------- run
    def run(self) -> GroundTruth:
        tr, params, dt = self.track, self.params, self.dt
        st = self._initial_states()
        ids = [LEADER, FOLLOWER]

        max_steps = int(self.n_laps * 110.0 / dt) + 2000
        rec = {c: {k: np.zeros(max_steps, dtype=float) for k in
                   ("s", "s_total", "v", "E", "P_mguk", "P_ice", "harvest", "fuel")}
               for c in ids}
        rec_lap = {c: np.zeros(max_steps, dtype=np.int32) for c in ids}
        rec_aero = {c: np.zeros(max_steps, dtype=np.uint8) for c in ids}
        rec_reg = {c: np.zeros(max_steps, dtype=np.uint8) for c in ids}
        gap_arr = np.zeros(max_steps)
        order_arr = np.zeros(max_steps, dtype=np.uint8)
        reg_code = {"accel": 0, "brake": 1, "corner": 2}

        per_lap = {c: {"dep": np.zeros(self.n_laps + 2), "har": np.zeros(self.n_laps + 2),
                       "mom": np.zeros(self.n_laps + 2), "open": np.full(self.n_laps + 2, np.nan),
                       "time": np.full(self.n_laps + 2, np.nan)} for c in ids}
        lap_start_t = {c: 0.0 for c in ids}
        pending_mom = {c: 0.0 for c in ids}
        mom_lap_granted = {c: -1 for c in ids}
        resolved: set[tuple[int, str]] = set()
        order_ahead = LEADER  # race classification; only a resolved pass changes it
        # peak speed reached on each overtake-zone straight, per car -- this is
        # the "speed at the end of the straight" the pass model wants, measured
        # at the braking point rather than at the apex
        zone_peak = {c: {z.name: 0.0 for z in tr.zones} for c in ids}
        zone_peak_gap = {c: {z.name: np.inf for z in tr.zones} for c in ids}
        ALONGSIDE_M = 5.0     # how far a car may draw level before the pass resolves
        events: list[Event] = []
        for c in ids:
            if st[c].lap >= 0:
                per_lap[c]["open"][st[c].lap] = st[c].E

        t = 0.0
        n = 0
        while n < max_steps:
            ahead = order_ahead
            behind = FOLLOWER if ahead == LEADER else LEADER
            d_m = st[ahead].s_total - st[behind].s_total
            gap = d_m / max(st[behind].v, 1.0)

            # One decay, two physically distinct consequences. The tow takes
            # drag off the following car; the dirty air takes DOWNFORCE off it.
            # Previously both came out as one number that multiplied apex speed,
            # which made a wake effect measured in m/s and hid which of the two
            # was doing the work.
            decay = np.exp(-max(gap, 0.0) / self.tow["tau_s"])
            tow_factor = 1.0 - self.tow["cda_reduction"] * decay
            downforce_factor = 1.0 - self._dirty_air_loss * decay
            if self._cla_active:
                # Corner speed follows from the lost load, not from a config
                # number that happens to be in speed units.
                grip = cornering_grip_from_downforce(
                    st[behind].v, params.cla_corner, self._rho,
                    st[behind].mass, downforce_factor)
            else:
                grip = 1.0 - self._legacy_grip_loss * decay

            s_prev_all = {}
            for c in ids:
                s_prev = st[c].s
                s_prev_all[c] = s_prev
                is_behind = (c == behind)
                laps_left = self.n_laps - max(st[c].lap, 0)
                gap_ahead = gap if is_behind else None
                gap_behind = gap if not is_behind else None
                pt = tr.point(st[c].s)
                lap_before = st[c].lap
                demand = self.policies[c].demand(
                    tr, st[c].s, st[c].v, st[c].E, gap_ahead, gap_behind,
                    laps_left, is_corner=pt[2], lap=st[c].lap)
                ctx = None
                if self._cla_active:
                    ctx = PhysicsContext(
                        environment=self._env, tyre=None, w_parallel_ms=0.0,
                        wind_source=self._wind_source,
                        downforce_factor=downforce_factor if is_behind else 1.0)
                out = step(tr, st[c], params, demand, dt,
                           tow_factor=tow_factor if is_behind else 1.0,
                           grip=grip if is_behind else 1.0,
                           physics_context=ctx)
                # Feed the DELIVERED power back, not the requested power: a
                # budgeted policy must account against what the integrator
                # actually put through the MGU-K after the taper and the store
                # limit clipped the request.
                # `st[c].lap` is POST-step here -- step() mutates then records
                # -- and `p_mguk` is the power applied DURING the step, so the
                # lap read before the call is the one that spent the energy.
                self.policies[c].note_deployed(
                    lap_before, float(out["p_mguk"]) * dt)

                r = rec[c]
                r["s"][n] = st[c].s
                r["s_total"][n] = st[c].s_total
                r["v"][n] = st[c].v
                r["E"][n] = st[c].E
                r["P_mguk"][n] = out["p_mguk"]
                r["P_ice"][n] = out["p_ice"]
                r["harvest"][n] = out["harvest"]
                r["fuel"][n] = st[c].fuel
                rec_lap[c][n] = st[c].lap
                rec_aero[c][n] = 1 if out["aero"] == "corner" else 0
                rec_reg[c][n] = reg_code[out["regime"]]

                # ------------------------------------------- zone peak speed
                zc = tr.zone_at(st[c].s)
                if zc is not None and st[c].s < zc.s_straight_end:
                    if _crossed(s_prev, st[c].s, zc.s_straight_start, tr.length):
                        zone_peak[c][zc.name] = 0.0
                    if st[c].v > zone_peak[c][zc.name]:
                        zone_peak[c][zc.name] = st[c].v
                        zone_peak_gap[c][zc.name] = gap

                # ------------------------------------------- MOM detection
                if is_behind:
                    for s_d in tr.detection_points:
                        if (_crossed(s_prev, st[c].s, s_d, tr.length) and gap <= MOM_GAP_S
                                and mom_lap_granted[c] != st[c].lap):
                            mom_lap_granted[c] = st[c].lap
                            pending_mom[c] = MOM_DEPLOYMENT_ALLOWANCE_J
                            events.append(Event(t, max(st[c].lap, 0), "mom_grant", c,
                                                {"detection_s": s_d, "gap_s": gap}))

                # ------------------------------------------- lap boundary
                if out["crossed_line"]:
                    lap_done = st[c].lap - 1
                    if lap_done >= 0:
                        per_lap[c]["dep"][lap_done] = st[c].deployed_lap
                        per_lap[c]["har"][lap_done] = st[c].harvested_lap
                        per_lap[c]["time"][lap_done] = t - lap_start_t[c]
                    lap_start_t[c] = t
                    st[c].deployed_lap = 0.0
                    st[c].harvested_lap = 0.0
                    # Manual Override grants a per-lap LEGAL ALLOWANCE, not a
                    # battery top-up. It raises this lap's recharge ceiling from
                    # 7.0 to 7.5 MJ and unlocks the Overtake power curve (full
                    # 350 kW to 337.5 km/h instead of 290). The old code ran
                    # `st.E += min(0.5 MJ, room)`, which created energy out of a
                    # regulation; the per-lap balance test only closed because it
                    # added the same mom term to both sides. The allowance is
                    # assigned, not accumulated: eligibility is decided afresh at
                    # each lap's detection point.
                    st[c].manual_overtake_allocation_j = 0.0
                    st[c].overtake_active = False
                    if pending_mom[c] > 0.0 and st[c].lap < self.n_laps + 1:
                        st[c].manual_overtake_allocation_j = pending_mom[c]
                        st[c].overtake_active = True
                        per_lap[c]["mom"][st[c].lap] = pending_mom[c]
                        pending_mom[c] = 0.0
                    if st[c].lap < len(per_lap[c]["open"]):
                        per_lap[c]["open"][st[c].lap] = st[c].E
                    events.append(Event(t, st[c].lap, "lap", c, {}))

            # ------------------------------- alongside clamp (1-D stand-in for
            # two cars side by side: longitudinal position is held, speed is
            # left free, and the pass is resolved at the next corner entry)
            if st[behind].s_total > st[ahead].s_total + ALONGSIDE_M:
                over = st[behind].s_total - (st[ahead].s_total + ALONGSIDE_M)
                st[behind].s_total -= over
                st[behind].s -= over
                if st[behind].s < 0.0:
                    st[behind].s += tr.length
                    st[behind].lap -= 1

            # ----------------------------------------------- overtake resolve
            zb = None
            for z in tr.zones:
                if _crossed(s_prev_all[behind], st[behind].s, z.s_straight_end, tr.length):
                    zb = z
                    break
            if zb is not None:
                key = (st[behind].lap, zb.name)
                if key not in resolved:
                    resolved.add(key)
                    dv = zone_peak[behind][zb.name] - zone_peak[ahead][zb.name]
                    gap_brake = min(zone_peak_gap[behind][zb.name], gap)
                    if gap_brake <= 0.9 and dv > 0.0:
                        # Pinned to the brief anchors, not the module default.
                        # `dv` here is the OBSERVED peak-speed difference between
                        # two cars at this zone, not the energy-implied delta_v
                        # the decision path feeds `p_pass`; the two quantities
                        # have different ranges, so one anchor cannot serve both.
                        # Every golden trace was also generated against these.
                        p = p_pass(dv, max(gap_brake, 0.0), zb,
                                   ASSUMED_BRIEF_COEFFS)
                        success = bool(self.rng.random() < p)
                        events.append(Event(t, max(st[behind].lap, 0), "overtake", behind,
                                            {"zone": zb.name, "p": p, "delta_v": dv,
                                             "gap_s": gap_brake, "success": success}))
                        if success:
                            shift = (st[ahead].s_total + 6.0) - st[behind].s_total
                            _shift_car(st[behind], shift, tr.length)
                            st[ahead].v *= 0.97  # compromised line through the apex
                            order_ahead = behind
                        else:
                            # the lunge fails: yield the place back and lose time
                            shift = (st[ahead].s_total - 10.0) - st[behind].s_total
                            _shift_car(st[behind], shift, tr.length)
                            st[behind].v = min(st[behind].v, st[ahead].v)

            gap_arr[n] = gap
            order_arr[n] = 0 if order_ahead == LEADER else 1
            t += dt
            n += 1
            if min(st[c].lap for c in ids) >= self.n_laps:
                break

        sl = slice(0, n)
        aero_names = np.array(["straight", "corner"])
        reg_names = np.array(["accel", "brake", "corner"])
        cars = {}
        for c in ids:
            r = rec[c]
            cars[c] = CarTrace(
                s=r["s"][sl].copy(), s_total=r["s_total"][sl].copy(), v=r["v"][sl].copy(),
                E=r["E"][sl].copy(), P_mguk=r["P_mguk"][sl].copy(),
                P_ice=r["P_ice"][sl].copy(), harvest=r["harvest"][sl].copy(),
                fuel=r["fuel"][sl].copy(), lap=rec_lap[c][sl].copy(),
                aero_mode=aero_names[rec_aero[c][sl]], regime=reg_names[rec_reg[c][sl]],
                deployed_lap=per_lap[c]["dep"][:self.n_laps].copy(),
                harvested_lap=per_lap[c]["har"][:self.n_laps].copy(),
                mom_lap=per_lap[c]["mom"][:self.n_laps].copy(),
                e_lap_open=per_lap[c]["open"][:self.n_laps + 1].copy(),
                lap_time=per_lap[c]["time"][:self.n_laps].copy())

        return GroundTruth(
            t=np.arange(n) * dt, cars=cars, gap_s=gap_arr[sl].copy(),
            order=order_arr[sl].copy(), events=events, track=tr, dt=dt,
            n_laps=self.n_laps, policies={c: self.policies[c].name for c in ids},
            seed=self.seed)


def _shift_car(st: CarState, shift: float, length: float) -> None:
    """Move a car along the track, keeping s / s_total / lap consistent."""
    st.s_total += shift
    st.s += shift
    while st.s >= length:
        st.s -= length
        st.lap += 1
    while st.s < 0.0:
        st.s += length
        st.lap -= 1


def _crossed(s_prev: float, s_now: float, s_mark: float, length: float) -> bool:
    """Did the car pass ``s_mark`` this step, wrap-safe?"""
    if s_now >= s_prev:
        return s_prev < s_mark <= s_now
    return s_mark > s_prev or s_mark <= s_now


def run_sim(cfg: dict, seed: int = 42, **kw) -> GroundTruth:
    return Simulator(cfg, seed=seed, **kw).run()
