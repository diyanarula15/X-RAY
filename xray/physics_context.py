"""What the force model is allowed to know about conditions, bundled once.

Deliberately two fields. `PhysicsContext` is the *physical* situation the car is
in right now -- the air it is driving through and the tyres it is driving on.
Pit strategy is not a force and does not belong here: a planned stop changes
which decisions are worth making, not the acceleration of the car this
millisecond. `PitContext` therefore lives in `xray.stint` and reaches the DP,
never `vehicle.step`.

Passing this object is optional everywhere. When it is absent the physics falls
back to the P0 behaviour exactly -- `params.rho`, no wind, no downforce term --
so that a P0 regression test measures P0 and a P1 test has to ask for P1.
"""
from __future__ import annotations

from dataclasses import dataclass

from .environment import EnvironmentalState


@dataclass(frozen=True)
class PhysicsContext:
    environment: EnvironmentalState
    tyre: "object | None" = None          # TyreState once xray.tyres is wired
    w_parallel_ms: float = 0.0            # already projected; see wind_parallel
    wind_source: str = "none"
    downforce_factor: float = 1.0         # wake retention, 1.0 = clean air

    @property
    def rho(self) -> float:
        return float(self.environment.rho)
