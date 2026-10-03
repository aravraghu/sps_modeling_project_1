"""Types that cross module boundaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One feasible way to serve one endpoint: a vehicle, a route, and a load.

    ``value`` is this endpoint's contribution to the world's coverage quality Q, i.e.
    (demand_j / total demand) * q_j, so the chosen candidates' values sum to Q directly.

    ``tokens`` is dispatch + fuel. ``fuel`` is tracked separately because the score's
    efficiency multiplier depends on fuel alone, so two candidates with equal total cost
    but a different dispatch/fuel split are genuinely different (see METHOD.md).
    """

    endpoint: int
    vehicle: str
    checkpoints: tuple[int, ...]
    load_kg: float
    omit_load: bool  # True when load == demand, so the plan JSON can omit the field
    value: float
    quality: float
    tokens: float
    fuel: float
    distance_m: float
    travel_days: float

    @property
    def dispatch(self) -> float:
        return self.tokens - self.fuel
