"""Starter instance generator and reference scoring helpers for Prompt 1.

Coordinates and distances use SI units (meters) (as they should). One model time unit is one day. The default seed (67) is the public test case. Organizers may grade on additional screened seeds by changing ``--seed``. This file is a reference model and scorer, not your final solution! You need to add extra code!!!
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Global challenge parameters
# ---------------------------------------------------------------------------

PUBLIC_SEED = 67
INITIAL_TOKENS = 1000.0
CHARACTERISTIC_MAP_LENGTH_M = 1.0e6  # 1000 km
N_CHECKPOINTS = 50
N_ENDPOINTS = 50
MIN_CHECKPOINT_TIME_DAYS = 1.0

FOOD_RETAINED_PER_DAY = 0.95
NUTRITION_DECAY_RATE_PER_DAY = 0.02
BANDIT_BASE_RATE_PER_DAY = 0.10
BANDIT_DETOUR_SUPPRESSION = 3.0
ATTACK_LOSS_FRACTION = 0.25
DETOUR_WEAR_RATE = 0.20

FINAL_SCORE_WEIGHTS = {
    "disk": 1.0,
    "sphere": 2.0,
    "torus": 3.0,
    "klein": 5.0,
}


@dataclass(frozen=True)
class VehicleSpec:
    dispatch_cost_tokens: float
    fuel_cost_tokens_per_unit: float
    mileage_m_per_fuel_unit: float
    speed_m_per_day: float
    capacity_kg: float
    max_leg_m: float
    lifetime_days: float
    can_cross_ocean: bool
    can_serve_island: bool


VEHICLES: dict[str, VehicleSpec] = {
    "horse": VehicleSpec(
        dispatch_cost_tokens=25.0,
        fuel_cost_tokens_per_unit=5.0,
        mileage_m_per_fuel_unit=100_000.0,
        speed_m_per_day=50_000.0,
        capacity_kg=100.0,
        max_leg_m=50_000.0,
        lifetime_days=3650.0,  # 10 years
        can_cross_ocean=False,
        can_serve_island=False,
    ),
    "truck": VehicleSpec(
        dispatch_cost_tokens=100.0,
        fuel_cost_tokens_per_unit=10.0,
        mileage_m_per_fuel_unit=300_000.0,
        speed_m_per_day=500_000.0,
        capacity_kg=1200.0,
        max_leg_m=750_000.0,
        lifetime_days=9125.0,  # 25 years
        can_cross_ocean=False,
        can_serve_island=False,
    ),
    "jet": VehicleSpec(
        dispatch_cost_tokens=300.0,
        fuel_cost_tokens_per_unit=100.0,
        mileage_m_per_fuel_unit=1_500_000.0,
        speed_m_per_day=5_000_000.0,
        capacity_kg=5000.0,
        max_leg_m=3_000_000.0,
        lifetime_days=9125.0,  # 25 years
        can_cross_ocean=True,
        can_serve_island=True,
    ),
}


@dataclass
class MapInstance:
    name: str
    scale_m: float
    start: np.ndarray
    checkpoints: np.ndarray
    endpoints: np.ndarray
    endpoint_demand_kg: np.ndarray
    endpoint_island_mask: np.ndarray
    ocean_centers: np.ndarray
    ocean_radii_m: np.ndarray

    def validate(self) -> None:
        if self.name not in FINAL_SCORE_WEIGHTS:
            raise ValueError(f"Unknown manifold: {self.name}")
        if len(self.checkpoints) != N_CHECKPOINTS:
            raise ValueError("Unexpected number of checkpoints")
        if len(self.endpoints) != N_ENDPOINTS:
            raise ValueError("Unexpected number of endpoints")
        if len(self.endpoint_demand_kg) != N_ENDPOINTS:
            raise ValueError("Demand vector has wrong length")
        if len(self.endpoint_island_mask) != N_ENDPOINTS:
            raise ValueError("Island mask has wrong length")
        if len(self.ocean_centers) != len(self.ocean_radii_m):
            raise ValueError("Ocean center/radius count mismatch")


@dataclass(frozen=True)
class DeliveryMetrics:
    feasible: bool
    reason: str
    endpoint: int
    vehicle: str
    load_kg: float
    route_distance_m: float
    travel_time_days: float
    dispatch_tokens: float
    fuel_tokens: float
    total_tokens: float
    attack_probability: float
    food_retained_fraction: float
    nutrition_fraction: float
    wear_fraction: float
    endpoint_quality_fraction: float


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sample_disk(rng: np.random.Generator, n: int, radius_m: float) -> np.ndarray:
    # sqrt(U) is required for uniform area density.
    r = radius_m * np.sqrt(rng.random(n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def _sample_sphere(rng: np.random.Generator, n: int, radius_m: float) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return radius_m * x


def _sample_flat_square(rng: np.random.Generator, n: int, side_m: float) -> np.ndarray:
    return rng.uniform(0.0, side_m, size=(n, 2))


def _torus_delta(a: np.ndarray, b: np.ndarray, side_m: float) -> np.ndarray:
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return (d + side_m / 2.0) % side_m - side_m / 2.0


def _klein_best_lift(a: np.ndarray, b: np.ndarray, side_m: float) -> np.ndarray:
    """Return the closest image of b to a in the Euclidean covering space.

    We use the square fundamental domain with identifications
      (x, y) ~ (x + L, y),
      (x, y) ~ (-x, y + L).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    best = None
    best_d2 = math.inf
    for n in range(-2, 3):
        reflected_x = ((-1) ** n) * b[0]
        y = b[1] + n * side_m
        for m in range(-2, 3):
            candidate = np.array([reflected_x + m * side_m, y], dtype=float)
            d2 = float(np.dot(candidate - a, candidate - a))
            if d2 < best_d2:
                best_d2 = d2
                best = candidate
    assert best is not None
    return best


def _reduce_klein_cover(point: np.ndarray, side_m: float) -> np.ndarray:
    x, y = map(float, point)
    n = math.floor(y / side_m)
    y -= n * side_m
    if n % 2:
        x = -x
    x %= side_m
    return np.array([x, y], dtype=float)


def geodesic_distance(instance: MapInstance, a: Sequence[float], b: Sequence[float]) -> float:
    """Shortest geodesic distance for the reference geometry."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if instance.name == "disk":
        return float(np.linalg.norm(b - a))

    if instance.name == "sphere":
        r = instance.scale_m
        cos_angle = float(np.dot(a, b) / (r * r))
        angle = math.acos(np.clip(cos_angle, -1.0, 1.0))
        return r * angle

    if instance.name == "torus":
        return float(np.linalg.norm(_torus_delta(a, b, instance.scale_m)))

    if instance.name == "klein":
        lift = _klein_best_lift(a, b, instance.scale_m)
        return float(np.linalg.norm(lift - a))

    raise ValueError(f"Unsupported manifold: {instance.name}")


def _geodesic_samples(
    instance: MapInstance,
    a: Sequence[float],
    b: Sequence[float],
    n: int = 96,
) -> np.ndarray:
    """Sample a reference shortest geodesic, used for ocean intersection checks."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ts = np.linspace(0.0, 1.0, n)

    if instance.name == "disk":
        return a[None, :] + ts[:, None] * (b - a)[None, :]

    if instance.name == "torus":
        d = _torus_delta(a, b, instance.scale_m)
        return (a[None, :] + ts[:, None] * d[None, :]) % instance.scale_m

    if instance.name == "klein":
        lift = _klein_best_lift(a, b, instance.scale_m)
        cover = a[None, :] + ts[:, None] * (lift - a)[None, :]
        return np.vstack([_reduce_klein_cover(p, instance.scale_m) for p in cover])

    if instance.name == "sphere":
        r = instance.scale_m
        ua, ub = a / r, b / r
        omega = math.acos(np.clip(float(np.dot(ua, ub)), -1.0, 1.0))
        if omega < 1e-12:
            return np.repeat(a[None, :], n, axis=0)
        sin_omega = math.sin(omega)
        if abs(sin_omega) < 1e-10:  # nearly antipodal: normalized interpolation fallback
            raw = (1.0 - ts[:, None]) * ua[None, :] + ts[:, None] * ub[None, :]
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            return r * raw / norms
        p = (
            np.sin((1.0 - ts) * omega)[:, None] / sin_omega * ua[None, :]
            + np.sin(ts * omega)[:, None] / sin_omega * ub[None, :]
        )
        return r * p

    raise ValueError(f"Unsupported manifold: {instance.name}")


def point_in_ocean(instance: MapInstance, point: Sequence[float]) -> bool:
    for center, radius in zip(instance.ocean_centers, instance.ocean_radii_m):
        if geodesic_distance(instance, point, center) <= float(radius):
            return True
    return False


def leg_crosses_ocean(instance: MapInstance, a: Sequence[float], b: Sequence[float]) -> bool:
    """Reference ocean-crossing test used by the competition scorer.

    This function samples the reference geodesic at 96 points through
    ``_geodesic_samples``. For official scoring this implementation is authoritative;
    an optimizer may use a faster approximation internally, but its final plan is
    always re-evaluated here.
    """
    if len(instance.ocean_centers) == 0:
        return False
    for point in _geodesic_samples(instance, a, b):
        if point_in_ocean(instance, point):
            return True
    return False


def route_distance(instance: MapInstance, points: Sequence[Sequence[float]]) -> tuple[float, list[float]]:
    if len(points) < 2:
        return 0.0, []
    legs = [geodesic_distance(instance, a, b) for a, b in zip(points[:-1], points[1:])]
    return float(sum(legs)), legs


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def _sample_points_for_manifold(
    rng: np.random.Generator, name: str, n: int, scale_m: float
) -> np.ndarray:
    if name == "disk":
        return _sample_disk(rng, n, scale_m)
    if name == "sphere":
        return _sample_sphere(rng, n, scale_m)
    if name in {"torus", "klein"}:
        return _sample_flat_square(rng, n, scale_m)
    raise ValueError(name)


def _make_instance(
    rng: np.random.Generator,
    name: str,
    scale_m: float,
) -> MapInstance:
    checkpoints = _sample_points_for_manifold(rng, name, N_CHECKPOINTS, scale_m)
    endpoints = _sample_points_for_manifold(rng, name, N_ENDPOINTS, scale_m)

    if name == "disk":
        start = np.array([0.0, 0.0])
    elif name == "sphere":
        start = np.array([0.0, 0.0, scale_m])
    else:
        start = np.array([0.0, 0.0])

    n_oceans = int(rng.integers(1, 4))
    ocean_centers = _sample_points_for_manifold(rng, name, n_oceans, scale_m)
    # Ocean radii are tied to the 1000 km characteristic length, not the flat-square side.
    ocean_radii = rng.uniform(
        0.10 * CHARACTERISTIC_MAP_LENGTH_M,
        0.25 * CHARACTERISTIC_MAP_LENGTH_M,
        size=n_oceans,
    )

    endpoint_demand = rng.integers(40, 401, size=N_ENDPOINTS).astype(float)

    instance = MapInstance(
        name=name,
        scale_m=scale_m,
        start=start,
        checkpoints=checkpoints,
        endpoints=endpoints,
        endpoint_demand_kg=endpoint_demand,
        endpoint_island_mask=np.zeros(N_ENDPOINTS, dtype=bool),
        ocean_centers=ocean_centers,
        ocean_radii_m=ocean_radii,
    )

    # Endpoints inside a water region are considered island destinations. Add a few
    # small isolated islands as well so the public seed always exercises the jet mode.
    ocean_islands = np.array([point_in_ocean(instance, p) for p in endpoints], dtype=bool)
    random_islands = rng.random(N_ENDPOINTS) < 0.08
    island_mask = ocean_islands | random_islands
    if island_mask.sum() < 2:
        island_mask[rng.choice(N_ENDPOINTS, size=2, replace=False)] = True
    instance.endpoint_island_mask = island_mask
    instance.validate()
    return instance


def generate_instances(seed: int = PUBLIC_SEED) -> dict[str, MapInstance]:
    rng = np.random.default_rng(seed)
    return {
        "disk": _make_instance(rng, "disk", CHARACTERISTIC_MAP_LENGTH_M),
        "sphere": _make_instance(rng, "sphere", CHARACTERISTIC_MAP_LENGTH_M),
        "torus": _make_instance(rng, "torus", 2.0 * CHARACTERISTIC_MAP_LENGTH_M),
        "klein": _make_instance(rng, "klein", 2.0 * CHARACTERISTIC_MAP_LENGTH_M),
    }


# ---------------------------------------------------------------------------
# Reference scoring helpers
# ---------------------------------------------------------------------------

def compute_mileage(transportation_method: str) -> float:
    """Distance in meters traveled per fuel unit."""
    try:
        return VEHICLES[transportation_method].mileage_m_per_fuel_unit
    except KeyError as exc:
        raise ValueError(f"Unknown transportation method: {transportation_method}") from exc


def degrade_food(food_quantity: float, days: float = 1.0) -> float:
    return float(food_quantity) * FOOD_RETAINED_PER_DAY ** float(days)


def nutrition_fraction(days: float) -> float:
    return math.exp(-NUTRITION_DECAY_RATE_PER_DAY * float(days))


def _infeasible(
    endpoint: int,
    vehicle: str,
    load_kg: float,
    reason: str,
    route_distance_m: float = math.inf,
) -> DeliveryMetrics:
    return DeliveryMetrics(
        feasible=False,
        reason=reason,
        endpoint=endpoint,
        vehicle=vehicle,
        load_kg=float(load_kg),
        route_distance_m=float(route_distance_m),
        travel_time_days=math.inf,
        dispatch_tokens=0.0,
        fuel_tokens=0.0,
        total_tokens=math.inf,
        attack_probability=1.0,
        food_retained_fraction=0.0,
        nutrition_fraction=0.0,
        wear_fraction=0.0,
        endpoint_quality_fraction=0.0,
    )


def evaluate_delivery(
    instance: MapInstance,
    endpoint: int,
    vehicle: str,
    checkpoint_indices: Sequence[int] = (),
    load_kg: float | None = None,
) -> DeliveryMetrics:
    """Evaluate one route from the depot through optional checkpoints to an endpoint.

    The score is deterministic and uses *expected* bandit losses rather than sampling an
    attack. A route is invalid if it violates capacity/range, uses a ground vehicle over
    an ocean-crossing geodesic leg, or uses a ground vehicle to serve an island.
    """
    if endpoint < 0 or endpoint >= N_ENDPOINTS:
        raise IndexError("endpoint out of range")
    if vehicle not in VEHICLES:
        raise ValueError(f"Unknown vehicle: {vehicle}")

    spec = VEHICLES[vehicle]
    demand = float(instance.endpoint_demand_kg[endpoint])
    load = demand if load_kg is None else float(load_kg)
    if load <= 0:
        return _infeasible(endpoint, vehicle, load, "load must be positive")
    if load > spec.capacity_kg + 1e-12:
        return _infeasible(endpoint, vehicle, load, "vehicle capacity exceeded")
    if bool(instance.endpoint_island_mask[endpoint]) and not spec.can_serve_island:
        return _infeasible(endpoint, vehicle, load, "island destination requires a jet")

    cps = []
    for idx in checkpoint_indices:
        if idx < 0 or idx >= N_CHECKPOINTS:
            raise IndexError(f"checkpoint index out of range: {idx}")
        cps.append(instance.checkpoints[idx])

    points = [instance.start, *cps, instance.endpoints[endpoint]]
    total_distance, legs = route_distance(instance, points)
    if any(d > spec.max_leg_m + 1e-9 for d in legs):
        return _infeasible(endpoint, vehicle, load, "maximum leg range/stamina exceeded", total_distance)

    if not spec.can_cross_ocean:
        for a, b in zip(points[:-1], points[1:]):
            if leg_crosses_ocean(instance, a, b):
                return _infeasible(endpoint, vehicle, load, "ground route crosses an ocean region", total_distance)

    travel_time = total_distance / spec.speed_m_per_day + MIN_CHECKPOINT_TIME_DAYS * len(cps)
    if travel_time > spec.lifetime_days:
        return _infeasible(endpoint, vehicle, load, "vehicle lifetime exceeded", total_distance)

    dispatch_tokens = spec.dispatch_cost_tokens
    fuel_units = total_distance / spec.mileage_m_per_fuel_unit
    fuel_tokens = fuel_units * spec.fuel_cost_tokens_per_unit
    total_tokens = dispatch_tokens + fuel_tokens

    direct_distance = max(
        geodesic_distance(instance, instance.start, instance.endpoints[endpoint]),
        1.0,
    )
    detour_ratio = max(1.0, total_distance / direct_distance)

    # More detour can reduce exposure to the predictable bandit corridor, but it also
    # increases mechanical wear. This is intentionally a trade-off rather than a free win.
    attack_rate = BANDIT_BASE_RATE_PER_DAY * math.exp(
        -BANDIT_DETOUR_SUPPRESSION * (detour_ratio - 1.0)
    )
    attack_probability = 1.0 - math.exp(-attack_rate * travel_time)
    attack_retention = 1.0 - ATTACK_LOSS_FRACTION * attack_probability
    wear_fraction = math.exp(-DETOUR_WEAR_RATE * (detour_ratio - 1.0))
    food_fraction = FOOD_RETAINED_PER_DAY ** travel_time
    nutrition = nutrition_fraction(travel_time)

    served_fraction = min(load / demand, 1.0)
    endpoint_quality = served_fraction * food_fraction * nutrition * attack_retention * wear_fraction

    return DeliveryMetrics(
        feasible=True,
        reason="ok",
        endpoint=endpoint,
        vehicle=vehicle,
        load_kg=load,
        route_distance_m=total_distance,
        travel_time_days=travel_time,
        dispatch_tokens=dispatch_tokens,
        fuel_tokens=fuel_tokens,
        total_tokens=total_tokens,
        attack_probability=attack_probability,
        food_retained_fraction=food_fraction,
        nutrition_fraction=nutrition,
        wear_fraction=wear_fraction,
        endpoint_quality_fraction=float(np.clip(endpoint_quality, 0.0, 1.0)),
    )


def score_manifold(
    instance: MapInstance,
    deliveries: Iterable[Mapping[str, object]],
    token_budget: float = INITIAL_TOKENS,
) -> dict[str, object]:
    """Score a collection of deliveries on one manifold.

    Each delivery mapping accepts:
      endpoint: int (required)
      vehicle: 'horse' | 'truck' | 'jet' (required)
      checkpoints: sequence[int] (optional)
      load_kg: float (optional; defaults to destination demand)

    If more than one delivery targets the same endpoint, only the best quality counts,
    while all token expenditure still counts. Exceeding the token budget invalidates the
    manifold score.
    """
    best_quality = np.zeros(N_ENDPOINTS, dtype=float)
    metrics: list[DeliveryMetrics] = []
    total_tokens = 0.0
    total_fuel_tokens = 0.0

    for delivery in deliveries:
        endpoint = int(delivery["endpoint"])
        vehicle = str(delivery["vehicle"])
        checkpoints = delivery.get("checkpoints", ())
        load_kg = delivery.get("load_kg", None)
        m = evaluate_delivery(
            instance,
            endpoint=endpoint,
            vehicle=vehicle,
            checkpoint_indices=tuple(int(x) for x in checkpoints),
            load_kg=None if load_kg is None else float(load_kg),
        )
        metrics.append(m)
        if math.isfinite(m.total_tokens):
            total_tokens += m.total_tokens
            total_fuel_tokens += m.fuel_tokens
        if m.feasible:
            best_quality[endpoint] = max(best_quality[endpoint], m.endpoint_quality_fraction)

    demand = instance.endpoint_demand_kg
    coverage_quality = float(np.dot(demand, best_quality) / np.sum(demand))

    if total_tokens > token_budget + 1e-9:
        manifold_score = 0.0
        valid_budget = False
    else:
        valid_budget = True
        fuel_efficiency = 1.0 - min(total_fuel_tokens / token_budget, 1.0)
        manifold_score = 100.0 * coverage_quality * (0.85 + 0.15 * fuel_efficiency)

    return {
        "manifold": instance.name,
        "score": manifold_score,
        "coverage_quality": coverage_quality,
        "tokens_used": total_tokens,
        "fuel_tokens_used": total_fuel_tokens,
        "budget_valid": valid_budget,
        "deliveries": [asdict(m) for m in metrics],
    }


def compute_final_score(C_disk: float, C_sphere: float, C_torus: float, C_klein: float) -> float:
    """Normalized weighted score in [0, 100] when every C is in [0, 100]."""
    raw = C_disk + 2.0 * C_sphere + 3.0 * C_torus + 5.0 * C_klein
    return raw / 11.0


def weighted_final_score(scores: Mapping[str, float]) -> float:
    missing = set(FINAL_SCORE_WEIGHTS) - set(scores)
    if missing:
        raise KeyError(f"Missing manifold scores: {sorted(missing)}")
    raw = sum(FINAL_SCORE_WEIGHTS[k] * float(scores[k]) for k in FINAL_SCORE_WEIGHTS)
    return raw / sum(FINAL_SCORE_WEIGHTS.values())


def score_plan(
    instances: Mapping[str, MapInstance],
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Score one JSON-style plan across all four worlds.

    Expected format::

        {
          "disk":  [{"endpoint": 0, "vehicle": "truck", "checkpoints": []}],
          "sphere": [...],
          "torus":  [...],
          "klein":  [...]
        }

    A missing world is treated as an empty delivery list. Extra top-level keys are
    ignored so teams may include their own metadata.
    """
    world_results: dict[str, object] = {}
    world_scores: dict[str, float] = {}
    for name, instance in instances.items():
        deliveries = plan.get(name, [])
        if not isinstance(deliveries, list):
            raise TypeError(f"Plan entry {name!r} must be a list of deliveries")
        result = score_manifold(instance, deliveries)
        world_results[name] = result
        world_scores[name] = float(result["score"])

    return {
        "worlds": world_results,
        "final_score": weighted_final_score(world_scores),
    }


def load_plan(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Plan JSON must contain an object at the top level")
    return payload


def seed_has_clear_depots(seed: int) -> tuple[bool, list[str]]:
    """Return whether no depot lies inside an ocean for this seed.

    Organizers use this helper to screen hidden seeds. It does not alter instance
    generation, so seed 67 and all participant-facing behavior remain unchanged.
    """
    instances = generate_instances(seed)
    bad = [name for name, inst in instances.items() if point_in_ocean(inst, inst.start)]
    return (len(bad) == 0, bad)


# ---------------------------------------------------------------------------
# Export / command-line interface
# ---------------------------------------------------------------------------

def _array(x: np.ndarray) -> list:
    return np.asarray(x).tolist()


def instance_to_jsonable(instance: MapInstance) -> dict[str, object]:
    return {
        "name": instance.name,
        "scale_m": instance.scale_m,
        "start": _array(instance.start),
        "checkpoints": _array(instance.checkpoints),
        "endpoints": _array(instance.endpoints),
        "endpoint_demand_kg": _array(instance.endpoint_demand_kg),
        "endpoint_island_mask": _array(instance.endpoint_island_mask.astype(int)),
        "ocean_centers": _array(instance.ocean_centers),
        "ocean_radii_m": _array(instance.ocean_radii_m),
    }


def export_json(instances: Mapping[str, MapInstance], path: str | Path, seed: int) -> None:
    payload = {
        "seed": int(seed),
        "initial_tokens": INITIAL_TOKENS,
        "vehicle_specs": {name: asdict(spec) for name, spec in VEHICLES.items()},
        "constants": {
            "min_checkpoint_time_days": MIN_CHECKPOINT_TIME_DAYS,
            "food_retained_per_day": FOOD_RETAINED_PER_DAY,
            "nutrition_decay_rate_per_day": NUTRITION_DECAY_RATE_PER_DAY,
            "bandit_base_rate_per_day": BANDIT_BASE_RATE_PER_DAY,
            "attack_loss_fraction": ATTACK_LOSS_FRACTION,
            "detour_wear_rate": DETOUR_WEAR_RATE,
            "final_score_weights": FINAL_SCORE_WEIGHTS,
        },
        "instances": {name: instance_to_jsonable(inst) for name, inst in instances.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(instances: Mapping[str, MapInstance], seed: int) -> None:
    print(f"Routing on Curved Worlds public instance summary (seed={seed})")
    print(f"Token budget: {INITIAL_TOKENS:.0f}")
    print("Vehicles:")
    for name, spec in VEHICLES.items():
        print(
            f"  {name:5s}: dispatch={spec.dispatch_cost_tokens:5.0f}, "
            f"speed={spec.speed_m_per_day/1000:7.0f} km/day, "
            f"capacity={spec.capacity_kg:6.0f} kg, "
            f"max leg={spec.max_leg_m/1000:6.0f} km"
        )
    print("Maps:")
    for name, inst in instances.items():
        direct = np.array([geodesic_distance(inst, inst.start, p) for p in inst.endpoints])
        print(
            f"  {name:6s}: checkpoints={len(inst.checkpoints):2d}, "
            f"endpoints={len(inst.endpoints):2d}, oceans={len(inst.ocean_centers)}, "
            f"islands={int(inst.endpoint_island_mask.sum()):2d}, "
            f"median direct distance={np.median(direct)/1000:7.1f} km"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PUBLIC_SEED, help="RNG seed (default: 67)")
    parser.add_argument("--json", type=Path, help="Write generated instance data to this JSON file")
    parser.add_argument(
        "--score-plan",
        type=Path,
        metavar="PLAN.json",
        help="Score a plan JSON file on the selected seed",
    )
    parser.add_argument(
        "--score-json",
        type=Path,
        metavar="RESULT.json",
        help="When --score-plan is used, also write the full scoring result as JSON",
    )
    parser.add_argument(
        "--check-depot",
        action="store_true",
        help="Report whether any depot lies inside an ocean for the selected seed",
    )
    args = parser.parse_args()

    instances = generate_instances(args.seed)
    print_summary(instances, args.seed)

    if args.check_depot:
        bad = [name for name, inst in instances.items() if point_in_ocean(inst, inst.start)]
        if bad:
            print("Depot-in-ocean worlds:", ", ".join(bad))
        else:
            print("Depot screening: clear on all four worlds")

    if args.json:
        export_json(instances, args.json, args.seed)
        print(f"Wrote {args.json}")

    if args.score_plan:
        result = score_plan(instances, load_plan(args.score_plan))
        print("Scores:")
        for name in FINAL_SCORE_WEIGHTS:
            r = result["worlds"][name]
            print(
                f"  {name:6s}: {float(r['score']):8.4f}  "
                f"tokens={float(r['tokens_used']):8.2f}  "
                f"budget_valid={bool(r['budget_valid'])}"
            )
        print(f"Final weighted score C = {float(result['final_score']):.6f}")
        if args.score_json:
            args.score_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Wrote {args.score_json}")


if __name__ == "__main__":
    main()
