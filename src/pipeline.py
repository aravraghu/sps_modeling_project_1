"""Per-world pipeline: geometry tables, candidate construction, selection.

Stages, for one world:
  1. WorldTables      -- node layout, pairwise geodesics, the ocean-crossing mask
  2. build_candidates -- per (endpoint, vehicle), the K best routes, scored exactly
  3. selection.solve  -- choose which endpoints to serve within the token budget

Endpoint nodes are sinks in the graph: the scorer's route format is
[depot, *checkpoints, endpoint], so a route cannot pass through another endpoint."""

from __future__ import annotations

import math
import time

import numpy as np

import optimal_transport as ot
from include.constants import (
    DEFAULT_K,
    DEFAULT_MAX_HOPS,
    HORSE_MAX_RADIUS_M,
    JET_GROUND_QUALITY_GATE,
    VEHICLE_ORDER,
    WORLDS,
)
from include.types import Candidate
from src import geometry
from src import selection
from src.routing import k_shortest_paths

# ---------------------------------------------------------------------------
# Per-world geometry tables (vehicle independent, computed once)
# ---------------------------------------------------------------------------

class WorldTables:
    """Node layout, pairwise distances, and the ocean-crossing mask for one world."""

    def __init__(self, instance: ot.MapInstance):
        self.instance = instance
        self.n_cp = len(instance.checkpoints)
        self.n_ep = len(instance.endpoints)

        # Node order: 0 = depot, 1..n_cp = checkpoints, then endpoints.
        start = np.atleast_2d(np.asarray(instance.start, dtype=float))
        self.nodes = np.vstack([start, instance.checkpoints, instance.endpoints])
        self.depot = 0
        self.cp0 = 1
        self.ep0 = 1 + self.n_cp
        self.n_nodes = len(self.nodes)

        self.dist = geometry.pairwise_geodesic(instance, self.nodes, self.nodes)
        np.fill_diagonal(self.dist, 0.0)

        # Ocean crossing is a property of the leg, not the vehicle, so build it once.
        iu, ju = np.triu_indices(self.n_nodes, k=1)
        crossed = geometry.legs_cross_ocean(
            instance, self.nodes[iu], self.nodes[ju]
        )
        self.ocean = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        self.ocean[iu, ju] = crossed
        self.ocean[ju, iu] = crossed

        # Direct depot -> endpoint distance, the D0 in rho = D / D0.
        self.direct = self.dist[self.depot, self.ep0 :].copy()

    def ep_node(self, ep: int) -> int:
        return self.ep0 + ep

    def weights(self, vehicle: str, metric: str,
                node_radius_m: float | None = None) -> np.ndarray:
        """Edge weight matrix for one mode under one metric; inf means no edge.

        ``node_radius_m`` drops every node further than that from the depot, which is
        how the horse gate is applied (see HORSE_MAX_RADIUS_M).
        """
        spec = ot.VEHICLES[vehicle]
        allowed = self.dist <= spec.max_leg_m + 1e-9
        if not spec.can_cross_ocean:
            allowed &= ~self.ocean
        if node_radius_m is not None:
            near = self.dist[self.depot] <= node_radius_m
            allowed &= near[:, None] & near[None, :]
        np.fill_diagonal(allowed, False)
        allowed[self.ep0 :, :] = False  # endpoints are sinks

        if metric == "tokens":
            # Dispatch is path-independent and the fuel term is proportional to
            # distance, so this is distance in different units: it always returns the
            # same K paths in the same order. Kept because it is the natural cost view.
            base = self.dist / spec.mileage_m_per_fuel_unit * spec.fuel_cost_tokens_per_unit
        elif metric == "time":
            # Each arrival at a checkpoint costs one extra day on top of travel.
            base = self.dist / spec.speed_m_per_day
            hop = np.zeros(self.n_nodes)
            hop[self.cp0 : self.ep0] = ot.MIN_CHECKPOINT_TIME_DAYS
            base = base + hop[None, :]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        W = np.where(allowed, base, np.inf)
        return W


# ---------------------------------------------------------------------------
# Exact candidate scoring (mirrors ot.evaluate_delivery, no ocean re-sampling)
# ---------------------------------------------------------------------------

def score_route(
    tables: WorldTables,
    ep: int,
    vehicle: str,
    cp_indices: tuple[int, ...],
    distance_m: float,
    weight_sum: float,
) -> Candidate | None:
    """Build a Candidate, or None if the route is infeasible.

    Feasibility conditions already enforced by the graph: per-leg range, and ocean
    avoidance for ground modes. Still checked here: island access, capacity, and
    vehicle lifetime.
    """
    inst = tables.instance
    spec = ot.VEHICLES[vehicle]
    demand = float(inst.endpoint_demand_kg[ep])

    if bool(inst.endpoint_island_mask[ep]) and not spec.can_serve_island:
        return None

    # Cost is independent of load (dispatch + fuel(distance) only), while quality is
    # increasing in load, so the best load is always as much as will fit.
    load = min(demand, spec.capacity_kg)
    if load <= 0:
        return None

    n_cp = len(cp_indices)
    travel_days = distance_m / spec.speed_m_per_day + ot.MIN_CHECKPOINT_TIME_DAYS * n_cp
    if travel_days > spec.lifetime_days:
        return None

    dispatch = spec.dispatch_cost_tokens
    fuel = distance_m / spec.mileage_m_per_fuel_unit * spec.fuel_cost_tokens_per_unit
    tokens = dispatch + fuel

    direct = max(float(tables.direct[ep]), 1.0)
    rho = max(1.0, distance_m / direct)

    attack_rate = ot.BANDIT_BASE_RATE_PER_DAY * math.exp(
        -ot.BANDIT_DETOUR_SUPPRESSION * (rho - 1.0)
    )
    p_attack = 1.0 - math.exp(-attack_rate * travel_days)
    attack_retention = 1.0 - ot.ATTACK_LOSS_FRACTION * p_attack
    wear = math.exp(-ot.DETOUR_WEAR_RATE * (rho - 1.0))
    food = ot.FOOD_RETAINED_PER_DAY ** travel_days
    nutrition = math.exp(-ot.NUTRITION_DECAY_RATE_PER_DAY * travel_days)

    served = min(load / demand, 1.0)
    q = float(np.clip(served * food * nutrition * attack_retention * wear, 0.0, 1.0))

    total_demand = float(np.sum(inst.endpoint_demand_kg))
    return Candidate(
        endpoint=ep,
        vehicle=vehicle,
        checkpoints=cp_indices,
        load_kg=load,
        omit_load=abs(load - demand) < 1e-9,
        value=(demand / total_demand) * q,
        quality=q,
        tokens=tokens,
        fuel=fuel,
        distance_m=distance_m,
        travel_days=travel_days,
    )


# ---------------------------------------------------------------------------
# Per-world solve
# ---------------------------------------------------------------------------

def _vehicle_worth_trying(
    tables: WorldTables,
    vehicle: str,
    ep: int,
    per_ep: list[list[Candidate]],
) -> bool:
    """Should we run the path search for this (vehicle, endpoint) pair at all?

    Called with ground vehicles already processed for this endpoint, so the jet gate
    can look at what the ground modes actually achieved.
    """
    inst = tables.instance

    if vehicle == "horse":
        # Outside the radius a horse is dominated at every demand (see the constant).
        return bool(tables.direct[ep] <= HORSE_MAX_RADIUS_M)

    if vehicle == "jet":
        if bool(inst.endpoint_island_mask[ep]):
            return True  # only a jet may serve an island
        ground = per_ep[ep]
        if not ground:
            return True  # nothing on the ground can reach it
        return max(c.quality for c in ground) < JET_GROUND_QUALITY_GATE

    return True  # trucks are always worth trying


def build_candidates(
    tables: WorldTables,
    K: int,
    metric: str,
    max_hops: int | None,
    vehicles=VEHICLE_ORDER,
) -> tuple[list[list[Candidate]], dict]:
    """Enumerate K-shortest-path candidates for every (endpoint, vehicle) pair."""
    per_ep: list[list[Candidate]] = [[] for _ in range(tables.n_ep)]
    stats = {v: {"paths": 0, "feasible": 0, "endpoints": 0} for v in vehicles}

    for vehicle in vehicles:
        radius = HORSE_MAX_RADIUS_M if vehicle == "horse" else None
        W = tables.weights(vehicle, metric, node_radius_m=radius)
        reachable = 0
        for ep in range(tables.n_ep):
            if not _vehicle_worth_trying(tables, vehicle, ep, per_ep):
                continue
            target = tables.ep_node(ep)
            paths = k_shortest_paths(W, tables.depot, target, K, max_hops=max_hops)
            if paths:
                reachable += 1
            for _, path in paths:
                stats[vehicle]["paths"] += 1
                cps = tuple(node - tables.cp0 for node in path[1:-1])
                # True geodesic length, independent of whichever metric ranked it.
                distance_m = float(
                    sum(tables.dist[a, b] for a, b in zip(path[:-1], path[1:]))
                )
                cand = score_route(
                    tables, ep, vehicle, cps, distance_m, 0.0
                )
                if cand is not None:
                    stats[vehicle]["feasible"] += 1
                    per_ep[ep].append(cand)
        stats[vehicle]["endpoints"] = reachable

    return per_ep, stats


def solve_world(
    instance: ot.MapInstance,
    K: int = DEFAULT_K,
    metric: str = "time",
    max_hops: int | None = DEFAULT_MAX_HOPS,
    budget: float = ot.INITIAL_TOKENS,
    fuel_correction: bool = True,
    method: str = "bnb",
) -> dict:
    t0 = time.perf_counter()
    tables = WorldTables(instance)
    t_geom = time.perf_counter() - t0

    t1 = time.perf_counter()
    per_ep, stats = build_candidates(tables, K, metric, max_hops)
    t_cand = time.perf_counter() - t1

    raw_total = sum(len(c) for c in per_ep)
    groups = [selection.pareto_filter(c) for c in per_ep if c]
    kept_total = sum(len(g) for g in groups)

    t2 = time.perf_counter()
    chosen, info, lam_trace = selection.solve(
        groups, budget_tokens=budget, fuel_correction=fuel_correction, method=method
    )
    t_knap = time.perf_counter() - t2

    deliveries = []
    for c in sorted(chosen, key=lambda x: x.endpoint):
        d: dict = {
            "endpoint": int(c.endpoint),
            "vehicle": c.vehicle,
            "checkpoints": [int(x) for x in c.checkpoints],
        }
        if not c.omit_load:
            d["load_kg"] = float(c.load_kg)
        deliveries.append(d)

    return {
        "deliveries": deliveries,
        "chosen": chosen,
        "info": info,
        "stats": stats,
        "candidates_raw": raw_total,
        "candidates_kept": kept_total,
        "lambda_trace": lam_trace,
        "timing": {"geometry": t_geom, "candidates": t_cand, "knapsack": t_knap},
    }


def solve_all(
    seed: int,
    K: int = DEFAULT_K,
    metric: str = "time",
    max_hops: int | None = DEFAULT_MAX_HOPS,
    fuel_correction: bool = True,
    method: str = "bnb",
) -> tuple[dict, dict]:
    instances = ot.generate_instances(seed)
    plan: dict = {}
    results: dict = {}
    for name in WORLDS:
        res = solve_world(instances[name], K=K, metric=metric, max_hops=max_hops,
                          fuel_correction=fuel_correction, method=method)
        plan[name] = res["deliveries"]
        results[name] = res
    return plan, results


