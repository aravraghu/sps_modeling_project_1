"""Routing on Curved Worlds: solver for SPS Modelling Competition 2026, Prompt 1.

Usage:
    python solve.py --seed N --out plan.json

Pipeline (run independently on each world):
    Stage 1  Geometry: 101-node distance matrix and ocean-crossing matrix
    Stage 2  Legal-leg matrices per vehicle
    Stage 3  Layered shortest paths (exact shortest distance for each checkpoint count k)
    Stage 4  Convert routes to delivery options (exact quality, tokens); drop dominated ones
    Stage 5  Multiple-choice knapsack DP over (dispatch total, fuel total)
    Stage 6  Exact world score, then local-search polish on that exact score
    Stage 7  Verify against the reference scorer
    Stage 8  Write plan JSON

Options:
    --baseline   write the simple baseline plan instead (cheapest direct delivery per
                 endpoint, added greedily by value per token)
"""

from __future__ import annotations

# Standard library
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Third party
import numpy as np

# Reference model (do not modify optimal_transport.py)
from optimal_transport import (
    # Instance generation and types
    PUBLIC_SEED,
    MapInstance,
    VehicleSpec,
    generate_instances,
    # Reference scoring (used to verify every plan)
    score_manifold,
    score_plan,
    # Constants (imported, never hard-coded)
    VEHICLES,
    INITIAL_TOKENS,
    N_CHECKPOINTS,
    N_ENDPOINTS,
    MIN_CHECKPOINT_TIME_DAYS,
    FOOD_RETAINED_PER_DAY,
    NUTRITION_DECAY_RATE_PER_DAY,
    BANDIT_BASE_RATE_PER_DAY,
    BANDIT_DETOUR_SUPPRESSION,
    ATTACK_LOSS_FRACTION,
    DETOUR_WEAR_RATE,
    FINAL_SCORE_WEIGHTS,
)


# Wall-clock reference for the time guards below (set as early as possible).
_T_START = time.perf_counter()


# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------

# Node layout shared by every world: one combined list of 101 points.
DEPOT_NODE = 0
CP_OFFSET = 1                       # checkpoint c -> node CP_OFFSET + c
EP_OFFSET = 1 + N_CHECKPOINTS       # endpoint j   -> node EP_OFFSET + j
N_NODES = 1 + N_CHECKPOINTS + N_ENDPOINTS

# Must match the reference sampler (_geodesic_samples uses n=96).
N_OCEAN_SAMPLES = 96

# Longest leg any ground vehicle may drive. Only legs this short need an ocean check.
GROUND_RANGE_M = max(s.max_leg_m for s in VEHICLES.values() if not s.can_cross_ocean)
RANGE_SLACK_M = 1.0

# Klein lifts searched by the reference (_klein_best_lift loops n outer, m inner, over
# -2..2). Flattening a meshgrid with indexing="ij" keeps that same order, so argmin
# breaks ties exactly as the reference's strict "<" comparison does.
_KLEIN_N, _KLEIN_M = (
    g.ravel() for g in np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij")
)
_KLEIN_SIGN = np.where(_KLEIN_N % 2 == 0, 1.0, -1.0)  # (-1)**n

# Legs per batch in the ocean check (bounds memory use).
OCEAN_BATCH_LEGS = 256

# A leg counts as within range only if it is at least this much shorter than the max leg.
# Our distances agree with the reference to ~1e-8 m; the margin means float noise can
# never make us accept a leg the reference would reject.
LEG_SAFETY_M = 1e-6

# Most checkpoints considered on one route. Each checkpoint costs a day (~7% quality),
# and 5 checkpoints already cover the longest truck route on the sphere (pi R / 750 km).
MAX_CHECKPOINTS = 5

# Fuel resolution of the knapsack DP, in tokens. Fuel is rounded up, so every plan the
# DP accepts is within budget; Stage 6 then re-optimizes on exact fuel.
FUEL_BIN_TOKENS = 0.1

# If the reference ever rejects a chosen delivery, ban it and re-solve (at most this often).
MAX_REPAIR_ROUNDS = 3

# Time guards. The organizers allow 120 s per seed; a typical seed takes about 1 s.
# These only matter on a pathologically slow machine, and keep a timeout (which would
# score zero for the whole seed) impossible:
#   after SOFT_DEADLINE_S, remaining worlds use the fast baseline instead of the optimizer
#   after HARD_DEADLINE_S, remaining worlds submit no deliveries
#   local search stops at MAX_POLISH_ROUNDS improvements or at the soft deadline
TIME_LIMIT_S = 120.0
SOFT_DEADLINE_S = 60.0
HARD_DEADLINE_S = 100.0
MAX_POLISH_ROUNDS = 200


def elapsed_s() -> float:
    """Seconds since solve.py was imported."""
    return time.perf_counter() - _T_START


# ---------------------------------------------------------------------------
# Stage 1: Geometry (distance matrix, ocean-crossing matrix)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    """Per-world geometry shared by every later stage.

    coords         (N_NODES, dim)      node positions: depot, checkpoints, endpoints
    dist           (N_NODES, N_NODES)  geodesic leg lengths in meters
    crosses_ocean  (N_NODES, N_NODES)  True if the reference geodesic for leg i -> j
                                       crosses an ocean. Only legs a ground vehicle could
                                       ever drive are checked (from the depot or a
                                       checkpoint, into a checkpoint or endpoint, at most
                                       GROUND_RANGE_M long). All other entries are left
                                       True so a ground vehicle can never use them.
    """
    coords: np.ndarray
    dist: np.ndarray
    crosses_ocean: np.ndarray


def node_coordinates(inst: MapInstance) -> np.ndarray:
    """Stack depot, checkpoints and endpoints into one (N_NODES, dim) array."""
    start = np.asarray(inst.start, dtype=float)[None, :]
    return np.vstack([start, inst.checkpoints, inst.endpoints]).astype(float)


def best_displacement(inst: MapInstance, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Displacement from a to the nearest image of b on a flat world.

    a and b are (..., 2) arrays that broadcast against each other. This is the
    covering-space picture: unroll the world onto the plane, then take the shortest
    straight segment from a to any copy of b. Each branch mirrors the reference exactly.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    L = inst.scale_m

    if inst.name == "disk":
        return b - a

    if inst.name == "torus":
        # Same expression as _torus_delta: wrap each component into [-L/2, L/2).
        return (b - a + L / 2.0) % L - L / 2.0

    if inst.name == "klein":
        # Copies of b: (+-bx + m L, by + n L), with x mirrored when n is odd.
        cx = _KLEIN_SIGN * b[..., 0][..., None] + _KLEIN_M * L   # (..., 25)
        cy = b[..., 1][..., None] + _KLEIN_N * L
        dx = cx - a[..., 0][..., None]
        dy = cy - a[..., 1][..., None]
        k = np.argmin(dx * dx + dy * dy, axis=-1)[..., None]
        return np.concatenate(
            [np.take_along_axis(dx, k, axis=-1), np.take_along_axis(dy, k, axis=-1)],
            axis=-1,
        )

    raise ValueError(f"best_displacement is only defined for flat worlds, not {inst.name}")


def pair_distance(inst: MapInstance, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic distance in meters between broadcastable point arrays a and b."""
    if inst.name == "sphere":
        r = inst.scale_m
        cos_angle = np.sum(np.asarray(a, float) * np.asarray(b, float), axis=-1) / (r * r)
        return r * np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.linalg.norm(best_displacement(inst, a, b), axis=-1)


def distance_matrix(inst: MapInstance, coords: np.ndarray) -> np.ndarray:
    """(N, N) matrix of geodesic distances between all node pairs."""
    dist = pair_distance(inst, coords[:, None, :], coords[None, :, :])
    np.fill_diagonal(dist, 0.0)   # arccos(1 - eps) on the sphere is not exactly zero
    return dist


def _reduce_klein(points: np.ndarray, L: float) -> np.ndarray:
    """Vectorized _reduce_klein_cover: map covering-space points into [0, L)^2."""
    x = points[..., 0]
    y = points[..., 1]
    n = np.floor(y / L)
    y = y - n * L
    x = np.where(n.astype(np.int64) % 2 != 0, -x, x)
    x = np.mod(x, L)              # np.mod matches Python's % for floats
    return np.stack([x, y], axis=-1)


def geodesic_samples(inst: MapInstance, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized _geodesic_samples for M legs at once.

    a, b: (M, dim) leg start and end points. Returns (M, N_OCEAN_SAMPLES, dim) points on
    the reference geodesic, expressed in the world's own coordinates.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ts = np.linspace(0.0, 1.0, N_OCEAN_SAMPLES)
    t = ts[None, :, None]                             # (1, n, 1)
    L = inst.scale_m

    if inst.name == "disk":
        return a[:, None, :] + t * (b - a)[:, None, :]

    if inst.name == "torus":
        d = best_displacement(inst, a, b)
        return (a[:, None, :] + t * d[:, None, :]) % L

    if inst.name == "klein":
        d = best_displacement(inst, a, b)             # lift(b) - a
        cover = a[:, None, :] + t * d[:, None, :]
        return _reduce_klein(cover, L)

    if inst.name == "sphere":
        r = inst.scale_m
        ua, ub = a / r, b / r
        omega = np.arccos(np.clip(np.sum(ua * ub, axis=-1), -1.0, 1.0))   # (M,)
        sin_omega = np.sin(omega)

        # General case: spherical linear interpolation (slerp).
        with np.errstate(divide="ignore", invalid="ignore"):
            w_a = np.sin((1.0 - ts)[None, :] * omega[:, None]) / sin_omega[:, None]
            w_b = np.sin(ts[None, :] * omega[:, None]) / sin_omega[:, None]
        slerp = r * (w_a[..., None] * ua[:, None, :] + w_b[..., None] * ub[:, None, :])

        # Nearly antipodal: normalized linear interpolation, as in the reference.
        raw = (1.0 - t) * ua[:, None, :] + t * ub[:, None, :]
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        antipodal = r * raw / norms

        # Same point: constant samples.
        same = np.broadcast_to(a[:, None, :], slerp.shape)

        out = np.where((np.abs(sin_omega) < 1e-10)[:, None, None], antipodal, slerp)
        return np.where((omega < 1e-12)[:, None, None], same, out)

    raise ValueError(f"Unsupported manifold: {inst.name}")


def points_in_ocean(inst: MapInstance, points: np.ndarray) -> np.ndarray:
    """Vectorized point_in_ocean. points: (..., dim). Returns a bool array of shape (...)."""
    inside = np.zeros(points.shape[:-1], dtype=bool)
    for center, radius in zip(inst.ocean_centers, inst.ocean_radii_m):
        inside |= pair_distance(inst, points, np.asarray(center, float)) <= float(radius)
    return inside


def ocean_crossing_matrix(inst: MapInstance, coords: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(N, N) bool matrix: True if leg i -> j crosses an ocean or is never checked.

    Only legs a ground vehicle could use are sampled: sources are the depot and the
    checkpoints, targets are checkpoints and endpoints, and the leg is at most
    GROUND_RANGE_M long. Directed legs are checked separately because the reference
    samples the geodesic in the direction of travel.
    """
    crosses = np.ones((N_NODES, N_NODES), dtype=bool)

    src, tgt = np.meshgrid(np.arange(EP_OFFSET), np.arange(1, N_NODES), indexing="ij")
    src, tgt = src.ravel(), tgt.ravel()
    keep = (src != tgt) & (dist[src, tgt] <= GROUND_RANGE_M + RANGE_SLACK_M)
    src, tgt = src[keep], tgt[keep]

    for lo in range(0, len(src), OCEAN_BATCH_LEGS):
        i = src[lo:lo + OCEAN_BATCH_LEGS]
        j = tgt[lo:lo + OCEAN_BATCH_LEGS]
        samples = geodesic_samples(inst, coords[i], coords[j])
        crosses[i, j] = points_in_ocean(inst, samples).any(axis=1)

    return crosses


def build_geometry(inst: MapInstance) -> Geometry:
    """Stage 1 entry point: everything later stages need to know about the world's shape."""
    coords = node_coordinates(inst)
    dist = distance_matrix(inst, coords)
    crosses = ocean_crossing_matrix(inst, coords, dist)
    return Geometry(coords=coords, dist=dist, crosses_ocean=crosses)


# ---------------------------------------------------------------------------
# Stage 2: Legal-leg matrices per vehicle
# ---------------------------------------------------------------------------

def legal_leg_matrix(geo: Geometry, spec: VehicleSpec) -> np.ndarray:
    """(N, N) leg lengths in meters for legs this vehicle may drive, inf elsewhere.

    A leg i -> j is legal if it starts at the depot or a checkpoint, ends at a checkpoint
    or an endpoint, fits within the vehicle's max leg, and, for ground vehicles, does not
    cross an ocean. Island endpoints are handled per vehicle in Stage 4.
    """
    legal = geo.dist <= spec.max_leg_m - LEG_SAFETY_M
    legal[:, DEPOT_NODE] = False     # routes never return to the depot
    legal[EP_OFFSET:, :] = False     # an endpoint is always the last stop
    np.fill_diagonal(legal, False)
    if not spec.can_cross_ocean:
        legal &= ~geo.crosses_ocean
    return np.where(legal, geo.dist, np.inf)


# ---------------------------------------------------------------------------
# Stage 3: Layered shortest paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayeredPaths:
    """Shortest routes from the depot to every endpoint, one per checkpoint count k.

    dist[k, j]  length (m) of the shortest route to endpoint j using exactly k checkpoints,
                inf if there is none
    via[k, j]   checkpoint visited just before endpoint j on that route (-1 when k = 0)
    pred[k, c]  checkpoint visited just before checkpoint c on the shortest k-checkpoint
                route ending at c (-1 for k <= 1, where the predecessor is the depot)
    """
    dist: np.ndarray
    via: np.ndarray
    pred: np.ndarray

    def checkpoints(self, k: int, j: int) -> list[int]:
        """Checkpoint indices (0-based, as in the plan JSON) for the route (k, j)."""
        if k == 0:
            return []
        path = [int(self.via[k, j])]
        for layer in range(k, 1, -1):
            path.append(int(self.pred[layer, path[-1]]))
        return path[::-1]


def layered_shortest_paths(legs: np.ndarray, max_k: int = MAX_CHECKPOINTS) -> LayeredPaths:
    """Hop-limited shortest paths from the depot (Bellman-Ford stopped after max_k rounds).

    best_k[c] is the shortest route depot -> checkpoint c through exactly k checkpoints:
        best_1[c]     = leg(depot, c)
        best_{k+1}[c] = min over c' of best_k[c'] + leg(c', c)
    and endpoint j is then reached with k checkpoints as min over c of best_k[c] + leg(c, j).

    Keeping one number per (k, c) is exact because route length is additive: the part of
    a shortest k-checkpoint route up to its last checkpoint is itself a shortest route with
    k - 1 checkpoints. Distances are summed leg by leg in travel order, the same order the
    reference uses, so route lengths match it to the last bit.
    """
    cp = slice(CP_OFFSET, EP_OFFSET)
    ep = slice(EP_OFFSET, N_NODES)
    cp_to_cp = legs[cp, cp]
    cp_to_ep = legs[cp, ep]
    cols_cp = np.arange(N_CHECKPOINTS)
    cols_ep = np.arange(N_ENDPOINTS)

    dist = np.full((max_k + 1, N_ENDPOINTS), np.inf)
    via = np.full((max_k + 1, N_ENDPOINTS), -1)
    pred = np.full((max_k + 1, N_CHECKPOINTS), -1)

    dist[0] = legs[DEPOT_NODE, ep]
    best = legs[DEPOT_NODE, cp].copy()
    for k in range(1, max_k + 1):
        if k > 1:
            totals = best[:, None] + cp_to_cp          # (from c', to c)
            pred[k] = np.argmin(totals, axis=0)
            best = totals[pred[k], cols_cp]
        totals = best[:, None] + cp_to_ep              # (from c, to endpoint j)
        via[k] = np.argmin(totals, axis=0)
        dist[k] = totals[via[k], cols_ep]
    return LayeredPaths(dist=dist, via=via, pred=pred)


# ---------------------------------------------------------------------------
# Stage 4: Route options (quality, cost, value) and dominance pruning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Option:
    """One candidate delivery: a vehicle and route to one endpoint, with its exact score."""
    endpoint: int
    vehicle: str
    checkpoints: tuple[int, ...]
    load_kg: float
    distance_m: float
    travel_days: float
    quality: float      # q_j in [0, 1]
    value: float        # demand_j * q_j, this option's contribution to sum_j D_j q_j (kg)
    dispatch: float     # tokens
    fuel: float         # tokens

    @property
    def cost(self) -> float:
        return self.dispatch + self.fuel


def delivery_quality(distance, n_checkpoints, direct, spec: VehicleSpec, served_fraction):
    """Reference quality q_j, vectorized over routes.

    Same formula and operation order as evaluate_delivery. The bandit term is the expected
    loss, 1 - 0.25 * p_attack, which the scorer computes deterministically, so it is exact
    here too. Callers pass finite distances only.
    """
    t = distance / spec.speed_m_per_day + MIN_CHECKPOINT_TIME_DAYS * n_checkpoints
    rho = np.maximum(1.0, distance / np.maximum(direct, 1.0))
    attack_rate = BANDIT_BASE_RATE_PER_DAY * np.exp(-BANDIT_DETOUR_SUPPRESSION * (rho - 1.0))
    attack_probability = 1.0 - np.exp(-attack_rate * t)
    attack_retention = 1.0 - ATTACK_LOSS_FRACTION * attack_probability
    wear = np.exp(-DETOUR_WEAR_RATE * (rho - 1.0))
    food = FOOD_RETAINED_PER_DAY ** t
    nutrition = np.exp(-NUTRITION_DECAY_RATE_PER_DAY * t)
    return np.clip(served_fraction * food * nutrition * attack_retention * wear, 0.0, 1.0)


def _weakly_dominates(a: Option, b: Option) -> bool:
    return a.value >= b.value and a.dispatch <= b.dispatch and a.fuel <= b.fuel


def prune_dominated(options: list[Option]) -> list[Option]:
    """Drop options that another option for the same endpoint beats on every axis.

    B is dropped if some A has value >= B, dispatch <= B and fuel <= B (ties broken by
    list order). Swapping B for A in any plan keeps it within budget, does not lower total
    value, and does not raise fuel, so the score cannot go down: removing B never changes
    the optimum. Dispatch and fuel are compared separately, not just total cost, because
    the fuel-efficiency factor depends on fuel alone.
    """
    keep = []
    for i, b in enumerate(options):
        beaten = any(
            _weakly_dominates(a, b)
            and (a.value > b.value or a.dispatch < b.dispatch or a.fuel < b.fuel or k < i)
            for k, a in enumerate(options)
            if k != i
        )
        if not beaten:
            keep.append(b)
    return sorted(keep, key=lambda o: o.cost)


def route_options(inst: MapInstance, geo: Geometry, max_k: int = MAX_CHECKPOINTS) -> list[list[Option]]:
    """Every non-dominated way to serve each endpoint: options[j] is a list of Options.

    For each vehicle and each checkpoint count k, only the shortest route is kept. For
    k <= 2 that route is provably the best one (shorter is strictly better for fixed k;
    see the README), so nothing is lost.

    Dominated choices removed by construction:
      * load is always min(demand, capacity): q rises with load and cost does not
      * at most one delivery per endpoint: only the best q counts, every delivery costs
      * ground vehicles never serve islands, and never use an ocean-crossing leg
    """
    demand = inst.endpoint_demand_kg.astype(float)
    direct = geo.dist[DEPOT_NODE, EP_OFFSET:]
    ks = np.arange(max_k + 1)[:, None]
    options: list[list[Option]] = [[] for _ in range(N_ENDPOINTS)]

    for name, spec in VEHICLES.items():
        paths = layered_shortest_paths(legal_leg_matrix(geo, spec), max_k)
        load = np.minimum(demand, spec.capacity_kg)
        served = np.minimum(load / demand, 1.0)

        usable = np.isfinite(paths.dist)
        if not spec.can_serve_island:
            usable &= ~inst.endpoint_island_mask[None, :]
        D = np.where(usable, paths.dist, 0.0)     # placeholder 0 keeps the math finite
        t = D / spec.speed_m_per_day + MIN_CHECKPOINT_TIME_DAYS * ks
        usable &= t <= spec.lifetime_days
        q = delivery_quality(D, ks, direct[None, :], spec, served[None, :])
        fuel = D / spec.mileage_m_per_fuel_unit * spec.fuel_cost_tokens_per_unit

        for k, j in zip(*np.nonzero(usable)):
            options[j].append(Option(
                endpoint=int(j),
                vehicle=name,
                checkpoints=tuple(paths.checkpoints(int(k), int(j))),
                load_kg=float(load[j]),
                distance_m=float(D[k, j]),
                travel_days=float(t[k, j]),
                quality=float(q[k, j]),
                value=float(demand[j] * q[k, j]),
                dispatch=float(spec.dispatch_cost_tokens),
                fuel=float(fuel[k, j]),
            ))

    return [prune_dominated(opts) for opts in options]


# ---------------------------------------------------------------------------
# Stage 5: Multiple-choice knapsack DP
# ---------------------------------------------------------------------------

def knapsack(options: list[list[Option]], total_demand: float,
             budget: float = INITIAL_TOKENS) -> list[Option]:
    """Choose at most one option per endpoint to maximize the world score within budget.

    State (d, f): total dispatch = d * unit tokens (unit = gcd of the dispatch costs, 25),
    total fuel = f bins of FUEL_BIN_TOKENS, rounded up per option.
        dp[d, f] = best total value sum(D_j q_j) over plans with exactly that state.
    Keeping dispatch and fuel apart, instead of only total cost, lets us evaluate the true
    score at the end: 100 * Q * (0.85 + 0.15 * (1 - fuel / budget)) is a product, so it
    cannot be maximized by a plain additive knapsack. Rounding fuel up means every state
    the DP accepts is within budget for real.
    """
    unit = math.gcd(*(int(round(s.dispatch_cost_tokens)) for s in VEHICLES.values()))
    n_d = int(budget // unit) + 1
    n_f = int(math.floor(budget / FUEL_BIN_TOKENS)) + 1

    def footprint(opt: Option) -> tuple[int, int]:
        return int(round(opt.dispatch / unit)), math.ceil(opt.fuel / FUEL_BIN_TOKENS)

    dp = np.full((n_d, n_f), -np.inf)
    dp[0, 0] = 0.0
    history: list[tuple[int, np.ndarray]] = []   # (endpoint, chosen option index per state)

    for j, opts in enumerate(options):
        if not opts:
            continue
        new = dp.copy()
        pick = np.full((n_d, n_f), -1, dtype=np.int16)
        for idx, opt in enumerate(opts):
            dd, ff = footprint(opt)
            if dd >= n_d or ff >= n_f:
                continue
            cand = np.full_like(dp, -np.inf)
            cand[dd:, ff:] = dp[:n_d - dd, :n_f - ff] + opt.value
            better = cand > new
            new[better] = cand[better]
            pick[better] = idx
        dp = new
        history.append((j, pick))

    # Exact score of every reachable, in-budget state; take the best.
    d_tok = (np.arange(n_d) * unit)[:, None]
    f_tok = (np.arange(n_f) * FUEL_BIN_TOKENS)[None, :]
    in_budget = (d_tok + f_tok <= budget + 1e-9) & np.isfinite(dp)
    fuel_factor = 0.85 + 0.15 * (1.0 - np.minimum(f_tok / budget, 1.0))
    score = np.where(in_budget, 100.0 * dp / total_demand * fuel_factor, -np.inf)
    d, f = np.unravel_index(int(np.argmax(score)), score.shape)

    chosen = []
    for j, pick in reversed(history):
        idx = int(pick[d, f])
        if idx >= 0:
            opt = options[j][idx]
            chosen.append(opt)
            dd, ff = footprint(opt)
            d, f = d - dd, f - ff
    return chosen[::-1]


# ---------------------------------------------------------------------------
# Stage 6: Exact objective with fuel-efficiency factor
# ---------------------------------------------------------------------------

def plan_score(chosen: list[Option], total_demand: float, budget: float = INITIAL_TOKENS) -> float:
    """World score C_M exactly as score_manifold defines it, from exact option values."""
    if sum(o.cost for o in chosen) > budget + 1e-9:
        return 0.0
    fuel = sum(o.fuel for o in chosen)
    coverage = sum(o.value for o in chosen) / total_demand
    return 100.0 * coverage * (0.85 + 0.15 * (1.0 - min(fuel / budget, 1.0)))


def polish(chosen: list[Option], options: list[list[Option]], total_demand: float,
           budget: float = INITIAL_TOKENS) -> list[Option]:
    """Best-improvement local search on the exact score.

    Moves: change one endpoint's option (including adding or dropping that endpoint), or
    drop one served endpoint and add an option for an unserved one. This removes any loss
    from the DP's fuel rounding. It only ever accepts improvements, so it cannot make the
    plan worse. Bounded by MAX_POLISH_ROUNDS and the soft deadline; stopping early just
    returns the best plan found so far.
    """
    current = {o.endpoint: o for o in chosen}
    best = plan_score(list(current.values()), total_demand, budget)

    def neighbours(plan: dict[int, Option]):
        for j in range(N_ENDPOINTS):
            for alt in [None, *options[j]]:
                if alt is plan.get(j):
                    continue
                trial = dict(plan)
                if alt is None:
                    trial.pop(j, None)
                else:
                    trial[j] = alt
                yield trial
        for out in plan:
            for j in range(N_ENDPOINTS):
                if j in plan:
                    continue
                for alt in options[j]:
                    trial = dict(plan)
                    del trial[out]
                    trial[j] = alt
                    yield trial

    for _ in range(MAX_POLISH_ROUNDS):
        if elapsed_s() > SOFT_DEADLINE_S:
            break
        step_best, step_plan = best, None
        for trial in neighbours(current):
            s = plan_score(list(trial.values()), total_demand, budget)
            if s > step_best + 1e-12:
                step_best, step_plan = s, trial
        if step_plan is None:
            break
        best, current = step_best, step_plan
    return sorted(current.values(), key=lambda o: o.endpoint)


# ---------------------------------------------------------------------------
# Stage 7: Verification against the reference scorer
# ---------------------------------------------------------------------------

def to_delivery(opt: Option) -> dict[str, object]:
    """Plan-JSON form of one option. Indices are the reference's 0-based ones."""
    return {
        "endpoint": int(opt.endpoint),
        "vehicle": opt.vehicle,
        "checkpoints": [int(c) for c in opt.checkpoints],
        "load_kg": float(opt.load_kg),
    }


def verify(inst: MapInstance, chosen: list[Option]) -> tuple[dict[str, object], list[Option]]:
    """Score the plan with the reference scorer. Returns its result and any options it
    rejected (none, if our geometry and quality agree with it as they should)."""
    result = score_manifold(inst, [to_delivery(o) for o in chosen])
    rejected = [o for o, m in zip(chosen, result["deliveries"]) if not m["feasible"]]
    return result, rejected


def solve_world(inst: MapInstance) -> tuple[list[Option], dict[str, object]]:
    """Full pipeline for one world: Stages 1-7."""
    geo = build_geometry(inst)
    options = route_options(inst, geo)
    total_demand = float(np.sum(inst.endpoint_demand_kg))

    for _ in range(MAX_REPAIR_ROUNDS):
        chosen = polish(knapsack(options, total_demand), options, total_demand)
        result, rejected = verify(inst, chosen)
        if not rejected:
            break
        print(f"  [{inst.name}] reference rejected {len(rejected)} deliveries; re-solving",
              file=sys.stderr)
        for bad in rejected:
            options[bad.endpoint] = [o for o in options[bad.endpoint] if o is not bad]
    else:
        chosen = [o for o in chosen if o not in rejected]
        result, _ = verify(inst, chosen)

    predicted = plan_score(chosen, total_demand)
    if abs(predicted - float(result["score"])) > 1e-6:
        print(f"  [{inst.name}] warning: predicted {predicted:.8f}, reference "
              f"{float(result['score']):.8f}", file=sys.stderr)
    return chosen, result


def baseline_world(inst: MapInstance) -> tuple[list[Option], dict[str, object]]:
    """Simple baseline for comparison: the cheapest feasible direct delivery (no
    checkpoints) for each endpoint, added greedily by value per token until the budget
    runs out."""
    geo = build_geometry(inst)
    cheapest = [min(opts, key=lambda o: o.cost) for opts in route_options(inst, geo, max_k=0) if opts]
    chosen, spent = [], 0.0
    for opt in sorted(cheapest, key=lambda o: o.value / o.cost, reverse=True):
        if spent + opt.cost <= INITIAL_TOKENS:
            chosen.append(opt)
            spent += opt.cost
    result, _ = verify(inst, chosen)
    return chosen, result


# ---------------------------------------------------------------------------
# Stage 8: Plan output and command-line entry point
# ---------------------------------------------------------------------------

def solve(seed: int, baseline: bool = False, verbose: bool = True) -> dict[str, list]:
    """Build the plan for all four worlds of one seed."""
    instances = generate_instances(seed)
    plan: dict[str, list] = {}
    for name, inst in instances.items():
        t0 = time.perf_counter()
        if elapsed_s() > HARD_DEADLINE_S:
            print(f"  [{name}] {elapsed_s():.0f} s elapsed; submitting no deliveries", file=sys.stderr)
            plan[name] = []
            continue
        use_baseline = baseline or elapsed_s() > SOFT_DEADLINE_S
        if use_baseline and not baseline:
            print(f"  [{name}] {elapsed_s():.0f} s elapsed; falling back to the baseline", file=sys.stderr)
        try:
            chosen, result = (baseline_world if use_baseline else solve_world)(inst)
        except Exception as exc:   # a crash in one world must not cost the other three
            print(f"  [{name}] solver failed ({exc!r}); submitting no deliveries", file=sys.stderr)
            plan[name] = []
            continue
        plan[name] = [to_delivery(o) for o in chosen]
        if verbose:
            vehicles = ", ".join(f"{o.vehicle}->{o.endpoint}" + (f" via {list(o.checkpoints)}" if o.checkpoints else "")
                                 for o in chosen)
            print(f"  {name:6s} score {float(result['score']):8.4f}  tokens {float(result['tokens_used']):7.2f}"
                  f"  deliveries {len(chosen):2d}  ({time.perf_counter() - t0:.2f} s)")
            print(f"         {vehicles}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=PUBLIC_SEED, help=f"instance seed (default {PUBLIC_SEED})")
    parser.add_argument("--out", type=Path, default=Path("plan.json"), help="output plan path")
    parser.add_argument("--baseline", action="store_true", help="write the simple baseline plan instead")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-world summary")
    args = parser.parse_args()

    if not args.quiet:
        print(f"{'Baseline' if args.baseline else 'Solver'} on seed {args.seed}")
    plan = solve(args.seed, baseline=args.baseline, verbose=not args.quiet)
    args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if not args.quiet and elapsed_s() < SOFT_DEADLINE_S:
        final = score_plan(generate_instances(args.seed), plan)["final_score"]
        print(f"Final weighted score C = {final:.6f}   (reference scorer)")
    if not args.quiet:
        print(f"Wrote {args.out}; total {elapsed_s():.2f} s of the {TIME_LIMIT_S:.0f} s limit")


if __name__ == "__main__":
    main()