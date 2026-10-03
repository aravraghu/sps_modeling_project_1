"""Routing on Curved Worlds -- solver entry point.

    python solve.py --seed N --out plan.json

Pipeline, per world and per vehicle:

  1. Build a dense node graph over {depot} + 50 checkpoints + 50 endpoints. Endpoint
     nodes are sinks (no outgoing edges) because the scorer's route format is
     [depot, *checkpoints, endpoint] -- a route cannot pass through another endpoint.
     An edge exists only if the leg is within the mode's max leg, and, for horses and
     trucks, only if the leg's geodesic misses every ocean region.
  2. Run Yen's algorithm for K loopless depot -> endpoint paths under a choice of edge
     metric (distance, tokens, or time; see --metric).
  3. Score every candidate path with the exact reference quality formula, giving a
     (tokens, quality) pair per candidate.
  4. Pareto-filter per endpoint, then solve a multiple-choice knapsack against the
     1000-token budget, accounting for the global fuel-efficiency multiplier.

As K grows this enumerates more of the feasible route space, so the candidate set
approaches "all simple paths" and the knapsack becomes exact over it. See NOTES.md for
why only simple paths matter, why loads are forced, and why duplicate deliveries never
help.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import geom_fast
import mckp
import optimal_transport as ot
from mckp import Candidate
from yen import k_shortest_paths

WORLDS = ("disk", "sphere", "torus", "klein")
VEHICLE_ORDER = ("horse", "truck", "jet")
METRICS = ("tokens", "time")  # distance omitted: provably identical ranking to tokens

# Hop cap on routes: hops == checkpoints + 1, so 7 hops allows up to 6 checkpoints.
# This is a real constraint enforced inside the shortest-path search, not a post-filter
# (see yen.hop_limited_shortest). Intermediate stops cost a full day of decay each, so
# the sensitivity study finds nothing beyond 1 checkpoint is ever selected.
DEFAULT_MAX_CHECKPOINTS = 6
DEFAULT_MAX_HOPS = DEFAULT_MAX_CHECKPOINTS + 1

# Paths per (endpoint, vehicle) from Yen's algorithm.
DEFAULT_K = 15


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

        self.dist = geom_fast.pairwise_geodesic(instance, self.nodes, self.nodes)
        np.fill_diagonal(self.dist, 0.0)

        # Ocean crossing is a property of the leg, not the vehicle, so build it once.
        iu, ju = np.triu_indices(self.n_nodes, k=1)
        crossed = geom_fast.legs_cross_ocean(
            instance, self.nodes[iu], self.nodes[ju]
        )
        self.ocean = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        self.ocean[iu, ju] = crossed
        self.ocean[ju, iu] = crossed

        # Direct depot -> endpoint distance, the D0 in rho = D / D0.
        self.direct = self.dist[self.depot, self.ep0 :].copy()

    def ep_node(self, ep: int) -> int:
        return self.ep0 + ep

    def weights(self, vehicle: str, metric: str) -> np.ndarray:
        """Edge weight matrix for one mode under one metric; inf means no edge."""
        spec = ot.VEHICLES[vehicle]
        allowed = self.dist <= spec.max_leg_m + 1e-9
        if not spec.can_cross_ocean:
            allowed &= ~self.ocean
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
        W = tables.weights(vehicle, metric)
        reachable = 0
        for ep in range(tables.n_ep):
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
    groups = [mckp.pareto_filter(c) for c in per_ep if c]
    kept_total = sum(len(g) for g in groups)

    t2 = time.perf_counter()
    chosen, info, lam_trace = mckp.solve(
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


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def self_test(seed: int, n_pairs: int = 400, verbose: bool = True) -> bool:
    """Check the vectorized geometry against the reference, world by world.

    The reference functions are authoritative for scoring, so agreement here is what
    licenses using the fast versions inside the optimizer.
    """
    rng = np.random.default_rng(12345)
    instances = ot.generate_instances(seed)
    ok = True

    for name in WORLDS:
        inst = instances[name]
        pts = np.vstack(
            [np.atleast_2d(inst.start), inst.checkpoints, inst.endpoints, inst.ocean_centers]
        )
        ia = rng.integers(0, len(pts), n_pairs)
        ib = rng.integers(0, len(pts), n_pairs)
        A, B = pts[ia], pts[ib]

        fast_d = np.array(
            [geom_fast.pairwise_geodesic(inst, A[i : i + 1], B[i : i + 1])[0, 0]
             for i in range(n_pairs)]
        )
        ref_d = np.array([ot.geodesic_distance(inst, A[i], B[i]) for i in range(n_pairs)])
        d_err = float(np.max(np.abs(fast_d - ref_d)))

        # Symmetry, as the prompt's Part I asks: d(A,B) == d(B,A).
        rev_d = np.array([ot.geodesic_distance(inst, B[i], A[i]) for i in range(n_pairs)])
        sym_err = float(np.max(np.abs(ref_d - rev_d)))

        fast_o = geom_fast.legs_cross_ocean(inst, A, B)
        ref_o = np.array([ot.leg_crosses_ocean(inst, A[i], B[i]) for i in range(n_pairs)])
        o_mismatch = int(np.sum(fast_o != ref_o))

        good = d_err < 1e-6 and o_mismatch == 0 and sym_err < 1e-6
        ok &= good
        if verbose:
            flag = "ok " if good else "FAIL"
            print(
                f"  [{flag}] {name:6s} max distance err={d_err:.3e}  "
                f"symmetry err={sym_err:.3e}  ocean mismatches={o_mismatch}/{n_pairs}"
            )

    # Candidate scoring must reproduce the reference scorer exactly.
    if verbose:
        print("  candidate metrics vs ot.evaluate_delivery:")
    for name in WORLDS:
        inst = instances[name]
        tables = WorldTables(inst)
        worst_q = 0.0
        worst_t = 0.0
        checked = 0
        for vehicle in VEHICLE_ORDER:
            W = tables.weights(vehicle, "time")
            for ep in rng.choice(tables.n_ep, size=12, replace=False):
                paths = k_shortest_paths(W, tables.depot, tables.ep_node(int(ep)), 3,
                                         max_hops=None)
                for _, path in paths:
                    cps = tuple(n - tables.cp0 for n in path[1:-1])
                    dist = float(sum(tables.dist[a, b] for a, b in zip(path[:-1], path[1:])))
                    cand = score_route(tables, int(ep), vehicle, cps, dist, 0.0)
                    if cand is None:
                        continue
                    m = ot.evaluate_delivery(
                        inst, endpoint=int(ep), vehicle=vehicle,
                        checkpoint_indices=cps, load_kg=cand.load_kg,
                    )
                    if not m.feasible:
                        print(f"  [FAIL] {name} ep{ep} {vehicle} {cps}: ref says {m.reason}")
                        ok = False
                        continue
                    worst_q = max(worst_q, abs(m.endpoint_quality_fraction - cand.quality))
                    worst_t = max(worst_t, abs(m.total_tokens - cand.tokens))
                    checked += 1
        good = worst_q < 1e-9 and worst_t < 1e-9
        ok &= good
        if verbose:
            flag = "ok " if good else "FAIL"
            print(
                f"  [{flag}] {name:6s} {checked:4d} routes  max q err={worst_q:.2e}  "
                f"max token err={worst_t:.2e}"
            )
    return bool(ok)


def verify_plan(seed: int, plan: dict) -> dict:
    """Re-score the plan with the authoritative reference scorer."""
    return ot.score_plan(ot.generate_instances(seed), plan)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=ot.PUBLIC_SEED)
    p.add_argument("--out", type=Path, default=None, help="write plan JSON here")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help=f"paths per (endpoint, vehicle) from Yen's (default: {DEFAULT_K})")
    p.add_argument("--metric", choices=METRICS, default="time",
                   help="Yen's edge metric (default: time)")
    p.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS,
                   help=f"cap edges per route; hops = checkpoints + 1 "
                        f"(default: {DEFAULT_MAX_HOPS}, i.e. {DEFAULT_MAX_CHECKPOINTS} checkpoints)")
    p.add_argument("--knapsack", choices=("bnb", "dp"), default="bnb",
                   help="exact branch and bound on real costs, or the gridded DP")
    p.add_argument("--no-fuel-correction", dest="fuel_correction",
                   action="store_false", default=True,
                   help="skip the fuel-efficiency multiplier optimization")
    p.add_argument("--report", action="store_true", help="print a diagnostic breakdown")
    p.add_argument("--verify", action="store_true", help="re-score with the reference scorer")
    p.add_argument("--self-test", action="store_true", help="check fast geometry vs reference")
    args = p.parse_args()

    if args.self_test:
        print(f"Self-test (seed={args.seed}):")
        ok = self_test(args.seed)
        print("All checks passed." if ok else "SELF-TEST FAILED.")
        if not args.out:
            raise SystemExit(0 if ok else 1)

    t0 = time.perf_counter()
    plan, results = solve_all(args.seed, K=args.k, metric=args.metric,
                              max_hops=args.max_hops,
                              fuel_correction=args.fuel_correction,
                              method=args.knapsack)
    elapsed = time.perf_counter() - t0

    if args.out:
        args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    internal = {n: results[n]["info"]["score"] for n in WORLDS}
    final = ot.weighted_final_score(internal)
    print(f"seed={args.seed}  K={args.k}  metric={args.metric}  "
          f"runtime={elapsed:.2f}s  (limit 120s)")
    for n in WORLDS:
        i = results[n]["info"]
        print(f"  {n:6s}: C={i['score']:7.4f}  Q={i['coverage_quality']:.4f}  "
              f"tokens={i['tokens']:7.2f}  fuel={i['fuel_tokens']:6.2f}  "
              f"deliveries={i['n_deliveries']:2d}")
    print(f"Final weighted score C = {final:.6f}")
    if args.out:
        print(f"Wrote {args.out}")

    if args.report:
        print("\nDiagnostics:")
        for n in WORLDS:
            r = results[n]
            tm = r["timing"]
            print(f"  {n}:")
            for v in VEHICLE_ORDER:
                s = r["stats"][v]
                print(f"    {v:5s}: reachable endpoints={s['endpoints']:2d}  "
                      f"paths found={s['paths']:3d}  feasible candidates={s['feasible']:3d}")
            used = {}
            for c in r["chosen"]:
                used[c.vehicle] = used.get(c.vehicle, 0) + 1
            n_cp_used = sum(1 for c in r["chosen"] if c.checkpoints)
            print(f"    selected: {used}  with-checkpoints={n_cp_used}")
            print(f"    candidates {r['candidates_raw']} -> {r['candidates_kept']} after Pareto")
            print(f"    lambda trace: {[f'{x:.3e}' for x in r['lambda_trace']]}")
            print(f"    timing: geometry={tm['geometry']:.2f}s "
                  f"candidates={tm['candidates']:.2f}s knapsack={tm['knapsack']:.2f}s")

    if args.verify:
        res = verify_plan(args.seed, plan)
        print("\nReference scorer (authoritative):")
        agree = True
        for n in WORLDS:
            w = res["worlds"][n]
            delta = abs(float(w["score"]) - internal[n])
            agree &= delta < 1e-6 and bool(w["budget_valid"])
            print(f"  {n:6s}: C={float(w['score']):7.4f}  "
                  f"tokens={float(w['tokens_used']):7.2f}  "
                  f"budget_valid={bool(w['budget_valid'])}  |delta|={delta:.2e}")
        print(f"Reference final C = {float(res['final_score']):.6f}")
        print("Internal and reference scores agree."
              if agree else "MISMATCH between internal and reference scores.")


if __name__ == "__main__":
    main()
