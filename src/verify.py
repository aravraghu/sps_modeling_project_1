"""Verification against the authoritative reference implementation.

``self_test`` checks the vectorized geometry and the fast candidate scoring against
``optimal_transport.py`` -- agreement is what licenses using the fast versions inside
the optimizer. ``verify_plan`` re-scores a finished plan with the reference scorer."""

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
from src.pipeline import WorldTables, score_route

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
            [geometry.pairwise_geodesic(inst, A[i : i + 1], B[i : i + 1])[0, 0]
             for i in range(n_pairs)]
        )
        ref_d = np.array([ot.geodesic_distance(inst, A[i], B[i]) for i in range(n_pairs)])
        d_err = float(np.max(np.abs(fast_d - ref_d)))

        # Symmetry, as the prompt's Part I asks: d(A,B) == d(B,A).
        rev_d = np.array([ot.geodesic_distance(inst, B[i], A[i]) for i in range(n_pairs)])
        sym_err = float(np.max(np.abs(ref_d - rev_d)))

        fast_o = geometry.legs_cross_ocean(inst, A, B)
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


