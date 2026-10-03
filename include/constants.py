"""Tuned and derived constants, each with the reasoning that fixes its value.

Numbers here are derived from the reference constants in ``optimal_transport.py``, not
guessed. Where a value is a judgement call rather than a proof, the comment says so.
"""

from __future__ import annotations

WORLDS = ("disk", "sphere", "torus", "klein")
VEHICLE_ORDER = ("horse", "truck", "jet")  # ground first: the jet gate reads their results
METRICS = ("tokens", "time")  # distance omitted: provably the same ranking as tokens

# --- Routing search breadth -------------------------------------------------

# Paths per (endpoint, vehicle) from Yen's algorithm. Measured: K=1 and K=15 give
# identical scores on every seed tested, because for a fixed vehicle and hop count the
# shortest route beats every other on quality AND cost at once, so the alternatives are
# dominated. Kept at 15 for margin; 1 would be defensible.
DEFAULT_K = 15

# Hops == checkpoints + 1. Each checkpoint costs a flat day (~9% of quality), so deep
# routes are worthless; measured, nothing beyond 1 checkpoint is ever selected, and
# capping at 0 costs ~1.05 of C on the sphere.
DEFAULT_MAX_CHECKPOINTS = 6
DEFAULT_MAX_HOPS = DEFAULT_MAX_CHECKPOINTS + 1

# --- Vehicle gating ---------------------------------------------------------

# A horse covers 50 km per leg at 50 km/day, so reaching D metres costs ceil(D/50km)
# legs plus ceil(D/50km)-1 checkpoint days even with perfectly spaced checkpoints.
# Against a direct truck on value-per-token under that best case, the horse wins out to
# ~320 km at low demand and loses at every demand beyond ~350 km. A useful horse route is
# therefore at most ~350 km long, so every node on one lies within that of the depot.
# 400 km is a safe superset and shrinks the horse graph from 101 nodes to a handful.
HORSE_MAX_RADIUS_M = 400_000.0

# A jet never beats a truck on value-per-token at any distance: the ratio peaks at 0.84
# (3000 km, against a 4-hop truck) and is ~0.4 at short range, because 300 tokens of
# dispatch buys three jets against ten trucks. Its niche is legality, not distance --
# islands, and endpoints no ground vehicle can reach, both of which are always kept.
# Beyond those we also keep it where the best ground option is badly degraded, since
# that is where its speed advantage is largest. This last clause is a heuristic, not a
# proof, so the gates are re-verified against ungated scores seed by seed.
JET_GROUND_QUALITY_GATE = 0.65

# --- Selection --------------------------------------------------------------

# Cost grid for the gridded DP only (``--knapsack dp``). The default solver works on
# real-valued costs and needs no grid. Costs are rounded UP so a DP-feasible plan is
# truly feasible, which means the grid must be fine: at 1/100 token, a plan truly
# costing 199.98 of 200 rounds to 200.01 and is wrongly rejected, losing a whole
# delivery. At 1/1000 and ~10 deliveries the worst case is 0.01 tokens.
COST_SCALE = 1000

# Multiples of the converged shadow price to probe after the fixed point converges.
# A single price can overlook plans just inside the boundary of what prices reach;
# see METHOD.md section 6.
LAMBDA_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)

# The score's efficiency multiplier is (1 - MULTIPLIER_SLOPE * fuel), from
# 0.85 + 0.15 * (1 - fuel/1000).
MULTIPLIER_SLOPE = 0.00015
