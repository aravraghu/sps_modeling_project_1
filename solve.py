"""Routing on Curved Worlds -- competition entry point.

    python solve.py --seed N --out plan.json

This file is the required grading interface and holds only the command line. The work
lives in two packages:

    include/    shared declarations -- tuned constants (with their derivations) and the
                Candidate type that crosses module boundaries
    src/        implementation -- geometry, routing, the per-world pipeline, selection,
                and verification against the reference scorer

Pipeline, per world and per vehicle:

  1. Build a dense node graph over {depot} + 50 checkpoints + 50 endpoints. Endpoint
     nodes are sinks, because the scorer's route format is [depot, *checkpoints,
     endpoint] -- a route cannot pass through another endpoint. An edge exists only if
     the leg is within the mode's max leg and, for ground modes, misses every ocean.
  2. Gate the vehicle (see include/constants.py): horses only reach nodes within 400 km
     of the depot; jets are only tried for islands, unreachable endpoints, or badly
     degraded ground options.
  3. Run Yen's algorithm for K loopless depot -> endpoint paths.
  4. Score every candidate with the exact reference quality formula, giving a
     (tokens, fuel, quality) triple each.
  5. Pareto-filter per endpoint, then choose a subset within the 1000-token budget,
     pricing fuel to account for the efficiency multiplier (see METHOD.md).

Useful flags:
    --self-test   check the vectorized geometry against the reference implementation
    --verify      re-score the finished plan with the reference scorer
    --report      per-world diagnostics: candidate counts, gating, timings
    --knapsack    bnb (default) | dp | frontier -- three solvers that should agree
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optimal_transport as ot
from include.constants import (
    DEFAULT_K,
    DEFAULT_MAX_CHECKPOINTS,
    DEFAULT_MAX_HOPS,
    METRICS,
    VEHICLE_ORDER,
    WORLDS,
)
from src.pipeline import solve_all
from src.verify import self_test, verify_plan

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
    p.add_argument("--knapsack", choices=("bnb", "dp", "frontier"), default="bnb",
                   help="bnb: priced fuel + branch and bound (default, see METHOD.md); "
                        "dp: priced fuel + gridded table; "
                        "frontier: exact fuel/quality scan, for cross-checking")
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
            print(f"    frontier/lambda trace: {[f'{x:.3e}' for x in r['lambda_trace']]}")
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
