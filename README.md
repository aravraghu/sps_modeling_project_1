# SPS Modeling Project #1: Routing in Curved Worlds

Run `python solve.py --seed N --out plan.json` to produce a delivery plan for all four
worlds on any integer seed `N` (needs Python 3.9+ and NumPy — `pip install numpy` is the
only dependency), then score it with the organizers' unmodified reference scorer. The
seed defines the whole instance — checkpoint and endpoint positions, demands, oceans,
islands — so the same `N` must go to both commands; mismatching them scores the plan
against a different world and fails quietly rather than erroring. Add `--verify` to score
in-process with the same instance and avoid that, `--self-test` to check the vectorized
geometry against the reference implementation, `--report` for per-world diagnostics, or
`--knapsack {bnb,dp,frontier}` to switch between three independent selection solvers that
should all agree. `METHOD.md` explains how the score's fuel-efficiency multiplier is
handled; the implementation is in `src/`, and the tuned constants, each with the
derivation that fixes its value, are in `include/constants.py`.

```bash
python solve.py --seed 424242 --out plan.json                     # write a plan
python optimal_transport.py --seed 424242 --score-plan plan.json  # score it
python solve.py --seed 424242 --out plan.json --verify            # both in one step
python solve.py --seed 424242                                     # scores only, no file
```
