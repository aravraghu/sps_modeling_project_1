# SPS Modeling Project #1: Routing in Curved Worlds

Run `python solve.py --seed 67 --out plan.json` to produce a delivery plan for all four
worlds (needs Python 3.9+ and NumPy — `pip install numpy` is the only dependency), then
`python optimal_transport.py --seed 67 --score-plan plan.json` to score that plan with
the organizers' unmodified reference scorer; any seed works in place of 67. Add
`--verify` to have `solve.py` re-score its own output, `--self-test` to check the
vectorized geometry against the reference implementation, `--report` for per-world
diagnostics, or `--knapsack {bnb,dp,frontier}` to switch between three independent
selection solvers that should all agree. `METHOD.md` explains how the score's
fuel-efficiency multiplier is handled; the implementation is in `src/` and the tuned
constants, each with the derivation that fixes its value, are in `include/constants.py`.
