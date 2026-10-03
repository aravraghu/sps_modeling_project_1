# How we handle the fuel multiplier

This document explains one specific part of the solver: the part that decides which
deliveries to make when the score formula punishes fuel in an awkward way. It assumes no
optimization background.

---

## 1. There are two separate problems. Fuel only lives in the second one.

This is the thing to get straight first, because the two problems use different
algorithms and it is easy to mix them up.

```
┌─────────────────────────────────────────────────────────────────────┐
│  PROBLEM 1 — ROUTING            "How do I get to endpoint 17?"      │
│                                                                      │
│  input    a map of legal legs between the depot, 50 checkpoints      │
│           and 50 endpoints                                           │
│  output   for each endpoint: some good routes, with their exact      │
│           distance, travel time, quality and cost                    │
│  solved by   Dijkstra / Bellman-Ford-by-hop-count / Yen's            │
│                                                                      │
│  Fuel is IRRELEVANT here. One route's fuel is just a number.         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │  ~97 candidate deliveries per world
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PROBLEM 2 — SELECTION   "Which 8 of the 50 endpoints do I serve?"   │
│                                                                      │
│  input    the candidate deliveries from Problem 1                    │
│  output   a chosen subset that fits in the 1000-token budget         │
│  solved by   knapsack + fuel pricing                                 │
│                                                                      │
│  THE FUEL PROBLEM LIVES ENTIRELY HERE.                               │
└─────────────────────────────────────────────────────────────────────┘
```

Fuel cannot be dealt with in Problem 1 because the penalty depends on the **total** fuel
across every delivery you end up making — which is unknowable while you are routing a
single endpoint.

Everything below is about Problem 2 only.

---

## 2. Why the score formula is awkward

For one world the score is

```
score  =  100 × Q × (0.85 + 0.15 × (1 − F/1000))
       =  100 × Q × (1 − 0.00015 × F)              for F ≤ 1000
```

- **Q** = the fraction of demand-weighted food that actually arrives (0 to 1)
- **F** = total **fuel** tokens spent (note: fuel only — dispatch cost is not in here)

Both `Q` and `F` are simple sums. If you make three deliveries:

```
Q = q₁ + q₂ + q₃                F = f₁ + f₂ + f₃
```

But the score is their **product**, and that is the whole difficulty:

```
score = 100 × (q₁+q₂+q₃) × (1 − 0.00015 × (f₁+f₂+f₃))
                  └── sum ──┘          └───── sum ─────┘
                           └──── product ────┘      ← NOT a sum
```

**Why that matters.** The standard way to pick items under a budget is a *knapsack*
algorithm, and every knapsack algorithm needs each item to have a fixed value that you
simply add up. Here, delivery 1's worth depends on how much fuel deliveries 2 and 3
happen to burn — because their fuel shrinks the multiplier, and the shrunken multiplier
applies to delivery 1's quality too. **The deliveries are tangled together.** No fixed
per-delivery value exists, so no plain knapsack applies.

---

## 3. What a knapsack does, when it *is* a sum

Set the multiplier aside for a second. If the score were just `Q`, the problem would be:

> 50 endpoints, each with candidate routes that have a token cost and a quality value.
> Pick at most one route per endpoint, total cost ≤ 1000, maximize total value.

We solve that **exactly** by branch and bound: explore the choices as a tree, and
whenever the best conceivable completion of a partial plan is already worse than a
complete plan we have in hand, abandon that whole branch unexamined. It works directly on
the real-valued token costs — no rounding anywhere. This is `solve_bnb` in `mckp.py`, and
it is validated against exhaustive search on 400 random instances (400/400 exact).

So we have an exact engine for sums. We just need to turn our product into a sum.

---

## 4. The idea: put a price on fuel

Charge a made-up price **λ** (lambda) per token of fuel, and give each delivery the value

```
adjusted value  =  (quality it delivers)  −  λ × (fuel it burns)
```

That **is** a fixed per-delivery number, so the deliveries are untangled and branch and
bound applies again. The entire trick is choosing the right λ.

### The picture that makes λ click

Draw every possible plan as a dot: fuel across, quality up. Real numbers from seed 67's
disk world:

```
  Q (quality delivered)
          │
  0.2282  ┤                                      ● A   F=171.4  Q=0.228173
          │
  0.2280  ┤              ● B   F=151.7  Q=0.227970
          │
  0.2278  ┤
          │
  0.2276  ┤   ● C   F=140.0  Q=0.22720   (made up, for illustration)
          │
          └───┬────────┬────────┬────────┬────────┬───── F (fuel tokens)
             140      150      160      170      180
```

Maximizing `Q − λF` has a clean geometric meaning: **tilt a ruler to slope λ, slide it
down from the top-left, and the first dot it touches wins.**

```
   λ = 0  (fuel is free)              λ = 3.5e-5  (fuel is expensive)
   ruler is horizontal ───────        ruler is tilted      ╱
                                                          ╱
        ════════════════● A  ← wins              ╱── ● B  ← wins
              ● B                              ╱   ● A  (touched later)
     ● C                                     ╱● C
   highest Q wins, whatever the fuel         low fuel now worth paying for
```

So λ is a **dial**: turn it up and the solver trades quality away to save fuel.

Check the arithmetic on the two real plans:

| plan | Q | F | `Q − λF` at λ=0 | `Q − λF` at λ=3.5e-5 |
|---|---|---|---|---|
| **A** | 0.228173 | 171.4 | **0.228173** ← wins | 0.222174 |
| **B** | 0.227970 | 151.7 | 0.227970 | **0.222661** ← wins |

The two plans swap places at `λ = (Q_A − Q_B)/(F_A − F_B) = 0.000203/19.69 = 1.03e-5`.
Above that price, B is preferred.

And B really is the better plan under the true formula:

```
A:  100 × 0.228173 × (1 − 0.00015 × 171.4)  =  22.2307
B:  100 × 0.227970 × (1 − 0.00015 × 151.7)  =  22.2783   ← better
```

B carries **less** food but burns **20 fewer fuel tokens**, and the fuel saving more than
pays for the lost food. That is exactly the trade we need λ to find for us.

---

## 5. Choosing λ: the fixed point

What *should* λ be? Think about what one extra fuel token actually costs. It shrinks the
multiplier by `0.00015`, and that shrink applies to your whole `Q`. So the honest price is

```
            0.00015 × Q
   λ  =  ─────────────────
          1 − 0.00015 × F
```

which depends on `Q` and `F` — the very things we are trying to compute. **Circular.**

The way out is to guess, then let the answer correct the guess, and repeat until the
guess stops changing. A value that comes back out unchanged is called a **fixed point**.

```
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                                          │
   ┌─────────┐    ┌──────────────┐    ┌────────────────────┐
   │ guess λ │───▶│ solve for the│───▶│ read off its Q and │
   └─────────┘    │ best plan at │    │ F, compute the λ   │
        ▲         │ that price   │    │ it implies         │
        │         └──────────────┘    └────────────────────┘
        │                                          │
        └───── different? use it as the new guess ─┘

                   same? ──▶  STOP. that is the fixed point.
```

### The real trace, seed 67, disk world

Straight from `solve.py --report`:

```
lambda trace: 0.000e+00  ->  3.513e-05  ->  3.499e-05  ->  (converged)
```

Step by step, with the arithmetic you can check by hand:

| round | λ in | plan it picks | Q | F | λ it implies |
|---|---|---|---|---|---|
| 1 | `0` | A | 0.228173 | 171.39 | `0.00015×0.228173 / (1−0.00015×171.39)` = **3.513e-5** |
| 2 | `3.513e-5` | B | 0.227970 | 151.70 | `0.00015×0.227970 / (1−0.00015×151.70)` = **3.499e-5** |
| 3 | `3.499e-5` | B | 0.227970 | 151.70 | **3.499e-5** — unchanged, stop |

Two rounds. The price settled, the plan improved from 22.2307 to **22.2783**.

---

## 6. The sweep, and why we need it

Pricing has a genuine weak spot. Go back to the ruler picture: a sliding straight edge can
only ever touch dots on the **outer boundary** of the cloud. A dot tucked just inside that
boundary is invisible to *every* possible λ — no matter how you tilt the ruler, something
else is always touched first.

```
   Q │                          ● ← on the boundary: some λ picks it
     │                    ●
     │              ●
     │         ○  ← tucked inside: NO λ ever picks it first,
     │      ●       even though it might be the true best plan
     │   ●
     └─────────────────────────── F
```

The reason such a dot can still be best is that the true score contours are gently
**curved** (`Q × (1 − 0.00015F)`), while the ruler is **straight**. A straight edge cannot
reach into a dent that a curve can.

So after the fixed point converges we also try λ at `0, 0.25×, 0.5×, 0.75×, 1.25×, 1.5×,
2×, 3×, 5×` the converged value. Each price tilts the ruler differently and surfaces a
different candidate plan. **Every resulting plan is then scored with the exact formula and
we keep the best**, so the method can never report a number it did not genuinely achieve.

### How much does this weak spot cost? Almost nothing — and here is why

The curvature is what creates the gap, and here the curvature is very weak: the
multiplier only ever ranges over `0.85 … 1.00`. The objective is therefore *nearly*
straight, so the boundary captures nearly every plan worth having.

Measured, not assumed:

| test | result |
|---|---|
| 3000 random instances vs exhaustive search | **0 misses** |
| 4000 *adversarial* instances (tight budgets, extreme fuel spreads) | **1 miss in 4000**, short by 0.000125 |
| seeds 1–100, real instances | agrees with the exact method on **every seed** |

For context, 0.000125 is about **one millionth** of a typical world score of ~21.

---

## 7. Why we chose this over the exact alternative

There is a provably exact method, implemented as `--knapsack frontier`. It makes fuel an
explicit dimension: for each fuel ceiling, find the best plan under that ceiling, then
score every result exactly. The optimum must be one of those, so nothing can be missed.

We did **not** make it the default:

| | fuel pricing (default) | frontier scan |
|---|---|---|
| guarantee | none, but measured error ≤ 1e-4 | provably exact |
| score, seeds 1–100 | identical | identical |
| explaining it | one dial, one loop | needs Pareto frontiers |

Since the scores are identical on every seed we tested, the choice costs nothing, and we
preferred the method that is easier to present. The exact method stays in the codebase as
a cross-check: `--knapsack frontier` and `--knapsack dp` should always agree with the
default, and we verify that they do.

---

## 8. Where this lives in the code

| file | what |
|---|---|
| `mckp.py` → `solve()` | the fixed point and the λ sweep |
| `mckp.py` → `solve_bnb()` | exact branch and bound for a fixed λ |
| `mckp.py` → `true_objective()` | the exact score formula; every candidate plan goes through it |
| `mckp.py` → `solve_frontier()` | the exact alternative, for cross-checking |
| `mckp.py` → `pareto_filter()` | drops candidate routes that are beaten on quality, dispatch *and* fuel at once |

One detail worth noting: `pareto_filter` compares **dispatch and fuel separately** rather
than just total cost. That is deliberate — the multiplier depends on fuel alone, so a
route that is pricier overall but burns less fuel is not necessarily worse.

---

## 9. One-paragraph summary

The score multiplies total quality by a penalty on total fuel, which tangles every
delivery together and defeats a plain knapsack. We untangle them by charging a price λ per
unit of fuel, which makes each delivery's worth a fixed number again. The correct price
depends on the answer, so we guess, let the answer correct the guess, and repeat until the
price stops moving — a fixed point, reached in two or three rounds. Because a fixed price
can overlook plans sitting just inside the boundary of what prices can reach, we also
sweep prices around the converged value, score every resulting plan with the exact
formula, and keep the best. Measured against exhaustive search, this finds the true
optimum on all 3000 random test instances and misses by at most 0.000125 in one of 4000
adversarial ones.
