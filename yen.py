"""Yen's K-shortest loopless paths on a small dense weighted digraph.

The graphs here are ~101 nodes (depot + 50 checkpoints + 50 endpoints) with a
sizeable fraction of finite edges, so a dense numpy Dijkstra beats a heap-based
sparse one: each relaxation step is a handful of vectorized ops on 101-element
arrays, and there is no priority-queue bookkeeping.

Paths are returned as node-index lists including both endpoints, cheapest first.
``np.inf`` in the weight matrix means "no edge".
"""

from __future__ import annotations

import numpy as np


def dijkstra(W: np.ndarray, src: int, dst: int) -> tuple[list[int] | None, float]:
    """Shortest path src -> dst. Returns (path, cost) or (None, inf)."""
    n = W.shape[0]
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    prev = np.full(n, -1, dtype=np.int32)
    done = np.zeros(n, dtype=bool)

    for _ in range(n):
        masked = np.where(done, np.inf, dist)
        u = int(np.argmin(masked))
        du = float(masked[u])
        if not np.isfinite(du):
            break
        if u == dst:
            break
        done[u] = True
        cand = du + W[u]
        upd = (cand < dist) & ~done
        dist[upd] = cand[upd]
        prev[upd] = u

    if not np.isfinite(dist[dst]):
        return None, np.inf

    path = [dst]
    while path[-1] != src:
        p = int(prev[path[-1]])
        if p < 0:
            return None, np.inf
        path.append(p)
    path.reverse()
    return path, float(dist[dst])


def hop_limited_shortest(
    W: np.ndarray, src: int, dst: int, max_hops: int
) -> tuple[list[int] | None, float]:
    """Cheapest src -> dst path using at most ``max_hops`` edges.

    Filtering an unconstrained Dijkstra result by hop count is wrong: it discards the
    path instead of finding the best one that fits. This is a layered DP over exact hop
    counts (Bellman-Ford by levels), so D[h, v] is the cheapest cost to reach v in
    exactly h edges and the answer is the best over h <= max_hops.
    """
    n = W.shape[0]
    D = np.full((max_hops + 1, n), np.inf)
    P = np.full((max_hops + 1, n), -1, dtype=np.int32)
    D[0, src] = 0.0
    cols = np.arange(n)

    for h in range(1, max_hops + 1):
        cand = D[h - 1][:, None] + W  # cand[u, v] = reach u in h-1 hops, then edge u->v
        u = np.argmin(cand, axis=0)
        D[h] = cand[u, cols]
        P[h] = np.where(np.isfinite(D[h]), u, -1)

    h_best = int(np.argmin(D[:, dst]))
    if not np.isfinite(D[h_best, dst]):
        return None, np.inf

    path = [dst]
    h = h_best
    while h > 0:
        p = int(P[h, path[-1]])
        if p < 0:
            return None, np.inf
        path.append(p)
        h -= 1
    path.reverse()
    if path[0] != src:
        return None, np.inf
    return path, float(D[h_best, dst])


def shortest(
    W: np.ndarray, src: int, dst: int, max_hops: int | None = None
) -> tuple[list[int] | None, float]:
    """Dijkstra, or the layered DP when a hop limit applies."""
    if max_hops is None:
        return dijkstra(W, src, dst)
    if max_hops < 1:
        return (None, np.inf)
    return hop_limited_shortest(W, src, dst, max_hops)


def path_cost(W: np.ndarray, path: list[int]) -> float:
    return float(sum(W[a, b] for a, b in zip(path[:-1], path[1:])))


def k_shortest_paths(
    W: np.ndarray,
    src: int,
    dst: int,
    K: int,
    max_hops: int | None = None,
) -> list[tuple[float, list[int]]]:
    """Yen's algorithm: up to K loopless src -> dst paths, cheapest first.

    ``max_hops`` caps the number of edges in a returned path, and is enforced inside
    every shortest-path call (initial and spur) rather than by discarding results.
    Every intermediate stop costs a full day of food decay, so deep paths are
    worthless here; capping also keeps the spur search small.
    """
    first, cost = shortest(W, src, dst, max_hops)
    if first is None:
        return []
    accepted: list[tuple[float, list[int]]] = [(cost, first)]
    candidates: list[tuple[float, list[int]]] = []

    while len(accepted) < K:
        prev_path = accepted[-1][1]

        for i in range(len(prev_path) - 1):
            spur = prev_path[i]
            root = prev_path[: i + 1]
            Wm = W.copy()

            # Ban the edges that would retrace an already-accepted path sharing
            # this root, so the spur search is forced somewhere new.
            for _, taken in accepted:
                if len(taken) > i + 1 and taken[: i + 1] == root:
                    Wm[taken[i], taken[i + 1]] = np.inf

            # Remove the root's interior nodes to keep the result loopless.
            for node in root[:-1]:
                Wm[node, :] = np.inf
                Wm[:, node] = np.inf

            # The root already consumed i edges, so the spur gets the rest.
            spur_budget = None if max_hops is None else max_hops - i
            spur_path, _ = shortest(Wm, spur, dst, spur_budget)
            if spur_path is None:
                continue

            total = root[:-1] + spur_path
            if any(total == p for _, p in accepted) or any(total == p for _, p in candidates):
                continue
            candidates.append((path_cost(W, total), total))

        if not candidates:
            break
        candidates.sort(key=lambda t: t[0])
        accepted.append(candidates.pop(0))

    return accepted
