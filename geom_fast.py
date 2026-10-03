"""Vectorized geometry for all four worlds.

Every function here mirrors the reference implementation in ``optimal_transport.py``
exactly -- same 96-point geodesic sampling, same clipping branches, same ``<=``
comparison against ocean radii, same tie-breaking order in the Klein lift search --
but operates on whole arrays of points at once.

The reference ``leg_crosses_ocean`` is authoritative for official scoring, and it is
also far too slow to call inside an optimizer loop: one call costs ~96 x n_ocean
Python-level ``geodesic_distance`` evaluations. Building the 101x101 leg table for one
world needs ~5k of those calls, which is ~1.5M geodesic evaluations. These batched
equivalents do the same work in numpy. ``solve.py --self-test`` checks that they agree
with the reference on random inputs in every world.
"""

from __future__ import annotations

import numpy as np

import optimal_transport as ref

N_OCEAN_SAMPLES = 96  # must match the default n in ref._geodesic_samples


# ---------------------------------------------------------------------------
# Pairwise geodesic distance
# ---------------------------------------------------------------------------

def _klein_pairwise(A: np.ndarray, B: np.ndarray, side: float) -> np.ndarray:
    """Mirror of ref._klein_best_lift's minimum, over all pairs at once."""
    best = np.full((len(A), len(B)), np.inf)
    ax = A[:, 0][:, None]
    ay = A[:, 1][:, None]
    for n in range(-2, 3):
        reflected_x = ((-1.0) ** n) * B[:, 0][None, :]
        dy = (B[:, 1][None, :] + n * side) - ay
        for m in range(-2, 3):
            dx = (reflected_x + m * side) - ax
            np.minimum(best, np.sqrt(dx * dx + dy * dy), out=best)
    return best


def pairwise_geodesic(instance: ref.MapInstance, A, B) -> np.ndarray:
    """(n,d) x (m,d) -> (n,m) geodesic distances. Mirrors ref.geodesic_distance."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    name = instance.name

    if name == "disk":
        return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)

    if name == "sphere":
        r = instance.scale_m
        cos_angle = (A @ B.T) / (r * r)
        return r * np.arccos(np.clip(cos_angle, -1.0, 1.0))

    side = instance.scale_m

    if name == "torus":
        d = B[None, :, :] - A[:, None, :]
        d = (d + side / 2.0) % side - side / 2.0
        return np.linalg.norm(d, axis=-1)

    if name == "klein":
        return _klein_pairwise(A, B, side)

    raise ValueError(f"Unsupported manifold: {name}")


# ---------------------------------------------------------------------------
# Batched geodesic sampling (for the ocean test)
# ---------------------------------------------------------------------------

def _klein_best_lift_batch(A: np.ndarray, B: np.ndarray, side: float) -> np.ndarray:
    """Closest image of B[i] to A[i] in the covering space, for every i.

    Iteration order and the strict ``<`` update match ref._klein_best_lift so that
    ties resolve identically.
    """
    best_d2 = np.full(len(A), np.inf)
    out = np.zeros((len(A), 2))
    for n in range(-2, 3):
        reflected_x = ((-1.0) ** n) * B[:, 0]
        y = B[:, 1] + n * side
        dy = y - A[:, 1]
        for m in range(-2, 3):
            cx = reflected_x + m * side
            dx = cx - A[:, 0]
            d2 = dx * dx + dy * dy
            upd = d2 < best_d2
            best_d2[upd] = d2[upd]
            out[upd, 0] = cx[upd]
            out[upd, 1] = y[upd]
    return out


def _reduce_klein_cover_batch(P: np.ndarray, side: float) -> np.ndarray:
    """Mirror of ref._reduce_klein_cover over a (..., 2) array."""
    x = P[..., 0]
    y = P[..., 1]
    n = np.floor(y / side)
    y = y - n * side
    x = np.where(np.mod(n, 2) != 0, -x, x)
    x = np.mod(x, side)
    return np.stack([x, y], axis=-1)


def geodesic_samples_batch(
    instance: ref.MapInstance,
    A: np.ndarray,
    B: np.ndarray,
    n_samples: int = N_OCEAN_SAMPLES,
) -> np.ndarray:
    """(k,d) x (k,d) -> (k,n_samples,d) samples along each geodesic A[i] -> B[i].

    Mirrors ref._geodesic_samples, including its two degenerate sphere branches.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    ts = np.linspace(0.0, 1.0, n_samples)
    name = instance.name

    if name == "disk":
        return A[:, None, :] + ts[None, :, None] * (B - A)[:, None, :]

    if name == "torus":
        side = instance.scale_m
        d = B - A
        d = (d + side / 2.0) % side - side / 2.0
        return (A[:, None, :] + ts[None, :, None] * d[:, None, :]) % side

    if name == "klein":
        side = instance.scale_m
        lift = _klein_best_lift_batch(A, B, side)
        cover = A[:, None, :] + ts[None, :, None] * (lift - A)[:, None, :]
        return _reduce_klein_cover_batch(cover, side)

    if name == "sphere":
        r = instance.scale_m
        ua, ub = A / r, B / r
        dot = np.clip(np.einsum("ij,ij->i", ua, ub), -1.0, 1.0)
        omega = np.arccos(dot)
        sin_omega = np.sin(omega)

        # Main slerp branch, computed everywhere then overwritten where degenerate.
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_sin = np.where(np.abs(sin_omega) < 1e-10, 1.0, sin_omega)
            ca = np.sin((1.0 - ts)[None, :] * omega[:, None]) / safe_sin[:, None]
            cb = np.sin(ts[None, :] * omega[:, None]) / safe_sin[:, None]
            out = r * (ca[:, :, None] * ua[:, None, :] + cb[:, :, None] * ub[:, None, :])

        # Nearly antipodal: normalized linear interpolation (ref fallback).
        lerp_mask = np.abs(sin_omega) < 1e-10
        if lerp_mask.any():
            raw = (1.0 - ts)[None, :, None] * ua[:, None, :] + ts[None, :, None] * ub[:, None, :]
            norms = np.linalg.norm(raw, axis=-1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            out[lerp_mask] = (r * raw / norms)[lerp_mask]

        # Coincident points: ref returns a repeated, and checks this first.
        same_mask = omega < 1e-12
        if same_mask.any():
            out[same_mask] = np.repeat(A[same_mask][:, None, :], n_samples, axis=1)

        return out

    raise ValueError(f"Unsupported manifold: {name}")


# ---------------------------------------------------------------------------
# Ocean crossing
# ---------------------------------------------------------------------------

def legs_cross_ocean(
    instance: ref.MapInstance,
    A: np.ndarray,
    B: np.ndarray,
    chunk: int = 512,
) -> np.ndarray:
    """(k,) bool: does the geodesic A[i] -> B[i] touch an ocean region?

    Chunked because the Klein distance expands each sample point into 25 lifts.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    out = np.zeros(len(A), dtype=bool)
    if len(instance.ocean_centers) == 0:
        return out

    radii = np.asarray(instance.ocean_radii_m, dtype=float)[None, :]
    for s in range(0, len(A), chunk):
        a = A[s : s + chunk]
        b = B[s : s + chunk]
        pts = geodesic_samples_batch(instance, a, b)
        flat = pts.reshape(-1, pts.shape[-1])
        # ref.point_in_ocean calls geodesic_distance(point, center) in that order.
        dist = pairwise_geodesic(instance, flat, instance.ocean_centers)
        inside = (dist <= radii).any(axis=1)
        out[s : s + chunk] = inside.reshape(len(a), -1).any(axis=1)
    return out


def points_in_ocean(instance: ref.MapInstance, P: np.ndarray) -> np.ndarray:
    """(k,) bool: is P[i] inside an ocean region? Mirrors ref.point_in_ocean."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if len(instance.ocean_centers) == 0:
        return np.zeros(len(P), dtype=bool)
    dist = pairwise_geodesic(instance, P, instance.ocean_centers)
    radii = np.asarray(instance.ocean_radii_m, dtype=float)[None, :]
    return (dist <= radii).any(axis=1)
