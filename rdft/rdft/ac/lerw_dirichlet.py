"""
rdft.ac.lerw_dirichlet
======================
LERW on Z^d boxes with killing (Dirichlet) boundary.

This is the natural setting for Kenyon's theorem (2000):
    LERW in the scaling limit on a 2D domain has fractal dimension
    d_f = 5/4, i.e. E[|gamma|] ~ L^{5/4}
when gamma is sampled from origin to boundary-hit on a box of side L.

Complementary to `rdft.ac.lerw_scaling` which uses periodic tori (no
killing boundary, so Kenyon's exponent is washed out by finite-size).

The box is [0, L-1]^d with killing boundary at every face. A walker
that steps outside is stopped; we then loop-erase its visited trace
and return its length.

This module is the numerical face of Proposition 1 of
paper/cfac/enumerative_boundary.tex in its natural asymptotic regime:
LERW as a ratio of Kirchhoff polynomials on the box graph, whose
finite-size scaling gives Kenyon's exponent in d=2 and Kozma's
d_f ~ 1/1.624 ~ 0.616 in d=3 (here we work with |gamma| ~ L^{d_f}
to stabilise the fit).
"""

from __future__ import annotations
from typing import Sequence
import numpy as np


# ------------------------------------------------------------------ #
#  Dirichlet-box LERW sampling
# ------------------------------------------------------------------ #

def sample_lerw_to_boundary(L: int, d: int,
                            rng: np.random.Generator,
                            start: tuple[int, ...] | None = None,
                            max_steps: int | None = None,
                            ) -> list[tuple[int, ...]]:
    """Sample one LERW from `start` to the boundary of [0, L-1]^d.

    Run SRW from `start`; stop the first time the walker exits the
    box, then loop-erase the trajectory chronologically and return
    the simple path. By default `start` is the centre of the box.

    The boundary-hit point is whichever lattice site was last inside
    before the step outside; the first step outside is what kills
    the walk. This matches the standard Dirichlet LERW definition.
    """
    if start is None:
        c = L // 2
        start = tuple(c for _ in range(d))
    if max_steps is None:
        max_steps = 50 * L ** d

    path: list[tuple[int, ...]] = [start]
    first_visit: dict[tuple[int, ...], int] = {start: 0}
    current = start
    steps = 0
    while True:
        axis = int(rng.integers(d))
        shift = 1 if rng.random() < 0.5 else -1
        nxt = list(current)
        nxt[axis] += shift
        if any(c < 0 or c >= L for c in nxt):
            # exited the box: LERW terminates at `current`
            return path
        nxt_t = tuple(nxt)
        if nxt_t in first_visit:
            cut = first_visit[nxt_t]
            for drop in path[cut + 1:]:
                del first_visit[drop]
            path = path[:cut + 1]
            current = nxt_t
        else:
            first_visit[nxt_t] = len(path)
            path.append(nxt_t)
            current = nxt_t
        steps += 1
        if steps >= max_steps:
            raise RuntimeError(
                f"LERW walker did not exit box after {max_steps} steps "
                f"(L={L}, d={d})")


def mean_lerw_length_dirichlet(L: int, d: int, n_samples: int,
                               rng: np.random.Generator,
                               ) -> tuple[float, float]:
    """Monte Carlo estimate of <|LERW from centre to boundary>|> on
    [0, L-1]^d with killing boundary. Returns (mean, sem).
    """
    lengths = np.empty(n_samples, dtype=float)
    for s in range(n_samples):
        path = sample_lerw_to_boundary(L, d, rng)
        lengths[s] = len(path) - 1
    return float(lengths.mean()), \
        float(lengths.std(ddof=1) / np.sqrt(n_samples))


# ------------------------------------------------------------------ #
#  Scaling extraction
# ------------------------------------------------------------------ #

def lerw_dirichlet_sweep(L_vals: Sequence[int], d: int,
                         n_samples: int = 2000,
                         seed: int = 0,
                         ) -> dict[int, tuple[float, float]]:
    """For each L in L_vals, estimate <|gamma|> for LERW from centre
    to boundary on the d-dim box of side L. Returns {L: (mean, sem)}.
    """
    rng = np.random.default_rng(seed)
    out: dict[int, tuple[float, float]] = {}
    for L in L_vals:
        m, e = mean_lerw_length_dirichlet(L, d, n_samples, rng)
        out[L] = (m, e)
    return out


def fit_fractal_dimension(L_vals: Sequence[int],
                          means: Sequence[float],
                          sems: Sequence[float] | None = None,
                          ) -> tuple[float, float]:
    """Least-squares fit log(mean) = d_f * log(L) + c.

    If sems is given, uses log-space inverse-variance weights
    (sem / mean)**-2. Returns (d_f, c).
    """
    x = np.log(np.asarray(L_vals, dtype=float))
    y = np.log(np.asarray(means, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    if sems is not None:
        sigma = np.asarray(sems, dtype=float) / np.asarray(means, dtype=float)
        W = np.diag(1.0 / sigma ** 2)
        ATA = A.T @ W @ A
        ATy = A.T @ W @ y
        coef = np.linalg.solve(ATA, ATy)
    else:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])
