"""
rdft.ac.lerw_tube
=================
Tier: 3 (research)

LERW on Z^3 tubes (N x N periodic cross-section, finite length L).

The initial hope: each N gives an algebraic generating function
(finite Markov chain on cross-sectional states), and the sequence
d_f^{(N)} as N -> infinity should converge to the Z^3 LERW
fractal dimension d_f^{(3)} ~ 1.624.

**Empirical finding (documented here as a negative result).**
For every finite N the tube LERW from one end cap to the other is
essentially ballistic: <|gamma|> ~ L, giving d_f^{(N)} ~ 1.  The
reason is structural: a narrow tube is quasi-1D, and LERW on a
1D path is deterministic --- loop erasure collapses every walk to
a minimal monotone path.  Cross-sectional wandering contributes
a sub-leading constant factor, not a different power of L.  So
the sequence d_f^{(N)} does NOT approach 1.624 as N grows; it
stays locked at 1 until N becomes comparable to L, at which point
the "tube" is no longer a tube.

Conclusion. The tube-transfer-matrix animal is the wrong tower
for the Z^3 LERW fractal dimension: d_f is an isotropic-box
observable, and tubes quotient out the transverse degrees of
freedom that carry it. The right algebraic animal is the
extrapolation schemes of rdft/ac/lerw_extrap.py applied to the
isotropic-box data of rdft/ac/lerw_dirichlet.py --- each scheme
is a rational transformation of a finite sample, each produces
an exact algebraic estimate of d_f^{(3)}, and the scheme-to-scheme
consistency is the witness of convergence.

This module is retained as a documented negative result: running
`tube_scaling_sweep` reproduces the quasi-1D collapse so future
readers see the obstruction concretely.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np


def sample_lerw_tube(N: int, L: int,
                     rng: np.random.Generator,
                     max_steps: int | None = None,
                     ) -> list[tuple[int, int, int]]:
    """Sample LERW on the N x N (periodic) x L (Dirichlet) tube.

    Start at (0, 0, L // 2), stop when z-coordinate hits -1 or L.
    Returns the simple path.
    """
    if max_steps is None:
        max_steps = 2000 * N * N * L
    start = (0, 0, L // 2)
    path = [start]
    first: dict[tuple[int, int, int], int] = {start: 0}
    cur = start
    steps = 0
    while True:
        axis = int(rng.integers(3))
        shift = 1 if rng.random() < 0.5 else -1
        nx = list(cur)
        nx[axis] += shift
        if axis == 2:
            if nx[2] < 0 or nx[2] >= L:
                return path
        else:
            nx[axis] %= N
        nt = (nx[0], nx[1], nx[2])
        if nt in first:
            cut = first[nt]
            for drop in path[cut + 1:]:
                del first[drop]
            path = path[:cut + 1]
            cur = nt
        else:
            first[nt] = len(path)
            path.append(nt)
            cur = nt
        steps += 1
        if steps >= max_steps:
            raise RuntimeError(
                f"tube walk did not exit after {max_steps} steps "
                f"(N={N}, L={L})")


def tube_scaling_sweep(N: int, L_vals: Sequence[int],
                       n_samples: int = 300,
                       seed: int = 0,
                       ) -> dict[int, tuple[float, float]]:
    """Measure <|gamma|> on the N-tube for each L in L_vals.

    Returns {L: (mean, sem)}.
    """
    rng = np.random.default_rng(seed)
    out: dict[int, tuple[float, float]] = {}
    for L in L_vals:
        lens = np.empty(n_samples)
        for s in range(n_samples):
            p = sample_lerw_tube(N, L, rng)
            lens[s] = len(p) - 1
        out[L] = (float(lens.mean()),
                  float(lens.std(ddof=1) / np.sqrt(n_samples)))
    return out


def fit_tube_d_f(sweep: dict[int, tuple[float, float]]) -> float:
    """Log-log slope of mean length vs L."""
    Ls = sorted(sweep.keys())
    x = np.log(np.asarray(Ls, dtype=float))
    y = np.log(np.asarray([sweep[L][0] for L in Ls]))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, *_ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)
