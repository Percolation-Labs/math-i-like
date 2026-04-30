"""
rdft.ac.lerw_hierarchical
=========================
Tier: 3 (research)

LERW on self-similar hierarchical lattices ("combinatorial animals"
for Z^d LERW).

Two families are implemented:

1. Migdal-Kadanoff (b, s) diamond lattice.
   G_0 = single edge. G_{k+1} = every edge of G_k replaced by a
   diamond (b parallel branches, s edges each in series).
   Effective dimension d_eff(b, s) = 1 + log(b)/log(s).
   Setting (b, s) = (4, 2) gives d_eff = 3.
   **Failure mode**: the MK diamond is series-parallel. The UST
   on such a graph deterministically selects one branch per
   diamond, and the source-to-target path has length equal to
   the diameter. Hence d_f^{MK}(b, s) = 1 for all b, s >= 2 ---
   LERW is degenerate on series-parallel graphs because loop
   erasure acts trivially. This is a **negative result** that
   diagnoses the MK approximation.

2. Sierpinski gasket (triangle-based).
   G_0 = single triangle. G_{k+1} = three copies of G_k glued at
   the three top-level corners (the classical 'triforce'
   construction).
   Hausdorff dimension log(3)/log(2) ~ 1.585. The gasket is NOT
   series-parallel: every sub-triangle carries three edges around
   a cycle, so loop erasure is non-trivial.
   **Result**: d_f^{Sierp} strictly between 1 and 2, extractable
   from the scaling of <|gamma|>_{G_k} with the diameter 2^k.

Algebraic / compositional reading (Flajolet-Sedgewick).
The hierarchical construction G_{k+1} = phi(G_k) is a combinatorial
SEQUENCE / PRODUCT construction applied to a finite pattern. At the
level of generating functions, the LERW length GF at level k+1 is
obtained by substituting the level-k GF into a fixed rational
transformation phi. Fractal dimensions are eigenvalues of this
transformation at its renormalisation fixed point. This is the
precise combinatorial-algebraic content of Kadanoff's block-spin
renormalisation, rephrased in Flajolet's symbolic method.
For LERW specifically, the transformation is determined by the
Kirchhoff ratio (Prop 1) restricted to the recursive building block.

What this tells us about Z^d LERW.
- The Z^d GF is not D-finite in the thermodynamic limit, so
  standard AC extraction does not terminate in closed form.
- Each hierarchical animal is D-finite at every finite level
  (rational GF) and its scaling exponent is an algebraic number
  (eigenvalue of a finite matrix).
- Comparing animal exponents to MC on Z^d quantifies what the
  animal captures and what it misses. The MK diamond loses
  everything (d_f = 1); the Sierpinski gasket captures non-trivial
  loop-erasure but at the wrong fractal dimension.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np


Graph = tuple[int, list[tuple[int, int]], int, int]
# (n_vertices, edges as (u, v) with u < v, source_id, target_id)


def diamond_lattice(k: int, b: int = 4, s: int = 2) -> Graph:
    """Build the level-k (b, s) diamond lattice.

    Returns (n_vertices, edges, source, target). Source = 0, target
    is a specific vertex id determined by the construction.
    """
    if k == 0:
        # Two terminals joined by one edge
        return (2, [(0, 1)], 0, 1)
    # Recursive build: take G_{k-1} and replace every edge by a diamond.
    sub = diamond_lattice(k - 1, b, s)
    n_sub, edges_sub, src_sub, tgt_sub = sub
    new_edges: list[tuple[int, int]] = []
    next_vertex = n_sub  # new vertex id counter
    for (u, v) in edges_sub:
        # Replace edge (u, v) by b parallel branches of s edges each.
        for _ in range(b):
            # chain: u -- w_1 -- w_2 -- ... -- w_{s-1} -- v
            prev = u
            for i in range(s - 1):
                w = next_vertex
                next_vertex += 1
                a, c = (min(prev, w), max(prev, w))
                new_edges.append((a, c))
                prev = w
            a, c = (min(prev, v), max(prev, v))
            new_edges.append((a, c))
    return (next_vertex, new_edges, src_sub, tgt_sub)


def diamond_diameter(k: int, s: int = 2) -> int:
    """Diameter of G_k: each level multiplies the source-target
    distance by s.
    """
    return s ** k


def diamond_n_edges(k: int, b: int = 4, s: int = 2) -> int:
    """Number of edges in G_k: (b*s)^k."""
    return (b * s) ** k


# ------------------------------------------------------------------ #
#  Wilson sampling of LERW
# ------------------------------------------------------------------ #

def _adjacency(G: Graph) -> list[list[int]]:
    n, edges, _, _ = G
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def sample_lerw_hierarchical(G: Graph,
                             rng: np.random.Generator,
                             max_steps: int | None = None,
                             ) -> list[int]:
    """Run Wilson's loop-erased random walk on G from source to
    target. Returns the simple path as a list of vertex ids.

    This samples the Kirchhoff-ratio LERW measure of Prop 1 exactly
    (Wilson's theorem: loop-erasing SRW from s stopped at first hit
    of t gives the UST-path law).
    """
    n, _, src, tgt = G
    adj = _adjacency(G)
    if max_steps is None:
        # Sierpinski-type fractals have anomalous diffusion with walk
        # dimension d_w > 2, so cover time scales as a higher power
        # of n than on Z^d. Give a generous budget to avoid spurious
        # timeouts on larger hierarchical levels.
        max_steps = max(500 * n, 100000)
    path = [src]
    first_visit = {src: 0}
    current = src
    steps = 0
    while current != tgt:
        if steps >= max_steps:
            raise RuntimeError(
                f"Wilson walk did not hit target after {max_steps} steps "
                f"(n_vertices={n})")
        nbrs = adj[current]
        current = nbrs[rng.integers(len(nbrs))]
        if current in first_visit:
            cut = first_visit[current]
            for drop in path[cut + 1:]:
                del first_visit[drop]
            path = path[:cut + 1]
        else:
            first_visit[current] = len(path)
            path.append(current)
        steps += 1
    return path


def mean_lerw_length_hierarchical(G: Graph, n_samples: int,
                                  rng: np.random.Generator,
                                  ) -> tuple[float, float]:
    """Monte Carlo estimate of <|gamma|> on the hierarchical graph.

    Returns (mean, sem).
    """
    lengths = np.empty(n_samples)
    for s in range(n_samples):
        p = sample_lerw_hierarchical(G, rng)
        lengths[s] = len(p) - 1
    return float(lengths.mean()), \
        float(lengths.std(ddof=1) / np.sqrt(n_samples))


# ------------------------------------------------------------------ #
#  Fractal dimension extraction
# ------------------------------------------------------------------ #

def hierarchical_dimension_sweep(k_vals: Sequence[int],
                                 b: int = 4, s: int = 2,
                                 n_samples: int = 1500,
                                 seed: int = 0,
                                 ) -> dict[int, dict]:
    """For each k in k_vals, construct G_k, run Wilson sampling,
    measure <|gamma|>, and return a dict keyed by k with the
    measurement + graph stats.
    """
    rng = np.random.default_rng(seed)
    out: dict[int, dict] = {}
    for k in k_vals:
        G = diamond_lattice(k, b, s)
        mean, sem = mean_lerw_length_hierarchical(G, n_samples, rng)
        out[k] = {
            'n_vertices': G[0],
            'n_edges': len(G[1]),
            'diameter': diamond_diameter(k, s),
            'mean_length': mean,
            'sem': sem,
        }
    return out


def fit_hierarchical_d_f(sweep: dict[int, dict],
                         s: int = 2) -> tuple[float, float]:
    """Fit <|gamma|>_k = A * s^{k * d_f} from the sweep.

    Returns (d_f, intercept_log).
    """
    ks = sorted(sweep.keys())
    x = np.array([k * np.log(s) for k in ks], dtype=float)
    y = np.log(np.array([sweep[k]['mean_length'] for k in ks]))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


# ------------------------------------------------------------------ #
#  Sierpinski gasket (non-series-parallel animal)
# ------------------------------------------------------------------ #

def sierpinski_gasket(k: int) -> Graph:
    """Build the level-k Sierpinski gasket as a Graph.

    G_0: single triangle, vertices {0, 1, 2}, corners = [0, 1, 2].
    G_{k+1}: three copies of G_k glued at top-level corners
             (triforce arrangement).

    Source = corners[0] (top), target = corners[1] (bottom-left).
    """
    if k == 0:
        return (3, [(0, 1), (0, 2), (1, 2)], 0, 1)
    sub = _sierp_recursive_build(k)
    n, edges, corners = sub
    return (n, edges, corners[0], corners[1])


def _sierp_recursive_build(k: int) -> tuple[int, list[tuple[int, int]], list[int]]:
    """Recursive builder that returns (n_vertices, edges, corners).
    corners is the three top-level corners of the current level.
    """
    if k == 0:
        return (3, [(0, 1), (0, 2), (1, 2)], [0, 1, 2])
    sub_n, sub_edges, sub_corners = _sierp_recursive_build(k - 1)
    # Allocate six top-level-adjacent vertices:
    # Three outer corners (a, b, c) of the new triangle, plus three
    # shared midpoints (b0, c0, b1). Interior vertices of each sub-copy
    # are fresh.
    a, b, c = 0, 1, 2
    b0, c0, b1 = 3, 4, 5
    next_vid = 6
    # Copy 0 (top):        sub_corners -> (a, b0, c0)
    # Copy 1 (bottom-left): sub_corners -> (b0, b, b1)
    # Copy 2 (bottom-right):sub_corners -> (c0, b1, c)
    copy_maps = [
        {sub_corners[0]: a, sub_corners[1]: b0, sub_corners[2]: c0},
        {sub_corners[0]: b0, sub_corners[1]: b, sub_corners[2]: b1},
        {sub_corners[0]: c0, sub_corners[1]: b1, sub_corners[2]: c},
    ]
    new_edges: list[tuple[int, int]] = []
    for corner_map in copy_maps:
        vmap: dict[int, int] = {}
        for j in range(sub_n):
            if j in corner_map:
                vmap[j] = corner_map[j]
            else:
                vmap[j] = next_vid
                next_vid += 1
        for (u, v) in sub_edges:
            uu, vv = vmap[u], vmap[v]
            x, y = (min(uu, vv), max(uu, vv))
            new_edges.append((x, y))
    return (next_vid, new_edges, [a, b, c])


def sierpinski_diameter(k: int) -> int:
    """Graph distance between two top-level corners of G_k.
    Doubles at every level: diameter(G_{k+1}) = 2 * diameter(G_k),
    with diameter(G_0) = 1.
    """
    return 2 ** k


def sierpinski_dimension_sweep(k_vals: Sequence[int],
                               n_samples: int = 1500,
                               seed: int = 0,
                               ) -> dict[int, dict]:
    """Sampled <|gamma|> on the Sierpinski gasket at each level."""
    rng = np.random.default_rng(seed)
    out: dict[int, dict] = {}
    for k in k_vals:
        G = sierpinski_gasket(k)
        mean, sem = mean_lerw_length_hierarchical(G, n_samples, rng)
        out[k] = {
            'n_vertices': G[0],
            'n_edges': len(G[1]),
            'diameter': sierpinski_diameter(k),
            'mean_length': mean,
            'sem': sem,
        }
    return out


def fit_sierpinski_d_f(sweep: dict[int, dict]) -> tuple[float, float]:
    """Fit <|gamma|>_k = A * 2^{k * d_f} from the Sierpinski sweep."""
    return fit_hierarchical_d_f(sweep, s=2)
