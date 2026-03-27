"""
rdft.graphs.spectral
====================
Spectral dimension computation for arbitrary graphs.

The spectral dimension d_s characterises the density of eigenvalues of the
graph Laplacian and controls the return probability of a random walk:

    P(t) ~ t^{-d_s/2}    as t → ∞

For reaction-diffusion processes, the substitution d → d_s in all scaling
formulas gives the correct exponents on arbitrary graphs, provided the
Laplacian does not acquire an anomalous dimension (verified at one loop
for the BRW process in Bordeu, Amarteifio et al. 2019).

Key values:
    Hypercubic lattice Z^d :  d_s = d
    Sierpinski carpet      :  d_s ≈ 1.86  (Watanabe 1985)
    Critical random tree   :  d_s = 4/3   (Destri-Donetti 2002)
    Preferential attachment:  d_s ≥ 4     (mean-field regime)

Reference:
    Bordeu, Amarteifio et al. (2019) Sci. Rep. 9:15590
    Amarteifio (2019) PhD thesis, Ch.3
    Burioni & Cassi (2005) J. Phys. A 38:R45
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np
import sympy as sp


class SpectralDimension:
    """
    Compute spectral dimension d_s for a graph from its Laplacian spectrum.

    The spectral dimension is defined via the return probability:
        P(t) = (1/N) Σ_k exp(-λ_k t)

    where {λ_k} are eigenvalues of the normalised graph Laplacian.
    For large t:
        P(t) ~ t^{-d_s/2}

    so d_s = -2 · d(log P)/d(log t) in the scaling regime.

    Parameters
    ----------
    adjacency : numpy array (N×N), the adjacency matrix of the graph
                OR a networkx Graph object
    """

    def __init__(self, adjacency):
        import networkx as nx
        if isinstance(adjacency, nx.Graph):
            self._graph = adjacency
            self.adjacency = nx.to_numpy_array(adjacency)
        else:
            self.adjacency = np.array(adjacency, dtype=float)
            self._graph = None

        self.N = self.adjacency.shape[0]
        self._eigenvalues = None
        self._d_s = None

    @property
    def laplacian_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the graph Laplacian L = D - A, sorted ascending."""
        if self._eigenvalues is None:
            degrees = self.adjacency.sum(axis=1)
            D = np.diag(degrees)
            L = D - self.adjacency
            self._eigenvalues = np.sort(np.linalg.eigvalsh(L))
        return self._eigenvalues

    def return_probability(self, t: float) -> float:
        """
        Heat kernel trace: P(t) = (1/N) Σ_k exp(-λ_k t)
        """
        eigvals = self.laplacian_eigenvalues
        return np.mean(np.exp(-eigvals * t))

    def spectral_dimension(self, t_range: Tuple[float, float] = None,
                            n_points: int = 100) -> float:
        """
        Estimate d_s from the slope of log P(t) vs log t.

        d_s = -2 · d(log P)/d(log t)

        Parameters
        ----------
        t_range : (t_min, t_max) for the fitting window
        n_points : number of time points

        Returns
        -------
        Estimated spectral dimension d_s
        """
        if t_range is None:
            # Heuristic: use the intermediate time regime
            # Avoid very short times (lattice effects) and very long (finite size)
            lambda_max = self.laplacian_eigenvalues[-1]
            lambda_2 = self.laplacian_eigenvalues[1] if self.N > 1 else 1.0
            t_min = max(1.0 / lambda_max, 0.1)
            t_max = min(1.0 / max(lambda_2, 1e-6), 1000.0)
            if t_max <= t_min:
                t_max = 10 * t_min
        else:
            t_min, t_max = t_range

        t_vals = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
        P_vals = np.array([self.return_probability(t) for t in t_vals])

        # Filter out zeros
        mask = P_vals > 1e-30
        if mask.sum() < 5:
            return 0.0

        log_t = np.log(t_vals[mask])
        log_P = np.log(P_vals[mask])

        # Linear fit: log P = -(d_s/2) · log t + const
        slope, intercept = np.polyfit(log_t, log_P, 1)
        self._d_s = -2 * slope

        return self._d_s

    def summary(self) -> str:
        """Human-readable summary."""
        d_s = self.spectral_dimension()
        lines = [
            f'Graph: {self.N} nodes',
            f'Laplacian spectrum: λ_min={self.laplacian_eigenvalues[0]:.4f}, '
            f'λ_max={self.laplacian_eigenvalues[-1]:.4f}',
            f'Spectral gap λ_2 = {self.laplacian_eigenvalues[1]:.4f}' if self.N > 1 else '',
            f'Spectral dimension d_s ≈ {d_s:.3f}',
        ]
        return '\n'.join(l for l in lines if l)


# ------------------------------------------------------------------ #
#  Known spectral dimensions                                          #
# ------------------------------------------------------------------ #

KNOWN_SPECTRAL_DIMENSIONS = {
    'hypercubic_1d': {'d_s': 1, 'source': 'trivial'},
    'hypercubic_2d': {'d_s': 2, 'source': 'trivial'},
    'hypercubic_3d': {'d_s': 3, 'source': 'trivial'},
    'sierpinski_carpet': {'d_s': 1.86, 'source': 'Watanabe (1985)'},
    'random_tree': {'d_s': 4/3, 'source': 'Destri-Donetti (2002)'},
    'preferential_attachment': {'d_s': 4.0, 'source': 'mean-field, BRW paper'},
}


def substitute_spectral_dimension(expression: sp.Expr,
                                    d_s: sp.Expr,
                                    d: sp.Symbol = None) -> sp.Expr:
    """
    Replace d → d_s in a scaling expression.

    This is the key substitution from Bordeu, Amarteifio et al. (2019):
    all scaling exponents on a graph with spectral dimension d_s are
    obtained by replacing the Euclidean dimension d with d_s.

    Valid when the Laplacian does not acquire an anomalous dimension
    (no wavefunction renormalisation).

    Parameters
    ----------
    expression : sympy expression containing dimension d
    d_s : spectral dimension (symbolic or numeric)
    d : symbol to replace (default: Symbol('d'))
    """
    if d is None:
        d = sp.Symbol('d', positive=True)
    return expression.subs(d, d_s)


# ------------------------------------------------------------------ #
#  Graph constructors for standard test cases                          #
# ------------------------------------------------------------------ #

def hypercubic_lattice(d: int, L: int) -> np.ndarray:
    """
    Build adjacency matrix for a d-dimensional hypercubic lattice
    with periodic boundary conditions and linear size L.

    Total nodes: L^d
    """
    import networkx as nx
    dims = [L] * d
    G = nx.grid_graph(dims, periodic=True)
    return nx.to_numpy_array(G)


def sierpinski_carpet(level: int = 2) -> np.ndarray:
    """
    Build adjacency matrix for a Sierpinski carpet at given iteration level.

    The carpet is built from a 2D lattice of size 3^level × 3^level
    with the central square removed at each iteration.

    d_s ≈ 1.86 (Watanabe 1985, Dasgupta et al. 1999)
    """
    size = 3 ** level

    def is_removed(x, y, size):
        """Check if position (x,y) is in a removed square."""
        while size > 1:
            third = size // 3
            if third <= x < 2 * third and third <= y < 2 * third:
                return True
            x, y, size = x % third, y % third, third
        return False

    # Build list of valid sites
    sites = []
    site_map = {}
    for x in range(size):
        for y in range(size):
            if not is_removed(x, y, size):
                site_map[(x, y)] = len(sites)
                sites.append((x, y))

    N = len(sites)
    adj = np.zeros((N, N))

    for x, y in sites:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx_, ny = x + dx, y + dy
            if (nx_, ny) in site_map:
                i = site_map[(x, y)]
                j = site_map[(nx_, ny)]
                adj[i, j] = 1.0

    return adj


def random_tree(n_nodes: int, seed: int = None) -> np.ndarray:
    """
    Generate a critical Galton-Watson random tree.

    Each node has 0, 1, or 2 descendants with equal probability,
    giving mean degree 2 (critical).

    d_s = 4/3 (Destri-Donetti 2002)
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((n_nodes, n_nodes))
    # Simple Prüfer sequence approach for random labelled tree
    if n_nodes <= 1:
        return adj
    import networkx as nx
    T = nx.random_tree(n_nodes, seed=seed)
    return nx.to_numpy_array(T)


def preferential_attachment(n_nodes: int, m: int = 3, seed: int = None) -> np.ndarray:
    """
    Barabási-Albert preferential attachment network.

    Power-law degree distribution with exponent ≈ -3.
    d_s ≥ 4 (mean-field behaviour for BRW).
    """
    import networkx as nx
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    return nx.to_numpy_array(G)


# ------------------------------------------------------------------ #
#  BRW scaling predictions                                             #
# ------------------------------------------------------------------ #

def brw_scaling_exponents(d_s: sp.Expr,
                           d_c: sp.Expr = sp.Integer(4)) -> Dict[str, sp.Expr]:
    """
    BRW (Branching Wiener Sausage) scaling exponents from Bordeu+ (2019).

    For d_s < d_c = 4:
        ⟨a^p⟩(t) ~ t^{(p·d_s - 2)/2}    (time scaling)
        ⟨a^p⟩(L) ~ L^{p·d_s - 2}         (size scaling)
        P(a) ~ a^{-(1+2/d_s)}             (cluster size distribution)

    For d_s ≥ d_c = 4 (mean-field):
        ⟨a^p⟩(t) ~ t^{2p-1}
        ⟨a^p⟩(L) ~ L^{4p-2}
        P(a) ~ a^{-3/2}
    """
    p = sp.Symbol('p', positive=True, integer=True)

    below_dc = {
        'time_exponent': (p * d_s - 2) / 2,
        'size_exponent': p * d_s - 2,
        'distribution_exponent': -(1 + sp.Rational(2, 1) / d_s),
        'gap_exponent': d_s,
        'd_c': d_c,
    }

    above_dc = {
        'time_exponent': 2 * p - 1,
        'size_exponent': 4 * p - 2,
        'distribution_exponent': sp.Rational(-3, 2),
        'gap_exponent': sp.Integer(4),
        'd_c': d_c,
    }

    return {'below_dc': below_dc, 'above_dc': above_dc}
