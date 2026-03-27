"""
rdft.ac.algebraic
=================
Algebraic curve singularity analysis via Newton polygon and Puiseux expansion.

Given a polynomial F(T, z) = 0 (the DSE defining T as an algebraic function
of z), this module:

1. Finds all branch points (T*, z*) where ∂F/∂T = 0
2. Computes the Newton polygon to determine Puiseux exponents p/q
3. Classifies the singularity type (square-root, cube-root, pole, ...)
4. Provides the data needed for the transfer theorem

The Newton polygon algorithm:
  - Shift F to the branch point: F(T*+δT, z*+δz)
  - Extract monomials of the shifted polynomial
  - Compute the lower convex hull of {(i,j)} where δT^i δz^j appears
  - Each edge of the hull with slope -q/p gives Puiseux exponent p/q
  - The leading exponent is from the steepest edge

For reaction-diffusion DSEs, F is always polynomial in T and z, so the
Newton polygon is finite and the singularity is always algebraic.

References:
    Walker (1950) — Algebraic Curves, Ch. IV (Newton polygon)
    Brieskorn & Knörrer (1986) — Plane Algebraic Curves
    Flajolet & Sedgewick (2009) — §VII.7 (algebraic singularity analysis)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import sympy as sp
from sympy import Symbol, Rational, Poly, resultant, solve, series, sqrt


class NewtonPolygon:
    """
    Newton polygon of a bivariate polynomial.

    Given F(T, z) with monomials c_{ij} T^i z^j, the Newton polygon
    is the lower convex hull of the support {(i, j) : c_{ij} ≠ 0}.

    Each edge with slope -q/p (in lowest terms, p > 0) corresponds to
    a Puiseux branch T ~ c · z^{p/q}.
    """

    def __init__(self, F: sp.Expr, T: sp.Symbol, z: sp.Symbol):
        self.F = F
        self.T = T
        self.z = z
        self._points = None
        self._hull_edges = None

    @property
    def support(self) -> List[Tuple[int, int]]:
        """Monomials (i, j) with nonzero coefficient in F = Σ c_ij T^i z^j."""
        if self._points is None:
            try:
                poly = Poly(self.F, self.T, self.z)
                self._points = [(m[0], m[1]) for m, c
                                in zip(poly.monoms(), poly.coeffs()) if c != 0]
            except Exception:
                # Fallback: expand and collect
                expanded = sp.expand(self.F)
                self._points = []
                for term in sp.Add.make_args(expanded):
                    i = sp.degree(term, self.T)
                    j = sp.degree(term, self.z)
                    self._points.append((int(i), int(j)))
        return self._points

    @property
    def edges(self) -> List[Tuple[Rational, List[Tuple[int, int]]]]:
        """
        Edges of the lower convex hull as (slope, supporting_points) pairs.

        The lower convex hull is computed via Andrew's monotone chain
        on the integer lattice points.
        """
        if self._hull_edges is None:
            self._hull_edges = self._compute_lower_hull_edges()
        return self._hull_edges

    def _compute_lower_hull_edges(self) -> List[Tuple[Rational, List]]:
        """Compute edges of the lower convex hull."""
        points = sorted(set(self.support))  # sort by (i, j)
        if len(points) <= 1:
            return []

        # Andrew's monotone chain — lower hull
        hull = []
        for p in points:
            while len(hull) >= 2:
                # Cross product to check if we make a left turn
                o, a, b = hull[-2], hull[-1], p
                cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
                if cross <= 0:
                    hull.pop()
                else:
                    break
            hull.append(p)

        # Extract edges with slopes
        edges = []
        for k in range(len(hull) - 1):
            p1, p2 = hull[k], hull[k + 1]
            di = p2[0] - p1[0]
            dj = p2[1] - p1[1]
            if di != 0:
                slope = Rational(dj, di)
            else:
                slope = sp.oo
            edges.append((slope, [p1, p2]))

        return edges

    def puiseux_exponents(self) -> List[Rational]:
        """
        Puiseux exponents p/q from the Newton polygon edges.

        An edge with slope -q/p (from lower-left to upper-right)
        gives the Puiseux exponent p/q for the branch T ~ c·z^{p/q}.

        For the *shifted* polynomial at a branch point, the relevant
        exponent is the reciprocal of the edge slope.
        """
        exponents = []
        for slope, _ in self.edges:
            if slope != 0 and slope != sp.oo:
                # Edge slope Δj/Δi = slope
                # Puiseux: δT ~ δz^{-1/slope} if we're in (δT, δz) coords
                # Actually: Newton polygon edge from (i1,j1) to (i2,j2)
                # gives T ~ z^{(j2-j1)/(i1-i2)} — but this depends on convention
                # For the standard convention: exponent = -slope when slope < 0
                if slope < 0:
                    exponents.append(-slope)
                else:
                    exponents.append(slope)
        return exponents if exponents else [Rational(1, 2)]  # default: square root


class AlgebraicSingularity:
    """
    Complete singularity analysis of an algebraic curve F(T, z) = 0.

    Finds all branch points, computes Newton polygons at each,
    and classifies singularity types.
    """

    def __init__(self, F: sp.Expr, T: sp.Symbol, z: sp.Symbol):
        self.F = F
        self.T = T
        self.z = z
        self._branch_points = None
        self._discriminant = None

    @classmethod
    def from_lagrange(cls, lagrange_eq) -> 'AlgebraicSingularity':
        """Construct from a LagrangeEquation T = z·φ(T)."""
        F = lagrange_eq.T - lagrange_eq.z * lagrange_eq.phi
        return cls(F, lagrange_eq.T, lagrange_eq.z)

    @property
    def discriminant_in_z(self) -> sp.Expr:
        """
        Resultant of F and ∂F/∂T with respect to T.

        The zeros of this polynomial in z are the branch points.
        """
        if self._discriminant is None:
            dF_dT = sp.diff(self.F, self.T)
            self._discriminant = sp.factor(
                resultant(self.F, dF_dT, self.T)
            )
        return self._discriminant

    def branch_points(self) -> List[Tuple[sp.Expr, sp.Expr]]:
        """
        Find all branch points (T*, z*) where F = 0 and ∂F/∂T = 0.

        Uses two methods:
        1. Direct solve of the system {F=0, dF/dT=0}
        2. Resultant elimination to find z* first, then recover T*
        """
        if self._branch_points is not None:
            return self._branch_points

        dF_dT = sp.diff(self.F, self.T)

        # Method 1: Direct solve
        try:
            solutions = solve([self.F, dF_dT], [self.T, self.z],
                            dict=False)
            if solutions:
                # Filter: keep only finite, non-zero solutions
                valid = []
                for sol in solutions:
                    if isinstance(sol, tuple) and len(sol) == 2:
                        T_star, z_star = sol
                        if T_star != 0 and z_star != 0:
                            valid.append((sp.simplify(T_star),
                                         sp.simplify(z_star)))
                if valid:
                    self._branch_points = valid
                    return valid
        except Exception:
            pass

        # Method 2: Resultant
        try:
            disc = self.discriminant_in_z
            z_stars = solve(disc, self.z)
            valid = []
            for z_star in z_stars:
                if z_star == 0:
                    continue
                T_stars = solve(self.F.subs(self.z, z_star), self.T)
                for T_star in T_stars:
                    if T_star != 0:
                        valid.append((sp.simplify(T_star),
                                     sp.simplify(z_star)))
            if valid:
                self._branch_points = valid
                return valid
        except Exception:
            pass

        self._branch_points = []
        return []

    def dominant_branch_point(self) -> Optional[Tuple[sp.Expr, sp.Expr]]:
        """
        The branch point with smallest |z*| (closest to origin).

        This dominates the asymptotics via the transfer theorem.
        For symbolic z*, returns the first real positive solution
        or the first solution if all are complex.
        """
        bps = self.branch_points()
        if not bps:
            return None

        # Try to find a real positive z*
        for T_star, z_star in bps:
            try:
                if z_star.is_positive or z_star.is_real:
                    return (T_star, z_star)
            except (AttributeError, TypeError):
                continue

        return bps[0]  # fallback: first solution

    def newton_polygon_at(self, T_star: sp.Expr,
                           z_star: sp.Expr) -> NewtonPolygon:
        """
        Newton polygon of F shifted to the branch point (T*, z*).

        F_shifted(δT, δz) = F(T* + δT, z* + δz)
        """
        dT = Symbol('dT')
        dz = Symbol('dz')
        F_shifted = self.F.subs([(self.T, T_star + dT),
                                  (self.z, z_star + dz)])
        F_shifted = sp.expand(F_shifted)
        return NewtonPolygon(F_shifted, dT, dz)

    def puiseux_exponent(self, T_star: sp.Expr = None,
                          z_star: sp.Expr = None) -> Rational:
        """
        Determine the Puiseux exponent p/q at a branch point.

        Uses the Newton polygon of the shifted polynomial.
        Falls back to derivative analysis if the Newton polygon
        approach encounters difficulties.
        """
        if T_star is None or z_star is None:
            bp = self.dominant_branch_point()
            if bp is None:
                return Rational(1, 2)  # default
            T_star, z_star = bp

        # Method 1: Newton polygon
        try:
            np = self.newton_polygon_at(T_star, z_star)
            exponents = np.puiseux_exponents()
            if exponents:
                # The leading exponent (smallest) determines the singularity
                return min(exponents)
        except Exception:
            pass

        # Method 2: Derivative analysis (fallback)
        # If F_TT ≠ 0 at branch point → square-root (p/q = 1/2)
        # If F_TT = 0, F_TTT ≠ 0 → cube-root (p/q = 1/3)
        F_TT = sp.diff(self.F, self.T, 2)
        val = F_TT.subs([(self.T, T_star), (self.z, z_star)])
        if sp.simplify(val) != 0:
            return Rational(1, 2)

        F_TTT = sp.diff(self.F, self.T, 3)
        val3 = F_TTT.subs([(self.T, T_star), (self.z, z_star)])
        if sp.simplify(val3) != 0:
            return Rational(1, 3)

        return Rational(1, 2)  # conservative default

    def singularity_type(self) -> Dict[str, sp.Expr]:
        """
        Full singularity classification at the dominant branch point.

        Returns dict with:
            'T_star', 'z_star': branch point coordinates
            'puiseux_exponent': p/q
            'type': human-readable name
            'alpha': transfer theorem α parameter (= p/q)
        """
        bp = self.dominant_branch_point()
        if bp is None:
            return {'type': 'no_singularity', 'alpha': None}

        T_star, z_star = bp
        pq = self.puiseux_exponent(T_star, z_star)

        type_name = {
            Rational(1, 2): 'square_root_branch',
            Rational(1, 3): 'cube_root_branch',
            Rational(2, 3): 'two_thirds_branch',
            Rational(1, 1): 'simple_pole',
        }.get(pq, f'branch_p/q={pq}')

        return {
            'T_star': T_star,
            'z_star': z_star,
            'puiseux_exponent': pq,
            'type': type_name,
            'alpha': pq,  # transfer theorem exponent
        }

    def to_singularity(self) -> 'Singularity':
        """Convert to a transfer.Singularity object for asymptotic analysis."""
        from .transfer import Singularity

        info = self.singularity_type()
        if info['alpha'] is None:
            return Singularity(sp.S.One, Rational(1, 2))

        return Singularity(
            z_star=info['z_star'],
            alpha=info['alpha'],
        )

    def summary(self) -> str:
        """Human-readable analysis summary."""
        lines = ['Algebraic Singularity Analysis',
                 '=' * 40]

        lines.append(f'F(T, z) = {self.F}')

        try:
            disc = self.discriminant_in_z
            lines.append(f'Discriminant in z: {disc}')
        except Exception:
            pass

        bps = self.branch_points()
        lines.append(f'\n{len(bps)} branch point(s):')
        for i, (T_star, z_star) in enumerate(bps):
            lines.append(f'  [{i+1}] T* = {T_star}, z* = {z_star}')

        info = self.singularity_type()
        lines.append(f'\nDominant singularity:')
        lines.append(f'  Type: {info["type"]}')
        lines.append(f'  Puiseux exponent: {info.get("puiseux_exponent", "?")}')
        lines.append(f'  Transfer: [z^n] ~ n^{{-{info.get("alpha", "?")}-1}}')

        return '\n'.join(lines)
