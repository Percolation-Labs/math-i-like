"""
ibp_solver.core
===============

Core IBP machinery: representation of a propagator family, generation
of IBP identities, and Laporta-style reduction.

The mathematical setup:

  A propagator family is given by:
    - L loop momenta {k_1, ..., k_L}
    - E external momenta {p_1, ..., p_E}
    - N "inverse propagators" D_i (polynomials of degree 2 in the
      loop momenta, with optional masses and external invariants)
    - A list of independent scalar invariants S = {p_i . p_j}
      with values fixed at the chosen renormalisation point

  An integral in the family is parametrised by integer exponents
  (a_1, ..., a_N), one per propagator:
       I[a_1, ..., a_N] = integral over k_1, ..., k_L of
                          1 / (D_1^{a_1} D_2^{a_2} ... D_N^{a_N}).

  IBP identities arise from
       integral d^d k_j  d/dk_j^mu (v^mu / Pi D_i^{a_i}) = 0
  where v in {k_1, ..., k_L, p_1, ..., p_E}.

  Computing the derivative produces a linear combination of integrals
  with shifted exponents (a_i +/- 1), with rational coefficients in
  the dimension d and the invariants in S.

Algorithm (Laporta):
  1. Choose a "seed set" of integrals to generate identities on.
  2. For each seed, generate L * (L + E) = O(L * (L+E)) identities.
  3. Order the integrals by Laporta priority (sum of exponents, etc.).
  4. Solve the linear system top-down (Gauss elimination on the
     priority order).
  5. The remaining un-eliminated integrals are the masters.

This reference implementation is minimal:
  - Symbolic arithmetic in SymPy over Q(d, invariants).
  - No modular-arithmetic optimization (KIRA's main perf trick).
  - No subgraph-based pre-reduction (Lee/Pomeransky tricks).
  - Targets small families (N <= 5, L <= 2) tractable in seconds.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
import sympy as sp


# ─────────────────────────────────────────────────────────────────────
#  Type for an integral: a tuple of integer exponents
# ─────────────────────────────────────────────────────────────────────
ExpTuple = Tuple[int, ...]


def laporta_priority(exps: ExpTuple) -> Tuple[int, int, int]:
    """
    Laporta priority for an integral (a_1, ..., a_N).
    Lower priority = simpler integral (closer to a master).

    Standard choice: (sector, sum of denominators, max exponent).
    Sector = number of nonzero exponents (which propagators are present).
    """
    sector = sum(1 for a in exps if a > 0)
    sum_pos = sum(a for a in exps if a > 0)
    sum_neg = sum(-a for a in exps if a < 0)
    max_pos = max((a for a in exps if a > 0), default=0)
    return (sector, sum_pos + sum_neg, max_pos)


# ─────────────────────────────────────────────────────────────────────
#  Symbolic representation of an integral (placeholder symbol)
# ─────────────────────────────────────────────────────────────────────
class IntegralSymbol(sp.Function):
    """
    Symbolic placeholder for I[a_1, ..., a_N].  SymPy treats these
    as opaque symbolic functions; we manipulate linear combinations
    of them and substitute reductions.
    """
    @classmethod
    def make(cls, exps: ExpTuple) -> sp.Expr:
        return cls(*exps)

    def __new__(cls, *args):
        # All args should be ints
        return sp.Function.__new__(cls, *args)


def I(*exps: int) -> sp.Expr:
    """Construct the symbolic integral I[exps]."""
    return IntegralSymbol(*exps)


# ─────────────────────────────────────────────────────────────────────
#  IBP family representation
# ─────────────────────────────────────────────────────────────────────
@dataclass
class IBPFamily:
    """
    A propagator family.

    Attributes:
      loop_momenta: list of SymPy symbols representing loop momenta
      external_momenta: list of SymPy symbols representing externals
      propagators: list of inverse-propagator polynomials D_i,
                   each as a SymPy expression in scalar products
      scalar_product_table: dict {(a, b): value} for fixed invariants
                   (e.g. {(p, p): mu_squared, (p, k1): 0, ...}).
                   At the symmetric subtraction point all such
                   external-only products are fixed.
      dim_symbol: the dimension d (default sp.Symbol('d')).

    The internal representation works with SCALAR PRODUCTS as the
    fundamental variables, not the momenta themselves.  We invert the
    map (scalar products) <-> (inverse propagators) to express
    everything in propagators + fixed externals.
    """
    loop_momenta: List[sp.Symbol]
    external_momenta: List[sp.Symbol]
    propagators: List[sp.Expr]      # in terms of scalar-product symbols
    sp_symbols: Dict[Tuple[str, str], sp.Symbol]   # k1.k2 -> sp_k1_k2
    fixed_scalar_products: Dict[Tuple[str, str], sp.Expr]  # external invariants
    dim: sp.Symbol = field(default_factory=lambda: sp.Symbol('d'))

    @property
    def n_props(self) -> int:
        return len(self.propagators)

    @property
    def n_loops(self) -> int:
        return len(self.loop_momenta)

    @property
    def free_scalar_products(self) -> List[sp.Symbol]:
        """All k_i.k_j and k_i.p_j scalar products (excludes
        purely-external p_i.p_j which are fixed)."""
        result = []
        loops = [str(k) for k in self.loop_momenta]
        for i, ki in enumerate(self.loop_momenta):
            for j, kj in enumerate(self.loop_momenta):
                if i <= j:
                    sp_sym = self.sp_symbols.get(
                        (str(ki), str(kj)),
                        self.sp_symbols.get((str(kj), str(ki)))
                    )
                    if sp_sym is not None and sp_sym not in result:
                        result.append(sp_sym)
            for ext in self.external_momenta:
                sp_sym = self.sp_symbols.get(
                    (str(ki), str(ext)),
                    self.sp_symbols.get((str(ext), str(ki)))
                )
                if sp_sym is not None and sp_sym not in result:
                    result.append(sp_sym)
        return result


# ─────────────────────────────────────────────────────────────────────
#  IBP identity generation
# ─────────────────────────────────────────────────────────────────────
@dataclass
class IBPIdentity:
    """
    A single IBP identity.

    Represented as a linear relation:
       sum_k  c_k * I[exps_k]  =  0
    where c_k are coefficients in Q(d, invariants) and exps_k are
    integer-exponent tuples.

    Internally stored as a SymPy expression (linear combination of
    IntegralSymbol calls).
    """
    relation: sp.Expr     # = 0
    seed_exps: ExpTuple   # the seed integral the identity was derived from
    derivative_label: str # e.g. "d/dk1 . k1"

    def __repr__(self):
        return f"IBPIdentity(seed={self.seed_exps}, label={self.derivative_label})"


def differentiate_inverse_prop(
    D: sp.Expr,
    momentum: sp.Symbol,
    family: IBPFamily,
) -> sp.Expr:
    """
    Compute (d/d momentum^mu)(v^mu D) where the result is again
    expressed in scalar products (the d-vector index mu is contracted
    appropriately).

    This is the non-trivial step that requires some Lorentz/index
    bookkeeping.  We use a simplified "projection" approach: for each
    scalar product s in D, compute d s / d momentum^mu summed against v^mu.

    For s = momentum.q (where q is another momentum), the result is q.
    For s = q.q (no momentum dep), it's 0.

    Concretely: d(k_i . k_j) / d k_l^mu summed against v^mu = v.k_i (l=j) + v.k_j (l=i).
    """
    # SymPy's differentiate with respect to a Symbol treats scalar-product
    # symbols as atomic; we handle it via the chain rule explicitly using
    # the structure we built.

    # Strategy: build a derivative dictionary {(s, momentum): partial_derivative_value}
    # where partial_derivative_value is itself a scalar product expression.
    return D  # placeholder; concrete logic in subclasses


# ─────────────────────────────────────────────────────────────────────
#  IBP system: collection of identities + reduction logic
# ─────────────────────────────────────────────────────────────────────
@dataclass
class IBPSystem:
    """
    A system of IBP identities for a given family.  Provides:
      - generate_identities(seeds): generate all IBP identities on the seed set
      - reduce(target): reduce the target integral to masters
    """
    family: IBPFamily
    identities: List[IBPIdentity] = field(default_factory=list)

    def generate_identities(self, seeds: List[ExpTuple]) -> None:
        """Generate IBP identities on each seed (concrete impl in subclass)."""
        raise NotImplementedError

    def reduce(self, target: ExpTuple) -> sp.Expr:
        """Reduce the target integral to a Q(d)-linear combination of masters."""
        raise NotImplementedError
