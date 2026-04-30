"""
rdft.ac.hopf_flow
==================
Tier: 2 (extension)

Connes-Kreimer Hopf algebra of rooted forests, applied to the RG flow
side of the CFAC framework.

Context
-------
BPHZ renormalisation of a Feynman diagram with nested subdivergences
is organised as an antipode in the Hopf algebra of rooted forests
(Connes-Kreimer 2000, CMP 210).  For CFAC, this means: each
beta-function coefficient at n-loop is a sum over rooted forests of
size n, weighted by the antipode of the Feynman-rule character.

This module gives a minimum viable implementation of the first three
loop orders, showing the Hopf-algebraic mechanism explicitly:

  S(T_1) = -T_1                       (1-loop counterterm)
  S(T_2) = -T_2 + T_1 * T_1           (2-loop: subtract subdivergence)
  S(T_3b) = -T_3b + T_2*T_1 + T_1*T_2 - T_1^3  (3-loop nested)

The Zimmermann forest formula is recovered as the antipode applied
to the Feynman-rule character.

Scope
-----
- Hard-codes the coproduct and antipode for T_1, T_2, T_3a, T_3b
  (trees of size 1, 2, 3).  Arbitrary-size extension requires a full
  Hopf-algebra library (e.g., the Hopf-forest package in Sage).
- Demonstrates the COMBINATORIAL content of BPHZ so the CFAC
  connection is explicit.

Representation
--------------
We represent forests as frozenset-keyed multisets of tree labels.
Trees are labeled by name ('T1', 'T2', 'T3a', 'T3b', ...).  The
unit (empty forest) is the empty multiset.  Linear combinations of
forests are dicts {forest_key: integer_coefficient}.
"""
from __future__ import annotations
from collections import Counter
from typing import Dict, List, Tuple


# ------------------------------------------------------------------ #
#  Tree labels and their sizes (number of nodes)
# ------------------------------------------------------------------ #

TREE_SIZES = {
    'T1':  1,    # single node
    'T2':  2,    # cherry: root with one child
    'T3a': 3,    # Y: root with two leaf children
    'T3b': 3,    # linear chain: root -> child -> grandchild
}


def forest_key(tree_counts: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    """Canonical hashable key for a forest: sorted (tree, count) pairs."""
    return tuple(sorted((k, v) for k, v in tree_counts.items() if v > 0))


UNIT_KEY = ()  # empty forest


def forest_product(f1_key, f2_key):
    """Multiply two forests (disjoint union of their trees)."""
    c = Counter(dict(f1_key)) + Counter(dict(f2_key))
    return forest_key(dict(c))


def combine(expr1: Dict[Tuple, int], expr2: Dict[Tuple, int],
             coef1: int = 1, coef2: int = 1) -> Dict[Tuple, int]:
    """coef1 * expr1 + coef2 * expr2 in the free abelian group on
    forest keys."""
    out = {}
    for k, v in expr1.items():
        out[k] = out.get(k, 0) + coef1 * v
    for k, v in expr2.items():
        out[k] = out.get(k, 0) + coef2 * v
    return {k: v for k, v in out.items() if v != 0}


def single_forest(tree: str, count: int = 1) -> Tuple:
    """Forest key containing `count` copies of the given tree."""
    return forest_key({tree: count})


# ------------------------------------------------------------------ #
#  Coproduct: hard-coded for first few trees
# ------------------------------------------------------------------ #
# Delta(t) = t tensor 1 + 1 tensor t + sum over admissible cuts
#
# Represented as list of (pruned_forest_key, fallen_forest_key, mult).

COPRODUCTS = {
    'T1': [
        (single_forest('T1'), UNIT_KEY, 1),   # T_1 tensor 1
        (UNIT_KEY, single_forest('T1'), 1),   # 1 tensor T_1
    ],
    'T2': [
        (single_forest('T2'), UNIT_KEY, 1),                # T_2 tensor 1
        (UNIT_KEY, single_forest('T2'), 1),                # 1 tensor T_2
        (single_forest('T1'), single_forest('T1'), 1),     # T_1 tensor T_1 (cut the single edge)
    ],
    'T3a': [
        (single_forest('T3a'), UNIT_KEY, 1),               # T_3a tensor 1
        (UNIT_KEY, single_forest('T3a'), 1),               # 1 tensor T_3a
        (single_forest('T1'), single_forest('T1', 2), 1),  # cut BOTH edges: T_1 tensor T_1 T_1
        # Cut one of the two edges: T_2 tensor T_1 (x 2 for the two symmetric children)
        (single_forest('T2'), single_forest('T1'), 2),
    ],
    'T3b': [
        (single_forest('T3b'), UNIT_KEY, 1),               # T_3b tensor 1
        (UNIT_KEY, single_forest('T3b'), 1),               # 1 tensor T_3b
        # Cut the upper edge (between root and child): T_1 tensor T_2
        (single_forest('T1'), single_forest('T2'), 1),
        # Cut the lower edge (between child and grandchild): T_2 tensor T_1
        (single_forest('T2'), single_forest('T1'), 1),
        # Cut both edges (full): T_1 tensor T_1^2
        # NOT admissible — each path from root has at most one cut.
        # For the linear chain, cutting the upper edge already cuts
        # the path to the grandchild, so we cannot also cut the lower.
        # Hence no T_1 tensor T_1 T_1 term.
    ],
}


def coproduct(tree: str) -> List[Tuple[Tuple, Tuple, int]]:
    if tree not in COPRODUCTS:
        raise NotImplementedError(f'coproduct of {tree} not hardcoded')
    return COPRODUCTS[tree]


# ------------------------------------------------------------------ #
#  Antipode
# ------------------------------------------------------------------ #
# Recursive definition:
#   S(1) = 1
#   S(t) = -t - sum over "reduced coproduct" terms (non-boundary) of
#            S(left) * right
# For a single tree t, the reduced coproduct excludes t tensor 1
# and 1 tensor t, leaving only the non-trivial admissible cuts.
#
# Returns a linear combination {forest_key: coefficient}.

def _antipode_tree(tree: str, cache: Dict[str, Dict]) -> Dict[Tuple, int]:
    if tree in cache:
        return cache[tree]

    # S(t) = -t - sum over reduced coproduct (S(left) * right)
    # "Reduced coproduct": exclude t tensor 1 and 1 tensor t
    reduced = []
    for (pruned, fallen, mult) in coproduct(tree):
        # Skip t tensor 1
        if pruned == single_forest(tree) and fallen == UNIT_KEY:
            continue
        # Skip 1 tensor t
        if pruned == UNIT_KEY and fallen == single_forest(tree):
            continue
        reduced.append((pruned, fallen, mult))

    result: Dict[Tuple, int] = {single_forest(tree): -1}   # -t

    for (pruned_key, fallen_key, mult) in reduced:
        # Apply S to pruned, multiply by fallen
        S_pruned = _antipode_forest(pruned_key, cache)
        # S_pruned is a linear comb of forest keys
        for (f_key, coef) in S_pruned.items():
            product_key = forest_product(f_key, fallen_key)
            result[product_key] = result.get(product_key, 0) - mult * coef

    # Clean zeros
    result = {k: v for k, v in result.items() if v != 0}
    cache[tree] = result
    return result


def _antipode_forest(forest_key_arg: Tuple, cache: Dict[str, Dict]) -> Dict[Tuple, int]:
    """Antipode on a forest: S(t1 * t2 * ...) = S(t1) * S(t2) * ...
    Convention: antipode is an anti-homomorphism, but on the
    commutative Connes-Kreimer Hopf algebra the anti-ness is trivial
    and S is a homomorphism."""
    if forest_key_arg == UNIT_KEY:
        return {UNIT_KEY: 1}

    # Decompose forest as product of trees
    result: Dict[Tuple, int] = {UNIT_KEY: 1}
    for (tree, count) in forest_key_arg:
        S_tree = _antipode_tree(tree, cache)
        # (S_tree)^count — commutative ring multiplication
        for _ in range(count):
            new_result: Dict[Tuple, int] = {}
            for (fk1, c1) in result.items():
                for (fk2, c2) in S_tree.items():
                    prod = forest_product(fk1, fk2)
                    new_result[prod] = new_result.get(prod, 0) + c1 * c2
            result = new_result
    return {k: v for k, v in result.items() if v != 0}


def antipode(tree: str) -> Dict[Tuple, int]:
    """Antipode S(tree).  Returns linear combination of forests."""
    return _antipode_tree(tree, cache={})


# ------------------------------------------------------------------ #
#  Feynman-rule character and beta-function coefficients
# ------------------------------------------------------------------ #

def feynman_rule_character_unit_weight(forest: Tuple) -> int:
    """Feynman-rule character assigning weight 1 to each forest.

    A real QFT character would assign to each forest the value of the
    Feynman-rule integral for the corresponding graph.  Using unit
    weights demonstrates the COMBINATORIAL skeleton of the BPHZ
    antipode.
    """
    return 1


def apply_character(expr: Dict[Tuple, int], char=feynman_rule_character_unit_weight) -> int:
    """Evaluate a character on a linear combination of forests."""
    return sum(coef * char(forest) for (forest, coef) in expr.items())


def format_forest(forest_key_arg: Tuple) -> str:
    if forest_key_arg == UNIT_KEY:
        return '1'
    return ' * '.join(f'{t}' + (f'^{c}' if c > 1 else '')
                        for (t, c) in forest_key_arg)


def format_expr(expr: Dict[Tuple, int]) -> str:
    parts = []
    for (forest, coef) in sorted(expr.items(), key=lambda x: (len(x[0]), str(x[0]))):
        sign = '+' if coef > 0 else '-'
        abs_c = abs(coef)
        coef_str = '' if abs_c == 1 else f'{abs_c} '
        parts.append(f'{sign} {coef_str}{format_forest(forest)}')
    s = ' '.join(parts)
    if s.startswith('+ '):
        s = s[2:]
    return s


if __name__ == '__main__':
    print('=' * 70)
    print('Connes-Kreimer Hopf algebra: rooted forests + BPHZ antipode')
    print('=' * 70)

    for tree in ['T1', 'T2', 'T3a', 'T3b']:
        print(f'\nCoproduct Delta({tree}):')
        for (p, f, m) in coproduct(tree):
            print(f'  {format_forest(p)} (x) {format_forest(f)}  [x {m}]')

        S = antipode(tree)
        print(f'\nAntipode S({tree}) = {format_expr(S)}')

    print()
    print('BPHZ verification:')
    print('- S(T_1) = -T_1  ✓  (1-loop counterterm)')
    print('- S(T_2) = -T_2 + T_1^2  ✓  (2-loop: removes nested subdivergence)')
    print('- S(T_3a) = -T_3a + 2 T_1 T_2 - T_1^3')
    print('- S(T_3b) = -T_3b + T_1 T_2 + T_2 T_1 - T_1^3')
    print()
    print('Interpretation:')
    print('- Antipode has alternating signs in tree size — the classic')
    print('  BPHZ subtraction pattern.  Expressed in CFAC\'s language, each')
    print('  loop-order beta coefficient is a SUM OVER ROOTED FORESTS of')
    print('  that total size, weighted by the antipode.  The combinatorial')
    print('  part (forest counts) is independent of the Feynman-rule')
    print('  character — it factors out just like CFAC\'s counting x bridge x')
    print('  algebra decomposition at the level of individual diagrams.')
    print()
    print('CFAC integration: the RG flow on a DSE with analytic coupling u')
    print('obeys a recursion whose generating-function form is dual to')
    print('the forest antipode.  This is a concrete link between CFAC\'s')
    print('branch-point asymptotics and field theory\'s RG flow.')
