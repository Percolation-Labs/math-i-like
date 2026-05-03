"""
rdft.crn.symmetry
=================

Two routes to the symmetry factor ``s(Gamma) = |Aut(Gamma)|``:

1. ``aut_phi_tree(shape)`` -- the AC route via the rooted plane phi-tree:
   |Aut(T)| = 2^{k(T)}, where k(T) counts internal nodes whose two children
   are isomorphic sub-trees.

2. ``aut_directed_graph(diagram)`` -- the directed Feynman graph route:
   counts vertex-swap and internal-line-swap automorphisms that fix
   external-leg identities.

The two should agree on graphs where the phi-tree representation is
unambiguous. ``cross_check_aut(diagram)`` runs both and asserts.
"""
from __future__ import annotations
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Route 1: phi-tree |Aut| from shape string
# ---------------------------------------------------------------------------

def parse_shape(shape: str):
    """Parse a phi-tree shape string into a nested tuple.

    >>> parse_shape("L")
    'L'
    >>> parse_shape("(L,L)")
    ('L', 'L')
    >>> parse_shape("(L,(L,L))")
    ('L', ('L', 'L'))
    """
    s = shape.replace(" ", "")
    if s == "L":
        return "L"
    assert s[0] == "(" and s[-1] == ")", f"bad shape {shape!r}"
    # find the comma at depth 0 inside the outer parens
    depth = 0
    inner = s[1:-1]
    for i, c in enumerate(inner):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            return (parse_shape(inner[:i]), parse_shape(inner[i + 1:]))
    raise ValueError(f"bad shape: {shape!r}")


def aut_phi_tree(shape: str) -> int:
    """|Aut(T)| for a rooted plane binary phi-tree, given by 2^k(T).

    k(T) = number of internal nodes whose two children are isomorphic sub-trees.
    A "leaf" L counts as identical to any other leaf L; a sub-tree is identical
    to another iff their parsed structures match.
    """
    if isinstance(shape, str):
        tree = parse_shape(shape)
    else:
        tree = shape
    return _aut_count(tree)


def _aut_count(tree) -> int:
    if tree == "L":
        return 1
    left, right = tree
    factor = 2 if left == right else 1
    return factor * _aut_count(left) * _aut_count(right)


def k_symmetric_nodes(shape: str) -> int:
    """Number of internal nodes in T with isomorphic children (the k in 2^k)."""
    tree = parse_shape(shape) if isinstance(shape, str) else shape
    return _k_count(tree)


def _k_count(tree) -> int:
    if tree == "L":
        return 0
    left, right = tree
    here = 1 if left == right else 0
    return here + _k_count(left) + _k_count(right)


# ---------------------------------------------------------------------------
# Route 2: directed-graph |Aut| for a 1-loop bubble (2 vertices, 2 internal lines)
# ---------------------------------------------------------------------------

def aut_bubble(v1_name: str, v2_name: str,
               line1_species: str, line1_dir: str,
               line2_species: str, line2_dir: str) -> int:
    """|Aut| for the 1-loop bubble: 2 vertices, 2 internal lines.

    Two automorphism sources:
      - Internal-line swap: the two lines are indistinguishable iff they have
        the same species AND the same direction.
      - Vertex swap: v1 <-> v2; flips each line's direction. Is an automorphism
        iff v1 and v2 are the same vertex type AND the line set (as a multiset
        of (species, direction) pairs) is invariant under the flip.
    """
    aut = 1
    # Internal-line swap
    if line1_species == line2_species and line1_dir == line2_dir:
        aut *= 2
    # Vertex swap
    if v1_name == v2_name:
        flip = {"lr": "rl", "rl": "lr"}
        old = sorted([(line1_species, line1_dir), (line2_species, line2_dir)])
        new = sorted([(line1_species, flip[line1_dir]),
                      (line2_species, flip[line2_dir])])
        if old == new:
            aut *= 2
    return aut


def aut_tadpole() -> int:
    """|Aut| for a single-vertex tadpole (self-loop).

    The two endpoints of the self-loop are an in-leg and an out-leg of the
    same species, which are distinguishable as field types. So |Aut| = 1.
    """
    return 1


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------

def cross_check_bubble_aut(shape: str, v_name: str,
                            line1_species: str, line1_dir: str,
                            line2_species: str, line2_dir: str) -> Tuple[int, int, bool]:
    """Compute |Aut| via both routes (phi-tree and directed graph) for a
    1-loop bubble and report them with a match flag.

    Returns ``(aut_tree, aut_graph, match)``.
    """
    aut_tree = aut_phi_tree(shape)
    aut_graph = aut_bubble(v_name, v_name,
                           line1_species, line1_dir,
                           line2_species, line2_dir)
    return aut_tree, aut_graph, (aut_tree == aut_graph)
