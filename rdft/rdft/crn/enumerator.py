"""
rdft.crn.enumerate
==================

Enumerators that take a ``CRN`` (or its ``phi(G)`` polynomial) and produce
a list of ``Diagram`` objects with full provenance:

* ``enumerate_phi_trees(phi, n)``     -- plane-tree shape strings, Catalan-style.
* ``enumerate_bubbles(crn)``           -- 2-vertex 1-loop bubbles with multi-species.
* ``enumerate_tadpoles(crn)``          -- 1-vertex self-loop tadpoles.
* ``classify_reggeon_topology(shape)`` -- shape -> (ladder|box|ice-cream|reducible).
"""
from __future__ import annotations
from itertools import product
from typing import Dict, List, Tuple

from rdft.crn.crn import CRN, Vertex
from rdft.crn.diagram import Diagram, Provenance
from rdft.crn.symmetry import (aut_phi_tree, aut_bubble, aut_tadpole,
                                k_symmetric_nodes)


# ---------------------------------------------------------------------------
# phi-tree enumeration
# ---------------------------------------------------------------------------

def enumerate_phi_trees(n: int, phi_polynomial=None) -> List[str]:
    """Enumerate plane phi-trees of size n for phi(G) = 1 + G^2.

    Returns a list of shape strings. ``[z^n]G`` is then ``len(result)``.
    """
    # Currently only phi(G) = 1+G^2 (plane binary trees) is implemented.
    return list(_enumerate_binary_trees(n))


def _enumerate_binary_trees(n: int):
    if n == 1:
        yield "L"
        return
    if n % 2 == 0:
        return
    for k in range(1, n - 1, 2):
        for left in _enumerate_binary_trees(k):
            for right in _enumerate_binary_trees(n - 1 - k):
                yield f"({left},{right})"


# ---------------------------------------------------------------------------
# Reggeon-DP topology classification
# ---------------------------------------------------------------------------

REGGEON_DP_TOPOLOGY_TABLE: Dict[str, Dict[str, object]] = {
    # shape -> {topology, is_1PI, label, reason}
    "(L,(L,(L,L)))": {
        "topology": "ladder", "is_1PI": True, "label": "ladder",
        "reason": "1PI: no single propagator cut disconnects",
    },
    "(L,((L,L),L))": {
        "topology": "box", "is_1PI": True, "label": "box",
        "reason": "1PI: no single propagator cut disconnects",
    },
    "((L,L),(L,L))": {
        "topology": "ice-cream", "is_1PI": True, "label": "ice-cream cone",
        "reason": "1PI: no single propagator cut disconnects",
    },
    "((L,(L,L)),L)": {
        "topology": "Sigma1_psi", "is_1PI": False,
        "label": "reducible: Sigma_1 on psi-leg",
        "reason": ("REDUCIBLE: deep nested subtree on a single branch "
                   "=> cut external psi-leg propagator => tree-vertex (+) Sigma_1"),
    },
    "(((L,L),L),L)": {
        "topology": "Sigma1_psit", "is_1PI": False,
        "label": "reducible: Sigma_1 on psi-tilde-leg",
        "reason": ("REDUCIBLE: deep nested subtree on a single branch "
                   "=> cut external psit-leg propagator => tree-vertex (+) Sigma_1"),
    },
    # 1-loop bubble: size 3
    "(L,L)": {
        "topology": "bubble", "is_1PI": True, "label": "1-loop self-energy bubble",
        "reason": "1PI: a single internal-line cut leaves the other connecting v1, v2",
    },
}


def classify_reggeon_topology(shape: str) -> Dict[str, object]:
    """Look up topology metadata for a Reggeon-DP shape string."""
    return REGGEON_DP_TOPOLOGY_TABLE.get(shape, {
        "topology": "unknown", "is_1PI": False, "label": shape, "reason": "",
    })


# ---------------------------------------------------------------------------
# phi-tree -> Diagram with provenance
# ---------------------------------------------------------------------------

def diagram_from_phi_tree(shape: str, *, crn_name: str = "Reggeon DP",
                          loop_order: int = None) -> Diagram:
    """Construct a Diagram from a plane phi-tree shape string."""
    classify = classify_reggeon_topology(shape)
    n_internal = shape.count("(")            # = number of internal nodes
    L = (shape.count(",") + 0) // 2 if shape != "L" else 0  # rough
    L = n_internal - 1 if n_internal > 0 else 0
    if loop_order is not None:
        L = loop_order

    s_aut = aut_phi_tree(shape)
    k = k_symmetric_nodes(shape)

    d = Diagram(
        id=f"{crn_name}.{shape}",
        shape=shape,
        topology=str(classify.get("topology", "")),
        is_1PI=bool(classify.get("is_1PI", True)),
        s_aut=s_aut,
    )
    d.add_provenance("Layer 1", "Doi shift -> phi(G)",
                     f"{crn_name}: phi(G) = 1+G^2 (cubic interactions)",
                     reference="rdft/crn/crn.py: CRN.phi_polynomial")
    d.add_provenance("Layer 2", "Lagrange inversion",
                     f"shape {shape!r} appears in plane phi-tree enumeration at size {n_internal*2+1}",
                     reference="rdft/crn/enumerate.py: enumerate_phi_trees")
    d.add_provenance("Layer 2", "phi-tree |Aut| via 2^k",
                     f"k(T) = {k} symmetric internal node(s) => |Aut| = 2^{k} = {s_aut}",
                     reference="rdft/crn/symmetry.py: aut_phi_tree")
    if classify.get("topology"):
        d.add_provenance("Layer 3", "1PI verdict from shape",
                         classify.get("reason", ""),
                         reference="rdft/crn/enumerate.py: classify_reggeon_topology")
    return d


# ---------------------------------------------------------------------------
# Multi-species bubble enumeration (port from poc_brw.py)
# ---------------------------------------------------------------------------

def _vertex_species_legs(v: Vertex):
    """Return (in_a, in_b, out_a, out_b) for a 2-species (A,B) vertex."""
    return (v.in_dict().get("A", 0), v.in_dict().get("B", 0),
            v.out_dict().get("A", 0), v.out_dict().get("B", 0))


def enumerate_bubbles(vertices: Tuple[Vertex, ...]) -> List[Diagram]:
    """Enumerate all 1-loop bubble topologies (V=2, I=2) over the given vertex
    set, with internal lines decorated by species (A/B) and direction (lr/rl)."""
    species_options = ("A", "B")
    direction_options = ("lr", "rl")
    raw = []
    for v1 in vertices:
        for v2 in vertices:
            for (s1, d1), (s2, d2) in product(
                product(species_options, direction_options),
                product(species_options, direction_options),
            ):
                if not _bubble_valid(v1, v2, s1, d1, s2, d2):
                    continue
                key = _bubble_signature(v1, v2, s1, d1, s2, d2)
                raw.append((key, v1, v2, s1, d1, s2, d2))

    seen = set()
    diagrams = []
    for key, v1, v2, s1, d1, s2, d2 in raw:
        if key in seen:
            continue
        seen.add(key)
        ext = _bubble_externals(v1, v2, s1, d1, s2, d2)
        s_aut = aut_bubble(v1.name, v2.name, s1, d1, s2, d2)
        E = sum(ext.values())
        d = Diagram(
            id=f"bubble.{v1.name}+{v2.name}.{s1}{d1}-{s2}{d2}.E{E}",
            vertex_names=(v1.name, v2.name),
            internal_lines=((s1, d1), (s2, d2)),
            external_legs=dict(ext),
            is_1PI=True,
            s_aut=s_aut,
        )
        d.add_provenance("Layer 1", "Doi shift",
                         f"vertex pair ({v1.name}, {v2.name}) from CRN",
                         reference="rdft/crn/crn.py")
        d.add_provenance("Layer 2", "bubble enumeration",
                         f"V=2, I=2 with internal species ({s1},{s2}) and directions ({d1},{d2})",
                         reference="rdft/crn/enumerate.py: enumerate_bubbles")
        d.add_provenance("Layer 2", "directed-graph |Aut|",
                         _aut_explain_bubble(v1, v2, s1, d1, s2, d2, s_aut),
                         reference="rdft/crn/symmetry.py: aut_bubble")
        d.add_provenance("Layer 3", "1PI verdict",
                         "1PI: removing any one internal line leaves v1 connected to v2 via the other",
                         reference="cut rule")
        diagrams.append(d)
    return diagrams


def _bubble_valid(v1, v2, s1, d1, s2, d2) -> bool:
    used = {("v1", "in_a"): 0, ("v1", "in_b"): 0,
            ("v1", "out_a"): 0, ("v1", "out_b"): 0,
            ("v2", "in_a"): 0, ("v2", "in_b"): 0,
            ("v2", "out_a"): 0, ("v2", "out_b"): 0}
    for s, d in [(s1, d1), (s2, d2)]:
        sp_low = s.lower()
        if d == "lr":
            used[("v1", f"out_{sp_low}")] += 1
            used[("v2", f"in_{sp_low}")] += 1
        else:
            used[("v1", f"in_{sp_low}")] += 1
            used[("v2", f"out_{sp_low}")] += 1
    v1_a_in, v1_b_in, v1_a_out, v1_b_out = _vertex_species_legs(v1)
    v2_a_in, v2_b_in, v2_a_out, v2_b_out = _vertex_species_legs(v2)
    return (v1_a_in >= used[("v1", "in_a")] and v1_b_in >= used[("v1", "in_b")]
            and v1_a_out >= used[("v1", "out_a")] and v1_b_out >= used[("v1", "out_b")]
            and v2_a_in >= used[("v2", "in_a")] and v2_b_in >= used[("v2", "in_b")]
            and v2_a_out >= used[("v2", "out_a")] and v2_b_out >= used[("v2", "out_b")])


def _bubble_externals(v1, v2, s1, d1, s2, d2) -> Dict[Tuple[str, str], int]:
    used = {("v1", "in_a"): 0, ("v1", "in_b"): 0,
            ("v1", "out_a"): 0, ("v1", "out_b"): 0,
            ("v2", "in_a"): 0, ("v2", "in_b"): 0,
            ("v2", "out_a"): 0, ("v2", "out_b"): 0}
    for s, d in [(s1, d1), (s2, d2)]:
        sp_low = s.lower()
        if d == "lr":
            used[("v1", f"out_{sp_low}")] += 1
            used[("v2", f"in_{sp_low}")] += 1
        else:
            used[("v1", f"in_{sp_low}")] += 1
            used[("v2", f"out_{sp_low}")] += 1
    v1_a_in, v1_b_in, v1_a_out, v1_b_out = _vertex_species_legs(v1)
    v2_a_in, v2_b_in, v2_a_out, v2_b_out = _vertex_species_legs(v2)
    return {
        ("A", "in"):  (v1_a_in - used[("v1", "in_a")]) + (v2_a_in - used[("v2", "in_a")]),
        ("A", "out"): (v1_a_out - used[("v1", "out_a")]) + (v2_a_out - used[("v2", "out_a")]),
        ("B", "in"):  (v1_b_in - used[("v1", "in_b")]) + (v2_b_in - used[("v2", "in_b")]),
        ("B", "out"): (v1_b_out - used[("v1", "out_b")]) + (v2_b_out - used[("v2", "out_b")]),
    }


def _bubble_signature(v1, v2, s1, d1, s2, d2):
    flip = {"lr": "rl", "rl": "lr"}
    cand_a = ((v1.name, v2.name), tuple(sorted([(s1, d1), (s2, d2)])))
    cand_b = ((v2.name, v1.name), tuple(sorted([(s1, flip[d1]), (s2, flip[d2])])))
    canon = min(cand_a, cand_b)
    ext = _bubble_externals(v1, v2, s1, d1, s2, d2)
    return (canon, tuple(sorted(ext.items())))


def _aut_explain_bubble(v1, v2, s1, d1, s2, d2, s_aut) -> str:
    parts = []
    if s1 == s2 and d1 == d2:
        parts.append("internal-line swap (same species and direction)")
    if v1.name == v2.name:
        flip = {"lr": "rl", "rl": "lr"}
        old = sorted([(s1, d1), (s2, d2)])
        new = sorted([(s1, flip[d1]), (s2, flip[d2])])
        if old == new:
            parts.append("vertex-swap (same vertex type, line set invariant)")
    if not parts:
        return f"|Aut| = {s_aut}: no nontrivial automorphism"
    return f"|Aut| = {s_aut} from: " + " + ".join(parts)


# ---------------------------------------------------------------------------
# Single-vertex tadpole enumeration
# ---------------------------------------------------------------------------

def enumerate_tadpoles(vertices: Tuple[Vertex, ...]) -> List[Diagram]:
    """Enumerate 1-vertex self-loop tadpoles."""
    diagrams = []
    for v in vertices:
        for sp_name in ("A", "B"):
            v_in = v.in_dict().get(sp_name, 0)
            v_out = v.out_dict().get(sp_name, 0)
            if v_in < 1 or v_out < 1:
                continue
            ext = {
                ("A", "in"):  v.in_dict().get("A", 0)  - (1 if sp_name == "A" else 0),
                ("A", "out"): v.out_dict().get("A", 0) - (1 if sp_name == "A" else 0),
                ("B", "in"):  v.in_dict().get("B", 0)  - (1 if sp_name == "B" else 0),
                ("B", "out"): v.out_dict().get("B", 0) - (1 if sp_name == "B" else 0),
            }
            E = sum(ext.values())
            d = Diagram(
                id=f"tadpole.{v.name}.{sp_name}.E{E}",
                vertex_names=(v.name,),
                internal_lines=((sp_name, "self"),),
                external_legs=ext,
                is_1PI=True,
                s_aut=aut_tadpole(),
            )
            d.add_provenance("Layer 1", "Doi shift",
                             f"vertex {v.name} from CRN")
            d.add_provenance("Layer 2", "tadpole enumeration",
                             f"V=1, self-loop on species {sp_name}",
                             reference="rdft/crn/enumerate.py: enumerate_tadpoles")
            d.add_provenance("Layer 2", "directed-graph |Aut|",
                             "|Aut| = 1: in-leg and out-leg of self-loop are distinguishable",
                             reference="rdft/crn/symmetry.py: aut_tadpole")
            diagrams.append(d)
    return diagrams
