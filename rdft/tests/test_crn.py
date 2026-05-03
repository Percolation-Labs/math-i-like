"""Tests for the rdft.crn package: end-to-end CRN -> RG pipeline."""
from __future__ import annotations

import sympy as sp

from rdft.crn import CRN, RGProgram
from rdft.crn.symmetry import (aut_phi_tree, aut_bubble, aut_tadpole,
                                k_symmetric_nodes, parse_shape)
from rdft.crn.enumerator import (enumerate_phi_trees, diagram_from_phi_tree,
                                  enumerate_bubbles, enumerate_tadpoles)


# ---------------------------------------------------------------------------
# Layer 1: CRN builders + Doi shift
# ---------------------------------------------------------------------------

def test_reggeon_dp_phi_polynomial():
    crn = CRN.reggeon_dp()
    G = sp.Symbol("G")
    assert crn.phi_polynomial(G, max_legs=3) == 1 + G**2


def test_dyadic_brw_phi_polynomial():
    crn = CRN.dyadic_brw()
    G = sp.Symbol("G")
    assert crn.phi_polynomial(G, max_legs=3) == 1 + G**2


def test_brw_thesis_has_seven_vertices():
    brw = CRN.brw_thesis()
    assert len(brw.vertices) == 7
    assert "V_branch" in {v.name for v in brw.vertices}


# ---------------------------------------------------------------------------
# Layer 2: Lagrange counts and phi-tree |Aut|
# ---------------------------------------------------------------------------

def test_lagrange_counts_catalan():
    # phi(G) = 1+G^2 produces Catalan numbers at odd sizes
    assert len(enumerate_phi_trees(1)) == 1
    assert len(enumerate_phi_trees(3)) == 1
    assert len(enumerate_phi_trees(5)) == 2
    assert len(enumerate_phi_trees(7)) == 5


def test_phi_tree_aut_rule():
    # AC rule: |Aut(T)| = 2^k(T)
    assert aut_phi_tree("L") == 1
    assert aut_phi_tree("(L,L)") == 2          # bubble (size 3)
    assert aut_phi_tree("(L,(L,(L,L)))") == 2   # ladder
    assert aut_phi_tree("(L,((L,L),L))") == 2   # box
    assert aut_phi_tree("((L,L),(L,L))") == 8   # ice-cream
    assert aut_phi_tree("((L,(L,L)),L)") == 2   # Sigma_1 on psi
    assert aut_phi_tree("(((L,L),L),L)") == 2   # Sigma_1 on psitilde


def test_k_symmetric_nodes():
    assert k_symmetric_nodes("(L,L)") == 1
    assert k_symmetric_nodes("((L,L),(L,L))") == 3
    assert k_symmetric_nodes("(L,(L,(L,L)))") == 1


def test_parse_shape_roundtrip():
    for s in ["L", "(L,L)", "(L,(L,L))", "((L,L),(L,L))"]:
        parsed = parse_shape(s)
        assert parsed is not None


# ---------------------------------------------------------------------------
# Bubble + tadpole enumeration
# ---------------------------------------------------------------------------

def test_brw_thesis_36_bubbles_with_seven_e3():
    brw = CRN.brw_thesis()
    bubbles = enumerate_bubbles(brw.vertices)
    assert len(bubbles) == 36
    e3 = [b for b in bubbles if sum(b.external_legs.values()) == 3]
    assert len(e3) == 7


def test_brw_thesis_seven_tadpoles():
    brw = CRN.brw_thesis()
    tadpoles = enumerate_tadpoles(brw.vertices)
    assert len(tadpoles) == 7


def test_canonical_bubble_aut_is_two():
    """Thesis Eq. (3.25): s(V_branch + V_branch, AA, opposite-direction) = 2."""
    s = aut_bubble("V_branch", "V_branch", "A", "lr", "A", "rl")
    assert s == 2


def test_tadpole_aut_is_one():
    assert aut_tadpole() == 1


# ---------------------------------------------------------------------------
# Reggeon-DP topology classification
# ---------------------------------------------------------------------------

def test_reggeon_dp_topology_classification():
    shapes_to_topology = {
        "(L,(L,(L,L)))":   ("ladder",       True),
        "(L,((L,L),L))":   ("box",          True),
        "((L,L),(L,L))":   ("ice-cream",    True),
        "((L,(L,L)),L)":   ("Sigma1_psi",   False),
        "(((L,L),L),L)":   ("Sigma1_psit",  False),
    }
    for shape, (expected_topo, is_1pi) in shapes_to_topology.items():
        d = diagram_from_phi_tree(shape)
        assert d.topology == expected_topo
        assert d.is_1PI == is_1pi


# ---------------------------------------------------------------------------
# Provenance: every Diagram has a non-empty lineage chain
# ---------------------------------------------------------------------------

def test_diagram_lineage_populated():
    d = diagram_from_phi_tree("(L,(L,(L,L)))")
    assert len(d.lineage) >= 3
    layers = {p.layer for p in d.lineage}
    assert "Layer 1" in layers
    assert "Layer 2" in layers


# ---------------------------------------------------------------------------
# RGProgram end-to-end (slow but the linchpin test)
# ---------------------------------------------------------------------------

def test_rgprogram_reggeon_dp_zero_residual():
    rg = RGProgram(CRN.reggeon_dp(), loop_order=2).run()
    res = rg.exponents.compare_to_jt05()
    assert all(res.values()), f"non-zero residuals: {rg.exponents.residuals}"


def test_rgprogram_double_poles_match_jt05_eq57():
    rg = RGProgram(CRN.reggeon_dp(), loop_order=2).run()
    expected = {"psi": sp.Rational(7, 32), "lambda": sp.Rational(13, 128),
                "tau": sp.Rational(1, 2), "u": sp.Rational(7, 2)}
    for X, exp_v in expected.items():
        assert rg.zfactors[X].double_pole == exp_v


def test_rgprogram_simple_poles_match_jt05_eq57():
    rg = RGProgram(CRN.reggeon_dp(), loop_order=2).run()
    L = sp.Symbol("L")
    # JT05 Eq.(57) simple poles
    expected = {
        "psi":    sp.Rational(-3, 32) + sp.Rational(9, 64) * L,
        "lambda": sp.Rational(-31, 512) + sp.Rational(35, 256) * L,
        "tau":    sp.Rational(-5, 32),
        "u":      sp.Rational(-7, 8),
    }
    for X, exp_v in expected.items():
        zf = rg.zfactors[X]
        got = zf.simple_pole_rat + zf.simple_pole_L * L
        assert sp.simplify(got - exp_v) == 0


def test_rgprogram_audit_runs():
    rg = RGProgram(CRN.reggeon_dp(), loop_order=2).run()
    txt = rg.audit()
    assert "AUDIT TRAIL" in txt
    assert "ALL MATCH (zero residual): True" in txt


# ---------------------------------------------------------------------------
# Legendre transform numerics (slow)
# ---------------------------------------------------------------------------

def test_legendre_reggeon_dp_w_and_gamma_coefficients():
    from rdft.crn.legendre import legendre_reggeon_dp
    r = legendre_reggeon_dp(N_g=5, J_max=4)
    # Sanity: Gamma at g=0 is the free action
    Phi, Phit = r.fields
    g = r.coupling
    assert sp.expand(sp.Poly(r.Gamma, g).coeff_monomial(g**0)) == Phi * Phit
    # The flagship 2-loop numbers
    assert r.W_coef(2, 1, 5) == -7608
    assert r.Gamma_coef(2, 1, 5) == -504
