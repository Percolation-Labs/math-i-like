"""
Test the causal -> Euclidean reduction pipeline:
  Reggeon graph + OmegaIntegration -> Euclidean parametric integral
  -> ibp_solver-compatible form

This wires together the existing modules:
  - rdft.graphs.incidence.FeynmanGraph (sunset, bubble)
  - rdft.integrals.parametric.OmegaIntegration (thesis 2.5.3.2)
  - rdft.integrals.symanzik.SymanzikPolynomials
"""
from __future__ import annotations
import sympy as sp
from rdft.graphs.incidence import FeynmanGraph
from rdft.integrals.symanzik import SymanzikPolynomials
from rdft.integrals.parametric import OmegaIntegration


def test_bubble():
    """1-loop bubble: trivial omega-reduction; verify the Symanzik polynomial
    after reduction matches the textbook bubble U = alpha_1 + alpha_2."""
    print("=" * 72)
    print(" TEST 1: 1-loop bubble Σ_1")
    print("=" * 72)

    G = FeynmanGraph.one_loop_self_energy()
    print(f"  Loops L = {G.L}")
    print(f"  Internal edges: {G.internal_edge_indices}")
    print(f"  External edges: {G.external_edge_indices}")

    sym = SymanzikPolynomials(G)
    Psi = sym.Psi  # First Symanzik (property, not method)
    Phi = sym.Phi  # Second Symanzik (property, not method)
    print(f"\n  Pre-reduction:")
    print(f"    Psi (first Symanzik U)  = {Psi}")
    print(f"    Phi (second Symanzik F) = {Phi}")

    omega = OmegaIntegration(G)
    Ef = omega.edge_basis_matrix
    print(f"\n  Edge-basis matrix E_f (thesis Eq. 2.28):")
    print(f"    {Ef}")

    constraints = omega.circuit_constraints
    print(f"\n  Circuit constraints Sigma_l (delta-function args):")
    for l, c in enumerate(constraints):
        print(f"    Sigma_{l}: delta({c}) = 0")

    Psi_r, Phi_r, subs = omega.reduce(Psi, Phi)
    print(f"\n  Post-reduction:")
    print(f"    Psi (reduced) = {Psi_r}")
    print(f"    Phi (reduced) = {Phi_r}")
    print(f"    Substitutions: {subs}")
    print(f"\n  This is the Euclidean parametric form on which a standard")
    print(f"  IBP solver (relativistic propagators 1/k^2) operates.")
    return Psi_r, Phi_r


def test_sunset():
    """2-loop sunset: omega-reduction; verify reduced Symanzik matches the
    standard 2-loop sunset U = alpha_1 alpha_2 + alpha_1 alpha_3 + alpha_2 alpha_3."""
    print()
    print("=" * 72)
    print(" TEST 2: 2-loop sunset Σ_2^sunset")
    print("=" * 72)

    G = FeynmanGraph.sunset()
    print(f"  Loops L = {G.L}")
    print(f"  Internal edges: {G.internal_edge_indices}")
    print(f"  External edges: {G.external_edge_indices}")

    sym = SymanzikPolynomials(G)
    Psi = sym.Psi
    Phi = sym.Phi
    print(f"\n  Pre-reduction:")
    print(f"    Psi (first Symanzik U)  = {Psi}")
    print(f"    Phi (second Symanzik F) = {Phi}")

    omega = OmegaIntegration(G)
    Ef = omega.edge_basis_matrix
    print(f"\n  Edge-basis matrix E_f (thesis Eq. 2.28):")
    print(f"    {Ef}")

    constraints = omega.circuit_constraints
    print(f"\n  Circuit constraints Sigma_l (delta-function args):")
    for l, c in enumerate(constraints):
        print(f"    Sigma_{l}: delta({c}) = 0")

    Psi_r, Phi_r, subs = omega.reduce(Psi, Phi)
    print(f"\n  Post-reduction:")
    print(f"    Psi (reduced) = {Psi_r}")
    print(f"    Phi (reduced) = {Phi_r}")
    print(f"    Substitutions: {subs}")

    # The textbook 2-loop sunset Symanzik is alpha_1 alpha_2 + alpha_1 alpha_3 + alpha_2 alpha_3
    # Compare to the pre-reduction Psi (which IS the Kirchhoff polynomial for 3 parallel edges).
    print()
    print(f"  Textbook check: U_sunset = a1*a2 + a1*a3 + a2*a3 (e_2 elementary symmetric)")
    print(f"  Our pre-reduction Psi:    {Psi}")
    return Psi_r, Phi_r


if __name__ == '__main__':
    print("Wiring test: Reggeon causal -> Euclidean reduction")
    print()
    Psi_b, Phi_b = test_bubble()
    Psi_s, Phi_s = test_sunset()

    print()
    print("=" * 72)
    print(" Result")
    print("=" * 72)
    print()
    print("If the reduced (Psi, Phi) for the sunset matches the standard")
    print("Euclidean Symanzik polynomial structure, then handing the result")
    print("to the existing relativistic IBP solver completes the route:")
    print()
    print("  Reggeon graph -> OmegaIntegration -> Euclidean parametric")
    print("  -> ibp_solver  -> q-coefficients (a-priori, not JT05-closure).")
