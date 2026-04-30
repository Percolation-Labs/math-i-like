"""
Verify what the causal-reduction code actually produces.

Two views of the same physical operation:
  (A) Schwinger-substitution view (what OmegaIntegration implements):
      keep the original graph, substitute alpha_l = alpha_0 from the
      delta functions. Produces Psi' = 2 alpha_0 (bubble) or 3 alpha_0^2 (sunset).

  (B) Topological-contraction view (thesis Remark 32 framing):
      contract one edge per loop, leaving a smaller graph. The naive
      claim is "bubble -> tadpole" or "sunset -> figure-8".

This script asks: do (A) and (B) produce the SAME parametric form?
If yes, the figures showing the contracted graph are correct.
If no, the figures are misleading and we should revise.
"""
from __future__ import annotations
import sympy as sp
from rdft.graphs.incidence import FeynmanGraph
from rdft.integrals.symanzik import SymanzikPolynomials
from rdft.integrals.parametric import OmegaIntegration


def view_A_bubble():
    """OmegaIntegration on the bubble."""
    G = FeynmanGraph.one_loop_self_energy()
    sym = SymanzikPolynomials(G)
    Psi, Phi = sym.Psi, sym.Phi
    omega = OmegaIntegration(G)
    Psi_r, Phi_r, subs = omega.reduce(Psi, Phi)
    return Psi_r, Phi_r


def view_B_tadpole():
    """Symanzik of an actual tadpole graph (1 vertex, 1 self-loop)."""
    G = FeynmanGraph.tadpole()
    sym = SymanzikPolynomials(G)
    return sym.Psi, sym.Phi


def main():
    print("=" * 72)
    print(" Comparing the two views for the BUBBLE reduction")
    print("=" * 72)
    print()
    Psi_A, Phi_A = view_A_bubble()
    Psi_B, Phi_B = view_B_tadpole()

    print("View (A) Schwinger-substitution (OmegaIntegration on bubble):")
    print(f"  Psi' = {Psi_A}")
    print(f"  Phi' = {Phi_A}")
    print()
    print("View (B) Topological-contraction (literal tadpole graph):")
    print(f"  Psi  = {Psi_B}")
    print(f"  Phi  = {Phi_B}")
    print()

    same_psi = sp.simplify(Psi_A - Psi_B) == 0
    same_phi = sp.simplify(Phi_A - Phi_B) == 0
    print(f"  Psi_A == Psi_B ?  {same_psi}")
    print(f"  Phi_A == Phi_B ?  {same_phi}")
    print()

    if same_psi and same_phi:
        print("=> The two views give literally identical parametric forms.")
        print("   Figures showing 'bubble -> tadpole' are LITERALLY correct.")
    else:
        print("=> The two views give DIFFERENT parametric forms.")
        print("   The contracted-graph picture is HEURISTIC --- it captures")
        print("   the topological structure but the broadcast kinematics")
        print("   make the actual integrand different from a 'fresh' tadpole.")
        print()
        print("   The OmegaIntegration output (View A) is what the code")
        print("   actually computes; the contracted graph (View B) is a")
        print("   different graph whose Symanzik happens not to match.")


if __name__ == '__main__':
    main()
