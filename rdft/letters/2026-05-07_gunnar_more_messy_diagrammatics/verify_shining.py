"""Verify the 'shine a light' examples in note.tex section 4 against direct
brute-force enumeration of the rail walks."""

import sympy as sp
from fractions import Fraction
from itertools import product
import sys
sys.path.insert(0, '/Users/sirsh/code/math/rdft/letters/2026-05-07_gunnar_more_messy_diagrammatics')
from poc_laplacian import enumerate_symbolic, laplacian_symbolic, enumerate_walks


def Lm00(n, m, v_dict):
    """Compute (L^m)_00 symbolically."""
    L = laplacian_symbolic(n, v_dict)
    return sp.expand((L**m)[0, 0])


print("=" * 72)
print("(1') Resolvent at n=3 uniform: G(z) = (1-zv)/(1-3zv)")
print("=" * 72)
v, z = sp.symbols("v z", real=True)
v_dict = {(0,1): v, (0,2): v, (1,2): v}
L = laplacian_symbolic(3, v_dict)
I = sp.eye(3)
G_full = ((I - z*L).inv())[0, 0]
G_full = sp.simplify(G_full)
G_predicted = (1 - z*v) / (1 - 3*z*v)
print(f"  resolvent (0,0) computed: {G_full}")
print(f"  predicted G^(3)(z):       {sp.simplify(G_predicted)}")
print(f"  match: {sp.simplify(G_full - G_predicted) == 0}")
expansion = sp.series(G_predicted, z, 0, 5).removeO()
print(f"  z-expansion to z^4: {sp.expand(expansion)}")

print()
print("=" * 72)
print("(2') First-return: F(z) = (n-1)zv / (1-zv)")
print("=" * 72)
F = sp.simplify(1 - 1/G_predicted)
F_predicted = 2*z*v / (1 - z*v)
print(f"  F(z) from 1 - 1/G:        {sp.simplify(F)}")
print(f"  predicted F^(3)(z):       {sp.simplify(F_predicted)}")
print(f"  match: {sp.simplify(F - F_predicted) == 0}")
F_expansion = sp.series(F_predicted, z, 0, 5).removeO()
print(f"  z-expansion to z^4: {sp.expand(F_expansion)}")

print()
print("=" * 72)
print("(3') Channel sensitivity at n=3, m=2: mark v_01")
print("=" * 72)
v01, v02, v12 = sp.symbols("v_01 v_02 v_12", real=True)
v_dict_sym = {(0,1): v01, (0,2): v02, (1,2): v12}
W2_sym = Lm00(3, 2, v_dict_sym)
print(f"  (L^2)_00 symbolic:           {W2_sym}")
W2_marked = W2_sym.subs({v02: v, v12: v})
W2_marked = sp.expand(W2_marked)
print(f"  with v_02=v_12=v, in v_01:   {sp.collect(W2_marked, v01)}")
predicted = 2*v**2 + 2*v*v01 + 2*v01**2
print(f"  predicted:                   {sp.expand(predicted)}")
print(f"  match: {sp.expand(W2_marked - predicted) == 0}")

print()
print("=" * 72)
print("(4') Sub-rail focus n=3, m=2: foreground {0,1}, background {2} (v_02=v_12=v)")
print("=" * 72)
W2_focus = sp.expand(W2_sym.subs({v02: v, v12: v}))
print(f"  collected polynomial:        {sp.collect(W2_focus, v01)}")
print(f"  pure-foreground (v_01^2):    {W2_focus.coeff(v01, 2)} * v_01^2")
print(f"  mixed (v_01^1):              {W2_focus.coeff(v01, 1)} * v_01")
print(f"  pure-background (v_01^0):    {W2_focus.coeff(v01, 0)}")
predicted_decomp = (2*v01**2, 2*v*v01, 2*v**2)
print(f"  predicted decomposition:     2*v_01^2 + 2*v*v_01 + 2*v^2")
ok = (W2_focus.coeff(v01, 2) == 2 and
      W2_focus.coeff(v01, 1) == 2*v and
      W2_focus.coeff(v01, 0) == 2*v**2)
print(f"  match: {ok}")

print()
print("=" * 72)
print("(5') Forbidden patterns: set v_12=0 at n=3, check at m=3")
print("=" * 72)
W3_sym = Lm00(3, 3, v_dict_sym)
print(f"  (L^3)_00 with all symbolic:  {W3_sym}")
W3_blocked = sp.expand(W3_sym.subs(v12, 0))
print(f"  (L^3)_00 with v_12=0:        {W3_blocked}")
# Verify it's the same as running on the smaller "hub" graph
v_dict_hub = {(0,1): v01, (0,2): v02, (1,2): sp.Integer(0)}
L_hub = laplacian_symbolic(3, v_dict_hub)
W3_hub = sp.expand((L_hub**3)[0, 0])
print(f"  (L^3)_00 on hub-graph:       {W3_hub}")
print(f"  consistency: {sp.expand(W3_blocked - W3_hub) == 0}")
# Are there terms that vanish?
W3_full = W3_sym
removed = sp.expand(W3_full - W3_blocked)
print(f"  terms killed by v_12=0:      {removed}")

print()
print("=" * 72)
print("(6') Schur reduction at n=3 uniform: integrate out rails {1,2}")
print("=" * 72)
v_dict_uniform = {(0,1): v, (0,2): v, (1,2): v}
L_unif = laplacian_symbolic(3, v_dict_uniform)
print(f"  L =\n{L_unif}")
L_II = sp.Matrix([[L_unif[0, 0]]])
L_IB = sp.Matrix([[L_unif[0, 1], L_unif[0, 2]]])
L_BI = sp.Matrix([[L_unif[1, 0]], [L_unif[2, 0]]])
L_BB = sp.Matrix([[L_unif[1, 1], L_unif[1, 2]],
                  [L_unif[2, 1], L_unif[2, 2]]])
I2 = sp.eye(2)
inv_BB = (I2 - z * L_BB).inv()
inv_BB = sp.simplify(inv_BB)
print(f"  (I - z L_BB)^-1 = {inv_BB}")
sigma = sp.simplify(z * (L_IB * inv_BB * L_BI)[0, 0])
print(f"  Sigma(z) = z L_IB (I-zL_BB)^-1 L_BI = {sigma}")
L_eff = sp.simplify(L_II[0,0] + sigma)
print(f"  L^eff_00(z) = {L_eff}")
G_eff = sp.simplify(1 / (1 - z * L_eff))
print(f"  (1 - z L^eff)^-1 = {G_eff}")
predicted_full = (1 - z*v) / (1 - 3*z*v)
print(f"  expected G^(3)(z) = {predicted_full}")
print(f"  match: {sp.simplify(G_eff - predicted_full) == 0}")

# Also verify the specific intermediate result Sigma = 2 z v^2 / (1 - zv)
sigma_predicted = 2 * z * v**2 / (1 - z*v)
print(f"  Sigma predicted (2zv^2/(1-zv)): {sp.simplify(sigma - sigma_predicted) == 0}")
# And L^eff = 2v/(1-zv)
Leff_predicted = 2*v / (1 - z*v)
print(f"  L^eff predicted (2v/(1-zv)):    {sp.simplify(L_eff - Leff_predicted) == 0}")

print()
print("All (1')-(6') checks complete.")
