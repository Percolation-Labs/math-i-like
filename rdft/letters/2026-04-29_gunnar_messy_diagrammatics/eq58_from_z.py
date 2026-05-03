"""Derive JT05 Eq.(58) RG functions from CFAC Z-factors via the Tauber relation.

This closes the gap that two_loop.py currently treats as 'input'.  We take only:
  - the 1-loop algebra factors a_X^(1) = {1/4, 1/8, 1/2, 2}     [CFAC C4]
  - the double-pole rationals from the Hopf antipode             [CFAC C5]
  - the 2-loop simple-pole rationals from the IBP closure        [CFAC C8]

and derive the four RG functions {gamma, zeta, kappa, beta} of JT05 Eq.(58),
verifying against the published series.

Tauber/MSbar relations (Tauber, *Critical Dynamics* 2014, Eq. 4.34;
Vasil'ev "Field Theoretic Renormalization Group"):

  For Z_X(u, eps) = 1 + (a_X/eps) u + (D_X/eps^2 + S_X/eps) u^2 + O(u^3),
  with simple-pole residue of ln(Z_X) being  sigma_X(u) = a_X u + S_X u^2 + ...,

      gamma_X(u) = -u * d/du [sigma_X(u)]   (Tauber)

  JT05's RG functions are then specific *combinations* of the gamma_X
  determined by the bare-vs-renormalised relations of (lambda, tau, u):

      gamma(u) = gamma_psi(u)                              (field anomalous dim)
      zeta(u)  = gamma_psi(u) - gamma_lambda(u)            (lambda runs through Z_psi/Z_lambda)
      kappa(u) = gamma_lambda(u) - gamma_tau(u)            (tau runs through Z_tau and lambda)
      beta(u)  = -eps*u + u^2 * d/du[sigma_psi - 2*sigma_psi]   (1-loop only here;
                                                                 2-loop needs MSbar pole closure)
"""
import sympy as sp

u, eps = sp.symbols('u varepsilon')
L = sp.log(sp.Rational(4, 3))

# --- CFAC inputs ---
# 1-loop algebra factors (Lagrange + Reggeon vertex algebra; gribov.tex C4):
a = {'psi': sp.Rational(1, 4),
     'lambda': sp.Rational(1, 8),
     'tau': sp.Rational(1, 2),
     'u': sp.Rational(2, 1)}

# 2-loop simple-pole residues Z_X^(2,1) (from ibp_coefficients.py table; gribov.tex C8):
S = {'psi':    sp.Rational(-3, 32)  + sp.Rational(9, 64)  * L,
     'lambda': sp.Rational(-31, 512) + sp.Rational(35, 256) * L,
     'tau':    sp.Rational(-5, 32),
     'u':      sp.Rational(-7, 8)}

# Simple-pole residue of ln(Z_X) to 2-loop:  sigma_X(u) = a_X u + S_X u^2.
sigma = {X: a[X] * u + S[X] * u**2 for X in a}

# Tauber relation: gamma_X(u) = -u * d/du sigma_X(u)
gamma_X = {X: sp.expand(-u * sp.diff(sigma[X], u)) for X in a}

# JT05 RG-function combinations
gamma_JT05_form = gamma_X['psi']
zeta_JT05_form  = sp.expand(gamma_X['psi'] - gamma_X['lambda'])
kappa_JT05_form = sp.expand(gamma_X['lambda'] - gamma_X['tau'])

# --- JT05 published Eq.(58) ---
gamma_JT05 = -u/4 + sp.Rational(3, 32) * u**2 * (2 - 3*L)
zeta_JT05  = -u/8 + (u**2 / 256) * (17 - 2*L)
kappa_JT05 = sp.Rational(3, 8) * u + (7 * u**2 / 256) * (-7 - 10*L)

beta_JT05 = u * (-eps + sp.Rational(3, 2) * u
                 + (u**2 / 128) * (-169 - 106 * L))

# --- Verify ---
def check(name, ours, theirs):
    diff = sp.expand(ours - theirs)
    status = 'MATCH' if diff == 0 else f'DIFFERS by {diff}'
    print(f"  {name}: ours = {ours}")
    print(f"    JT05 = {sp.expand(theirs)}")
    print(f"    {status}\n")
    return diff == 0

print("=" * 72)
print(" Deriving JT05 Eq.(58) from CFAC Z-factors via Tauber relation")
print("=" * 72)
print()
print("Step 1: gamma_X(u) = -u * d/du sigma_X(u)  [Tauber, MSbar]")
for X in a:
    print(f"  gamma_{X}(u) = {gamma_X[X]}")
print()
print("Step 2: JT05 combinations")

ok_gamma = check("gamma   = gamma_psi",                    gamma_JT05_form, gamma_JT05)
ok_zeta  = check("zeta    = gamma_psi - gamma_lambda",     zeta_JT05_form,  zeta_JT05)
ok_kappa = check("kappa   = gamma_lambda - gamma_tau",     kappa_JT05_form, kappa_JT05)

# Beta function: 1-loop part derived; 2-loop has MSbar pole-cancellation subtlety
# beyond the simple Tauber relation.  Show what we have.
print("Step 3: beta(u) -- 1-loop derived from CFAC, 2-loop closure pending")
beta_1loop = -eps * u + (a['u'] - 2 * a['psi']) * u**2  # 1-loop u_0 = mu^eps u Z_u/Z_psi^2
print(f"  beta(u) [1-loop, CFAC] = {beta_1loop}")
print(f"  beta_1 = a_u - 2*a_psi = {a['u'] - 2*a['psi']}  (= JT05 1-loop coefficient 3/2: "
      f"{(a['u'] - 2*a['psi']) == sp.Rational(3, 2)})")
print()
print("  NOTE on 2-loop beta:")
print("  The MSbar 2-loop beta-function coefficient is NOT simply 2 * (Z_u^(2,1) - 2*Z_psi^(2,1)).")
print("  Closing it from CFAC requires the standard pole-cancellation identity")
print("  in the 2-loop bare-vs-renormalised expansion (Vasil'ev Eq. 1.114).")
print("  This is bookkeeping algebra that we have not yet written out; once")
print("  closed, beta(u) in JT05 Eq.(58d) follows exactly from the same Z-factor")
print("  data the gamma/zeta/kappa derivations above use.")

print()
print("=" * 72)
print(f" RESULT: 3 of 4 RG functions in JT05 Eq.(58) DERIVED from CFAC Z-factors.")
print(f"   gamma:  {'OK' if ok_gamma else 'FAIL'}")
print(f"   zeta:   {'OK' if ok_zeta else 'FAIL'}")
print(f"   kappa:  {'OK' if ok_kappa else 'FAIL'}")
print(f"   beta:   1-loop OK; 2-loop coefficient pending MSbar pole-closure algebra.")
print("=" * 72)

assert ok_gamma and ok_zeta and ok_kappa, "RG derivation failed"
