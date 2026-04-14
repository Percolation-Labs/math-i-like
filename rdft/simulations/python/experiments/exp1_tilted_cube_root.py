"""
Experiment 1: tilted DSE for the cube-root CRN.

Take the cube-root CRN
    A -> 3A    rate r1 = 3
    2A -> A    rate r2 = 2
    2A -> 3A   rate r3 = 1
which gives phi_0(G) = 1 - G + 3 G^2 + G^3  (on C_3).

Tilt the FORWARD reaction A -> 3A by e^k.  This is the simplest current-type
tilt and is a standard starting point in stochastic thermodynamics.  It breaks
exactly one rate, so it shifts off C_3 except at k = 0.

New vertex contributions:
    A -> 3A   rate 3 e^k  =>  +3 e^k G^2
    (others unchanged)

Tilted kernel:
    phi_k(G) = 1 - G + (1 + 3 e^k) G^2 + G^3
              (the "+1" is the 2A->A contribution -2 + 2A->3A contribution +2
               = 0, plus the A->3A's 3 e^k)

Wait — let me rederive carefully from the vertex list.

Vertex (3,1) A->3A: +r1 G^(3+1-2) = +r1 G^2        -> +3 e^k G^2
Vertex (1,2) 2A->A: -r2 G^(1+2-2) = -r2 G          -> -2 G
Vertex (2,2) 2A->A: -r2 G^(2+2-2) = -r2 G^2        -> -2 G^2
Vertex (3,2) 2A->3A: +r3 G^(3+2-2) = +r3 G^3       -> +1 G^3
Vertex (2,2) 2A->3A: +2 r3 G^(2+2-2) = +2 r3 G^2   -> +2 G^2
Vertex (1,2) 2A->3A: +r3 G^(1+2-2) = +r3 G         -> +1 G

Totals:
  G^0: 1
  G^1: -r2 + r3 = -2 + 1 = -1
  G^2: +r1 e^k - r2 + 2 r3 = 3 e^k - 2 + 2 = 3 e^k
  G^3: +r3 = +1

So phi_k(G) = 1 - G + 3 e^k G^2 + G^3.  At k=0: 1 - G + 3 G^2 + G^3 (on C_3).
b^3 = 27 c^2  <=>  (3 e^k)^3 = 27  <=>  e^{3k} = 1  <=>  k = 0 only.

Prediction: lambda(k) = -log|z*(k)| has a non-analytic kink at k = 0, of
order (k)^{4/3} in a neighbourhood (cube-root-branch singularity of lambda).
"""

from __future__ import annotations
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pathlib import Path


# ----------------------------------------------------------
# Closed-form discriminant of F(G, z) = G - z phi_k(G), as cubic in G.
# phi_k(G) = 1 - G + 3 e^k G^2 + G^3 has coefficients (a,b,c)=(-1, 3e^k, 1).
# F = -c z G^3 - b z G^2 + (1 - a z) G - z = -z G^3 - 3e^k z G^2 + (1+z) G - z.
# Discriminant of A G^3 + B G^2 + C G + D, with A=-z, B=-3e^k z, C=1+z, D=-z, is:
#   Δ = 18 ABCD - 4 B^3 D + B^2 C^2 - 4 A C^3 - 27 A^2 D^2.
# After factoring out an overall z (the trivial root), the nontrivial roots
# satisfy a cubic in z whose coefficients we worked out by hand:
#   P(z; k) = α z^3 + β z^2 + γ z + δ
#   with α = 9 e^{2k} - 54 e^k - 108 e^{3k} - 23,
#        β = 18 e^{2k} - 54 e^k + 12,
#        γ = 9 e^{2k} + 12,
#        δ = 4.
# Use numpy to find the smallest-|z| root of P at each k.

def disc_poly_in_z(k_val: float) -> np.ndarray:
    """Coefficients of P(z; k) — leading first for numpy.roots."""
    e1 = np.exp(k_val); e2 = np.exp(2 * k_val); e3 = np.exp(3 * k_val)
    alpha = 9 * e2 - 54 * e1 - 108 * e3 - 23
    beta  = 18 * e2 - 54 * e1 + 12
    gamma = 9 * e2 + 12
    delta = 4.0
    return np.array([alpha, beta, gamma, delta])


def dominant_branch(k_val: float) -> complex:
    """Smallest |z| nonzero root of the discriminant polynomial."""
    coeffs = disc_poly_in_z(k_val)
    roots = np.roots(coeffs)
    nontrivial = [r for r in roots if abs(r) > 1e-10]
    if not nontrivial:
        return np.nan + 0j
    return min(nontrivial, key=lambda r: abs(r))


def compute_lambda(k_val: float) -> tuple[float, complex]:
    """lambda(k) = -log|z*(k)| and z*(k)."""
    z_star = dominant_branch(k_val)
    if not np.isfinite(abs(z_star)) or abs(z_star) == 0:
        return np.nan, z_star
    return -float(np.log(abs(z_star))), z_star


# ----------------------------------------------------------
#  Scan k
# ----------------------------------------------------------
k_vals = np.linspace(-0.8, 0.8, 81)
lam_vals = np.zeros_like(k_vals)
zstar_abs = np.zeros_like(k_vals)
zstar_arg = np.zeros_like(k_vals)

print('Scanning tilted DSE...')
for i, k in enumerate(k_vals):
    lam, zs = compute_lambda(k)
    lam_vals[i] = lam
    zstar_abs[i] = abs(zs)
    zstar_arg[i] = np.angle(zs)
    if i % 10 == 0:
        print(f'  k={k:+.3f}  |z*|={abs(zs):.4f}  arg={np.angle(zs):+.3f}  lambda={lam:+.4f}')

# First and second differences to look for kink
dlam = np.gradient(lam_vals, k_vals)
d2lam = np.gradient(dlam, k_vals)

# Find k=0 index
idx0 = np.argmin(np.abs(k_vals))
print(f'\n  at k=0:  lambda={lam_vals[idx0]:.6f}, z*={zstar_abs[idx0]:.6f}')

# Kink detection: does d2lam have a spike near k=0?
print(f'  second-derivative d²λ/dk² near k=0:')
for j in range(idx0 - 3, idx0 + 4):
    print(f'    k={k_vals[j]:+.3f}  d²λ={d2lam[j]:+.4f}')

# Test the cube-root scaling: lambda(k) - lambda(0) ~ (k - k_*)^{4/3} ?
# Near k=0, if the branch is cube-root, then lambda ~ |k|^{4/3}
k_small = k_vals[np.abs(k_vals) < 0.3]
lam_small = lam_vals[np.abs(k_vals) < 0.3]
delta_lam = lam_small - lam_vals[idx0]
# Fit log|delta_lam| vs log|k|
mask = np.abs(k_small) > 0.02
if mask.sum() >= 4:
    log_k = np.log(np.abs(k_small[mask]))
    log_dl = np.log(np.abs(delta_lam[mask]) + 1e-300)
    slope, _ = np.polyfit(log_k, log_dl, 1)
    print(f'\n  Scaling fit: |lambda(k) - lambda(0)| ~ |k|^alpha')
    print(f'     fitted alpha = {slope:.4f}  (cube-root branch predicts 4/3 ≈ 1.333)')
    print(f'                                  (square-root predicts 3/2 = 1.500)')

# ----------------------------------------------------------
#  Plot
# ----------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

axs[0].plot(k_vals, lam_vals, 'b-', lw=1.8)
axs[0].axvline(0, color='red', lw=1, ls='--', alpha=0.6)
axs[0].set_xlabel('tilt parameter $k$')
axs[0].set_ylabel(r'$\lambda(k) = -\log|z_\star(k)|$')
axs[0].set_title('SCGF: linear in $k$ except at $k{=}0$')
axs[0].grid(True, alpha=0.3)

axs[1].plot(k_vals, dlam, 'g-', lw=1.8)
axs[1].axvline(0, color='red', lw=1, ls='--', alpha=0.6)
axs[1].set_xlabel('tilt parameter $k$')
axs[1].set_ylabel(r"$d\lambda/dk$")
axs[1].set_title(r'First derivative (mean current)')
axs[1].grid(True, alpha=0.3)

axs[2].plot(k_vals, d2lam, 'r-', lw=1.8)
axs[2].axvline(0, color='red', lw=1, ls='--', alpha=0.6)
axs[2].set_xlabel('tilt parameter $k$')
axs[2].set_ylabel(r"$d^2\lambda/dk^2$")
axs[2].set_title(r'Second derivative: kink at $\mathcal{C}_3$ crossing?')
axs[2].grid(True, alpha=0.3)

fig.suptitle(r'Experiment 1: tilted cube-root CRN SCGF $\lambda(k)$',
             fontsize=13, y=1.02)
fig.tight_layout()
outdir = Path(__file__).parent.parent.parent.parent / 'paper' / 'cfac' / 'figures'
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / 'exp1_tilted_cube_root.pdf', bbox_inches='tight')
fig.savefig(outdir / 'exp1_tilted_cube_root.png', bbox_inches='tight', dpi=150)

np.savez(outdir / 'exp1_tilted_cube_root.npz',
         k_vals=k_vals, lam_vals=lam_vals,
         zstar_abs=zstar_abs, zstar_arg=zstar_arg,
         dlam=dlam, d2lam=d2lam)

print(f'\nSaved {outdir / "exp1_tilted_cube_root.pdf"}')
print(f'Saved {outdir / "exp1_tilted_cube_root.npz"}')
