"""
Experiment 2: Schlögl II rate scan — map CRN parameters to Puiseux stratum.

Schlögl II: 2A <-> 3A at rates k_a (forward), k_b (reverse).  The DSE kernel is
    phi(G) = 1 + k_a G + (2 k_a - k_b) G^2 + (k_a - 2 k_b) G^3.

For each (k_a, k_b) we:
  (i)   compute the discriminant of F(G,z) = G - z phi(G) in G;
  (ii)  locate all nontrivial branch points z* in the complex z-plane;
  (iii) identify the multiplicity of the dominant (smallest |z*|) branch
        (this is k-1 for a 1/k Puiseux branch);
  (iv)  label the stratum C_k of the point (k_a, k_b).

Then we produce a 2D map of the stratum as a function of (k_a, k_b) and
overlay:
  - the C_3 crossing line b^3 = 27 c^2 (where the cube-root branch exists),
  - the C_4 surface (if the parameter range reaches it),
  - the locus of bistability (relevant for MFT quasipotential).
"""

from __future__ import annotations
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pathlib import Path


def schlogl_disc_coeffs(k_a: float, k_b: float) -> np.ndarray:
    """Coefficients of disc_G F(G,z) as polynomial in z, leading first.

    F = G - z phi(G), phi = 1 + a G + b G^2 + c G^3 with
        a = k_a, b = 2 k_a - k_b, c = k_a - 2 k_b.
    F as cubic in G: A G^3 + B G^2 + C G + D, with
        A = -c z, B = -b z, C = 1 - a z, D = -z.
    Discriminant of cubic = 18 A B C D - 4 B^3 D + B^2 C^2 - 4 A C^3 - 27 A^2 D^2.
    Computed once symbolically, evaluated here at (k_a, k_b) and returned as a
    polynomial in z.
    """
    a, b, c = k_a, 2 * k_a - k_b, k_a - 2 * k_b
    # Use sympy ONCE to get the coefficient formulas in z, then evaluate.
    z = sp.Symbol('z')
    A = -c * z; B = -b * z; C_ = 1 - a * z; D = -z
    disc = 18 * A * B * C_ * D - 4 * B**3 * D + B**2 * C_**2 - 4 * A * C_**3 - 27 * A**2 * D**2
    poly = sp.Poly(sp.expand(disc), z)
    coeffs = poly.all_coeffs()
    return np.array([float(c) for c in coeffs])


def root_multiplicity(roots: np.ndarray, z: complex, tol: float = 1e-3) -> int:
    """Count near-duplicates of `z` in `roots` (numerical multiplicity)."""
    return int(np.sum(np.abs(roots - z) < tol))


def dominant_branch_type(k_a: float, k_b: float) -> tuple[int, complex]:
    """Return (k_Puiseux, z_star) for Schlögl II at (k_a, k_b)."""
    coeffs = schlogl_disc_coeffs(k_a, k_b)
    # remove leading zeros
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-12:
        coeffs = coeffs[1:]
    if len(coeffs) < 2:
        return -1, np.nan + 0j
    roots = np.roots(coeffs)
    nontrivial = np.array([r for r in roots if abs(r) > 1e-9])
    if len(nontrivial) == 0:
        return -1, np.nan + 0j
    closest = min(nontrivial, key=lambda r: abs(r))
    mult = root_multiplicity(nontrivial, closest, tol=5e-3)
    k_puiseux = 1 + mult
    return k_puiseux, closest


# ----------------------------------------------------------
#  Scan (k_a, k_b) grid
# ----------------------------------------------------------
ka_grid = np.linspace(0.2, 4, 40)
kb_grid = np.linspace(0.2, 4, 40)
KA, KB = np.meshgrid(ka_grid, kb_grid)

K_puiseux = np.zeros_like(KA, dtype=int)
z_star_abs = np.zeros_like(KA)

print('Scanning Schlögl II rate space...')
for i in range(KA.shape[0]):
    for j in range(KA.shape[1]):
        k_a, k_b = KA[i, j], KB[i, j]
        k_p, zs = dominant_branch_type(k_a, k_b)
        K_puiseux[i, j] = k_p
        z_star_abs[i, j] = abs(zs) if not np.isnan(abs(zs)) else np.nan
    if i % 5 == 0:
        print(f'  row {i}/{KA.shape[0]}')

unique_k, counts = np.unique(K_puiseux, return_counts=True)
print('Stratum counts over scan grid:')
for k, n in zip(unique_k, counts):
    print(f'  C_{k}: {n} points')

# ----------------------------------------------------------
#  C_3 tuning curve in Schlögl coordinates
# ----------------------------------------------------------
# b = 2 k_a - k_b, c = k_a - 2 k_b.  C_3: b^3 = 27 c^2.
#   (2 k_a - k_b)^3 = 27 (k_a - 2 k_b)^2
# Solve for k_a given k_b (numerically).
from scipy.optimize import brentq
kb_curve = np.linspace(0.3, 3.5, 50)
ka_curve = []
for kb in kb_curve:
    def f(ka):
        return (2 * ka - kb) ** 3 - 27 * (ka - 2 * kb) ** 2
    # search in k_a in a reasonable range
    try:
        lo, hi = 0.1, 6.0
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi < 0:
            ka_curve.append(brentq(f, lo, hi))
        else:
            # scan for sign change
            k_scan = np.linspace(lo, hi, 100)
            v_scan = [f(k) for k in k_scan]
            signs = np.sign(v_scan)
            ka_root = np.nan
            for m in range(len(k_scan) - 1):
                if signs[m] * signs[m + 1] < 0:
                    ka_root = brentq(f, k_scan[m], k_scan[m + 1])
                    break
            ka_curve.append(ka_root)
    except Exception:
        ka_curve.append(np.nan)
ka_curve = np.array(ka_curve)

# ----------------------------------------------------------
#  Plot
# ----------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

ax = axs[0]
# Pseudocolor the Puiseux order
from matplotlib.colors import BoundaryNorm, ListedColormap
vals = sorted(set(K_puiseux.flatten()))
valid_vals = [v for v in vals if v > 0]
if not valid_vals:
    valid_vals = [2]
colour_list = ['#4a90c4', '#e07b00', '#c03030', '#60a060', '#a04040', '#404040']
n_colors = max(len(valid_vals), 1)
cmap = ListedColormap(colour_list[:n_colors])
norm = BoundaryNorm([v - 0.5 for v in valid_vals] + [valid_vals[-1] + 0.5], cmap.N)
im = ax.pcolormesh(ka_grid, kb_grid, K_puiseux, shading='auto', cmap=cmap, norm=norm)
cbar = fig.colorbar(im, ax=ax, ticks=valid_vals)
cbar.set_label(r'Puiseux order $k$  ($\tau = 1 + 1/k$)', fontsize=10)

# Overlay C_3 curve
mask_ok = ~np.isnan(ka_curve)
if mask_ok.any():
    ax.plot(ka_curve[mask_ok], kb_curve[mask_ok], 'k-', lw=2.4,
            label=r'$\mathcal{C}_3$ crossing: $(2k_a-k_b)^3 = 27(k_a-2k_b)^2$')

ax.set_xlabel(r'$k_a$ (Schlögl II forward rate)')
ax.set_ylabel(r'$k_b$ (Schlögl II reverse rate)')
ax.set_title('Schlögl II: Puiseux stratum of dominant branch')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.2)

ax = axs[1]
im = ax.pcolormesh(ka_grid, kb_grid, np.log10(z_star_abs + 1e-12),
                    shading='auto', cmap='magma_r')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$\log_{10}|z_\star|$ (radius of convergence)', fontsize=10)
if mask_ok.any():
    ax.plot(ka_curve[mask_ok], kb_curve[mask_ok], 'cyan', lw=2.4, alpha=0.9,
            label=r'$\mathcal{C}_3$ crossing')
ax.set_xlabel(r'$k_a$')
ax.set_ylabel(r'$k_b$')
ax.set_title(r'Schlögl II: dominant-branch radius $|z_\star(k_a, k_b)|$')
ax.legend(loc='lower right', fontsize=9)

fig.suptitle('Experiment 2: Schlögl II rate-space map', fontsize=13, y=1.02)
fig.tight_layout()
outdir = Path(__file__).parent.parent.parent.parent / 'paper' / 'cfac' / 'figures'
fig.savefig(outdir / 'exp2_schlogl_rate_scan.pdf', bbox_inches='tight')
fig.savefig(outdir / 'exp2_schlogl_rate_scan.png', bbox_inches='tight', dpi=150)

np.savez(outdir / 'exp2_schlogl_rate_scan.npz',
         ka_grid=ka_grid, kb_grid=kb_grid,
         K_puiseux=K_puiseux, z_star_abs=z_star_abs,
         kb_curve=kb_curve, ka_curve=ka_curve)

print(f'\nSaved {outdir / "exp2_schlogl_rate_scan.pdf"}')
print(f'Saved {outdir / "exp2_schlogl_rate_scan.npz"}')
