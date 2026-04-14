"""
Figure 2: CFAC universality surface.  Four panels.

  (A) 3D rate space with the cube-root surface C_3 and CRN trajectories.
  (B) The tau ladder tau_k = 1 + 1/k across k, with Banderier-Drmota status.
  (C) Canonical-family phase diagram (k, |beta|) -> universality class.
  (D) Radius-of-convergence surface |z*(b, c)| at a = -1; C_3 sits on the ridge.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from scipy.optimize import brentq


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'legend.fontsize': 9,
    'legend.framealpha': 0.95,
})


# ============================================================
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30,
                       left=0.05, right=0.95, top=0.92, bottom=0.05)

# ============================================================
# Panel A: 3D rate space with C_3
# ============================================================
axA = fig.add_subplot(gs[0, 0], projection='3d')
a_range = np.linspace(-5, 5, 20)
b_range = np.linspace(0.05, 5, 40)
A1, B1 = np.meshgrid(a_range, b_range)
C_plus = B1 ** 1.5 / (3 * np.sqrt(3))
C_minus = -C_plus

axA.plot_surface(A1, B1, C_plus, alpha=0.55, color='#d94848',
                 edgecolor='none', linewidth=0, antialiased=True)
axA.plot_surface(A1, B1, C_minus, alpha=0.55, color='#d94848',
                 edgecolor='none', linewidth=0, antialiased=True)

# Rim outline
for a_ in [-5, 5]:
    axA.plot(np.full_like(b_range, a_), b_range, b_range ** 1.5 / (3 * np.sqrt(3)),
             '-', color='#c03030', lw=1.8, alpha=0.9)
    axA.plot(np.full_like(b_range, a_), b_range, -b_range ** 1.5 / (3 * np.sqrt(3)),
             '-', color='#c03030', lw=1.8, alpha=0.9)

# DP family (c = 0)
a_DP = np.linspace(-5, 5, 40)
axA.plot(a_DP, np.full_like(a_DP, 1.0), np.full_like(a_DP, 0), '-',
         color='#1f6faa', lw=2.8, label='DP family ($c=0$)')

# Schlögl II family
k_a = np.linspace(0.3, 3.5, 60)
axA.plot(k_a, 2 * k_a - 1, k_a - 2, '--', color='#e07b00', lw=2.2,
         label=r"Schlögl II family")

# Cube-root CRN
axA.scatter([-1], [3], [1], s=250, marker='*', color='#c03030',
            edgecolor='black', linewidth=1.4, zorder=10,
            label='cube-root CRN')

# Schlögl II ∩ C_3
axA.scatter([1.4755], [1.9511], [-0.5245], s=160, marker='o',
            facecolors='#ffe060', edgecolor='#c03030', linewidth=2,
            zorder=10, label=r'Schlögl II $\cap\,\mathcal{C}_3$')

axA.set_xlabel(r'$a$', fontsize=11, labelpad=4)
axA.set_ylabel(r'$b$', fontsize=11, labelpad=4)
axA.set_zlabel(r'$c$', fontsize=11, labelpad=4)
axA.set_title(r'(A) Rate space $(a,b,c)$ of cubic DSE kernel' + '\n'
              + r'$\mathcal{C}_3:\; b^3 = 27\, c^2$ (red) separates universality classes',
              pad=6)
axA.view_init(elev=20, azim=-68)
axA.legend(loc='upper left', fontsize=8, framealpha=0.9)
# tighten
axA.xaxis.pane.set_alpha(0.06)
axA.yaxis.pane.set_alpha(0.06)
axA.zaxis.pane.set_alpha(0.06)


# ============================================================
# Panel B: tau ladder
# ============================================================
axB = fig.add_subplot(gs[0, 1])

ks = np.arange(2, 13)
taus = 1 + 1 / ks

# BD status: 1/k is dyadic iff k is a power of 2
bd_allowed = [(k & (k - 1)) == 0 for k in ks]

# Plot as descending ladder
for i, (k, t, allowed) in enumerate(zip(ks, taus, bd_allowed)):
    color = '#4a90c4' if allowed else '#c03030'
    marker = 'o' if allowed else 'X'
    axB.plot([k], [t], marker=marker, markersize=14, color=color,
             markeredgecolor='black', markeredgewidth=1.1, zorder=5)
    # label
    off_y = 0.022 if allowed else -0.035
    if k == 2:
        label = r'$\tau_2 = 3/2$' + '\n(DP, generic)'
    else:
        label = fr'$\tau_{{{k}}} = {sp.Rational(k+1, k)}$'
    axB.annotate(label, xy=(k, t), xytext=(k, t + off_y),
                 ha='center', va='bottom' if allowed else 'top',
                 fontsize=8.5, color=color)

# connect DP to the ladder with a thin line
axB.plot(ks, taus, '--', color='gray', lw=0.8, alpha=0.6, zorder=2)

# horizontal reference at τ=1
axB.axhline(1, color='grey', lw=0.6, alpha=0.6, ls=':')
axB.text(12, 1.02, r'$\tau \to 1$ (asymptotic)', fontsize=8.5, color='grey',
         ha='right', style='italic')

# shaded regions for universality class origin
# Allowed = dyadic, accessible to positive combinatorics
# Forbidden = non-dyadic, exclusive to CFAC
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', markersize=10, color='#4a90c4',
           markeredgecolor='black', lw=0, label=r'$1/k$ dyadic — allowed to $\mathbb{N}$-algebraic'),
    Line2D([0], [0], marker='X', markersize=10, color='#c03030',
           markeredgecolor='black', lw=0, label=r'$1/k$ non-dyadic — forbidden to positive systems,'
                                                 '\n        '
                                                 r'reachable only by signed Doi–Peliti CRNs'),
]
axB.legend(handles=legend_handles, loc='upper right', fontsize=8.5,
           framealpha=0.95, title=r'$\tau_k = 1 + 1/k$ stratification (Thm. A.2)')

axB.set_xlabel(r'Puiseux order $k$')
axB.set_ylabel(r'transfer exponent $\tau_k = 1 + 1/k$')
axB.set_title(r'(B) Universality ladder: an infinite family of exponents, every odd $k$ forbidden',
              pad=6)
axB.set_xticks(ks)
axB.set_xlim(1.5, 12.8)
axB.set_ylim(1.02, 1.58)
axB.grid(True, alpha=0.25)


# ============================================================
# Panel C: phase diagram (k, |beta|)
# ============================================================
axC = fig.add_subplot(gs[1, 0])

# Dominance thresholds (from separate scan, reported in the appendix)
thresholds = {3: 4, 4: 5, 5: 8, 6: 8, 7: 10, 8: 12}
ks_pd = list(thresholds.keys())

# Plot: for each k, the coloured "1/k universality" region (|beta| >= threshold)
# and the grey "DP" region (|beta| < threshold).
for i, k in enumerate(ks_pd):
    thr = thresholds[k]
    colour = '#c03030' if (k & (k - 1)) != 0 else '#4a90c4'
    # upper box
    axC.fill_between([i - 0.4, i + 0.4], thr, 25, color=colour, alpha=0.6, zorder=3)
    # lower (DP) box
    axC.fill_between([i - 0.4, i + 0.4], 2, thr, color='#e0e0e0', alpha=0.8, zorder=2)
    # threshold line
    axC.plot([i - 0.4, i + 0.4], [thr, thr], 'k-', lw=1.2, zorder=4)
    # threshold label
    axC.text(i, thr - 0.6, rf'$|\beta^\star|\!=\!{thr}$', ha='center',
             fontsize=8.3, color='0.3')
    # universality label up top
    axC.text(i, 22, rf'$\tau_{{{k}}}{{=}}{sp.Rational(k+1, k)}$',
             ha='center', fontsize=9.2, color='white', fontweight='bold')
    # DP label
    axC.text(i, (2 + thr) / 2, 'DP', ha='center', fontsize=9,
             color='0.4')

axC.set_xticks(range(len(ks_pd)))
axC.set_xticklabels([str(k) for k in ks_pd])
axC.set_xlim(-0.5, len(ks_pd) - 0.5)
axC.set_ylim(2, 25)
axC.set_xlabel(r'Puiseux order $k$')
axC.set_ylabel(r'$|\beta|$ in canonical family $\phi_{k,\beta} = (1+G)^k + \beta G$')
axC.set_title(r'(C) Phase diagram: red = Banderier-Drmota–forbidden stratum reached; '
              r'blue = allowed', pad=6)
axC.grid(True, axis='y', alpha=0.25)


# ============================================================
# Panel D: |z*| surface with C_3 ridge at a = -1
# ============================================================
axD = fig.add_subplot(gs[1, 1], projection='3d')

bgrid = np.linspace(-1.5, 5.5, 120)
cgrid = np.linspace(-2.2, 2.2, 120)
Bg, Cg = np.meshgrid(bgrid, cgrid)

a_fixed = -1.0
Z = np.zeros_like(Bg)
for i in range(Bg.shape[0]):
    for j in range(Bg.shape[1]):
        b_, c_ = Bg[i, j], Cg[i, j]
        # disc(G) of F = G - z phi(G) for phi = 1 + aG + bG^2 + cG^3
        def disc_at(zv):
            p3 = -c_ * zv; p2 = -b_ * zv; p1 = 1 - a_fixed * zv; p0 = -zv
            return (18 * p3 * p2 * p1 * p0 - 4 * p2**3 * p0 + p2**2 * p1**2
                    - 4 * p3 * p1**3 - 27 * p3**2 * p0**2)
        best_z = np.inf
        try:
            zs = np.concatenate([np.linspace(-1.2, -0.01, 60), np.linspace(0.01, 1.2, 60)])
            vals = np.array([disc_at(zv) for zv in zs])
            for m in range(len(zs) - 1):
                if vals[m] * vals[m + 1] < 0:
                    try:
                        r = brentq(disc_at, zs[m], zs[m + 1])
                        if abs(r) < best_z and abs(r) > 1e-6:
                            best_z = abs(r)
                    except ValueError:
                        pass
        except Exception:
            pass
        Z[i, j] = best_z if best_z < np.inf else np.nan

# Smooth and clip for plotting
from scipy.ndimage import gaussian_filter
Z_plot = np.clip(Z, 0.02, 0.55)
Z_plot = gaussian_filter(Z_plot, sigma=1.0, mode='nearest')
surf = axD.plot_surface(Bg, Cg, Z_plot, cmap='viridis', alpha=0.88,
                        linewidth=0, antialiased=True, rcount=80, ccount=80,
                        vmin=0.05, vmax=0.5, edgecolor='none')

# Draw C_3 curve lifted onto the surface
b_pos = np.linspace(0.1, 5.3, 80)
c_plus = b_pos ** 1.5 / (3 * np.sqrt(3))
c_minus = -c_plus

def z_on_C3(a_val, b_val, c_val):
    if c_val == 0: return np.nan
    G_star = -b_val / (3 * c_val)
    phi_star = 1 + a_val * G_star + b_val * G_star ** 2 + c_val * G_star ** 3
    return abs(G_star / phi_star) if phi_star != 0 else np.nan

z_plus = np.array([z_on_C3(-1, b_, c_) for b_, c_ in zip(b_pos, c_plus)])
z_minus = np.array([z_on_C3(-1, b_, c_) for b_, c_ in zip(b_pos, c_minus)])

axD.plot(b_pos, c_plus, np.clip(z_plus, 0.02, 0.9), color='#c03030', lw=3,
         label=r'$\mathcal{C}_3\cap\{a=-1\}$')
axD.plot(b_pos, c_minus, np.clip(z_minus, 0.02, 0.9), color='#c03030', lw=3)
axD.scatter([3], [1], [0.25], s=260, marker='*', color='#c03030',
            edgecolor='black', linewidth=1.6, zorder=10)

axD.set_xlabel(r'$b$', fontsize=11, labelpad=3)
axD.set_ylabel(r'$c$', fontsize=11, labelpad=3)
axD.set_zlabel(r'$|z_\star|$', fontsize=11, labelpad=3)
axD.set_title(r'(D) Radius-of-convergence surface $|z_\star(b,c)|$ at $a\!=\!-1$' + '\n'
              + r'$\mathcal{C}_3$ sits on the ridge (cube-root cusp)',
              pad=8)
axD.view_init(elev=28, azim=-50)
axD.xaxis.pane.set_alpha(0.04)
axD.yaxis.pane.set_alpha(0.04)
axD.zaxis.pane.set_alpha(0.04)
cbar = fig.colorbar(surf, ax=axD, pad=0.10, shrink=0.6, aspect=15)
cbar.set_label(r'$|z_\star|$', fontsize=9)
axD.legend(loc='upper left', fontsize=9, framealpha=0.9)

fig.suptitle(r'CFAC universality stratification: $\mathcal{C}_k$ $(k\geq 2)$ at a glance',
             fontsize=14.5, y=0.975)

outdir = Path(__file__).parent
fig.savefig(outdir / 'universality_surface.pdf', bbox_inches='tight')
fig.savefig(outdir / 'universality_surface.png', bbox_inches='tight', dpi=160)
print('saved universality_surface.{pdf,png}')
