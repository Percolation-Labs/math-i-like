"""
Figure 1: (b, c) rate space of cubic DSE kernels with the cube-root locus
          C_3: b^3 = 27 c^2 and a curated set of named CRNs.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# ---- shaded universality regions -----
b_range = np.linspace(-6, 8, 400)
c_range = np.linspace(-3, 3, 400)

# N-algebraic cone (BD-allowed): b >= 0, c >= 0
ax.axhspan(0, 3, xmin=6 / 14, alpha=0, color='w')  # placeholder
ax.fill_betweenx([0, 3], 0, 8, color='#4a90c4', alpha=0.08, zorder=0)

# ---- cube-root curve C_3: b^3 = 27 c^2 -----
b_pos = np.linspace(0, 8, 800)
c_plus = b_pos ** 1.5 / (3 * np.sqrt(3))
c_minus = -c_plus

# fill between the two branches of C_3 with a very subtle red glow
ax.fill_between(b_pos, c_minus, c_plus, color='#c03030', alpha=0.05, zorder=1)
ax.plot(b_pos, c_plus, color='#c03030', lw=2.6, zorder=3,
        label=r'$\mathcal{C}_3$: $b^3 = 27\,c^2$  ($\tau = 4/3$)')
ax.plot(b_pos, c_minus, color='#c03030', lw=2.6, zorder=3)

# ---- DP axis (c = 0) -----
ax.axhline(0, color='#2060a0', lw=1.4, alpha=0.7, zorder=2,
           label=r'DP axis ($c=0$, $\tau = 3/2$)')

# ---- Schlögl II family curve -----
k_a = np.linspace(0.1, 4, 200)
sch_b = 2 * k_a - 1
sch_c = k_a - 2
ax.plot(sch_b, sch_c, '--', color='#e07b00', lw=2.0, alpha=0.9, zorder=4,
        label=r"Schlögl II family ($k_b=1$, $k_a$ varies)")

# ---- CRN points (deduplicated; one marker per distinct location) -----
crns = [
    # label, b, c, marker, colour
    (r'DP family' + '\n(DP critical, pair coag.,' + '\n stem-cell niche, ...)',
     -1.0,  0.0, 'o', '#2060a0'),
    (r'Triplet annihilation $3A\to 2A$',
     0.0, -1.0, 's', '#6040a0'),
    (r"Schlögl II A ($k_a{=}k_b{=}1$)",
     -1.0, -1.0, 'v', '#e07b00'),
    (r"Schlögl II B ($k_a{=}2$)",
     3.0,  0.0, 'D', '#a04020'),
]
for label, b_val, c_val, marker, colour in crns:
    ax.scatter([b_val], [c_val], s=160, marker=marker,
               facecolors=colour, edgecolors='black', lw=1.1,
               zorder=6, label=label)

# ---- highlighted points -----
# Cube-root CRN (star)
ax.scatter([3], [1], s=420, marker='*', color='#c03030',
           edgecolors='black', lw=1.6, zorder=7,
           label=r'cube-root CRN (on $\mathcal{C}_3$)')

# Schlögl II ∩ C_3 (annotated separately)
def sch_cross(k):
    return (2 * k - 1) ** 3 - 27 * (k - 2) ** 2
k_cross = brentq(sch_cross, 1.4, 1.6)
bx, cx = 2 * k_cross - 1, k_cross - 2
ax.scatter([bx], [cx], s=220, marker='o',
           facecolors='#ffe060', edgecolors='#c03030', lw=2.2,
           zorder=7, label=rf"Schlögl II $\cap\,\mathcal{{C}}_3$  ($k_a\!\approx\!{k_cross:.3f}$)")

# ---- annotations ---------------------------------------------------------
ax.text(7, 2.55,
        r'$\mathbb{N}$-algebraic cone' + '\n'
        + '(positive combinatorics;\n'
        + r'Banderier–Drmota forbids $\alpha{=}1/3$)',
        ha='center', fontsize=8.8, color='#2060a0', alpha=0.9,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor='#2060a0', alpha=0.9, lw=0.7))

ax.annotate('cube-root CRN\n' + r'$\tau = 4/3$',
            xy=(3, 1), xytext=(4.8, 1.7),
            ha='center', fontsize=10.5, color='#c03030', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#c03030', lw=1.3,
                            connectionstyle='arc3,rad=-0.2'))

ax.annotate(fr"Schlögl II crosses $\mathcal{{C}}_3$" + '\n' +
            f'at $k_a/k_b\\approx{k_cross:.3f}$',
            xy=(bx, cx), xytext=(bx - 2.3, cx - 1.5),
            ha='center', fontsize=9.5, color='#c03030',
            arrowprops=dict(arrowstyle='->', color='#c03030', lw=1.2,
                            connectionstyle='arc3,rad=0.15'))

ax.annotate('DP universality\n' + r'($\tau = 3/2$)',
            xy=(-1, 0), xytext=(-4.2, 1.0),
            ha='center', fontsize=10, color='#2060a0',
            arrowprops=dict(arrowstyle='->', color='#2060a0', lw=1.2))

# ---- axes & decoration -----
ax.axvline(0, color='grey', lw=0.5, alpha=0.5, zorder=1)
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, zorder=0)
ax.set_xlabel(r'$b$   (coefficient of $G^{\,2}$ in DSE kernel $\phi$)')
ax.set_ylabel(r'$c$   (coefficient of $G^{\,3}$ in DSE kernel $\phi$)')
ax.set_title('CRN universality classes as positions in DSE rate space\n'
             r'(cubic slice; linear coefficient $a$ suppressed, $\mathcal{C}_3$ is $a$-independent)',
             pad=12)
ax.set_xlim(-6, 8)
ax.set_ylim(-3, 3)

# legend inside, compact, in the bottom-left corner where space is free
leg = ax.legend(loc='lower left', fontsize=9, framealpha=0.93,
                edgecolor='0.7', fancybox=True)
leg.get_frame().set_linewidth(0.6)

fig.tight_layout()
outdir = Path(__file__).parent
fig.savefig(outdir / 'algebraic_curve_rate_space.pdf', bbox_inches='tight')
fig.savefig(outdir / 'algebraic_curve_rate_space.png', bbox_inches='tight', dpi=160)
print('saved algebraic_curve_rate_space.{pdf,png}')
print(f"Schloegl II crosses C_3 at k_a = {k_cross:.6f}, (b, c) = ({bx:.4f}, {cx:.4f})")
