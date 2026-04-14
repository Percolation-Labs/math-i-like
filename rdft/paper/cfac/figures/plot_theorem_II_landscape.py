"""
Theorem II stratification landscape.

Shows the Puiseux-order / tau-exponent axis with the four regions
covered by CFAC:

  Theorem I (polynomial DSE):       dyadic ladder C_2, C_4, C_8, ...
                                     plus signed non-dyadic C_3, C_5, ...
  Theorem IIa (admissible):         C_2 generic + continuous stable-tree
                                     strip tau in (3/2, 2) for stable-index
                                     alpha in (1, 2]
  Theorem IIb (multivariate):       same strata but with per-species
                                     amplitude structure (Perron eigenvectors)
  Theorem IIc (log-corrected):      marginal tuning decorates a stratum
                                     with (log n)^beta factor

Named systems are placed at their (stratum, kernel-type) coordinates.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8.5,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))

# ------------------------------------------------------------------ #
# Horizontal strata = tau values, vertical zones = kernel-type
# ------------------------------------------------------------------ #
# y axis: tau (from 1 to 2)
# x axis: schematic "kernel type" bands
#   Band 1 (x in [0, 1]):  polynomial (Type I)
#   Band 2 (x in [1, 2]):  admissible (IIa)
#   Band 3 (x in [2, 3]):  multivariate (IIb)
#   Band 4 (x in [3, 4]):  log-corrected (IIc)

tau_min, tau_max = 0.95, 2.05
ax.set_ylim(tau_min, tau_max)
ax.set_xlim(-0.1, 4.2)

# Background bands for each Theorem region
band_colors = {
    'I':   '#e8eef5',
    'IIa': '#e9f4ea',
    'IIb': '#f4eae8',
    'IIc': '#f3ebf4',
}
band_labels = {
    'I':   'Theorem I\n(polynomial $\\phi$)',
    'IIa': 'Theorem IIa\n(admissible $\\phi$)',
    'IIb': 'Theorem IIb\n(multivariate $\\phi$)',
    'IIc': 'Theorem IIc\n(log-corrected)',
}

for i, key in enumerate(['I', 'IIa', 'IIb', 'IIc']):
    ax.axvspan(i, i + 1, color=band_colors[key], alpha=0.55, zorder=0)
    # Band title at top
    ax.text(i + 0.5, tau_max - 0.04, band_labels[key],
            ha='center', va='top', fontsize=9.5,
            color='#333333', fontweight='bold', zorder=5)

# ------------------------------------------------------------------ #
# Tau-stratum horizontal reference lines (the ladder)
# ------------------------------------------------------------------ #
strata = [
    (1.0,   r'$\mathcal{C}_\infty$ (Voter, $k\to\infty$)', 'grey', 0.4),
    (1.25,  r'$\mathcal{C}_4$  ($\tau = 5/4$)',           '#806020', 0.7),
    (4.0/3, r'$\mathcal{C}_3$  ($\tau = 4/3$)',           '#c03030', 0.9),
    (1.5,   r'$\mathcal{C}_2$  ($\tau = 3/2$, generic)',  '#2060a0', 0.9),
]

for tau, lbl, colour, alpha in strata:
    ax.axhline(tau, color=colour, lw=1.2, alpha=alpha,
                linestyle='-', zorder=1)
    ax.text(-0.04, tau, f'$\\tau\\!=\\!{tau:.3f}$',
            ha='right', va='center', fontsize=8.5,
            color=colour, zorder=5)

# ------------------------------------------------------------------ #
# Banderier-Drmota cap indicator
# ------------------------------------------------------------------ #
# N-algebraic systems (positive DSE) are capped at dyadic k in {2,4,8,...}
# Non-dyadic strata (C_3, C_5, ...) require SIGNED structure (conservation)
# Show this as a hatched band on the polynomial side
ax.add_patch(Rectangle((0, 4.0/3 - 0.015), 1, 0.03,
                        facecolor='none', edgecolor='#c03030',
                        hatch='///', lw=0, alpha=0.4, zorder=2))
ax.text(0.5, 4.0/3 + 0.055, 'Banderier–Drmota forbids\n$k=3$ in $\\mathbb{N}$-algebraic;\nsigned CRN needed',
        ha='center', va='bottom', fontsize=7.5, color='#c03030',
        fontstyle='italic', zorder=5)

# ------------------------------------------------------------------ #
# Stable-tree continuous strip (Theorem IIa Corollary)
# ------------------------------------------------------------------ #
# For alpha in (1, 2], tau = 1 + 1/alpha fills (3/2, 2) continuously.
# Shade this strip inside the IIa band.
alpha_vals = np.linspace(1.01, 2.0, 100)
tau_vals = 1.0 + 1.0 / alpha_vals
# Vertical gradient: fill the strip from tau=3/2 to tau=2 within IIa band
ax.fill_between([1.05, 1.95], 1.5, 2.0,
                 color='#6aa06a', alpha=0.25, zorder=2,
                 label='Stable-tree strip (Cor.~2)')
ax.text(1.5, 1.75, 'stable-tree\nregime\n$\\tau = 1 + 1/\\alpha$\n$\\alpha \\in (1, 2]$',
        ha='center', va='center', fontsize=8.5, color='#2a502a',
        fontstyle='italic', zorder=4)

# ------------------------------------------------------------------ #
# Log-correction decorations (Theorem IIc)
# ------------------------------------------------------------------ #
# Marginal tuning lives "ON" a stratum but decorates with (log n)^beta.
# Show as dashed tick marks on C_2 and C_3 in the IIc band.
for tau in [1.5, 4.0/3]:
    ax.plot([3.2, 3.8], [tau, tau], linestyle='--',
            color='#8040a0', lw=2.5, alpha=0.7, zorder=3)
ax.text(3.5, (1.5 + 4.0/3) / 2, '$(\\log n)^\\beta$\ncorrection',
        ha='center', va='center', fontsize=8.5, color='#5a2080',
        fontstyle='italic', zorder=4)

# ------------------------------------------------------------------ #
# Named systems (CRN-in-field + physical systems placed at coords)
# ------------------------------------------------------------------ #
#  (x, y, label, marker, size, color)
systems = [
    # Theorem I (polynomial)
    (0.25, 1.5,    'DP',                        'o',  90,  '#2060a0'),
    (0.50, 1.5,    'Schlögl II',                 's',  80,  '#e07b00'),
    (0.75, 1.5,    'pair coag.',                '^',  80,  '#2060a0'),
    (0.50, 4.0/3,  'Manna /\nC-DP',             '*',  220, '#c03030'),
    (0.25, 1.25,   'quartic CRN',               'D',  70,  '#806020'),
    # Theorem IIa (admissible)
    (1.20, 1.5,    'Cayley tree\n$\\phi{=}ze^G$',  'o', 100, '#2a8040'),
    (1.40, 1.5,    'Poisson\nbranching',         'h',  90,  '#2a8040'),
    (1.20, 1.7,    'stable trees\n$\\alpha{=}1.5$', 'p',  90,  '#408050'),
    (1.60, 1.83,   'stable trees\n$\\alpha{=}1.2$', 'p',  90,  '#408050'),
    # Theorem IIb (multivariate)
    (2.30, 1.5,    '2-type ant\ncolony',         'd',  100, '#a04020'),
    (2.60, 1.5,    'age-structured\nSIR',        'v',  90,  '#a04020'),
    # Theorem IIc (log-corrected)
    (3.20, 1.5,    '4D Ising\nsusceptibility',   'P',  100, '#5a2080'),
    (3.60, 1.5,    'Potts $q{=}4$',              'X',  100, '#5a2080'),
    (3.40, 4.0/3,  'KT transition\n(2D $XY$)',   '*',  130, '#5a2080'),
]

for x, y, label, marker, size, colour in systems:
    ax.scatter([x], [y], s=size, marker=marker,
                facecolors=colour, edgecolors='black', lw=0.8,
                zorder=6)
    ax.annotate(label, xy=(x, y),
                xytext=(5, 7), textcoords='offset points',
                fontsize=7.5, color='#222222',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.85),
                zorder=7)

# ------------------------------------------------------------------ #
# Axis labels & legend
# ------------------------------------------------------------------ #
ax.set_ylabel(r'cluster-size exponent $\tau$', fontsize=11)
ax.set_title(
    'CFAC stratification landscape with Theorem~II extensions\n'
    'Type I core (polynomial, dyadic-capped) + IIa/IIb/IIc extensions',
    fontsize=12, pad=12
)
ax.set_xticks([])  # x axis is categorical
ax.grid(False)

# ------------------------------------------------------------------ #
# Legend: marker types for the four theorem regions
# ------------------------------------------------------------------ #
legend_elems = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2060a0',
                markersize=9, label='Theorem I (polynomial core)', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='h', color='w', markerfacecolor='#2a8040',
                markersize=9, label='Theorem IIa (admissible)', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='#a04020',
                markersize=9, label='Theorem IIb (multivariate)', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#5a2080',
                markersize=9, label='Theorem IIc (log-corrected)', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#c03030',
                markersize=14, label='Manna/C-DP ($\\mathcal{C}_3$, signed)',
                markeredgecolor='k'),
    plt.Rectangle((0,0), 1, 1, facecolor='#6aa06a', alpha=0.4,
                   label='Stable-tree continuous strip (Cor.~2)'),
]
ax.legend(handles=legend_elems, loc='upper right', fontsize=8.5,
          framealpha=0.93, edgecolor='0.6', fancybox=True,
          bbox_to_anchor=(1.0, 1.12))

fig.tight_layout()
outdir = Path(__file__).parent
fig.savefig(outdir / 'theorem_II_landscape.pdf', bbox_inches='tight')
fig.savefig(outdir / 'theorem_II_landscape.png', bbox_inches='tight', dpi=160)
print(f"saved theorem_II_landscape.{{pdf,png}} to {outdir}")
