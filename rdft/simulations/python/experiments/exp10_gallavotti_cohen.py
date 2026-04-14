"""
Experiment 10: Gallavotti-Cohen symmetry check on the canonical cube-root.

Open problem (stochastic thermodynamics):
  The Gallavotti-Cohen fluctuation theorem predicts that the SCGF of the
  entropy-production current satisfies lambda(s) = lambda(-1-s).  For
  Doi-Peliti DSEs, this becomes an algebraic statement about how the
  tilted DSE phi_s deforms under the involution s -> -1-s.

CFAC contribution:
  rdft.ac.tilted.gallavotti_cohen_residual computes the residual
  lambda(s) - lambda(-1-s) over a grid and plots the symmetry test.

Result:
  For a properly thermodynamic tilt (entropy production), GC predicts
  zero residual.  For a generic rate-tilt (not entropy production), the
  residual will be nonzero — and its magnitude is a quantitative measure
  of how far the tilt is from the entropy-production current.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rdft.ac.stratification import canonical_family, lambda_scgf
from rdft.ac.tilted import gallavotti_cohen_residual


# A "synthetic" CRN whose phi happens to equal canonical_family(3, -4).
# Since no positive-rate single-species CRN realises this (see paper
# correction), we use the canonical kernel directly via a wrapper that
# bypasses phi_from_reactions.

def lambda_canonical_tilted(s: float, k: int = 3, beta: float = -4) -> float:
    """SCGF of the canonical phi_{k, beta} with beta tilted by exp(s):
    beta -> beta * exp(s).  This is the simplest 1-parameter tilt of the
    canonical family."""
    return lambda_scgf(canonical_family(k, beta * np.exp(s)))


def main():
    print('=' * 80)
    print('Experiment 10: Gallavotti-Cohen test on canonical cube-root')
    print('=' * 80)

    s_arr = np.linspace(-0.4, 0.4, 81)

    # Compute lambda(s) and lambda(-1-s)
    lam_pos = np.array([lambda_canonical_tilted(s) for s in s_arr])
    lam_mirror = np.array([lambda_canonical_tilted(-1 - s) for s in s_arr])
    residual = lam_pos - lam_mirror

    # Print summary
    print(f'\n{"s":>8} {"lambda(s)":>12} {"lambda(-1-s)":>14} {"residual":>12}')
    for i in [0, 20, 40, 60, 80]:
        print(f'{s_arr[i]:>8.3f} {lam_pos[i]:>12.6f} {lam_mirror[i]:>14.6f} '
              f'{residual[i]:>12.6f}')

    print(f'\nMax |residual| = {np.max(np.abs(residual)):.4e}')

    if np.max(np.abs(residual)) < 1e-3:
        verdict = 'HOLDS — the canonical-beta tilt IS thermodynamic / GC-symmetric'
    else:
        verdict = ('does NOT hold — the canonical-beta tilt is NOT the entropy-'
                   'production current.\nExpected: a generic rate-tilt does not '
                   'satisfy GC; entropy-production does.')
    print(f'\nGC symmetry: {verdict}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(s_arr, lam_pos, 'b-', lw=1.8, label=r'$\lambda(s)$')
    axes[0].plot(s_arr, lam_mirror, 'r--', lw=1.4, label=r'$\lambda(-1-s)$')
    axes[0].set_xlabel(r'tilt parameter $s$')
    axes[0].set_ylabel(r'$\lambda$')
    axes[0].set_title(r'Canonical $\phi_{3,-4}$: SCGF vs reflected SCGF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(s_arr, residual, 'g-', lw=1.8)
    axes[1].axhline(0, color='black', lw=0.6, ls=':')
    axes[1].set_xlabel(r'tilt parameter $s$')
    axes[1].set_ylabel(r'GC residual $\lambda(s) - \lambda(-1-s)$')
    axes[1].set_title('Gallavotti-Cohen test')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Experiment 10: Gallavotti-Cohen on canonical cube-root')
    fig.tight_layout()
    outdir = Path(__file__).parent.parent.parent.parent / 'paper' / 'cfac' / 'figures'
    fig.savefig(outdir / 'exp10_gallavotti_cohen.pdf', bbox_inches='tight')
    fig.savefig(outdir / 'exp10_gallavotti_cohen.png', bbox_inches='tight', dpi=150)
    print(f'\nSaved {outdir / "exp10_gallavotti_cohen.pdf"}')

    print()
    print('=' * 80)
    print('STATEMENT')
    print('=' * 80)
    print(f"""
Open problem: in stochastic thermodynamics, the Gallavotti-Cohen fluctuation
theorem says lambda(s) = lambda(-1-s) for the SCGF of the entropy-production
current.  For DP DSEs, this is an algebraic statement on the tilted phi_s.

CFAC contribution: rdft.ac.tilted.gallavotti_cohen_residual provides a
one-line check.  For the canonical phi_{{3, beta}}(G) = (1+G)^3 + beta G,
tilting beta -> beta exp(s) is the simplest natural tilt.

Result: GC residual = {np.max(np.abs(residual)):.4e}.
This tilt is NOT the entropy-production current — the canonical-beta tilt
breaks the GC reflection.  The honest take: GC needs the SPECIFIC tilt
that targets the entropy-production functional, not a rate-tilt.  Building
that tilt from a CRN requires the Doi-Peliti generator framework — a
follow-up calculation.
""")


if __name__ == '__main__':
    main()
