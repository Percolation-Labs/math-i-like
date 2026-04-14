"""
Experiment 13: GC fluctuation theorem via the proper entropy-production tilt.

Open problem (continuation of Exp 10):
  Build the entropy-production current sigma from the DP generator, tilt
  the DSE accordingly, and verify the Gallavotti-Cohen symmetry
  lambda_sigma(s) = lambda_sigma(-1 - s) for the resulting SCGF.

CFAC contribution:
  rdft.ac.tilted.entropy_production_tilt + gallavotti_cohen_test now
  implement the canonical thermodynamic tilt for paired forward/reverse
  reactions in any single-species CRN.

Test substrate:
  Schlögl-II forward/reverse pair: 2A <-> 3A at rates (k_a, k_b).
  Affinity A = ln(k_a / k_b).  GC predicts lambda(s) = lambda(-1-s) exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rdft.ac.tilted import gallavotti_cohen_test


def main():
    print('=' * 80)
    print('Experiment 13: Gallavotti-Cohen via the proper entropy-production tilt')
    print('=' * 80)

    cases = [
        ('Schlögl II symmetric (k_a=k_b=1)', [(2, 3, 1.0), (3, 2, 1.0)], [(0, 1)]),
        ('Schlögl II asymmetric (k_a=2, k_b=1)', [(2, 3, 2.0), (3, 2, 1.0)], [(0, 1)]),
        ('Schlögl II strongly asymmetric (k_a=4, k_b=1)', [(2, 3, 4.0), (3, 2, 1.0)], [(0, 1)]),
        ('Birth-death (A->2A:1, 2A->A:1)', [(1, 2, 1.0), (2, 1, 1.0)], [(0, 1)]),
        ('Birth-death asymmetric (A->2A:2, 2A->A:1)', [(1, 2, 2.0), (2, 1, 1.0)], [(0, 1)]),
    ]

    s_arr = np.linspace(-1.4, 0.4, 51)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for desc, reactions, pairs in cases:
        s, lam_pos, lam_mirror, residual = gallavotti_cohen_test(reactions, pairs, s_arr)

        # Filter out NaN
        ok = np.isfinite(residual)
        if not ok.any():
            print(f'\n{desc}: all NaN (likely no nontrivial branch)')
            continue

        max_res = np.max(np.abs(residual[ok]))
        print(f'\n{desc}')
        print(f'  max |lambda(s) - lambda(-1-s)| = {max_res:.4e}')
        verdict = '✓ GC PASSES' if max_res < 1e-3 else '✗ GC fails'
        print(f'  {verdict}')

        axes[0].plot(s[ok], lam_pos[ok], lw=1.5, label=f'{desc[:30]}')
        axes[1].plot(s[ok], residual[ok], lw=1.5, label=f'{desc[:30]} (max={max_res:.2e})')

    axes[0].set_ylabel(r'$\lambda_\sigma(s)$')
    axes[0].set_title(r'SCGF of entropy-production current')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color='black', lw=0.6, ls=':')
    axes[1].set_xlabel(r'$s$')
    axes[1].set_ylabel(r'$\lambda(s) - \lambda(-1-s)$ (GC residual)')
    axes[1].set_title('GC residual: zero means GC holds')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Experiment 13: entropy-production tilt + Gallavotti-Cohen')
    fig.tight_layout()
    outdir = Path(__file__).parent.parent.parent.parent / 'paper' / 'cfac' / 'figures'
    fig.savefig(outdir / 'exp13_entropy_production_tilt.pdf', bbox_inches='tight')
    fig.savefig(outdir / 'exp13_entropy_production_tilt.png', bbox_inches='tight', dpi=150)
    print(f'\nSaved {outdir / "exp13_entropy_production_tilt.pdf"}')

    print()
    print('=' * 80)
    print('STATEMENT')
    print('=' * 80)
    print("""
Open problem (Exp 10 continuation):
  Verify Gallavotti-Cohen lambda_sigma(s) = lambda_sigma(-1-s) for the
  SCGF of the entropy-production current in CFAC DSEs.

CFAC contribution:
  Library function entropy_production_tilt(reactions, pairs, s) constructs
  the proper tilt: forward rate × exp(s*A), reverse × exp(-s*A) with
  A = ln(rate_fwd / rate_rev) the affinity.  gallavotti_cohen_test
  computes the residual.

Result (honest negative):
  Only the SYMMETRIC birth-death case (affinity A=0) passes GC, and it
  passes trivially because the tilt does nothing.  All asymmetric cases
  give nonzero residual (0.38--3.49).  The simple rate-tilt
  rate_fwd -> rate_fwd * exp(s*A), rate_rev -> rate_rev * exp(-s*A)
  with A = ln(rate_fwd/rate_rev) is NOT the canonical entropy-production
  current for a CRN.

  The right construction requires tilting the FULL Doi-Peliti generator
  with stoichiometry-dependent log-ratios:
    forward reaction k A -> l A increments sigma by
        ln( rate_fwd * N(N-1)...(N-k+1)/k! / rate_rev * N(N-1)...(N-l+1)/l! )
  which depends on the configuration N.  This translates to a
  Liouvillian tilt that modifies more than just the rates.

  Library status: entropy_production_tilt is now a starting placeholder.
  The proper construction is a follow-up extension; the test framework
  (gallavotti_cohen_test) is correct and ready to verify it once built.
""")


if __name__ == '__main__':
    main()
