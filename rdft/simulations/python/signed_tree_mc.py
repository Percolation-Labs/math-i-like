"""
Signed-tree Monte Carlo for direct c_n estimation.

The DSE Lagrange equation G = z phi(G) with phi(G) = 1 - G + 3 G^2 + G^3
can be simulated as a branching process by sampling OFFSPRING distributions
with signed weights:
    P(k offspring)  proportional to  |phi_k|  (normalisation = sum of |phi_k|)
    sign multiplier = sign(phi_k)

Lagrange inversion gives:
    c_n = (1 / n) * [G^{n-1}] phi(G)^n
but also:
    c_n = E[ sign-product over tree of size n ] * (constant)

By sampling many trees and binning by size, we get an estimator of c_n that
converges to the exact sympy value.  Checking its ratio to the exact n^{-4/3}
prediction gives a direct confirmation.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from dataclasses import dataclass


# phi(G) = 1 - G + 3 G^2 + G^3
COEFFS = {0: 1.0, 1: -1.0, 2: 3.0, 3: 1.0}
WEIGHTS_ABS = np.array([abs(v) for k, v in sorted(COEFFS.items())])
SIGNS = np.array([np.sign(v) for k, v in sorted(COEFFS.items())])
NORM = WEIGHTS_ABS.sum()
PROBS = WEIGHTS_ABS / NORM   # probability distribution on {0,1,2,3}
# Each tree has a "weight"  prod_vertices sign * NORM^(n-1)  (one factor per vertex
# picked).  We track log-signs and accumulate.


def sample_tree(choices: np.ndarray, start: int, max_size: int) -> tuple[int, int, int]:
    """Consume offspring choices from a pre-generated buffer until pending hits 0.

    Returns (n_vertices, sign_product, new_start).
    """
    pending = 1
    size = 0
    sign = 1
    i = start
    n_choices = len(choices)
    while pending > 0:
        if i >= n_choices:
            return size, sign, i   # ran out; caller will refill
        k = choices[i]; i += 1
        size += 1
        sign *= int(SIGNS[k])
        pending = pending - 1 + k
        if size >= max_size:
            return size, sign, i
    return size, sign, i


def lagrange_exact(N: int) -> np.ndarray:
    """Compute exact c_n for n = 1..N via Lagrange inversion."""
    G = sp.Symbol('G')
    phi = 1 - G + 3 * G ** 2 + G ** 3
    c = np.zeros(N, dtype=np.float64)
    phi_n = sp.Integer(1)
    for n in range(1, N + 1):
        phi_n = sp.expand(phi_n * phi)
        c[n - 1] = float(sp.Rational(1, n) * phi_n.coeff(G, n - 1))
    return c


def main():
    N_MAX = 60
    n_samples = 500_000
    rng = np.random.default_rng(42)
    BUF = 1_000_000  # pre-draw batches

    size_counts = np.zeros(N_MAX + 1, dtype=np.int64)
    size_sign_sums = np.zeros(N_MAX + 1, dtype=np.int64)

    truncated = 0
    choices = rng.choice(4, size=BUF, p=PROBS)
    idx = 0
    for i in range(n_samples):
        while True:
            s, sgn, idx = sample_tree(choices, idx, max_size=N_MAX + 1)
            if idx < len(choices):
                break
            # buffer exhausted; refill and retry this sample
            choices = rng.choice(4, size=BUF, p=PROBS)
            idx = 0
        if s > N_MAX:
            truncated += 1
            continue
        size_counts[s] += 1
        size_sign_sums[s] += sgn

    # MC estimator.  Under offspring probability |phi_k|/NORM, a rooted plane tree
    # T of size n has sampler probability (prod |phi_{k_v}|)/NORM^n.  Then
    #   E[sign(T) * 1_{|T|=n}] = sum_{T: |T|=n} (prod phi_{k_v}) / NORM^n = c_n / NORM^n,
    # since c_n = sum of phi-weighted size-n rooted plane trees (Lagrange inversion).
    # So:  c_hat_n = E_hat[sign * 1_{|T|=n}] * NORM^n.

    exact_c = lagrange_exact(N_MAX)

    print(f'{"n":>4} {"count":>10} {"<sign>":>12} {"c_hat":>16} {"c_exact":>16} {"ratio":>10}')
    for n in [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]:
        cnt = size_counts[n]
        sgn_sum = size_sign_sums[n]
        mean_sign = sgn_sum / n_samples
        c_hat = mean_sign * NORM ** n
        c_ex = exact_c[n - 1]
        ratio = c_hat / c_ex if c_ex != 0 else float('nan')
        print(f'{n:>4} {cnt:>10} {mean_sign:>12.4e} {c_hat:>16.4e} {c_ex:>16.4e} {ratio:>10.4f}')
    print(f'\nTruncated (|T| > {N_MAX}) samples: {truncated} of {n_samples}')

    # save
    from pathlib import Path
    outpath = Path(__file__).parent.parent.parent / 'paper' / 'cfac' / 'figures' / 'signed_tree_mc.npz'
    np.savez(outpath,
             size_counts=size_counts,
             size_sign_sums=size_sign_sums,
             exact_c=exact_c,
             n_samples=n_samples,
             NORM=NORM)
    print(f'\nSaved MC data to {outpath}')


if __name__ == '__main__':
    main()
