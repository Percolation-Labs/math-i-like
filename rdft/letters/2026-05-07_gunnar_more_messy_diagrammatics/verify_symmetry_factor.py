"""
Verify the closed-form symmetry-factor / multiplicity formula for each kernel
(chunk-length pattern) at n=3, m=5 against direct enumeration.

Claim: for an unordered partition lambda = (l_1, ..., l_k) of m+1, the signed
multiplicity of the kernel labelled by lambda at uniform coupling v=1 on K_n is

    mu(lambda; n, m) = C(lambda) * A_{00}^(k-1)(n) * (n-1)^(m+1-k) * (-1)^(k-1)

where:
  C(lambda)         = k! / prod_p c_p!     (number of time-orderings of lambda)
  c_p               = number of chunks of length p in lambda
  A = J_n - I_n     adjacency of K_n minus self-loops
  A_{00}^(k-1)(n)   = ((n-1)^(k-1) + (n-1)(-1)^(k-1)) / n
                    (number of length-(k-1) rail-jump sequences 0 -> 0)
  (n-1)^(m+1-k)     = product of (n-1) over the (m+1-k) total stays
  (-1)^(k-1)        = sign from k-1 swaps

This script:
 (1) enumerates trajectories of length m on K_n, computes signed sum per
     unordered chunk-length partition lambda;
 (2) computes the closed form above for each lambda;
 (3) compares.
"""

from collections import Counter
from fractions import Fraction
from math import factorial


def chunks_of(traj):
    """Return the multiset of chunk lengths (as sorted-descending tuple)."""
    chunks = []
    cur, run = traj[0], 1
    for r in traj[1:]:
        if r == cur:
            run += 1
        else:
            chunks.append(run)
            cur, run = r, 1
    chunks.append(run)
    return tuple(sorted(chunks, reverse=True))


def trajectories(n, m, state=0, depth=0, traj=None, sign=1):
    if traj is None:
        traj = [0]
    if depth == m:
        if state == 0:
            yield sign, tuple(traj)
        return
    others = [r for r in range(n) if r != state]
    for j in others:                       # stay
        yield from trajectories(n, m, state, depth+1, traj+[state], sign)
    for b in others:                       # swap
        yield from trajectories(n, m, b, depth+1, traj+[b], -sign)


def comp_count(lam):
    """k! / prod_p c_p!  (number of distinct time-orderings)."""
    k = len(lam)
    cp = Counter(lam)
    denom = 1
    for c in cp.values():
        denom *= factorial(c)
    return factorial(k) // denom


def A_pow_at_00(n, p):
    """( (n-1)^p + (n-1)(-1)^p ) / n  --- a non-negative integer."""
    num = (n - 1) ** p + (n - 1) * ((-1) ** p)
    assert num % n == 0, "A^p_{00} should be an integer."
    return num // n


def mu_closed_form(lam, n, m):
    k = len(lam)
    # composition count
    C = comp_count(lam)
    # rail-labeling count via no-self-loop random walk
    R = A_pow_at_00(n, k - 1)
    # per-walk signed weight at v = 1
    stays = (m + 1) - k                   # total stays
    w = (n - 1) ** stays * ((-1) ** (k - 1))
    return C * R * w


def main(n=3, m=5):
    pattern_signed = Counter()
    pattern_count = Counter()
    for sign, traj in trajectories(n, m):
        lam = chunks_of(traj)
        pattern_signed[lam] += sign
        pattern_count[lam] += 1

    print(f"n={n}, m={m}")
    print(f"{'partition lambda':30} {'#walks':>8} {'signed':>10} "
          f"{'closed form':>14} {'match':>6}")
    print("-" * 76)
    total_signed = 0
    for lam in sorted(pattern_signed, key=lambda l: (-len(l), l)):
        signed = pattern_signed[lam]
        predicted = mu_closed_form(lam, n, m)
        match = (signed == predicted)
        total_signed += signed
        print(f"{str(lam):30} {pattern_count[lam]:>8} {signed:>10} "
              f"{predicted:>14} {match!s:>6}")
        assert match, f"mismatch at {lam}"
    expected_total = Fraction(n - 1, n) * n ** m
    print("-" * 76)
    print(f"Sum of signed multiplicities: {total_signed}")
    print(f"Expected (n-1)/n * n^m       : {expected_total}")
    assert total_signed == expected_total
    print("\nAll closed-form predictions match enumeration.")


if __name__ == "__main__":
    main(n=3, m=5)
    print()
    main(n=3, m=4)
    print()
    main(n=4, m=4)
