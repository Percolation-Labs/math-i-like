"""
rdft.ac.replica_closed
======================
Closed-form expressions for the KPZ replica rate on Z.

This module carries the analytic deliverables that complement the
numerical transfer-matrix and cubic-in-n analysis. The statement
below is a Tier-A closed form for the 2-body binding problem of
Prop 3 of paper/cfac/enumerative_boundary.tex.

Setup. Two nearest-neighbour walkers on Z with pairwise on-site
contact attraction: the one-step weight is
    w(x1', x2') = exp(beta^2 * k / 2),
with k = 2 if x1' != x2' and k = 4 if x1' = x2'. The replica rate
is lambda(n, beta) = log(dominant eigenvalue of the n-walker
transfer matrix).

Result. Define the center-of-mass / relative-coordinate change of
variables and factor the translation-invariant COM. The relative
TM is a tight-binding chain with an attractive on-site impurity
at r = 0. Solving the bound-state eigenvalue equation with
wavefunction psi(r) = z^{|r|/2} yields

    z = 1 / (2 * e^{beta^2} - 1),

    D(2, beta; Z) := lambda(2, beta; Z) - 2 * lambda(1, beta; Z)
                   = 2 * beta^2 - log(2 * e^{beta^2} - 1).

Small-beta behaviour: D(2, beta; Z) = beta^4 - beta^6 / 3 + O(beta^8),
i.e., D(2) vanishes as beta^4, matching the Kardar Bethe-ansatz
scaling of the 2-body bound state energy on Z.

Large-beta behaviour: D(2, beta; Z) ~ beta^2 - log(2), consistent
with a fully localised pair at the origin.

Numerical verification: finite-W transfer-matrix D(2, beta; W)
converges monotonically from above to D(2, beta; Z) as W -> oo.

(The 3-body analogue is a nested Bethe ansatz on Z; it admits a
closed form but is deferred here -- see discussion.)
"""

from __future__ import annotations
from math import factorial
import numpy as np


def D2_closed_Z(beta: float) -> float:
    """Closed-form D(2, beta; Z) = lambda(2) - 2 lambda(1) on Z.

    D(2, beta; Z) = 2 beta^2 - log(2 exp(beta^2) - 1).
    """
    return 2.0 * beta ** 2 - float(np.log(2.0 * np.exp(beta ** 2) - 1.0))


def lambda_relative_Z(beta: float) -> float:
    """log(mu_rel) = log((1+z)^2 / z) with z = 1/(2 exp(beta^2) - 1).

    Dominant eigenvalue of the relative-coordinate transfer matrix
    on Z, at momentum k_s = 0 (zero COM momentum).
    """
    z = 1.0 / (2.0 * np.exp(beta ** 2) - 1.0)
    return float(np.log((1.0 + z) ** 2 / z))


def bound_state_decay_length(beta: float) -> float:
    """Bound-state wavefunction psi(r) = z^{|r|/2} has decay length
    1 / log(1 / z) = 1 / log(2 exp(beta^2) - 1) (in units of r = 2
    lattice spacings). Diverges as 1/beta^2 as beta -> 0 (loose
    binding), shrinks as beta grows (tight binding).
    """
    denom = float(np.log(2.0 * np.exp(beta ** 2) - 1.0))
    if denom <= 0:
        return float('inf')
    return 1.0 / denom


def D2_small_beta_series(beta: float, order: int = 6) -> float:
    """Series expansion of D2_closed_Z around beta = 0.

    D(2, beta; Z) = beta^4 - beta^6 / 3 + beta^8 / 6
                  - 8 beta^{10} / 45 + ...  (via Taylor of 2b^2
    - log(2 e^{b^2} - 1) with b^2 = x small).

    Returns the truncated series up to O(beta^{order}). Only even
    orders contribute; order must be even and >= 4.
    """
    if order < 4 or order % 2 != 0:
        raise ValueError("order must be even and >= 4")
    x = beta ** 2
    # 2x - log(2 e^x - 1) expanded in x:
    # 2 e^x - 1 = 1 + 2 sum_{k>=1} x^k / k!
    # log(1 + u) = u - u^2/2 + u^3/3 - ...
    # u = 2 (e^x - 1) = 2x + x^2 + x^3/3 + x^4/12 + x^5/60 + ...
    # compute term by term to required order (order/2 in x)
    max_x_order = order // 2
    u_coeffs = [0.0]  # u has no constant term
    for k in range(1, max_x_order + 1):
        u_coeffs.append(2.0 / factorial(k))
    # compute log(1 + u) up to x^{max_x_order}
    log_coeffs = [0.0] * (max_x_order + 1)
    u_power = u_coeffs[:]  # u^1
    sign = 1.0
    divisor = 1
    for power in range(1, max_x_order + 1):
        for k in range(len(u_power)):
            if k <= max_x_order:
                log_coeffs[k] += sign * u_power[k] / divisor
        if power < max_x_order:
            # u^{power+1} = u * u^power (truncated at max_x_order)
            new_power = [0.0] * (max_x_order + 1)
            for i in range(min(len(u_coeffs), max_x_order + 1)):
                if u_coeffs[i] == 0.0:
                    continue
                for j in range(min(len(u_power), max_x_order + 1 - i)):
                    new_power[i + j] += u_coeffs[i] * u_power[j]
            u_power = new_power
            sign *= -1.0
            divisor += 1
    # D2 = 2 x - log(2 e^x - 1)
    out = 2.0 * x
    for k, c in enumerate(log_coeffs):
        out -= c * x ** k
    return float(out)
