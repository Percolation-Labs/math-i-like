"""
Tests for the rank-k bridge function additions to rdft.ac.bridge.
"""

import numpy as np
import pytest

from rdft.ac.bridge import (
    bridge_rank_k,
    bridge_rank_k_numeric,
    upper_critical_dim_for_cusp,
    gamma_k_anomalous,
    tau_dressed,
)


def test_bridge_rank_k_closed_form():
    """bridge_rank_k(k) = 2 / [(k-1)! (4pi)^k] by Feynman + standard
    momentum-integration formulae."""
    from math import factorial
    for k in [2, 3, 4, 5]:
        expected = 2.0 / (factorial(k - 1) * (4 * np.pi) ** k)
        assert bridge_rank_k(k) == pytest.approx(expected)


def test_bridge_rank_k_matches_scalar_at_k_2():
    """At k=2 the rank-k formula must reproduce the standard scalar bubble."""
    from rdft.ac.bridge import one_loop_pole, bridge_scalar
    expected = one_loop_pole(d_c=4) * bridge_scalar()  # 2 (4pi)^{-2}
    assert bridge_rank_k(2) == pytest.approx(expected)


def test_rank_3_bridge_mass_independent_numerically():
    """Numerical test (Dirichlet Monte-Carlo) — rank-3 bridge is mass-independent
    to ~5% across mass ratios spanning a decade.  MC tolerance reflects
    sampling noise; for tighter tests use deterministic Feynman quadrature."""
    predicted = bridge_rank_k(3)
    masses_list = [
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 5.0],
        [0.1, 1.0, 10.0],
        [0.5, 0.5, 5.0],
    ]
    devs = []
    for masses in masses_list:
        v = bridge_rank_k_numeric(3, masses, eps=1e-4, n_samples=50000, rng_seed=42)
        devs.append(abs(v - predicted) / predicted)
    assert max(devs) < 0.10, f'mass-independence violated: max rel dev = {max(devs):.3f}'


def test_d_c_for_cusp():
    """d_c(k) = 2k/(k-1).  Standard engineering-dim result."""
    assert upper_critical_dim_for_cusp(2) == pytest.approx(4.0)
    assert upper_critical_dim_for_cusp(3) == pytest.approx(3.0)
    assert upper_critical_dim_for_cusp(4) == pytest.approx(8 / 3)
    assert upper_critical_dim_for_cusp(5) == pytest.approx(2.5)


def test_gamma_k_zero_at_d_c():
    """At d = d_c(k), eps = 0, so the one-loop gamma_k vanishes."""
    for k in [2, 3, 4, 5]:
        d_c = upper_critical_dim_for_cusp(k)
        assert gamma_k_anomalous(k, d_c) == 0.0


def test_gamma_k_zero_above_d_c():
    """Above d_c, mean-field is exact, gamma = 0."""
    for k in [3, 4, 5]:
        assert gamma_k_anomalous(k, upper_critical_dim_for_cusp(k) + 1) == 0.0


def test_tau_dressed_equals_skeleton_at_dc():
    """At d = d_c(k), tau equals the skeleton 1 + 1/k."""
    for k in [2, 3, 4, 5]:
        d_c = upper_critical_dim_for_cusp(k)
        assert tau_dressed(k, d_c) == pytest.approx(1 + 1 / k)


def test_tau_dressed_below_dc():
    """Below d_c, tau gets a one-loop shift -eps/2."""
    k = 3
    d_c = upper_critical_dim_for_cusp(k)  # = 3
    # At d = 2: eps = 1, gamma = -1/2, tau = 4/3 - 1/2 = 5/6
    assert tau_dressed(k, 2.0) == pytest.approx(4 / 3 - 0.5)
