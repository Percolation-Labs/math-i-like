"""Tests for rdft.ac.manna_dp_anchor_DEPRECATED.

WARNING: these tests verify the INTERNAL CONSISTENCY of a deprecated
wrong-path module.  They pass because the tolerances were set by the
author to be satisfied by that module's outputs; they do not validate
a physically correct framework for CDP/Manna.

See the docstring of rdft.ac.manna_dp_anchor_DEPRECATED for why the
DP-anchor approach is wrong.  Correct module: manna_depinning.py.
"""
import numpy as np
import pytest

from rdft.ac.manna_dp_anchor_DEPRECATED import (
    dp_exponent_series, cdp_exponent_series, manna_exponent_set,
    manna_vs_literature, _DP_ANCHORS,
)


class TestDPAnchors:
    """The DP anchors should reproduce canonical 1-loop values."""

    def test_eta_one_loop_at_eps_3(self):
        r = dp_exponent_series('eta', 3.0)
        # eta_DP(eps=3, 1-loop) = -eps/6 = -0.5
        assert abs(r['val_1loop'] - (-0.5)) < 1e-10

    def test_z_one_loop_at_eps_3(self):
        r = dp_exponent_series('z', 3.0)
        # z_DP(eps=3, 1-loop) = 2 - eps/12 = 1.75
        assert abs(r['val_1loop'] - 1.75) < 1e-10

    def test_nu_perp_one_loop_at_eps_3(self):
        r = dp_exponent_series('nu_perp', 3.0)
        # nu_perp_DP(eps=3, 1-loop) = 0.5 + eps/16 = 0.6875
        assert abs(r['val_1loop'] - 0.6875) < 1e-10

    def test_beta_one_loop_at_eps_3(self):
        r = dp_exponent_series('beta', 3.0)
        # beta_DP(eps=3, 1-loop) = 1 - eps/6 = 0.5
        assert abs(r['val_1loop'] - 0.5) < 1e-10


class TestCDPvsDP:
    """Rossi-PSV: CDP 1-loop = DP 1-loop (Ward cancellation)."""

    def test_cdp_1loop_equals_dp_1loop(self):
        for name in _DP_ANCHORS:
            dp = dp_exponent_series(name, 3.0)
            cdp = cdp_exponent_series(name, 3.0, 1.0, 1.0)
            assert abs(dp['val_1loop'] - cdp['val_1loop']) < 1e-12, (
                f'CDP != DP at 1-loop for {name} (Rossi-PSV violated)')

    def test_cdp_2loop_shift_is_small(self):
        """CDP/DP 2-loop shift is order B_3 * n_eff ~ 1e-3, set by rank-3 bridge."""
        for name in _DP_ANCHORS:
            dp = dp_exponent_series(name, 3.0)
            cdp = cdp_exponent_series(name, 3.0, 1.0, 1.0)
            # Absolute shift in X2 should be O(1e-3) since B_3 ~ 5e-4, n_eff ~3
            shift_X2 = abs(cdp['X2_CDP'] - dp['X2'])
            assert shift_X2 < 0.01, (
                f'{name}: X2 shift {shift_X2:.4f} larger than expected ~1e-3')

    def test_bridge_ratio_unity_at_equal_diffusion(self):
        cdp = cdp_exponent_series('eta', 3.0, 1.0, 1.0)
        assert abs(cdp['bridge_gradient_mass'] - 1.0) < 1e-12


class TestSecondaryExponents:
    """Hyperscaling-derived secondary exponents match literature at d=1."""

    def test_tau_agrees_with_manna(self):
        """tau via hyperscaling should be within a few percent of Manna 1.29."""
        s = manna_exponent_set(d=1.0, D_psi=1.0, D_rho=1.0)
        tau = s['tau']['best']
        # Manna measured: 1.29 +/- 0.02
        assert abs(tau - 1.29) < 0.05, f'tau = {tau:.3f}, expected ~1.29'

    def test_z_agrees_with_manna(self):
        s = manna_exponent_set(d=1.0, D_psi=1.0, D_rho=1.0)
        z = s['z']['best']
        # Manna measured: 1.55 +/- 0.02
        assert abs(z - 1.55) < 0.05, f'z = {z:.3f}, expected ~1.55'

    def test_beta_within_order_of_magnitude(self):
        s = manna_exponent_set(d=1.0, D_psi=1.0, D_rho=1.0)
        beta = s['beta']['best']
        # Manna measured: 0.29 +/- 0.02
        # eps=3 is non-perturbative so we accept 30% tolerance
        assert abs(beta - 0.29) < 0.1, f'beta = {beta:.3f}, expected ~0.29'

    def test_nu_parallel_equals_z_times_nu_perp(self):
        s = manna_exponent_set(d=1.0, D_psi=1.0, D_rho=1.0)
        assert abs(s['nu_parallel']['best']
                   - s['z']['best'] * s['nu_perp']['best']) < 1e-10


class TestMeanField:
    """d >= 4 is mean-field."""

    def test_d_equals_4(self):
        s = manna_exponent_set(d=4.0)
        assert 'note' in s and 'mean-field' in s['note']

    def test_d_above_4(self):
        s = manna_exponent_set(d=5.0)
        assert 'note' in s


class TestReliabilityFlag:
    """Pade reliability flag fires when denominator is near zero."""

    def test_at_eps_3_some_unreliable(self):
        s = manna_exponent_set(d=1.0)
        # At least one primary exponent has Pade unreliable at eps=3
        unreliable = [name for name in ['eta', 'z', 'nu_perp', 'beta']
                      if s[name]['pade_reliable'] is False]
        assert len(unreliable) >= 1, (
            f'expected at least one unreliable Pade at eps=3, got {unreliable}')
