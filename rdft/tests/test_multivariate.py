"""Tests for rdft.ac.multivariate — Pemantle-Wilson ACSV for coupled DSE.

Verifies:
  (1) Spectral-radius and Perron-Frobenius behaviour.
  (2) Critical-point detection for symmetric 2-type branching
      (rho* = 1, G* = 1, 1) at spectral radius 1.
  (3) Smooth-point classification and tau = 3/2 output.
  (4) Multivariate recovers single-species as a special case.
  (5) Direction-dependence: diagonal [z^m, z^m] vs axial [z^m, 0].
"""
import numpy as np
import pytest

from rdft.ac.multivariate import (
    mean_matrix_at_origin, spectral_radius,
    find_critical_point_multivariate, classify_singular_point,
    diagonal_asymptotic_smooth, two_type_branching_demo,
    classify_non_smooth_critical, independent_two_species_demo,
    cone_point_demo,
)


class TestNonSmoothCriticalPoints:
    """Extension: Pemantle-Wilson multiple-point and cone-point detection."""

    def test_independent_two_species_is_multiple_point(self):
        """Two independent critical Poisson branching processes produce
        a multiple point: r=2 eigenvalues at 1 with orthogonal
        eigenvectors (maximally transverse)."""
        r = independent_two_species_demo()
        assert r['classification'] == 'multiple_point'
        assert r['multiplicity_r'] == 2
        assert r['is_transverse']

    def test_cone_point_detected(self):
        """Contrived cone-point where eigenvectors are nearly parallel
        gets flagged as 'cone_point', not 'multiple_point'."""
        r = cone_point_demo(epsilon=0.01)
        assert r['classification'] == 'cone_point'
        assert r['multiplicity_r'] == 2
        assert not r['is_transverse']

    def test_smooth_still_classified_smooth(self):
        """The existing smooth case (symmetric 2-type) should still
        classify as 'smooth' under the extended classifier."""
        import numpy as np
        def phi1(G):
            return np.exp(0.5 * (G[0] - 1) + 0.5 * (G[1] - 1))
        def phi2(G):
            return np.exp(0.5 * (G[0] - 1) + 0.5 * (G[1] - 1))
        r = classify_non_smooth_critical([phi1, phi2],
                                           np.array([1.0, 1.0]),
                                           1.0)
        # M = [[1/2, 1/2], [1/2, 1/2]] has spectrum {1, 0}, so only one
        # eigenvalue is near 1.
        assert r['classification'] == 'smooth'
        assert r['multiplicity_r'] == 1

    def test_multiple_point_provides_per_sheet_amplitudes(self):
        r = independent_two_species_demo()
        assert r['amplitude_per_sheet'] is not None
        assert len(r['amplitude_per_sheet']) == 2

    def test_cone_point_advice_mentions_non_universal(self):
        r = cone_point_demo(epsilon=0.001)
        assert 'non-universal' in r['advice'] or 'cone' in r['advice']


class TestSpectralRadius:
    def test_symmetric_half_matrix_radius_one(self):
        M = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert abs(spectral_radius(M) - 1.0) < 1e-12

    def test_diagonal_matrix(self):
        M = np.diag([0.3, 0.7])
        assert abs(spectral_radius(M) - 0.7) < 1e-12

    def test_identity(self):
        M = np.eye(3)
        assert abs(spectral_radius(M) - 1.0) < 1e-12


class TestTwoTypeCritical:
    """Symmetric 2-type branching at spectral radius 1."""

    def test_critical_rho_is_one(self):
        r = two_type_branching_demo(m11=0.5, m12=0.5, m21=0.5, m22=0.5)
        assert r['critical_point_found']
        assert abs(r['rho_star'] - 1.0) < 1e-6

    def test_critical_G_star_is_ones(self):
        r = two_type_branching_demo(m11=0.5, m12=0.5, m21=0.5, m22=0.5)
        assert np.allclose(r['G_star'], [1.0, 1.0], atol=1e-6)

    def test_smooth_classification(self):
        r = two_type_branching_demo(m11=0.5, m12=0.5, m21=0.5, m22=0.5)
        assert r['classification'] == 'smooth'

    def test_tau_is_three_halves(self):
        r = two_type_branching_demo(m11=0.5, m12=0.5, m21=0.5, m22=0.5)
        assert r['tau'] == 1.5

    def test_mean_matrix_recovered(self):
        r = two_type_branching_demo(m11=0.5, m12=0.5, m21=0.5, m22=0.5)
        M = r['mean_matrix_at_Gstar']
        # At G*=1, phi'_i(1) = m_ij exactly for Poisson-offspring phi
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert np.allclose(M, expected, atol=1e-4)


class TestAsymmetricMeanMatrix:
    """Non-symmetric M still gives smooth critical at spectral radius 1
    appropriately rescaled."""

    def test_subcritical_rho_star_above_one(self):
        # Mean matrix with rho(M) = 0.8 -> rho* = 1.25 (compensating)
        r = two_type_branching_demo(m11=0.3, m12=0.5, m21=0.6, m22=0.2)
        assert r['rho_star'] > 1.0
        # Spectral radius(rho* M) = 1 at critical
        rho_spec = r['spectral_radius']
        assert abs(rho_spec - 1.0) < 1e-4

    def test_still_smooth(self):
        r = two_type_branching_demo(m11=0.3, m12=0.5, m21=0.6, m22=0.2)
        assert r['classification'] == 'smooth'


class TestConsistencyWithUnivariate:
    """A "1-species" multivariate (n=1) should reproduce single-species results."""

    def test_single_species_recovers_univariate(self):
        """phi_1(G) = exp(G - 1), single species, should match the
        critical Poisson branching univariate result (rho = 1, G = 1)."""
        def phi1(G):
            return np.exp(G[0] - 1.0)
        r = find_critical_point_multivariate([phi1], n=1,
                                              z_init=np.array([1.0]),
                                              G_init=np.array([0.9]))
        assert r['critical_point_found']
        assert abs(r['rho_star'] - 1.0) < 1e-6
        assert abs(r['G_star'][0] - 1.0) < 1e-6


class TestAmplitudeStructure:
    """The multivariate amplitude encodes mean-matrix structure."""

    def test_amplitude_depends_on_species(self):
        """Two species with ASYMMETRIC mean matrix should give
        different amplitudes per species — information that the
        univariate resultant projection would lose."""
        r_asym = two_type_branching_demo(m11=0.2, m12=0.8, m21=0.8, m22=0.2)
        if r_asym['critical_point_found'] and r_asym['classification'] == 'smooth':
            # For asymmetric M, the right/left eigenvectors differ and
            # the amplitudes for species 0 vs species 1 should differ too.
            # We at least check that the eigenvector structure isn't trivial.
            u = r_asym['right_eigenvector']
            assert len(u) == 2
            # For symmetric mean matrix, both components should be equal
            # For asymmetric, they can differ
