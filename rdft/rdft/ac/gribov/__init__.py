"""
rdft.ac.gribov
==============

CFAC framework for the Gribov process / directed percolation:
2-loop renormalisation reproducing Janssen-Tauber 2005.

Submodules:
  - assembly:        Lagrange-inversion counts and 1-loop algebra factors
  - actrick:         AC trick at 2 loops (signed bivariate Lagrange,
                     Symanzik polynomials, nested-graph derivation)
  - simple_poles:    BPHZ closed-form counterterm and primitive residue
  - ibp_coefficients: 12 IBP coefficients from CFAC structural constraints
  - two_loop:        Algebraic pipeline Z -> beta -> u* -> exponents
                     (reproduces JT05 Eq. 60 with zero residual)
  - ibp_plugin:      FORM backend plugin (sunset Laurent expansion)
  - run_all_tests:   End-to-end test harness (8/8 passing)

Companion paper: paper/cfac/gribov.tex
"""
