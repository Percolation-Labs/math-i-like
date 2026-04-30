"""
ibp_solver
==========

Minimal in-Python Laporta-style IBP reducer for small Feynman-integral
families.  Built from scratch following:

  - Laporta, Int. J. Mod. Phys. A 15, 5087 (2000); arXiv:hep-ph/0102033.
  - Smirnov, "Analytic Tools for Feynman Integrals", Ch. 6 (2012).
  - Chetyrkin & Tkachov, Nucl. Phys. B 192, 159 (1981).

Scope: this implementation handles propagator families with up to
3-4 propagators in 2-loop topologies (sunset, simple vertex graphs).
For larger families one would defer to KIRA/FIRE.

Public API:

    from ibp_solver import IBPFamily, SunsetMassless, reduce_integral

The library is intentionally minimal --- about 500 lines.  It serves
as a reference implementation and a plugin for the CFAC pipeline.
"""

from .core import IBPFamily, IBPIdentity, IBPSystem

__all__ = ['IBPFamily', 'IBPIdentity', 'IBPSystem']
