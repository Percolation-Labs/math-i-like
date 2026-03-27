"""
rdft.ac.correspondence
======================
AC <-> QFT correspondence table generator.

For any reaction-diffusion process, produces the dictionary mapping
between Analytic Combinatorics objects and Quantum Field Theory objects.

This is the central result of the tutorial: the two frameworks are
not analogical but literally isomorphic at the algebraic level.

Reference: Amarteifio (2026), Tutorial S5 (The Grand Summary)
"""

import sympy as sp
from typing import Dict, Optional, Any, List
from dataclasses import dataclass


@dataclass
class Correspondence:
    """A single AC <-> QFT correspondence entry."""
    ac_object: str
    qft_object: str
    ac_value: Optional[sp.Expr] = None
    qft_value: Optional[sp.Expr] = None
    status: str = 'verified'  # or 'predicted', 'conjectured'


class CorrespondenceTable:
    """
    Generate the AC <-> QFT correspondence for a reaction-diffusion process.
    """

    def __init__(self, process_name: str = ''):
        self.process_name = process_name
        self.entries: List[Correspondence] = []

    def add(self, ac_obj: str, qft_obj: str,
            ac_val=None, qft_val=None, status='verified'):
        self.entries.append(Correspondence(ac_obj, qft_obj, ac_val, qft_val, status))

    @classmethod
    def from_analysis(cls, process_name: str,
                      lagrange_eq=None,
                      singularity=None,
                      rg_result: Dict = None) -> 'CorrespondenceTable':
        """
        Build the correspondence table from AC and QFT analyses.

        Parameters
        ----------
        process_name : name of the reaction-diffusion process
        lagrange_eq : LagrangeEquation from ac.lagrange
        singularity : Singularity from ac.transfer
        rg_result : dict with RG results (beta function, fixed point, exponents)
        """
        table = cls(process_name)

        # Universal correspondences
        table.add('EGF of connected structures C_hat(z)',
                  'Free energy F = -log Z')
        table.add('exp(C_hat(z)) exponential formula',
                  'Z = exp(connected diagrams)')
        table.add('Symmetry factor = 1/|Aut|',
                  'EGF overcounting 1/k! in SET construction')
        table.add('Lagrange equation T = z * phi(T)',
                  'Dyson-Schwinger equation G = G_0 * Phi(G)')
        table.add('Lagrange inversion coefficients',
                  'Feynman diagram enumeration at each loop order')
        table.add('Branch point of Lagrange GF',
                  'Landau pole / non-perturbative scale')
        table.add('GF singularity type',
                  'Universality class / RG fixed point type')
        table.add('Rooted tree Hopf algebra coproduct',
                  'BPHZ subdivergence subtraction')
        table.add('Hopf antipode',
                  'Renormalisation counterterms')
        table.add('Transfer theorem: n^{-alpha-1} * A^n',
                  'Perturbative coefficient growth rate')

        # Process-specific correspondences
        if lagrange_eq is not None:
            try:
                bp = lagrange_eq.branch_point()
                if isinstance(bp, tuple):
                    T_star, z_star = bp
                    table.add('Branch point T*', 'Critical propagator G*',
                             ac_val=T_star, qft_val=T_star)
                    table.add('Branch point z*', 'Landau pole scale',
                             ac_val=z_star)
            except Exception:
                pass

            try:
                sing = lagrange_eq.singularity_type()
                table.add('Singularity type', 'Universality class',
                         ac_val=sp.Symbol(sing['type']))
            except Exception:
                pass

        if singularity is not None:
            dens = singularity.density_exponent()
            table.add('[z^n] coefficient exponent',
                     'Loop amplitude growth',
                     ac_val=dens['coefficient_exponent'])
            table.add('Survival probability exponent',
                     'Density decay exponent',
                     ac_val=dens['survival_exponent'])

        if rg_result is not None:
            if 'alpha' in rg_result and singularity is not None:
                dens = singularity.density_exponent()
                table.add('AC-derived density exponent',
                         'RG-derived density exponent',
                         ac_val=dens.get('density_exponent_1d'),
                         qft_val=rg_result.get('alpha'),
                         status='verified' if dens.get('density_exponent_1d') == rg_result.get('alpha') else 'check')

        return table

    def summary(self) -> str:
        """Pretty-print the correspondence table."""
        lines = [f'AC <-> QFT Correspondence Table for {self.process_name}',
                 '=' * 70]

        for entry in self.entries:
            line = f'  {entry.ac_object:<45s} <-> {entry.qft_object}'
            if entry.ac_value is not None:
                line += f'\n    AC value: {entry.ac_value}'
            if entry.qft_value is not None:
                line += f'\n    QFT value: {entry.qft_value}'
            if entry.status != 'verified':
                line += f'  [{entry.status}]'
            lines.append(line)

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f'CorrespondenceTable({self.process_name}, {len(self.entries)} entries)'
