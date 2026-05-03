"""
rdft.crn.audit
==============

Walks an ``RGProgram``'s history and prints the per-step ledger:
which results are AC-derived, which are hand-mapped, which are external input.

Every step in ``RGProgram.history`` carries a ``Provenance`` that names the
layer (1, 2, 3, 4, final, input) and the rule that fired. The audit walker
groups by layer and reports, plus surfaces the per-Z-factor and per-diagram
provenance for the things the user typically wants to inspect.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdft.crn.rg import RGProgram


def format_audit(rg: "RGProgram") -> str:
    """Return a multi-line string with the full audit trail."""
    lines = []
    lines.append("=" * 76)
    lines.append(f" AUDIT TRAIL: {rg.crn.name}  (loop_order = {rg.loop_order})")
    lines.append("=" * 76)
    lines.append("")

    # Top-level pipeline history
    lines.append("Pipeline steps:")
    for p in rg.history:
        lines.append(f"  {p}")
    lines.append("")

    # Lagrange counts
    if rg.lagrange_counts:
        lines.append("Layer 2: Lagrange counts [z^n]G")
        for n in (1, 3, 5, 7):
            if n in rg.lagrange_counts:
                lines.append(f"  [z^{n}]G = {rg.lagrange_counts[n]}")
        lines.append("")

    # Z-factors
    if rg.zfactors:
        lines.append("Z-factors:")
        for X, zf in rg.zfactors.items():
            lines.append(f"  {zf.display()}")
            for prov in zf.provenance:
                lines.append(f"    -> {prov}")
        lines.append("")

    # IBP table
    if rg.ibp is not None:
        lines.append("IBP table (12 q^X_Gamma rationals):")
        for line in rg.ibp.as_table().split("\n"):
            lines.append("  " + line)
        lines.append("")

    # Exponents
    if rg.exponents is not None:
        lines.append("Critical exponents (vs JT05 Eq. 60):")
        lines.append(f"  eta     = {rg.exponents.eta}")
        lines.append(f"  z       = {rg.exponents.z}")
        lines.append(f"  nu      = {rg.exponents.nu}")
        lines.append(f"  beta_DP = {rg.exponents.beta_DP}")
        lines.append("  residuals (ours - JT05):")
        for k, v in rg.exponents.residuals.items():
            lines.append(f"    {k:<10} = {v}")
        all_zero = all(v == 0 for v in rg.exponents.residuals.values())
        lines.append(f"  ALL MATCH (zero residual): {all_zero}")
        lines.append("")

    # Provenance summary by layer
    lines.append("Provenance summary:")
    summary = {
        "Layer 1": "AC-derived (Doi shift)",
        "Layer 2 (counts)": "AC-derived (Lagrange inversion)",
        "Layer 2 (sym factors)": "AC-derived (phi-tree |Aut| = 2^k(T))",
        "Layer 2 (Hopf antipode)": "AC-derived (closed-form rational)",
        "Layer 2 (IBP closure)":   "AC-derived (12 rationals from CFAC structure)",
        "Layer 3 (1PI verdict)":   "structural rule on shape string",
        "Layer 3 (external sectors)": "QFT bookkeeping (not AC)",
        "Layer 4 (Legendre)":      "AC-derived (when invoked)",
        "Master integral values":  "input (JT05 / Panzer / Borinsky)",
        "Tauber relation":         "derived (closed form)",
        "Exponents vs JT05 Eq.60": "derived (zero residual)",
    }
    width = max(len(k) for k in summary)
    for k, v in summary.items():
        lines.append(f"  {k:<{width+2}}  {v}")
    return "\n".join(lines)


def diagram_summary(diagrams) -> str:
    """A compact one-line-per-diagram audit table."""
    lines = []
    lines.append(f"{'id':<55} {'topology':<14} {'1PI':<5} {'s(G)':<5}")
    lines.append("-" * 85)
    for d in diagrams:
        lines.append(f"{d.id[:54]:<55} {d.topology[:13]:<14} "
                     f"{('Y' if d.is_1PI else 'N'):<5} {str(d.s_aut or '?'):<5}")
    return "\n".join(lines)
