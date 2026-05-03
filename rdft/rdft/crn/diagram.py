"""
rdft.crn.diagram
================

The ``Diagram`` dataclass is the auditable artifact of the CRN -> RG pipeline.
Each ``Diagram`` carries:

* a shape string (the plane phi-tree representation, when applicable);
* the directed Feynman graph (vertices + lines + externals);
* a 1PI verdict;
* a symmetry factor ``s_aut = |Aut(Gamma)|``;
* a ``Provenance`` chain explaining which rule produced each field.

The provenance chain is what makes the pipeline auditable: every ``Diagram``
can answer "where did this number come from?" by walking ``self.lineage``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Provenance:
    """One line of derivation: which rule fired and what it produced."""
    layer: str         # "Layer 1" | "Layer 2" | "Layer 3" | "Layer 4" | "input"
    rule: str          # e.g. "Doi shift", "Lagrange inversion", "phi-tree |Aut|"
    detail: str        # human-readable detail, e.g. "[z^3]G = 1 from (1/3)C(3,1)"
    reference: str = ""  # optional pointer: section / equation / paper

    def __str__(self) -> str:
        ref = f" ({self.reference})" if self.reference else ""
        return f"[{self.layer}] {self.rule}: {self.detail}{ref}"


@dataclass
class Diagram:
    """A 1PI / connected Feynman graph with full provenance."""
    id: str
    shape: str = ""                                  # phi-tree shape string, if any
    topology: str = ""                               # "ladder", "box", "ice-cream", etc.
    vertex_names: tuple = ()                         # vertex types from the CRN
    internal_lines: tuple = ()                       # ((species, dir), ...)
    external_legs: dict = field(default_factory=dict)
    is_1PI: bool = True
    s_aut: Optional[int] = None                      # |Aut(Gamma)|
    relevance: str = ""                              # "RG-relevant" / "trace observable" / ""
    lineage: list = field(default_factory=list)      # list[Provenance]

    def add_provenance(self, layer: str, rule: str, detail: str, reference: str = "") -> None:
        self.lineage.append(Provenance(layer=layer, rule=rule, detail=detail, reference=reference))

    def explain(self) -> str:
        """Plain-text walk-through of how each field was derived."""
        lines = [f"Diagram {self.id}"]
        if self.shape:
            lines.append(f"  shape    : {self.shape}")
        if self.topology:
            lines.append(f"  topology : {self.topology}")
        if self.vertex_names:
            lines.append(f"  vertices : {' + '.join(self.vertex_names)}")
        if self.internal_lines:
            lines.append(f"  internal : {self.internal_lines}")
        if self.external_legs:
            lines.append(f"  external : {dict(self.external_legs)}")
        lines.append(f"  1PI?     : {self.is_1PI}")
        if self.s_aut is not None:
            lines.append(f"  s(G)     : {self.s_aut}")
        if self.relevance:
            lines.append(f"  relevance: {self.relevance}")
        if self.lineage:
            lines.append("  lineage  :")
            for p in self.lineage:
                lines.append(f"    - {p}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.explain()
