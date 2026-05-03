"""Generate the BRW vertex-primitive and 1-loop-bubble figures via rdft.crn.

Thin wrapper that builds the thesis BRW CRN, enumerates its bubbles, and
hands them to ``rdft.crn.viz`` for rendering with full annotations.

Outputs:
  brw_corollas.pdf  -- the 7 BRW interaction primitives.
  brw_bubbles.pdf   -- the 7 distinct 1-loop bubble topologies (E=3 sector).
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from rdft.crn import CRN
from rdft.crn.enumerator import enumerate_bubbles
from rdft.crn.symmetry import aut_bubble
from rdft.crn.viz import draw_corolla, draw_bubble, COLOR_A, COLOR_B


PROVENANCE = {
    "V_branch": ("Eq. 3.16", r"BRW branching $A \to 2A$"),
    "V_QC1":    ("Eq. 3.21", r"$\Lambda$-trace observable (bilinear)"),
    "V_QC2":    ("Eq. 3.21", r"$\Lambda$-trace observable"),
    "V_QC3":    ("Eq. 3.21", r"$\Lambda$-trace observable"),
    "V_QC4":    ("Eq. 3.21", r"$\Lambda$-trace observable"),
    "V_QC5":    ("Eq. 3.21", r"$\Lambda$-trace observable"),
    "V_QC6":    ("Eq. 3.21", r"$\Lambda$-trace observable"),
}


def figure_corollas(crn):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    fig.suptitle("BRW interaction primitives (thesis Eqs. 3.16+3.21) "
                 "via rdft.crn", fontsize=13, y=0.99)
    for i, v in enumerate(crn.vertices):
        ax = axes[i // 4, i % 4]
        ia = v.in_dict().get("A", 0); ib = v.in_dict().get("B", 0)
        oa = v.out_dict().get("A", 0); ob = v.out_dict().get("B", 0)
        eq, role = PROVENANCE[v.name]
        leg_str = (rf"$\tilde\psi_a^{{{ia}}}\tilde\psi_b^{{{ib}}}"
                   rf"\psi_a^{{{oa}}}\psi_b^{{{ob}}}$")
        title = (f"{v.name}  (sign {v.sign:+d}, {eq})\n{leg_str}\n{role}")
        draw_corolla(ax, ia, ib, oa, ob, title=title)
    axes[1, 3].axis("off")
    axes[1, 3].text(0.05, 0.65, "Legend", fontsize=12, fontweight="bold",
                    transform=axes[1, 3].transAxes)
    axes[1, 3].plot([0.05, 0.30], [0.50, 0.50], color=COLOR_A, lw=2,
                    transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.35, 0.50, r"A species ($\psi_a, \tilde\psi_a$)",
                    fontsize=10, va="center", transform=axes[1, 3].transAxes)
    axes[1, 3].plot([0.05, 0.30], [0.35, 0.35], color=COLOR_B, lw=2,
                    transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.35, 0.35, r"B species ($\psi_b, \tilde\psi_b$)",
                    fontsize=10, va="center", transform=axes[1, 3].transAxes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("brw_corollas.pdf", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved brw_corollas.pdf")


def figure_bubbles(crn):
    bubbles = enumerate_bubbles(crn.vertices)
    e3 = [b for b in bubbles if sum(b.external_legs.values()) == 3]
    assert len(e3) == 7

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    fig.suptitle("7 distinct 1-loop loop integrals at $E=3$ "
                 "(matches thesis p.91-92, Eqs. 2.122-2.123) via rdft.crn",
                 fontsize=13, y=0.99)
    for i, b in enumerate(e3):
        ax = axes[i // 4, i % 4]
        ext = b.external_legs
        ext_str = " ".join(f"{c}{k[0]}-{k[1][:3]}" for k, c in sorted(ext.items()) if c > 0)
        title = (f"{b.vertex_names[0]} + {b.vertex_names[1]}    int: "
                 f"{b.internal_lines[0][0]}{b.internal_lines[1][0]}\n"
                 f"ext: {ext_str}    $s(G)={b.s_aut}$")
        draw_bubble(ax, b, title=title)

        # Relevance + lineage from Diagram
        is_relev = (ext.get(("A","in"), 0) == 1 and ext.get(("A","out"), 0) == 2
                    and ext.get(("B","in"), 0) == 0 and ext.get(("B","out"), 0) == 0)
        verdict = "RG-RELEVANT" if is_relev else "TRACE OBSERVABLE"
        color = "tab:green" if is_relev else "tab:gray"
        ax.text(0.5, -0.05, verdict, ha='center', va='top', fontsize=9,
                fontweight='bold', color=color, transform=ax.transAxes)
        # Lineage note: pull the |Aut| explanation from the Diagram
        aut_note = next((p.detail for p in b.lineage if "|Aut|" in p.detail),
                        f"|Aut| = {b.s_aut}")
        ax.text(0.5, -0.18, aut_note, ha='center', va='top',
                fontsize=7, color='dimgray', wrap=True, transform=ax.transAxes)

    # 8th panel: canonical s=2 bubble (E=2, thesis Eq. 3.25)
    ax_canon = axes[1, 3]
    s_canon = aut_bubble("V_branch", "V_branch", "A", "lr", "A", "rl")
    ax_canon.set_aspect("equal"); ax_canon.axis("off")
    ax_canon.set_xlim(-1.5, 1.5); ax_canon.set_ylim(-1.2, 1.2)
    ax_canon.plot(-0.55, 0, "ko", ms=9, zorder=5)
    ax_canon.plot(0.55, 0, "ko", ms=9, zorder=5)
    theta = np.linspace(0, math.pi, 60)
    rx, ry = 0.55, 0.40
    ax_canon.plot(rx*np.cos(theta), ry*np.sin(theta), color=COLOR_A, lw=1.6)
    ax_canon.plot(rx*np.cos(theta), -ry*np.sin(theta), color=COLOR_A, lw=1.6)
    ax_canon.set_title(rf"$V_{{\rm branch}} + V_{{\rm branch}}$    int: AA"
                       "\n"
                       rf"ext: 2A-out    $s(G)={s_canon}$  [thesis Eq.~3.25]",
                       fontsize=9, pad=4)
    ax_canon.text(0.5, -0.05, "REFERENCE BUBBLE (E=2)", ha='center', va='top',
                  fontsize=9, fontweight='bold', color='tab:blue',
                  transform=ax_canon.transAxes)
    ax_canon.text(0.5, -0.18, "vertex-swap symmetry (same vertex type, lines invariant)",
                  ha='center', va='top', fontsize=7, color='dimgray',
                  wrap=True, transform=ax_canon.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("brw_bubbles.pdf", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved brw_bubbles.pdf")


def main():
    crn = CRN.brw_thesis()
    figure_corollas(crn)
    figure_bubbles(crn)


if __name__ == "__main__":
    main()
