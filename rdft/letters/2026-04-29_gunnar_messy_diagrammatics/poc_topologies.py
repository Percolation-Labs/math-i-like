"""POC: CRN -> labelled animals -> Legendre, pictorial.

Thin wrapper around the rdft.crn API. Enumerates plane phi-trees of size 7
(the 5 connected 2-loop vertex topologies of Reggeon DP), classifies them
1PI vs reducible from the shape string, and renders a multi-panel figure
showing the W -> Gamma transition.

Output: poc_topologies.pdf
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from rdft.crn.enumerator import enumerate_phi_trees, diagram_from_phi_tree

# The Legendre coefficients are fixed AC outputs; re-running the symbolic
# transform here is wasteful. They are recomputed and asserted by the
# dedicated poc_legendre.py script. Hard-coding them keeps this drawing
# script fast.
W_g5_psi2psit = -7608   # = legendre_reggeon_dp(5,4).W_coef(2,1,5)
Gamma_g5_Phi2Phit = -504  # = legendre_reggeon_dp(5,4).Gamma_coef(2,1,5)


# Reggeon-graph drawing primitives (kept here, since they are presentation-only)
def draw_phi_tree(ax, shape: str, x=0.5, y=0.85, dx=0.5, dy=0.18, color="black"):
    def _parse(s):
        s = s.replace(" ", "")
        if s == "L":
            return None
        depth = 0
        inner = s[1:-1]
        for i, c in enumerate(inner):
            if c == "(": depth += 1
            elif c == ")": depth -= 1
            elif c == "," and depth == 0:
                return (inner[:i], inner[i+1:])
        return None

    def _height(s):
        if s == "L":
            return 1
        a, b = _parse(s)
        return 1 + max(_height(a), _height(b))

    def _draw(s, x, y, dx):
        if s == "L":
            ax.plot(x, y, marker='o', markersize=7, color=color, mfc='white', mec=color)
            return
        ax.plot(x, y, marker='o', markersize=10, color=color, mfc=color)
        a, b = _parse(s)
        if a is not None:
            lx, ly = x - dx / 2, y - dy
            ax.plot([x, lx], [y, ly], color=color, lw=1.2)
            _draw(a, lx, ly, dx / 1.6)
        if b is not None:
            rx, ry = x + dx / 2, y - dy
            ax.plot([x, rx], [y, ry], color=color, lw=1.2)
            _draw(b, rx, ry, dx / 1.6)

    _draw(shape, x, y, dx)


def draw_reggeon_topology(ax, name: str):
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")
    if name == "ice-cream":
        v = [(0, 0.6), (-0.5, -0.2), (0.5, -0.2)]
        for i in range(3):
            ax.plot([v[i][0], v[(i+1)%3][0]], [v[i][1], v[(i+1)%3][1]], 'k-', lw=1.2)
        for vx, vy in v: ax.plot(vx, vy, 'ko', ms=6)
        ax.add_patch(mpatches.Ellipse((0, 0.85), 0.35, 0.25, fill=False, lw=1.2))
        ax.plot([0, 0], [0.6, 0.72], 'k-', lw=1.2); ax.plot(0, 0.72, 'ko', ms=4)
        ax.plot([-0.5, -0.9], [-0.2, -0.5], 'k-', lw=1.2)
        ax.plot([0.5, 0.9], [-0.2, -0.5], 'k-', lw=1.2)
        ax.plot([0, 0], [0.97, 1.15], 'k-', lw=1.2)
    elif name == "box":
        v = [(-0.45, 0.45), (0.45, 0.45), (0.45, -0.45), (-0.45, -0.45)]
        for i in range(4):
            ax.plot([v[i][0], v[(i+1)%4][0]], [v[i][1], v[(i+1)%4][1]], 'k-', lw=1.2)
        ax.plot(0, 0, 'ko', ms=6)
        ax.plot([v[0][0], 0], [v[0][1], 0], 'k-', lw=1.2)
        ax.plot([v[2][0], 0], [v[2][1], 0], 'k-', lw=1.2)
        for vx, vy in v: ax.plot(vx, vy, 'ko', ms=6)
        ax.plot([-0.45, -0.85], [0.45, 0.75], 'k-', lw=1.2)
        ax.plot([0.45, 0.85], [0.45, 0.75], 'k-', lw=1.2)
        ax.plot([0.45, 0.85], [-0.45, -0.75], 'k-', lw=1.2)
    elif name == "ladder":
        v0, v1, v2, v3, vc = (0,0.6),(-0.55,0),(0.55,0),(0,-0.6),(0,0)
        for a, b in [(v0,v1),(v0,v2),(v3,v1),(v3,v2),(v1,vc),(vc,v2)]:
            ax.plot([a[0],b[0]], [a[1],b[1]], 'k-', lw=1.2)
        for vx, vy in [v0, v1, v2, v3, vc]: ax.plot(vx, vy, 'ko', ms=6)
        ax.plot([v0[0], v0[0]], [v0[1], v0[1]+0.35], 'k-', lw=1.2)
        ax.plot([v3[0], v3[0]], [v3[1], v3[1]-0.35], 'k-', lw=1.2)
        ax.plot([v2[0], v2[0]+0.35], [v2[1], v2[1]], 'k-', lw=1.2)
    elif name == "Sigma1_psi":
        v = [(0, 0.4), (-0.4, -0.3), (0.4, -0.3)]
        for i in range(3):
            ax.plot([v[i][0], v[(i+1)%3][0]], [v[i][1], v[(i+1)%3][1]], 'k-', lw=1.2)
        for vx, vy in v: ax.plot(vx, vy, 'ko', ms=6)
        ax.plot([0, 0], [0.4, 0.55], 'k-', lw=1.2)
        ax.plot([-0.4, -0.55], [-0.3, -0.45], 'k-', lw=1.2)
        ax.add_patch(mpatches.Ellipse((-0.7, -0.6), 0.25, 0.18, angle=45, fill=False, lw=1.2))
        ax.plot(-0.55, -0.45, 'ko', ms=4); ax.plot(-0.85, -0.75, 'ko', ms=4)
        ax.plot([-0.85, -1.0], [-0.75, -0.95], 'k-', lw=1.2)
        ax.plot([0.4, 0.85], [-0.3, -0.6], 'k-', lw=1.2)
    elif name == "Sigma1_psit":
        v = [(0, 0.4), (-0.4, -0.3), (0.4, -0.3)]
        for i in range(3):
            ax.plot([v[i][0], v[(i+1)%3][0]], [v[i][1], v[(i+1)%3][1]], 'k-', lw=1.2)
        for vx, vy in v: ax.plot(vx, vy, 'ko', ms=6)
        ax.plot([0, 0], [0.4, 0.55], 'k-', lw=1.2)
        ax.add_patch(mpatches.Ellipse((0, 0.78), 0.32, 0.22, fill=False, lw=1.2))
        ax.plot(0, 0.55, 'ko', ms=4); ax.plot(0, 1.01, 'ko', ms=4)
        ax.plot([0, 0], [1.01, 1.15], 'k-', lw=1.2)
        ax.plot([-0.4, -0.85], [-0.3, -0.6], 'k-', lw=1.2)
        ax.plot([0.4, 0.85], [-0.3, -0.6], 'k-', lw=1.2)


def main():
    diagrams = [diagram_from_phi_tree(s) for s in enumerate_phi_trees(7)]
    assert len(diagrams) == 5

    W_3pt = W_g5_psi2psit
    Gamma_3pt = Gamma_g5_Phi2Phit

    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(r"CRN $\phi(G)=1+G^2$ $\to$ labelled animals $\to$ Legendre $\to$ 1PI",
                 fontsize=14, y=0.98)
    gs = GridSpec(4, 5, figure=fig, height_ratios=[1.3, 1.3, 1.0, 1.0],
                  hspace=0.55, wspace=0.25)

    for i, d in enumerate(diagrams):
        col_color = "tab:green" if d.is_1PI else "tab:red"
        ax_tree = fig.add_subplot(gs[0, i])
        ax_tree.set_xlim(0, 1); ax_tree.set_ylim(0, 1); ax_tree.axis("off")
        draw_phi_tree(ax_tree, d.shape, x=0.5, y=0.85, dx=0.5, dy=0.18, color=col_color)
        ax_tree.set_title(d.shape, fontsize=9, pad=4)

        ax_graph = fig.add_subplot(gs[1, i])
        draw_reggeon_topology(ax_graph, d.topology)
        ax_graph.set_title(d.topology, fontsize=10, color=col_color, pad=4)

        ax_ann = fig.add_subplot(gs[2, i])
        ax_ann.axis("off"); ax_ann.set_xlim(0, 1); ax_ann.set_ylim(0, 1)
        verdict = "RELEVANT (1PI)" if d.is_1PI else "IRRELEVANT (reducible)"
        ax_ann.text(0.5, 0.95, verdict, ha='center', va='top',
                    fontsize=10, fontweight='bold', color=col_color, transform=ax_ann.transAxes)
        ax_ann.text(0.5, 0.72, fr"$s(G)={d.s_aut}$", ha='center', va='top',
                    fontsize=12, transform=ax_ann.transAxes)
        # Pull the |Aut| lineage from the diagram itself
        aut_note = next((p.detail for p in d.lineage if "|Aut|" in p.detail or "2^k" in p.detail),
                        f"|Aut| from phi-tree shape")
        ax_ann.text(0.5, 0.45, aut_note, ha='center', va='top', fontsize=7,
                    color='dimgray', wrap=True, transform=ax_ann.transAxes, style='italic')

    # Bottom row: W vs Gamma
    ax_W = fig.add_subplot(gs[3, 0:2])
    ax_W.axis("off")
    ax_W.text(0.0, 0.95, "W (connected, 5 graphs)", fontsize=12, fontweight='bold',
              transform=ax_W.transAxes, va='top')
    ax_W.text(0.0, 0.65, fr"$[g^5\,\psi^2\tilde\psi]\,W = {W_3pt}$",
              fontsize=14, transform=ax_W.transAxes, va='top')

    ax_arrow = fig.add_subplot(gs[3, 2]); ax_arrow.axis("off")
    ax_arrow.annotate("", xy=(0.95, 0.55), xytext=(0.05, 0.55), xycoords='axes fraction',
                      arrowprops=dict(arrowstyle="->", lw=2.5, color='black'))
    ax_arrow.text(0.5, 0.7, "Legendre", fontsize=11, fontweight='bold',
                  ha='center', va='bottom', transform=ax_arrow.transAxes)

    ax_G = fig.add_subplot(gs[3, 3:5]); ax_G.axis("off")
    ax_G.text(0.0, 0.95, r"$\Gamma$ (1PI, after Legendre)", fontsize=12, fontweight='bold',
              transform=ax_G.transAxes, va='top')
    ax_G.text(0.0, 0.65, fr"$[g^5\,\Phi^2\tilde\Phi]\,\Gamma = {Gamma_3pt}$",
              fontsize=14, transform=ax_G.transAxes, va='top')
    ax_G.text(0.0, 0.30, fr"$W-\Gamma = {W_3pt - Gamma_3pt}$ (the 2 reducibles)",
              fontsize=9, transform=ax_G.transAxes, va='top')

    fig.text(0.005, 0.88, r"$\phi$-trees" + "\n(size 7,\nLagrange):",
             fontsize=10, fontweight='bold', va='center')
    fig.text(0.005, 0.65, "Reggeon-DP\ntopology:",
             fontsize=10, fontweight='bold', va='center')
    fig.text(0.005, 0.43, "Relevance\n& $s(G)$:",
             fontsize=10, fontweight='bold', va='center')

    out = "poc_topologies.pdf"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")

    # Console summary
    print()
    print(f"  W:     [g^5 psi^2 psi-tilde]  = {W_3pt}")
    print(f"  Gamma: [g^5 Phi^2 Phi-tilde]  = {Gamma_3pt}")
    print(f"  Diff                          = {W_3pt - Gamma_3pt}")


if __name__ == "__main__":
    main()
