"""Generate PDF report for multiscale OWT training progress."""

import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def split_train_val(entries):
    train = [e for e in entries if "loss" in e and "val_loss" not in e]
    val = [e for e in entries if "val_loss" in e]
    return train, val


# ── Styling ──────────────────────────────────────────────────────
COLORS = {
    "ms_train": "#2196F3",
    "ms_val": "#1565C0",
    "orig_train": "#BDBDBD",
    "orig_val": "#757575",
    "slow": "#E91E63",
    "fast": "#4CAF50",
    "all": "#FF9800",
    "accent": "#7C4DFF",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig_loss_curves(ms_train, ms_val, orig_train, orig_val):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Train loss
    steps_o = [e["step"] for e in orig_train]
    loss_o = [e["loss"] for e in orig_train]
    steps_m = [e["step"] for e in ms_train]
    loss_m = [e["loss"] for e in ms_train]

    ax1.plot(steps_o, loss_o, color=COLORS["orig_train"], lw=1, label="Original (single-field)", alpha=0.7)
    ax1.plot(steps_m, loss_m, color=COLORS["ms_train"], lw=1.2, label="Multiscale")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)

    # Val PPL
    steps_ov = [e["step"] for e in orig_val]
    ppl_o = [e["ppl"] for e in orig_val]
    steps_mv = [e["step"] for e in ms_val]
    ppl_m = [e["ppl"] for e in ms_val]

    ax2.plot(steps_ov, ppl_o, "o-", color=COLORS["orig_val"], ms=3, lw=1.2,
             label="Original (single-field)")
    ax2.plot(steps_mv, ppl_m, "s-", color=COLORS["ms_val"], ms=3, lw=1.2,
             label="Multiscale")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("Validation Perplexity")
    ax2.legend(fontsize=8)

    fig.suptitle("Multiscale vs Single-Field Training on OpenWebText (125M params)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_retention_evolution():
    """Plot slow field retention half-life over training steps."""
    steps = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
             10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
             18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000,
             26000, 27000, 28000, 29000]
    hl_l0 = [200, 199, 196, 195, 197, 199, 203, 206, 209, 213,
             214, 218, 220, 223, 225, 229, 232, 235, 237, 240,
             242, 244, 248, 249, 251, 253, 254, 257, 260, 261]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, hl_l0, "o-", color=COLORS["slow"], ms=4, lw=1.5,
            label="Layer 0 (longest memory)")
    ax.axhline(y=200, color="gray", ls="--", alpha=0.5, label="Initialization (200)")
    ax.fill_between(steps, 200, hl_l0, alpha=0.1, color=COLORS["slow"])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Half-life (tokens)")
    ax.set_title("Slow Field Retention — Layer 0 Half-Life Over Training")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_retention_by_layer():
    """Show retention across all layers at two snapshots."""
    layers = list(range(12))

    hl_step5k = [199, 91, 90, 78, 68, 51, 53, 53, 48, 40, 37, 38]
    hl_step25k = [253, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    bars1 = ax1.bar(layers, hl_step5k, color=COLORS["slow"], alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Half-life (tokens)")
    ax1.set_title("Step 5,000 — Early Differentiation")
    ax1.set_xticks(layers)
    for bar, val in zip(bars1, hl_step5k):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(val), ha="center", va="bottom", fontsize=7)

    bars2 = ax2.bar(layers, hl_step25k, color=COLORS["accent"], alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Half-life (tokens)")
    ax2.set_title("Step 25,000 — Dramatic Specialization")
    ax2.set_xticks(layers)
    for bar, val in zip(bars2, hl_step25k):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(val), ha="center", va="bottom", fontsize=7)

    fig.suptitle("Learned Slow-Field Half-Life by Layer",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_ablation_comparison():
    """Ablation results at step 5K vs step 25K."""
    chunks = list(range(8))
    chunk_labels = [f"C{c}\n({c*128}-{(c+1)*128})" for c in chunks]

    # Step 5K ablation
    slow_delta_5k = [0.0000, -0.0005, -0.0006, -0.0006, -0.0007, 0.0001, -0.0013, -0.0008]
    all_delta_5k =  [0.0000, -0.0013, -0.0000, -0.0001, 0.0011, 0.0050, 0.0053, 0.0088]

    # Step 25K ablation
    slow_delta_25k = [0.0000, -0.0005, -0.0002, 0.0007, 0.0012, 0.0032, 0.0040, 0.0032]
    all_delta_25k =  [0.0000, -0.0010, -0.0002, 0.0008, 0.0012, 0.0057, 0.0063, 0.0086]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    x = np.arange(len(chunks))
    w = 0.35

    # Step 5K
    ax = axes[0]
    b1 = ax.bar(x - w/2, slow_delta_5k, w, label="Slow field Δ", color=COLORS["slow"], alpha=0.8)
    b2 = ax.bar(x + w/2, all_delta_5k, w, label="All fields Δ", color=COLORS["all"], alpha=0.8)
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(chunk_labels, fontsize=7)
    ax.set_ylabel("Loss delta (↑ = field helps)")
    ax.set_title("Step 5,000 — Slow field not yet contributing")
    ax.legend(fontsize=8)

    # Step 25K
    ax = axes[1]
    b1 = ax.bar(x - w/2, slow_delta_25k, w, label="Slow field Δ", color=COLORS["slow"], alpha=0.8)
    b2 = ax.bar(x + w/2, all_delta_25k, w, label="All fields Δ", color=COLORS["all"], alpha=0.8)
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(chunk_labels, fontsize=7)
    ax.set_title("Step 25,000 — Slow field now active in later chunks")
    ax.legend(fontsize=8)

    fig.suptitle("Per-Chunk Ablation: Impact of Removing Field Modulation",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_slow_field_share():
    """Show what fraction of total field effect comes from slow field at step 25K."""
    chunks = [3, 4, 5, 6, 7]
    chunk_labels = [f"Chunk {c}" for c in chunks]
    slow_pct = [85.8, 96.9, 56.0, 63.0, 36.7]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(chunk_labels, slow_pct, color=COLORS["accent"], alpha=0.85,
                  edgecolor="white", linewidth=1.5)
    ax.axhline(y=50, color="gray", ls="--", alpha=0.4, label="50% line")
    ax.set_ylabel("Slow field share of total field effect (%)")
    ax.set_title("Slow Field Contribution — Step 25,000")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, slow_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def page_samples(pdf):
    """Create text sample pages."""
    samples = [
        (1000, 26.7,
         "The meaning of life is that it is both the process of discovering the true "
         "value of life and the process of realizing itself. So I saw the difference in "
         "the difference between the two concepts and I think this's why, because it "
         "means that the concept of \"living\" comes from the process of discovering..."),
        (5000, 30.1,
         "The meaning of life is that we live in a world where we live in isolation, "
         "where we feel we have an obligation to live in a world where we feel that we "
         "are alone.\" The film is a reimagining of contemporary life in the North. The "
         "film is about how the British government has been able to control the..."),
        (10000, 29.4,
         "The meaning of life is to be alive. I have no idea how to explain it, but "
         "once I know what life will do for us, I'll be able to explain it in a way "
         "that is realistic. So, for those who might not already know, life is the "
         "most beautiful thing. For those who might not know, life is the most realistic..."),
        (17000, 28.6,
         "The meaning of life is not only simple but complex. It is also the reason "
         "why life is not a random state. It is what it is. It is not a random thing. "
         "It is like the Earth, it is what it is. It is not a random thing. It is just "
         "an ordinary thing. The Earth has been created by a random person, it is h..."),
        (22000, 28.0,
         "The meaning of life is not the point of having to live the life of a man, "
         "but a life that is the beginning of a life. The first and only possibility "
         "that life and death have for our ancestors. For my father to have the freedom "
         "to live in this life and to be there for it to come, was the first and on..."),
        (26000, 27.5,
         "The meaning of life is the meaning of freedom. At the end of the book, he "
         "describes the life of a man in a suit and tie. The phrase was used to describe "
         "the life of a man in a suit and tie. The meaning of life is the meaning of "
         "freedom. \"There are some that have difficulty with this,\" he says..."),
        (29000, 27.3,
         "The meaning of life is never really clear. The author does not necessarily "
         "represent The New American. He is an opinion writer for the magazine. "
         "I have been doing this for a while. I've been using the same method for the "
         "past few years, but this time I'm using the same method for differ..."),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    y = 0.95
    ax.text(0.5, y, "Generated Text Samples During Training",
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            ha="center", va="top")
    y -= 0.03
    ax.text(0.5, y, "Prompt: \"The meaning of life is...\"",
            transform=ax.transAxes, fontsize=11, fontstyle="italic",
            ha="center", va="top", color="#666666")
    y -= 0.04

    for step, ppl, text in samples:
        ax.text(0.05, y, f"Step {step:,}  (PPL {ppl})",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", color=COLORS["ms_val"])
        y -= 0.025
        wrapped = text
        ax.text(0.05, y, wrapped,
                transform=ax.transAxes, fontsize=8.5, va="top",
                wrap=True, family="serif",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                          edgecolor="#E0E0E0", alpha=0.9))
        y -= 0.115

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_summary(pdf):
    """Summary page with key findings."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    lines = [
        ("Multiscale Stigmergic Field — OWT 125M Training Report", 16, "bold", COLORS["ms_val"]),
        ("", 6, "normal", "black"),
        ("Architecture", 13, "bold", "#333"),
        ("• 125M parameter GPT with chunk-parallel stigmergic attention", 10, "normal", "#444"),
        ("• Dual-scale social field: fast (fixed evap) + slow (learnable retention)", 10, "normal", "#444"),
        ("• Warm-started from 60K-step single-field checkpoint on OpenWebText", 10, "normal", "#444"),
        ("", 6, "normal", "black"),
        ("Key Findings at Step 29,000 / 70,000", 13, "bold", "#333"),
        ("", 4, "normal", "black"),
        ("1. PPL surpassed single-field baseline", 11, "bold", COLORS["slow"]),
        ("   Multiscale PPL 27.3 vs single-field best 27.4 — and still improving", 10, "normal", "#444"),
        ("", 4, "normal", "black"),
        ("2. Slow field is actively contributing", 11, "bold", COLORS["slow"]),
        ("   At step 25K, removing slow field hurts chunks 3-7 (tokens 384-1024)", 10, "normal", "#444"),
        ("   Slow field accounts for 37-97% of total field effect in later chunks", 10, "normal", "#444"),
        ("", 4, "normal", "black"),
        ("3. Dramatic layer specialization", 11, "bold", COLORS["slow"]),
        ("   Layer 0 slow field: half-life 253 tokens (long-range memory)", 10, "normal", "#444"),
        ("   Layers 1-11: half-life collapsed to 1-5 tokens (local processing)", 10, "normal", "#444"),
        ("   Model self-organized: one long-memory layer + fast processing layers", 10, "normal", "#444"),
        ("", 4, "normal", "black"),
        ("4. Field effect grows with sequence position", 11, "bold", COLORS["slow"]),
        ("   Ablation delta: 0.0 at chunk 0 → +0.009 at chunk 7", 10, "normal", "#444"),
        ("   Confirms field carries useful context forward across chunks", 10, "normal", "#444"),
        ("", 4, "normal", "black"),
        ("5. Retention evolves during training", 11, "bold", COLORS["slow"]),
        ("   Layer 0 half-life: 200 → 261 tokens over 29K steps (still growing)", 10, "normal", "#444"),
        ("   Deeper layers collapsed from 200 → 1 token (learned to be local)", 10, "normal", "#444"),
        ("", 8, "normal", "black"),
        ("Training Status", 13, "bold", "#333"),
        ("• Step 29,000 / 70,000 (41% complete)", 10, "normal", "#444"),
        ("• Throughput: 74K tokens/sec on L40S", 10, "normal", "#444"),
        ("• ETA: ~6.5 hours remaining", 10, "normal", "#444"),
        ("• LR in cosine decay phase (4.1e-4, peak was 6e-4)", 10, "normal", "#444"),
    ]

    y = 0.95
    for text, size, weight, color in lines:
        if text == "":
            y -= size * 0.003
            continue
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
                fontweight=weight, va="top", color=color)
        y -= size * 0.0035

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ms_entries = load_jsonl("experiments/owt/owt_multiscale_log.jsonl")
    orig_entries = load_jsonl("experiments/owt/owt_social_log.jsonl")

    ms_train, ms_val = split_train_val(ms_entries)
    orig_train, orig_val = split_train_val(orig_entries)

    output_path = "experiments/owt/multiscale_training_report.pdf"

    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        page_summary(pdf)

        # Page 2: Loss curves
        fig = fig_loss_curves(ms_train, ms_val, orig_train, orig_val)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Retention evolution
        fig = fig_retention_evolution()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Retention by layer
        fig = fig_retention_by_layer()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5: Ablation comparison
        fig = fig_ablation_comparison()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 6: Slow field share
        fig = fig_slow_field_share()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 7: Sample texts
        page_samples(pdf)

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
