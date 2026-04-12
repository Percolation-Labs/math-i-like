"""Analysis lab for social field models.

Visualises the stigmergic dynamics: field evolution, deposits,
per-position loss, and head specialisation.

Usage:
    python analyze.py --baseline checkpoints/baseline_latest.pt \
                      --social checkpoints/social_latest.pt \
                      --text "The dog found a shiny key and gave it to the cat"

    python analyze.py --social checkpoints/social_latest.pt --aggregate
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sip_sim.neural import get_tokenizer, load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text, enc):
    """Tokenize text, return (token_ids, token_strings)."""
    ids = enc.encode(text)
    strs = [enc.decode([t]) for t in ids]
    return ids, strs


# ---------------------------------------------------------------------------
# Per-position loss comparison
# ---------------------------------------------------------------------------

def compute_position_loss(model, token_ids, device):
    """Compute cross-entropy loss at each token position."""
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(idx)
    losses = []
    for t in range(len(token_ids) - 1):
        loss = F.cross_entropy(logits[0, t:t+1], idx[0, t+1:t+2])
        losses.append(loss.item())
    return losses


def plot_position_loss(baseline_losses, social_losses, token_strs, out_path):
    """Plot per-position loss for baseline vs social field."""
    T = len(baseline_losses)
    fig, axes = plt.subplots(2, 1, figsize=(max(14, T * 0.15), 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(range(T), baseline_losses, label="Baseline", color="#2196F3",
            alpha=0.8, linewidth=1.5)
    ax.plot(range(T), social_losses, label="Social field", color="#FF5722",
            alpha=0.8, linewidth=1.5)
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Per-position loss: baseline vs social field")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    diff = [b - s for b, s in zip(baseline_losses, social_losses)]
    colors = ["#4CAF50" if d > 0 else "#F44336" for d in diff]
    ax2.bar(range(T), diff, color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Baseline - Social\n(+ve = social better)")
    ax2.set_xlabel("Token position")
    ax2.grid(True, alpha=0.3)

    if T <= 80:
        ax2.set_xticks(range(T))
        ax2.set_xticklabels(token_strs[1:], rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    avg_diff = sum(diff) / len(diff)
    social_wins = sum(1 for d in diff if d > 0)
    print(f"  Average advantage (social): {avg_diff:.4f}")
    print(f"  Social wins at {social_wins}/{T} positions ({100*social_wins/T:.0f}%)")

    if T > 20:
        early = diff[:T//3]
        late = diff[2*T//3:]
        print(f"  Early positions (0-{T//3}): avg diff = {sum(early)/len(early):.4f}")
        print(f"  Late positions ({2*T//3}-{T}): avg diff = {sum(late)/len(late):.4f}")


# ---------------------------------------------------------------------------
# Field state evolution
# ---------------------------------------------------------------------------

def capture_field_states(model, token_ids, device):
    """Run model with diagnostics capture, return field states and deposits."""
    for block in model.blocks:
        if block.attn.use_field:
            block.attn._capture = True

    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(idx)

    layer_fields = []
    layer_deposits = []
    for block in model.blocks:
        attn = block.attn
        if attn.use_field and hasattr(attn, "_field_states"):
            layer_fields.append(attn._field_states)
            layer_deposits.append(attn._deposits)
            del attn._field_states, attn._deposits
            attn._capture = False

    return layer_fields, layer_deposits


def plot_field_evolution(layer_fields, token_strs, chunk_size, out_path):
    """Heatmap of field magnitude per head across chunks, for each layer."""
    n_layers = len(layer_fields)
    n_heads = layer_fields[0][0].shape[1]

    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 2.5 * n_layers),
                             squeeze=False)

    for layer_idx, field_states in enumerate(layer_fields):
        ax = axes[layer_idx, 0]
        n_chunks = len(field_states)
        magnitudes = np.zeros((n_heads, n_chunks))
        for ci, fs in enumerate(field_states):
            mag = fs[0].norm(dim=-1).numpy()
            magnitudes[:, ci] = mag

        im = ax.imshow(magnitudes, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest")
        ax.set_ylabel(f"Layer {layer_idx}\nHead")
        ax.set_yticks(range(n_heads))

        chunk_labels = [f"{ci*chunk_size}" for ci in range(n_chunks)]
        ax.set_xticks(range(n_chunks))
        ax.set_xticklabels(chunk_labels, fontsize=8)
        plt.colorbar(im, ax=ax, label="||field||")

    axes[-1, 0].set_xlabel("Position (chunk boundary)")
    axes[0, 0].set_title("Field magnitude per head across sequence")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_field_pca(layer_fields, layer_idx, out_path):
    """PCA projection of field states to show trajectory in 2D."""
    field_states = layer_fields[layer_idx]
    n_heads = field_states[0].shape[1]
    n_chunks = len(field_states)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for h in range(min(n_heads, 6)):
        ax = axes[h]
        vecs = np.stack([fs[0, h].numpy() for fs in field_states])

        if vecs.shape[0] < 2:
            continue

        centered = vecs - vecs.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T

        for i in range(len(proj) - 1):
            ax.annotate("", xy=proj[i+1], xytext=proj[i],
                       arrowprops=dict(arrowstyle="->", color=plt.cm.viridis(i/len(proj)),
                                       lw=1.5))
        ax.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)),
                  cmap="viridis", s=40, zorder=5)
        ax.scatter(proj[0, 0], proj[0, 1], c="green", s=100, marker="o",
                  zorder=6, label="start")
        ax.scatter(proj[-1, 0], proj[-1, 1], c="red", s=100, marker="s",
                  zorder=6, label="end")
        ax.set_title(f"Head {h}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Field trajectory (PCA) — Layer {layer_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Deposit analysis
# ---------------------------------------------------------------------------

def plot_deposit_heatmap(layer_deposits, layer_idx, token_strs, chunk_size, out_path):
    """Show what each token deposits into the field, per head."""
    deposits = layer_deposits[layer_idx]
    n_heads = deposits[0].shape[2]

    all_deposits = torch.cat(deposits, dim=1)[0]
    dep_mag = all_deposits.norm(dim=-1).numpy()
    T = dep_mag.shape[0]

    fig, ax = plt.subplots(figsize=(max(14, T * 0.15), 4))
    im = ax.imshow(dep_mag.T, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_ylabel("Head")
    ax.set_yticks(range(n_heads))
    ax.set_xlabel("Token position")

    for ci in range(1, (T + chunk_size - 1) // chunk_size):
        ax.axvline(x=ci * chunk_size - 0.5, color="red", linewidth=0.5,
                   linestyle="--", alpha=0.5)

    if T <= 80:
        ax.set_xticks(range(T))
        ax.set_xticklabels(token_strs[:T], rotation=90, fontsize=6)

    plt.colorbar(im, ax=ax, label="||deposit||")
    ax.set_title(f"Deposit magnitude per token per head — Layer {layer_idx}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Head specialisation
# ---------------------------------------------------------------------------

def plot_head_specialisation(layer_deposits, token_strs, chunk_size, out_path):
    """Cosine similarity between heads' deposit patterns."""
    n_layers = len(layer_deposits)

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]

    for layer_idx, deposits in enumerate(layer_deposits):
        all_dep = torch.cat(deposits, dim=1)[0]
        n_heads = all_dep.shape[1]

        avg_dep = all_dep.mean(dim=0)
        avg_dep = F.normalize(avg_dep, dim=-1)
        sim = (avg_dep @ avg_dep.T).numpy()

        ax = axes[layer_idx]
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Head")
        ax.set_ylabel("Head")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_heads))
        plt.colorbar(im, ax=ax, label="cosine sim")

    plt.suptitle("Head deposit specialisation (cosine similarity)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Dashboard: all plots for one text
# ---------------------------------------------------------------------------

def run_analysis(baseline_path, social_path, text, out_dir, device):
    """Run full analysis suite on a text."""
    os.makedirs(out_dir, exist_ok=True)
    enc = get_tokenizer()
    token_ids, token_strs = tokenize(text, enc)
    T = len(token_ids)
    print(f"Text: {text[:80]}...")
    print(f"Tokens: {T}")

    if baseline_path and social_path:
        print("\n=== Per-position loss ===")
        baseline_model, _, _ = load_model_from_checkpoint(baseline_path, device)
        social_model, social_args, _ = load_model_from_checkpoint(social_path, device)

        baseline_losses = compute_position_loss(baseline_model, token_ids, device)
        social_losses = compute_position_loss(social_model, token_ids, device)
        plot_position_loss(baseline_losses, social_losses, token_strs,
                          os.path.join(out_dir, "position_loss.png"))
        del baseline_model
    elif social_path:
        social_model, social_args, _ = load_model_from_checkpoint(social_path, device)
    else:
        print("Need at least --social checkpoint")
        return

    print("\n=== Field state evolution ===")
    chunk_size = social_args["chunk_size"]
    layer_fields, layer_deposits = capture_field_states(
        social_model, token_ids, device)

    plot_field_evolution(layer_fields, token_strs, chunk_size,
                        os.path.join(out_dir, "field_evolution.png"))

    for li in [0, len(layer_fields) - 1]:
        plot_field_pca(layer_fields, li,
                      os.path.join(out_dir, f"field_pca_layer{li}.png"))

    print("\n=== Deposit analysis ===")
    for li in [0, len(layer_deposits) - 1]:
        plot_deposit_heatmap(layer_deposits, li, token_strs, chunk_size,
                            os.path.join(out_dir, f"deposits_layer{li}.png"))

    print("\n=== Head specialisation ===")
    plot_head_specialisation(layer_deposits, token_strs, chunk_size,
                            os.path.join(out_dir, "head_specialisation.png"))

    print(f"\nAll plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Multi-text analysis for aggregate stats
# ---------------------------------------------------------------------------

def run_aggregate_analysis(baseline_path, social_path, out_dir, device):
    """Run per-position loss on many texts, aggregate by position bucket."""
    os.makedirs(out_dir, exist_ok=True)
    enc = get_tokenizer()

    texts = [
        "Once upon a time there was a little girl named Sara who loved to play in the garden with her dog Max and her cat Whiskers every single day after school",
        "The old wizard opened his ancient book of spells and began to read the words carefully one by one until he found the enchantment he had been searching for all these years",
        "Tom found a mysterious key hidden under the old oak tree in the backyard so he decided to try every locked door in the house to see which one it would open",
        "The little bird flew high above the clouds and looked down at the tiny houses and cars below wondering what all the people were doing on this beautiful sunny morning",
        "Sara gave her favorite teddy bear to her friend Tom because he was feeling sad and she wanted to make him smile again like he always did for her when she was upset",
        "One day the sky turned a strange shade of green and all the animals in the forest started running because they knew that something very unusual was about to happen",
        "The brave knight rode his horse through the dark forest following the narrow path that the wise old woman had told him about when he visited her cottage yesterday",
        "In a small village by the sea there lived a fisherman who caught a golden fish that promised to grant him three wishes if he would let it go back into the water",
    ]

    print("=== Aggregate per-position loss analysis ===")
    baseline_model, _, _ = load_model_from_checkpoint(baseline_path, device)
    social_model, _, _ = load_model_from_checkpoint(social_path, device)

    all_diff = []
    bucket_size = 10
    max_pos = max(len(tokenize(t, enc)[0]) for t in texts)
    n_buckets = (max_pos + bucket_size - 1) // bucket_size
    bucket_diffs = [[] for _ in range(n_buckets)]

    for text in texts:
        ids, _ = tokenize(text, enc)
        bl = compute_position_loss(baseline_model, ids, device)
        sl = compute_position_loss(social_model, ids, device)
        for pos, (b, s) in enumerate(zip(bl, sl)):
            d = b - s
            all_diff.append(d)
            bucket_diffs[pos // bucket_size].append(d)

    fig, ax = plt.subplots(figsize=(12, 5))
    means = [np.mean(bd) if bd else 0 for bd in bucket_diffs]
    stds = [np.std(bd) if bd else 0 for bd in bucket_diffs]
    x = [i * bucket_size + bucket_size // 2 for i in range(n_buckets)]

    colors = ["#4CAF50" if m > 0 else "#F44336" for m in means]
    ax.bar(x, means, width=bucket_size * 0.8, color=colors, alpha=0.7)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", alpha=0.5, capsize=3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Loss difference (baseline - social)\n+ve = social better")
    ax.set_title(f"Aggregate per-position advantage ({len(texts)} texts)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "aggregate_position_loss.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    avg = np.mean(all_diff)
    social_wins = sum(1 for d in all_diff if d > 0)
    total = len(all_diff)
    print(f"  Overall: social better at {social_wins}/{total} positions "
          f"({100*social_wins/total:.1f}%)")
    print(f"  Average advantage: {avg:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze social field models")
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--social", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--out-dir", type=str, default="analysis")
    args = parser.parse_args()

    device = get_device()

    if args.aggregate and args.baseline:
        run_aggregate_analysis(args.baseline, args.social, args.out_dir, device)
    elif args.text:
        run_analysis(args.baseline, args.social, args.text, args.out_dir, device)
    else:
        text = ("The dog found a shiny key hidden under the old tree. "
                "He picked it up and carried it to his friend the cat. "
                "The cat looked at the key and said she knew which door it opened. "
                "Together they walked to the big red house at the end of the street.")
        run_analysis(args.baseline, args.social, text, args.out_dir, device)


if __name__ == "__main__":
    main()
