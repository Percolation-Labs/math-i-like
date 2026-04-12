"""Generate comprehensive analysis plots for the paper."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sip_sim.neural import get_device, get_tokenizer, load_model_from_checkpoint
from analyze import (tokenize, compute_position_loss,
                     capture_field_states, plot_position_loss,
                     plot_field_evolution, plot_field_pca,
                     plot_deposit_heatmap, plot_head_specialisation)


def plot_field_magnitude_timeseries(layer_fields, chunk_size, out_path):
    """Line plot of field magnitude per head over the sequence — cleaner than heatmap."""
    n_layers = len(layer_fields)
    n_heads = layer_fields[0][0].shape[1]

    # Pick layers 0, middle, last
    layers_to_plot = [0, n_layers // 2, n_layers - 1]

    fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(12, 3 * len(layers_to_plot)),
                             sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, n_heads))

    for ax_idx, li in enumerate(layers_to_plot):
        ax = axes[ax_idx]
        field_states = layer_fields[li]
        n_chunks = len(field_states)
        positions = [ci * chunk_size for ci in range(n_chunks)]

        for h in range(n_heads):
            mags = [fs[0, h].norm().item() for fs in field_states]
            ax.plot(positions, mags, color=colors[h], label=f"H{h}",
                    linewidth=2, marker="o", markersize=4)

        ax.set_ylabel(f"Layer {li}\n||field||")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=n_heads, fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Token position")
    axes[0].set_title("Field magnitude per head across sequence")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_deposit_selectivity(layer_deposits, token_strs, layer_idx, out_path):
    """For each head, show top-5 tokens by deposit magnitude — what does each head care about?"""
    deposits = layer_deposits[layer_idx]
    all_dep = torch.cat(deposits, dim=1)[0]  # (T, H, d_f)
    dep_mag = all_dep.norm(dim=-1)  # (T, H)
    T, H = dep_mag.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for h in range(min(H, 6)):
        ax = axes[h]
        mags = dep_mag[:, h].numpy()
        # Top tokens
        top_idx = np.argsort(mags)[::-1][:10]
        top_tokens = [token_strs[i] for i in top_idx]
        top_mags = [mags[i] for i in top_idx]

        bars = ax.barh(range(len(top_tokens)), top_mags, color=plt.cm.Blues(0.6))
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels([f"'{t.strip()}' (pos {i})" for t, i in zip(top_tokens, top_idx)],
                          fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("||deposit||")
        ax.set_title(f"Head {h} — top deposits")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle(f"What each head deposits most strongly — Layer {layer_idx}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_multi_text_position_loss(baseline_model, social_model, texts, enc, device, out_path):
    """Per-position loss comparison across many texts, with smoothing."""
    fig, axes = plt.subplots(len(texts), 1, figsize=(14, 3 * len(texts)), squeeze=False)

    for i, text in enumerate(texts):
        ids, strs = tokenize(text, enc)
        bl = compute_position_loss(baseline_model, ids, device)
        sl = compute_position_loss(social_model, ids, device)
        diff = [b - s for b, s in zip(bl, sl)]
        T = len(diff)

        ax = axes[i, 0]
        colors = ["#4CAF50" if d > 0 else "#F44336" for d in diff]
        ax.bar(range(T), diff, color=colors, alpha=0.7, width=0.9)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Running average
        if T > 5:
            window = min(5, T // 3)
            smoothed = np.convolve(diff, np.ones(window)/window, mode="valid")
            ax.plot(range(window//2, window//2 + len(smoothed)), smoothed,
                    color="black", linewidth=2, label=f"{window}-token avg")
            ax.legend(fontsize=8)

        ax.set_ylabel("B-S loss diff")
        title = text[:60] + "..." if len(text) > 60 else text
        ax.set_title(f'"{title}"', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Token labels if short enough
        if T <= 50:
            ax.set_xticks(range(T))
            ax.set_xticklabels(strs[1:], rotation=90, fontsize=5)

    axes[-1, 0].set_xlabel("Token position")
    plt.suptitle("Per-position loss: green = social field better", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_aggregate_position_buckets(baseline_model, social_model, texts, enc, device, out_path):
    """Aggregate loss difference bucketed by position, with error bars."""
    bucket_size = 5
    max_pos = 0

    all_results = []
    for text in texts:
        ids, _ = tokenize(text, enc)
        bl = compute_position_loss(baseline_model, ids, device)
        sl = compute_position_loss(social_model, ids, device)
        diff = [b - s for b, s in zip(bl, sl)]
        all_results.append(diff)
        max_pos = max(max_pos, len(diff))

    n_buckets = (max_pos + bucket_size - 1) // bucket_size
    bucket_vals = [[] for _ in range(n_buckets)]

    for diff in all_results:
        for pos, d in enumerate(diff):
            bucket_vals[pos // bucket_size].append(d)

    # Filter empty buckets
    valid = [(i, bv) for i, bv in enumerate(bucket_vals) if len(bv) >= 2]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = [i * bucket_size + bucket_size // 2 for i, _ in valid]
    means = [np.mean(bv) for _, bv in valid]
    sems = [np.std(bv) / np.sqrt(len(bv)) for _, bv in valid]

    colors = ["#4CAF50" if m > 0 else "#F44336" for m in means]
    ax.bar(x, means, width=bucket_size * 0.8, color=colors, alpha=0.7)
    ax.errorbar(x, means, yerr=sems, fmt="none", color="black", alpha=0.7, capsize=3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xlabel("Token position")
    ax.set_ylabel("Loss diff (baseline - social)\n+ve = social better")
    ax.set_title(f"Aggregate per-position advantage ({len(texts)} texts, "
                 f"mean +/- SEM)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Stats
    all_diff = [d for r in all_results for d in r]
    early = [d for r in all_results for i, d in enumerate(r) if i < 15]
    late = [d for r in all_results for i, d in enumerate(r) if i >= 15]
    print(f"  Overall: social better at {sum(1 for d in all_diff if d>0)}/{len(all_diff)} "
          f"({100*sum(1 for d in all_diff if d>0)/len(all_diff):.1f}%)")
    print(f"  Early (<15): {np.mean(early):.4f} +/- {np.std(early)/np.sqrt(len(early)):.4f}")
    print(f"  Late (>=15): {np.mean(late):.4f} +/- {np.std(late)/np.sqrt(len(late)):.4f}")
    return np.mean(early), np.mean(late)


def plot_entity_tracking(baseline_model, social_model, enc, device, out_path):
    """Texts specifically designed to test entity tracking — where field should help."""
    entity_texts = [
        "Tom had a red ball. Sara had a blue ball. Tom gave his ball to Sara. Now Sara had",
        "The key was on the table. The cat jumped on the table and knocked the key to the floor. The dog picked up the key and carried it to",
        "Alice baked cookies. Bob ate three cookies. Alice counted and found that she now had only five cookies left because Bob had eaten",
        "The wizard put the magic stone in the chest. The thief opened the chest and took the stone. The wizard returned and opened the chest but the stone was",
    ]

    fig, axes = plt.subplots(len(entity_texts), 1, figsize=(14, 3.5 * len(entity_texts)),
                             squeeze=False)

    for i, text in enumerate(entity_texts):
        ids, strs = tokenize(text, enc)
        bl = compute_position_loss(baseline_model, ids, device)
        sl = compute_position_loss(social_model, ids, device)
        diff = [b - s for b, s in zip(bl, sl)]
        T = len(diff)

        ax = axes[i, 0]
        colors = ["#4CAF50" if d > 0 else "#F44336" for d in diff]
        ax.bar(range(T), diff, color=colors, alpha=0.7, width=0.9)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("B-S diff")

        if T <= 60:
            ax.set_xticks(range(T))
            ax.set_xticklabels(strs[1:], rotation=90, fontsize=6)

        # Highlight the critical tokens (last few where entity state matters)
        for j in range(max(0, T-5), T):
            ax.get_children()[j].set_edgecolor("black")
            ax.get_children()[j].set_linewidth(1.5)

        title = text[:70] + "..." if len(text) > 70 else text
        ax.set_title(f'"{title}"', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Token position")
    plt.suptitle("Entity tracking: per-position loss on state-dependent completions\n"
                 "(green = social field better, outlined = critical tokens)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Print critical token stats
    print("  Critical token (last 5) advantages:")
    for text in entity_texts:
        ids, strs = tokenize(text, enc)
        bl = compute_position_loss(baseline_model, ids, device)
        sl = compute_position_loss(social_model, ids, device)
        diff = [b - s for b, s in zip(bl, sl)]
        crit = diff[-5:]
        print(f"    '{text[:40]}...': avg={np.mean(crit):.3f}")


def plot_field_across_sentences(social_model, enc, device, chunk_size, out_path):
    """Show field evolution on a long multi-sentence text with sentence boundaries marked."""
    text = ("The little girl found a golden key in the garden. "
            "She picked it up and put it in her pocket. "
            "Then she walked to the old house at the end of the road. "
            "She tried the key in the rusty lock and it opened with a click. "
            "Inside she found a room full of dusty old books. "
            "She picked up the biggest book and started to read. "
            "The book told the story of a brave knight who lost his sword. "
            "The knight searched for his sword in the dark forest.")

    ids, strs = tokenize(text, enc)
    T = len(ids)

    # Enable capture
    for block in social_model.blocks:
        if block.attn.use_field:
            block.attn._capture = True

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        social_model(idx)

    # Collect from all layers
    all_deposits = []
    for block in social_model.blocks:
        attn = block.attn
        if attn.use_field and hasattr(attn, "_deposits"):
            deps = torch.cat(attn._deposits, dim=1)[0]  # (T, H, d_f)
            all_deposits.append(deps)
            del attn._field_states, attn._deposits
            attn._capture = False

    if not all_deposits:
        return

    # Find sentence boundaries (periods)
    sent_bounds = [i for i, s in enumerate(strs) if "." in s]

    n_layers = len(all_deposits)
    n_heads = all_deposits[0].shape[1]

    # Average deposit magnitude across heads for each layer
    fig, axes = plt.subplots(n_layers, 1, figsize=(max(16, T * 0.12), 2 * n_layers),
                             sharex=True)

    for li in range(n_layers):
        ax = axes[li]
        dep_mag = all_deposits[li].norm(dim=-1)  # (T, H)

        # Plot each head as a line
        colors = plt.cm.Set2(np.linspace(0, 1, n_heads))
        for h in range(n_heads):
            ax.plot(range(T), dep_mag[:, h].numpy(), color=colors[h],
                    alpha=0.6, linewidth=1, label=f"H{h}" if li == 0 else None)

        # Mean across heads
        mean_mag = dep_mag.mean(dim=1).numpy()
        ax.plot(range(T), mean_mag, color="black", linewidth=2, label="mean" if li == 0 else None)

        # Sentence boundaries
        for sb in sent_bounds:
            ax.axvline(x=sb, color="red", linewidth=0.5, linestyle="--", alpha=0.5)

        # Chunk boundaries
        for ci in range(1, (T + chunk_size - 1) // chunk_size):
            ax.axvline(x=ci * chunk_size, color="blue", linewidth=0.5,
                       linestyle=":", alpha=0.3)

        ax.set_ylabel(f"L{li}")
        ax.grid(True, alpha=0.2)

    if T <= 100:
        axes[-1].set_xticks(range(T))
        axes[-1].set_xticklabels(strs, rotation=90, fontsize=5)
    axes[-1].set_xlabel("Token position")
    axes[0].legend(ncol=n_heads + 1, fontsize=7, loc="upper right")
    axes[0].set_title("Deposit magnitude across a multi-sentence story\n"
                      "(red dashed = sentence boundary, blue dotted = chunk boundary)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_loss_curves_from_logs(out_path):
    """Plot training loss curves from checkpoint logs."""
    logs = {}
    for name, path in [("baseline", "checkpoints/baseline_log.jsonl"),
                        ("social", "checkpoints/social_log.jsonl"),
                        ("baseline_1_8x", "checkpoints/baseline_1_8x_log.jsonl")]:
        if not os.path.exists(path):
            continue
        steps, losses, val_steps, val_losses = [], [], [], []
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                if "val_loss" in entry:
                    val_steps.append(entry["step"])
                    val_losses.append(entry["val_loss"])
                elif "loss" in entry:
                    steps.append(entry["step"])
                    losses.append(entry["loss"])
        logs[name] = (steps, losses, val_steps, val_losses)

    if not logs:
        print("No log files found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax = axes[0]
    colors = {"baseline": "#2196F3", "social": "#FF5722", "baseline_1_8x": "#4CAF50"}
    labels = {"baseline": "Baseline (30M)", "social": "Social field (31M)",
              "baseline_1_8x": "Baseline 1.8x (53M)"}
    for name in ["baseline", "social", "baseline_1_8x"]:
        if name not in logs:
            continue
        steps, losses, _, _ = logs[name]
        ax.plot(steps, losses, color=colors[name], label=labels[name],
                linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[1]
    for name in ["baseline", "social", "baseline_1_8x"]:
        if name not in logs:
            continue
        _, _, val_steps, val_losses = logs[name]
        ax.plot(val_steps, val_losses, color=colors[name], label=labels[name],
                linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("TinyStories training dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    device = get_device()
    enc = get_tokenizer()
    out_dir = "analysis"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading models...")
    baseline_model, _, _ = load_model_from_checkpoint("checkpoints/baseline_latest.pt", device)
    social_model, social_args, _ = load_model_from_checkpoint("checkpoints/social_latest.pt", device)
    chunk_size = social_args["chunk_size"]

    # 1. Training curves
    print("\n" + "="*60)
    print("1. Training curves")
    print("="*60)
    plot_loss_curves_from_logs(os.path.join(out_dir, "training_curves.png"))

    # 2. Multi-text per-position loss
    print("\n" + "="*60)
    print("2. Per-position loss comparison")
    print("="*60)
    texts = [
        "The dog found a shiny key hidden under the old tree. He picked it up and carried it to his friend the cat. The cat looked at the key and said she knew which door it opened.",
        "Once upon a time there was a little girl named Sara who loved to play in the garden with her dog Max and her cat Whiskers every single day after school.",
        "Tom found a mysterious key hidden under the old oak tree in the backyard so he decided to try every locked door in the house to see which one it would open.",
        "The brave knight rode his horse through the dark forest following the narrow path that the wise old woman had told him about when he visited her cottage yesterday.",
    ]
    plot_multi_text_position_loss(baseline_model, social_model, texts, enc, device,
                                  os.path.join(out_dir, "multi_text_loss.png"))

    # 3. Aggregate position analysis (many texts)
    print("\n" + "="*60)
    print("3. Aggregate position analysis")
    print("="*60)
    many_texts = texts + [
        "The little bird flew high above the clouds and looked down at the tiny houses below wondering what all the people were doing on this beautiful morning.",
        "Sara gave her favorite teddy bear to her friend Tom because he was feeling sad and she wanted to make him smile again like he always did for her.",
        "One day the sky turned a strange shade of green and all the animals in the forest started running because they knew something unusual was about to happen.",
        "In a small village by the sea there lived a fisherman who caught a golden fish that promised to grant him three wishes if he let it go.",
        "The old wizard opened his ancient book of spells and began to read carefully one by one until he found the enchantment he had been searching for.",
        "Lily had a beautiful garden full of flowers but one day she noticed that someone had been picking them without asking so she decided to find out who.",
        "The little mouse lived in a hole in the wall and every night he would sneak out to find cheese in the kitchen while the big cat was sleeping.",
        "There was a magic tree in the forest that grew different fruit every season and the children from the village would visit it to see what new fruit appeared.",
    ]
    plot_aggregate_position_buckets(baseline_model, social_model, many_texts, enc, device,
                                    os.path.join(out_dir, "aggregate_loss_12texts.png"))

    # 4. Entity tracking
    print("\n" + "="*60)
    print("4. Entity tracking analysis")
    print("="*60)
    plot_entity_tracking(baseline_model, social_model, enc, device,
                         os.path.join(out_dir, "entity_tracking.png"))

    # 5. Field evolution on longer text
    print("\n" + "="*60)
    print("5. Field evolution (multi-sentence)")
    print("="*60)
    long_text = ("The dog found a shiny key hidden under the old tree. "
                 "He picked it up and carried it to his friend the cat. "
                 "The cat looked at the key and said she knew which door it opened. "
                 "Together they walked to the big red house at the end of the street. "
                 "The cat tried the key in the lock and the door swung open.")
    ids, strs = tokenize(long_text, enc)
    layer_fields, layer_deposits = capture_field_states(social_model, ids, device)

    plot_field_evolution(layer_fields, strs, chunk_size,
                         os.path.join(out_dir, "field_evolution_long.png"))
    plot_field_magnitude_timeseries(layer_fields, chunk_size,
                                    os.path.join(out_dir, "field_timeseries.png"))

    # PCA for first, middle, last layer
    for li in [0, len(layer_fields)//2, len(layer_fields)-1]:
        plot_field_pca(layer_fields, li,
                       os.path.join(out_dir, f"field_pca_layer{li}_long.png"))

    # 6. Deposit analysis on longer text
    print("\n" + "="*60)
    print("6. Deposit analysis")
    print("="*60)
    for li in [0, len(layer_deposits)//2, len(layer_deposits)-1]:
        plot_deposit_heatmap(layer_deposits, li, strs, chunk_size,
                             os.path.join(out_dir, f"deposits_layer{li}_long.png"))
        plot_deposit_selectivity(layer_deposits, strs, li,
                                 os.path.join(out_dir, f"deposit_selectivity_layer{li}.png"))

    # 7. Head specialisation
    print("\n" + "="*60)
    print("7. Head specialisation")
    print("="*60)
    plot_head_specialisation(layer_deposits, strs, chunk_size,
                             os.path.join(out_dir, "head_specialisation_long.png"))

    # 8. Field across sentences (longest text)
    print("\n" + "="*60)
    print("8. Deposits across sentences")
    print("="*60)
    plot_field_across_sentences(social_model, enc, device, chunk_size,
                                os.path.join(out_dir, "deposits_across_sentences.png"))

    print(f"\n{'='*60}")
    print(f"All plots saved to {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
