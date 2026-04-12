"""
Learnability sweep: micro-GPT over (dropout × weight_decay).

Produces phase diagrams analogous to the ant trail (noise × evaporation) sweeps.
Tests whether C_ST = structure × plasticity predicts generalization.

Task: in-context function learning. Each sequence shows examples of a hidden
function f(x) = (a*x + b) mod V, and the model must predict f on a new input.
This requires both structure (learning what functions look like) and plasticity
(adapting to each new function per sequence).

Architecture follows Karpathy's micro-GPT spirit: minimal, no bias, RMSNorm,
ReLU. Scaled to ~60K params (2 layers, 4 heads, d_model=48).

Usage:
    uv run python learn/learnability_sweep.py              # full sweep
    uv run python learn/learnability_sweep.py --single      # single diagnostic run
"""

import time
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent

# ── Hyperparameters ─────────────────────────────────────────────

VOCAB_SIZE = 16          # numbers 0-15
N_EXAMPLES = 8           # in-context examples per sequence
SEQ_LEN = 2 * (N_EXAMPLES + 1)  # 18 tokens: x1 y1 x2 y2 ... x9 y9
D_MODEL = 48
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 64
BASE_LR = 3e-3
TRAIN_STEPS = 2000
MEASURE_EVERY = 100      # measure observables every N steps
AUTOCORR_LAG = 200       # steps between attention snapshots for autocorrelation
SEED = 42

# Sweep grid
DROPOUT_VALUES = np.linspace(0.0, 0.55, 10)
WD_VALUES = np.linspace(0.0, 0.50, 10)

# Function train/test split
N_TRAIN_FUNCS = 200
N_TEST_FUNCS = 40


# ── Model ───────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )
        self.last_attn = None  # instrumentation: saved attention weights

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, max_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, dropout, max_len) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def get_attention_maps(self):
        """Return list of attention tensors (B, n_head, T, T) per layer."""
        return [block.attn.last_attn for block in self.blocks]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data generation ─────────────────────────────────────────────

def make_functions(vocab_size, seed=0):
    """Generate all (a, b) pairs for f(x) = (a*x + b) mod V, a != 0.

    Returns (train_funcs, test_funcs) as lists of (a, b) tuples.
    """
    rng = np.random.default_rng(seed)
    all_funcs = [(a, b) for a in range(1, vocab_size) for b in range(vocab_size)]
    rng.shuffle(all_funcs)
    return all_funcs[:N_TRAIN_FUNCS], all_funcs[N_TRAIN_FUNCS:N_TRAIN_FUNCS + N_TEST_FUNCS]


def generate_batch(batch_size, n_examples, vocab_size, functions, rng):
    """Generate in-context function learning sequences.

    Sequence: [x1, y1, x2, y2, ..., x_{n+1}, y_{n+1}]
    Length: 2 * (n_examples + 1) = 18

    Returns:
        input_ids: (B, SEQ_LEN-1) — all tokens except last
        targets:   (B, SEQ_LEN-1) — shifted by 1, with -100 at x-prediction positions
        last_targets: (B,) — y_{n+1} values for final accuracy measurement
    """
    B = batch_size
    n_total = n_examples + 1
    full_len = 2 * n_total  # 18

    tokens = np.zeros((B, full_len), dtype=np.int64)

    for b in range(B):
        fi = rng.integers(len(functions))
        a, b_coeff = functions[fi]
        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            y = (a * x + b_coeff) % vocab_size
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = y

    # Standard causal LM: input = tokens[:-1], target = tokens[1:]
    input_ids = tokens[:, :-1]   # (B, 17)
    shifted = tokens[:, 1:]      # (B, 17)

    # Mask x-prediction positions (odd indices in shifted = even indices in input after first)
    # In shifted: position 0 = y1, pos 1 = x2, pos 2 = y2, pos 3 = x3, ...
    # y-positions in shifted: 0, 2, 4, 6, 8, 10, 12, 14, 16
    # x-positions in shifted: 1, 3, 5, 7, 9, 11, 13, 15
    targets = shifted.copy()
    for i in range(1, full_len - 1, 2):  # odd indices
        targets[:, i] = -100

    # Last y-prediction for accuracy
    last_targets = tokens[:, -1]  # y_{n+1}

    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


# ── Instrumentation ─────────────────────────────────────────────

def effective_rank(model):
    """Participation ratio of singular values across weight matrices.

    PR = (sum sigma_i)^2 / sum(sigma_i^2), normalized by min dimension.
    Returns mean across all weight matrices.
    """
    ranks = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and param.shape[0] >= 4 and param.shape[1] >= 4:
            with torch.no_grad():
                s = torch.linalg.svdvals(param)
                s_sum = s.sum()
                s_sq_sum = (s ** 2).sum()
                if s_sq_sum > 1e-10:
                    pr = (s_sum ** 2 / s_sq_sum).item()
                    max_rank = min(param.shape)
                    ranks.append(pr / max_rank)  # normalized to [0, 1]
    return float(np.mean(ranks)) if ranks else 0.0


def attention_entropy(model):
    """Mean entropy of attention distributions across all heads and layers."""
    entropies = []
    for attn_map in model.get_attention_maps():
        if attn_map is None:
            continue
        # attn_map: (B, n_head, T, T)
        # Entropy per position per head: -sum(p * log(p))
        p = attn_map.clamp(min=1e-10)
        ent = -(p * p.log()).sum(dim=-1)  # (B, n_head, T)
        entropies.append(ent.mean().item())
    return float(np.mean(entropies)) if entropies else 0.0


def attention_pattern_flat(model, input_ids):
    """Run model on input and return flattened attention pattern for correlation."""
    model.eval()
    with torch.no_grad():
        model(input_ids)
        maps = model.get_attention_maps()
        flat = torch.cat([m.mean(dim=0).reshape(-1) for m in maps if m is not None])
    model.train()
    return flat.cpu().numpy()


def pattern_correlation(a, b):
    """Pearson correlation between two flattened patterns."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2))
    if denom < 1e-10:
        return 1.0
    return float(np.sum(a * b) / denom)


# ── Training ────────────────────────────────────────────────────

def train_one(dropout, weight_decay, train_funcs, test_funcs, probe_batch,
              train_steps=TRAIN_STEPS, verbose=False):
    """Train one model, return measurements dict."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model = MicroGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=weight_decay, betas=(0.85, 0.99))

    # Linear LR decay (following Karpathy)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps
    )

    # Storage for measurements
    measurements = {
        "steps": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "eff_rank": [],
        "attn_entropy": [],
        "attn_autocorr": [],
    }

    # Attention pattern history for autocorrelation
    prev_pattern = None
    prev_pattern_step = -AUTOCORR_LAG

    probe_input = probe_batch[0]  # fixed probe for attention autocorrelation

    for step in range(train_steps):
        model.train()
        input_ids, targets, last_tgt = generate_batch(
            BATCH_SIZE, N_EXAMPLES, VOCAB_SIZE, train_funcs, rng
        )

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── Measure ──
        if step % MEASURE_EVERY == 0 or step == train_steps - 1:
            model.eval()
            with torch.no_grad():
                # Train accuracy (last position)
                pred = logits[:, -1].argmax(dim=-1)
                train_acc = (pred == last_tgt).float().mean().item()

                # Validation accuracy (held-out functions)
                val_in, val_tgt_full, val_last = generate_batch(
                    256, N_EXAMPLES, VOCAB_SIZE, test_funcs, np.random.default_rng(0)
                )
                val_logits = model(val_in)
                val_pred = val_logits[:, -1].argmax(dim=-1)
                val_acc = (val_pred == val_last).float().mean().item()

                # Effective rank
                er = effective_rank(model)

                # Attention entropy
                ae = attention_entropy(model)

                # Attention autocorrelation
                curr_pattern = attention_pattern_flat(model, probe_input)
                if prev_pattern is not None and (step - prev_pattern_step) >= AUTOCORR_LAG:
                    ac = pattern_correlation(curr_pattern, prev_pattern)
                    prev_pattern = curr_pattern
                    prev_pattern_step = step
                elif prev_pattern is None:
                    ac = 0.0
                    prev_pattern = curr_pattern
                    prev_pattern_step = step
                else:
                    ac = measurements["attn_autocorr"][-1] if measurements["attn_autocorr"] else 0.0

            measurements["steps"].append(step)
            measurements["train_loss"].append(loss.item())
            measurements["train_acc"].append(train_acc)
            measurements["val_acc"].append(val_acc)
            measurements["eff_rank"].append(er)
            measurements["attn_entropy"].append(ae)
            measurements["attn_autocorr"].append(ac)

            if verbose:
                print(f"  step {step:4d}  loss={loss.item():.3f}  "
                      f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                      f"rank={er:.3f}  entropy={ae:.3f}  autocorr={ac:.3f}")

            model.train()

    return measurements


# ── Sweep ───────────────────────────────────────────────────────

def run_sweep():
    print(f"Micro-GPT learnability sweep")
    print(f"Model: {N_LAYER} layers, {N_HEAD} heads, d={D_MODEL}")

    # Create model once to count params
    tmp = MicroGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER, 0.0)
    print(f"Parameters: {tmp.count_params():,}")
    del tmp

    print(f"Task: f(x) = (ax+b) mod {VOCAB_SIZE}, {N_EXAMPLES} in-context examples")
    print(f"Sweep: {len(DROPOUT_VALUES)} dropout x {len(WD_VALUES)} weight_decay "
          f"= {len(DROPOUT_VALUES) * len(WD_VALUES)} runs")
    print(f"Training: {TRAIN_STEPS} steps, batch_size={BATCH_SIZE}, lr={BASE_LR}")
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    print(f"Functions: {len(train_funcs)} train, {len(test_funcs)} test")

    # Fixed probe batch for attention autocorrelation (same across all runs)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    n_do = len(DROPOUT_VALUES)
    n_wd = len(WD_VALUES)
    n_total = n_do * n_wd

    # Result grids (using final measurements)
    val_acc_grid = np.zeros((n_wd, n_do))
    train_acc_grid = np.zeros((n_wd, n_do))
    eff_rank_grid = np.zeros((n_wd, n_do))
    attn_autocorr_grid = np.zeros((n_wd, n_do))
    attn_entropy_grid = np.zeros((n_wd, n_do))
    composite_grid = np.zeros((n_wd, n_do))

    t0 = time.perf_counter()

    for wi, wd in enumerate(WD_VALUES):
        for di, do_rate in enumerate(DROPOUT_VALUES):
            idx = wi * n_do + di + 1
            m = train_one(do_rate, wd, train_funcs, test_funcs, probe_batch)

            # Use mean of last 3 measurements for stability
            def last_mean(key, n=3):
                vals = m[key][-n:]
                return float(np.mean(vals)) if vals else 0.0

            va = last_mean("val_acc")
            ta = last_mean("train_acc")
            er = last_mean("eff_rank")
            ac = last_mean("attn_autocorr")
            ae = last_mean("attn_entropy")

            # Composite: structure × plasticity
            # structure = effective rank (normalized, higher = more structured)
            # plasticity = 1 - attention autocorrelation (lower autocorr = more plastic)
            cst = er * (1.0 - ac)

            val_acc_grid[wi, di] = va
            train_acc_grid[wi, di] = ta
            eff_rank_grid[wi, di] = er
            attn_autocorr_grid[wi, di] = ac
            attn_entropy_grid[wi, di] = ae
            composite_grid[wi, di] = cst

            elapsed = time.perf_counter() - t0
            eta = (elapsed / idx) * (n_total - idx)
            print(f"  [{idx:3d}/{n_total}] do={do_rate:.2f} wd={wd:.2f}  "
                  f"val={va:.3f} rank={er:.3f} ac={ac:.3f} C={cst:.3f}  "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s")

    # ── Find peaks ──────────────────────────────────────────────
    comp_peak = np.unravel_index(np.argmax(composite_grid), composite_grid.shape)
    val_peak = np.unravel_index(np.argmax(val_acc_grid), val_acc_grid.shape)

    peak_wd_c = WD_VALUES[comp_peak[0]]
    peak_do_c = DROPOUT_VALUES[comp_peak[1]]
    peak_wd_v = WD_VALUES[val_peak[0]]
    peak_do_v = DROPOUT_VALUES[val_peak[1]]

    print(f"Composite C_ST peak: dropout={peak_do_c:.2f}, wd={peak_wd_c:.2f}, "
          f"value={composite_grid[comp_peak]:.3f}")
    print(f"Val accuracy peak:   dropout={peak_do_v:.2f}, wd={peak_wd_v:.2f}, "
          f"value={val_acc_grid[val_peak]:.3f}")

    # ── Visualize ───────────────────────────────────────────────
    extent = [DROPOUT_VALUES[0], DROPOUT_VALUES[-1], WD_VALUES[0], WD_VALUES[-1]]

    # Triptych: effective rank, attention autocorrelation, composite
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = axes[0].imshow(eff_rank_grid, origin="lower", aspect="auto",
                          extent=extent, cmap="magma", vmin=0)
    axes[0].set_title("Effective Rank\n(representational structure)", fontsize=12)
    axes[0].set_xlabel("Dropout")
    axes[0].set_ylabel("Weight Decay")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(attn_autocorr_grid, origin="lower", aspect="auto",
                          extent=extent, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[1].set_title("Attention Autocorrelation\n(representational persistence)", fontsize=12)
    axes[1].set_xlabel("Dropout")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(composite_grid, origin="lower", aspect="auto",
                          extent=extent, cmap="inferno", vmin=0)
    axes[2].set_title("C_ST = rank × (1 - autocorr)\n(learnability)", fontsize=12)
    axes[2].set_xlabel("Dropout")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.plot(peak_do_c, peak_wd_c, "w*", markersize=15, markeredgecolor="k",
                label="C_ST peak")
        ax.plot(peak_do_v, peak_wd_v, "cs", markersize=10, markeredgecolor="k",
                label="Val acc peak")

    axes[2].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Micro-GPT Phase Diagram — Dropout × Weight Decay\n"
                 f"C_ST peak: do={peak_do_c:.2f}, wd={peak_wd_c:.2f}  |  "
                 f"Val peak: do={peak_do_v:.2f}, wd={peak_wd_v:.2f}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_triptych_gpt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: heatmap_triptych_gpt.png")

    # Validation accuracy heatmap (the ground truth)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(val_acc_grid, origin="lower", aspect="auto",
                          extent=extent, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Validation Accuracy\n(generalization)", fontsize=12)
    axes[0].set_xlabel("Dropout")
    axes[0].set_ylabel("Weight Decay")
    plt.colorbar(im0, ax=axes[0])
    axes[0].plot(peak_do_v, peak_wd_v, "w*", markersize=15, markeredgecolor="k")

    im1 = axes[1].imshow(train_acc_grid - val_acc_grid, origin="lower", aspect="auto",
                          extent=extent, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[1].set_title("Generalization Gap\n(train - val accuracy)", fontsize=12)
    axes[1].set_xlabel("Dropout")
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle("Micro-GPT: Generalization Landscape", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_generalization_gpt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: heatmap_generalization_gpt.png")

    # 3D surface of composite
    D, W = np.meshgrid(DROPOUT_VALUES, WD_VALUES)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(D, W, composite_grid, cmap="inferno",
                           edgecolor="k", linewidth=0.2, alpha=0.9)
    ax.set_xlabel("Dropout", fontsize=12, labelpad=10)
    ax.set_ylabel("Weight Decay", fontsize=12, labelpad=10)
    ax.set_zlabel("C_ST (learnability)", fontsize=12, labelpad=10)
    ax.set_title("Spatiotemporal Complexity — Micro-GPT\n"
                 f"C_ST = effective_rank × (1 - attn_autocorrelation)\n"
                 f"Peak: do={peak_do_c:.2f}, wd={peak_wd_c:.2f}",
                 fontsize=13, pad=20)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="C_ST")
    ax.view_init(elev=30, azim=225)
    fig.savefig(OUT_DIR / "surface_cst_gpt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: surface_cst_gpt.png")

    # Scatter: C_ST vs validation accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    cst_flat = composite_grid.ravel()
    val_flat = val_acc_grid.ravel()
    ax.scatter(cst_flat, val_flat, c=val_flat, cmap="viridis", edgecolors="k",
               linewidths=0.5, s=50)
    ax.set_xlabel("C_ST (learnability parameter)", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("Does C_ST predict generalization?", fontsize=13)

    # Correlation
    corr = np.corrcoef(cst_flat, val_flat)[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_cst_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: scatter_cst_vs_accuracy.png (r={corr:.3f})")

    # Save CSV
    csv_path = OUT_DIR / "sweep_results_gpt.csv"
    with open(csv_path, "w") as f:
        f.write("dropout,weight_decay,val_acc,train_acc,eff_rank,"
                "attn_autocorr,attn_entropy,composite\n")
        for wi, wd in enumerate(WD_VALUES):
            for di, do_rate in enumerate(DROPOUT_VALUES):
                f.write(f"{do_rate:.4f},{wd:.4f},{val_acc_grid[wi,di]:.4f},"
                        f"{train_acc_grid[wi,di]:.4f},{eff_rank_grid[wi,di]:.4f},"
                        f"{attn_autocorr_grid[wi,di]:.4f},"
                        f"{attn_entropy_grid[wi,di]:.4f},"
                        f"{composite_grid[wi,di]:.4f}\n")
    print(f"Saved: {csv_path.name}")


# ── Single diagnostic run ───────────────────────────────────────

def run_single():
    """Single training run with detailed time-series output."""
    print("Single diagnostic run (dropout=0.10, weight_decay=0.05)")

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    model = MicroGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER, 0.0)
    print(f"Parameters: {model.count_params():,}")
    del model

    m = train_one(
        dropout=0.10, weight_decay=0.05,
        train_funcs=train_funcs, test_funcs=test_funcs,
        probe_batch=probe_batch, train_steps=5000, verbose=True,
    )

    # Plot time series
    steps = m["steps"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Panel 1: accuracy
    axes[0].plot(steps, m["train_acc"], "b-", label="Train accuracy", alpha=0.8)
    axes[0].plot(steps, m["val_acc"], "r-", label="Val accuracy", alpha=0.8)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_title("Learning Dynamics — Single Run (do=0.10, wd=0.05)")

    # Panel 2: components
    axes[1].plot(steps, m["eff_rank"], "g-", label="Effective rank (S)", alpha=0.8)
    axes[1].plot(steps, [1 - a for a in m["attn_autocorr"]], "m-",
                 label="1 - Attn autocorr (P)", alpha=0.8)
    axes[1].set_ylabel("Component value")
    axes[1].legend()

    # Panel 3: composite
    cst = [s * (1 - a) for s, a in zip(m["eff_rank"], m["attn_autocorr"])]
    axes[2].plot(steps, cst, "k-", linewidth=2, label="C_ST = S × P")
    axes[2].set_ylabel("C_ST")
    axes[2].set_xlabel("Training step")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "timeseries_single_run.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: timeseries_single_run.png")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Single diagnostic run")
    args = parser.parse_args()

    if args.single:
        run_single()
    else:
        run_sweep()
