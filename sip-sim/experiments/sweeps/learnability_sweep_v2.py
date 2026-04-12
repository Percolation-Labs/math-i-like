"""
Learnability sweep v2: fixes from v1 failure analysis.

v1 problem: attention autocorrelation saturated at ~0.99 for ALL settings by
end of training. C_ST had zero variance → no predictive power.

v2 fixes:
  1. Cumulative C_ST — integral over training, not end snapshot
  2. Weight velocity as P — ‖ΔW‖/‖W‖ per measurement interval
  3. Finer time resolution — measure every 50 steps
  4. Longer training — 5000 steps to see post-freeze dynamics
  5. Multiple P candidates compared: weight velocity, attention autocorr,
     gradient norm, loss curvature (Δloss variance)

Also computes:
  - C_cumul = Σ S(t) × P(t) over all measurement points
  - Peak C_ST and time-of-peak
  - "Learning window" = steps where C_ST > 50% of its peak

Usage:
    uv run python learn/learnability_sweep_v2.py
"""

import time
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

VOCAB_SIZE = 16
N_EXAMPLES = 8
SEQ_LEN = 2 * (N_EXAMPLES + 1)  # 18
D_MODEL = 48
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 64
BASE_LR = 3e-3
TRAIN_STEPS = 5000        # longer than v1 (was 2000)
MEASURE_EVERY = 50        # finer than v1 (was 100)
SEED = 42

# Sweep grid — 12x12
DROPOUT_VALUES = np.linspace(0.0, 0.55, 12)
WD_VALUES = np.linspace(0.0, 0.50, 12)

N_TRAIN_FUNCS = 200
N_TEST_FUNCS = 40


# ── Model (identical to v1) ─────────────────────────────────────

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
        self.last_attn = None

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
        return [block.attn.last_attn for block in self.blocks]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data generation (identical to v1) ───────────────────────────

def make_functions(vocab_size, seed=0):
    rng = np.random.default_rng(seed)
    all_funcs = [(a, b) for a in range(1, vocab_size) for b in range(vocab_size)]
    rng.shuffle(all_funcs)
    return all_funcs[:N_TRAIN_FUNCS], all_funcs[N_TRAIN_FUNCS:N_TRAIN_FUNCS + N_TEST_FUNCS]


def generate_batch(batch_size, n_examples, vocab_size, functions, rng):
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)
    for b in range(batch_size):
        fi = rng.integers(len(functions))
        a, b_coeff = functions[fi]
        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            y = (a * x + b_coeff) % vocab_size
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = y
    input_ids = tokens[:, :-1]
    shifted = tokens[:, 1:]
    targets = shifted.copy()
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]
    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


# ── Instrumentation (v2: multiple P candidates) ────────────────

def effective_rank(model):
    """Participation ratio of SVD across weight matrices, normalized."""
    ranks = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and param.shape[0] >= 4 and param.shape[1] >= 4:
            with torch.no_grad():
                s = torch.linalg.svdvals(param)
                s_sum = s.sum()
                s_sq_sum = (s ** 2).sum()
                if s_sq_sum > 1e-10:
                    pr = (s_sum ** 2 / s_sq_sum).item()
                    ranks.append(pr / min(param.shape))
    return float(np.mean(ranks)) if ranks else 0.0


def weight_snapshot(model):
    """Flat vector of all parameters for velocity computation."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def weight_velocity(w_now, w_prev):
    """Relative weight change: ‖ΔW‖ / ‖W‖."""
    if w_prev is None:
        return 1.0
    delta = (w_now - w_prev).norm().item()
    norm = w_now.norm().item()
    return delta / max(norm, 1e-10)


def attention_entropy(model):
    """Mean attention entropy across heads and layers."""
    entropies = []
    for attn_map in model.get_attention_maps():
        if attn_map is None:
            continue
        p = attn_map.clamp(min=1e-10)
        ent = -(p * p.log()).sum(dim=-1)
        entropies.append(ent.mean().item())
    return float(np.mean(entropies)) if entropies else 0.0


def attention_pattern_flat(model, input_ids):
    """Run probe batch, return flattened attention for correlation."""
    model.eval()
    with torch.no_grad():
        model(input_ids)
        maps = model.get_attention_maps()
        flat = torch.cat([m.mean(dim=0).reshape(-1) for m in maps if m is not None])
    model.train()
    return flat.cpu().numpy()


def pattern_correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2))
    if denom < 1e-10:
        return 1.0
    return float(np.sum(a * b) / denom)


# ── Training ────────────────────────────────────────────────────

def train_one(dropout, weight_decay, train_funcs, test_funcs, probe_batch,
              train_steps=TRAIN_STEPS, verbose=False):
    """Train one model. Returns dict with time series AND summary statistics."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model = MicroGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=weight_decay, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps
    )

    probe_input = probe_batch[0]

    # Time series storage
    ts = {
        "steps": [], "train_loss": [], "train_acc": [], "val_acc": [],
        "eff_rank": [], "attn_entropy": [],
        "weight_vel": [], "attn_autocorr": [],
        "cst_wv": [],   # C_ST using weight velocity
        "cst_ac": [],   # C_ST using attention autocorrelation
    }

    # State for measurements
    prev_weights = None
    prev_attn_pattern = None
    prev_attn_step = -999

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

        if step % MEASURE_EVERY == 0 or step == train_steps - 1:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1].argmax(dim=-1)
                train_acc = (pred == last_tgt).float().mean().item()

                val_in, _, val_last = generate_batch(
                    256, N_EXAMPLES, VOCAB_SIZE, test_funcs, np.random.default_rng(0)
                )
                val_logits = model(val_in)
                val_pred = val_logits[:, -1].argmax(dim=-1)
                val_acc = (val_pred == val_last).float().mean().item()

                er = effective_rank(model)
                ae = attention_entropy(model)

                # Weight velocity
                curr_weights = weight_snapshot(model)
                wv = weight_velocity(curr_weights, prev_weights)
                prev_weights = curr_weights

                # Attention autocorrelation (lag = MEASURE_EVERY * 2 = 100 steps)
                curr_pattern = attention_pattern_flat(model, probe_input)
                if prev_attn_pattern is not None and (step - prev_attn_step) >= MEASURE_EVERY * 2:
                    ac = pattern_correlation(curr_pattern, prev_attn_pattern)
                    prev_attn_pattern = curr_pattern
                    prev_attn_step = step
                elif prev_attn_pattern is None:
                    ac = 0.0
                    prev_attn_pattern = curr_pattern
                    prev_attn_step = step
                else:
                    ac = ts["attn_autocorr"][-1] if ts["attn_autocorr"] else 0.0

                # Two C_ST variants
                cst_wv = er * wv        # structure × weight_velocity
                cst_ac = er * (1 - ac)   # structure × (1 - attn_autocorr)

            ts["steps"].append(step)
            ts["train_loss"].append(loss.item())
            ts["train_acc"].append(train_acc)
            ts["val_acc"].append(val_acc)
            ts["eff_rank"].append(er)
            ts["attn_entropy"].append(ae)
            ts["weight_vel"].append(wv)
            ts["attn_autocorr"].append(ac)
            ts["cst_wv"].append(cst_wv)
            ts["cst_ac"].append(cst_ac)

            if verbose and step % (MEASURE_EVERY * 4) == 0:
                print(f"  step {step:5d}  loss={loss.item():.3f}  "
                      f"val={val_acc:.3f}  rank={er:.3f}  "
                      f"wv={wv:.4f}  ac={ac:.3f}  "
                      f"Cwv={cst_wv:.4f}  Cac={cst_ac:.4f}")

            model.train()

    # ── Summary statistics ──────────────────────────────────────
    def last_n(key, n=5):
        return float(np.mean(ts[key][-n:])) if ts[key] else 0.0

    # Cumulative C_ST (integral) — the v2 key metric
    cumul_cst_wv = float(np.sum(ts["cst_wv"]))
    cumul_cst_ac = float(np.sum(ts["cst_ac"]))

    # Peak C_ST and when it occurred
    if ts["cst_wv"]:
        peak_idx_wv = int(np.argmax(ts["cst_wv"]))
        peak_cst_wv = ts["cst_wv"][peak_idx_wv]
        peak_step_wv = ts["steps"][peak_idx_wv]
    else:
        peak_cst_wv, peak_step_wv = 0.0, 0

    # Learning window: steps where C_ST(wv) > 50% of peak
    if peak_cst_wv > 0:
        threshold = peak_cst_wv * 0.5
        window_steps = [s for s, c in zip(ts["steps"], ts["cst_wv"]) if c > threshold]
        learning_window = len(window_steps) * MEASURE_EVERY if window_steps else 0
    else:
        learning_window = 0

    summary = {
        "val_acc": last_n("val_acc"),
        "train_acc": last_n("train_acc"),
        "eff_rank": last_n("eff_rank"),
        "attn_autocorr": last_n("attn_autocorr"),
        "attn_entropy": last_n("attn_entropy"),
        "weight_vel_final": last_n("weight_vel"),
        "cumul_cst_wv": cumul_cst_wv,
        "cumul_cst_ac": cumul_cst_ac,
        "peak_cst_wv": peak_cst_wv,
        "peak_step_wv": peak_step_wv,
        "learning_window": learning_window,
    }

    return ts, summary


# ── Sweep ───────────────────────────────────────────────────────

def run_sweep():
    print(f"Micro-GPT learnability sweep v2")
    print(f"Fixes: cumulative C_ST, weight velocity, finer resolution, longer training")
    print(f"Model: {N_LAYER}L {N_HEAD}H d={D_MODEL}")

    tmp = MicroGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER, 0.0)
    print(f"Parameters: {tmp.count_params():,}")
    del tmp

    print(f"Task: f(x)=(ax+b) mod {VOCAB_SIZE}, {N_EXAMPLES} in-context examples")
    print(f"Sweep: {len(DROPOUT_VALUES)}×{len(WD_VALUES)} = {len(DROPOUT_VALUES)*len(WD_VALUES)} runs")
    print(f"Training: {TRAIN_STEPS} steps, batch={BATCH_SIZE}, lr={BASE_LR}, measure every {MEASURE_EVERY}")
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    n_do = len(DROPOUT_VALUES)
    n_wd = len(WD_VALUES)
    n_total = n_do * n_wd

    # Result grids
    grids = {}
    for key in ["val_acc", "train_acc", "eff_rank", "attn_autocorr", "attn_entropy",
                "weight_vel_final", "cumul_cst_wv", "cumul_cst_ac",
                "peak_cst_wv", "peak_step_wv", "learning_window"]:
        grids[key] = np.zeros((n_wd, n_do))

    t0 = time.perf_counter()

    for wi, wd in enumerate(WD_VALUES):
        for di, do_rate in enumerate(DROPOUT_VALUES):
            idx = wi * n_do + di + 1
            _, summary = train_one(do_rate, wd, train_funcs, test_funcs, probe_batch)

            for key in grids:
                grids[key][wi, di] = summary[key]

            elapsed = time.perf_counter() - t0
            eta = (elapsed / idx) * (n_total - idx)
            print(f"  [{idx:3d}/{n_total}] do={do_rate:.2f} wd={wd:.2f}  "
                  f"val={summary['val_acc']:.3f}  "
                  f"Σwv={summary['cumul_cst_wv']:.2f}  "
                  f"peak={summary['peak_cst_wv']:.4f}@{summary['peak_step_wv']}  "
                  f"window={summary['learning_window']}  "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    # ── Find peaks ──────────────────────────────────────────────
    extent = [DROPOUT_VALUES[0], DROPOUT_VALUES[-1], WD_VALUES[0], WD_VALUES[-1]]

    def find_peak(grid, name):
        idx = np.unravel_index(np.argmax(grid), grid.shape)
        wd = WD_VALUES[idx[0]]
        do = DROPOUT_VALUES[idx[1]]
        val = grid[idx]
        print(f"{name} peak: dropout={do:.2f}, wd={wd:.2f}, value={val:.3f}")
        return do, wd

    pk_val_do, pk_val_wd = find_peak(grids["val_acc"], "Val accuracy")
    pk_cwv_do, pk_cwv_wd = find_peak(grids["cumul_cst_wv"], "Cumul C_ST(wv)")
    pk_cac_do, pk_cac_wd = find_peak(grids["cumul_cst_ac"], "Cumul C_ST(ac)")
    pk_win_do, pk_win_wd = find_peak(grids["learning_window"], "Learning window")

    # ── Triptych: structure, weight velocity C_ST, val accuracy ──
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    im0 = axes[0].imshow(grids["eff_rank"], origin="lower", aspect="auto",
                          extent=extent, cmap="magma", vmin=0)
    axes[0].set_title("Effective Rank\n(structure S)", fontsize=11)
    axes[0].set_xlabel("Dropout"); axes[0].set_ylabel("Weight Decay")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(grids["cumul_cst_wv"], origin="lower", aspect="auto",
                          extent=extent, cmap="inferno")
    axes[1].set_title("Cumulative C_ST (weight velocity)\nΣ rank × ‖ΔW‖/‖W‖", fontsize=11)
    axes[1].set_xlabel("Dropout")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(grids["learning_window"], origin="lower", aspect="auto",
                          extent=extent, cmap="YlOrRd")
    axes[2].set_title("Learning Window\n(steps with C_ST > 50% peak)", fontsize=11)
    axes[2].set_xlabel("Dropout")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(grids["val_acc"], origin="lower", aspect="auto",
                          extent=extent, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("Validation Accuracy\n(generalization)", fontsize=11)
    axes[3].set_xlabel("Dropout")
    plt.colorbar(im3, ax=axes[3])

    for ax in axes:
        ax.plot(pk_val_do, pk_val_wd, "cs", markersize=10, markeredgecolor="k", label="Val peak")
        ax.plot(pk_cwv_do, pk_cwv_wd, "w*", markersize=14, markeredgecolor="k", label="C_ST peak")
    axes[3].legend(loc="lower right", fontsize=8)

    fig.suptitle("Micro-GPT Phase Diagram v2 — Cumulative C_ST with Weight Velocity", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v2_triptych.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: v2_triptych.png")

    # ── Scatter: cumulative C_ST vs val accuracy ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, key, label in [
        (axes[0], "cumul_cst_wv", "Cumul C_ST (weight velocity)"),
        (axes[1], "cumul_cst_ac", "Cumul C_ST (attn autocorr)"),
        (axes[2], "learning_window", "Learning window (steps)"),
    ]:
        x = grids[key].ravel()
        y = grids["val_acc"].ravel()
        ax.scatter(x, y, c=y, cmap="viridis", edgecolors="k", linewidths=0.5, s=50)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Validation Accuracy", fontsize=11)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"r = {r:.3f}", fontsize=12)
        ax.text(0.05, 0.95, f"Pearson r = {r:.3f}", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle("Does C_ST predict generalization? (v2 — cumulative measures)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v2_scatter_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: v2_scatter_comparison.png")

    # ── 3D surfaces ─────────────────────────────────────────────
    D, W = np.meshgrid(DROPOUT_VALUES, WD_VALUES)

    for grid_key, cmap, title, fname in [
        ("cumul_cst_wv", "inferno", "Cumulative C_ST (weight velocity)", "v2_surface_cst_wv.png"),
        ("val_acc", "viridis", "Validation Accuracy", "v2_surface_val_acc.png"),
        ("learning_window", "YlOrRd", "Learning Window (steps)", "v2_surface_window.png"),
    ]:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(D, W, grids[grid_key], cmap=cmap,
                               edgecolor="k", linewidth=0.2, alpha=0.9)
        ax.set_xlabel("Dropout", fontsize=12, labelpad=10)
        ax.set_ylabel("Weight Decay", fontsize=12, labelpad=10)
        ax.set_zlabel(grid_key, fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=13, pad=20)
        fig.colorbar(surf, ax=ax, shrink=0.5)
        ax.view_init(elev=30, azim=225)
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

    # ── Generalization gap ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(grids["val_acc"], origin="lower", aspect="auto",
                      extent=extent, cmap="viridis", vmin=0, vmax=1)
    ax1.set_title("Validation Accuracy", fontsize=12)
    ax1.set_xlabel("Dropout"); ax1.set_ylabel("Weight Decay")
    plt.colorbar(im1, ax=ax1)
    ax1.plot(pk_val_do, pk_val_wd, "w*", markersize=15, markeredgecolor="k")

    gap = grids["train_acc"] - grids["val_acc"]
    im2 = ax2.imshow(gap, origin="lower", aspect="auto",
                      extent=extent, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax2.set_title("Generalization Gap (train - val)", fontsize=12)
    ax2.set_xlabel("Dropout")
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Micro-GPT v2: Generalization Landscape", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v2_generalization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: v2_generalization.png")

    # ── Peak timing heatmap ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grids["peak_step_wv"], origin="lower", aspect="auto",
                    extent=extent, cmap="plasma")
    ax.set_title("When does C_ST peak?\n(training step of peak weight-velocity C_ST)", fontsize=12)
    ax.set_xlabel("Dropout"); ax.set_ylabel("Weight Decay")
    plt.colorbar(im, ax=ax, label="Step")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v2_peak_timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: v2_peak_timing.png")

    # ── Save CSV ────────────────────────────────────────────────
    csv_path = OUT_DIR / "v2_sweep_results.csv"
    with open(csv_path, "w") as f:
        cols = ["dropout", "weight_decay"] + list(grids.keys())
        f.write(",".join(cols) + "\n")
        for wi, wd in enumerate(WD_VALUES):
            for di, do_rate in enumerate(DROPOUT_VALUES):
                row = [f"{do_rate:.4f}", f"{wd:.4f}"]
                for key in grids:
                    row.append(f"{grids[key][wi, di]:.6f}")
                f.write(",".join(row) + "\n")
    print(f"Saved: {csv_path.name}")

    # ── Print correlation summary ───────────────────────────────
    print("\n=== Correlation with validation accuracy ===")
    val_flat = grids["val_acc"].ravel()
    for key in ["cumul_cst_wv", "cumul_cst_ac", "learning_window",
                "peak_cst_wv", "eff_rank", "attn_entropy"]:
        r = np.corrcoef(grids[key].ravel(), val_flat)[0, 1]
        print(f"  {key:25s}  r = {r:+.3f}")


if __name__ == "__main__":
    run_sweep()
