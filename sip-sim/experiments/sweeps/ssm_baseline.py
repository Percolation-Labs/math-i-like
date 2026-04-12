"""
SSM Baselines for Parity Length Generalization.

Implements minimal diagonal S4D, selective SSM (Mamba-style), and GRU baselines
for the same running-parity task as length_gen_sweep.py.

All models are pure PyTorch — no CUDA dependencies.

Usage:
    uv run python -u learn/ssm_baseline.py
"""

import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from length_gen_sweep import (
    generate_parity_batch, n_to_seqlen,
    VOCAB_SIZE, D_MODEL, N_LAYER, BATCH_SIZE, BASE_LR,
    TRAIN_STEPS, MEASURE_EVERY, FIXED_DROPOUT, FIXED_WD, SEEDS,
    DEFAULT_TRAIN_N, DEFAULT_TEST_NS,
)

OUT_DIR = Path(__file__).resolve().parent


# ── Building blocks ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class MLPBlock(nn.Module):
    def __init__(self, d_model, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── S4D: Diagonal State Space Model ────────────────────────────

class S4DLayer(nn.Module):
    """Diagonal S4D layer with HiPPO-like initialization.

    Uses real parameterization: A is real negative, discretized via ZOH.
    Computes the SSM convolution in the time domain for variable-length inputs.
    """

    def __init__(self, d_model, state_dim=64, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # HiPPO-like initialization: A_n = -(n + 1/2)
        # Parameterize as log(-A) for stability
        A_init = torch.log(torch.arange(1, state_dim + 1).float() + 0.5)
        self.log_neg_A = nn.Parameter(A_init)  # log(-A), so A = -exp(log_neg_A)

        # B, C as learned real parameters
        self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)

        # Discretization step size (log-parameterized)
        self.log_dt = nn.Parameter(torch.zeros(d_model) - 1.0)  # dt ~ 0.37

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        """u: (B, T, D) -> (B, T, D)"""
        B_batch, T, D = u.shape

        dt = self.log_dt.exp()          # (D,)
        A = -self.log_neg_A.exp()       # (N,) — negative real

        # ZOH discretization: A_bar = exp(A * dt), B_bar = (A_bar - 1) / A * B
        A_dt = A.unsqueeze(0) * dt.unsqueeze(1)  # (D, N)
        A_bar = A_dt.exp()                         # (D, N)
        B_bar = self.B * (A_bar - 1.0) / (A_dt + 1e-8)  # (D, N)

        # Scan: x_{t+1} = A_bar * x_t + B_bar * u_t; y_t = C * x_t + D * u_t
        # For variable length, run sequential scan (no FFT needed for our sizes)
        x = torch.zeros(B_batch, D, self.state_dim, device=u.device)  # (B, D, N)
        ys = []

        for t in range(T):
            x = A_bar.unsqueeze(0) * x + B_bar.unsqueeze(0) * u[:, t, :].unsqueeze(-1)
            y = (self.C.unsqueeze(0) * x).sum(-1)  # (B, D)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, T, D)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u
        return self.dropout(y)


class S4DBlock(nn.Module):
    def __init__(self, d_model, state_dim=64, dropout=0.05):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = S4DLayer(d_model, state_dim, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLPBlock(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class S4DModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, state_dim=64, dropout=0.05):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            S4DBlock(d_model, state_dim, dropout) for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Selective SSM (Mamba-style) ─────────────────────────────────

class SelectiveSSMLayer(nn.Module):
    """Selective SSM: B, C, and dt are input-dependent (projected from x).

    This is the core Mamba idea — the selection mechanism makes the SSM
    content-aware rather than fixed.
    """

    def __init__(self, d_model, state_dim=64, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Fixed A (HiPPO-like, not input-dependent in Mamba)
        A_init = torch.log(torch.arange(1, state_dim + 1).float() + 0.5)
        self.log_neg_A = nn.Parameter(A_init)

        # Input-dependent projections for B, C, dt
        self.proj_B = nn.Linear(d_model, state_dim, bias=False)
        self.proj_C = nn.Linear(d_model, state_dim, bias=False)
        self.proj_dt = nn.Linear(d_model, d_model, bias=True)

        # Initialize dt projection bias so dt starts ~ 0.3-1.0
        nn.init.constant_(self.proj_dt.bias, -1.0)

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        """u: (B, T, D) -> (B, T, D)"""
        B_batch, T, D = u.shape

        A = -self.log_neg_A.exp()  # (N,)

        # Input-dependent parameters
        B_t = self.proj_B(u)           # (B, T, N)
        C_t = self.proj_C(u)           # (B, T, N)
        dt = F.softplus(self.proj_dt(u))  # (B, T, D) — positive

        # Sequential scan with input-dependent discretization
        x = torch.zeros(B_batch, D, self.state_dim, device=u.device)
        ys = []

        for t in range(T):
            dt_t = dt[:, t, :]  # (B, D)
            A_dt = A.unsqueeze(0) * dt_t.unsqueeze(-1)  # (B, D, N)
            A_bar = A_dt.exp()
            B_bar_t = B_t[:, t, :].unsqueeze(1) * (A_bar - 1.0) / (A_dt + 1e-8)  # (B, D, N)

            x = A_bar * x + B_bar_t * u[:, t, :].unsqueeze(-1)
            y = (C_t[:, t, :].unsqueeze(1) * x).sum(-1)  # (B, D)
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u
        return self.dropout(y)


class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model, state_dim=64, dropout=0.05):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = SelectiveSSMLayer(d_model, state_dim, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLPBlock(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SelectiveSSMModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, state_dim=64, dropout=0.05):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SelectiveSSMBlock(d_model, state_dim, dropout) for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── GRU Baseline ────────────────────────────────────────────────

class GRUModel(nn.Module):
    """Stacked GRU baseline with same layer structure."""

    def __init__(self, vocab_size, d_model, n_layer, dropout=0.05):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            d_model, d_model, num_layers=n_layer,
            batch_first=True, dropout=dropout if n_layer > 1 else 0.0,
        )
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.drop(self.tok_emb(idx))
        x, _ = self.gru(x)
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Configurations ──────────────────────────────────────────────

# Tune state_dim to hit ~209K params (close to transformer baseline)
# S4D: embedding(1024) + 4*[RMSNorm(64) + S4D(D*N*3 + D*2) + RMSNorm(64) + MLP(64*256*2)] + head(1024)
# We'll use state_dim=16 to keep params in range

SSM_CONFIGS = {
    "s4d": lambda: S4DModel(
        VOCAB_SIZE, D_MODEL, N_LAYER, state_dim=64, dropout=FIXED_DROPOUT,
    ),
    "selective_ssm": lambda: SelectiveSSMModel(
        VOCAB_SIZE, D_MODEL, N_LAYER, state_dim=64, dropout=FIXED_DROPOUT,
    ),
    "gru": lambda: GRUModel(
        VOCAB_SIZE, 96, N_LAYER, dropout=FIXED_DROPOUT,
    ),
}

SSM_CONFIG_NAMES = list(SSM_CONFIGS.keys())


# ── Training + Evaluation ──────────────────────────────────────

def train_and_evaluate_ssm(config_name, train_n, test_ns, seed=42, verbose=False):
    """Train SSM model on parity, evaluate at test lengths."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = SSM_CONFIGS[config_name]()
    n_params = model.count_params()
    train_T = n_to_seqlen(train_n)

    if verbose:
        print(f"  Config: {config_name}, params: {n_params:,}, "
              f"train n={train_n} (T={train_T}), seed={seed}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR,
        weight_decay=FIXED_WD, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=TRAIN_STEPS)

    train_curve = {"steps": [], "train_acc": [], "val_acc": []}

    for step in range(TRAIN_STEPS):
        model.train()
        inp, tgt, labels = generate_parity_batch(
            BATCH_SIZE, train_n, VOCAB_SIZE, rng)
        logits = model(inp)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % MEASURE_EVERY == 0 or step == TRAIN_STEPS - 1:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1].argmax(-1)
                train_acc = (pred == labels).float().mean().item()

                val_rng = np.random.default_rng(0)
                val_in, _, val_lab = generate_parity_batch(
                    512, train_n, VOCAB_SIZE, val_rng)
                val_logits = model(val_in)
                val_acc = (val_logits[:, -1].argmax(-1) == val_lab).float().mean().item()

            train_curve["steps"].append(step)
            train_curve["train_acc"].append(train_acc)
            train_curve["val_acc"].append(val_acc)

            if verbose and step % (MEASURE_EVERY * 4) == 0:
                print(f"    step {step:5d}  loss={loss.item():.3f}  "
                      f"train={train_acc:.3f}  val={val_acc:.3f}")

            model.train()

    # Evaluate at each test length
    model.eval()
    N_EVAL = 1024
    length_results = {}

    with torch.no_grad():
        for test_n in test_ns:
            eval_rng = np.random.default_rng(12345)
            eval_in, _, eval_lab = generate_parity_batch(
                N_EVAL, test_n, VOCAB_SIZE, eval_rng)
            eval_logits = model(eval_in)
            eval_pred = eval_logits[:, -1].argmax(-1)
            acc = (eval_pred == eval_lab).float().mean().item()
            length_results[test_n] = acc

            test_T = n_to_seqlen(test_n)
            if verbose:
                ratio = test_T / train_T
                ood = " (OOD)" if test_n > train_n else ""
                print(f"    n={test_n:3d} T={test_T:3d} ({ratio:.1f}x){ood}: "
                      f"acc={acc:.3f}")

    train_acc_final = float(np.mean(train_curve["val_acc"][-5:]))
    summary = {
        "config": config_name,
        "seed": seed,
        "train_n": train_n,
        "n_params": n_params,
        "train_acc": train_acc_final,
        **{f"acc_n{n}": a for n, a in length_results.items()},
    }

    return train_curve, length_results, summary


# ── Sweep ───────────────────────────────────────────────────────

def run_ssm_sweep(train_n=DEFAULT_TRAIN_N, test_ns=None):
    if test_ns is None:
        test_ns = DEFAULT_TEST_NS

    train_T = n_to_seqlen(train_n)
    test_Ts = [n_to_seqlen(n) for n in test_ns]

    print(f"SSM Baseline Sweep: Parity Task")
    print(f"Train: n={train_n} examples (T={train_T})")
    print(f"Test:  n={test_ns} -> T={test_Ts}")
    print(f"Model: {N_LAYER}L d={D_MODEL}, V={VOCAB_SIZE}")
    print(f"Seeds: {SEEDS}")
    print(f"Configs: {SSM_CONFIG_NAMES}")
    print()

    for name in SSM_CONFIG_NAMES:
        m = SSM_CONFIGS[name]()
        print(f"  {name:25s}  params={m.count_params():,}")
        del m
    print()

    all_results = []
    all_length_results = {}
    all_curves = {}
    t0 = time.perf_counter()
    total_runs = len(SSM_CONFIG_NAMES) * len(SEEDS)
    idx = 0

    for config_name in SSM_CONFIG_NAMES:
        for seed in SEEDS:
            idx += 1
            curve, length_res, summary = train_and_evaluate_ssm(
                config_name, train_n, test_ns, seed=seed, verbose=True)

            all_results.append(summary)
            all_length_results[(config_name, seed)] = length_res
            all_curves[(config_name, seed)] = curve

            elapsed = time.perf_counter() - t0
            eta = (elapsed / idx) * (total_runs - idx)

            acc_str = "  ".join(f"n{n}={length_res[n]:.2f}" for n in test_ns)
            print(f"  [{idx:2d}/{total_runs}] {config_name:25s} s={seed}  "
                  f"{acc_str}  ({elapsed:.0f}s, ~{eta:.0f}s left)")
            print()

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    plot_ssm_results(all_results, all_length_results, all_curves, train_n, test_ns)
    print_ssm_summary(all_results, test_ns)


def print_ssm_summary(results, test_ns):
    print(f"\n{'='*80}")
    print(f"SSM BASELINE SUMMARY (mean +/- std over {len(SEEDS)} seeds)")
    print(f"{'='*80}")

    header = f"{'Config':25s}"
    for n in test_ns:
        T = n_to_seqlen(n)
        header += f"  {'n='+str(n)+f'(T={T})':>14s}"
    print(header)
    print("-" * (25 + 16 * len(test_ns)))

    for config_name in SSM_CONFIG_NAMES:
        rows = [r for r in results if r["config"] == config_name]
        if not rows:
            continue
        line = f"{config_name:25s}"
        for n in test_ns:
            key = f"acc_n{n}"
            vals = [r[key] for r in rows]
            m, s = np.mean(vals), np.std(vals)
            line += f"  {m:.3f}+/-{s:.3f} "
        print(line)

    n_train = test_ns[0]
    n_max = test_ns[-1]
    print(f"\nGeneralization gap (n={n_train} -> n={n_max}):")
    for config_name in SSM_CONFIG_NAMES:
        rows = [r for r in results if r["config"] == config_name]
        if not rows:
            continue
        acc_train = np.mean([r[f"acc_n{n_train}"] for r in rows])
        acc_max = np.mean([r[f"acc_n{n_max}"] for r in rows])
        gap = acc_train - acc_max
        print(f"  {config_name:25s}  {acc_train:.3f} -> {acc_max:.3f}  "
              f"(gap={gap:+.3f})")


def plot_ssm_results(results, all_length_results, all_curves, train_n, test_ns):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SSM_CONFIG_NAMES)))
    train_T = n_to_seqlen(train_n)

    # 1. Length generalization curves
    for ci, config_name in enumerate(SSM_CONFIG_NAMES):
        accs_by_n = {n: [] for n in test_ns}
        for seed in SEEDS:
            key = (config_name, seed)
            if key in all_length_results:
                for n, a in all_length_results[key].items():
                    accs_by_n[n].append(a)

        test_Ts = [n_to_seqlen(n) for n in test_ns]
        means = [np.mean(accs_by_n[n]) for n in test_ns]
        stds = [np.std(accs_by_n[n]) for n in test_ns]

        axes[0].errorbar(test_Ts, means, yerr=stds, fmt="o-",
                         color=colors[ci], label=config_name,
                         capsize=3, linewidth=2, markersize=6)

    axes[0].axvline(x=train_T, color="gray", linestyle="--", alpha=0.7,
                    label=f"train T={train_T}")
    axes[0].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5,
                    label="random")
    axes[0].set_xlabel("Sequence length T")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Length Generalization: Parity (SSM Baselines)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.35, 1.05)

    # 2. Accuracy ratio
    for ci, config_name in enumerate(SSM_CONFIG_NAMES):
        accs_by_n = {n: [] for n in test_ns}
        for seed in SEEDS:
            key = (config_name, seed)
            if key in all_length_results:
                for n, a in all_length_results[key].items():
                    accs_by_n[n].append(a)

        train_acc = np.mean(accs_by_n[test_ns[0]])
        if train_acc > 0.55:
            ratios = [np.mean(accs_by_n[n]) / train_acc for n in test_ns]
            length_ratios = [n_to_seqlen(n) / train_T for n in test_ns]
            axes[1].plot(length_ratios, ratios, "o-", color=colors[ci],
                         label=config_name, linewidth=2, markersize=6)

    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Length ratio (T_test / T_train)")
    axes[1].set_ylabel("Accuracy ratio (test / train)")
    axes[1].set_title("Normalized Generalization")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. Training curves
    for ci, config_name in enumerate(SSM_CONFIG_NAMES):
        key = (config_name, SEEDS[0])
        if key in all_curves:
            curve = all_curves[key]
            axes[2].plot(curve["steps"], curve["val_acc"],
                         color=colors[ci], label=config_name, alpha=0.8)
    axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Val Accuracy (at train length)")
    axes[2].set_title(f"Training Curves (seed={SEEDS[0]})")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"SSM Baselines: Parity, Train n={train_n} (T={train_T})\n"
        f"{N_LAYER}L d={D_MODEL} V={VOCAB_SIZE} | {len(SEEDS)} seeds",
        fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / f"ssm_parity_n{train_n}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ssm_sweep()
