"""
Experiment 1: RetNet-style exponential decay in attention.

The "one-line change": multiply attention scores by D[i,j] = γ^{i-j} before
softmax. This makes old tokens decay exponentially, analogous to pheromone
evaporation in the ant system.

Sweep over γ (retention rate) to find whether decay alone produces a phase
transition with meaningful C_ST dynamics.

Usage:
    uv run python -u learn/retnet_decay_sweep.py
    uv run python -u learn/retnet_decay_sweep.py --single 0.92 --verbose
    uv run python -u learn/retnet_decay_sweep.py --mode 2d
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
TRAIN_STEPS = 5000
MEASURE_EVERY = 50
SEED = 42

# Fixed regularization (near-optimal from v2 sweep)
FIXED_DROPOUT = 0.05
FIXED_WD = 0.10

# 1D sweep: γ only
GAMMA_VALUES = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90,
                         0.92, 0.94, 0.96, 0.98, 0.99, 1.00])

# 2D sweep: γ × weight_decay
GAMMA_VALUES_2D = np.array([0.70, 0.80, 0.85, 0.90, 0.94, 0.96, 0.98, 1.00])
WD_VALUES_2D = np.linspace(0.0, 0.40, 8)

N_TRAIN_FUNCS = 200
N_TEST_FUNCS = 40


# ── Model ───────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class DecayAttention(nn.Module):
    """Causal self-attention with RetNet-style exponential decay.

    The ONLY change from standard attention: before softmax, multiply scores
    by D[i,j] = γ^{i-j}. When γ=1.0, this is identical to standard attention.
    """

    def __init__(self, d_model, n_head, dropout, max_len, gamma=1.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.gamma = gamma
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        # Decay matrix: D[i,j] = γ^{i-j} for i >= j
        # Pre-compute for max_len, slice at runtime
        if gamma < 1.0:
            positions = torch.arange(max_len).float()
            # D[i,j] = γ^{i-j} = γ^{positions[i] - positions[j]}
            decay = gamma ** (positions.unsqueeze(1) - positions.unsqueeze(0))
            decay = torch.tril(decay)  # causal
            self.register_buffer("decay", decay.view(1, 1, max_len, max_len))
        else:
            self.decay = None  # γ=1.0 → standard attention, no decay needed

        self.last_attn = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # ═══════════════════════════════════════════════════════════
        # THE ONE LINE: multiply by decay mask before masking/softmax
        if self.decay is not None:
            att = att * self.decay[:, :, :T, :T]
        # ═══════════════════════════════════════════════════════════

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len, gamma=1.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = DecayAttention(d_model, n_head, dropout, max_len, gamma)
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


class DecayGPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout, gamma=1.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, dropout, max_len, gamma) for _ in range(n_layer)]
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


# ── Data generation ────────────────────────────────────────────

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
    targets = tokens[:, 1:].copy()
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]
    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


# ── Instrumentation ───────────────────────────────────────────

def effective_rank(model):
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
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def weight_velocity(w_now, w_prev):
    if w_prev is None:
        return 1.0
    delta = (w_now - w_prev).norm().item()
    norm = w_now.norm().item()
    return delta / max(norm, 1e-10)


def attention_pattern_flat(model, input_ids):
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


# ── Training ──────────────────────────────────────────────────

def train_one(gamma, dropout, weight_decay, train_funcs, test_funcs, probe_batch,
              train_steps=TRAIN_STEPS, verbose=False):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model = DecayGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER, dropout, gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=weight_decay, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps
    )

    probe_input = probe_batch[0]

    ts = {
        "steps": [], "train_loss": [], "train_acc": [], "val_acc": [],
        "eff_rank": [], "weight_vel": [], "attn_autocorr": [],
        "cst_wv": [], "cst_ac": [],
    }

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

                curr_weights = weight_snapshot(model)
                wv = weight_velocity(curr_weights, prev_weights)
                prev_weights = curr_weights

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

                cst_wv = er * wv
                cst_ac = er * (1 - ac)

            ts["steps"].append(step)
            ts["train_loss"].append(loss.item())
            ts["train_acc"].append(train_acc)
            ts["val_acc"].append(val_acc)
            ts["eff_rank"].append(er)
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

    # ── Summary ──────────────────────────────────────────────
    def last_n(key, n=5):
        return float(np.mean(ts[key][-n:])) if ts[key] else 0.0

    cumul_cst_wv = float(np.sum(ts["cst_wv"]))
    cumul_cst_ac = float(np.sum(ts["cst_ac"]))

    if ts["cst_wv"]:
        peak_idx_wv = int(np.argmax(ts["cst_wv"]))
        peak_cst_wv = ts["cst_wv"][peak_idx_wv]
        peak_step_wv = ts["steps"][peak_idx_wv]
    else:
        peak_cst_wv, peak_step_wv = 0.0, 0

    # Learning window
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
        "weight_vel_final": last_n("weight_vel"),
        "cumul_cst_wv": cumul_cst_wv,
        "cumul_cst_ac": cumul_cst_ac,
        "peak_cst_wv": peak_cst_wv,
        "peak_step_wv": peak_step_wv,
        "learning_window": learning_window,
    }

    return ts, summary


# ── 1D Sweep: γ only ─────────────────────────────────────────

def run_1d_sweep():
    print(f"RetNet decay sweep: γ ∈ [{GAMMA_VALUES[0]:.2f}, {GAMMA_VALUES[-1]:.2f}]")
    print(f"Fixed: dropout={FIXED_DROPOUT}, wd={FIXED_WD}")
    print(f"Model: DecayGPT {N_LAYER}L {N_HEAD}H d={D_MODEL}")

    tmp = DecayGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT, 0.9)
    print(f"Parameters: {tmp.count_params():,}")
    del tmp

    print(f"Task: f(x)=(ax+b) mod {VOCAB_SIZE}, {N_EXAMPLES} in-context examples")
    print(f"Training: {TRAIN_STEPS} steps, batch={BATCH_SIZE}")
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    results = []
    all_ts = {}
    t0 = time.perf_counter()

    for gi, gamma in enumerate(GAMMA_VALUES):
        ts, summary = train_one(gamma, FIXED_DROPOUT, FIXED_WD,
                                train_funcs, test_funcs, probe_batch)
        summary["gamma"] = gamma
        results.append(summary)
        all_ts[gamma] = ts

        elapsed = time.perf_counter() - t0
        eta = (elapsed / (gi + 1)) * (len(GAMMA_VALUES) - gi - 1)
        print(f"  [{gi+1:2d}/{len(GAMMA_VALUES)}] γ={gamma:.2f}  "
              f"val={summary['val_acc']:.3f}  rank={summary['eff_rank']:.3f}  "
              f"ac={summary['attn_autocorr']:.3f}  "
              f"Σcst_wv={summary['cumul_cst_wv']:.2f}  "
              f"window={summary['learning_window']}  "
              f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    plot_1d_results(results, all_ts)
    save_1d_csv(results)


def plot_1d_results(results, all_ts):
    gammas = [r["gamma"] for r in results]
    val_accs = [r["val_acc"] for r in results]
    eff_ranks = [r["eff_rank"] for r in results]
    autocorrs = [r["attn_autocorr"] for r in results]
    cumul_cst = [r["cumul_cst_wv"] for r in results]
    windows = [r["learning_window"] for r in results]

    # ── Main result: val accuracy vs γ ──────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(gammas, val_accs, "o-", color="tab:blue", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("γ (retention rate)", fontsize=11)
    axes[0, 0].set_ylabel("Validation Accuracy", fontsize=11)
    axes[0, 0].set_title("Generalization vs Decay Rate", fontsize=12)
    axes[0, 0].axhline(y=1/VOCAB_SIZE, color="gray", linestyle="--", alpha=0.5, label="Random")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(gammas, eff_ranks, "s-", color="tab:orange", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("γ (retention rate)", fontsize=11)
    axes[0, 1].set_ylabel("Effective Rank (S)", fontsize=11)
    axes[0, 1].set_title("Structure vs Decay Rate", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(gammas, autocorrs, "^-", color="tab:red", linewidth=2, markersize=8)
    axes[0, 2].set_xlabel("γ (retention rate)", fontsize=11)
    axes[0, 2].set_ylabel("Attention Autocorrelation", fontsize=11)
    axes[0, 2].set_title("Temporal Stability vs Decay Rate", fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)

    # P = 1 - autocorr
    plasticity = [1 - ac for ac in autocorrs]
    axes[1, 0].plot(gammas, plasticity, "D-", color="tab:green", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("γ (retention rate)", fontsize=11)
    axes[1, 0].set_ylabel("Plasticity P = 1 - autocorr", fontsize=11)
    axes[1, 0].set_title("Plasticity vs Decay Rate", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(gammas, cumul_cst, "p-", color="tab:purple", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("γ (retention rate)", fontsize=11)
    axes[1, 1].set_ylabel("Cumul C_ST (weight vel)", fontsize=11)
    axes[1, 1].set_title("Cumulative Complexity vs Decay Rate", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    # C_ST vs val accuracy scatter
    axes[1, 2].scatter(cumul_cst, val_accs, c=gammas, cmap="viridis",
                       edgecolors="k", linewidths=0.5, s=100)
    for i, g in enumerate(gammas):
        axes[1, 2].annotate(f"γ={g:.2f}", (cumul_cst[i], val_accs[i]),
                            fontsize=7, ha="center", va="bottom")
    r = np.corrcoef(cumul_cst, val_accs)[0, 1]
    axes[1, 2].set_xlabel("Cumul C_ST", fontsize=11)
    axes[1, 2].set_ylabel("Validation Accuracy", fontsize=11)
    axes[1, 2].set_title(f"C_ST vs Generalization  (r = {r:.3f})", fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("RetNet Decay Sweep: Phase Transition in γ\n"
                 f"dropout={FIXED_DROPOUT}, wd={FIXED_WD}  |  "
                 f"Does mandatory decay produce edge-of-chaos dynamics?", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "retnet_decay_1d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: retnet_decay_1d.png")

    # ── Time series comparison for select γ values ──────────
    select_gammas = [g for g in [0.50, 0.80, 0.92, 0.98, 1.00] if g in all_ts]
    if len(select_gammas) >= 3:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(select_gammas)))

        for g, color in zip(select_gammas, colors):
            ts = all_ts[g]
            label = f"γ={g:.2f}"
            axes[0, 0].plot(ts["steps"], ts["val_acc"], color=color, label=label, alpha=0.8)
            axes[0, 1].plot(ts["steps"], ts["eff_rank"], color=color, label=label, alpha=0.8)
            axes[1, 0].plot(ts["steps"], ts["attn_autocorr"], color=color, label=label, alpha=0.8)
            axes[1, 1].plot(ts["steps"], ts["cst_wv"], color=color, label=label, alpha=0.8)

        axes[0, 0].set_title("Validation Accuracy over Training"); axes[0, 0].set_ylabel("Val Acc")
        axes[0, 1].set_title("Effective Rank over Training"); axes[0, 1].set_ylabel("Eff Rank")
        axes[1, 0].set_title("Attention Autocorrelation"); axes[1, 0].set_ylabel("Autocorr")
        axes[1, 1].set_title("C_ST (weight velocity)"); axes[1, 1].set_ylabel("C_ST")

        for ax in axes.ravel():
            ax.set_xlabel("Step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Training Dynamics at Different Decay Rates", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "retnet_decay_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved: retnet_decay_timeseries.png")


def save_1d_csv(results):
    csv_path = OUT_DIR / "retnet_decay_1d.csv"
    with open(csv_path, "w") as f:
        cols = list(results[0].keys())
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                            for k in cols) + "\n")
    print(f"Saved: {csv_path.name}")


# ── 2D Sweep: γ × weight_decay ──────────────────────────────

def run_2d_sweep():
    print(f"RetNet decay 2D sweep: γ × weight_decay")
    print(f"γ: {GAMMA_VALUES_2D}")
    print(f"WD: {WD_VALUES_2D}")
    print(f"Fixed dropout={FIXED_DROPOUT}")
    n_total = len(GAMMA_VALUES_2D) * len(WD_VALUES_2D)
    print(f"Total: {n_total} runs")
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    n_g = len(GAMMA_VALUES_2D)
    n_w = len(WD_VALUES_2D)

    grids = {}
    for key in ["val_acc", "eff_rank", "attn_autocorr", "cumul_cst_wv",
                "cumul_cst_ac", "learning_window"]:
        grids[key] = np.zeros((n_w, n_g))

    t0 = time.perf_counter()
    idx = 0

    for wi, wd in enumerate(WD_VALUES_2D):
        for gi, gamma in enumerate(GAMMA_VALUES_2D):
            idx += 1
            _, summary = train_one(gamma, FIXED_DROPOUT, wd,
                                   train_funcs, test_funcs, probe_batch)

            for key in grids:
                grids[key][wi, gi] = summary[key]

            elapsed = time.perf_counter() - t0
            eta = (elapsed / idx) * (n_total - idx)
            print(f"  [{idx:3d}/{n_total}] γ={gamma:.2f} wd={wd:.2f}  "
                  f"val={summary['val_acc']:.3f}  "
                  f"ac={summary['attn_autocorr']:.3f}  "
                  f"Σcst={summary['cumul_cst_wv']:.2f}  "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    plot_2d_results(grids)
    save_2d_csv(grids)


def plot_2d_results(grids):
    extent = [GAMMA_VALUES_2D[0], GAMMA_VALUES_2D[-1],
              WD_VALUES_2D[0], WD_VALUES_2D[-1]]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for ax, key, cmap, title in [
        (axes[0, 0], "val_acc", "viridis", "Validation Accuracy"),
        (axes[0, 1], "eff_rank", "magma", "Effective Rank (S)"),
        (axes[0, 2], "attn_autocorr", "RdYlBu_r", "Attention Autocorrelation"),
        (axes[1, 0], "cumul_cst_wv", "inferno", "Cumul C_ST (weight vel)"),
        (axes[1, 1], "cumul_cst_ac", "inferno", "Cumul C_ST (attn autocorr)"),
        (axes[1, 2], "learning_window", "YlOrRd", "Learning Window (steps)"),
    ]:
        im = ax.imshow(grids[key], origin="lower", aspect="auto",
                       extent=extent, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("γ (retention rate)")
        ax.set_ylabel("Weight Decay")
        plt.colorbar(im, ax=ax)

        # Mark peak
        peak_idx = np.unravel_index(np.argmax(grids[key]), grids[key].shape)
        peak_g = GAMMA_VALUES_2D[peak_idx[1]]
        peak_w = WD_VALUES_2D[peak_idx[0]]
        ax.plot(peak_g, peak_w, "w*", markersize=14, markeredgecolor="k")

    # Mark val_acc peak on all panels
    val_peak = np.unravel_index(np.argmax(grids["val_acc"]), grids["val_acc"].shape)
    vpg = GAMMA_VALUES_2D[val_peak[1]]
    vpw = WD_VALUES_2D[val_peak[0]]
    for ax in axes.ravel():
        ax.plot(vpg, vpw, "cs", markersize=10, markeredgecolor="k")

    fig.suptitle(f"RetNet Decay 2D Phase Diagram: γ × Weight Decay\n"
                 f"dropout={FIXED_DROPOUT}  |  Val acc peak at γ={vpg:.2f}, wd={vpw:.2f}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "retnet_decay_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: retnet_decay_2d.png")

    # Scatter: C_ST vs val accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, key, label in [
        (ax1, "cumul_cst_wv", "Cumul C_ST (weight velocity)"),
        (ax2, "cumul_cst_ac", "Cumul C_ST (attn autocorr)"),
    ]:
        x = grids[key].ravel()
        y = grids["val_acc"].ravel()
        ax.scatter(x, y, c=y, cmap="viridis", edgecolors="k", linewidths=0.5, s=50)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Validation Accuracy", fontsize=11)
        ax.set_title(f"r = {r:.3f}", fontsize=12)
        ax.text(0.05, 0.95, f"Pearson r = {r:.3f}", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle("Does C_ST predict generalization with mandatory decay?", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "retnet_decay_2d_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: retnet_decay_2d_scatter.png")


def save_2d_csv(grids):
    csv_path = OUT_DIR / "retnet_decay_2d.csv"
    with open(csv_path, "w") as f:
        cols = ["gamma", "weight_decay"] + list(grids.keys())
        f.write(",".join(cols) + "\n")
        for wi, wd in enumerate(WD_VALUES_2D):
            for gi, gamma in enumerate(GAMMA_VALUES_2D):
                row = [f"{gamma:.4f}", f"{wd:.4f}"]
                for key in grids:
                    row.append(f"{grids[key][wi, gi]:.6f}")
                f.write(",".join(row) + "\n")
    print(f"Saved: {csv_path.name}")


# ── Single run mode ──────────────────────────────────────────

def run_single(gamma):
    print(f"Single run: γ={gamma}, dropout={FIXED_DROPOUT}, wd={FIXED_WD}")
    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    ts, summary = train_one(gamma, FIXED_DROPOUT, FIXED_WD,
                            train_funcs, test_funcs, probe_batch, verbose=True)

    print(f"\nResults:")
    for k, v in summary.items():
        print(f"  {k:25s} = {v}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RetNet decay sweep")
    parser.add_argument("--mode", choices=["1d", "2d", "single"], default="1d")
    parser.add_argument("--single", type=float, default=0.92,
                        help="γ value for single run mode")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "single":
        run_single(args.single)
    elif args.mode == "2d":
        run_2d_sweep()
    else:
        run_1d_sweep()


if __name__ == "__main__":
    main()
