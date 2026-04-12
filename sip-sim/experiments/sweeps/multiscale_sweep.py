"""
Experiment 3: Multi-scale Stigmergic Attention.

Each head gets a different evaporation rate, creating a hierarchy of timescales.
Some heads maintain long-lived pheromone trails (low ε), others operate in the
transient regime (critical ε), others have fast decay (high ε).

The key parameter is ε_spread: how much the per-head rates diverge from the mean.
With 4 heads and mean ε_mean, the rates are spaced geometrically:
  ε_h = ε_mean * r^h  where r = spread factor

Also introduces "productive plasticity" — a better P component for C_ST that
measures loss-correlated field change rather than total field change.

Usage:
    uv run python -u learn/multiscale_sweep.py
    uv run python -u learn/multiscale_sweep.py --single 0.15 --verbose
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

FIXED_DROPOUT = 0.05
FIXED_WD = 0.10

# Multi-scale configurations to sweep
# Each is (name, [ε_head0, ε_head1, ε_head2, ε_head3])
CONFIGS = [
    # Uniform baselines (from 1D sweep)
    ("uniform_0.00", [0.00, 0.00, 0.00, 0.00]),
    ("uniform_0.05", [0.05, 0.05, 0.05, 0.05]),
    ("uniform_0.10", [0.10, 0.10, 0.10, 0.10]),
    ("uniform_0.15", [0.15, 0.15, 0.15, 0.15]),
    ("uniform_0.20", [0.20, 0.20, 0.20, 0.20]),

    # Multi-scale: span frozen→critical
    ("multi_narrow",  [0.05, 0.10, 0.15, 0.20]),
    ("multi_wide",    [0.00, 0.07, 0.15, 0.25]),
    ("multi_wider",   [0.00, 0.05, 0.15, 0.30]),

    # Multi-scale: one critical head, rest frozen
    ("one_critical",  [0.00, 0.00, 0.00, 0.15]),
    ("two_critical",  [0.00, 0.00, 0.15, 0.15]),

    # Multi-scale: one frozen head, rest critical
    ("one_frozen",    [0.15, 0.15, 0.15, 0.00]),

    # Extreme spread
    ("extreme",       [0.00, 0.05, 0.20, 0.50]),

    # All near-critical
    ("all_critical",  [0.12, 0.14, 0.16, 0.18]),

    # Standard attention baseline (no pheromone)
    ("no_pheromone",  None),  # special case: skip field bias
]

N_TRAIN_FUNCS = 200
N_TEST_FUNCS = 40


# ── Model ───────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class MultiScaleStigmergicAttention(nn.Module):
    """Stigmergic Attention with per-head evaporation rates.

    Each head has its own decay matrix, creating a hierarchy of timescales.
    Head h's field: field_h[i] = Σ_{j≤i} (1-ε_h)^{i-j} * deposit[j,h]
    """

    def __init__(self, d_model, n_head, dropout, max_len, evap_rates=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.use_field = evap_rates is not None

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        if self.use_field:
            self.w_deposit = nn.Linear(d_model, n_head, bias=False)

            # Per-head decay matrices
            positions = torch.arange(max_len).float()
            dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)

            # Stack per-head decay matrices: (n_head, T, T)
            decay_matrices = []
            for h, eps in enumerate(evap_rates):
                dm = (1.0 - eps) ** dist
                dm = torch.tril(dm)
                decay_matrices.append(dm)
            self.register_buffer("decay_matrices",
                                 torch.stack(decay_matrices, dim=0))  # (n_head, T, T)

            self.evap_rates = evap_rates
        else:
            self.evap_rates = None

        self.last_attn = None
        self._last_field = None

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if self.use_field:
            deposits = self.w_deposit(x)  # (B, T, n_head)

            # Per-head field computation
            # decay_matrices: (n_head, T, T), deposits: (B, T, n_head)
            D = self.decay_matrices[:, :T, :T]  # (n_head, T, T)
            # field[b, i, h] = Σ_j D[h, i, j] * deposits[b, j, h]
            field = torch.einsum("hij,bjh->bih", D, deposits)  # (B, T, n_head)
            self._last_field = field.detach()

            field_bias = field.permute(0, 2, 1).unsqueeze(2)  # (B, n_head, 1, T)
            att = att + field_bias

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        return out

    def get_field_snapshot(self):
        return self._last_field


class MultiScaleBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len, evap_rates=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiScaleStigmergicAttention(
            d_model, n_head, dropout, max_len, evap_rates)
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


class MultiScaleGPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 evap_rates=None):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MultiScaleBlock(d_model, n_head, dropout, max_len, evap_rates)
            for _ in range(n_layer)
        ])
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

    def get_field_snapshots(self):
        return [block.attn.get_field_snapshot() for block in self.blocks]

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


def field_autocorrelation(model, prev_fields):
    curr_fields = model.get_field_snapshots()
    if prev_fields is None or any(f is None for f in curr_fields):
        return 0.0, curr_fields

    correlations = []
    for curr, prev in zip(curr_fields, prev_fields):
        if curr is None or prev is None:
            continue
        c = curr.reshape(-1).cpu().numpy()
        p = prev.reshape(-1).cpu().numpy()
        cm, pm = c.mean(), p.mean()
        cd, pd = c - cm, p - pm
        denom = np.sqrt(np.sum(cd ** 2)) * np.sqrt(np.sum(pd ** 2))
        if denom > 1e-10:
            correlations.append(float(np.sum(cd * pd) / denom))

    ac = float(np.mean(correlations)) if correlations else 1.0
    return ac, curr_fields


def productive_plasticity(loss_now, loss_prev, wv):
    """P component that measures loss-correlated change.

    Productive plasticity = weight_velocity × max(0, loss_improvement).
    Only counts change that reduces loss. Noise-induced change (which
    doesn't reduce loss) gets zero P.
    """
    if loss_prev is None:
        return 0.0
    improvement = max(0.0, loss_prev - loss_now)
    return wv * improvement


# ── Training ──────────────────────────────────────────────────

def train_one(evap_rates, dropout, weight_decay,
              train_funcs, test_funcs, probe_batch,
              train_steps=TRAIN_STEPS, verbose=False):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model = MultiScaleGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER,
                           dropout, evap_rates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=weight_decay, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps
    )

    ts = {
        "steps": [], "train_loss": [], "train_acc": [], "val_acc": [],
        "eff_rank": [], "weight_vel": [],
        "field_autocorr": [],
        "cst_field": [],
        "productive_p": [],  # new: loss-correlated plasticity
        "cst_productive": [],  # new: S × productive_P
    }

    prev_weights = None
    prev_fields = None
    prev_field_step = -999
    prev_loss = None

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

                # Field autocorrelation
                model(probe_batch[0])
                if (step - prev_field_step) >= MEASURE_EVERY * 2:
                    fac, prev_fields = field_autocorrelation(model, prev_fields)
                    prev_field_step = step
                elif prev_fields is None:
                    fac = 0.0
                    _, prev_fields = field_autocorrelation(model, None)
                    prev_field_step = step
                else:
                    fac = ts["field_autocorr"][-1] if ts["field_autocorr"] else 0.0

                # Productive plasticity
                current_loss = loss.item()
                pp = productive_plasticity(current_loss, prev_loss, wv)
                prev_loss = current_loss

                cst_field = er * (1.0 - fac)
                cst_prod = er * pp

            ts["steps"].append(step)
            ts["train_loss"].append(current_loss)
            ts["train_acc"].append(train_acc)
            ts["val_acc"].append(val_acc)
            ts["eff_rank"].append(er)
            ts["weight_vel"].append(wv)
            ts["field_autocorr"].append(fac)
            ts["cst_field"].append(cst_field)
            ts["productive_p"].append(pp)
            ts["cst_productive"].append(cst_prod)

            if verbose and step % (MEASURE_EVERY * 4) == 0:
                print(f"  step {step:5d}  loss={current_loss:.3f}  "
                      f"val={val_acc:.3f}  rank={er:.3f}  "
                      f"fac={fac:.3f}  pp={pp:.4f}  "
                      f"Cp={cst_prod:.4f}")

            model.train()

    # ── Summary ──────────────────────────────────────────────
    def last_n(key, n=5):
        return float(np.mean(ts[key][-n:])) if ts[key] else 0.0

    cumul_cst_field = float(np.sum(ts["cst_field"]))
    cumul_cst_prod = float(np.sum(ts["cst_productive"]))

    summary = {
        "val_acc": last_n("val_acc"),
        "train_acc": last_n("train_acc"),
        "eff_rank": last_n("eff_rank"),
        "field_autocorr": last_n("field_autocorr"),
        "cumul_cst_field": cumul_cst_field,
        "cumul_cst_productive": cumul_cst_prod,
    }

    return ts, summary


# ── Sweep ────────────────────────────────────────────────────

def run_sweep():
    print(f"Multi-scale Stigmergic Attention sweep")
    print(f"Model: MultiScaleGPT {N_LAYER}L {N_HEAD}H d={D_MODEL}")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Fixed: dropout={FIXED_DROPOUT}, wd={FIXED_WD}")
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    results = []
    all_ts = {}
    t0 = time.perf_counter()

    for ci, (name, evap_rates) in enumerate(CONFIGS):
        ts, summary = train_one(evap_rates, FIXED_DROPOUT, FIXED_WD,
                                train_funcs, test_funcs, probe_batch)
        summary["name"] = name
        summary["evap_rates"] = str(evap_rates) if evap_rates else "none"
        results.append(summary)
        all_ts[name] = ts

        elapsed = time.perf_counter() - t0
        eta = (elapsed / (ci + 1)) * (len(CONFIGS) - ci - 1)
        rates_str = ",".join(f"{e:.2f}" for e in evap_rates) if evap_rates else "none"
        print(f"  [{ci+1:2d}/{len(CONFIGS)}] {name:20s} [{rates_str:20s}]  "
              f"val={summary['val_acc']:.3f}  rank={summary['eff_rank']:.3f}  "
              f"fac={summary['field_autocorr']:.3f}  "
              f"Σcf={summary['cumul_cst_field']:.2f}  "
              f"Σcp={summary['cumul_cst_productive']:.2f}  "
              f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    plot_results(results, all_ts)
    save_csv(results)


def plot_results(results, all_ts):
    names = [r["name"] for r in results]
    val_accs = [r["val_acc"] for r in results]
    eff_ranks = [r["eff_rank"] for r in results]
    field_acs = [r["field_autocorr"] for r in results]
    cumul_cst = [r["cumul_cst_field"] for r in results]
    cumul_prod = [r["cumul_cst_productive"] for r in results]

    # ── Bar chart: val accuracy by configuration ──────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    x = np.arange(len(names))

    axes[0, 0].barh(x, val_accs, color=colors, edgecolor="k", linewidth=0.5)
    axes[0, 0].set_yticks(x)
    axes[0, 0].set_yticklabels(names, fontsize=8)
    axes[0, 0].set_xlabel("Validation Accuracy")
    axes[0, 0].set_title("Generalization by Configuration")
    axes[0, 0].axvline(x=1/VOCAB_SIZE, color="gray", linestyle="--", alpha=0.5)

    axes[0, 1].barh(x, eff_ranks, color=colors, edgecolor="k", linewidth=0.5)
    axes[0, 1].set_yticks(x)
    axes[0, 1].set_yticklabels(names, fontsize=8)
    axes[0, 1].set_xlabel("Effective Rank (S)")
    axes[0, 1].set_title("Representational Structure")

    # Scatter: cumul_cst_productive vs val_acc
    r_prod = np.corrcoef(cumul_prod, val_accs)[0, 1]
    axes[1, 0].scatter(cumul_prod, val_accs, c=colors, edgecolors="k",
                       linewidths=0.5, s=100)
    for i, n in enumerate(names):
        axes[1, 0].annotate(n, (cumul_prod[i], val_accs[i]),
                            fontsize=6, rotation=20, ha="left")
    axes[1, 0].set_xlabel("Cumul C_ST (productive)")
    axes[1, 0].set_ylabel("Validation Accuracy")
    axes[1, 0].set_title(f"Productive C_ST vs Generalization (r={r_prod:.3f})")
    axes[1, 0].grid(True, alpha=0.3)

    # Scatter: cumul_cst_field vs val_acc
    r_field = np.corrcoef(cumul_cst, val_accs)[0, 1]
    axes[1, 1].scatter(cumul_cst, val_accs, c=colors, edgecolors="k",
                       linewidths=0.5, s=100)
    for i, n in enumerate(names):
        axes[1, 1].annotate(n, (cumul_cst[i], val_accs[i]),
                            fontsize=6, rotation=20, ha="left")
    axes[1, 1].set_xlabel("Cumul C_ST (field autocorr)")
    axes[1, 1].set_ylabel("Validation Accuracy")
    axes[1, 1].set_title(f"Field C_ST vs Generalization (r={r_field:.3f})")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Multi-Scale Stigmergic Attention: Configuration Comparison\n"
                 "Does mixing timescales improve generalization?", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multiscale_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: multiscale_comparison.png")

    # ── Time series for select configs ──────────────────────
    select = ["no_pheromone", "uniform_0.15", "multi_wide", "all_critical", "one_critical"]
    select = [s for s in select if s in all_ts]

    if len(select) >= 3:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 0.5, len(select)))

        for name, color in zip(select, colors):
            ts = all_ts[name]
            axes[0, 0].plot(ts["steps"], ts["val_acc"], color=color, label=name, alpha=0.8)
            axes[0, 1].plot(ts["steps"], ts["eff_rank"], color=color, label=name, alpha=0.8)
            axes[1, 0].plot(ts["steps"], ts["field_autocorr"], color=color, label=name, alpha=0.8)
            axes[1, 1].plot(ts["steps"], ts["cst_productive"], color=color, label=name, alpha=0.8)

        axes[0, 0].set_title("Validation Accuracy"); axes[0, 0].set_ylabel("Val Acc")
        axes[0, 1].set_title("Effective Rank"); axes[0, 1].set_ylabel("Eff Rank")
        axes[1, 0].set_title("Field Autocorrelation"); axes[1, 0].set_ylabel("FAC")
        axes[1, 1].set_title("Productive C_ST"); axes[1, 1].set_ylabel("S × PP")

        for ax in axes.ravel():
            ax.set_xlabel("Step")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Multi-Scale: Training Dynamics Comparison", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "multiscale_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved: multiscale_timeseries.png")


def save_csv(results):
    csv_path = OUT_DIR / "multiscale_results.csv"
    with open(csv_path, "w") as f:
        cols = list(results[0].keys())
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(
                f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                for k in cols
            ) + "\n")
    print(f"Saved: {csv_path.name}")

    # Print correlation summary
    print("\n=== Correlation with validation accuracy ===")
    val = np.array([r["val_acc"] for r in results])
    for key in ["cumul_cst_field", "cumul_cst_productive", "eff_rank",
                "field_autocorr"]:
        x = np.array([r[key] for r in results])
        r = np.corrcoef(x, val)[0, 1]
        print(f"  {key:25s}  r = {r:+.3f}")


# ── Single run mode ──────────────────────────────────────────

def run_single(name):
    config = next((c for c in CONFIGS if c[0] == name), None)
    if config is None:
        # Try parsing as comma-separated rates
        try:
            rates = [float(x) for x in name.split(",")]
            config = (f"custom_{name}", rates)
        except ValueError:
            print(f"Unknown config: {name}")
            print(f"Available: {[c[0] for c in CONFIGS]}")
            return

    print(f"Single run: {config[0]}, rates={config[1]}")
    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    ts, summary = train_one(config[1], FIXED_DROPOUT, FIXED_WD,
                            train_funcs, test_funcs, probe_batch, verbose=True)

    print(f"\nResults:")
    for k, v in summary.items():
        print(f"  {k:25s} = {v}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-scale Stigmergic sweep")
    parser.add_argument("--mode", choices=["sweep", "single"], default="sweep")
    parser.add_argument("--single", type=str, default="multi_wide",
                        help="Config name or comma-separated rates")
    args = parser.parse_args()

    if args.mode == "single":
        run_single(args.single)
    else:
        run_sweep()


if __name__ == "__main__":
    main()
