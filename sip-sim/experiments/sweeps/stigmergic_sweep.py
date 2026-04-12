"""
Experiment 2: Stigmergic Attention — decaying memory field.

The key insight from Experiment 1: pre-softmax decay (RetNet γ mask) doesn't
break the frozen-attention equilibrium because softmax normalizes away the
effect. We need decay on the STORED STATE itself.

StigmergicAttention maintains a persistent field F that:
  1. DECAYS every forward pass (evaporation — real information loss)
  2. DIFFUSES causally (spatial smoothing)
  3. Is READ via attention (Q from input, K/V from field)
  4. Is WRITTEN to by a deposit mechanism (input → field)

The evaporation rate ε is the phase control parameter:
  ε → 0: field accumulates → frozen (like standard transformer)
  ε → 1: field erased each step → disorder (memoryless)
  ε = ε_c: transient field → edge of chaos

Usage:
    uv run python -u learn/stigmergic_sweep.py
    uv run python -u learn/stigmergic_sweep.py --single 0.15 --verbose
    uv run python -u learn/stigmergic_sweep.py --mode 2d
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

# Fixed regularization
FIXED_DROPOUT = 0.05
FIXED_WD = 0.10

# 1D sweep: ε (evaporation rate)
EVAP_VALUES = np.array([0.00, 0.01, 0.03, 0.05, 0.10, 0.15,
                         0.20, 0.30, 0.40, 0.50, 0.70, 0.90])

# 2D sweep: ε × dropout (evaporation × noise — the ant analogy)
EVAP_VALUES_2D = np.array([0.00, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.25])
DROPOUT_VALUES_2D = np.array([0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])

N_TRAIN_FUNCS = 200
N_TEST_FUNCS = 40


# ── Model ───────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class StigmergicAttention(nn.Module):
    """Attention with a decaying memory field.

    The field F is a persistent state that:
      - Decays by (1-ε) each forward pass (evaporation)
      - Gets smoothed causally (diffusion)
      - Is queried via attention (keys/values from field)
      - Is written to via deposit mechanism

    This creates genuine non-equilibrium dynamics: information in the field
    is destroyed unless actively reinforced.
    """

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.1, diffuse_rate=0.05):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.evap_rate = evap_rate

        # Standard Q/K/V (input → input attention, as in normal transformer)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Deposit: each position deposits a scalar "pheromone" per head
        # This scalar becomes an additive bias in the attention logits
        self.w_deposit = nn.Linear(d_model, n_head, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        # Decay matrix: D[i,j] = (1-ε)^{i-j} for causal decaying sum
        # Position j's deposit decays exponentially for later positions.
        positions = torch.arange(max_len).float()
        decay_matrix = (1.0 - evap_rate) ** (positions.unsqueeze(0) - positions.unsqueeze(1))
        decay_matrix = torch.tril(decay_matrix)  # causal
        self.register_buffer("decay_matrix", decay_matrix)

        self.last_attn = None
        self._last_field = None

    def forward(self, x):
        B, T, C = x.shape

        # ── Standard Q/K/V from input ────────────────────────
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # ── Pheromone field bias ─────────────────────────────
        # Each position deposits a scalar per head. The deposit from
        # position j decays as (1-ε)^{i-j} when read by position i.
        # This biases attention: high-pheromone positions get attended more.
        deposits = self.w_deposit(x)  # (B, T, n_head)
        D = self.decay_matrix[:T, :T]  # (T, T)
        # field[i, h] = Σ_{j≤i} (1-ε)^{i-j} * deposit[j, h]
        field = torch.einsum("ij,bjh->bih", D, deposits)  # (B, T, n_head)
        self._last_field = field.detach()

        # Add field as attention bias: position j's accumulated pheromone
        # biases how much position i attends to j.
        # field_bias[i,j,h] = field[j,h] = pheromone at position j
        field_bias = field.permute(0, 2, 1).unsqueeze(2)  # (B, n_head, 1, T)
        att = att + field_bias  # broadcast over query positions

        # ── Causal masking + softmax ─────────────────────────
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        return out

    def get_field_snapshot(self):
        """Return field state for instrumentation."""
        return self._last_field


class StigmergicBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = StigmergicAttention(d_model, n_head, dropout, max_len,
                                         evap_rate)
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


class StigmergicGPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 evap_rate=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            StigmergicBlock(d_model, n_head, dropout, max_len, evap_rate)
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
    """Autocorrelation of the memory field itself (not attention patterns).

    This is the key new measurement: does the field state change between
    measurement intervals? In standard transformers, there IS no field.
    In StigmergicGPT, the field is the pheromone analogue.
    """
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


def field_magnitude(model):
    """Average magnitude of the memory field — measures accumulation vs decay."""
    fields = model.get_field_snapshots()
    mags = []
    for f in fields:
        if f is not None:
            mags.append(f.abs().mean().item())
    return float(np.mean(mags)) if mags else 0.0


# ── Training ──────────────────────────────────────────────────

def train_one(evap_rate, dropout, weight_decay,
              train_funcs, test_funcs, probe_batch,
              train_steps=TRAIN_STEPS, verbose=False):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model = StigmergicGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER,
                           dropout, evap_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=weight_decay, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps
    )

    ts = {
        "steps": [], "train_loss": [], "train_acc": [], "val_acc": [],
        "eff_rank": [], "weight_vel": [],
        "field_autocorr": [], "field_mag": [],
        "cst_wv": [], "cst_field": [],
    }

    prev_weights = None
    prev_fields = None
    prev_field_step = -999

    for step in range(train_steps):
        model.train()
        # Reset fields between batches (each batch is independent sequences)


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
                fm = field_magnitude(model)

                # Weight velocity
                curr_weights = weight_snapshot(model)
                wv = weight_velocity(curr_weights, prev_weights)
                prev_weights = curr_weights

                # Field autocorrelation (measure at lag = 2*MEASURE_EVERY)
                # Run a probe batch to get consistent field state
        
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

                cst_wv = er * wv
                cst_field = er * (1.0 - fac)  # S × (1 - field_autocorr)

            ts["steps"].append(step)
            ts["train_loss"].append(loss.item())
            ts["train_acc"].append(train_acc)
            ts["val_acc"].append(val_acc)
            ts["eff_rank"].append(er)
            ts["weight_vel"].append(wv)
            ts["field_autocorr"].append(fac)
            ts["field_mag"].append(fm)
            ts["cst_wv"].append(cst_wv)
            ts["cst_field"].append(cst_field)

            if verbose and step % (MEASURE_EVERY * 4) == 0:
                print(f"  step {step:5d}  loss={loss.item():.3f}  "
                      f"val={val_acc:.3f}  rank={er:.3f}  "
                      f"fac={fac:.3f}  fmag={fm:.4f}  "
                      f"Cf={cst_field:.4f}")

            model.train()

    # ── Summary ──────────────────────────────────────────────
    def last_n(key, n=5):
        return float(np.mean(ts[key][-n:])) if ts[key] else 0.0

    cumul_cst_wv = float(np.sum(ts["cst_wv"]))
    cumul_cst_field = float(np.sum(ts["cst_field"]))

    if ts["cst_field"]:
        peak_idx = int(np.argmax(ts["cst_field"]))
        peak_cst = ts["cst_field"][peak_idx]
        peak_step = ts["steps"][peak_idx]
    else:
        peak_cst, peak_step = 0.0, 0

    # Learning window
    if peak_cst > 0:
        threshold = peak_cst * 0.5
        window_steps = [s for s, c in zip(ts["steps"], ts["cst_field"]) if c > threshold]
        learning_window = len(window_steps) * MEASURE_EVERY if window_steps else 0
    else:
        learning_window = 0

    summary = {
        "val_acc": last_n("val_acc"),
        "train_acc": last_n("train_acc"),
        "eff_rank": last_n("eff_rank"),
        "field_autocorr": last_n("field_autocorr"),
        "field_mag": last_n("field_mag"),
        "weight_vel_final": last_n("weight_vel"),
        "cumul_cst_wv": cumul_cst_wv,
        "cumul_cst_field": cumul_cst_field,
        "peak_cst_field": peak_cst,
        "peak_step_field": peak_step,
        "learning_window": learning_window,
    }

    return ts, summary


# ── 1D Sweep: ε only ─────────────────────────────────────────

def run_1d_sweep():
    print(f"Stigmergic Attention sweep: ε ∈ [{EVAP_VALUES[0]:.2f}, {EVAP_VALUES[-1]:.2f}]")
    print(f"Fixed: dropout={FIXED_DROPOUT}, wd={FIXED_WD}")
    print(f"Model: StigmergicGPT {N_LAYER}L {N_HEAD}H d={D_MODEL}")

    tmp = StigmergicGPT(VOCAB_SIZE, SEQ_LEN - 1, D_MODEL, N_HEAD, N_LAYER,
                         FIXED_DROPOUT, 0.1)
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

    for ei, evap in enumerate(EVAP_VALUES):
        ts, summary = train_one(evap, FIXED_DROPOUT, FIXED_WD,
                                train_funcs, test_funcs, probe_batch)
        summary["evap_rate"] = evap
        results.append(summary)
        all_ts[evap] = ts

        elapsed = time.perf_counter() - t0
        eta = (elapsed / (ei + 1)) * (len(EVAP_VALUES) - ei - 1)
        print(f"  [{ei+1:2d}/{len(EVAP_VALUES)}] ε={evap:.2f}  "
              f"val={summary['val_acc']:.3f}  rank={summary['eff_rank']:.3f}  "
              f"fac={summary['field_autocorr']:.3f}  "
              f"fmag={summary['field_mag']:.4f}  "
              f"Σcf={summary['cumul_cst_field']:.2f}  "
              f"window={summary['learning_window']}  "
              f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    plot_1d_results(results, all_ts)
    save_1d_csv(results)

    # Print correlation summary
    print("\n=== Correlation with validation accuracy ===")
    val_flat = np.array([r["val_acc"] for r in results])
    for key in ["cumul_cst_field", "cumul_cst_wv", "eff_rank",
                "field_autocorr", "field_mag", "learning_window"]:
        x = np.array([r[key] for r in results])
        r = np.corrcoef(x, val_flat)[0, 1]
        print(f"  {key:25s}  r = {r:+.3f}")


def plot_1d_results(results, all_ts):
    evaps = [r["evap_rate"] for r in results]
    val_accs = [r["val_acc"] for r in results]
    eff_ranks = [r["eff_rank"] for r in results]
    field_acs = [r["field_autocorr"] for r in results]
    field_mags = [r["field_mag"] for r in results]
    cumul_cst = [r["cumul_cst_field"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Val accuracy vs ε
    axes[0, 0].plot(evaps, val_accs, "o-", color="tab:blue", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("ε (evaporation rate)", fontsize=11)
    axes[0, 0].set_ylabel("Validation Accuracy", fontsize=11)
    axes[0, 0].set_title("Generalization vs Evaporation", fontsize=12)
    axes[0, 0].axhline(y=1/VOCAB_SIZE, color="gray", linestyle="--", alpha=0.5, label="Random")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Effective rank vs ε
    axes[0, 1].plot(evaps, eff_ranks, "s-", color="tab:orange", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("ε (evaporation rate)", fontsize=11)
    axes[0, 1].set_ylabel("Effective Rank (S)", fontsize=11)
    axes[0, 1].set_title("Structure vs Evaporation", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # Field autocorrelation vs ε — THIS IS THE KEY PLOT
    axes[0, 2].plot(evaps, field_acs, "^-", color="tab:red", linewidth=2, markersize=8)
    axes[0, 2].set_xlabel("ε (evaporation rate)", fontsize=11)
    axes[0, 2].set_ylabel("Field Autocorrelation", fontsize=11)
    axes[0, 2].set_title("Field Temporal Stability\n(does autocorr vary with ε?)", fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)

    # Field magnitude vs ε
    axes[1, 0].plot(evaps, field_mags, "D-", color="tab:green", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("ε (evaporation rate)", fontsize=11)
    axes[1, 0].set_ylabel("Field Magnitude", fontsize=11)
    axes[1, 0].set_title("Field Accumulation vs Evaporation", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # Cumul C_ST(field) vs ε
    axes[1, 1].plot(evaps, cumul_cst, "p-", color="tab:purple", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("ε (evaporation rate)", fontsize=11)
    axes[1, 1].set_ylabel("Cumul C_ST (field)", fontsize=11)
    axes[1, 1].set_title("Cumul S × (1 - field_autocorr)", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    # C_ST vs val accuracy scatter — THE MONEY PLOT
    r_val = np.corrcoef(cumul_cst, val_accs)[0, 1]
    axes[1, 2].scatter(cumul_cst, val_accs, c=[e for e in evaps], cmap="plasma",
                       edgecolors="k", linewidths=0.5, s=100)
    for i, e in enumerate(evaps):
        axes[1, 2].annotate(f"ε={e:.2f}", (cumul_cst[i], val_accs[i]),
                            fontsize=7, ha="center", va="bottom")
    axes[1, 2].set_xlabel("Cumul C_ST (field)", fontsize=11)
    axes[1, 2].set_ylabel("Validation Accuracy", fontsize=11)
    axes[1, 2].set_title(f"C_ST vs Generalization  (r = {r_val:.3f})", fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("Stigmergic Attention: Phase Transition in ε\n"
                 f"dropout={FIXED_DROPOUT}, wd={FIXED_WD}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stigmergic_1d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: stigmergic_1d.png")

    # ── Time series comparison ──────────────────────────────
    select_evaps = [e for e in [0.00, 0.05, 0.15, 0.30, 0.70] if e in all_ts]
    if len(select_evaps) >= 3:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(select_evaps)))

        for e, color in zip(select_evaps, colors):
            ts = all_ts[e]
            label = f"ε={e:.2f}"
            axes[0, 0].plot(ts["steps"], ts["val_acc"], color=color, label=label, alpha=0.8)
            axes[0, 1].plot(ts["steps"], ts["eff_rank"], color=color, label=label, alpha=0.8)
            axes[0, 2].plot(ts["steps"], ts["field_autocorr"], color=color, label=label, alpha=0.8)
            axes[1, 0].plot(ts["steps"], ts["field_mag"], color=color, label=label, alpha=0.8)
            axes[1, 1].plot(ts["steps"], ts["cst_field"], color=color, label=label, alpha=0.8)
            axes[1, 2].plot(ts["steps"], ts["weight_vel"], color=color, label=label, alpha=0.8)

        axes[0, 0].set_title("Validation Accuracy"); axes[0, 0].set_ylabel("Val Acc")
        axes[0, 1].set_title("Effective Rank (S)"); axes[0, 1].set_ylabel("Eff Rank")
        axes[0, 2].set_title("Field Autocorrelation"); axes[0, 2].set_ylabel("Field AC")
        axes[1, 0].set_title("Field Magnitude"); axes[1, 0].set_ylabel("Magnitude")
        axes[1, 1].set_title("C_ST (field)"); axes[1, 1].set_ylabel("S × (1-FAC)")
        axes[1, 2].set_title("Weight Velocity"); axes[1, 2].set_ylabel("‖ΔW‖/‖W‖")

        for ax in axes.ravel():
            ax.set_xlabel("Step")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Stigmergic Attention: Training Dynamics at Different Evaporation Rates",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "stigmergic_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved: stigmergic_timeseries.png")


def save_1d_csv(results):
    csv_path = OUT_DIR / "stigmergic_1d.csv"
    with open(csv_path, "w") as f:
        cols = list(results[0].keys())
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                            for k in cols) + "\n")
    print(f"Saved: {csv_path.name}")


# ── 2D Sweep: ε × dropout ────────────────────────────────────
# Direct analogy to ant (evaporation × noise)

def run_2d_sweep():
    print(f"Stigmergic Attention 2D sweep: ε × dropout")
    print(f"ε: {EVAP_VALUES_2D}")
    print(f"dropout: {DROPOUT_VALUES_2D}")
    n_total = len(EVAP_VALUES_2D) * len(DROPOUT_VALUES_2D)
    print(f"Total: {n_total} runs")
    print(f"Fixed: wd={FIXED_WD}")
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    n_e = len(EVAP_VALUES_2D)
    n_do = len(DROPOUT_VALUES_2D)

    grids = {}
    for key in ["val_acc", "eff_rank", "field_autocorr", "field_mag",
                "cumul_cst_field", "learning_window"]:
        grids[key] = np.zeros((n_do, n_e))

    t0 = time.perf_counter()
    idx = 0

    for di, do_rate in enumerate(DROPOUT_VALUES_2D):
        for ei, evap in enumerate(EVAP_VALUES_2D):
            idx += 1
            _, summary = train_one(evap, do_rate, FIXED_WD,
                                   train_funcs, test_funcs, probe_batch)

            for key in grids:
                grids[key][di, ei] = summary[key]

            elapsed = time.perf_counter() - t0
            eta = (elapsed / idx) * (n_total - idx)
            print(f"  [{idx:3d}/{n_total}] ε={evap:.2f} do={do_rate:.2f}  "
                  f"val={summary['val_acc']:.3f}  "
                  f"fac={summary['field_autocorr']:.3f}  "
                  f"Σcf={summary['cumul_cst_field']:.2f}  "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    plot_2d_results(grids)
    save_2d_csv(grids)

    # Correlation summary
    print("\n=== Correlation with validation accuracy ===")
    val_flat = grids["val_acc"].ravel()
    for key in ["cumul_cst_field", "eff_rank", "field_autocorr",
                "field_mag", "learning_window"]:
        r = np.corrcoef(grids[key].ravel(), val_flat)[0, 1]
        print(f"  {key:25s}  r = {r:+.3f}")


def plot_2d_results(grids):
    extent = [EVAP_VALUES_2D[0], EVAP_VALUES_2D[-1],
              DROPOUT_VALUES_2D[0], DROPOUT_VALUES_2D[-1]]

    # Find val_acc peak
    val_peak = np.unravel_index(np.argmax(grids["val_acc"]), grids["val_acc"].shape)
    vpd = DROPOUT_VALUES_2D[val_peak[0]]
    vpe = EVAP_VALUES_2D[val_peak[1]]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for ax, key, cmap, title in [
        (axes[0, 0], "val_acc", "viridis", "Validation Accuracy"),
        (axes[0, 1], "eff_rank", "magma", "Effective Rank (S)"),
        (axes[0, 2], "field_autocorr", "RdYlBu_r", "Field Autocorrelation"),
        (axes[1, 0], "field_mag", "hot", "Field Magnitude"),
        (axes[1, 1], "cumul_cst_field", "inferno", "Cumul C_ST (field)"),
        (axes[1, 2], "learning_window", "YlOrRd", "Learning Window"),
    ]:
        im = ax.imshow(grids[key], origin="lower", aspect="auto",
                       extent=extent, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("ε (evaporation)")
        ax.set_ylabel("Dropout (noise)")
        plt.colorbar(im, ax=ax)

        # Mark peak of this metric
        peak_idx = np.unravel_index(np.argmax(grids[key]), grids[key].shape)
        peak_e = EVAP_VALUES_2D[peak_idx[1]]
        peak_d = DROPOUT_VALUES_2D[peak_idx[0]]
        ax.plot(peak_e, peak_d, "w*", markersize=14, markeredgecolor="k")
        # Mark val_acc peak on all panels
        ax.plot(vpe, vpd, "cs", markersize=10, markeredgecolor="k")

    axes[0, 0].legend(["Metric peak", "Val acc peak"], loc="lower right", fontsize=8)

    fig.suptitle(f"Stigmergic Attention: Phase Diagram (ε × dropout)\n"
                 f"Evaporation × Noise — direct analogy to ant trail system  |  "
                 f"Val peak at ε={vpe:.2f}, do={vpd:.2f}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stigmergic_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: stigmergic_2d.png")

    # Scatter: C_ST vs val accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x1 = grids["cumul_cst_field"].ravel()
    y = grids["val_acc"].ravel()
    r1 = np.corrcoef(x1, y)[0, 1]
    ax1.scatter(x1, y, c=y, cmap="viridis", edgecolors="k", linewidths=0.5, s=50)
    ax1.set_xlabel("Cumul C_ST (field)", fontsize=11)
    ax1.set_ylabel("Validation Accuracy", fontsize=11)
    ax1.set_title(f"C_ST vs Generalization  (r = {r1:.3f})", fontsize=12)

    x2 = grids["eff_rank"].ravel()
    r2 = np.corrcoef(x2, y)[0, 1]
    ax2.scatter(x2, y, c=y, cmap="viridis", edgecolors="k", linewidths=0.5, s=50)
    ax2.set_xlabel("Effective Rank", fontsize=11)
    ax2.set_ylabel("Validation Accuracy", fontsize=11)
    ax2.set_title(f"Eff Rank vs Generalization  (r = {r2:.3f})", fontsize=12)

    fig.suptitle("Stigmergic Attention: Does C_ST predict generalization?", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stigmergic_2d_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: stigmergic_2d_scatter.png")


def save_2d_csv(grids):
    csv_path = OUT_DIR / "stigmergic_2d.csv"
    with open(csv_path, "w") as f:
        cols = ["evap_rate", "dropout"] + list(grids.keys())
        f.write(",".join(cols) + "\n")
        for di, do_rate in enumerate(DROPOUT_VALUES_2D):
            for ei, evap in enumerate(EVAP_VALUES_2D):
                row = [f"{evap:.4f}", f"{do_rate:.4f}"]
                for key in grids:
                    row.append(f"{grids[key][di, ei]:.6f}")
                f.write(",".join(row) + "\n")
    print(f"Saved: {csv_path.name}")


# ── Single run mode ──────────────────────────────────────────

def run_single(evap_rate):
    print(f"Single run: ε={evap_rate}, dropout={FIXED_DROPOUT}, wd={FIXED_WD}")

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    probe_rng = np.random.default_rng(999)
    probe_batch = generate_batch(32, N_EXAMPLES, VOCAB_SIZE, train_funcs, probe_rng)

    ts, summary = train_one(evap_rate, FIXED_DROPOUT, FIXED_WD,
                            train_funcs, test_funcs, probe_batch, verbose=True)

    print(f"\nResults:")
    for k, v in summary.items():
        print(f"  {k:25s} = {v}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stigmergic Attention sweep")
    parser.add_argument("--mode", choices=["1d", "2d", "single"], default="1d")
    parser.add_argument("--single", type=float, default=0.15,
                        help="ε value for single run mode")
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
