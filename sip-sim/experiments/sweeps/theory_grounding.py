"""
Theory Grounding Experiments for Stigmergic Attention.

Three experiments to move from empirical results to theory:

Experiment A: Learnable ε
  Make ε a trainable parameter (via sigmoid). If gradient descent converges to
  ε ≈ 0.15 across random seeds, that's self-organized criticality — the system
  finds its own edge of chaos without hyperparameter search.

Experiment B: Task Variation
  Vary vocab_size and n_examples while keeping the architecture fixed.
  If critical ε shifts predictably with task properties, we have a scaling law.
  Hypothesis: ε_c ∝ 1/T (longer sequences need slower decay to retain signal).

Experiment C: Mechanistic Inspection
  At the critical point, what does the pheromone field actually compute?
  Inspect deposits, field values, and attention patterns with/without field bias.
  Does the field mark "important" positions? Does it create specialization?

Usage:
    uv run python -u learn/theory_grounding.py --mode learnable
    uv run python -u learn/theory_grounding.py --mode task_variation
    uv run python -u learn/theory_grounding.py --mode mechanistic
    uv run python -u learn/theory_grounding.py --mode all
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

# ── Shared hyperparameters ─────────────────────────────────────

D_MODEL = 48
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 64
BASE_LR = 3e-3
TRAIN_STEPS = 5000
MEASURE_EVERY = 50
FIXED_DROPOUT = 0.05
FIXED_WD = 0.10

N_TRAIN_FUNCS = 200
N_TEST_FUNCS = 40


# ── Model components ───────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class LearnableStigmergicAttention(nn.Module):
    """Stigmergic Attention with LEARNABLE evaporation rate.

    ε is parameterized as sigmoid(raw_epsilon) so it stays in (0, 1).
    The model learns its own decay rate via gradient descent.
    """

    def __init__(self, d_model, n_head, dropout, max_len,
                 init_epsilon=0.15, per_head=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.max_len = max_len
        self.per_head = per_head

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_deposit = nn.Linear(d_model, n_head, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        # Learnable ε: parameterize via inverse-sigmoid for unconstrained optimization
        # sigmoid(raw) = init_epsilon → raw = log(init / (1 - init))
        raw_init = np.log(init_epsilon / (1.0 - init_epsilon))
        if per_head:
            self.raw_epsilon = nn.Parameter(torch.full((n_head,), raw_init))
        else:
            self.raw_epsilon = nn.Parameter(torch.tensor(raw_init))

        # Pre-compute position distances
        positions = torch.arange(max_len).float()
        self.register_buffer(
            "pos_dist", positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        )

        self.last_attn = None
        self._last_field = None

    @property
    def epsilon(self):
        return torch.sigmoid(self.raw_epsilon)

    def _build_decay_matrix(self, T):
        """Build decay matrix from current (learned) ε."""
        eps = self.epsilon
        dist = self.pos_dist[:T, :T]  # (T, T)

        if self.per_head:
            # eps: (n_head,), dist: (T, T) → decay: (n_head, T, T)
            decay = (1.0 - eps.view(-1, 1, 1)) ** dist.unsqueeze(0)
            causal = torch.tril(torch.ones(T, T, device=dist.device)).unsqueeze(0)
            return decay * causal
        else:
            decay = (1.0 - eps) ** dist
            return torch.tril(decay)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Pheromone field with LEARNED decay
        deposits = self.w_deposit(x)  # (B, T, n_head)
        D = self._build_decay_matrix(T)

        if self.per_head:
            # D: (n_head, T, T), deposits: (B, T, n_head)
            field = torch.einsum("hij,bjh->bih", D, deposits)
        else:
            # D: (T, T), deposits: (B, T, n_head)
            field = torch.einsum("ij,bjh->bih", D, deposits)

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


class FixedStigmergicAttention(nn.Module):
    """Same architecture but with fixed ε (for comparison / mechanistic)."""

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.15):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_deposit = nn.Linear(d_model, n_head, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        positions = torch.arange(max_len).float()
        decay_matrix = (1.0 - evap_rate) ** (positions.unsqueeze(0) - positions.unsqueeze(1))
        decay_matrix = torch.tril(decay_matrix)
        self.register_buffer("decay_matrix", decay_matrix)

        self.last_attn = None
        self._last_field = None
        self._last_deposits = None
        self._last_att_no_bias = None

    def forward(self, x, return_decomposed=False):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if return_decomposed:
            # Save attention BEFORE field bias
            att_no_bias = att.clone()
            att_no_bias = att_no_bias.masked_fill(
                self.mask[:, :, :T, :T] == 0, float("-inf"))
            self._last_att_no_bias = F.softmax(att_no_bias, dim=-1).detach()

        deposits = self.w_deposit(x)
        self._last_deposits = deposits.detach()

        D = self.decay_matrix[:T, :T]
        field = torch.einsum("ij,bjh->bih", D, deposits)
        self._last_field = field.detach()

        field_bias = field.permute(0, 2, 1).unsqueeze(2)
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


class Block(nn.Module):
    def __init__(self, attn_cls, d_model, n_head, dropout, max_len, **attn_kwargs):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = attn_cls(d_model, n_head, dropout, max_len, **attn_kwargs)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        x = x + self.attn(self.ln1(x), **kwargs)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 attn_cls=FixedStigmergicAttention, **attn_kwargs):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(attn_cls, d_model, n_head, dropout, max_len, **attn_kwargs)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, **kwargs):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x, **kwargs)
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data generation ────────────────────────────────────────────

def make_functions(vocab_size, n_train=N_TRAIN_FUNCS, n_test=N_TEST_FUNCS, seed=0):
    rng = np.random.default_rng(seed)
    all_funcs = [(a, b) for a in range(1, vocab_size) for b in range(vocab_size)]
    rng.shuffle(all_funcs)
    # Cap train/test to available functions (small vocab = fewer functions)
    total_needed = n_train + n_test
    if len(all_funcs) < total_needed:
        n_train = max(1, int(len(all_funcs) * 0.8))
        n_test = max(1, len(all_funcs) - n_train)
    return all_funcs[:n_train], all_funcs[n_train:n_train + n_test]


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


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT A: Learnable ε
# ═══════════════════════════════════════════════════════════════

def train_learnable(vocab_size, n_examples, seed, init_eps=0.15,
                    per_head=False, train_steps=TRAIN_STEPS):
    """Train with learnable ε. Track where ε converges."""
    seq_len = 2 * (n_examples + 1)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = GPT(vocab_size, seq_len - 1, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                attn_cls=LearnableStigmergicAttention,
                init_epsilon=init_eps, per_head=per_head)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=FIXED_WD, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps)

    train_funcs, test_funcs = make_functions(vocab_size, seed=0)

    ts = {"steps": [], "val_acc": [], "train_acc": [], "eff_rank": []}
    # Track ε per layer, per head (if per_head)
    n_eps = N_HEAD if per_head else 1
    for layer in range(N_LAYER):
        for h in range(n_eps):
            ts[f"eps_L{layer}_H{h}"] = []

    for step in range(train_steps):
        model.train()
        input_ids, targets, last_tgt = generate_batch(
            BATCH_SIZE, n_examples, vocab_size, train_funcs, rng)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
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
                    256, n_examples, vocab_size, test_funcs, np.random.default_rng(0))
                val_logits = model(val_in)
                val_pred = val_logits[:, -1].argmax(dim=-1)
                val_acc = (val_pred == val_last).float().mean().item()

                er = effective_rank(model)

            ts["steps"].append(step)
            ts["val_acc"].append(val_acc)
            ts["train_acc"].append(train_acc)
            ts["eff_rank"].append(er)

            # Record learned ε values
            for li, block in enumerate(model.blocks):
                eps = block.attn.epsilon.detach()
                if per_head:
                    for h in range(N_HEAD):
                        ts[f"eps_L{li}_H{h}"].append(eps[h].item())
                else:
                    ts[f"eps_L{li}_H0"].append(eps.item())

            model.train()

    # Final ε values
    final_eps = {}
    for li, block in enumerate(model.blocks):
        eps = block.attn.epsilon.detach()
        if per_head:
            for h in range(N_HEAD):
                final_eps[f"L{li}_H{h}"] = eps[h].item()
        else:
            final_eps[f"L{li}"] = eps.item()

    final_val = float(np.mean(ts["val_acc"][-5:]))
    final_rank = float(np.mean(ts["eff_rank"][-5:]))

    return ts, final_eps, final_val, final_rank


def run_learnable_experiment():
    """Experiment A: Does gradient descent find the critical ε?"""
    print("=" * 70)
    print("EXPERIMENT A: Learnable Evaporation Rate")
    print("=" * 70)
    print()

    # Test 1: Multiple seeds, same init → convergence test
    print("Test A1: 8 seeds, shared ε (init=0.15), vocab=16, n_examples=8")
    print("-" * 60)

    seeds = [42, 137, 256, 314, 512, 777, 999, 1337]
    results_a1 = []
    t0 = time.perf_counter()

    for i, seed in enumerate(seeds):
        ts, final_eps, val_acc, rank = train_learnable(
            vocab_size=16, n_examples=8, seed=seed, init_eps=0.15, per_head=False)
        results_a1.append({"seed": seed, "final_eps": final_eps,
                          "val_acc": val_acc, "rank": rank, "ts": ts})
        elapsed = time.perf_counter() - t0
        eps_str = ", ".join(f"{k}={v:.4f}" for k, v in final_eps.items())
        print(f"  seed={seed:4d}  val={val_acc:.3f}  rank={rank:.3f}  "
              f"ε=[{eps_str}]  ({elapsed:.0f}s)")

    eps_all = [list(r["final_eps"].values()) for r in results_a1]
    eps_flat = [e for row in eps_all for e in row]
    print(f"\n  ε statistics: mean={np.mean(eps_flat):.4f}, "
          f"std={np.std(eps_flat):.4f}, "
          f"min={np.min(eps_flat):.4f}, max={np.max(eps_flat):.4f}")

    # Test 2: Different initial ε → does it converge to same point?
    print("\nTest A2: 5 different initial ε, seed=42")
    print("-" * 60)

    init_eps_values = [0.01, 0.05, 0.15, 0.30, 0.50]
    results_a2 = []

    for init_eps in init_eps_values:
        ts, final_eps, val_acc, rank = train_learnable(
            vocab_size=16, n_examples=8, seed=42, init_eps=init_eps, per_head=False)
        results_a2.append({"init_eps": init_eps, "final_eps": final_eps,
                          "val_acc": val_acc, "rank": rank, "ts": ts})
        eps_str = ", ".join(f"{k}={v:.4f}" for k, v in final_eps.items())
        print(f"  init_ε={init_eps:.2f} → final ε=[{eps_str}]  val={val_acc:.3f}")

    # Test 3: Per-head learnable ε → do heads specialize?
    print("\nTest A3: Per-head learnable ε, 4 seeds")
    print("-" * 60)

    results_a3 = []
    for seed in [42, 137, 256, 314]:
        ts, final_eps, val_acc, rank = train_learnable(
            vocab_size=16, n_examples=8, seed=seed, init_eps=0.15, per_head=True)
        results_a3.append({"seed": seed, "final_eps": final_eps,
                          "val_acc": val_acc, "rank": rank, "ts": ts})
        eps_str = ", ".join(f"{k}={v:.4f}" for k, v in final_eps.items())
        print(f"  seed={seed:4d}  val={val_acc:.3f}  ε=[{eps_str}]")

    total = time.perf_counter() - t0
    print(f"\nExperiment A complete in {total:.0f}s ({total/60:.1f}min)")

    plot_learnable_results(results_a1, results_a2, results_a3)
    return results_a1, results_a2, results_a3


def plot_learnable_results(results_a1, results_a2, results_a3):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # A1: ε convergence across seeds
    ax = axes[0, 0]
    for r in results_a1:
        ts = r["ts"]
        # Plot first layer ε trajectory
        ax.plot(ts["steps"], ts["eps_L0_H0"], alpha=0.6, label=f"L0 s={r['seed']}")
    ax.set_xlabel("Step")
    ax.set_ylabel("ε")
    ax.set_title("A1: Layer 0 ε convergence (8 seeds)")
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="init=0.15")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for r in results_a1:
        ts = r["ts"]
        ax.plot(ts["steps"], ts["eps_L1_H0"], alpha=0.6, label=f"L1 s={r['seed']}")
    ax.set_xlabel("Step")
    ax.set_ylabel("ε")
    ax.set_title("A1: Layer 1 ε convergence (8 seeds)")
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # A2: Different initializations
    ax = axes[0, 2]
    for r in results_a2:
        ts = r["ts"]
        ax.plot(ts["steps"], ts["eps_L0_H0"], linewidth=2,
                label=f"init={r['init_eps']:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("ε")
    ax.set_title("A2: Convergence from different initial ε")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # A1: Val accuracy across seeds
    ax = axes[1, 0]
    for r in results_a1:
        ts = r["ts"]
        ax.plot(ts["steps"], ts["val_acc"], alpha=0.5, label=f"s={r['seed']}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("A1: Generalization (learnable ε)")
    ax.axhline(y=0.853, color="red", linestyle="--", alpha=0.5, label="fixed ε=0.15")
    ax.axhline(y=0.658, color="gray", linestyle="--", alpha=0.5, label="no pheromone")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # A3: Per-head ε — do heads specialize?
    ax = axes[1, 1]
    seed_colors = plt.cm.tab10(np.linspace(0, 0.4, len(results_a3)))
    for ri, r in enumerate(results_a3):
        eps_vals = list(r["final_eps"].values())
        x = np.arange(len(eps_vals))
        labels = list(r["final_eps"].keys())
        ax.bar(x + ri * 0.2, eps_vals, width=0.18, color=seed_colors[ri],
               label=f"seed={r['seed']}")
    ax.set_xticks(np.arange(len(labels)) + 0.3)
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_ylabel("Learned ε")
    ax.set_title("A3: Per-head ε (do heads specialize?)")
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # A3: Per-head ε trajectories for seed=42
    ax = axes[1, 2]
    if results_a3:
        ts = results_a3[0]["ts"]
        for key in sorted(ts.keys()):
            if key.startswith("eps_"):
                ax.plot(ts["steps"], ts[key], label=key.replace("eps_", ""), linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("ε")
        ax.set_title(f"A3: Per-head ε trajectory (seed={results_a3[0]['seed']})")
        ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment A: Does Gradient Descent Find Critical ε?", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theory_learnable_eps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: theory_learnable_eps.png")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT B: Task Variation
# ═══════════════════════════════════════════════════════════════

def find_critical_eps(vocab_size, n_examples, seed=42, eps_values=None):
    """Find critical ε for a given task configuration by sweep."""
    if eps_values is None:
        eps_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    seq_len = 2 * (n_examples + 1)
    train_funcs, test_funcs = make_functions(vocab_size, seed=0)

    rng_base = np.random.default_rng(seed)
    best_val = 0.0
    best_eps = 0.0
    results = []

    for eps in eps_values:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        model = GPT(vocab_size, seq_len - 1, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                     attn_cls=FixedStigmergicAttention, evap_rate=eps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                      weight_decay=FIXED_WD, betas=(0.85, 0.99))
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=TRAIN_STEPS)

        for step in range(TRAIN_STEPS):
            model.train()
            input_ids, targets, last_tgt = generate_batch(
                BATCH_SIZE, n_examples, vocab_size, train_funcs, rng)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                   targets.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Final eval
        model.eval()
        with torch.no_grad():
            val_in, _, val_last = generate_batch(
                512, n_examples, vocab_size, test_funcs, np.random.default_rng(0))
            val_logits = model(val_in)
            val_pred = val_logits[:, -1].argmax(dim=-1)
            val_acc = (val_pred == val_last).float().mean().item()

        results.append({"eps": eps, "val_acc": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_eps = eps

    return best_eps, best_val, results


def run_task_variation_experiment():
    """Experiment B: How does critical ε depend on task properties?"""
    print("=" * 70)
    print("EXPERIMENT B: Task Variation — Does Critical ε Scale?")
    print("=" * 70)
    print()

    eps_grid = [0.00, 0.05, 0.10, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]

    # Test B1: Vary n_examples (sequence length)
    print("Test B1: Vary n_examples (seq_len = 2*(n+1))")
    print("-" * 60)

    n_examples_list = [2, 4, 6, 8, 12, 16]
    results_b1 = []
    t0 = time.perf_counter()

    for n_ex in n_examples_list:
        seq_len = 2 * (n_ex + 1)
        best_eps, best_val, sweep = find_critical_eps(
            vocab_size=16, n_examples=n_ex, eps_values=eps_grid)
        results_b1.append({"n_examples": n_ex, "seq_len": seq_len,
                          "critical_eps": best_eps, "best_val": best_val,
                          "sweep": sweep})
        elapsed = time.perf_counter() - t0
        print(f"  n_ex={n_ex:2d} (T={seq_len:2d})  ε_c={best_eps:.2f}  "
              f"val={best_val:.3f}  ({elapsed:.0f}s)")

    # Test B2: Vary vocab_size
    print("\nTest B2: Vary vocab_size (fixed n_examples=8)")
    print("-" * 60)

    vocab_sizes = [8, 12, 16, 24, 32]
    results_b2 = []

    for vs in vocab_sizes:
        best_eps, best_val, sweep = find_critical_eps(
            vocab_size=vs, n_examples=8, eps_values=eps_grid)
        results_b2.append({"vocab_size": vs, "critical_eps": best_eps,
                          "best_val": best_val, "sweep": sweep})
        elapsed = time.perf_counter() - t0
        print(f"  vocab={vs:2d}  ε_c={best_eps:.2f}  val={best_val:.3f}  ({elapsed:.0f}s)")

    # Test B3: Learnable ε across task configs → where does it converge?
    print("\nTest B3: Learnable ε across task configs")
    print("-" * 60)

    results_b3 = []
    configs = [
        (8, 4), (8, 8), (16, 4), (16, 8), (16, 12), (24, 8), (32, 8),
    ]
    for vs, n_ex in configs:
        _, final_eps, val_acc, _ = train_learnable(
            vocab_size=vs, n_examples=n_ex, seed=42, init_eps=0.15)
        eps_mean = np.mean(list(final_eps.values()))
        results_b3.append({"vocab_size": vs, "n_examples": n_ex,
                          "learned_eps": eps_mean, "val_acc": val_acc,
                          "final_eps": final_eps})
        elapsed = time.perf_counter() - t0
        print(f"  vocab={vs:2d} n_ex={n_ex:2d} (T={2*(n_ex+1):2d})  "
              f"learned_ε={eps_mean:.4f}  val={val_acc:.3f}  ({elapsed:.0f}s)")

    total = time.perf_counter() - t0
    print(f"\nExperiment B complete in {total:.0f}s ({total/60:.1f}min)")

    plot_task_variation_results(results_b1, results_b2, results_b3)
    return results_b1, results_b2, results_b3


def plot_task_variation_results(results_b1, results_b2, results_b3):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # B1: ε curves for different n_examples
    ax = axes[0, 0]
    for r in results_b1:
        eps = [s["eps"] for s in r["sweep"]]
        vals = [s["val_acc"] for s in r["sweep"]]
        ax.plot(eps, vals, "o-", label=f"n={r['n_examples']} (T={r['seq_len']})",
                linewidth=1.5, markersize=5)
    ax.set_xlabel("ε (evaporation rate)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("B1: Phase curves vs sequence length")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # B1: Critical ε vs sequence length
    ax = axes[0, 1]
    seq_lens = [r["seq_len"] for r in results_b1]
    crit_eps = [r["critical_eps"] for r in results_b1]
    ax.plot(seq_lens, crit_eps, "o-", color="tab:red", linewidth=2, markersize=10)
    for r in results_b1:
        ax.annotate(f"n={r['n_examples']}", (r["seq_len"], r["critical_eps"]),
                    fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Sequence Length (T)")
    ax.set_ylabel("Critical ε")
    ax.set_title("B1: Critical ε vs Sequence Length")
    # Overlay 1/T fit
    T_arr = np.array(seq_lens, dtype=float)
    if np.std(crit_eps) > 0.001:
        # Fit ε_c = a / T + b
        A = np.vstack([1.0/T_arr, np.ones_like(T_arr)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, np.array(crit_eps), rcond=None)
        T_fit = np.linspace(min(seq_lens), max(seq_lens), 50)
        ax.plot(T_fit, coeffs[0] / T_fit + coeffs[1], "--", color="gray",
                label=f"fit: ε_c ≈ {coeffs[0]:.1f}/T + {coeffs[1]:.2f}")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # B2: ε curves for different vocab sizes
    ax = axes[0, 2]
    for r in results_b2:
        eps = [s["eps"] for s in r["sweep"]]
        vals = [s["val_acc"] for s in r["sweep"]]
        ax.plot(eps, vals, "o-", label=f"V={r['vocab_size']}", linewidth=1.5, markersize=5)
    ax.set_xlabel("ε (evaporation rate)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("B2: Phase curves vs vocab size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # B2: Critical ε vs vocab size
    ax = axes[1, 0]
    vocabs = [r["vocab_size"] for r in results_b2]
    crit_eps_v = [r["critical_eps"] for r in results_b2]
    ax.plot(vocabs, crit_eps_v, "s-", color="tab:blue", linewidth=2, markersize=10)
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Critical ε")
    ax.set_title("B2: Critical ε vs Vocab Size")
    ax.grid(True, alpha=0.3)

    # B3: Learned ε vs sequence length
    ax = axes[1, 1]
    for r in results_b3:
        T = 2 * (r["n_examples"] + 1)
        ax.scatter(T, r["learned_eps"], s=r["vocab_size"] * 5,
                   c=[r["val_acc"]], cmap="viridis", vmin=0, vmax=1,
                   edgecolors="k", linewidths=0.5, zorder=5)
        ax.annotate(f"V={r['vocab_size']}", (T, r["learned_eps"]),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Sequence Length (T)")
    ax.set_ylabel("Learned ε")
    ax.set_title("B3: Where does learned ε converge?")
    ax.grid(True, alpha=0.3)

    # B3: Learned ε vs best fixed ε
    ax = axes[1, 2]
    # Cross-reference with B1/B2 sweep results where configs match
    all_learned = [r["learned_eps"] for r in results_b3]
    all_vals = [r["val_acc"] for r in results_b3]
    configs = [f"V{r['vocab_size']}_n{r['n_examples']}" for r in results_b3]
    x = np.arange(len(configs))
    ax.bar(x, all_learned, color="tab:blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=7, rotation=45)
    ax.set_ylabel("Learned ε")
    ax.set_title("B3: Learned ε across task configs")
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="ε=0.15 (Exp 2 critical)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment B: Does Critical ε Scale with Task Properties?", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theory_task_variation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: theory_task_variation.png")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT C: Mechanistic Inspection
# ═══════════════════════════════════════════════════════════════

def run_mechanistic_experiment():
    """Experiment C: What does the pheromone field compute at the critical point?"""
    print("=" * 70)
    print("EXPERIMENT C: Mechanistic Inspection at Critical ε")
    print("=" * 70)
    print()

    vocab_size = 16
    n_examples = 8
    seq_len = 2 * (n_examples + 1)

    # Train three models: no pheromone, critical ε=0.15, frozen ε=0.00
    configs = [
        ("critical_0.15", 0.15),
        ("frozen_0.00", 0.00),
        ("high_0.30", 0.30),
    ]

    models = {}
    train_funcs, test_funcs = make_functions(vocab_size, seed=0)
    t0 = time.perf_counter()

    for name, eps in configs:
        print(f"Training {name}...")
        rng = np.random.default_rng(42)
        torch.manual_seed(42)

        model = GPT(vocab_size, seq_len - 1, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                     attn_cls=FixedStigmergicAttention, evap_rate=eps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                      weight_decay=FIXED_WD, betas=(0.85, 0.99))
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=TRAIN_STEPS)

        for step in range(TRAIN_STEPS):
            model.train()
            input_ids, targets, last_tgt = generate_batch(
                BATCH_SIZE, n_examples, vocab_size, train_funcs, rng)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                   targets.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_in, _, val_last = generate_batch(
                256, n_examples, vocab_size, test_funcs, np.random.default_rng(0))
            val_logits = model(val_in)
            val_pred = val_logits[:, -1].argmax(dim=-1)
            val_acc = (val_pred == val_last).float().mean().item()

        models[name] = model
        elapsed = time.perf_counter() - t0
        print(f"  {name}: val={val_acc:.3f} ({elapsed:.0f}s)")

    # Now inspect: run a batch of test sequences through each model
    print("\nInspecting field and attention patterns...")

    # Generate a consistent inspection batch
    inspect_rng = np.random.default_rng(123)
    inspect_in, _, inspect_last = generate_batch(
        32, n_examples, vocab_size, test_funcs, inspect_rng)

    inspection_data = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            # Run with decomposed output
            logits = model(inspect_in, return_decomposed=True)
            pred = logits[:, -1].argmax(dim=-1)
            acc = (pred == inspect_last).float().mean().item()

            # Collect per-layer data
            layers_data = []
            for li, block in enumerate(model.blocks):
                attn = block.attn
                data = {
                    "attention_with_field": attn.last_attn.cpu().numpy(),  # (B, H, T, T)
                    "field": attn._last_field.cpu().numpy() if attn._last_field is not None else None,
                    "deposits": attn._last_deposits.cpu().numpy() if attn._last_deposits is not None else None,
                }
                if attn._last_att_no_bias is not None:
                    data["attention_no_field"] = attn._last_att_no_bias.cpu().numpy()
                layers_data.append(data)

            inspection_data[name] = {
                "layers": layers_data,
                "accuracy": acc,
                "predictions": pred.cpu().numpy(),
            }
            print(f"  {name}: inspect_acc={acc:.3f}")

    plot_mechanistic_results(inspection_data, inspect_in.numpy())

    total = time.perf_counter() - t0
    print(f"\nExperiment C complete in {total:.0f}s ({total/60:.1f}min)")


def plot_mechanistic_results(inspection_data, input_tokens):
    """Visualize what the pheromone field actually does."""
    # Sequence structure: x0 y0 x1 y1 ... x8 y8
    # Positions: 0=x0, 1=y0, 2=x1, 3=y1, ..., 16=x8 (query)
    # Even positions = x (input), odd positions = y (output), last = query

    n_positions = input_tokens.shape[1]  # 17
    pos_labels = []
    for i in range(n_positions):
        if i % 2 == 0:
            pos_labels.append(f"x{i//2}")
        else:
            pos_labels.append(f"y{i//2}")

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))

    crit_data = inspection_data.get("critical_0.15", {})
    frozen_data = inspection_data.get("frozen_0.00", {})

    for col, (name, data) in enumerate([("critical_0.15", crit_data),
                                         ("frozen_0.00", frozen_data)]):
        if not data or not data["layers"]:
            continue

        # Layer 0, Head 0: attention with field (averaged over batch)
        L0 = data["layers"][0]
        att_with = L0["attention_with_field"].mean(axis=0)  # (H, T, T)

        # Row 0: Attention maps with field (head 0 and head 1)
        for h_idx in range(2):
            ax = axes[0, col * 2 + h_idx]
            im = ax.imshow(att_with[h_idx], cmap="viridis", aspect="auto")
            ax.set_title(f"{name}\nL0 H{h_idx} attention (with field)", fontsize=9)
            ax.set_xticks(range(n_positions))
            ax.set_xticklabels(pos_labels, fontsize=6, rotation=45)
            ax.set_yticks(range(n_positions))
            ax.set_yticklabels(pos_labels, fontsize=6)
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 1: Field values and deposits for critical model
    if crit_data and crit_data["layers"]:
        L0 = crit_data["layers"][0]

        # Deposits: mean over batch (B, T, H) → (T, H)
        if L0["deposits"] is not None:
            deposits_mean = L0["deposits"].mean(axis=0)  # (T, H)
            ax = axes[1, 0]
            im = ax.imshow(deposits_mean.T, cmap="RdBu_r", aspect="auto")
            ax.set_title("Critical: L0 Deposits (mean over batch)", fontsize=9)
            ax.set_xticks(range(n_positions))
            ax.set_xticklabels(pos_labels, fontsize=6, rotation=45)
            ax.set_yticks(range(N_HEAD))
            ax.set_yticklabels([f"H{h}" for h in range(N_HEAD)], fontsize=8)
            ax.set_xlabel("Position")
            ax.set_ylabel("Head")
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Field values: (B, T, H) → (T, H)
        if L0["field"] is not None:
            field_mean = L0["field"].mean(axis=0)  # (T, H)
            ax = axes[1, 1]
            im = ax.imshow(field_mean.T, cmap="hot", aspect="auto")
            ax.set_title("Critical: L0 Field values (mean over batch)", fontsize=9)
            ax.set_xticks(range(n_positions))
            ax.set_xticklabels(pos_labels, fontsize=6, rotation=45)
            ax.set_yticks(range(N_HEAD))
            ax.set_yticklabels([f"H{h}" for h in range(N_HEAD)], fontsize=8)
            ax.set_xlabel("Position")
            ax.set_ylabel("Head")
            plt.colorbar(im, ax=ax, fraction=0.046)

    # Frozen model deposits and field
    if frozen_data and frozen_data["layers"]:
        L0 = frozen_data["layers"][0]
        if L0["deposits"] is not None:
            deposits_mean = L0["deposits"].mean(axis=0)
            ax = axes[1, 2]
            im = ax.imshow(deposits_mean.T, cmap="RdBu_r", aspect="auto")
            ax.set_title("Frozen: L0 Deposits (mean over batch)", fontsize=9)
            ax.set_xticks(range(n_positions))
            ax.set_xticklabels(pos_labels, fontsize=6, rotation=45)
            ax.set_yticks(range(N_HEAD))
            ax.set_yticklabels([f"H{h}" for h in range(N_HEAD)], fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)

        if L0["field"] is not None:
            field_mean = L0["field"].mean(axis=0)
            ax = axes[1, 3]
            im = ax.imshow(field_mean.T, cmap="hot", aspect="auto")
            ax.set_title("Frozen: L0 Field values (mean over batch)", fontsize=9)
            ax.set_xticks(range(n_positions))
            ax.set_xticklabels(pos_labels, fontsize=6, rotation=45)
            ax.set_yticks(range(N_HEAD))
            ax.set_yticklabels([f"H{h}" for h in range(N_HEAD)], fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Attention difference (with field - without field) for critical model
    if crit_data and crit_data["layers"]:
        L0 = crit_data["layers"][0]
        if "attention_no_field" in L0 and L0["attention_no_field"] is not None:
            att_with = L0["attention_with_field"].mean(axis=0)
            att_without = L0["attention_no_field"].mean(axis=0)
            att_diff = att_with - att_without

            for h_idx in range(2):
                ax = axes[2, h_idx]
                vmax = np.abs(att_diff[h_idx]).max()
                im = ax.imshow(att_diff[h_idx], cmap="RdBu_r", aspect="auto",
                               vmin=-vmax, vmax=vmax)
                ax.set_title(f"Critical: L0 H{h_idx}\nΔAttention (field effect)", fontsize=9)
                ax.set_xticks(range(n_positions))
                ax.set_xticklabels(pos_labels, fontsize=6, rotation=45)
                ax.set_yticks(range(n_positions))
                ax.set_yticklabels(pos_labels, fontsize=6)
                ax.set_xlabel("Key position")
                ax.set_ylabel("Query position")
                plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2 col 2-3: Deposit statistics — do x positions vs y positions differ?
    if crit_data and crit_data["layers"]:
        L0 = crit_data["layers"][0]
        if L0["deposits"] is not None:
            # Separate x-positions (even) and y-positions (odd)
            deposits = L0["deposits"]  # (B, T, H)
            x_pos = list(range(0, n_positions, 2))
            y_pos = list(range(1, n_positions, 2))

            ax = axes[2, 2]
            x_deps = deposits[:, x_pos, :].reshape(-1)
            y_deps = deposits[:, y_pos, :].reshape(-1)
            ax.hist(x_deps, bins=50, alpha=0.5, label="x positions (inputs)",
                    density=True, color="tab:blue")
            ax.hist(y_deps, bins=50, alpha=0.5, label="y positions (outputs)",
                    density=True, color="tab:red")
            ax.set_xlabel("Deposit value")
            ax.set_ylabel("Density")
            ax.set_title("Critical: Deposit distribution\nx vs y positions", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Per-head mean deposit at x vs y positions
            ax = axes[2, 3]
            x_mean = deposits[:, x_pos, :].mean(axis=(0, 1))  # (H,)
            y_mean = deposits[:, y_pos, :].mean(axis=(0, 1))  # (H,)
            x_pos_data = np.arange(N_HEAD)
            w = 0.35
            ax.bar(x_pos_data - w/2, x_mean, w, label="x pos (input)", color="tab:blue")
            ax.bar(x_pos_data + w/2, y_mean, w, label="y pos (output)", color="tab:red")
            ax.set_xticks(x_pos_data)
            ax.set_xticklabels([f"H{h}" for h in range(N_HEAD)])
            ax.set_ylabel("Mean deposit")
            ax.set_title("Critical: Per-head deposit\nby position type", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment C: What Does the Pheromone Field Compute?", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theory_mechanistic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: theory_mechanistic.png")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Theory grounding experiments")
    parser.add_argument("--mode", choices=["learnable", "task_variation", "mechanistic", "all"],
                        default="all")
    args = parser.parse_args()

    if args.mode in ("learnable", "all"):
        run_learnable_experiment()
        print()

    if args.mode in ("task_variation", "all"):
        run_task_variation_experiment()
        print()

    if args.mode in ("mechanistic", "all"):
        run_mechanistic_experiment()
        print()

    print("All experiments complete.")


if __name__ == "__main__":
    main()
