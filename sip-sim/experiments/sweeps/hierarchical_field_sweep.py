"""
Experiment: Hierarchical Stigmergic Attention.

Tests the multi-scale and cross-layer field extensions from Section 8 of the
stigmergic attention paper.  Uses a 4-layer model with:

  1. Multi-scale fields: each head gets a fast and slow field at different
     evaporation rates.  Both bias attention logits additively.
  2. Cross-layer fields: each layer's slow field terminal state seeds the
     next layer's slow field initial condition via a learned scalar gate.
  3. Baseline comparisons: standard attention (no field), single-scale
     uniform field, per-head multi-scale (no cross-layer).

Task: state tracking — predict the sign of the running sum of +1/-1 tokens.
This is provably hard for bounded-depth transformers and trivially easy with
a scalar recurrence, making it the ideal testbed for the field mechanism.

Sequence structure:
  [t_1, t_2, ..., t_T, QUERY, ANSWER]
  where t_i ∈ {+1, -1}, QUERY is a special token, ANSWER ∈ {POS, NEG}.

Usage:
    uv run python -u learn/hierarchical_field_sweep.py
    uv run python -u learn/hierarchical_field_sweep.py --mode single --config feedback_only
    uv run python -u learn/hierarchical_field_sweep.py --mode single --config hierarchical --verbose
    uv run python -u learn/hierarchical_field_sweep.py --seq-len 64
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

# Vocab: 0 = +1 token, 1 = -1 token, 2 = QUERY, 3 = POS, 4 = NEG
VOCAB_SIZE = 5
TOK_PLUS = 0
TOK_MINUS = 1
TOK_QUERY = 2
TOK_POS = 3
TOK_NEG = 4

DEFAULT_SEQ_LEN = 34      # 32 +/- tokens + QUERY + ANSWER
D_MODEL = 64
N_HEAD = 4
N_LAYER = 4
BATCH_SIZE = 64
BASE_LR = 3e-3
TRAIN_STEPS = 8000
MEASURE_EVERY = 100
SEED = 42

FIXED_DROPOUT = 0.05
FIXED_WD = 0.10

# Field parameters
EVAP_FAST = 0.15           # fast field: τ ≈ 6 positions
EVAP_SLOW = 0.02           # slow field: τ ≈ 50 positions

# Seeds for multi-seed runs
SEEDS = [42, 137, 2024]


# ── Model components ────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class HierarchicalStigmergicAttention(nn.Module):
    """Stigmergic attention with optional fast/slow fields and cross-layer input.

    Modes (controlled by constructor args):
      - no field:     use_field=False
      - single-scale: use_field=True, use_slow=False
      - multi-scale:  use_field=True, use_slow=True
      - cross-layer:  use_field=True, use_slow=True, receives field_init from previous layer
    """

    def __init__(self, d_model, n_head, dropout, max_len,
                 evap_fast=0.15, evap_slow=0.02,
                 use_field=True, use_slow=False, feedback=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.use_field = use_field
        self.use_slow = use_slow
        self.feedback = feedback

        # Standard Q/K/V
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        if self.use_field:
            # Deposit projection: input (or output) → scalar per head
            self.w_deposit = nn.Linear(d_model, n_head, bias=False)

            # Fast field decay matrix
            positions = torch.arange(max_len).float()
            dist = positions.unsqueeze(0) - positions.unsqueeze(1)
            decay_fast = (1.0 - evap_fast) ** dist
            decay_fast = torch.tril(decay_fast)
            self.register_buffer("decay_fast", decay_fast)

            if self.use_slow:
                # Slow field decay matrix
                decay_slow = (1.0 - evap_slow) ** dist
                decay_slow = torch.tril(decay_slow)
                self.register_buffer("decay_slow", decay_slow)

                # Projection from fast field → slow field deposits
                self.w_fast_to_slow = nn.Linear(n_head, n_head, bias=False)

                # Cross-layer gate: scalar per head, maps incoming field_init
                self.cross_layer_gate = nn.Parameter(torch.zeros(n_head))

        self.last_attn = None
        self._last_field_fast = None
        self._last_field_slow = None

    def forward(self, x, field_init=None):
        """
        Args:
            x: (B, T, d_model) input embeddings
            field_init: (B, n_head) optional cross-layer slow field initial condition
        Returns:
            out: (B, T, d_model) attention output
        """
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if self.use_field:
            if self.feedback:
                # Feedback mode: sequential, deposits from output
                field_bias = self._feedback_field(x, q, k, v, att, field_init)
            else:
                # Non-feedback mode: parallel, deposits from input
                field_bias = self._input_field(x, field_init)

            att = att + field_bias

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        return out

    def _input_field(self, x, field_init):
        """Non-feedback field: deposits from input, fully parallelisable."""
        B, T, C = x.shape
        deposits = self.w_deposit(x)  # (B, T, n_head)

        # Fast field
        D_f = self.decay_fast[:T, :T]
        field_fast = torch.einsum("ij,bjh->bih", D_f, deposits)  # (B, T, n_head)
        self._last_field_fast = field_fast.detach()

        if self.use_slow:
            # Slow field deposits come from fast field values
            slow_deposits = self.w_fast_to_slow(field_fast)  # (B, T, n_head)
            D_s = self.decay_slow[:T, :T]
            field_slow = torch.einsum("ij,bjh->bih", D_s, slow_deposits)

            # Cross-layer: add initial condition that decays over positions
            if field_init is not None:
                gate = torch.sigmoid(self.cross_layer_gate)  # (n_head,)
                init_contrib = gate * field_init  # (B, n_head)
                # Decay the initial condition: α_slow^i * init
                alpha_slow = 1.0 - self.decay_slow[0, 0].new_tensor(
                    1.0 - (self.decay_slow[1, 0] / self.decay_slow[0, 0]).item()
                    if T > 1 else 0.02)
                # Use the first column of decay_slow as the decay profile
                decay_profile = self.decay_slow[:T, 0]  # (T,) = α^0, α^1, ..., α^{T-1}
                # init_contrib: (B, n_head), decay_profile: (T,)
                init_field = torch.einsum("bh,t->bth", init_contrib, decay_profile)
                field_slow = field_slow + init_field

            self._last_field_slow = field_slow.detach()
            field_total = field_fast + field_slow
        else:
            field_total = field_fast

        # Reshape to attention bias: (B, n_head, 1, T)
        return field_total.permute(0, 2, 1).unsqueeze(2)

    def _feedback_field(self, x, q, k, v, att_logits, field_init):
        """Feedback field: deposits from attention output, sequential over positions."""
        B, T, C = x.shape
        device = x.device

        alpha_f = 1.0 - self.decay_fast.new_tensor(
            1.0 - self.decay_fast[1, 0].item() / self.decay_fast[0, 0].item()
            if T > 1 else EVAP_FAST)

        if self.use_slow:
            # Extract alpha_slow from decay matrix
            alpha_s = self.decay_slow[1, 0].item() if T > 1 else (1.0 - EVAP_SLOW)
        else:
            alpha_s = None

        # Initialise per-head field states
        phi_fast = torch.zeros(B, self.n_head, device=device)
        phi_slow = torch.zeros(B, self.n_head, device=device) if self.use_slow else None

        # Cross-layer initial condition for slow field
        if self.use_slow and field_init is not None:
            gate = torch.sigmoid(self.cross_layer_gate)
            phi_slow = gate * field_init  # (B, n_head)

        # Collect field biases per position
        field_biases = torch.zeros(B, self.n_head, T, device=device)

        # We need to accumulate deposits; the deposit from position i-1 feeds position i
        prev_deposit = torch.zeros(B, self.n_head, device=device)
        prev_slow_deposit = torch.zeros(B, self.n_head, device=device)

        for i in range(T):
            # Update fields with previous deposit
            phi_fast = alpha_f * phi_fast + prev_deposit
            if self.use_slow:
                phi_slow = alpha_s * phi_slow + prev_slow_deposit

            # Record bias for this position (bias on key side)
            if self.use_slow:
                field_biases[:, :, i] = phi_fast + phi_slow
            else:
                field_biases[:, :, i] = phi_fast

            # Compute attention at position i using field bias up to i
            logits_i = att_logits[:, :, i, :i+1]  # (B, n_head, i+1)
            bias_i = field_biases[:, :, :i+1]     # (B, n_head, i+1)
            logits_i = logits_i + bias_i
            weights_i = F.softmax(logits_i, dim=-1)  # (B, n_head, i+1)

            # Compute output at position i
            v_i = v[:, :, :i+1, :]  # (B, n_head, i+1, head_dim)
            o_i = (weights_i.unsqueeze(-1) * v_i).sum(dim=2)  # (B, n_head, head_dim)
            # Concatenate heads → (B, d_model)
            o_i_cat = o_i.transpose(1, 2).contiguous().view(B, self.d_model)
            o_i_proj = self.out_proj(o_i_cat)  # (B, d_model)

            # Deposit from output
            prev_deposit = self.w_deposit(o_i_proj)  # (B, n_head)

            # Slow field deposits from fast field
            if self.use_slow:
                prev_slow_deposit = self.w_fast_to_slow(phi_fast.detach())

        self._last_field_fast = phi_fast.detach()
        if self.use_slow:
            self._last_field_slow = phi_slow.detach()

        return field_biases.unsqueeze(2)  # (B, n_head, 1, T)

    def get_terminal_slow_field(self):
        """Return terminal slow field state for cross-layer passing.
        Returns (B, n_head) — the field value at the last sequence position."""
        if self._last_field_slow is None:
            return None
        if self._last_field_slow.ndim == 3:
            return self._last_field_slow[:, -1, :]  # (B, T, n_head) → (B, n_head)
        return self._last_field_slow  # already (B, n_head) from feedback mode

    def get_field_snapshot(self):
        """Return fast field for instrumentation."""
        return self._last_field_fast


class HierarchicalBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len,
                 evap_fast=0.15, evap_slow=0.02,
                 use_field=True, use_slow=False, feedback=False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = HierarchicalStigmergicAttention(
            d_model, n_head, dropout, max_len,
            evap_fast, evap_slow, use_field, use_slow, feedback)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, field_init=None):
        x = x + self.attn(self.ln1(x), field_init=field_init)
        x = x + self.mlp(self.ln2(x))
        return x


class HierarchicalGPT(nn.Module):
    """4-layer GPT with optional hierarchical stigmergic fields.

    Configurations:
      - no_field:       standard transformer, no pheromone
      - feedback_only:  single-scale feedback field (Section 3.4 of paper)
      - multiscale:     fast+slow fields, no cross-layer (Section 8.1)
      - hierarchical:   fast+slow fields + cross-layer passing (Section 8.1+8.2)
    """

    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 evap_fast=0.15, evap_slow=0.02,
                 use_field=True, use_slow=False, use_crosslayer=False,
                 feedback=False):
        super().__init__()
        self.use_crosslayer = use_crosslayer
        self.use_slow = use_slow

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            HierarchicalBlock(d_model, n_head, dropout, max_len,
                              evap_fast, evap_slow, use_field, use_slow, feedback)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Cross-layer projections: map terminal slow field from layer L to
        # initial condition of layer L+1
        if use_crosslayer and use_slow:
            self.cross_projs = nn.ModuleList([
                nn.Linear(n_head, n_head, bias=False)
                for _ in range(n_layer - 1)
            ])

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        field_init = None
        for i, block in enumerate(self.blocks):
            x = block(x, field_init=field_init)

            # Pass slow field terminal state to next layer
            if self.use_crosslayer and self.use_slow and i < len(self.blocks) - 1:
                terminal = block.attn.get_terminal_slow_field()
                if terminal is not None:
                    field_init = self.cross_projs[i](terminal)
                else:
                    field_init = None
            else:
                field_init = None

        return self.head(self.ln_f(x))

    def get_attention_maps(self):
        return [block.attn.last_attn for block in self.blocks]

    def get_field_snapshots(self):
        return [(block.attn._last_field_fast, block.attn._last_field_slow)
                for block in self.blocks]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Experiment configurations ───────────────────────────────────

def make_config(name, seq_len):
    """Return model kwargs for a named configuration."""
    max_len = seq_len - 1  # input length (without last token)
    base = dict(vocab_size=VOCAB_SIZE, max_len=max_len, d_model=D_MODEL,
                n_head=N_HEAD, n_layer=N_LAYER, dropout=FIXED_DROPOUT)

    configs = {
        # Baseline: no field at all
        "no_field": {**base, "use_field": False},

        # Single-scale input field (non-feedback, Section 3.3 negative result)
        "input_only": {**base, "use_field": True, "use_slow": False,
                       "feedback": False, "evap_fast": EVAP_FAST},

        # Single-scale feedback field (Section 3.4)
        "feedback_only": {**base, "use_field": True, "use_slow": False,
                          "feedback": True, "evap_fast": EVAP_FAST},

        # Multi-scale: fast + slow fields, no cross-layer (Section 8.1)
        "multiscale": {**base, "use_field": True, "use_slow": True,
                       "use_crosslayer": False, "feedback": False,
                       "evap_fast": EVAP_FAST, "evap_slow": EVAP_SLOW},

        # Multi-scale + feedback (Section 8.1 with feedback deposits)
        "multiscale_feedback": {**base, "use_field": True, "use_slow": True,
                                "use_crosslayer": False, "feedback": True,
                                "evap_fast": EVAP_FAST, "evap_slow": EVAP_SLOW},

        # Full hierarchical: multi-scale + cross-layer (Sections 8.1 + 8.2)
        "hierarchical": {**base, "use_field": True, "use_slow": True,
                         "use_crosslayer": True, "feedback": False,
                         "evap_fast": EVAP_FAST, "evap_slow": EVAP_SLOW},

        # Full hierarchical + feedback
        "hierarchical_feedback": {**base, "use_field": True, "use_slow": True,
                                  "use_crosslayer": True, "feedback": True,
                                  "evap_fast": EVAP_FAST, "evap_slow": EVAP_SLOW},
    }

    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Choose from {list(configs.keys())}")
    return configs[name]


# ── Data generation: state tracking ─────────────────────────────

def generate_state_tracking_batch(batch_size, seq_len, rng):
    """Generate state-tracking sequences.

    Each sequence has (seq_len - 2) tokens of +1/-1, then QUERY, then ANSWER.
    ANSWER = POS if running sum > 0, NEG if running sum <= 0.

    Returns:
        input_ids: (B, seq_len - 1)  — everything except ANSWER
        targets:   (B, seq_len - 1)  — ANSWER at last position, -100 elsewhere
        labels:    (B,)              — the correct answer token
    """
    n_tokens = seq_len - 2  # number of +/- tokens
    tokens = np.zeros((batch_size, seq_len), dtype=np.int64)

    for b in range(batch_size):
        signs = rng.choice([TOK_PLUS, TOK_MINUS], size=n_tokens)
        tokens[b, :n_tokens] = signs
        tokens[b, n_tokens] = TOK_QUERY

        # Running sum: +1 maps to +1, -1 maps to -1
        running_sum = np.sum(np.where(signs == TOK_PLUS, 1, -1))
        tokens[b, n_tokens + 1] = TOK_POS if running_sum > 0 else TOK_NEG

    input_ids = tokens[:, :-1]
    targets = np.full_like(input_ids, -100)
    targets[:, -1] = tokens[:, -1]  # only predict the answer
    labels = tokens[:, -1]

    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(labels),
    )


# ── Instrumentation ─────────────────────────────────────────────

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


def field_magnitude(model):
    """Average magnitude of fast fields across layers."""
    snapshots = model.get_field_snapshots()
    mags = []
    for fast, slow in snapshots:
        if fast is not None:
            mags.append(fast.abs().mean().item())
        if slow is not None:
            mags.append(slow.abs().mean().item())
    return float(np.mean(mags)) if mags else 0.0


# ── Training ────────────────────────────────────────────────────

def train_one(config_name, seq_len, seed=SEED, verbose=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model_kwargs = make_config(config_name, seq_len)
    model = HierarchicalGPT(**model_kwargs)

    n_params = model.count_params()
    if verbose:
        print(f"  Config: {config_name}, params: {n_params:,}, seed: {seed}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=FIXED_WD, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=TRAIN_STEPS
    )

    ts = {
        "steps": [], "train_loss": [], "train_acc": [], "val_acc": [],
        "eff_rank": [], "weight_vel": [], "field_mag": [],
    }

    prev_weights = None

    for step in range(TRAIN_STEPS):
        model.train()
        input_ids, targets, labels = generate_state_tracking_batch(
            BATCH_SIZE, seq_len, rng)

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % MEASURE_EVERY == 0 or step == TRAIN_STEPS - 1:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                pred = logits[:, -1].argmax(dim=-1)
                train_acc = (pred == labels).float().mean().item()

                # Validation accuracy (fresh rng, larger batch)
                val_rng = np.random.default_rng(0)
                val_in, _, val_labels = generate_state_tracking_batch(
                    512, seq_len, val_rng)
                val_logits = model(val_in)
                val_pred = val_logits[:, -1].argmax(dim=-1)
                val_acc = (val_pred == val_labels).float().mean().item()

                er = effective_rank(model)
                fm = field_magnitude(model)

                curr_weights = weight_snapshot(model)
                wv = weight_velocity(curr_weights, prev_weights)
                prev_weights = curr_weights

            ts["steps"].append(step)
            ts["train_loss"].append(loss.item())
            ts["train_acc"].append(train_acc)
            ts["val_acc"].append(val_acc)
            ts["eff_rank"].append(er)
            ts["weight_vel"].append(wv)
            ts["field_mag"].append(fm)

            if verbose and step % (MEASURE_EVERY * 4) == 0:
                print(f"    step {step:5d}  loss={loss.item():.3f}  "
                      f"train={train_acc:.3f}  val={val_acc:.3f}  "
                      f"rank={er:.3f}  fmag={fm:.4f}")

            model.train()

    # Summary: average over last 5 measurements
    def last_n(key, n=5):
        return float(np.mean(ts[key][-n:])) if ts[key] else 0.0

    summary = {
        "config": config_name,
        "seed": seed,
        "seq_len": seq_len,
        "n_params": n_params,
        "val_acc": last_n("val_acc"),
        "train_acc": last_n("train_acc"),
        "eff_rank": last_n("eff_rank"),
        "field_mag": last_n("field_mag"),
        "weight_vel_final": last_n("weight_vel"),
    }

    return ts, summary


# ── Full sweep ──────────────────────────────────────────────────

CONFIG_NAMES = [
    "no_field",
    "input_only",
    "feedback_only",
    "multiscale",
    "multiscale_feedback",
    "hierarchical",
    "hierarchical_feedback",
]


def run_sweep(seq_len):
    print(f"Hierarchical Stigmergic Attention Sweep")
    print(f"Task: state tracking (running sum sign)")
    print(f"Sequence length: {seq_len} ({seq_len - 2} +/- tokens)")
    print(f"Model: {N_LAYER}L {N_HEAD}H d={D_MODEL}")
    print(f"ε_fast={EVAP_FAST}, ε_slow={EVAP_SLOW}")
    print(f"Seeds: {SEEDS}")
    print(f"Configs: {CONFIG_NAMES}")
    print()

    # Show param counts
    for name in CONFIG_NAMES:
        kwargs = make_config(name, seq_len)
        m = HierarchicalGPT(**kwargs)
        print(f"  {name:30s}  params={m.count_params():,}")
        del m
    print()

    all_results = []
    all_ts = {}
    t0 = time.perf_counter()
    total_runs = len(CONFIG_NAMES) * len(SEEDS)
    idx = 0

    for config_name in CONFIG_NAMES:
        for seed in SEEDS:
            idx += 1
            ts, summary = train_one(config_name, seq_len, seed=seed)
            all_results.append(summary)
            all_ts[(config_name, seed)] = ts

            elapsed = time.perf_counter() - t0
            eta = (elapsed / idx) * (total_runs - idx)
            print(f"  [{idx:2d}/{total_runs}] {config_name:30s} seed={seed}  "
                  f"val={summary['val_acc']:.3f}  train={summary['train_acc']:.3f}  "
                  f"rank={summary['eff_rank']:.3f}  "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    plot_sweep_results(all_results, all_ts, seq_len)
    save_csv(all_results, seq_len)
    print_summary_table(all_results)


def print_summary_table(results):
    """Print a summary table averaging over seeds."""
    print("\n=== Summary (mean ± std over seeds) ===")
    print(f"{'Config':30s}  {'Val Acc':>12s}  {'Train Acc':>12s}  {'Eff Rank':>10s}  {'Params':>8s}")
    print("-" * 80)

    for config_name in CONFIG_NAMES:
        rows = [r for r in results if r["config"] == config_name]
        if not rows:
            continue
        val = np.array([r["val_acc"] for r in rows])
        train = np.array([r["train_acc"] for r in rows])
        rank = np.array([r["eff_rank"] for r in rows])
        params = rows[0]["n_params"]
        print(f"{config_name:30s}  {val.mean():.3f} ± {val.std():.3f}  "
              f"{train.mean():.3f} ± {train.std():.3f}  "
              f"{rank.mean():.3f}     {params:>8,}")


def plot_sweep_results(results, all_ts, seq_len):
    """Plot training curves and bar chart comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Validation accuracy training curves (seed 42 only for clarity)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONFIG_NAMES)))
    for ci, config_name in enumerate(CONFIG_NAMES):
        key = (config_name, SEEDS[0])
        if key in all_ts:
            ts = all_ts[key]
            axes[0].plot(ts["steps"], ts["val_acc"],
                         color=colors[ci], label=config_name, alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title(f"Training Curves (seed={SEEDS[0]})")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # 2. Bar chart: final val accuracy (mean ± std over seeds)
    means, stds, labels = [], [], []
    for config_name in CONFIG_NAMES:
        rows = [r for r in results if r["config"] == config_name]
        vals = [r["val_acc"] for r in rows]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        labels.append(config_name)

    x = np.arange(len(labels))
    bars = axes[1].bar(x, means, yerr=stds, capsize=4,
                       color=colors[:len(labels)], edgecolor="k", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Final Accuracy (mean ± std)")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    # 3. Field magnitude training curves
    for ci, config_name in enumerate(CONFIG_NAMES):
        key = (config_name, SEEDS[0])
        if key in all_ts:
            ts = all_ts[key]
            if any(v > 0 for v in ts["field_mag"]):
                axes[2].plot(ts["steps"], ts["field_mag"],
                             color=colors[ci], label=config_name, alpha=0.8)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Field Magnitude")
    axes[2].set_title("Field Accumulation")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Hierarchical Stigmergic Attention: State Tracking (T={seq_len})\n"
                 f"{N_LAYER}L {N_HEAD}H d={D_MODEL}, ε_fast={EVAP_FAST}, ε_slow={EVAP_SLOW}",
                 fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / f"hierarchical_T{seq_len}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def save_csv(results, seq_len):
    csv_path = OUT_DIR / f"hierarchical_T{seq_len}.csv"
    with open(csv_path, "w") as f:
        cols = list(results[0].keys())
        f.write(",".join(cols) + "\n")
        for r in results:
            row = []
            for k in cols:
                v = r[k]
                if isinstance(v, float):
                    row.append(f"{v:.6f}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")
    print(f"Saved: {csv_path.name}")


# ── Single run mode ─────────────────────────────────────────────

def run_single(config_name, seq_len, seed=SEED):
    print(f"Single run: {config_name}, T={seq_len}, seed={seed}")
    print(f"Model: {N_LAYER}L {N_HEAD}H d={D_MODEL}")
    print(f"ε_fast={EVAP_FAST}, ε_slow={EVAP_SLOW}")

    ts, summary = train_one(config_name, seq_len, seed=seed, verbose=True)

    print(f"\nResults:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.4f}")
        else:
            print(f"  {k:25s} = {v}")


# ── Diagnostic probes ───────────────────────────────────────────

def run_diagnose(config_name, seq_len, seed=SEED):
    """Train a model then run mechanistic probes to check if the field
    is doing something non-trivial.

    Probes:
      1. Field ablation: zero out field at inference, measure accuracy drop
      2. Field-sum correlation: does the field track the running sum?
      3. Deposit sign alignment: do deposits match token signs?
      4. Per-layer field contribution: which layers' fields matter most?
    """
    print(f"=== DIAGNOSTIC MODE: {config_name}, T={seq_len}, seed={seed} ===")
    print(f"Model: {N_LAYER}L {N_HEAD}H d={D_MODEL}")
    print()

    # ── Train ───────────────────────────────────────────────────
    print("Phase 1: Training...")
    ts, summary = train_one(config_name, seq_len, seed=seed, verbose=True)
    print(f"\nTrained val_acc: {summary['val_acc']:.3f}")
    print()

    # Rebuild model at end state (train_one doesn't return model, so retrain)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model_kwargs = make_config(config_name, seq_len)
    model = HierarchicalGPT(**model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=FIXED_WD, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=TRAIN_STEPS)

    for step in range(TRAIN_STEPS):
        model.train()
        inp, tgt, labels = generate_state_tracking_batch(BATCH_SIZE, seq_len, rng)
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()

    # ── Probe data ──────────────────────────────────────────────
    probe_rng = np.random.default_rng(9999)
    N_PROBE = 1024
    probe_in, probe_tgt, probe_labels = generate_state_tracking_batch(
        N_PROBE, seq_len, probe_rng)

    n_tokens = seq_len - 2

    # Compute true running sums for each sequence
    # tokens: positions 0..n_tokens-1 are +/- tokens
    true_sums = np.zeros((N_PROBE, n_tokens))
    for b in range(N_PROBE):
        cumsum = 0
        for i in range(n_tokens):
            tok = probe_in[b, i].item()
            cumsum += 1 if tok == TOK_PLUS else -1
            true_sums[b, i] = cumsum

    # ── Probe 1: Field ablation ─────────────────────────────────
    print("Phase 2: Probes")
    print()
    print("Probe 1: Field ablation (zero out field at inference)")

    with torch.no_grad():
        # Normal inference
        logits_normal = model(probe_in)
        pred_normal = logits_normal[:, -1].argmax(dim=-1)
        acc_normal = (pred_normal == probe_labels).float().mean().item()

        # Ablated inference: temporarily zero all decay matrices
        saved_states = {}
        for li, block in enumerate(model.blocks):
            attn = block.attn
            if attn.use_field:
                saved_states[li] = {
                    'decay_fast': attn.decay_fast.clone(),
                }
                attn.decay_fast.zero_()
                if attn.use_slow and hasattr(attn, 'decay_slow'):
                    saved_states[li]['decay_slow'] = attn.decay_slow.clone()
                    attn.decay_slow.zero_()

        logits_ablated = model(probe_in)
        pred_ablated = logits_ablated[:, -1].argmax(dim=-1)
        acc_ablated = (pred_ablated == probe_labels).float().mean().item()

        # Restore
        for li, states in saved_states.items():
            attn = model.blocks[li].attn
            attn.decay_fast.copy_(states['decay_fast'])
            if 'decay_slow' in states:
                attn.decay_slow.copy_(states['decay_slow'])

    delta = acc_normal - acc_ablated
    print(f"  Accuracy with field:    {acc_normal:.3f}")
    print(f"  Accuracy without field: {acc_ablated:.3f}")
    print(f"  Delta (field contrib):  {delta:+.3f}")
    if abs(delta) < 0.02:
        print(f"  → FIELD IS INERT (delta < 0.02)")
    elif delta > 0.05:
        print(f"  → FIELD IS HELPFUL (delta > 0.05)")
    else:
        print(f"  → FIELD HAS MARGINAL EFFECT")
    print()

    # ── Probe 2: Field-sum correlation ──────────────────────────
    print("Probe 2: Does the field track the running sum?")

    with torch.no_grad():
        _ = model(probe_in)  # populate field snapshots
        snapshots = model.get_field_snapshots()

    # For non-feedback input-deposit models, the fast field is (B, T, n_head)
    # For feedback models, it's (B, n_head) — only terminal state
    # We need per-position fields, so this probe works for non-feedback modes
    for li, (fast, slow) in enumerate(snapshots):
        if fast is None:
            continue

        if fast.ndim == 3:
            # (B, T, n_head) — we have per-position fields
            field_np = fast.cpu().numpy()[:, :n_tokens, :]  # (B, n_tokens, n_head)

            # Correlate each head's field with the running sum
            for h in range(N_HEAD):
                field_h = field_np[:, :, h].ravel()
                sums_flat = true_sums.ravel()
                r = np.corrcoef(field_h, sums_flat)[0, 1]
                print(f"  Layer {li}, Head {h}: field-sum corr = {r:+.3f}")
        elif fast.ndim == 2:
            # (B, n_head) — only terminal state from feedback mode
            field_np = fast.cpu().numpy()  # (B, n_head)
            final_sums = true_sums[:, -1]  # (B,)
            for h in range(N_HEAD):
                r = np.corrcoef(field_np[:, h], final_sums)[0, 1]
                print(f"  Layer {li}, Head {h}: terminal field-sum corr = {r:+.3f}")
    print()

    # ── Probe 3: Deposit sign alignment ─────────────────────────
    print("Probe 3: Do deposits match token signs?")

    with torch.no_grad():
        # Get deposits from the first layer's attention module
        for li, block in enumerate(model.blocks):
            attn = block.attn
            if not attn.use_field:
                continue

            # Compute deposits from input (what w_deposit produces)
            x_normed = block.ln1(model.drop(
                model.tok_emb(probe_in) +
                model.pos_emb(torch.arange(probe_in.shape[1], device=probe_in.device))
            ))
            if li > 0:
                # For deeper layers we'd need to propagate through earlier blocks
                # Only probe layer 0 for simplicity
                break

            deposits = attn.w_deposit(x_normed)  # (B, T, n_head)
            dep_np = deposits.cpu().numpy()[:, :n_tokens, :]

            for h in range(N_HEAD):
                dep_h = dep_np[:, :, h].ravel()
                # Token signs: +1 for TOK_PLUS, -1 for TOK_MINUS
                tok_signs = np.where(probe_in[:, :n_tokens].numpy() == TOK_PLUS, 1.0, -1.0).ravel()
                r = np.corrcoef(dep_h, tok_signs)[0, 1]
                # Also: fraction of deposits with correct sign
                sign_match = np.mean(np.sign(dep_h) == tok_signs)
                print(f"  Layer {li}, Head {h}: deposit-sign corr = {r:+.3f}, "
                      f"sign match = {sign_match:.1%}")

            break  # only layer 0
    print()

    # ── Probe 4: Per-layer ablation ─────────────────────────────
    print("Probe 4: Per-layer field contribution")

    with torch.no_grad():
        for target_layer in range(N_LAYER):
            # Zero out field in only this layer
            saved = {}
            attn = model.blocks[target_layer].attn
            if not attn.use_field:
                continue
            saved['decay_fast'] = attn.decay_fast.clone()
            attn.decay_fast.zero_()
            if attn.use_slow and hasattr(attn, 'decay_slow'):
                saved['decay_slow'] = attn.decay_slow.clone()
                attn.decay_slow.zero_()

            logits_abl = model(probe_in)
            pred_abl = logits_abl[:, -1].argmax(dim=-1)
            acc_abl = (pred_abl == probe_labels).float().mean().item()

            attn.decay_fast.copy_(saved['decay_fast'])
            if 'decay_slow' in saved:
                attn.decay_slow.copy_(saved['decay_slow'])

            drop = acc_normal - acc_abl
            print(f"  Ablate layer {target_layer}: acc={acc_abl:.3f} "
                  f"(drop={drop:+.3f})")

    print()

    # ── Summary verdict ─────────────────────────────────────────
    print("=== DIAGNOSTIC SUMMARY ===")
    print(f"  Config:            {config_name}")
    print(f"  Val accuracy:      {acc_normal:.3f}")
    print(f"  Field ablation Δ:  {delta:+.3f}")
    if delta > 0.05:
        print(f"  VERDICT: Field is mechanistically active — proceed to full sweep")
    elif delta > 0.02:
        print(f"  VERDICT: Field has marginal effect — may need tuning")
    else:
        print(f"  VERDICT: Field appears inert — investigate before scaling up")

    # ── Plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training curve
    axes[0].plot(ts["steps"], ts["val_acc"], "b-", label="val_acc")
    axes[0].plot(ts["steps"], ts["train_acc"], "r--", alpha=0.5, label="train_acc")
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Field-sum scatter (layer 0, head 0 if available)
    if snapshots[0][0] is not None and snapshots[0][0].ndim == 3:
        f0 = snapshots[0][0].cpu().numpy()[:, :n_tokens, 0].ravel()
        s0 = true_sums.ravel()
        axes[1].scatter(s0, f0, alpha=0.05, s=2, c="steelblue")
        r = np.corrcoef(f0, s0)[0, 1]
        axes[1].set_xlabel("True Running Sum")
        axes[1].set_ylabel("Field Value (L0 H0)")
        axes[1].set_title(f"Field vs Running Sum (r={r:.3f})")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "N/A\n(feedback mode:\nno per-position field)",
                     ha="center", va="center", fontsize=11)
        axes[1].set_title("Field vs Running Sum")

    # Ablation bar chart
    layer_drops = []
    with torch.no_grad():
        for target_layer in range(N_LAYER):
            attn = model.blocks[target_layer].attn
            if not attn.use_field:
                layer_drops.append(0.0)
                continue
            saved = {'decay_fast': attn.decay_fast.clone()}
            attn.decay_fast.zero_()
            if attn.use_slow and hasattr(attn, 'decay_slow'):
                saved['decay_slow'] = attn.decay_slow.clone()
                attn.decay_slow.zero_()
            logits_abl = model(probe_in)
            acc_abl = (logits_abl[:, -1].argmax(-1) == probe_labels).float().mean().item()
            attn.decay_fast.copy_(saved['decay_fast'])
            if 'decay_slow' in saved:
                attn.decay_slow.copy_(saved['decay_slow'])
            layer_drops.append(acc_normal - acc_abl)

    colors_bar = ["steelblue" if d > 0 else "salmon" for d in layer_drops]
    axes[2].bar(range(N_LAYER), layer_drops, color=colors_bar, edgecolor="k", linewidth=0.5)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Accuracy Drop")
    axes[2].set_title("Per-Layer Field Ablation")
    axes[2].axhline(y=0, color="gray", linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Diagnostics: {config_name} (T={seq_len}, seed={seed})\n"
                 f"val_acc={acc_normal:.3f}, field_Δ={delta:+.3f}", fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / f"diagnose_{config_name}_T{seq_len}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path.name}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Stigmergic Attention sweep")
    parser.add_argument("--mode", choices=["sweep", "single", "diagnose"],
                        default="sweep")
    parser.add_argument("--config", type=str, default="hierarchical",
                        help=f"Config for single/diagnose mode: {CONFIG_NAMES}")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="Sequence length (default: 34)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "single":
        run_single(args.config, args.seq_len, seed=args.seed)
    elif args.mode == "diagnose":
        run_diagnose(args.config, args.seq_len, seed=args.seed)
    else:
        run_sweep(args.seq_len)


if __name__ == "__main__":
    main()
