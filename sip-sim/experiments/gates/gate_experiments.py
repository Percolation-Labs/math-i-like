"""
Gate experiments for Stigmergic Attention theory.

Three gates that determine whether this architecture is worth scaling:

Gate 1: PE Ablation
  Is the pheromone field just a positional encoding?
  Compare: {learned PE, no PE, sinusoidal PE, RoPE} × {with field, without field}
  If the field helps WITH good PEs, it's computing something PEs can't.

Gate 2: Linear Recurrence
  Verify that field[i] = (1-ε)·field[i-1] + deposit[i] produces identical
  outputs to the matrix formulation. Then benchmark training speed.

Gate 3: Second Task
  In-context retrieval (lookup table) instead of modular arithmetic.
  Does ε_c ≈ 1.6/T still hold? Does the field learn task-appropriate markers?

Usage:
    uv run python -u learn/gate_experiments.py --gate 1
    uv run python -u learn/gate_experiments.py --gate 2
    uv run python -u learn/gate_experiments.py --gate 3
    uv run python -u learn/gate_experiments.py --gate all
"""

import time
import math
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
FIXED_EVAP = 0.20  # critical ε for correct (decaying) field

N_TRAIN_FUNCS = 200
N_TEST_FUNCS = 40


# ── Model components ───────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class RotaryEmbedding(nn.Module):
    """RoPE positional encoding."""

    def __init__(self, dim, max_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, T, device):
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        return torch.cat([freqs, freqs], dim=-1)  # (T, dim)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, freqs):
    """Apply RoPE to tensor x of shape (B, H, T, D)."""
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class StigmergicAttention(nn.Module):
    """Attention with optional pheromone field bias.

    field_mode controls the field behavior:
      "decay"    — correct: field[i] = (1-ε)·field[i-1] + deposit[i], α < 1
      "amplify"  — buggy original: field[i] = (1/(1-ε))·field[i-1] + deposit[i], α > 1
      "bounded"  — amplify but clamp field magnitude to prevent explosion
      "feedback" — deposits from attention OUTPUT (closes the feedback loop)
                   field → attention → output → deposit → field
    """

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.15,
                 use_field=True, use_rope=False, field_mode="decay"):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.use_field = use_field
        self.use_rope = use_rope
        self.field_mode = field_mode

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        if use_field:
            self.w_deposit = nn.Linear(d_model, n_head, bias=False)
            if field_mode in ("decay", "bounded", "feedback"):
                # Recurrent field computation
                self.retain = 1.0 - evap_rate
            if field_mode == "decay":
                positions = torch.arange(max_len).float()
                decay_matrix = (1.0 - evap_rate) ** (
                    positions.unsqueeze(1) - positions.unsqueeze(0))
                decay_matrix = torch.tril(decay_matrix)
                self.register_buffer("decay_matrix", decay_matrix)
            elif field_mode == "amplify":
                positions = torch.arange(max_len).float()
                # Original buggy: D[i,j] = (1-ε)^(j-i) → amplifies old deposits
                decay_matrix = (1.0 - evap_rate) ** (
                    positions.unsqueeze(0) - positions.unsqueeze(1))
                decay_matrix = torch.tril(decay_matrix)
                self.register_buffer("decay_matrix", decay_matrix)
            # "bounded" and "feedback" use recurrent computation in forward()

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_len)

        self.last_attn = None
        self._last_field = None

    def _compute_field_bounded(self, deposits):
        """Bounded amplification: α > 1 recurrence with field clamping."""
        B, T, H = deposits.shape
        alpha = 1.0 / self.retain  # α = 1/(1-ε) > 1
        fields = []
        prev = deposits[:, 0, :]
        fields.append(prev)
        for i in range(1, T):
            prev = torch.clamp(alpha * prev + deposits[:, i, :], -10.0, 10.0)
            fields.append(prev)
        return torch.stack(fields, dim=1)

    def _forward_feedback(self, x):
        """Feedback mode: deposits from attention output, position-by-position.

        Closes the loop: field → attention → output → deposit → field

        The field is per-position: φ[j,h] = deposit at position j decayed by
        α^(i-j) when read at position i. This matches the non-feedback version
        structurally (field_bias at key position j biases attention toward j)
        but the deposit at j comes from j's attention OUTPUT, not j's INPUT.

        Instead of maintaining per-position state across all T steps (O(T²)),
        we accumulate deposits into a vector and apply the decay matrix once.
        The sequential constraint is only that position j's deposit depends
        on the field from positions 0..j-1.
        """
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            freqs = self.rope(T, x.device)
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        # Pre-compute raw attention scores (before field bias and masking)
        raw_att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Causal mask
        mask = self.mask[:, :, :T, :T]

        # Collect deposits position-by-position, then form the field
        # deposits: (B, T, H) — filled in sequentially
        deposits = torch.zeros(B, T, self.n_head, device=x.device, dtype=x.dtype)
        # Running field state via recurrence: field[i] = α·field[i-1] + deposit[i]
        field_state = torch.zeros(B, self.n_head, device=x.device, dtype=x.dtype)
        # field_at_pos[i] = field state just BEFORE position i processes
        field_at_pos = torch.zeros(B, T, self.n_head, device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(T):
            # 1. Build field bias for position i's attention
            # field_at_pos[j] for j<i = recurrence state after processing j
            # Position i sees this as the "pheromone" at each past position
            field_bias = field_at_pos.permute(0, 2, 1).unsqueeze(2)  # (B, H, 1, T)

            # 2. Attention at position i with field bias
            att_i = raw_att[:, :, i:i+1, :] + field_bias  # (B, H, 1, T)
            att_i = att_i.masked_fill(mask[:, :, i:i+1, :] == 0, float("-inf"))
            att_i = F.softmax(att_i, dim=-1)
            att_i = self.attn_drop(att_i)

            # 3. Compute output at position i
            out_i = (att_i @ v).squeeze(2)  # (B, H, head_dim)
            out_i = out_i.transpose(1, 2).contiguous().view(B, C)  # (B, C)
            outputs.append(out_i)

            # 4. Deposit from output (not input!)
            deposit_i = self.w_deposit(out_i)  # (B, H)
            deposits[:, i, :] = deposit_i

            # 5. Update recurrence: field_state = α·field_state + deposit
            field_state = self.retain * field_state + deposit_i
            # Store for next iteration (i+1 will see this as field_at_pos[i])
            if i < T - 1:
                field_at_pos[:, i + 1, :] = field_state

        # Stack outputs: (B, T, C)
        out = torch.stack(outputs, dim=1)
        out = self.resid_drop(self.out_proj(out))

        # Store field for diagnostics
        self._last_field = field_at_pos.detach()

        return out

    def forward(self, x):
        B, T, C = x.shape

        if self.use_field and self.field_mode == "feedback":
            return self._forward_feedback(x)

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            freqs = self.rope(T, x.device)
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if self.use_field:
            deposits = self.w_deposit(x)
            if self.field_mode == "bounded":
                field = self._compute_field_bounded(deposits)
            else:
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


class StigmergicAttentionRecurrent(nn.Module):
    """Same architecture but field computed via linear recurrence (scan)."""

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.15):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.evap_rate = evap_rate
        self.retain = 1.0 - evap_rate

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_deposit = nn.Linear(d_model, n_head, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        # Also keep the matrix version for equivalence checking
        positions = torch.arange(max_len).float()
        # D[i,j] = (1-ε)^(i-j): old deposits DECAY (correct sign)
        decay_matrix = (1.0 - evap_rate) ** (
            positions.unsqueeze(1) - positions.unsqueeze(0))
        decay_matrix = torch.tril(decay_matrix)
        self.register_buffer("decay_matrix", decay_matrix)

        self._last_field = None

    def _field_recurrent(self, deposits):
        """Compute field via linear recurrence: field[i] = retain * field[i-1] + deposit[i]"""
        B, T, H = deposits.shape
        field = torch.zeros(B, T, H, device=deposits.device, dtype=deposits.dtype)
        field[:, 0, :] = deposits[:, 0, :]
        for i in range(1, T):
            field[:, i, :] = self.retain * field[:, i-1, :] + deposits[:, i, :]
        return field

    def _field_matrix(self, deposits):
        """Compute field via matrix multiply (original formulation)."""
        T = deposits.shape[1]
        D = self.decay_matrix[:T, :T]
        return torch.einsum("ij,bjh->bih", D, deposits)

    def forward(self, x, mode="recurrent"):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        deposits = self.w_deposit(x)
        if mode == "recurrent":
            field = self._field_recurrent(deposits)
        else:
            field = self._field_matrix(deposits)

        self._last_field = field.detach()
        field_bias = field.permute(0, 2, 1).unsqueeze(2)
        att = att + field_bias

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        return out


class DualChannelAttention(nn.Module):
    """Dual-channel attention: individual lookup + social field routing.

    The social field is a per-head vector recurrence whose output modulates
    the key space of attention.  Deposits come from attention output (closing
    the feedback loop).

    mod_type controls how the field modulates keys:
      "additive"   — k' = k + W_mod @ field  (Architecture I)
      "gating"     — k' = k * sigmoid(W_gate @ field + b)  (Architecture II)
    """

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.10,
                 use_rope=False, mod_type="additive", field_dim=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.use_rope = use_rope
        self.mod_type = mod_type
        self.d_f = field_dim if field_dim is not None else self.head_dim

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        # Social field: deposit projection (output → field space)
        self.w_deposit = nn.Linear(d_model, n_head * self.d_f, bias=False)
        self.retain = 1.0 - evap_rate

        # Key modulation projection (field space → key space)
        if mod_type == "additive":
            self.w_mod = nn.Linear(self.d_f, self.head_dim, bias=False)
        elif mod_type == "gating":
            self.w_gate = nn.Linear(self.d_f, self.head_dim)
            # Initialize bias so gates start near 1 (field starts as no-op)
            nn.init.constant_(self.w_gate.bias, 2.0)

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_len)

        self._last_field = None

    def forward(self, x):
        B, T, C = x.shape
        H, d_h, d_f = self.n_head, self.head_dim, self.d_f

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)  # (B, H, T, d_h)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        if self.use_rope:
            freqs = self.rope(T, x.device)
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        # Sequential processing: position-by-position for feedback loop
        field_state = torch.zeros(B, H, d_f, device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(T):
            # 1. Modulate keys at all positions j <= i using current social field
            k_i_raw = k[:, :, i:i+1, :]  # current position key (unmodulated)

            if i > 0:
                field_expanded = field_history[:, :, :i, :]  # (B, H, i, d_f)
                k_prev = k[:, :, :i, :]  # keys at positions 0..i-1

                if self.mod_type == "additive":
                    mod = self.w_mod(field_expanded)  # (B, H, i, d_h)
                    k_prev_mod = k_prev + mod
                elif self.mod_type == "gating":
                    gates = torch.sigmoid(self.w_gate(field_expanded))
                    k_prev_mod = k_prev * gates

                # Concatenate modulated past keys with unmodulated current key
                k_mod = torch.cat([k_prev_mod, k_i_raw], dim=2)
            else:
                k_mod = k_i_raw

            # 2. Attention at position i with modulated keys
            q_i = q[:, :, i:i+1, :]  # (B, H, 1, d_h)
            att_i = (q_i @ k_mod.transpose(-2, -1)) * (d_h ** -0.5)  # (B, H, 1, i+1)

            # Causal mask (only need positions 0..i)
            att_i = F.softmax(att_i, dim=-1)
            att_i = self.attn_drop(att_i)

            # 3. Aggregate
            v_i = v[:, :, :i+1, :]  # (B, H, i+1, d_h)
            out_i = (att_i @ v_i).squeeze(2)  # (B, H, d_h)
            out_flat = out_i.transpose(1, 2).contiguous().view(B, C)  # (B, C)
            outputs.append(out_flat)

            # 4. Deposit from output (injection)
            deposit = self.w_deposit(out_flat)  # (B, H * d_f)
            deposit = deposit.view(B, H, d_f)

            # 5. Update social field (erosion + injection)
            field_state = self.retain * field_state + deposit

            # Store field history for key modulation at future positions
            if i == 0:
                field_history = field_state.unsqueeze(2)  # (B, H, 1, d_f)
            else:
                field_history = torch.cat([
                    field_history, field_state.unsqueeze(2)], dim=2)

        out = torch.stack(outputs, dim=1)  # (B, T, C)
        out = self.resid_drop(self.out_proj(out))
        self._last_field = field_state.detach()
        return out


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.15,
                 use_field=True, use_rope=False, field_mode="decay",
                 attn_type="stigmergic", mod_type="additive", field_dim=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        if attn_type == "dual_channel":
            self.attn = DualChannelAttention(
                d_model, n_head, dropout, max_len, evap_rate,
                use_rope=use_rope, mod_type=mod_type, field_dim=field_dim)
        else:
            self.attn = StigmergicAttention(
                d_model, n_head, dropout, max_len, evap_rate,
                use_field=use_field, use_rope=use_rope, field_mode=field_mode)
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


class GPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 evap_rate=0.15, use_field=True, pe_type="learned", use_rope=False,
                 field_mode="decay", attn_type="stigmergic", mod_type="additive",
                 field_dim=None):
        """
        pe_type: "learned", "sinusoidal", "none"
        use_rope: if True, applies RoPE inside attention (orthogonal to pe_type)
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pe_type = pe_type

        if pe_type == "learned":
            self.pos_emb = nn.Embedding(max_len, d_model)
        elif pe_type == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                 -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_emb", pe)
        # pe_type == "none": no positional encoding

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, max_len, evap_rate,
                  use_field=use_field, use_rope=use_rope, field_mode=field_mode,
                  attn_type=attn_type, mod_type=mod_type, field_dim=field_dim)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)

        if self.pe_type == "learned":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif self.pe_type == "sinusoidal":
            x = x + self.pos_emb[:T]
        # pe_type == "none": just token embeddings

        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data generation ────────────────────────────────────────────

def make_functions(vocab_size, seed=0):
    rng = np.random.default_rng(seed)
    all_funcs = [(a, b) for a in range(1, vocab_size) for b in range(vocab_size)]
    rng.shuffle(all_funcs)
    n_train = min(N_TRAIN_FUNCS, int(len(all_funcs) * 0.8))
    n_test = min(N_TEST_FUNCS, len(all_funcs) - n_train)
    return all_funcs[:n_train], all_funcs[n_train:n_train + n_test]


def generate_batch_modular(batch_size, n_examples, vocab_size, functions, rng):
    """Original task: f(x) = (ax+b) mod V."""
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


def generate_batch_retrieval(batch_size, n_examples, vocab_size, rng):
    """Gate 3 task: in-context retrieval (lookup table).

    Each sequence defines a random mapping for n_examples keys, then queries one.
    Format: k0 v0 k1 v1 ... k_{n-1} v_{n-1} k_query v_query
    Keys are unique within a sequence. Values are random.
    The model must retrieve the value associated with k_query from the examples.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        # Pick n_total unique keys
        keys = rng.choice(vocab_size, size=n_total, replace=False)
        # Random values
        values = rng.integers(0, vocab_size, size=n_total)

        for i in range(n_total):
            tokens[b, 2 * i] = keys[i]
            tokens[b, 2 * i + 1] = values[i]

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    # Only predict values (odd positions)
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]
    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


def generate_batch_majority(batch_size, n_examples, vocab_size, rng):
    """Gate 3 task: in-context majority classification.

    Each sequence picks a random threshold t ∈ [1, V-1] and a random
    permutation of 2 class labels from [0, V-1].
    Format: x0 c0 x1 c1 ... x_{n-1} c_{n-1} x_query c_query
    where c_i = label_a if x_i < t, else label_b.

    The model must learn the threshold and label mapping from examples,
    then classify the query.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        # Random threshold (at least 1 item on each side)
        t = int(rng.integers(1, vocab_size))
        # Two distinct class labels
        labels = rng.choice(vocab_size, size=2, replace=False)

        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            c = labels[0] if x < t else labels[1]
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = c

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


def generate_batch_permutation(batch_size, n_examples, vocab_size, rng):
    """Gate 3 task: in-context permutation learning.

    Each sequence picks a random permutation π of [0, V-1].
    Format: x0 π(x0) x1 π(x1) ... x_query π(x_query)
    The model must learn π from examples and apply it to the query.

    This is harder than modular arithmetic (no algebraic structure to exploit)
    but easier than retrieval (the permutation is consistent across all examples,
    so the model can build up knowledge progressively).
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        perm = rng.permutation(vocab_size)
        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = perm[x]

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


# ── Training ──────────────────────────────────────────────────

def train_config(vocab_size, n_examples, seed, evap_rate, use_field,
                 pe_type, use_rope, generate_fn, gen_kwargs,
                 train_steps=TRAIN_STEPS, field_mode="decay",
                 val_gen_kwargs=None):
    """Train a single configuration. Returns (final_val_acc, timeseries).

    val_gen_kwargs: if provided, used for validation instead of gen_kwargs.
                   Use this to evaluate on held-out test functions.
    """
    seq_len = 2 * (n_examples + 1)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if val_gen_kwargs is None:
        val_gen_kwargs = gen_kwargs

    model = GPT(vocab_size, seq_len - 1, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                evap_rate=evap_rate, use_field=use_field,
                pe_type=pe_type, use_rope=use_rope, field_mode=field_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=FIXED_WD, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps)

    ts = {"steps": [], "val_acc": [], "train_acc": []}

    for step in range(train_steps):
        model.train()
        input_ids, targets, last_tgt = generate_fn(
            BATCH_SIZE, n_examples, vocab_size, rng=rng, **gen_kwargs)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               targets.view(-1), ignore_index=-100)
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

                val_in, _, val_last = generate_fn(
                    256, n_examples, vocab_size,
                    rng=np.random.default_rng(0), **val_gen_kwargs)
                val_logits = model(val_in)
                val_pred = val_logits[:, -1].argmax(dim=-1)
                val_acc = (val_pred == val_last).float().mean().item()

            ts["steps"].append(step)
            ts["val_acc"].append(val_acc)
            ts["train_acc"].append(train_acc)
            model.train()

    final_val = float(np.mean(ts["val_acc"][-5:]))
    return final_val, ts, model


# ── Wrapper for modular arithmetic task ──────────────────────

def gen_modular(batch_size, n_examples, vocab_size, rng, functions):
    return generate_batch_modular(batch_size, n_examples, vocab_size, functions, rng)


def gen_retrieval(batch_size, n_examples, vocab_size, rng):
    return generate_batch_retrieval(batch_size, n_examples, vocab_size, rng)


def gen_majority(batch_size, n_examples, vocab_size, rng):
    return generate_batch_majority(batch_size, n_examples, vocab_size, rng)


def gen_permutation(batch_size, n_examples, vocab_size, rng):
    return generate_batch_permutation(batch_size, n_examples, vocab_size, rng)


# ── State-tracking tasks ──────────────────────────────────────

def generate_batch_cumsum(batch_size, n_examples, vocab_size, rng):
    """State-tracking task: cumulative sum mod V.

    Format: x0 s0 x1 s1 ... xn sn
    where si = (x0 + x1 + ... + xi) mod V.

    This requires maintaining a running counter. Standard attention must
    attend to ALL previous x positions and sum them — O(T) work per
    position. A running state does it in O(1) per position.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        running_sum = 0
        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            running_sum = (running_sum + x) % vocab_size
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = running_sum

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    # Only predict cumsum positions (odd positions in the token stream)
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]
    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


def generate_batch_parity(batch_size, n_examples, vocab_size, rng):
    """State-tracking task: running parity.

    Format: x0 p0 x1 p1 ... xn pn
    where pi = (x0 + x1 + ... + xi) mod 2.

    Inputs are random in [0, V-1] but answer is always 0 or 1.
    Parity is a classic hard problem for feedforward/attention models.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        running_sum = 0
        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            running_sum = (running_sum + x) % 2
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = running_sum

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


def generate_batch_flipflop(batch_size, n_examples, vocab_size, rng):
    """State-tracking task: flip-flop state machine.

    State machine with V states. Input tokens are commands:
      - Token 0: no-op (state unchanged)
      - Token k (1 <= k < V): set state to k
    Answer at each position is the current state.

    Format: c0 s0 c1 s1 ... cn sn

    This tests whether the model can maintain and update a register.
    The challenge grows with sequence length: the model must track the
    LAST non-zero command, which could be arbitrarily far back.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        state = int(rng.integers(1, vocab_size))  # initial state
        for i in range(n_total):
            # ~50% no-op, ~50% set new state
            if rng.random() < 0.5:
                cmd = 0
            else:
                cmd = int(rng.integers(1, vocab_size))
                state = cmd
            tokens[b, 2 * i] = cmd
            tokens[b, 2 * i + 1] = state

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


def gen_cumsum(batch_size, n_examples, vocab_size, rng):
    return generate_batch_cumsum(batch_size, n_examples, vocab_size, rng)


def gen_parity(batch_size, n_examples, vocab_size, rng):
    return generate_batch_parity(batch_size, n_examples, vocab_size, rng)


def gen_flipflop(batch_size, n_examples, vocab_size, rng):
    return generate_batch_flipflop(batch_size, n_examples, vocab_size, rng)


# ═══════════════════════════════════════════════════════════════
# GATE 1: PE Ablation
# ═══════════════════════════════════════════════════════════════

def run_gate1():
    """Is the pheromone field just a positional encoding?"""
    print("=" * 70)
    print("GATE 1: PE Ablation — Is the Field Just a Positional Encoding?")
    print("=" * 70)
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)

    # 4 PE types × 2 field settings = 8 configs, 3 seeds each = 24 runs
    pe_configs = [
        ("learned", False),    # standard learned PE
        ("sinusoidal", False), # fixed sinusoidal PE
        ("none", False),       # no PE at all
        ("learned", True),     # learned PE + RoPE in attention
    ]
    field_settings = [True, False]  # with and without pheromone field
    seeds = [42, 137, 256]

    results = []
    t0 = time.perf_counter()
    idx = 0
    total = len(pe_configs) * len(field_settings) * len(seeds)

    for pe_type, use_rope in pe_configs:
        for use_field in field_settings:
            vals = []
            for seed in seeds:
                idx += 1
                val, ts, _ = train_config(
                    VOCAB_SIZE, N_EXAMPLES, seed, FIXED_EVAP, use_field,
                    pe_type, use_rope,
                    generate_fn=gen_modular,
                    gen_kwargs={"functions": train_funcs})

                vals.append(val)
                elapsed = time.perf_counter() - t0
                eta = (elapsed / idx) * (total - idx)

            name = f"{'rope+' if use_rope else ''}{pe_type}"
            field_str = "field" if use_field else "no_field"
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            results.append({
                "pe_type": pe_type, "use_rope": use_rope,
                "use_field": use_field, "name": f"{name}+{field_str}",
                "mean_val": mean_val, "std_val": std_val,
                "vals": vals,
            })
            print(f"  {name:20s} {field_str:10s}  "
                  f"val={mean_val:.3f}±{std_val:.3f}  "
                  f"[{', '.join(f'{v:.3f}' for v in vals)}]  "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    total_time = time.perf_counter() - t0
    print(f"\nGate 1 complete in {total_time:.0f}s ({total_time/60:.1f}min)")

    # Analysis: compute field benefit per PE type
    print("\n=== Field benefit by PE type ===")
    pe_names = []
    field_benefits = []
    for pe_type, use_rope in pe_configs:
        name = f"{'rope+' if use_rope else ''}{pe_type}"
        with_field = [r for r in results
                      if r["pe_type"] == pe_type and r["use_rope"] == use_rope
                      and r["use_field"]][0]
        no_field = [r for r in results
                    if r["pe_type"] == pe_type and r["use_rope"] == use_rope
                    and not r["use_field"]][0]
        benefit = with_field["mean_val"] - no_field["mean_val"]
        pe_names.append(name)
        field_benefits.append(benefit)
        print(f"  {name:20s}  Δval = {benefit:+.3f}  "
              f"({no_field['mean_val']:.3f} → {with_field['mean_val']:.3f})")

    plot_gate1(results, pe_names, field_benefits)
    return results


def plot_gate1(results, pe_names, field_benefits):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Bar chart: val accuracy for each config
    ax = axes[0]
    names = [r["name"] for r in results]
    vals = [r["mean_val"] for r in results]
    stds = [r["std_val"] for r in results]
    colors = ["tab:blue" if r["use_field"] else "tab:gray" for r in results]
    x = np.arange(len(names))
    ax.bar(x, vals, yerr=stds, color=colors, capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Val Accuracy by Configuration\n(blue=field, gray=no field)")
    ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle="--", alpha=0.3, label="Random")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Field benefit per PE type
    ax = axes[1]
    x = np.arange(len(pe_names))
    colors_b = ["tab:green" if b > 0.02 else "tab:red" if b < -0.02 else "tab:gray"
                for b in field_benefits]
    ax.bar(x, field_benefits, color=colors_b, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pe_names, fontsize=9)
    ax.set_ylabel("Δ Validation Accuracy (field − no field)")
    ax.set_title("Field Benefit by PE Type\n(green=helps, red=hurts)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Grouped comparison
    ax = axes[2]
    with_vals = [r["mean_val"] for r in results if r["use_field"]]
    without_vals = [r["mean_val"] for r in results if not r["use_field"]]
    ax.scatter(without_vals, with_vals, s=100, c="tab:blue", edgecolors="k", zorder=5)
    for i, name in enumerate(pe_names):
        ax.annotate(name, (without_vals[i], with_vals[i]),
                    fontsize=8, ha="left", va="bottom")
    lim_min = min(min(with_vals), min(without_vals)) - 0.05
    lim_max = max(max(with_vals), max(without_vals)) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Val Acc WITHOUT field")
    ax.set_ylabel("Val Acc WITH field")
    ax.set_title("Field vs No Field\n(above diagonal = field helps)")
    ax.legend(fontsize=8)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Gate 1: Is the Pheromone Field Just a Positional Encoding?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate1_pe_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: gate1_pe_ablation.png")


# ═══════════════════════════════════════════════════════════════
# GATE 2: Linear Recurrence Equivalence
# ═══════════════════════════════════════════════════════════════

def run_gate2():
    """Verify recurrent ↔ matrix equivalence, benchmark speed."""
    print("=" * 70)
    print("GATE 2: Linear Recurrence — Matrix vs Scan Equivalence")
    print("=" * 70)
    print()

    torch.manual_seed(SEED)

    # Build a single attention layer
    attn = StigmergicAttentionRecurrent(D_MODEL, N_HEAD, 0.0, SEQ_LEN, FIXED_EVAP)
    attn.eval()

    # Random input
    x = torch.randn(4, SEQ_LEN - 1, D_MODEL)

    # Compare outputs
    with torch.no_grad():
        out_matrix = attn(x, mode="matrix")
        out_recurrent = attn(x, mode="recurrent")

    max_diff = (out_matrix - out_recurrent).abs().max().item()
    mean_diff = (out_matrix - out_recurrent).abs().mean().item()

    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Equivalent: {'YES' if max_diff < 1e-5 else 'NO'}")

    # Also check field values
    deposits = attn.w_deposit(x)
    field_mat = attn._field_matrix(deposits)
    field_rec = attn._field_recurrent(deposits)
    field_diff = (field_mat - field_rec).abs().max().item()
    print(f"  Field max difference:     {field_diff:.2e}")

    # Benchmark
    print("\n  Benchmarking (100 iterations, T=17)...")
    import timeit

    def bench_matrix():
        with torch.no_grad():
            attn(x, mode="matrix")

    def bench_recurrent():
        with torch.no_grad():
            attn(x, mode="recurrent")

    t_mat = timeit.timeit(bench_matrix, number=100) / 100
    t_rec = timeit.timeit(bench_recurrent, number=100) / 100
    print(f"  Matrix:    {t_mat*1000:.2f}ms per forward")
    print(f"  Recurrent: {t_rec*1000:.2f}ms per forward")
    print(f"  Ratio: {t_rec/t_mat:.2f}x")

    # Benchmark at longer sequence length
    print("\n  Benchmarking at T=128...")
    attn_long = StigmergicAttentionRecurrent(D_MODEL, N_HEAD, 0.0, 256, FIXED_EVAP)
    attn_long.eval()
    x_long = torch.randn(4, 128, D_MODEL)

    def bench_matrix_long():
        with torch.no_grad():
            attn_long(x_long, mode="matrix")

    def bench_recurrent_long():
        with torch.no_grad():
            attn_long(x_long, mode="recurrent")

    t_mat_l = timeit.timeit(bench_matrix_long, number=50) / 50
    t_rec_l = timeit.timeit(bench_recurrent_long, number=50) / 50
    print(f"  Matrix:    {t_mat_l*1000:.2f}ms per forward")
    print(f"  Recurrent: {t_rec_l*1000:.2f}ms per forward")
    print(f"  Ratio: {t_rec_l/t_mat_l:.2f}x")

    # Verify gradients match
    print("\n  Verifying gradient equivalence...")
    attn.train()
    x_grad = torch.randn(4, SEQ_LEN - 1, D_MODEL, requires_grad=True)

    out_m = attn(x_grad, mode="matrix")
    loss_m = out_m.sum()
    loss_m.backward()
    grad_m = x_grad.grad.clone()

    x_grad.grad = None
    out_r = attn(x_grad, mode="recurrent")
    loss_r = out_r.sum()
    loss_r.backward()
    grad_r = x_grad.grad.clone()

    grad_diff = (grad_m - grad_r).abs().max().item()
    print(f"  Gradient max difference: {grad_diff:.2e}")
    print(f"  Gradients equivalent: {'YES' if grad_diff < 1e-4 else 'NO'}")

    print("\nGate 2 result: recurrence IS equivalent to matrix formulation.")
    print("The pheromone field is a 1D SSM: field[i] = (1-ε)·field[i-1] + deposit[i]")


# ═══════════════════════════════════════════════════════════════
# GATE 3: Second Task (In-Context Retrieval)
# ═══════════════════════════════════════════════════════════════

def run_gate3():
    """Does the theory generalize to different tasks?

    Tests three tasks:
    - Modular arithmetic (our main task, as control)
    - Majority classification (threshold-based, simpler)
    - Permutation learning (harder, no algebraic structure)

    For each task: ε sweep with 3 seeds, plus no-field baseline.
    """
    print("=" * 70)
    print("GATE 3: Second Task — Does the Phase Transition Generalize?")
    print("=" * 70)
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)

    eps_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    seeds = [42, 137, 256]

    tasks = [
        ("modular", gen_modular, {"functions": train_funcs}),
        ("majority", gen_majority, {}),
        ("permutation", gen_permutation, {}),
    ]

    all_results = {}
    t0 = time.perf_counter()

    for task_name, gen_fn, gen_kw in tasks:
        print(f"\n{'─'*60}")
        print(f"Task: {task_name} (vocab={VOCAB_SIZE}, n={N_EXAMPLES})")
        print(f"{'─'*60}")

        task_results = []
        for ei, eps in enumerate(eps_values):
            vals = []
            for seed in seeds:
                val, _, _ = train_config(
                    VOCAB_SIZE, N_EXAMPLES, seed, eps, use_field=True,
                    pe_type="learned", use_rope=False,
                    generate_fn=gen_fn, gen_kwargs=gen_kw)
                vals.append(val)

            mean_val = np.mean(vals)
            std_val = np.std(vals)
            task_results.append({"eps": eps, "mean_val": mean_val,
                                "std_val": std_val, "vals": vals})
            elapsed = time.perf_counter() - t0
            print(f"  ε={eps:.2f}  val={mean_val:.3f}±{std_val:.3f}  "
                  f"[{', '.join(f'{v:.3f}' for v in vals)}]")

        # No-field baseline
        baseline_vals = []
        for seed in seeds:
            val, _, _ = train_config(
                VOCAB_SIZE, N_EXAMPLES, seed, 0.0, use_field=False,
                pe_type="learned", use_rope=False,
                generate_fn=gen_fn, gen_kwargs=gen_kw)
            baseline_vals.append(val)
        baseline_mean = np.mean(baseline_vals)
        baseline_std = np.std(baseline_vals)
        print(f"  no_field  val={baseline_mean:.3f}±{baseline_std:.3f}")

        # Find best ε
        best = max(task_results, key=lambda r: r["mean_val"])
        print(f"  → Best ε={best['eps']:.2f} (val={best['mean_val']:.3f}), "
              f"benefit over no-field: {best['mean_val'] - baseline_mean:+.3f}")

        all_results[task_name] = {
            "sweep": task_results,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        }

    total_time = time.perf_counter() - t0
    print(f"\nGate 3 complete in {total_time:.0f}s ({total_time/60:.1f}min)")

    plot_gate3_v2(all_results)
    return all_results


def plot_gate3_v2(all_results):
    """Plot phase curves for all tasks side by side."""
    tasks = list(all_results.keys())
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]

    colors = {"modular": "tab:blue", "majority": "tab:green",
              "permutation": "tab:orange", "retrieval": "tab:red"}

    for ax, task_name in zip(axes, tasks):
        data = all_results[task_name]
        sweep = data["sweep"]
        eps = [r["eps"] for r in sweep]
        vals = [r["mean_val"] for r in sweep]
        stds = [r["std_val"] for r in sweep]

        color = colors.get(task_name, "tab:blue")
        ax.errorbar(eps, vals, yerr=stds, fmt="o-", color=color,
                    linewidth=2, markersize=8, capsize=4, label="With field")
        ax.axhline(y=data["baseline_mean"], color="tab:gray", linestyle="--",
                   linewidth=2, label=f"No field ({data['baseline_mean']:.3f})")
        ax.fill_between([eps[0], eps[-1]],
                        data["baseline_mean"] - data["baseline_std"],
                        data["baseline_mean"] + data["baseline_std"],
                        color="gray", alpha=0.1)
        ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle=":", alpha=0.3,
                   label="Random")
        ax.set_xlabel("ε (evaporation rate)")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(f"{task_name.title()} Task")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Gate 3: Phase Curves Across Tasks (correct decay, 3 seeds)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate3_tasks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: gate3_tasks.png")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run_sweep_correct():
    """Re-run the core ε sweep with correct decay matrix and multiple seeds.

    This establishes proper baselines since all original sweeps used the
    buggy amplify matrix.
    """
    print("=" * 70)
    print("SWEEP: Correct Decay ε Sweep (multi-seed baselines)")
    print("=" * 70)
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    eps_values = [0.00, 0.05, 0.10, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]
    seeds = [42, 137, 256, 314, 512]

    results = []
    t0 = time.perf_counter()

    for ei, eps in enumerate(eps_values):
        vals = []
        for seed in seeds:
            val, _, _ = train_config(
                VOCAB_SIZE, N_EXAMPLES, seed, eps, use_field=True,
                pe_type="learned", use_rope=False,
                generate_fn=gen_modular,
                gen_kwargs={"functions": train_funcs})
            vals.append(val)

        mean_val = np.mean(vals)
        std_val = np.std(vals)
        results.append({"eps": eps, "mean_val": mean_val, "std_val": std_val,
                        "vals": vals})
        elapsed = time.perf_counter() - t0
        eta = (elapsed / (ei + 1)) * (len(eps_values) - ei - 1)
        print(f"  ε={eps:.2f}  val={mean_val:.3f}±{std_val:.3f}  "
              f"[{', '.join(f'{v:.3f}' for v in vals)}]  "
              f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    # No-field baseline
    print("\n  No-field baseline (5 seeds)...")
    baseline_vals = []
    for seed in seeds:
        val, _, _ = train_config(
            VOCAB_SIZE, N_EXAMPLES, seed, 0.0, use_field=False,
            pe_type="learned", use_rope=False,
            generate_fn=gen_modular,
            gen_kwargs={"functions": train_funcs})
        baseline_vals.append(val)
    baseline_mean = np.mean(baseline_vals)
    baseline_std = np.std(baseline_vals)
    print(f"  no_field  val={baseline_mean:.3f}±{baseline_std:.3f}")

    total_time = time.perf_counter() - t0
    print(f"\nSweep complete in {total_time:.0f}s ({total_time/60:.1f}min)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    eps = [r["eps"] for r in results]
    vals = [r["mean_val"] for r in results]
    stds = [r["std_val"] for r in results]
    ax.errorbar(eps, vals, yerr=stds, fmt="o-", color="tab:blue",
                linewidth=2, markersize=8, capsize=4, label="With field (correct decay)")
    ax.axhline(y=baseline_mean, color="tab:gray", linestyle="--",
               linewidth=2, label=f"No field ({baseline_mean:.3f}±{baseline_std:.3f})")
    ax.fill_between([eps[0], eps[-1]],
                    baseline_mean - baseline_std, baseline_mean + baseline_std,
                    color="gray", alpha=0.1)
    ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle=":", alpha=0.3, label="Random")
    ax.set_xlabel("ε (evaporation rate)")
    ax.set_ylabel("Validation Accuracy (5 seeds)")
    ax.set_title("Correct Decay: ε Phase Curve\n(modular arithmetic, V=16, n=8)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_correct_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: sweep_correct_decay.png")

    return results


def run_field_modes():
    """Compare decay vs amplify vs bounded amplification.

    Tests whether bounded amplification captures the benefit of the
    buggy amplify matrix while being scalable (no explosion).
    """
    print("=" * 70)
    print("FIELD MODES: Decay vs Amplify vs Bounded Amplification")
    print("=" * 70)
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    eps_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    seeds = [42, 137, 256]
    modes = ["decay", "amplify", "bounded"]

    all_results = {}
    t0 = time.perf_counter()

    for mode in modes:
        print(f"\n{'─'*60}")
        print(f"Mode: {mode}")
        print(f"{'─'*60}")

        mode_results = []
        for eps in eps_values:
            vals = []
            for seed in seeds:
                val, _, _ = train_config(
                    VOCAB_SIZE, N_EXAMPLES, seed, eps, use_field=True,
                    pe_type="learned", use_rope=False,
                    generate_fn=gen_modular,
                    gen_kwargs={"functions": train_funcs},
                    val_gen_kwargs={"functions": test_funcs},
                    field_mode=mode)
                vals.append(val)

            mean_val = np.mean(vals)
            std_val = np.std(vals)
            mode_results.append({"eps": eps, "mean_val": mean_val,
                                "std_val": std_val, "vals": vals})
            print(f"  ε={eps:.2f}  val={mean_val:.3f}±{std_val:.3f}  "
                  f"[{', '.join(f'{v:.3f}' for v in vals)}]")

        all_results[mode] = mode_results

    # No-field baseline
    print(f"\n{'─'*60}")
    print("No-field baseline")
    baseline_vals = []
    for seed in seeds:
        val, _, _ = train_config(
            VOCAB_SIZE, N_EXAMPLES, seed, 0.0, use_field=False,
            pe_type="learned", use_rope=False,
            generate_fn=gen_modular,
            gen_kwargs={"functions": train_funcs},
            val_gen_kwargs={"functions": test_funcs})
        baseline_vals.append(val)
    baseline_mean = np.mean(baseline_vals)
    baseline_std = np.std(baseline_vals)
    print(f"  no_field  val={baseline_mean:.3f}±{baseline_std:.3f}")

    total_time = time.perf_counter() - t0
    print(f"\nField modes complete in {total_time:.0f}s ({total_time/60:.1f}min)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"decay": "tab:blue", "amplify": "tab:red", "bounded": "tab:green"}
    for mode in modes:
        data = all_results[mode]
        eps = [r["eps"] for r in data]
        vals = [r["mean_val"] for r in data]
        stds = [r["std_val"] for r in data]
        ax.errorbar(eps, vals, yerr=stds, fmt="o-", color=colors[mode],
                    linewidth=2, markersize=8, capsize=4, label=mode)

    ax.axhline(y=baseline_mean, color="tab:gray", linestyle="--",
               linewidth=2, label=f"No field ({baseline_mean:.3f})")
    ax.fill_between([0, 0.30], baseline_mean - baseline_std,
                    baseline_mean + baseline_std, color="gray", alpha=0.1)
    ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle=":", alpha=0.3,
               label="Random")
    ax.set_xlabel("ε (evaporation/amplification rate)")
    ax.set_ylabel("Validation Accuracy (3 seeds)")
    ax.set_title("Field Modes: Decay vs Amplify vs Bounded\n(modular arithmetic, V=16, n=8)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "field_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: field_modes.png")

    return all_results


def run_feedback():
    """Test the feedback hypothesis: deposits from attention output.

    The ant paper's recipe requires positive feedback (field → attention →
    output → deposit → field). Previous modes computed deposits from INPUT
    (no feedback loop). This test compares:
    - "decay"    — deposits from input, correct decay (baseline negative result)
    - "feedback" — deposits from output, correct decay (the fix)
    - no field   — standard attention

    ε sweep with 3 seeds each.
    """
    print("=" * 70)
    print("FEEDBACK: Deposits from Output (Closing the Loop)")
    print("=" * 70)
    print()

    train_funcs, test_funcs = make_functions(VOCAB_SIZE, seed=0)
    eps_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    seeds = [42, 137, 256]
    modes = ["decay", "feedback"]

    all_results = {}
    t0 = time.perf_counter()

    for mode in modes:
        print(f"\n{'─'*60}")
        print(f"Mode: {mode}")
        print(f"{'─'*60}")

        mode_results = []
        for ei, eps in enumerate(eps_values):
            vals = []
            for seed in seeds:
                val, _, _ = train_config(
                    VOCAB_SIZE, N_EXAMPLES, seed, eps, use_field=True,
                    pe_type="learned", use_rope=False,
                    generate_fn=gen_modular,
                    gen_kwargs={"functions": train_funcs},
                    val_gen_kwargs={"functions": test_funcs},
                    field_mode=mode)
                vals.append(val)

            mean_val = np.mean(vals)
            std_val = np.std(vals)
            mode_results.append({"eps": eps, "mean_val": mean_val,
                                "std_val": std_val, "vals": vals})
            elapsed = time.perf_counter() - t0
            print(f"  ε={eps:.2f}  val={mean_val:.3f}±{std_val:.3f}  "
                  f"[{', '.join(f'{v:.3f}' for v in vals)}]")

        all_results[mode] = mode_results

    # No-field baseline
    print(f"\n{'─'*60}")
    print("No-field baseline")
    baseline_vals = []
    for seed in seeds:
        val, _, _ = train_config(
            VOCAB_SIZE, N_EXAMPLES, seed, 0.0, use_field=False,
            pe_type="learned", use_rope=False,
            generate_fn=gen_modular,
            gen_kwargs={"functions": train_funcs},
            val_gen_kwargs={"functions": test_funcs})
        baseline_vals.append(val)
    baseline_mean = np.mean(baseline_vals)
    baseline_std = np.std(baseline_vals)
    print(f"  no_field  val={baseline_mean:.3f}±{baseline_std:.3f}")

    total_time = time.perf_counter() - t0
    print(f"\nFeedback experiment complete in {total_time:.0f}s ({total_time/60:.1f}min)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"decay": "tab:blue", "feedback": "tab:green"}
    for mode in modes:
        data = all_results[mode]
        eps = [r["eps"] for r in data]
        vals = [r["mean_val"] for r in data]
        stds = [r["std_val"] for r in data]
        label = {"decay": "Decay (deposits from input)",
                 "feedback": "Feedback (deposits from output)"}[mode]
        ax.errorbar(eps, vals, yerr=stds, fmt="o-", color=colors[mode],
                    linewidth=2, markersize=8, capsize=4, label=label)

    ax.axhline(y=baseline_mean, color="tab:gray", linestyle="--",
               linewidth=2, label=f"No field ({baseline_mean:.3f})")
    ax.fill_between([0, 0.40], baseline_mean - baseline_std,
                    baseline_mean + baseline_std, color="gray", alpha=0.1)
    ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle=":", alpha=0.3,
               label="Random")
    ax.set_xlabel("ε (evaporation rate)")
    ax.set_ylabel("Validation Accuracy (3 seeds, OOD)")
    ax.set_title("Feedback Loop Test: Does Closing the Loop Help?\n"
                 "field → attention → output → deposit → field")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feedback_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: feedback_test.png")

    return all_results


def run_statetrack():
    """The right problem: state-tracking tasks where sequential memory matters.

    Standard attention is random-access: it can query any position equally.
    But it's provably bad at maintaining running state (counting, parity,
    registers). The feedback field IS a running state — if there's a task
    where it helps, this is it.

    Tests three tasks × three modes × multiple sequence lengths:
    - cumsum:   predict running sum mod V (requires counter)
    - parity:   predict running parity (classic transformer limitation)
    - flipflop: predict state of a register (last non-zero write)

    Modes: no_field, decay (deposits from input), feedback (deposits from output)
    Lengths: n=8 (T=18), n=16 (T=34), n=32 (T=66)
    """
    print("=" * 70)
    print("STATE TRACKING: The Right Problem for Stigmergic Attention")
    print("=" * 70)
    print()

    tasks = [
        ("cumsum", gen_cumsum),
        ("parity", gen_parity),
        ("flipflop", gen_flipflop),
    ]

    modes = [
        ("no_field", False, "decay"),
        ("decay", True, "decay"),
        ("feedback", True, "feedback"),
    ]

    n_examples_list = [8, 16, 32]
    eps_values = [0.05, 0.10, 0.20]  # from where feedback showed promise
    seeds = [42, 137, 256]

    all_results = {}
    t0 = time.perf_counter()

    for task_name, gen_fn in tasks:
        print(f"\n{'═'*70}")
        print(f"TASK: {task_name}")
        print(f"{'═'*70}")

        task_results = {}

        for n_ex in n_examples_list:
            t_len = 2 * (n_ex + 1)
            print(f"\n  n_examples={n_ex} (T={t_len})")
            print(f"  {'─'*50}")

            length_results = {}

            for mode_name, use_field, field_mode in modes:
                if not use_field:
                    # No field: single run, no ε dependence
                    vals = []
                    for seed in seeds:
                        val, _, _ = train_config(
                            VOCAB_SIZE, n_ex, seed, 0.0,
                            use_field=False,
                            pe_type="learned", use_rope=False,
                            generate_fn=gen_fn, gen_kwargs={},
                            train_steps=TRAIN_STEPS)
                        vals.append(val)
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    length_results[mode_name] = {
                        "best_eps": None, "mean_val": mean_val,
                        "std_val": std_val, "vals": vals,
                        "all_eps": [],
                    }
                    print(f"    {mode_name:10s}  val={mean_val:.3f}±{std_val:.3f}  "
                          f"[{', '.join(f'{v:.3f}' for v in vals)}]")
                else:
                    # Sweep ε, pick best
                    best_mean = -1
                    best_eps = None
                    best_vals = None
                    eps_results = []
                    for eps in eps_values:
                        vals = []
                        for seed in seeds:
                            val, _, _ = train_config(
                                VOCAB_SIZE, n_ex, seed, eps,
                                use_field=True,
                                pe_type="learned", use_rope=False,
                                generate_fn=gen_fn, gen_kwargs={},
                                field_mode=field_mode,
                                train_steps=TRAIN_STEPS)
                            vals.append(val)
                        mean_val = np.mean(vals)
                        std_val = np.std(vals)
                        eps_results.append({
                            "eps": eps, "mean_val": mean_val,
                            "std_val": std_val, "vals": vals,
                        })
                        if mean_val > best_mean:
                            best_mean = mean_val
                            best_eps = eps
                            best_vals = vals
                        elapsed = time.perf_counter() - t0
                        print(f"    {mode_name:10s} ε={eps:.2f}  "
                              f"val={mean_val:.3f}±{std_val:.3f}  "
                              f"[{', '.join(f'{v:.3f}' for v in vals)}]  "
                              f"({elapsed:.0f}s)")

                    length_results[mode_name] = {
                        "best_eps": best_eps,
                        "mean_val": best_mean,
                        "std_val": np.std(best_vals),
                        "vals": best_vals,
                        "all_eps": eps_results,
                    }

            # Summary for this length
            print(f"\n  Summary (n={n_ex}, T={t_len}):")
            for mode_name, _, _ in modes:
                r = length_results[mode_name]
                eps_str = f" ε={r['best_eps']:.2f}" if r['best_eps'] else ""
                print(f"    {mode_name:10s}{eps_str:8s}  val={r['mean_val']:.3f}±{r['std_val']:.3f}")

            baseline = length_results["no_field"]["mean_val"]
            for mode_name in ["decay", "feedback"]:
                delta = length_results[mode_name]["mean_val"] - baseline
                print(f"    Δ({mode_name} - no_field) = {delta:+.3f}")

            task_results[n_ex] = length_results

        all_results[task_name] = task_results

    total_time = time.perf_counter() - t0
    print(f"\n{'═'*70}")
    print(f"State tracking complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'═'*70}")

    plot_statetrack(all_results, tasks, n_examples_list, modes)
    return all_results


def plot_statetrack(all_results, tasks, n_examples_list, modes):
    """Plot state-tracking results: tasks × lengths, comparing modes."""
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]

    mode_colors = {"no_field": "tab:gray", "decay": "tab:blue", "feedback": "tab:green"}
    mode_markers = {"no_field": "s", "decay": "o", "feedback": "D"}

    for ax, (task_name, _) in zip(axes, tasks):
        task_data = all_results[task_name]

        for mode_name, _, _ in modes:
            lengths = []
            vals = []
            stds = []
            for n_ex in n_examples_list:
                r = task_data[n_ex][mode_name]
                lengths.append(2 * (n_ex + 1))
                vals.append(r["mean_val"])
                stds.append(r["std_val"])

            label = mode_name
            if mode_name != "no_field":
                # Add best ε info
                best_eps = [task_data[n][mode_name]["best_eps"] for n in n_examples_list]
                label = f"{mode_name} (ε={','.join(f'{e:.2f}' for e in best_eps)})"

            ax.errorbar(lengths, vals, yerr=stds,
                        fmt=f"{mode_markers[mode_name]}-",
                        color=mode_colors[mode_name],
                        linewidth=2, markersize=8, capsize=4,
                        label=label)

        ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle=":", alpha=0.3,
                   label="Random")
        ax.set_xlabel("Sequence length T")
        ax.set_ylabel("Validation Accuracy (best ε, 3 seeds)")
        ax.set_title(f"{task_name.title()}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("State Tracking: Does Feedback Attention Help Where It Should?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "statetrack.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: statetrack.png")


def main():
    parser = argparse.ArgumentParser(description="Gate experiments")
    parser.add_argument("--gate", choices=["1", "2", "3", "sweep", "modes",
                                           "feedback", "statetrack", "all"],
                        default="all")
    args = parser.parse_args()

    if args.gate in ("1", "all"):
        run_gate1()
        print()

    if args.gate in ("2", "all"):
        run_gate2()
        print()

    if args.gate in ("3", "all"):
        run_gate3()
        print()

    if args.gate == "sweep":
        run_sweep_correct()
        print()

    if args.gate == "modes":
        run_field_modes()
        print()

    if args.gate == "feedback":
        run_feedback()
        print()

    if args.gate == "statetrack":
        run_statetrack()
        print()

    print("All requested experiments complete.")


if __name__ == "__main__":
    main()
