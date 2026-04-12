"""Stigmergic Residuals: Inter-Head Coupling via Key Trails.

Each head maintains an EWMA of its keys (the "pheromone trail").
Each head modulates its own keys by reading other heads' trails.
Per-head coupling strength β_h controls adherence to the collective.

Fully parallelizable: EWMA is a parallel scan, coupling is a matmul.

Tasks:
  cumsum   - cumulative sum mod 16 (4-bit state tracking)
  parity   - running parity (1-bit state tracking)
  selcopy  - selective copy (position-selective retrieval)

Modes:
  baseline        - standard attention, no coupling
  fixed_low       - fixed β = 0.1
  fixed_med       - fixed β = 0.5
  fixed_high      - fixed β = 2.0
  learnable       - per-head learnable β_h (init 0.1)
  learnable_alpha - learnable β_h and α

Probes (logged every 100 steps):
  - attention entropy per head
  - head-pair correlation matrix
  - trail contribution vs raw key
  - β_h values
  - α value
  - head ablation Δ (at eval only)

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_stigmergic_residuals.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# Components
# ══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T, device):
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, freqs):
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ══════════════════════════════════════════════════════════════
# Parallel EWMA scan
# ══════════════════════════════════════════════════════════════

def ewma_scan(x, alpha):
    """Compute EWMA: s_t = alpha * s_{t-1} + (1 - alpha) * x_t.

    x: (B, H, T, d_h)
    alpha: scalar or (H,)

    Returns: (B, H, T, d_h) of trail states.

    Sequential implementation — in production, replace with
    Blelloch parallel prefix scan for O(log T) depth.
    """
    B, H, T, d_h = x.shape
    out = torch.empty_like(x)
    state = torch.zeros(B, H, d_h, device=x.device, dtype=x.dtype)
    if isinstance(alpha, torch.Tensor) and alpha.dim() > 0:
        # per-head alpha: (H,) -> (1, H, 1)
        a = alpha.view(1, H, 1)
    else:
        a = alpha
    for t in range(T):
        state = a * state + (1 - a) * x[:, :, t, :]
        out[:, :, t, :] = state
    return out


# ══════════════════════════════════════════════════════════════
# Stigmergic Attention
# ══════════════════════════════════════════════════════════════

class StigmergicAttention(nn.Module):
    """Multi-head attention with inter-head coupling via key trails.

    Each head h computes trail s_{h,t} = alpha * s_{h,t-1} + (1-alpha) * K_{h,t}.
    Then modulates keys: K'_{h,t} = K_{h,t} + beta_h * W_coup * sum_{h'!=h} s_{h',t}.

    All operations are parallel — no position-by-position loop.
    """

    def __init__(self, d_model, n_head, dropout, max_len,
                 alpha=0.9, beta_init=0.1,
                 learnable_beta=False, learnable_alpha=False,
                 layer_idx=0, tie_coup=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.layer_idx = layer_idx

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

        # Trail persistence
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            init_logit = math.log(alpha / (1 - alpha + 1e-8))
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer("alpha_val", torch.tensor(alpha))

        # Per-head coupling strength
        self.learnable_beta = learnable_beta
        if learnable_beta:
            self.beta_logit = nn.Parameter(
                torch.full((n_head,), math.log(beta_init / (1 - beta_init + 1e-8))))
        else:
            self.register_buffer("beta_val", torch.tensor(beta_init))

        # Coupling projection (can be shared across layers via tie_coup)
        if tie_coup is not None:
            self.w_coup = tie_coup
        else:
            self.w_coup = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Probe storage (populated during forward)
        self._probe_data = {}

    def get_alpha(self):
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logit)
        return self.alpha_val

    def get_beta(self):
        if self.learnable_beta:
            return torch.sigmoid(self.beta_logit) * 4.0  # range [0, 4]
        return self.beta_val

    def forward(self, x, collect_probes=False, ablate_head=None):
        B, T, C = x.shape
        H, d_h = self.n_head, self.head_dim
        scale = d_h ** -0.5

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)  # (B, H, T, d_h)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        alpha = self.get_alpha()
        beta = self.get_beta()  # scalar or (H,)

        # Compute key trails via parallel scan
        trails = ewma_scan(k, alpha)  # (B, H, T, d_h)

        # Inter-head coupling: each head reads sum of OTHER heads' trails
        trail_sum = trails.sum(dim=1, keepdim=True)  # (B, 1, T, d_h)
        # Subtract own trail: sum_{h'!=h} s_{h'} = total - s_h
        other_trails = trail_sum - trails  # (B, H, T, d_h)

        # Apply coupling projection
        coupling = self.w_coup(other_trails)  # (B, H, T, d_h)

        # Optionally ablate a specific head's trail contribution
        if ablate_head is not None:
            mask_ab = torch.ones(1, H, 1, 1, device=x.device)
            mask_ab[0, ablate_head, 0, 0] = 0.0
            coupling = coupling * mask_ab

        # Modulate keys
        if isinstance(beta, torch.Tensor) and beta.dim() > 0:
            beta_view = beta.view(1, H, 1, 1)
        else:
            beta_view = beta
        k_mod = k + beta_view * coupling  # (B, H, T, d_h)

        # Standard causal attention with modulated keys
        att = (q @ k_mod.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att, dim=-1)
        att_weights = self.attn_drop(att_weights)
        out = (att_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        # Collect probe data
        if collect_probes:
            with torch.no_grad():
                probes = {}

                # 1. Attention entropy per head: (H,)
                # att_weights: (B, H, T, T), compute entropy along last dim
                log_att = torch.log(att_weights + 1e-10)
                entropy = -(att_weights * log_att).sum(dim=-1)  # (B, H, T)
                probes["attn_entropy"] = entropy.mean(dim=(0, 2)).cpu().tolist()

                # 2. Head-pair correlation: (H, H)
                # Compare attention distributions across heads
                # Flatten per-head attention: (B*T, H, T_keys) -> pick last T positions
                flat_att = att_weights.permute(0, 2, 1, 3).reshape(B * T, H, T)
                corr_matrix = torch.zeros(H, H)
                for h1 in range(H):
                    for h2 in range(h1, H):
                        cos = F.cosine_similarity(
                            flat_att[:, h1, :], flat_att[:, h2, :], dim=-1)
                        corr_matrix[h1, h2] = cos.mean()
                        corr_matrix[h2, h1] = cos.mean()
                probes["head_correlation"] = corr_matrix.cpu().tolist()

                # 3. Trail contribution vs raw key: per-head ratio
                trail_norm = (beta_view * coupling).norm(dim=-1).mean(dim=(0, 2))  # (H,)
                key_norm = k.norm(dim=-1).mean(dim=(0, 2))  # (H,)
                probes["trail_ratio"] = (trail_norm / (key_norm + 1e-8)).cpu().tolist()

                # 4. β_h values
                if isinstance(beta, torch.Tensor) and beta.dim() > 0:
                    probes["beta"] = beta.cpu().tolist()
                else:
                    probes["beta"] = [float(beta)] * H

                # 5. α value
                probes["alpha"] = float(alpha)

                self._probe_data = probes

        return out


class BaselineAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))
        self.rope = RotaryEmbedding(self.head_dim, max_len)
        self._probe_data = {}

    def forward(self, x, collect_probes=False, ablate_head=None):
        B, T, C = x.shape
        H, d = self.n_head, self.head_dim
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, H, d).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)
        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        att = (q @ k.transpose(-2, -1)) * (d ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att, dim=-1)
        att_weights = self.attn_drop(att_weights)
        out = (att_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        if collect_probes:
            with torch.no_grad():
                log_att = torch.log(att_weights + 1e-10)
                entropy = -(att_weights * log_att).sum(dim=-1)
                self._probe_data = {
                    "attn_entropy": entropy.mean(dim=(0, 2)).cpu().tolist(),
                    "beta": [0.0] * H,
                    "alpha": 0.0,
                    "trail_ratio": [0.0] * H,
                }
        return out


# ══════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len,
                 use_coupling=True, alpha=0.9, beta_init=0.1,
                 learnable_beta=False, learnable_alpha=False,
                 layer_idx=0, tie_coup=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        if use_coupling:
            self.attn = StigmergicAttention(
                d_model, n_head, dropout, max_len,
                alpha=alpha, beta_init=beta_init,
                learnable_beta=learnable_beta,
                learnable_alpha=learnable_alpha,
                layer_idx=layer_idx, tie_coup=tie_coup)
        else:
            self.attn = BaselineAttention(d_model, n_head, dropout, max_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, collect_probes=False, ablate_head=None):
        x = x + self.attn(self.ln1(x), collect_probes=collect_probes,
                          ablate_head=ablate_head)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT_Stigmergic(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 use_coupling=True, alpha=0.9, beta_init=0.1,
                 learnable_beta=False, learnable_alpha=False,
                 tie_coup_across_layers=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.n_layer = n_layer

        tie_coup = None
        if tie_coup_across_layers and use_coupling:
            head_dim = d_model // n_head
            tie_coup = nn.Linear(head_dim, head_dim, bias=False)
            self.shared_w_coup = tie_coup

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, max_len,
                  use_coupling=use_coupling, alpha=alpha,
                  beta_init=beta_init, learnable_beta=learnable_beta,
                  learnable_alpha=learnable_alpha,
                  layer_idx=i, tie_coup=tie_coup)
            for i in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, collect_probes=False, ablate_head=None):
        x = self.tok_emb(idx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, collect_probes=collect_probes,
                      ablate_head=ablate_head)
        return self.head(self.ln_f(x))

    def get_probe_data(self):
        """Collect probe data from all layers."""
        probes = {}
        for i, block in enumerate(self.blocks):
            for k, v in block.attn._probe_data.items():
                probes[f"L{i}_{k}"] = v
        return probes

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════
# Data generation
# ══════════════════════════════════════════════════════════════

def gen_cumsum(batch_size, n_examples, vocab_size, rng):
    """Cumulative sum mod vocab_size. Sequence: x0, s0, x1, s1, ..."""
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
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]
    return torch.tensor(input_ids), torch.tensor(targets), torch.tensor(last_targets)


def gen_parity(batch_size, n_examples, vocab_size, rng):
    """Running parity of binary inputs. Sequence: x0, p0, x1, p1, ...
    x_i in {0, 1}, p_i = (sum of x_0..x_i) mod 2.
    We use vocab_size tokens but only 0/1 for input, 0/1 for parity output.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)
    for b in range(batch_size):
        parity = 0
        for i in range(n_total):
            x = int(rng.integers(0, 2))  # binary input
            parity = (parity + x) % 2
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = parity
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]
    return torch.tensor(input_ids), torch.tensor(targets), torch.tensor(last_targets)


def gen_selective_copy(batch_size, n_examples, vocab_size, rng):
    """Selective copy: given markers, copy specific tokens.

    Sequence format: [content tokens..., MARKER, content tokens..., MARKER, ...]
    At each marker position, predict the token that appeared marker_value positions
    before the marker.

    Simplified version: pairs of (value, index) where index says which
    previous value to recall. Token at output position = value at index.

    Actually, let's do a cleaner version:
    - First half: N random values from {0..V-1}
    - Second half: N indices into first half, each followed by the correct value
    - Model must learn: given an index token, output the value at that position
    """
    # Use a 2-phase approach:
    # Phase 1: v0 v1 v2 ... v_{n-1}  (values to remember)
    # Phase 2: i0 a0 i1 a1 ...       (index, answer pairs)
    # Indices shifted by vocab_size to distinguish from values
    n_vals = n_examples
    n_queries = n_examples + 1
    total_len = n_vals + 2 * n_queries
    # Need vocab to include both values AND index tokens
    # Values: 0..V-1, Indices: V..V+n_vals-1
    # Total vocab needed: vocab_size + max_n_vals
    # For simplicity, use values in [0, V//2) and indices in [V//2, V)
    v_range = vocab_size // 2

    tokens = np.zeros((batch_size, total_len), dtype=np.int64)
    for b in range(batch_size):
        values = [int(rng.integers(0, v_range)) for _ in range(n_vals)]
        for i, val in enumerate(values):
            tokens[b, i] = val
        for q in range(n_queries):
            idx = int(rng.integers(0, n_vals))
            pos = n_vals + 2 * q
            tokens[b, pos] = v_range + idx  # index token
            tokens[b, pos + 1] = values[idx]  # answer

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    # Mask value positions and index tokens — only predict answers
    for i in range(total_len - 1):
        if i < n_vals - 1:
            targets[:, i] = -100  # value phase (except last which leads to first query)
        elif i >= n_vals:
            # In query phase: even offsets from n_vals are index tokens, odd are answers
            offset = i - n_vals
            if offset % 2 == 0:
                targets[:, i] = -100  # index token, don't predict next index
    last_targets = tokens[:, -1]
    return torch.tensor(input_ids), torch.tensor(targets), torch.tensor(last_targets)


TASK_GENERATORS = {
    "cumsum": gen_cumsum,
    "parity": gen_parity,
    "selcopy": gen_selective_copy,
}


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def train_and_eval(mode_cfg, task, vocab_size, n_examples, seed,
                   d_model, n_head, n_layer,
                   train_steps, batch_size, n_test_list,
                   probe_every=100):
    max_n = max(n_test_list)
    gen_fn = TASK_GENERATORS[task]

    if task == "selcopy":
        max_seq = max_n + 2 * (max_n + 1) + 2
        # Index tokens go up to vocab_size//2 + max_n - 1, so expand vocab
        effective_vocab = vocab_size // 2 + max_n
    else:
        max_seq = 2 * (max_n + 1) + 2
        effective_vocab = vocab_size

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = GPT_Stigmergic(
        effective_vocab, max_seq, d_model, n_head, n_layer, dropout=0.05,
        **mode_cfg,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                  weight_decay=0.10, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps)

    probe_log = []

    for step in range(train_steps):
        model.train()
        inp, tgt, _ = gen_fn(batch_size, n_examples, vocab_size, rng=rng)
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

        collect = (step % probe_every == 0)
        logits = model(inp, collect_probes=collect)
        loss = F.cross_entropy(logits.view(-1, effective_vocab),
                               tgt.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if collect:
            probes = model.get_probe_data()
            probes["step"] = step
            probes["loss"] = loss.item()
            probe_log.append(probes)

    # Evaluation
    model.eval()
    results = {}
    for n_test in n_test_list:
        with torch.no_grad():
            eval_rng = np.random.default_rng(9999)
            val_in, _, val_last = gen_fn(512, n_test, vocab_size, rng=eval_rng)
            val_in = val_in.to(DEVICE)
            val_logits = model(val_in, collect_probes=True)
            val_pred = val_logits[:, -1].argmax(dim=-1).cpu()
            val_acc = (val_pred == val_last).float().mean().item()
            results[n_test] = val_acc

    # Final probe data
    final_probes = model.get_probe_data()

    # Head ablation study (only for stigmergic models)
    ablation = {}
    if mode_cfg.get("use_coupling", True):
        with torch.no_grad():
            eval_rng = np.random.default_rng(9999)
            val_in, _, val_last = gen_fn(512, n_examples, vocab_size, rng=eval_rng)
            val_in = val_in.to(DEVICE)
            # Baseline accuracy (no ablation)
            base_logits = model(val_in)
            base_pred = base_logits[:, -1].argmax(dim=-1).cpu()
            base_acc = (base_pred == val_last).float().mean().item()
            ablation["base"] = base_acc

            # Ablate each head's trail
            for h in range(n_head):
                ab_logits = model(val_in, ablate_head=h)
                ab_pred = ab_logits[:, -1].argmax(dim=-1).cpu()
                ab_acc = (ab_pred == val_last).float().mean().item()
                ablation[f"ablate_h{h}"] = ab_acc
                ablation[f"delta_h{h}"] = base_acc - ab_acc

    return results, final_probes, probe_log, ablation, model.count_params()


# ══════════════════════════════════════════════════════════════
# Mode configurations
# ══════════════════════════════════════════════════════════════

MODE_CONFIGS = {
    "baseline": dict(
        use_coupling=False, alpha=0.9, beta_init=0.0,
        learnable_beta=False, learnable_alpha=False,
        tie_coup_across_layers=True),
    "fixed_low": dict(
        use_coupling=True, alpha=0.9, beta_init=0.1,
        learnable_beta=False, learnable_alpha=False,
        tie_coup_across_layers=True),
    "fixed_med": dict(
        use_coupling=True, alpha=0.9, beta_init=0.5,
        learnable_beta=False, learnable_alpha=False,
        tie_coup_across_layers=True),
    "fixed_high": dict(
        use_coupling=True, alpha=0.9, beta_init=2.0,
        learnable_beta=False, learnable_alpha=False,
        tie_coup_across_layers=True),
    "learnable": dict(
        use_coupling=True, alpha=0.9, beta_init=0.1,
        learnable_beta=True, learnable_alpha=False,
        tie_coup_across_layers=True),
    "learnable_alpha": dict(
        use_coupling=True, alpha=0.9, beta_init=0.1,
        learnable_beta=True, learnable_alpha=True,
        tie_coup_across_layers=True),
}


# ══════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════

def run_experiment(tasks_override=None):
    print("=" * 70)
    print("STIGMERGIC RESIDUALS: Inter-Head Coupling via Key Trails")
    print("=" * 70)

    V = 16
    D = 48
    H = 4
    N_EX = 8
    N_LAYER = 4
    STEPS = 5000
    BS = 64
    seeds = [42, 137, 256]
    n_test_list = [N_EX, N_EX * 2, N_EX * 3]
    tasks = tasks_override or ["cumsum", "parity", "selcopy"]
    modes = ["baseline", "fixed_low", "fixed_med", "fixed_high",
             "learnable", "learnable_alpha"]

    all_results = {}
    t0 = time.perf_counter()

    for task in tasks:
        print(f"\n{'█' * 70}")
        print(f"  TASK: {task}")
        print(f"{'█' * 70}")

        task_results = {}

        for mode in modes:
            cfg = MODE_CONFIGS[mode]
            print(f"\n{'─' * 70}")
            print(f"  {task} / {mode}")
            print(f"{'─' * 70}")

            seed_results = {n: [] for n in n_test_list}
            all_probes = []
            all_probe_logs = []
            all_ablations = []

            for seed in seeds:
                results, probes, probe_log, ablation, n_params = train_and_eval(
                    cfg, task, V, N_EX, seed,
                    d_model=D, n_head=H, n_layer=N_LAYER,
                    train_steps=STEPS, batch_size=BS,
                    n_test_list=n_test_list)
                for n in n_test_list:
                    seed_results[n].append(results[n])
                all_probes.append(probes)
                all_probe_logs.append(probe_log)
                all_ablations.append(ablation)
                elapsed = time.perf_counter() - t0
                accs = [f"{results[n]:.3f}" for n in n_test_list]
                print(f"    seed={seed}  [{', '.join(accs)}]  "
                      f"params={n_params}  ({elapsed:.0f}s)")

            summary = {
                "mode": mode,
                "task": task,
                "n_params": n_params,
                "results": {},
                "probes": all_probes,
                "probe_logs": all_probe_logs,
                "ablations": all_ablations,
            }
            for n in n_test_list:
                vals = seed_results[n]
                summary["results"][str(n)] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "vals": vals,
                }

            accs_str = "  ".join(
                f"T={2*(n+1)}:{np.mean(seed_results[n]):.3f}+/-{np.std(seed_results[n]):.3f}"
                for n in n_test_list)
            print(f"    => {accs_str}")

            # Print probe summary
            if all_probes and "L0_beta" in all_probes[-1]:
                betas = all_probes[-1].get("L0_beta", [])
                print(f"    β_h: {['%.3f' % b for b in betas]}")
            if all_probes and "L0_alpha" in all_probes[-1]:
                print(f"    α: {all_probes[-1].get('L0_alpha', 'N/A'):.3f}")
            if all_probes and "L0_trail_ratio" in all_probes[-1]:
                ratios = all_probes[-1].get("L0_trail_ratio", [])
                print(f"    trail/key: {['%.3f' % r for r in ratios]}")
            if all_ablations and all_ablations[-1]:
                ab = all_ablations[-1]
                deltas = [ab.get(f"delta_h{h}", 0) for h in range(H)]
                print(f"    ablation Δ: {['%.3f' % d for d in deltas]}")

            task_results[mode] = summary

        all_results[task] = task_results

        # Save after each task so we don't lose results on crash
        out_path = OUT_DIR / "exp_stigmergic_residuals.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  (saved intermediate results to {out_path})")

    # ── Summary tables ────────────────────────────────────────

    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")

    for task in tasks:
        if task not in all_results:
            continue
        print(f"\n  Task: {task}")
        header = f"    {'Mode':>16s}  {'Params':>7s}"
        for n in n_test_list:
            header += f"  {'T='+str(2*(n+1)):>14s}"
        print(header)
        print(f"    {'─' * 68}")

        for mode in modes:
            if mode not in all_results[task]:
                continue
            s = all_results[task][mode]
            line = f"    {mode:>16s}  {s['n_params']:>7d}"
            for n in n_test_list:
                r = s['results'][str(n)]
                line += f"  {r['mean']:.3f}+/-{r['std']:.3f}"
            print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_path = OUT_DIR / "exp_stigmergic_residuals.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")

    return all_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_experiment(tasks_override=sys.argv[1:])
    else:
        run_experiment()
