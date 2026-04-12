"""Linear Field: computation-dependent deposits, parallelizable readout.

The key insight: the social field's power comes from depositing attention
OUTPUT (computation-dependent). The serial bottleneck comes from key
MODULATION (non-linear feedback through softmax). If we add the field
to the output instead, the recurrence is linear and scan-parallelizable.

Architecture:
  1. Standard causal attention: A_t = Attn(Q_t, K_{1:t}, V_{1:t})  [parallel]
  2. Deposits from attention output: d_t = W_dep(A_t)                [parallel]
  3. Field via parallel scan: f_t = α·f_{t-1} + d_t                 [O(log T)]
  4. Output = A_t + W_read(f_t)                                     [parallel]

No serial loop. Deposits are computation-dependent (attention output).
The field carries forward the model's accumulated beliefs.

Modes:
  baseline     - standard attention
  faithful     - original social field (serial, key modulation) — gold standard
  linear_field - EWMA of attention outputs, added to output [PARALLEL]
  ewma_keys    - EWMA of keys, modulates keys (prior work) [PARALLEL]

Tasks: cumsum mod 16, parity

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_linear_field.py
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
# Parallel scan (sequential impl — replace with Blelloch for GPU)
# ══════════════════════════════════════════════════════════════

def linear_scan(deposits, alpha):
    """f_t = alpha * f_{t-1} + deposits_t.

    deposits: (B, H, T, d_f)
    alpha: scalar

    Returns: (B, H, T, d_f) field states.
    """
    B, H, T, d_f = deposits.shape
    out = torch.empty_like(deposits)
    state = torch.zeros(B, H, d_f, device=deposits.device, dtype=deposits.dtype)
    for t in range(T):
        state = alpha * state + deposits[:, :, t]
        out[:, :, t] = state
    return out


# ══════════════════════════════════════════════════════════════
# Attention variants
# ══════════════════════════════════════════════════════════════

class BaselineAttention(nn.Module):
    """Standard multi-head causal attention."""

    def __init__(self, d_model, n_head, dropout, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape
        H, d_h = self.n_head, self.head_dim
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)
        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        att = (q @ k.transpose(-2, -1)) * (d_h ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class LinearFieldAttention(nn.Module):
    """Attention + linear recurrence field on outputs.

    1. Standard causal attention → A  (parallel)
    2. Deposits = W_dep(A)             (parallel)
    3. Field = scan(deposits, α)       (parallel scan)
    4. Output = A + W_read(field)      (parallel)

    Deposits are computation-dependent (attention output).
    No key modulation. No serial loop.
    """

    def __init__(self, d_model, n_head, dropout, max_len,
                 alpha=0.9, d_field=None, shared_params=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.d_field = d_field or self.head_dim

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

        # Field parameters (can be shared across layers)
        if shared_params is not None:
            self.w_dep = shared_params["w_dep"]
            self.w_read = shared_params["w_read"]
            self.alpha_logit = shared_params["alpha_logit"]
        else:
            self.w_dep = nn.Linear(d_model, n_head * self.d_field, bias=False)
            self.w_read = nn.Linear(self.d_field, self.head_dim, bias=False)
            init_logit = math.log(alpha / (1 - alpha + 1e-8))
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit))

        nn.init.normal_(self.w_dep.weight, std=0.02)
        nn.init.zeros_(self.w_read.weight)

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, x):
        B, T, C = x.shape
        H, d_h, d_f = self.n_head, self.head_dim, self.d_field

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Standard causal attention (fully parallel)
        att = (q @ k.transpose(-2, -1)) * (d_h ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        A = att @ v  # (B, H, T, d_h)

        # Flatten for deposit projection
        A_flat = A.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Compute deposits from attention output (computation-dependent!)
        deposits = self.w_dep(A_flat).view(B, T, H, d_f)  # (B, T, H, d_f)
        deposits = deposits.transpose(1, 2)  # (B, H, T, d_f)

        # Parallel scan: field states
        alpha = self.get_alpha()
        field_states = linear_scan(deposits, alpha)  # (B, H, T, d_f)

        # Read from field and add to attention output
        field_contrib = self.w_read(field_states)  # (B, H, T, d_h)
        out = A + field_contrib  # (B, H, T, d_h)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FaithfulFieldAttention(nn.Module):
    """Original social field: serial, deposits output, modulates keys.

    Gold standard for comparison. O(T) serial loop.
    """

    def __init__(self, d_model, n_head, dropout, max_len,
                 evap_rate=0.10, shared_params=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.d_f = self.head_dim

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len)

        if shared_params is not None:
            self.w_deposit = shared_params["w_deposit"]
            self.w_mod = shared_params["w_mod"]
        else:
            self.w_deposit = nn.Linear(d_model, n_head * self.d_f, bias=False)
            self.w_mod = nn.Linear(self.d_f, self.head_dim, bias=False)

        self.retain = 1.0 - evap_rate

    def forward(self, x):
        B, T, C = x.shape
        H, d_h, d_f = self.n_head, self.head_dim, self.d_f

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        field_state = torch.zeros(B, H, d_f, device=x.device, dtype=x.dtype)
        field_states = []
        output_list = []

        for i in range(T):
            k_i_raw = k[:, :, i:i+1, :]
            if i > 0:
                fh = torch.stack(field_states, dim=2)
                mod = self.w_mod(fh)
                k_prev_mod = k[:, :, :i, :] + mod
                k_mod = torch.cat([k_prev_mod, k_i_raw], dim=2)
            else:
                k_mod = k_i_raw

            q_i = q[:, :, i:i+1, :]
            att_i = (q_i @ k_mod.transpose(-2, -1)) * (d_h ** -0.5)
            att_i = F.softmax(att_i, dim=-1)
            att_i = self.attn_drop(att_i)
            out_i = (att_i @ v[:, :, :i+1, :]).squeeze(2)
            out_flat = out_i.transpose(1, 2).contiguous().view(B, C)
            output_list.append(out_flat)

            deposit = self.w_deposit(out_flat).view(B, H, d_f)
            field_state = self.retain * field_state + deposit
            field_states.append(field_state)

        out = torch.stack(output_list, dim=1)
        return self.resid_drop(self.out_proj(out))


class EWMAKeyAttention(nn.Module):
    """EWMA key coupling (prior work). Parallel but input-derived."""

    def __init__(self, d_model, n_head, dropout, max_len, alpha=0.9):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))
        self.register_buffer("alpha", torch.tensor(alpha))

        self.beta = nn.Parameter(torch.tensor(0.1))
        self.w_coup = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        H, d_h = self.n_head, self.head_dim

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # EWMA of keys
        trails = linear_scan(k, self.alpha)
        trail_sum = trails.sum(dim=1, keepdim=True)
        other_trails = trail_sum - trails
        coupling = self.w_coup(other_trails)
        beta = torch.sigmoid(self.beta) * 4.0
        k_mod = k + beta * coupling

        att = (q @ k_mod.transpose(-2, -1)) * (d_h ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# ══════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len, mode="baseline",
                 shared_field_params=None, shared_faithful_params=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        if mode == "baseline":
            self.attn = BaselineAttention(d_model, n_head, dropout, max_len)
        elif mode == "faithful":
            self.attn = FaithfulFieldAttention(
                d_model, n_head, dropout, max_len,
                shared_params=shared_faithful_params)
        elif mode == "linear_field":
            self.attn = LinearFieldAttention(
                d_model, n_head, dropout, max_len,
                shared_params=shared_field_params)
        elif mode == "ewma_keys":
            self.attn = EWMAKeyAttention(d_model, n_head, dropout, max_len)
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
                 mode="baseline", tie_field=True):
        super().__init__()
        self.vocab_size = vocab_size

        # Shared field params for weight tying
        shared_field = None
        shared_faithful = None
        if tie_field and mode == "linear_field":
            d_f = d_model // n_head
            shared_field = {
                "w_dep": nn.Linear(d_model, n_head * d_f, bias=False),
                "w_read": nn.Linear(d_f, d_model // n_head, bias=False),
                "alpha_logit": nn.Parameter(
                    torch.tensor(math.log(0.9 / 0.1))),
            }
            self.shared_w_dep = shared_field["w_dep"]
            self.shared_w_read = shared_field["w_read"]
            self.shared_alpha_logit = shared_field["alpha_logit"]
        if tie_field and mode == "faithful":
            d_f = d_model // n_head
            shared_faithful = {
                "w_deposit": nn.Linear(d_model, n_head * d_f, bias=False),
                "w_mod": nn.Linear(d_f, d_model // n_head, bias=False),
            }
            self.shared_w_deposit = shared_faithful["w_deposit"]
            self.shared_w_mod = shared_faithful["w_mod"]

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, max_len, mode=mode,
                  shared_field_params=shared_field,
                  shared_faithful_params=shared_faithful)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════

def gen_cumsum(batch_size, n_examples, vocab_size, rng):
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
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)
    for b in range(batch_size):
        parity = 0
        for i in range(n_total):
            x = int(rng.integers(0, 2))
            parity = (parity + x) % 2
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = parity
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]
    return torch.tensor(input_ids), torch.tensor(targets), torch.tensor(last_targets)


TASK_GENERATORS = {"cumsum": gen_cumsum, "parity": gen_parity}


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def train_and_eval(mode, task, vocab_size, n_examples, seed,
                   d_model, n_head, n_layer,
                   train_steps, batch_size, n_test_list):
    max_n = max(n_test_list)
    max_seq = 2 * (max_n + 1) + 2
    gen_fn = TASK_GENERATORS[task]

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = GPT(
        vocab_size, max_seq, d_model, n_head, n_layer, dropout=0.05,
        mode=mode, tie_field=True,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                  weight_decay=0.10, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps)

    for step in range(train_steps):
        model.train()
        inp, tgt, _ = gen_fn(batch_size, n_examples, vocab_size, rng=rng)
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               tgt.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    results = {}
    for n_test in n_test_list:
        with torch.no_grad():
            eval_rng = np.random.default_rng(9999)
            val_in, _, val_last = gen_fn(512, n_test, vocab_size, rng=eval_rng)
            val_in = val_in.to(DEVICE)
            val_logits = model(val_in)
            val_pred = val_logits[:, -1].argmax(dim=-1).cpu()
            val_acc = (val_pred == val_last).float().mean().item()
            results[n_test] = val_acc

    # Get alpha if linear_field
    alpha_val = None
    for block in model.blocks:
        if hasattr(block.attn, 'get_alpha'):
            alpha_val = block.attn.get_alpha().item()
            break

    return results, model.count_params(), alpha_val


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("LINEAR FIELD: Computation-Dependent Deposits, Parallel Readout")
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
    tasks = ["cumsum", "parity"]
    modes = ["baseline", "faithful", "linear_field", "ewma_keys"]

    all_results = {}
    t0 = time.perf_counter()

    for task in tasks:
        print(f"\n{'█' * 70}")
        print(f"  TASK: {task}")
        print(f"{'█' * 70}")

        task_results = {}

        for mode in modes:
            print(f"\n{'─' * 70}")
            print(f"  {task} / {mode}")
            print(f"{'─' * 70}")

            seed_results = {n: [] for n in n_test_list}
            alphas = []

            for seed in seeds:
                t_seed = time.perf_counter()
                results, n_params, alpha_val = train_and_eval(
                    mode, task, V, N_EX, seed,
                    d_model=D, n_head=H, n_layer=N_LAYER,
                    train_steps=STEPS, batch_size=BS,
                    n_test_list=n_test_list)
                for n in n_test_list:
                    seed_results[n].append(results[n])
                if alpha_val is not None:
                    alphas.append(alpha_val)
                elapsed = time.perf_counter() - t0
                dt = time.perf_counter() - t_seed
                accs = [f"{results[n]:.3f}" for n in n_test_list]
                print(f"    seed={seed}  [{', '.join(accs)}]  "
                      f"params={n_params}  ({dt:.0f}s, total {elapsed:.0f}s)")

            summary = {
                "mode": mode, "task": task, "n_params": n_params,
                "results": {},
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
            if alphas:
                print(f"    α (learned): {np.mean(alphas):.3f}")
                summary["alpha"] = float(np.mean(alphas))

            task_results[mode] = summary

        all_results[task] = task_results

        # Save after each task
        out_path = OUT_DIR / "exp_linear_field.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")

    for task in tasks:
        print(f"\n  Task: {task}")
        header = f"    {'Mode':>15s}  {'Params':>7s}"
        for n in n_test_list:
            header += f"  {'T='+str(2*(n+1)):>14s}"
        print(header)
        print(f"    {'─' * 68}")

        for mode in modes:
            s = all_results[task][mode]
            line = f"    {mode:>15s}  {s['n_params']:>7d}"
            for n in n_test_list:
                r = s['results'][str(n)]
                line += f"  {r['mean']:.3f}+/-{r['std']:.3f}"
            if "alpha" in s:
                line += f"  α={s['alpha']:.3f}"
            print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_path = OUT_DIR / "exp_linear_field.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_experiment()
