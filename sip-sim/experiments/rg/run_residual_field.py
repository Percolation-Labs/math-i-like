"""Residual Stream Field: stigmergic field IN the residual stream.

Key insight from prior work:
  - Social field (serial): deposits attn output, modulates keys → powerful
    but O(T) serial loop (key modulation is non-linear through softmax).
  - Linear field (parallel): deposits attn output, adds to output → the
    field never influences what the model ATTENDS to → worse than baseline.
  - EWMA key coupling (parallel): smooths input-derived keys → input-derived
    signal is too weak for state tracking.

The residual stream field resolves this:
  - Deposit the FULL layer output into a decaying field (computation-dependent)
  - Inject the field into the stream BEFORE the next layer's attention
  - The next layer's Q,K,V projections see the field-modified input
  - This IS key modulation by proxy — through standard projections
  - The recurrence is linear and scan-parallelizable: O(log T)

Architecture at each layer l:
  1. Inject field:   x += β_l · W_read(LN(field))     [parallel]
  2. Attention:      x += Attn(LN(x))                  [parallel]
  3. MLP:            x += MLP(LN(x))                    [parallel]
  4. Deposit:        d_t = W_dep(x)                      [parallel]
  5. Scan:           new = scan(d_t, α)                  [O(log T)]
  6. Accumulate:     field += new                        [parallel]

Per-layer β_l is the critical exponent — how much each layer follows
the collective field vs computes independently.

W_dep, W_read shared across layers (memoisation — the deposit/read rule
is invariant, per GRN insight). α is learned. β_l is per-layer.

Modes:
  baseline        - standard transformer
  residual_field  - stigmergic field in the residual stream [NEW]
  faithful        - original social field (serial, gold standard)

Tasks: cumsum mod 16, parity

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_residual_field.py
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


def linear_scan(deposits, alpha):
    """f_t = alpha * f_{t-1} + deposits_t.  Sequential impl."""
    B, T, d_f = deposits.shape
    out = torch.empty_like(deposits)
    state = torch.zeros(B, d_f, device=deposits.device, dtype=deposits.dtype)
    for t in range(T):
        state = alpha * state + deposits[:, t]
        out[:, t] = state
    return out


# ══════════════════════════════════════════════════════════════
# Attention (standard, used by all modes)
# ══════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
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


# ══════════════════════════════════════════════════════════════
# Faithful social field (serial, gold standard)
# ══════════════════════════════════════════════════════════════

class FaithfulFieldAttention(nn.Module):
    """Original social field: serial, deposits output, modulates keys."""

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
            v_i = v[:, :, :i+1, :]
            att = (q_i @ k_mod.transpose(-2, -1)) * (d_h ** -0.5)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            o_i = (att @ v_i).squeeze(2)

            o_flat = o_i.transpose(0, 1).contiguous().view(B, C)
            deposit = self.w_deposit(o_flat).view(B, H, d_f)
            field_state = self.retain * field_state + deposit
            field_states.append(field_state)
            output_list.append(o_i)

        out = torch.stack(output_list, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# ══════════════════════════════════════════════════════════════
# Residual Stream Field
# ══════════════════════════════════════════════════════════════

class SharedFieldParams(nn.Module):
    """Field parameters shared across all layers (memoisation).

    W_dep, W_read, α are properties of the field, not of individual
    layers. Sharing them follows the GRN insight: the coupling rule
    is invariant across depth.
    """

    def __init__(self, d_model, d_field):
        super().__init__()
        self.d_field = d_field
        self.w_dep = nn.Linear(d_model, d_field, bias=False)
        self.w_read = nn.Linear(d_field, d_model, bias=False)
        self.ln_field = RMSNorm(d_field)
        init_logit = math.log(0.9 / 0.1)
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit))

        nn.init.normal_(self.w_dep.weight, std=0.02)
        nn.init.zeros_(self.w_read.weight)

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit)


class ResidualFieldBlock(nn.Module):
    """Transformer block with stigmergic field injection.

    The field is injected into the residual stream BEFORE attention,
    so Q,K,V projections see field-modified inputs. This is key
    modulation by proxy — through the standard linear projections.
    """

    def __init__(self, d_model, n_head, dropout, max_len,
                 shared_field, layer_idx):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, max_len)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )
        self.shared_field = shared_field
        self.layer_idx = layer_idx

        # Per-layer coupling strength (the critical exponent)
        self.log_beta = nn.Parameter(torch.tensor(0.0))

    @property
    def beta(self):
        return self.log_beta.exp()

    def forward(self, x, field):
        B, T, d = x.shape

        # 1. Inject field into stream (before attention)
        if field is not None:
            field_normed = self.shared_field.ln_field(field)
            field_read = self.shared_field.w_read(field_normed)
            x = x + self.beta * field_read

        # 2. Standard attention (unmodified — but sees field-enriched input)
        x = x + self.attn(self.ln1(x))

        # 3. MLP
        x = x + self.mlp(self.ln2(x))

        # 4. Deposit full layer output into field
        alpha = self.shared_field.get_alpha()
        deposits = self.shared_field.w_dep(x)  # (B, T, d_f)

        # 5. Scan across positions (parallelizable)
        new_scan = linear_scan(deposits, alpha)  # (B, T, d_f)

        # 6. Accumulate across layers (stigmergic: each layer adds to the field)
        if field is not None:
            field = field + new_scan
        else:
            field = new_scan

        return x, field


# ══════════════════════════════════════════════════════════════
# Standard Block (for baseline and faithful modes)
# ══════════════════════════════════════════════════════════════

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len, mode="baseline",
                 shared_faithful_params=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        if mode == "faithful":
            self.attn = FaithfulFieldAttention(
                d_model, n_head, dropout, max_len,
                shared_params=shared_faithful_params)
        else:
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


# ══════════════════════════════════════════════════════════════
# GPT
# ══════════════════════════════════════════════════════════════

class GPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 mode="baseline"):
        super().__init__()
        self.mode = mode
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if mode == "residual_field":
            self.shared_field = SharedFieldParams(d_model, d_field=d_model)
            self.blocks = nn.ModuleList([
                ResidualFieldBlock(d_model, n_head, dropout, max_len,
                                   self.shared_field, layer_idx=i)
                for i in range(n_layer)
            ])
        else:
            shared_faithful = None
            if mode == "faithful":
                d_f = d_model // n_head
                shared_faithful = {
                    "w_deposit": nn.Linear(d_model, n_head * d_f, bias=False),
                    "w_mod": nn.Linear(d_f, d_model // n_head, bias=False),
                }
                self.shared_w_deposit = shared_faithful["w_deposit"]
                self.shared_w_mod = shared_faithful["w_mod"]

            self.blocks = nn.ModuleList([
                Block(d_model, n_head, dropout, max_len, mode=mode,
                      shared_faithful_params=shared_faithful)
                for _ in range(n_layer)
            ])

    def forward(self, idx):
        x = self.tok_emb(idx)
        x = self.drop(x)

        if self.mode == "residual_field":
            field = None
            for block in self.blocks:
                x, field = block(x, field)
        else:
            for block in self.blocks:
                x = block(x)

        return self.head(self.ln_f(x))

    def get_probes(self):
        """Extract diagnostic values after training."""
        probes = {}
        if self.mode == "residual_field":
            probes["betas"] = [b.beta.item() for b in self.blocks]
            probes["alpha"] = self.shared_field.get_alpha().item()
            probes["w_dep_norm"] = self.shared_field.w_dep.weight.norm().item()
            probes["w_read_norm"] = self.shared_field.w_read.weight.norm().item()
        return probes

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
        mode=mode,
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

    probes = model.get_probes()

    return results, model.count_params(), probes


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("RESIDUAL STREAM FIELD: Stigmergic Field in the Residual Stream")
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
    modes = ["baseline", "residual_field", "faithful"]

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
            all_probes = []

            for seed in seeds:
                t_seed = time.perf_counter()
                results, n_params, probes = train_and_eval(
                    mode, task, V, N_EX, seed,
                    d_model=D, n_head=H, n_layer=N_LAYER,
                    train_steps=STEPS, batch_size=BS,
                    n_test_list=n_test_list)
                for n in n_test_list:
                    seed_results[n].append(results[n])
                all_probes.append(probes)
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

            if all_probes and all_probes[0]:
                summary["probes"] = all_probes
                if "betas" in all_probes[0]:
                    mean_betas = [
                        np.mean([p["betas"][i] for p in all_probes])
                        for i in range(N_LAYER)
                    ]
                    mean_alpha = np.mean([p["alpha"] for p in all_probes])
                    beta_str = ", ".join(f"{b:.3f}" for b in mean_betas)
                    print(f"    β per layer: [{beta_str}]")
                    print(f"    α (learned): {mean_alpha:.3f}")

            task_results[mode] = summary

        all_results[task] = task_results

        out_path = OUT_DIR / "exp_residual_field.json"
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
            print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_path = OUT_DIR / "exp_residual_field.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_experiment()
