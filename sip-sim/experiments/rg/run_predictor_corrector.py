"""Predictor-corrector: recover the feedback loop with 2 parallel passes.

Pass 1 (predict): input deposits → parallel field scan → attention → outputs
Pass 2 (correct): output deposits from Pass 1 → parallel field scan → attention

Both passes use standard causal attention (no position-by-position loop).
The field states are pre-computed via scan before attention runs.

Modes:
  baseline     - no field
  faithful     - position-by-position serial (gold standard)
  decoupled    - input deposits only, 1 pass (already tested)
  pred_corr_1  - predictor-corrector, 1 correction pass
  pred_corr_2  - predictor-corrector, 2 correction passes

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_predictor_corrector.py
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


# ── Components ────────────────────────────────────────────────

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


class SharedFieldParams(nn.Module):
    def __init__(self, d_model, n_head, d_f, head_dim, evap_rate=0.10):
        super().__init__()
        self.w_deposit = nn.Linear(d_model, n_head * d_f, bias=False)
        self.w_mod = nn.Linear(d_f, head_dim, bias=False)
        init_logit = math.log((1 - evap_rate) / evap_rate)
        self.retain_logit = nn.Parameter(torch.tensor(init_logit))

    def get_retain(self):
        return torch.sigmoid(self.retain_logit)


# ── Parallel field scan ───────────────────────────────────────

def parallel_field_scan(deposits, retain):
    """Compute field states from deposits via sequential scan.

    deposits: (B, T, H, d_f)
    retain: scalar

    Returns field_states: (B, T, H, d_f) where
      field_states[:, t] = retain * field_states[:, t-1] + deposits[:, t]

    Note: In a production implementation this would use Blelloch parallel
    scan in O(log T). Here we simulate the same result sequentially since
    we're measuring accuracy, not speed.
    """
    B, T, H, d_f = deposits.shape
    field_states = torch.empty_like(deposits)
    state = torch.zeros(B, H, d_f, device=deposits.device, dtype=deposits.dtype)
    for t in range(T):
        state = retain * state + deposits[:, t]
        field_states[:, t] = state
    return field_states


# ── Attention with pre-computed field states ──────────────────

def field_attention_from_states(q, k, v, field_states, w_mod, scale, mask):
    """Standard causal attention but keys are modulated by pre-computed fields.

    q, k, v: (B, H, T, d_h)
    field_states: (B, T, H, d_f) or None
    w_mod: Linear(d_f, d_h) or None
    """
    if field_states is not None:
        # field_states: (B, T, H, d_f) -> (B, H, T, d_f)
        fs = field_states.permute(0, 2, 1, 3)
        mod = w_mod(fs)  # (B, H, T, d_h)
        k = k + mod

    B, H, T, d_h = q.shape
    att = (q @ k.transpose(-2, -1)) * scale
    att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    out = att @ v  # (B, H, T, d_h)
    return out.transpose(1, 2).contiguous().view(B, T, H * d_h)


# ── Attention blocks ──────────────────────────────────────────

class PredCorrAttention(nn.Module):
    """Attention with predictor-corrector field computation."""

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.10,
                 layer_idx=0, shared_params=None, mode='faithful'):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.d_f = self.head_dim
        self.layer_idx = layer_idx
        self.mode = mode

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))
        self.rope = RotaryEmbedding(self.head_dim, max_len)
        self.shared_params = shared_params

        self._field_influence = torch.tensor(0.0)
        self._key_norm = torch.tensor(1.0)

    def _get_retain(self):
        return self.shared_params.get_retain()

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

        retain = self._get_retain()
        w_dep = self.shared_params.w_deposit
        w_mod = self.shared_params.w_mod
        scale = d_h ** -0.5

        if self.mode == 'baseline':
            out = field_attention_from_states(q, k, v, None, None, scale, self.mask)

        elif self.mode == 'faithful':
            out = self._faithful_forward(x, q, k, v, retain, w_dep, w_mod, B, T, C, H, d_h, d_f)

        elif self.mode == 'decoupled':
            deposits = w_dep(x).view(B, T, H, d_f)
            field_states = parallel_field_scan(deposits, retain)
            out = field_attention_from_states(q, k, v, field_states, w_mod, scale, self.mask)
            self._compute_influence(k, field_states, w_mod)

        elif self.mode.startswith('pred_corr'):
            n_corrections = int(self.mode.split('_')[-1])
            # Pass 0: predict with input deposits
            deposits = w_dep(x).view(B, T, H, d_f)
            field_states = parallel_field_scan(deposits, retain)
            out = field_attention_from_states(q, k, v, field_states, w_mod, scale, self.mask)

            # Correction passes: use outputs to recompute deposits
            for _ in range(n_corrections):
                deposits = w_dep(out).view(B, T, H, d_f)
                field_states = parallel_field_scan(deposits, retain)
                out = field_attention_from_states(q, k, v, field_states, w_mod, scale, self.mask)

            self._compute_influence(k, field_states, w_mod)

        out = self.resid_drop(self.out_proj(out))
        return out

    def _compute_influence(self, k, field_states, w_mod):
        with torch.no_grad():
            fs = field_states.permute(0, 2, 1, 3)
            mod = w_mod(fs)
            self._field_influence = mod.norm(dim=-1).mean()
            self._key_norm = k.norm(dim=-1).mean()

    def _faithful_forward(self, x, q, k, v, retain, w_dep, w_mod, B, T, C, H, d_h, d_f):
        """Position-by-position serial processing (gold standard)."""
        field_state = torch.zeros(B, H, d_f, device=x.device, dtype=x.dtype)
        field_states = []
        output_list = []
        total_mod_norm = torch.tensor(0.0, device=x.device)
        total_key_norm = torch.tensor(0.0, device=x.device)
        n_mod = 0

        for i in range(T):
            k_i_raw = k[:, :, i:i+1, :]
            if i > 0:
                fh = torch.stack(field_states, dim=2)
                mod = w_mod(fh)
                k_prev_mod = k[:, :, :i, :] + mod
                k_mod = torch.cat([k_prev_mod, k_i_raw], dim=2)
                total_mod_norm = total_mod_norm + mod.norm(dim=-1).mean()
                total_key_norm = total_key_norm + k[:, :, :i, :].norm(dim=-1).mean()
                n_mod += 1
            else:
                k_mod = k_i_raw

            q_i = q[:, :, i:i+1, :]
            att_i = (q_i @ k_mod.transpose(-2, -1)) * (d_h ** -0.5)
            att_i = F.softmax(att_i, dim=-1)
            att_i = self.attn_drop(att_i)
            out_i = (att_i @ v[:, :, :i+1, :]).squeeze(2)
            out_flat = out_i.transpose(1, 2).contiguous().view(B, C)
            output_list.append(out_flat)

            deposit = w_dep(out_flat).view(B, H, d_f)
            field_state = retain * field_state + deposit
            field_states.append(field_state)

        out = torch.stack(output_list, dim=1)
        if n_mod > 0:
            self._field_influence = total_mod_norm / n_mod
            self._key_norm = total_key_norm / n_mod
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
        self._field_influence = torch.tensor(0.0)
        self._key_norm = torch.tensor(1.0)

    def forward(self, x):
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
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.10,
                 use_field=True, layer_idx=0, shared_params=None, mode='faithful'):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        if use_field:
            self.attn = PredCorrAttention(
                d_model, n_head, dropout, max_len, evap_rate,
                layer_idx=layer_idx, shared_params=shared_params, mode=mode)
        else:
            self.attn = BaselineAttention(d_model, n_head, dropout, max_len)
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


class GPT_PC(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 evap_rate=0.10, use_field=True, mode='faithful'):
        super().__init__()
        self.vocab_size = vocab_size

        shared_params = None
        if use_field:
            head_dim = d_model // n_head
            shared_params = SharedFieldParams(d_model, n_head, head_dim, head_dim, evap_rate)
            self.shared_field = shared_params

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, max_len, evap_rate,
                  use_field=use_field, layer_idx=i,
                  shared_params=shared_params, mode=mode)
            for i in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def get_field_info(self):
        info = {}
        for i, block in enumerate(self.blocks):
            attn = block.attn
            fi = attn._field_influence
            kn = attn._key_norm
            if isinstance(fi, torch.Tensor):
                info[f"L{i}_influence"] = (fi / (kn + 1e-8)).item()
            else:
                info[f"L{i}_influence"] = 0.0
        return info

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data ──────────────────────────────────────────────────────

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


# ── Training ──────────────────────────────────────────────────

def train_and_eval(mode, vocab_size, n_examples, seed,
                   d_model, n_head, n_layer,
                   train_steps, batch_size, evap_rate, n_test_list):
    max_n = max(n_test_list)
    max_seq = 2 * (max_n + 1)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    use_field = mode != "baseline"

    model = GPT_PC(
        vocab_size, max_seq, d_model, n_head, n_layer, dropout=0.05,
        evap_rate=evap_rate, use_field=use_field, mode=mode,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                   weight_decay=0.10, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps)

    for step in range(train_steps):
        model.train()
        inp, tgt, _ = gen_cumsum(batch_size, n_examples, vocab_size, rng=rng)
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
            val_in, _, val_last = gen_cumsum(512, n_test, vocab_size, rng=eval_rng)
            val_in = val_in.to(DEVICE)
            val_logits = model(val_in)
            val_pred = val_logits[:, -1].argmax(dim=-1).cpu()
            val_acc = (val_pred == val_last).float().mean().item()
            results[n_test] = val_acc

    field_info = model.get_field_info()
    return results, field_info, model.count_params()


# ── Experiment ────────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("PREDICTOR-CORRECTOR: Recover feedback with parallel passes")
    print(f"d=48, H=4, V=16, n=8, T_train=18, 4 layers, tied weights")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    V = 16
    D = 48
    H = 4
    N_EX = 8
    N_LAYER = 4
    STEPS = 5000
    BS = 64
    EVAP = 0.10
    seeds = [42, 137, 256]
    n_test_list = [N_EX, N_EX * 2, N_EX * 3]

    modes = ["baseline", "faithful", "decoupled", "pred_corr_1", "pred_corr_2"]

    all_results = {}
    t0 = time.perf_counter()

    for mode in modes:
        print(f"\n{'─' * 70}")
        print(f"  {mode}")
        print(f"{'─' * 70}")

        seed_results = {n: [] for n in n_test_list}
        field_infos = []

        for seed in seeds:
            results, fi, n_params = train_and_eval(
                mode, V, N_EX, seed,
                d_model=D, n_head=H, n_layer=N_LAYER,
                train_steps=STEPS, batch_size=BS, evap_rate=EVAP,
                n_test_list=n_test_list)
            for n in n_test_list:
                seed_results[n].append(results[n])
            field_infos.append(fi)
            elapsed = time.perf_counter() - t0
            accs = [f"{results[n]:.3f}" for n in n_test_list]
            print(f"    seed={seed}  [{', '.join(accs)}]  "
                  f"params={n_params}  ({elapsed:.0f}s)")

        summary = {
            "label": mode,
            "n_layer": N_LAYER,
            "n_params": n_params,
            "results": {},
            "field_info": field_infos,
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

        if field_infos and any(f"L0_influence" in fi for fi in field_infos):
            fi = field_infos[-1]
            infs = [fi.get(f"L{l}_influence", 0) for l in range(N_LAYER)]
            print(f"    influence: {['%.4f' % v for v in infs]}")

        all_results[mode] = summary

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    header = f"  {'Mode':>15s}  {'Params':>7s}"
    for n in n_test_list:
        header += f"  {'T='+str(2*(n+1)):>14s}"
    print(header)
    print(f"  {'─' * 65}")

    for mode in modes:
        s = all_results[mode]
        line = f"  {mode:>15s}  {s['n_params']:>7d}"
        for n in n_test_list:
            r = s['results'][str(n)]
            line += f"  {r['mean']:.3f}+/-{r['std']:.3f}"
        print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_path = OUT_DIR / "exp_pred_corr_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_experiment()
