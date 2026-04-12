"""RG-Constrained Social Field: Faithful position-by-position experiment.

Uses the PROVEN DualChannelAttention (position-by-position field, deposits
from output, per-position key modulation) that achieved 0.999 on cumsum
at 2x in prior work. Tests whether the RG equalization constraint helps
on top of a field that already works, at 4 layers where collapse can occur.

Matches prior work parameters: d=48, H=4, n_examples=8 (T=18), V=16.
Only change: 4 layers instead of 2.

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_rg_faithful.py
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

def pick_device():
    """Benchmark MPS vs CPU for position-by-position workload."""
    if not torch.backends.mps.is_available():
        return "cpu"
    # Quick benchmark: 100 small matmuls sequentially
    for dev in ["cpu", "mps"]:
        x = torch.randn(64, 4, 1, 12, device=dev)
        k = torch.randn(64, 4, 17, 12, device=dev)
        if dev == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            _ = (x @ k.transpose(-2, -1))
        if dev == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {dev}: 200 small matmuls in {dt*1000:.0f}ms")
    # MPS has high dispatch overhead for sequential small ops; use CPU
    print("  -> Using CPU for sequential position-by-position processing")
    return "cpu"

DEVICE = pick_device()


# ── Model components ──────────────────────────────────────────

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


# ── Position-by-position DualChannelAttention with RG tracking ──

class DualChannelAttentionRG(nn.Module):
    """Exact reproduction of the proven DualChannelAttention architecture
    with added field influence tracking for RG measurement.

    Position-by-position processing: at each position i, the field state
    is updated from the attention OUTPUT, closing the feedback loop.
    Each key at position j is modulated by the field state at position j.
    This is the architecture that achieved 0.999 on cumsum at 2x.
    """

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.10,
                 learnable_retain=False, layer_idx=0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.d_f = self.head_dim
        self.layer_idx = layer_idx

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

        self.w_deposit = nn.Linear(d_model, n_head * self.d_f, bias=False)
        self.w_mod = nn.Linear(self.d_f, self.head_dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_len)

        self.learnable_retain = learnable_retain
        if learnable_retain:
            init_logit = math.log((1 - evap_rate) / evap_rate)
            self.retain_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.retain = 1.0 - evap_rate

        self._field_influence = torch.tensor(0.0)
        self._key_norm = torch.tensor(1.0)

    def _get_retain(self):
        if self.learnable_retain:
            return torch.sigmoid(self.retain_logit)
        return self.retain

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

        field_state = torch.zeros(B, H, d_f, device=x.device, dtype=x.dtype)
        field_states = []
        output_list = []
        total_mod_norm = torch.tensor(0.0, device=x.device)
        total_key_norm = torch.tensor(0.0, device=x.device)
        n_mod = 0

        for i in range(T):
            k_i_raw = k[:, :, i:i+1, :]

            if i > 0:
                fh = torch.stack(field_states, dim=2)  # (B, H, i, d_f)
                mod = self.w_mod(fh)
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

            deposit = self.w_deposit(out_flat).view(B, H, d_f)
            field_state = retain * field_state + deposit
            field_states.append(field_state)

        out = torch.stack(output_list, dim=1)
        out = self.resid_drop(self.out_proj(out))

        if n_mod > 0:
            self._field_influence = total_mod_norm / n_mod
            self._key_norm = total_key_norm / n_mod
        else:
            self._field_influence = torch.tensor(0.0, device=x.device)
            self._key_norm = torch.tensor(1.0, device=x.device)

        return out


class BaselineAttention(nn.Module):
    """Standard causal attention, no field."""

    def __init__(self, d_model, n_head, dropout, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
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
                 use_field=True, learnable_retain=False, layer_idx=0):
        super().__init__()
        self.use_field = use_field
        self.ln1 = RMSNorm(d_model)
        if use_field:
            self.attn = DualChannelAttentionRG(
                d_model, n_head, dropout, max_len, evap_rate,
                learnable_retain=learnable_retain, layer_idx=layer_idx)
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


class GPT_RG(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 evap_rate=0.10, use_field=True, learnable_retain=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, max_len, evap_rate,
                  use_field=use_field, learnable_retain=learnable_retain,
                  layer_idx=i)
            for i in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def rg_loss(self):
        influences = []
        for block in self.blocks:
            attn = block.attn
            fi = attn._field_influence
            kn = attn._key_norm
            if isinstance(fi, torch.Tensor) and fi.item() > 0:
                influences.append(fi / (kn + 1e-8))
        if len(influences) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        I = torch.stack(influences)
        return I.std() / (I.mean() + 1e-8)

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
            if hasattr(attn, 'retain_logit'):
                info[f"L{i}_retain"] = torch.sigmoid(attn.retain_logit).item()
            elif hasattr(attn, 'retain'):
                info[f"L{i}_retain"] = attn.retain
        return info

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data generation ───────────────────────────────────────────

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
                   train_steps, batch_size, evap_rate,
                   lambda_rg, n_test_list):
    max_n = max(n_test_list)
    max_seq = 2 * (max_n + 1)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    use_field = mode != "baseline"
    learnable_retain = mode in ("learnable", "rg_loss")

    model = GPT_RG(
        vocab_size, max_seq, d_model, n_head, n_layer, dropout=0.05,
        evap_rate=evap_rate, use_field=use_field,
        learnable_retain=learnable_retain,
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

        if lambda_rg > 0 and use_field:
            rg = model.rg_loss()
            loss = loss + lambda_rg * rg

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
    print("RG EXPERIMENT (FAITHFUL): Position-by-position DualChannel")
    print(f"Matching prior work: d=48, H=4, V=16, n=8, T_train=18")
    print(f"New: 4 layers (vs 2) to test collapse + RG constraint")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    V = 16
    D = 48
    H = 4
    N_EX = 8
    STEPS = 5000
    BS = 64
    EVAP = 0.10
    seeds = [42, 137, 256]
    n_test_list = [N_EX, N_EX * 2, N_EX * 3]

    layer_configs = [
        (2, "2L"),   # sanity check: reproduce prior 0.999 result
        (4, "4L"),   # collapse test + RG constraint
    ]

    # 2L only needs baseline+fixed to confirm field works
    # 4L needs the full sweep
    modes_2L = [
        ("baseline", 0.0),
        ("fixed", 0.0),
    ]
    modes_4L = [
        ("baseline", 0.0),
        ("fixed", 0.0),
        ("learnable", 0.0),
        ("rg_loss", 1.0),
        ("rg_loss", 5.0),
    ]

    all_results = {}
    t0 = time.perf_counter()

    for n_layer, layer_label in layer_configs:
        print(f"\n{'─' * 70}")
        print(f"  {layer_label} ({n_layer} layers)")
        print(f"{'─' * 70}")

        current_modes = modes_2L if n_layer == 2 else modes_4L
        for mode, lam in current_modes:
            label = f"{layer_label}_{mode}" if lam == 0 else f"{layer_label}_{mode}_lam={lam}"
            print(f"\n  {label}:")
            seed_results = {n: [] for n in n_test_list}
            field_infos = []

            for seed in seeds:
                results, fi, n_params = train_and_eval(
                    mode, V, N_EX, seed,
                    d_model=D, n_head=H, n_layer=n_layer,
                    train_steps=STEPS, batch_size=BS, evap_rate=EVAP,
                    lambda_rg=lam, n_test_list=n_test_list)
                for n in n_test_list:
                    seed_results[n].append(results[n])
                field_infos.append(fi)
                elapsed = time.perf_counter() - t0
                accs = [f"{results[n]:.3f}" for n in n_test_list]
                print(f"    seed={seed}  [{', '.join(accs)}]  ({elapsed:.0f}s)")

            summary = {
                "label": label,
                "mode": mode,
                "lambda_rg": lam,
                "n_layer": n_layer,
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
                infs = [fi.get(f"L{l}_influence", 0) for l in range(n_layer)]
                rets = [fi.get(f"L{l}_retain", 0) for l in range(n_layer)]
                print(f"    influence: {['%.4f' % v for v in infs]}")
                print(f"    retention: {['%.4f' % v for v in rets]}")

            all_results[label] = summary

    # Summary tables
    for n_layer, layer_label in layer_configs:
        print(f"\n{'═' * 70}")
        print(f"  SUMMARY: {layer_label}")
        print(f"{'═' * 70}")
        header = f"  {'Mode':>25s}"
        for n in n_test_list:
            header += f"  {'T='+str(2*(n+1)):>14s}"
        header += "  field_CV"
        print(header)
        print(f"  {'─' * 75}")

        for label, s in all_results.items():
            if not label.startswith(layer_label):
                continue
            line = f"  {label:>25s}"
            for n in n_test_list:
                r = s['results'][str(n)]
                line += f"  {r['mean']:.3f}+/-{r['std']:.3f}"
            fis = s['field_info']
            if fis and any(f"L0_influence" in fi for fi in fis):
                cvs = []
                for fi in fis:
                    vals = [fi.get(f"L{l}_influence", 0) for l in range(s['n_layer'])]
                    mu = np.mean(vals)
                    if mu > 0:
                        cvs.append(np.std(vals) / (mu + 1e-8))
                if cvs:
                    line += f"  {np.mean(cvs):.3f}"
            print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    with open(OUT_DIR / "exp_faithful_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {OUT_DIR / 'exp_faithful_results.json'}")
    return all_results


if __name__ == "__main__":
    run_experiment()
