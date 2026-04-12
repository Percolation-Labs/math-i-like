"""Weight-tying experiment: the RG fixed point as an architectural constraint.

Instead of a CV loss that the optimizer games, share w_deposit and w_mod
across all layers so collapse is structurally impossible.

Modes:
  baseline     - no field (control)
  untied       - per-layer w_dep, w_mod, learnable eps (reproduces collapse)
  tied_all     - shared w_dep, w_mod, shared retain across layers
  tied_weights - shared w_dep, w_mod; per-layer learnable retain
  field_drop   - per-layer weights; p=0.3 field zeroing at train time

Same task/hyperparams as run_rg_faithful.py: cumsum mod 16, d=48, H=4.

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_rg_tying.py
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
    if not torch.backends.mps.is_available():
        return "cpu"
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


# ── Shared field parameters container ────────────────────────

class SharedFieldParams(nn.Module):
    """Holds w_deposit and w_mod shared across all layers."""

    def __init__(self, d_model, n_head, d_f, head_dim, evap_rate=0.10):
        super().__init__()
        self.w_deposit = nn.Linear(d_model, n_head * d_f, bias=False)
        self.w_mod = nn.Linear(d_f, head_dim, bias=False)
        init_logit = math.log((1 - evap_rate) / evap_rate)
        self.retain_logit = nn.Parameter(torch.tensor(init_logit))

    def get_retain(self):
        return torch.sigmoid(self.retain_logit)


# ── Position-by-position attention ────────────────────────────

class DualChannelAttentionTied(nn.Module):
    """Position-by-position attention with optionally shared field weights.

    If shared_params is provided, w_deposit and w_mod come from there.
    Otherwise, this layer owns its own copies (untied mode).
    """

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.10,
                 learnable_retain=False, layer_idx=0,
                 shared_params=None, tie_retain=False,
                 field_drop_p=0.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.d_f = self.head_dim
        self.layer_idx = layer_idx
        self.field_drop_p = field_drop_p

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

        self.shared_params = shared_params
        self.tie_retain = tie_retain

        if shared_params is None:
            self.w_deposit = nn.Linear(d_model, n_head * self.d_f, bias=False)
            self.w_mod = nn.Linear(self.d_f, self.head_dim, bias=False)
        # else: use shared_params.w_deposit and shared_params.w_mod

        self.rope = RotaryEmbedding(self.head_dim, max_len)

        self.learnable_retain = learnable_retain
        if shared_params is not None and tie_retain:
            pass  # use shared_params.retain_logit
        elif learnable_retain:
            init_logit = math.log((1 - evap_rate) / evap_rate)
            self.retain_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.retain = 1.0 - evap_rate

        self._field_influence = torch.tensor(0.0)
        self._key_norm = torch.tensor(1.0)

    @property
    def _w_deposit(self):
        if self.shared_params is not None:
            return self.shared_params.w_deposit
        return self.w_deposit

    @property
    def _w_mod(self):
        if self.shared_params is not None:
            return self.shared_params.w_mod
        return self.w_mod

    def _get_retain(self):
        if self.shared_params is not None and self.tie_retain:
            return self.shared_params.get_retain()
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
        w_dep = self._w_deposit
        w_mod = self._w_mod

        field_state = torch.zeros(B, H, d_f, device=x.device, dtype=x.dtype)
        field_states = []
        output_list = []
        total_mod_norm = torch.tensor(0.0, device=x.device)
        total_key_norm = torch.tensor(0.0, device=x.device)
        n_mod = 0

        # Field dropout: zero the field for this entire forward pass
        field_active = True
        if self.field_drop_p > 0 and self.training:
            if torch.rand(1).item() < self.field_drop_p:
                field_active = False

        for i in range(T):
            k_i_raw = k[:, :, i:i+1, :]

            if i > 0 and field_active:
                fh = torch.stack(field_states, dim=2)
                mod = w_mod(fh)
                k_prev_mod = k[:, :, :i, :] + mod
                k_mod = torch.cat([k_prev_mod, k_i_raw], dim=2)

                total_mod_norm = total_mod_norm + mod.norm(dim=-1).mean()
                total_key_norm = total_key_norm + k[:, :, :i, :].norm(dim=-1).mean()
                n_mod += 1
            else:
                k_mod = k[:, :, :i+1, :]

            q_i = q[:, :, i:i+1, :]
            att_i = (q_i @ k_mod.transpose(-2, -1)) * (d_h ** -0.5)
            att_i = F.softmax(att_i, dim=-1)
            att_i = self.attn_drop(att_i)

            out_i = (att_i @ v[:, :, :i+1, :]).squeeze(2)
            out_flat = out_i.transpose(1, 2).contiguous().view(B, C)
            output_list.append(out_flat)

            if field_active:
                deposit = w_dep(out_flat).view(B, H, d_f)
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
                 use_field=True, learnable_retain=False, layer_idx=0,
                 shared_params=None, tie_retain=False, field_drop_p=0.0):
        super().__init__()
        self.use_field = use_field
        self.ln1 = RMSNorm(d_model)
        if use_field:
            self.attn = DualChannelAttentionTied(
                d_model, n_head, dropout, max_len, evap_rate,
                learnable_retain=learnable_retain, layer_idx=layer_idx,
                shared_params=shared_params, tie_retain=tie_retain,
                field_drop_p=field_drop_p)
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


class GPT_Tied(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout,
                 evap_rate=0.10, use_field=True, learnable_retain=False,
                 tie_field=False, tie_retain=False, field_drop_p=0.0):
        super().__init__()
        self.vocab_size = vocab_size

        shared_params = None
        if tie_field and use_field:
            head_dim = d_model // n_head
            shared_params = SharedFieldParams(
                d_model, n_head, head_dim, head_dim, evap_rate)
            self.shared_field = shared_params

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, max_len, evap_rate,
                  use_field=use_field, learnable_retain=learnable_retain,
                  layer_idx=i, shared_params=shared_params,
                  tie_retain=tie_retain, field_drop_p=field_drop_p)
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
            if hasattr(attn, 'retain_logit'):
                info[f"L{i}_retain"] = torch.sigmoid(attn.retain_logit).item()
            elif hasattr(attn, 'shared_params') and attn.shared_params is not None and attn.tie_retain:
                info[f"L{i}_retain"] = attn.shared_params.get_retain().item()
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

def train_and_eval(mode_cfg, vocab_size, n_examples, seed,
                   d_model, n_head, n_layer,
                   train_steps, batch_size, evap_rate, n_test_list):
    max_n = max(n_test_list)
    max_seq = 2 * (max_n + 1)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = GPT_Tied(
        vocab_size, max_seq, d_model, n_head, n_layer, dropout=0.05,
        evap_rate=evap_rate, **mode_cfg,
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

MODE_CONFIGS = {
    "baseline": dict(use_field=False, learnable_retain=False,
                     tie_field=False, tie_retain=False, field_drop_p=0.0),
    "untied": dict(use_field=True, learnable_retain=True,
                   tie_field=False, tie_retain=False, field_drop_p=0.0),
    "tied_all": dict(use_field=True, learnable_retain=False,
                     tie_field=True, tie_retain=True, field_drop_p=0.0),
    "tied_weights": dict(use_field=True, learnable_retain=True,
                         tie_field=True, tie_retain=False, field_drop_p=0.0),
    "field_drop": dict(use_field=True, learnable_retain=True,
                       tie_field=False, tie_retain=False, field_drop_p=0.3),
}


def run_experiment():
    print("=" * 70)
    print("WEIGHT-TYING EXPERIMENT: RG Fixed Point via Architecture")
    print(f"d=48, H=4, V=16, n=8, T_train=18, 4 layers")
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

    modes = ["baseline", "untied", "tied_all", "tied_weights", "field_drop"]

    all_results = {}
    t0 = time.perf_counter()

    for mode in modes:
        cfg = MODE_CONFIGS[mode]
        print(f"\n{'─' * 70}")
        print(f"  {mode}  (params: {cfg})")
        print(f"{'─' * 70}")

        seed_results = {n: [] for n in n_test_list}
        field_infos = []

        for seed in seeds:
            results, fi, n_params = train_and_eval(
                cfg, V, N_EX, seed,
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
            "mode": mode,
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
            rets = [fi.get(f"L{l}_retain", 0) for l in range(N_LAYER)]
            print(f"    influence: {['%.4f' % v for v in infs]}")
            print(f"    retention: {['%.4f' % v for v in rets]}")
            mu = np.mean(infs)
            cv = np.std(infs) / (mu + 1e-8) if mu > 0 else 0
            print(f"    CV={cv:.3f}  mean_I={mu:.4f}")

        all_results[mode] = summary

    # Summary table
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY: 4L Weight-Tying Experiment")
    print(f"{'═' * 70}")
    header = f"  {'Mode':>15s}  {'Params':>7s}"
    for n in n_test_list:
        header += f"  {'T='+str(2*(n+1)):>14s}"
    header += f"  {'CV':>6s}  {'mean_I':>8s}"
    print(header)
    print(f"  {'─' * 80}")

    for mode in modes:
        s = all_results[mode]
        line = f"  {mode:>15s}  {s['n_params']:>7d}"
        for n in n_test_list:
            r = s['results'][str(n)]
            line += f"  {r['mean']:.3f}+/-{r['std']:.3f}"
        fis = s['field_info']
        if fis and any(f"L0_influence" in fi for fi in fis):
            all_infs = []
            for fi in fis:
                vals = [fi.get(f"L{l}_influence", 0) for l in range(N_LAYER)]
                all_infs.append(vals)
            flat = [v for row in all_infs for v in row]
            last_infs = all_infs[-1]
            mu = np.mean(last_infs)
            cv = np.std(last_infs) / (mu + 1e-8) if mu > 0 else 0
            line += f"  {cv:>6.3f}  {mu:>8.4f}"
        else:
            line += f"  {'---':>6s}  {'---':>8s}"
        print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_path = OUT_DIR / "exp_tying_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")
    return all_results


if __name__ == "__main__":
    run_experiment()
