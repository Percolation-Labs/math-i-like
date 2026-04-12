"""RG-Constrained Social Field: Experiments 1 & 2.

Experiment 1: Single-stream cumsum mod 16, 4 layers.
  Tests whether learnable epsilon collapses to Layer-0 monopoly,
  and whether the RG influence equalization loss prevents it.

Experiment 2: Bimodal cumsum (two interleaved independent cumsums).
  Tests whether the RG constraint enables multimodal learning at
  smaller model size by preventing modality collapse.

Uses vectorized decay-matrix field computation (not position-by-position)
for tractable runtimes on MPS.

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_rg_experiment.py
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

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


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


# ── Chunk-parallel attention with RG instrumentation ──────────

class ChunkAttentionRG(nn.Module):
    """Chunk-parallel causal attention with social field and RG tracking.

    Field is computed via decay-matrix within each chunk, with cross-chunk
    field state carried forward. Key modulation is applied per-chunk.
    Much faster than position-by-position (vectorized within chunks).
    """

    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.05,
                 use_field=True, learnable_retain=False, layer_idx=0,
                 eps_base_param=None, eps_delta_param=None, chunk_size=8):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.use_field = use_field
        self.layer_idx = layer_idx
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, max_len)

        if use_field:
            self.field_dim = self.head_dim
            self.w_deposit = nn.Linear(d_model, n_head * self.field_dim, bias=False)
            nn.init.normal_(self.w_deposit.weight, std=0.02)
            self.w_mod = nn.Linear(self.field_dim, self.head_dim, bias=False)
            nn.init.zeros_(self.w_mod.weight)

            self.learnable_retain = learnable_retain
            self.eps_base_param = eps_base_param
            self.eps_delta_param = eps_delta_param

            if eps_base_param is not None:
                pass  # uses shared params
            elif learnable_retain:
                init_logit = math.log((1 - evap_rate) / evap_rate)
                self.retain_logit = nn.Parameter(torch.tensor(init_logit))
            else:
                self.fixed_retain = 1.0 - evap_rate

        self._field_influence = torch.tensor(0.0)
        self._key_norm = torch.tensor(1.0)

    def _get_retain(self):
        if not self.use_field:
            return 1.0
        if self.eps_base_param is not None:
            eps = torch.sigmoid(self.eps_base_param + self.layer_idx * self.eps_delta_param)
            return 1.0 - eps
        if self.learnable_retain:
            return torch.sigmoid(self.retain_logit)
        return self.fixed_retain

    def _attend(self, q, k, v):
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        Tq, Tk = q.shape[2], k.shape[2]
        offset = Tk - Tq
        mask = torch.ones(Tq, Tk, device=q.device, dtype=torch.bool).triu(diagonal=offset + 1)
        att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        return att @ v

    def forward(self, x):
        B, T, _ = x.shape
        H, d = self.n_head, self.head_dim
        CS = self.chunk_size

        qkv = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if not self.use_field:
            out = self._attend(q, k, v)
            out = out.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
            return self.resid_drop(self.out_proj(out))

        retain = self._get_retain()
        n_chunks = (T + CS - 1) // CS
        field_state = torch.zeros(B, H, self.field_dim, device=x.device)

        outputs = []
        k_cache = []
        v_cache = []
        mod_norms = []
        key_norms = []

        for ci in range(n_chunks):
            s = ci * CS
            e = min(s + CS, T)
            q_c = q[:, :, s:e, :]
            k_c = k[:, :, s:e, :]
            v_c = v[:, :, s:e, :]

            if ci > 0 and self.use_field:
                shift = self.w_mod(field_state)  # (B, H, d)
                k_c = k_c + shift.unsqueeze(2)
                mod_norms.append(shift.norm(dim=-1).mean())
                key_norms.append(k_c.norm(dim=-1).mean())

            if k_cache:
                k_full = torch.cat(k_cache + [k_c], dim=2)
                v_full = torch.cat(v_cache + [v_c], dim=2)
            else:
                k_full = k_c
                v_full = v_c

            out_c = self._attend(q_c, k_full, v_full)
            outputs.append(out_c)

            out_flat = out_c.permute(0, 2, 1, 3).reshape(B, e - s, self.d_model)

            # Update field state
            deposits = self.w_deposit(out_flat)
            deposits = deposits.reshape(B, e - s, H, self.field_dim)
            chunk_len = e - s

            if isinstance(retain, torch.Tensor):
                steps = torch.arange(chunk_len - 1, -1, -1, device=x.device).float()
                decay_weights = retain ** steps
                weighted = deposits * decay_weights.reshape(1, chunk_len, 1, 1)
                field_state = (retain ** chunk_len) * field_state + weighted.sum(dim=1)
            else:
                steps = torch.arange(chunk_len - 1, -1, -1, device=x.device).float()
                decay_weights = retain ** steps
                weighted = deposits * decay_weights.reshape(1, chunk_len, 1, 1)
                field_state = (retain ** chunk_len) * field_state + weighted.sum(dim=1)

            k_cache.append(k_c)
            v_cache.append(v_c)

        output = torch.cat(outputs, dim=2)
        output = output.permute(0, 2, 1, 3).reshape(B, T, self.d_model)

        if mod_norms:
            self._field_influence = torch.stack(mod_norms).mean()
            self._key_norm = torch.stack(key_norms).mean()
        else:
            self._field_influence = torch.tensor(0.0, device=x.device)
            self._key_norm = torch.tensor(1.0, device=x.device)

        return self.resid_drop(self.out_proj(output))


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len, evap_rate=0.05,
                 use_field=True, learnable_retain=False, layer_idx=0,
                 eps_base_param=None, eps_delta_param=None, chunk_size=8):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = ChunkAttentionRG(
            d_model, n_head, dropout, max_len, evap_rate,
            use_field=use_field, learnable_retain=learnable_retain,
            layer_idx=layer_idx, eps_base_param=eps_base_param,
            eps_delta_param=eps_delta_param, chunk_size=chunk_size)
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
                 evap_rate=0.05, use_field=True, learnable_retain=False,
                 rg_flow=False, chunk_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        eps_base = eps_delta = None
        if rg_flow:
            init_logit = math.log(evap_rate / (1 - evap_rate))
            self.eps_base = nn.Parameter(torch.tensor(init_logit))
            self.eps_delta = nn.Parameter(torch.tensor(0.0))
            eps_base = self.eps_base
            eps_delta = self.eps_delta
            learnable_retain = False

        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, max_len, evap_rate,
                  use_field=use_field, learnable_retain=learnable_retain,
                  layer_idx=i, eps_base_param=eps_base, eps_delta_param=eps_delta,
                  chunk_size=chunk_size)
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
            elif hasattr(attn, 'eps_base_param') and attn.eps_base_param is not None:
                info[f"L{i}_retain"] = attn._get_retain().item()
            elif hasattr(attn, 'fixed_retain'):
                info[f"L{i}_retain"] = attn.fixed_retain
        if hasattr(self, 'eps_base'):
            info["eps_base"] = torch.sigmoid(self.eps_base).item()
            info["eps_delta"] = self.eps_delta.item()
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


def gen_bimodal_cumsum(batch_size, n_examples, vocab_size, rng):
    """Two independent cumsums interleaved: a1 sA1 b1 sB1 a2 sA2 b2 sB2 ...

    The model must track two independent running sums simultaneously.
    We only evaluate on the LAST target (second stream's final cumsum).
    """
    n_total = n_examples + 1
    full_len = 4 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)
    for b in range(batch_size):
        sum_a = 0
        sum_b = 0
        for i in range(n_total):
            a = int(rng.integers(0, vocab_size))
            sum_a = (sum_a + a) % vocab_size
            bv = int(rng.integers(0, vocab_size))
            sum_b = (sum_b + bv) % vocab_size
            base = 4 * i
            tokens[b, base] = a
            tokens[b, base + 1] = sum_a
            tokens[b, base + 2] = bv
            tokens[b, base + 3] = sum_b
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    # Mask inputs (positions 0,2,4,...) — only predict cumsum outputs
    for i in range(full_len - 1):
        if i % 2 == 1:
            targets[:, i] = -100
    last_targets = tokens[:, -1]
    return torch.tensor(input_ids), torch.tensor(targets), torch.tensor(last_targets)


# ── Training ──────────────────────────────────────────────────

def train_and_eval(mode, vocab_size, n_examples, seed, gen_fn,
                   d_model=64, n_head=4, n_layer=4,
                   train_steps=5000, batch_size=64, evap_rate=0.05,
                   lambda_rg=0.0, n_test_list=None, chunk_size=8):
    if n_test_list is None:
        n_test_list = [n_examples]

    is_bimodal = (gen_fn == gen_bimodal_cumsum)
    max_n = max(n_test_list)
    max_seq = (4 if is_bimodal else 2) * (max_n + 1)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    use_field = mode != "baseline"
    learnable_retain = mode in ("learnable", "rg_loss")
    rg_flow = mode == "rg_flow"

    model = GPT_RG(
        vocab_size, max_seq, d_model, n_head, n_layer, dropout=0.05,
        evap_rate=evap_rate, use_field=use_field,
        learnable_retain=learnable_retain, rg_flow=rg_flow,
        chunk_size=chunk_size,
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
            val_in, _, val_last = gen_fn(512, n_test, vocab_size, rng=eval_rng)
            val_in = val_in.to(DEVICE)
            val_logits = model(val_in)
            val_pred = val_logits[:, -1].argmax(dim=-1).cpu()
            val_acc = (val_pred == val_last).float().mean().item()
            results[n_test] = val_acc

    field_info = model.get_field_info()
    return results, field_info, model.count_params()


# ── Experiment 1: Single-stream cumsum ────────────────────────

def run_experiment_1():
    print("=" * 70)
    print("EXPERIMENT 1: RG-Constrained Cumsum mod 16")
    print(f"4 layers, d=64, H=4, chunk=8, device={DEVICE}")
    print("=" * 70)

    V = 16
    n_train = 16
    n_test_list = [n_train, n_train * 2, n_train * 3]
    seeds = [42, 137, 256]

    modes = [
        ("baseline", 0.0),
        ("fixed", 0.0),
        ("learnable", 0.0),
        ("rg_loss", 0.1),
        ("rg_loss", 0.5),
        ("rg_loss", 1.0),
        ("rg_loss", 5.0),
        ("rg_flow", 0.0),
    ]

    all_results = {}
    t0 = time.perf_counter()

    for mode, lam in modes:
        label = mode if lam == 0 else f"{mode}_lam={lam}"
        print(f"\n  Mode: {label}")
        seed_results = {n: [] for n in n_test_list}
        field_infos = []

        for seed in seeds:
            results, fi, n_params = train_and_eval(
                mode, V, n_train, seed, gen_cumsum,
                d_model=64, n_head=4, n_layer=4,
                train_steps=5000, lambda_rg=lam, n_test_list=n_test_list,
                chunk_size=8)
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

        # Field influence summary
        if field_infos and any(f"L0_influence" in fi for fi in field_infos):
            for fi in field_infos[-1:]:
                layer_inf = [fi.get(f"L{l}_influence", 0) for l in range(4)]
                layer_ret = [fi.get(f"L{l}_retain", 0) for l in range(4)]
                print(f"    field influence: {['%.4f' % v for v in layer_inf]}")
                print(f"    retention:       {['%.4f' % v for v in layer_ret]}")

        all_results[label] = summary

    # Summary table
    print(f"\n{'_' * 80}")
    header = f"  {'Mode':>20s}"
    for n in n_test_list:
        header += f"  {'T='+str(2*(n+1)):>14s}"
    header += "  field_CV"
    print(header)
    print(f"{'_' * 80}")

    for label, s in all_results.items():
        line = f"  {label:>20s}"
        for n in n_test_list:
            r = s['results'][str(n)]
            line += f"  {r['mean']:.3f}+/-{r['std']:.3f}"
        fis = s['field_info']
        if fis and any(f"L0_influence" in fi for fi in fis):
            cvs = []
            for fi in fis:
                vals = [fi.get(f"L{l}_influence", 0) for l in range(4)]
                mu = np.mean(vals)
                if mu > 0:
                    cvs.append(np.std(vals) / (mu + 1e-8))
            if cvs:
                line += f"  {np.mean(cvs):.3f}"
        print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nExperiment 1 complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    with open(OUT_DIR / "exp1_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {OUT_DIR / 'exp1_results.json'}")
    return all_results


# ── Experiment 2: Bimodal cumsum ──────────────────────────────

def run_experiment_2():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Bimodal Cumsum — Modality Competition")
    print(f"4 layers, H=4, d_model sweep, chunk=8, device={DEVICE}")
    print("=" * 70)

    V = 16
    n_train = 12
    n_test_list = [n_train, n_train * 2]
    seeds = [42, 137, 256]
    d_models = [32, 48, 64, 96]

    modes = [
        ("baseline", 0.0),
        ("fixed", 0.0),
        ("learnable", 0.0),
        ("rg_loss", 1.0),
    ]

    all_results = {}
    t0 = time.perf_counter()

    for d_model in d_models:
        n_head = max(2, d_model // 16)

        print(f"\n  d_model={d_model}, H={n_head}")
        print(f"  {'_' * 55}")

        for mode, lam in modes:
            label = f"d{d_model}_{mode}" if lam == 0 else f"d{d_model}_{mode}_lam={lam}"
            seed_results = {n: [] for n in n_test_list}
            field_infos = []

            for seed in seeds:
                results, fi, n_params = train_and_eval(
                    mode, V, n_train, seed, gen_bimodal_cumsum,
                    d_model=d_model, n_head=n_head, n_layer=4,
                    train_steps=6000, lambda_rg=lam, n_test_list=n_test_list,
                    chunk_size=8)
                for n in n_test_list:
                    seed_results[n].append(results[n])
                field_infos.append(fi)

            summary = {
                "label": label,
                "d_model": d_model,
                "mode": mode,
                "lambda_rg": lam,
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
                f"T={4*(n+1)}:{np.mean(seed_results[n]):.3f}+/-{np.std(seed_results[n]):.3f}"
                for n in n_test_list)
            elapsed = time.perf_counter() - t0
            print(f"    {label:>25s}:  {accs_str}  ({elapsed:.0f}s)")

            all_results[label] = summary

    # Summary table
    print(f"\n{'_' * 80}")
    header = f"  {'Config':>25s}"
    for n in n_test_list:
        header += f"  {'T='+str(4*(n+1)):>14s}"
    header += "  field_CV"
    print(header)
    print(f"{'_' * 80}")

    for label, s in all_results.items():
        line = f"  {label:>25s}"
        for n in n_test_list:
            r = s['results'][str(n)]
            line += f"  {r['mean']:.3f}+/-{r['std']:.3f}"
        fis = s['field_info']
        if fis and any(f"L0_influence" in fi for fi in fis):
            cvs = []
            for fi in fis:
                vals = [fi.get(f"L{l}_influence", 0) for l in range(4)]
                mu = np.mean(vals)
                if mu > 0:
                    cvs.append(np.std(vals) / (mu + 1e-8))
            if cvs:
                line += f"  {np.mean(cvs):.3f}"
        print(line)

    elapsed = time.perf_counter() - t0
    print(f"\nExperiment 2 complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    with open(OUT_DIR / "exp2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {OUT_DIR / 'exp2_results.json'}")
    return all_results


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}\n")

    t0 = time.perf_counter()
    exp1 = run_experiment_1()
    exp2 = run_experiment_2()
    total = time.perf_counter() - t0

    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE in {total:.0f}s ({total/60:.1f}min)")
    print(f"{'=' * 70}")
