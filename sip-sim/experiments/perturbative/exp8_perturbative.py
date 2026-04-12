"""
Exp 8: Perturbative Ontological Corrections
============================================

Tests the perturbative correction architecture: small, context-dependent
corrections around the orbital (SharedM) path.

The orbital model gives the "classical path" — the mean-field prediction
from structural rules. The perturbation is the "quantum correction" —
small deviations that depend on context (boundary conditions).

Architecture:
  SharedM layer output + gate(h) * correction_mlp(h)

  gate(h): scalar, context-dependent. Reads residual stream.
           → 0 when context is weak (pure orbital output)
           → small positive when context provides specificity

  correction_mlp(h): small MLP (d → d_corr → d). Produces the perturbation.

  Constraint: perturbation acts on value stream only, never on coupling.
  Constraint: ||perturbation|| << ||orbital output|| (it's a correction, not a replacement)

Four models compared at matched ~10M total params:
  1. SharedM (baseline orbital)
  2. SharedM + Perturbation (the new design)
  3. SharedM + OldOnto (parallel attention, r=4, for comparison)
  4. Transformer (standard baseline)

Diagnostics:
  - Gate magnitude over training (should increase as context matures)
  - Perturbation norm vs orbital norm (should stay small)
  - Ablation: set gate=0 at inference, measure loss by sequence position
  - Per-position loss curves (perturbation should help late > early tokens)
"""

import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', help='Quick CPU test with tiny models')
args = parser.parse_args()

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
print("Starting exp8 (perturbative onto)...", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import math

if args.dry_run:
    print("CPU TEST MODE (tiny models, 200 steps)", flush=True)
    DEVICE = torch.device("cpu")
    TARGET = 200_000
    STEPS = 200
    BATCH_SIZE = 8
    SEQ_LEN = 64
    LR = 3e-4
    D_CORR = 16
else:
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    TARGET = 10_000_000
    STEPS = 5000
    BATCH_SIZE = 32
    SEQ_LEN = 256
    LR = 2e-4
    D_CORR = 64

OUT = Path(__file__).parent / "results" / "exp8"
OUT.mkdir(parents=True, exist_ok=True)

import tiktoken
ENC = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab
EOT = ENC.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


# ── Data ─────────────────────────────────────────────────────────────

def load_data():
    cache_path = Path(__file__).parent / "data" / "tinystories_tokens.pt"
    data = torch.load(cache_path, weights_only=True)
    return data["train"][:5_000_000], data["val"]


def get_batch(data):
    ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+SEQ_LEN] for i in ix]).to(DEVICE)
    y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix]).to(DEVICE)
    return x, y


# ── Model components ─────────────────────────────────────────────────

class SharedMBlock(nn.Module):
    """SharedM attention block (no perturbation)."""
    def __init__(self, d, h, A, B):
        super().__init__()
        self.h, self.dh = h, d // h
        self.A, self.B = A, B
        self.ln1 = nn.LayerNorm(d)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q = (h @ self.A).view(B, S, self.h, self.dh).transpose(1, 2)
        k = (h @ self.B).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.h, self.dh).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s.masked_fill_(torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1), float('-inf'))
        out = (F.softmax(s, -1) @ v).transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.Wo(out)
        return x + self.ff(self.ln2(x))


class PerturbativeBlock(nn.Module):
    """SharedM attention + context-gated perturbative correction.

    After the standard SharedM attention + FFN, adds:
      x = x + gate(x) * correction(x)

    gate: sigmoid(linear(x)) → scalar per token, context-dependent
    correction: small MLP (d → d_corr → d)

    The perturbation acts on the value/FFN output (the combined residual),
    NOT on the coupling. M remains the sole coupling authority.
    """
    def __init__(self, d, h, A, B, d_corr):
        super().__init__()
        self.h_heads, self.dh = h, d // h
        self.A, self.B = A, B
        self.ln1 = nn.LayerNorm(d)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

        # Perturbation components
        self.ln_corr = nn.LayerNorm(d)
        self.gate = nn.Linear(d, 1, bias=True)  # scalar gate per token
        self.correction = nn.Sequential(
            nn.Linear(d, d_corr), nn.GELU(), nn.Linear(d_corr, d)
        )
        # Init gate bias negative so perturbation starts near zero
        nn.init.constant_(self.gate.bias, -3.0)
        # Init correction output near zero
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x, collect_diagnostics=False):
        B, S, D = x.shape
        # Standard SharedM forward
        h = self.ln1(x)
        q = (h @ self.A).view(B, S, self.h_heads, self.dh).transpose(1, 2)
        k = (h @ self.B).view(B, S, self.h_heads, self.dh).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.h_heads, self.dh).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s.masked_fill_(torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1), float('-inf'))
        attn_weights = F.softmax(s, -1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.Wo(out)
        x = x + self.ff(self.ln2(x))

        # Perturbative correction
        h_corr = self.ln_corr(x)
        g = torch.sigmoid(self.gate(h_corr))    # (B, S, 1)
        delta = self.correction(h_corr)          # (B, S, D)
        x = x + g * delta

        if collect_diagnostics:
            # Attention entropy: -sum(p * log(p)) per token, averaged over heads
            # Higher entropy = diffuse attention, lower = sharp/confident
            attn_ent = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1)  # (B, H, S)
            attn_ent = attn_ent.mean(dim=1)  # (B, S) — average over heads
            return x, g.detach(), delta.detach(), attn_ent.detach()
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1 = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

    def forward(self, x, collect_diagnostics=False):
        B, S, D = x.shape
        h = self.ln1(x)
        q = self.Wq(h).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.Wk(h).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.h, self.dh).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s.masked_fill_(torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1), float('-inf'))
        out = (F.softmax(s, -1) @ v).transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.Wo(out)
        return x + self.ff(self.ln2(x))


# ── Full Models ──────────────────────────────────────────────────────

class SharedMLM(nn.Module):
    def __init__(self, d, L, h):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        self.A = nn.Parameter(torch.randn(d, d) * 0.02)
        self.B = nn.Parameter(torch.randn(d, d) * 0.02)
        self.blocks = nn.ModuleList([SharedMBlock(d, h, self.A, self.B) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)
        self.head.weight = self.emb.weight
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))


class PerturbativeLM(nn.Module):
    """SharedM + perturbative corrections at each layer."""
    def __init__(self, d, L, h, d_corr=D_CORR):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        self.A = nn.Parameter(torch.randn(d, d) * 0.02)
        self.B = nn.Parameter(torch.randn(d, d) * 0.02)
        self.blocks = nn.ModuleList([
            PerturbativeBlock(d, h, self.A, self.B, d_corr) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)
        self.head.weight = self.emb.weight
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, disable_perturbation=False):
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        for b in self.blocks:
            if disable_perturbation:
                # Run only the orbital part (zero out gate)
                B_sz, S, D = h.shape
                h_norm = b.ln1(h)
                q = (h_norm @ b.A).view(B_sz, S, b.h_heads, b.dh).transpose(1, 2)
                k = (h_norm @ b.B).view(B_sz, S, b.h_heads, b.dh).transpose(1, 2)
                v = b.Wv(h_norm).view(B_sz, S, b.h_heads, b.dh).transpose(1, 2)
                s = (q @ k.transpose(-2, -1)) / math.sqrt(b.dh)
                s.masked_fill_(torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1), float('-inf'))
                out = (F.softmax(s, -1) @ v).transpose(1, 2).contiguous().view(B_sz, S, D)
                h = h + b.Wo(out)
                h = h + b.ff(b.ln2(h))
                # Skip perturbation
            else:
                h = b(h)
        return self.head(self.ln(h))

    def collect_diagnostics(self, x):
        """Run forward collecting gate magnitudes, perturbation norms, and attention entropy."""
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        gates = []
        perturb_norms = []
        orbital_norms = []
        attn_entropies = []
        for b in self.blocks:
            h_before = h.clone()
            h, g, delta, attn_ent = b(h, collect_diagnostics=True)
            gates.append(g.mean().item())
            perturb_norms.append(delta.norm(dim=-1).mean().item())
            orbital_norms.append(h_before.norm(dim=-1).mean().item())
            attn_entropies.append(attn_ent.mean().item())
        return self.head(self.ln(h)), gates, perturb_norms, orbital_norms, attn_entropies


class TransformerLM(nn.Module):
    def __init__(self, d, L, h):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)
        self.head.weight = self.emb.weight
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))


# ── Config search ────────────────────────────────────────────────────

def count_sharedm(d, L):
    emb = VOCAB_SIZE * d + SEQ_LEN * d
    shared = 2 * d * d
    per_layer = 2 * d * d + 2 * d * 4 * d + 2 * d
    return emb + shared + L * per_layer + d

def count_perturbative(d, L, d_corr=D_CORR):
    base = count_sharedm(d, L)
    # Per-layer perturbation: LN(d) + gate(d→1) + correction(d→d_corr→d)
    per_layer_perturb = d + (d + 1) + (d * d_corr + d_corr + d_corr * d + d)
    return base + L * per_layer_perturb

def count_transformer(d, L):
    emb = VOCAB_SIZE * d + SEQ_LEN * d
    per_layer = 4 * d * d + 2 * d * 4 * d + 2 * d
    return emb + L * per_layer + d

def find_config(count_fn, target, n_heads=4, d_min=96, min_L=3, max_L=7, **kwargs):
    best, best_diff = None, float('inf')
    for d in range(d_min, 600, 4):
        if d % n_heads != 0: continue
        for L in range(min_L, max_L + 1):
            p = count_fn(d, L, **kwargs)
            if abs(p - target) < best_diff:
                best_diff = abs(p - target)
                best = (d, L, n_heads, p)
    if best is None or best[3] > target * 2:
        for d in range(32, d_min, 4):
            if d % n_heads != 0: continue
            for L in range(min_L, max_L + 1):
                p = count_fn(d, L, **kwargs)
                if abs(p - target) < best_diff:
                    best_diff = abs(p - target)
                    best = (d, L, n_heads, p)
    return best


# ── Training ─────────────────────────────────────────────────────────

def evaluate(model, val_data, n=30):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n):
            x, y = get_batch(val_data)
            logits = model(x)
            losses.append(F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)).item())
    return np.mean(losses)


def per_position_loss(model, val_data, n=30, disable_perturbation=False):
    """Compute loss at each sequence position."""
    model.eval()
    pos_losses = torch.zeros(SEQ_LEN, device=DEVICE)
    pos_counts = torch.zeros(SEQ_LEN, device=DEVICE)
    with torch.no_grad():
        for _ in range(n):
            x, y = get_batch(val_data)
            if disable_perturbation and hasattr(model, 'forward'):
                logits = model(x, disable_perturbation=True)
            else:
                logits = model(x)
            for pos in range(SEQ_LEN):
                loss_pos = F.cross_entropy(logits[:, pos, :], y[:, pos])
                pos_losses[pos] += loss_pos
                pos_counts[pos] += 1
    return (pos_losses / pos_counts).cpu().numpy()


@torch.no_grad()
def generate(model, prompt, max_new=100):
    model.eval()
    ids = torch.tensor([ENC.encode(prompt)], device=DEVICE)
    for _ in range(max_new):
        logits = model(ids[:, -SEQ_LEN:])[:, -1, :] / 0.8
        v, _ = torch.topk(logits, 40)
        logits[logits < v[:, [-1]]] = float('-inf')
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == EOT: break
    return ENC.decode(ids[0].tolist())


def train_model(model, train_data, val_data, name, steps=STEPS):
    model = model.to(DEVICE)
    p = sum(p.numel() for p in model.parameters())
    print(f"\n  {name}: {p:,} params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    warmup = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt,
        lambda s: s/warmup if s < warmup else 0.5*(1+math.cos(math.pi*(s-warmup)/(steps-warmup))))
    t0 = time.time()
    val_log = []
    diag_log = []

    for step in range(steps):
        model.train()
        x, y = get_batch(train_data)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        log_every = 50 if args.dry_run else 500
        if (step+1) % log_every == 0 or step == 0:
            val = evaluate(model, val_data)
            val_log.append((step+1, val))
            el = time.time() - t0
            extra = ""

            # Collect perturbation diagnostics if available
            if hasattr(model, 'collect_diagnostics'):
                model.eval()
                with torch.no_grad():
                    xd, _ = get_batch(val_data)
                    _, gates, pnorms, onorms, attn_ents = model.collect_diagnostics(xd)
                    avg_gate = np.mean(gates)
                    avg_ratio = np.mean([p/(o+1e-8) for p, o in zip(pnorms, onorms)])
                    avg_entropy = np.mean(attn_ents)
                    diag_log.append((step+1, avg_gate, avg_ratio, avg_entropy))
                    extra = f" | gate={avg_gate:.4f} | δ/h={avg_ratio:.4f} | attn_H={avg_entropy:.3f}"

            print(f"    step {step+1:5d} | val {val:.4f} | {el:.0f}s{extra}", flush=True)

    final = evaluate(model, val_data)
    print(f"    FINAL | val {final:.4f} | {time.time()-t0:.0f}s", flush=True)

    # Generate samples
    for prompt in ["Once upon a time", "The little"]:
        text = generate(model, prompt)
        print(f"    [{prompt}] {text[:150]}", flush=True)

    return {"name": name, "final_val": final, "params": p, "time": time.time()-t0,
            "val_log": val_log, "diag_log": diag_log}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("  EXP 8: PERTURBATIVE ONTOLOGICAL CORRECTIONS", flush=True)
    print("=" * 60, flush=True)

    train_data, val_data = load_data()

    # Find configs at matched params
    cfg_sm = find_config(count_sharedm, TARGET)
    cfg_pt = find_config(count_perturbative, TARGET, d_corr=D_CORR)
    cfg_tf = find_config(count_transformer, TARGET)

    print(f"\n  Configs (target {TARGET:,}):", flush=True)
    print(f"    SharedM:        d={cfg_sm[0]}, L={cfg_sm[1]}, ~{cfg_sm[3]:,}", flush=True)
    print(f"    Perturbative:   d={cfg_pt[0]}, L={cfg_pt[1]}, ~{cfg_pt[3]:,} (d_corr={D_CORR})", flush=True)
    print(f"    Transformer:    d={cfg_tf[0]}, L={cfg_tf[1]}, ~{cfg_tf[3]:,}", flush=True)

    results = {}

    # 1. SharedM (orbital baseline)
    print(f"\n{'─'*60}\n  [1/3] SharedM", flush=True)
    m1 = SharedMLM(*cfg_sm[:3])
    r1 = train_model(m1, train_data, val_data, "SharedM")
    results["sharedm"] = r1
    torch.save(m1.state_dict(), OUT / "ckpt_sharedm.pt")
    del m1; torch.mps.empty_cache() if DEVICE.type == "mps" else None

    # 2. Perturbative (the new design)
    print(f"\n{'─'*60}\n  [2/3] SharedM + Perturbation", flush=True)
    m2 = PerturbativeLM(*cfg_pt[:3], d_corr=D_CORR)
    r2 = train_model(m2, train_data, val_data, "Perturbative")
    results["perturbative"] = r2

    # Perturbation diagnostics
    print(f"\n  Perturbation ablation:", flush=True)
    m2.eval()
    with torch.no_grad():
        val_with = evaluate(m2, val_data)
        val_without = evaluate(m2, val_data)  # need disable
        # Per-position analysis
        pos_with = per_position_loss(m2, val_data, n=20)
        pos_without = per_position_loss(m2, val_data, n=20, disable_perturbation=True)
        improvement = pos_without - pos_with  # positive = perturbation helped

    quarter = SEQ_LEN // 4
    print(f"    With perturbation:    {val_with:.4f}", flush=True)
    print(f"    Without perturbation: {np.mean(pos_without):.4f}", flush=True)
    print(f"    Improvement early (pos 0-{quarter}):  {np.mean(improvement[:quarter]):.4f}", flush=True)
    print(f"    Improvement late  (pos {SEQ_LEN-quarter}-{SEQ_LEN}): {np.mean(improvement[SEQ_LEN-quarter:]):.4f}", flush=True)

    torch.save(m2.state_dict(), OUT / "ckpt_perturbative.pt")
    results["perturbative"]["ablation"] = {
        "val_with": val_with,
        "pos_improvement_early": float(np.mean(improvement[:quarter])),
        "pos_improvement_late": float(np.mean(improvement[SEQ_LEN-quarter:])),
    }
    del m2; torch.mps.empty_cache() if DEVICE.type == "mps" else None

    # 3. Transformer
    print(f"\n{'─'*60}\n  [3/3] Transformer", flush=True)
    m3 = TransformerLM(*cfg_tf[:3])
    r3 = train_model(m3, train_data, val_data, "Transformer")
    results["transformer"] = r3
    torch.save(m3.state_dict(), OUT / "ckpt_transformer.pt")
    del m3; torch.mps.empty_cache() if DEVICE.type == "mps" else None

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    base = results["transformer"]["final_val"]
    for key in ["sharedm", "perturbative", "transformer"]:
        r = results[key]
        delta = (r["final_val"] - base) / base * 100
        print(f"  {r['name']:<20s} {r['params']:>10,} {r['final_val']:>8.4f} ({delta:+.2f}%)", flush=True)

    # Perturbation diagnostic plot
    diag = results["perturbative"].get("diag_log", [])
    if diag:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Perturbation Diagnostics", fontsize=14, fontweight='bold')
        steps_d = [d[0] for d in diag]
        gates = [d[1] for d in diag]
        ratios = [d[2] for d in diag]
        entropies = [d[3] for d in diag] if len(diag[0]) > 3 else None

        axes[0,0].plot(steps_d, gates, 'o-', color='#1E88E5', linewidth=2)
        axes[0,0].set_xlabel("Step")
        axes[0,0].set_ylabel("Mean gate magnitude")
        axes[0,0].set_title("Context gate opens over training")
        axes[0,0].grid(alpha=0.3)

        axes[0,1].plot(steps_d, ratios, 'o-', color='#E53935', linewidth=2)
        axes[0,1].set_xlabel("Step")
        axes[0,1].set_ylabel("||δ|| / ||h||")
        axes[0,1].set_title("Perturbation ratio (should stay small)")
        axes[0,1].grid(alpha=0.3)

        if entropies:
            axes[1,0].plot(steps_d, entropies, 'o-', color='#43A047', linewidth=2)
            axes[1,0].set_xlabel("Step")
            axes[1,0].set_ylabel("Mean attention entropy")
            axes[1,0].set_title("Attention entropy over training")
            axes[1,0].grid(alpha=0.3)

            # Gate vs entropy correlation (prediction: gate opens when entropy drops)
            axes[1,1].scatter(entropies, gates, c=steps_d, cmap='viridis', s=60, edgecolors='black', linewidth=0.5)
            axes[1,1].set_xlabel("Attention entropy (high=diffuse, low=sharp)")
            axes[1,1].set_ylabel("Gate magnitude")
            axes[1,1].set_title("Gate vs entropy\n(prediction: negative correlation)")
            # Fit correlation
            if len(entropies) > 2:
                corr = np.corrcoef(entropies, gates)[0, 1]
                axes[1,1].text(0.05, 0.95, f"r = {corr:.3f}",
                              transform=axes[1,1].transAxes, fontsize=11, va='top',
                              bbox=dict(facecolor='lightyellow', edgecolor='orange', alpha=0.8))
            axes[1,1].grid(alpha=0.3)
            cb = fig.colorbar(axes[1,1].collections[0], ax=axes[1,1], label="Training step")
        else:
            axes[1,0].axis('off')
            axes[1,1].axis('off')

        plt.tight_layout()
        plt.savefig(OUT / "perturbation_diagnostics.png", dpi=150, bbox_inches='tight')
        print(f"\n  Diagnostics plot: {OUT / 'perturbation_diagnostics.png'}", flush=True)

        if entropies and len(entropies) > 2:
            corr = np.corrcoef(entropies, gates)[0, 1]
            print(f"  Gate-entropy correlation: r = {corr:.3f}", flush=True)
            print(f"  (negative = gate opens when attention sharpens — predicted)", flush=True)

    with open(OUT / "exp8_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUT / 'exp8_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
