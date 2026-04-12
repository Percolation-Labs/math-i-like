"""TinyStories: Residual Stream Field vs Baseline on real narratives.

Does the temporal field produce characteristically different text?
We train both models on TinyStories and compare:
  - Perplexity (does the field help?)
  - Generated samples (does the text read differently?)
  - β_l distribution (how does the model use the field on real text?)
  - Sample quality (temporal coherence, pronoun tracking, narrative arc)

Model: d=256, H=4, L=6 (~8M params), ctx=256
Data:  TinyStories from HuggingFace, ~10M tokens

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_tinystories.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════

def load_tinystories(n_tokens=10_000_000):
    cache_path = Path(__file__).resolve().parent / ".data"
    cache_path.mkdir(parents=True, exist_ok=True)
    tok_file = cache_path / f"tinystories_{n_tokens // 1_000_000}M_tokens.pt"

    if tok_file.exists():
        print(f"Loading cached tokens from {tok_file}")
        tokens = torch.load(tok_file, weights_only=True)
        print(f"  {len(tokens)/1e6:.1f}M tokens loaded")
        return tokens

    print(f"Loading TinyStories and tokenizing {n_tokens/1e6:.0f}M tokens...")
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    all_tokens = []
    for i, example in enumerate(ds):
        text = example["text"]
        tokens = enc.encode_ordinary(text)
        all_tokens.extend(tokens)
        all_tokens.append(eot)
        if len(all_tokens) >= n_tokens:
            break
        if (i + 1) % 10_000 == 0:
            print(f"  {i+1} stories, {len(all_tokens)/1e6:.1f}M tokens")

    all_tokens = all_tokens[:n_tokens]
    tensor = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(tensor, tok_file)
    print(f"  Cached {len(tensor)/1e6:.1f}M tokens to {tok_file}")
    return tensor


class TokenBuffer:
    def __init__(self, tokens, context_length, split="train", val_fraction=0.01):
        n_val = int(len(tokens) * val_fraction)
        if split == "train":
            self.tokens = tokens[:-n_val]
        else:
            self.tokens = tokens[-n_val:]
        self.context_length = context_length

    def sample_batch(self, batch_size, device):
        max_start = len(self.tokens) - self.context_length - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.tokens[s:s + self.context_length] for s in starts])
        y = torch.stack([self.tokens[s + 1:s + self.context_length + 1] for s in starts])
        return x.to(device), y.to(device)


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
    def __init__(self, dim, max_len=4096):
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


def linear_scan_parallel(deposits, alpha):
    """f_t = alpha * f_{t-1} + deposits_t via cumsum trick.

    deposits: (B, T, d_f), alpha: scalar tensor
    Returns: (B, T, d_f)
    """
    B, T, d_f = deposits.shape
    t_idx = torch.arange(T, device=deposits.device, dtype=deposits.dtype)
    inv_alpha_t = (1.0 / alpha) ** t_idx
    alpha_t = alpha ** t_idx
    y = deposits * inv_alpha_t.view(1, T, 1)
    C = torch.cumsum(y, dim=1)
    return C * alpha_t.view(1, T, 1)


# ══════════════════════════════════════════════════════════════
# Attention
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
# Residual Stream Field
# ══════════════════════════════════════════════════════════════

class SharedFieldParams(nn.Module):
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
    def __init__(self, d_model, n_head, dropout, max_len,
                 shared_field, layer_idx):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, max_len)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )
        self.shared_field = shared_field
        self.layer_idx = layer_idx
        self.log_beta = nn.Parameter(torch.tensor(0.0))

    @property
    def beta(self):
        return self.log_beta.exp()

    def forward(self, x, field):
        if field is not None:
            field_normed = self.shared_field.ln_field(field)
            field_read = self.shared_field.w_read(field_normed)
            x = x + self.beta * field_read

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        alpha = self.shared_field.get_alpha()
        deposits = self.shared_field.w_dep(x)
        new_scan = linear_scan_parallel(deposits, alpha)
        field = field + new_scan if field is not None else new_scan

        return x, field


class BaselineBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, max_len)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
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
    def __init__(self, vocab_size=50257, d_model=256, n_head=4, n_layer=6,
                 dropout=0.1, max_len=256, use_field=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_field = use_field

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        if use_field:
            self.shared_field = SharedFieldParams(d_model, d_field=d_model)
            self.blocks = nn.ModuleList([
                ResidualFieldBlock(d_model, n_head, dropout, max_len,
                                   self.shared_field, layer_idx=i)
                for i in range(n_layer)
            ])
        else:
            self.blocks = nn.ModuleList([
                BaselineBlock(d_model, n_head, dropout, max_len)
                for _ in range(n_layer)
            ])

        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight,
                            std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(block.mlp[-2].weight,
                            std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        x = self.drop(x)

        if self.use_field:
            field = None
            for block in self.blocks:
                x, field = block(x, field)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   targets.view(-1), ignore_index=-1)
            return loss, logits
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.max_len else idx[:, -self.max_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_field_info(self):
        if not self.use_field:
            return {}
        betas = [b.beta.item() for b in self.blocks]
        alpha = self.shared_field.get_alpha().item()
        return {"betas": betas, "alpha": alpha}


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

PROMPTS = [
    "Once upon a time there was a little",
    "Lily wanted to play with her",
    "One day, Sam went to the park and",
    "The big dog was very",
    "Mom said they could not go outside because",
]


def train_model(mode, tokens, config):
    d_model = config["d_model"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    ctx = config["ctx"]
    batch_size = config["batch_size"]
    train_steps = config["train_steps"]
    lr = config["lr"]
    grad_accum = config["grad_accum"]

    use_field = (mode == "residual_field")

    train_data = TokenBuffer(tokens, ctx, split="train")
    val_data = TokenBuffer(tokens, ctx, split="val")

    model = GPT(
        vocab_size=50257, d_model=d_model, n_head=n_head,
        n_layer=n_layer, dropout=0.1, max_len=ctx,
        use_field=use_field,
    ).to(DEVICE)

    n_params = model.count_params()
    print(f"  Model: {n_params/1e6:.2f}M params, field={use_field}")

    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))

    warmup_steps = min(200, train_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(train_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    enc = tiktoken.get_encoding("gpt2")

    log = []
    samples = {}
    t0 = time.perf_counter()
    running_loss = 0.0

    torch.manual_seed(42)
    model.train()

    for step in range(1, train_steps + 1):
        total_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_data.sample_batch(batch_size, DEVICE)
            loss, _ = model(x, y)
            loss = loss / grad_accum
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += total_loss

        if step % 100 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            avg_loss = running_loss / min(step, 100)
            eff_batch = batch_size * grad_accum
            total_tokens = step * eff_batch * ctx
            tok_per_sec = total_tokens / elapsed
            ppl = math.exp(min(avg_loss, 20))
            print(f"    step {step:5d}/{train_steps}  loss={avg_loss:.4f}  "
                  f"ppl={ppl:>8.1f}  {tok_per_sec:.0f} tok/s  ({elapsed:.0f}s)")
            running_loss = 0.0

        if step % 500 == 0 or step == train_steps:
            model.eval()

            # Validation
            val_losses = []
            with torch.no_grad():
                for _ in range(30):
                    vx, vy = val_data.sample_batch(batch_size, DEVICE)
                    vl, _ = model(vx, vy)
                    val_losses.append(vl.item())
            val_loss = np.mean(val_losses)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"    >>> val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")

            # Field diagnostics
            field_info = model.get_field_info()
            if field_info:
                betas = field_info["betas"]
                beta_str = ", ".join(f"{b:.3f}" for b in betas)
                print(f"    >>> β per layer: [{beta_str}]")
                print(f"    >>> α = {field_info['alpha']:.4f}")

            # Generate samples from all prompts
            step_samples = {}
            print(f"\n    --- Samples at step {step} ---")
            for prompt in PROMPTS:
                tok_ids = enc.encode(prompt)
                idx = torch.tensor([tok_ids], device=DEVICE)
                gen = model.generate(idx, max_new_tokens=80, temperature=0.8, top_k=40)
                text = enc.decode(gen[0].tolist())
                step_samples[prompt] = text
                short = text[:200].replace("\n", " ")
                print(f"    [{prompt[:30]:30s}] {short}")
            samples[f"step_{step}"] = step_samples
            print()

            model.train()

    # Final validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(50):
            vx, vy = val_data.sample_batch(batch_size, DEVICE)
            vl, _ = model(vx, vy)
            val_losses.append(vl.item())
    final_val = np.mean(val_losses)
    final_ppl = math.exp(min(final_val, 20))
    elapsed = time.perf_counter() - t0
    total_tokens = train_steps * batch_size * grad_accum * ctx
    avg_tok_s = total_tokens / elapsed

    result = {
        "mode": mode,
        "n_params": n_params,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "elapsed_s": elapsed,
        "avg_tok_s": avg_tok_s,
        "field_info": model.get_field_info(),
        "samples": samples,
    }
    print(f"\n  FINAL: val_loss={final_val:.4f}  ppl={final_ppl:.1f}  "
          f"{avg_tok_s:.0f} tok/s  {elapsed:.0f}s")
    return result


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("TINYSTORIES: Residual Stream Field vs Baseline")
    print("=" * 70)

    config = {
        "d_model": 256,
        "n_head": 4,
        "n_layer": 6,
        "ctx": 256,
        "batch_size": 16,
        "grad_accum": 2,
        "train_steps": 3000,
        "lr": 3e-4,
    }

    eff_batch = config["batch_size"] * config["grad_accum"]
    total_tokens = config["train_steps"] * eff_batch * config["ctx"]
    print(f"Config: d={config['d_model']}, H={config['n_head']}, "
          f"L={config['n_layer']}, ctx={config['ctx']}")
    print(f"Training: {config['train_steps']} steps, "
          f"eff_batch={eff_batch}, {total_tokens/1e6:.0f}M tokens seen")

    tokens = load_tinystories(n_tokens=10_000_000)

    modes = ["baseline", "residual_field"]
    all_results = {}

    for mode in modes:
        print(f"\n{'═' * 70}")
        print(f"  MODE: {mode}")
        print(f"{'═' * 70}")

        result = train_model(mode, tokens, config)
        all_results[mode] = result

        out_path = OUT_DIR / "exp_tinystories.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")

    for mode in modes:
        r = all_results[mode]
        print(f"\n  {mode}: {r['n_params']/1e6:.2f}M params  "
              f"val_ppl={r['final_val_ppl']:.1f}  "
              f"{r['avg_tok_s']:.0f} tok/s  {r['elapsed_s']:.0f}s")
        if r["field_info"]:
            betas = r["field_info"]["betas"]
            beta_str = ", ".join(f"{b:.3f}" for b in betas)
            print(f"    β: [{beta_str}]  α={r['field_info']['alpha']:.4f}")

    ppl_delta = all_results["residual_field"]["final_val_ppl"] - \
                all_results["baseline"]["final_val_ppl"]
    print(f"\n  PPL difference: {ppl_delta:+.1f} (negative = field better)")

    # Side-by-side samples
    print(f"\n{'═' * 70}")
    print(f"  SIDE-BY-SIDE GENERATION (final)")
    print(f"{'═' * 70}")

    final_key = f"step_{config['train_steps']}"
    for prompt in PROMPTS:
        print(f"\n  Prompt: \"{prompt}\"")
        for mode in modes:
            text = all_results[mode]["samples"][final_key][prompt]
            short = text[:300].replace("\n", " ")
            print(f"    [{mode:>15s}] {short}")

    out_path = OUT_DIR / "exp_tinystories.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
