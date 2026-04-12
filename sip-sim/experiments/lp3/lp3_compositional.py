"""
LP3: Compositional Generalisation — Does the Rule Generalise?
==============================================================

Construct a controlled language with entity-role structure.
Hold out 15% of character-object pairs from training.
Train SharedM and Transformer at matched params.
Evaluate: perplexity on held-out combinations, top-k accuracy.

The language:
  CHAR wanted/found/liked/saw/took OBJ .
  CHAR was happy/sad/scared/excited .
  CHAR and CHAR played with OBJ .

Characters: Lily, Tom, Bear, Fish, Dog, Cat, Bird, Fox, Bee, Ant (10)
Actions: wanted, found, liked, saw, took, held, dropped, made (8)
Objects: cake, ball, toy, book, hat, cup, box, ring, bell, kite (10)
Emotions: happy, sad, scared, excited, tired, proud, shy, brave (8)

Total CHAR-OBJ pairs: 10 x 10 = 100
Hold out 15 pairs (never seen in training).
Test: does the model predict held-out CHAR-OBJ combos?

python3 lp3_compositional.py
python3 lp3_compositional.py --dry-run
"""

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
print("LP3: Compositional Generalisation", flush=True)

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
import random

if args.dry_run:
    DEVICE = torch.device("cpu")
    STEPS = 500
    BATCH_SIZE = 32
    LR = 3e-4
    D_MODEL = 64
    N_LAYERS = 4
    N_HEADS = 4
    N_SENTENCES = 5000
else:
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    STEPS = 3000
    BATCH_SIZE = 64
    LR = 2e-4
    D_MODEL = 128
    N_LAYERS = 6
    N_HEADS = 4
    N_SENTENCES = 50000

OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ── Language ─────────────────────────────────────────────────────────

CHARS = ["Lily", "Tom", "Bear", "Fish", "Dog", "Cat", "Bird", "Fox", "Bee", "Ant"]
ACTIONS = ["wanted", "found", "liked", "saw", "took", "held", "dropped", "made"]
OBJECTS = ["cake", "ball", "toy", "book", "hat", "cup", "box", "ring", "bell", "kite"]
EMOTIONS = ["happy", "sad", "scared", "excited", "tired", "proud", "shy", "brave"]

# Build vocabulary
SPECIAL = ["<pad>", "<bos>", "<eos>"]
CONNECTORS = ["the", "a", "and", "was", "with", ".", "played"]
ALL_WORDS = SPECIAL + CHARS + ACTIONS + OBJECTS + EMOTIONS + CONNECTORS
WORD2ID = {w: i for i, w in enumerate(ALL_WORDS)}
ID2WORD = {i: w for w, i in WORD2ID.items()}
VOCAB_SIZE = len(ALL_WORDS)
PAD, BOS, EOS = WORD2ID["<pad>"], WORD2ID["<bos>"], WORD2ID["<eos>"]

print(f"  Vocabulary: {VOCAB_SIZE} tokens", flush=True)

# Hold out 15 CHAR-OBJ pairs
random.seed(42)
all_pairs = [(c, o) for c in CHARS for o in OBJECTS]
random.shuffle(all_pairs)
HOLDOUT_PAIRS = set((c, o) for c, o in all_pairs[:15])
TRAIN_PAIRS = set((c, o) for c, o in all_pairs[15:])

print(f"  Holdout pairs: {len(HOLDOUT_PAIRS)}", flush=True)
print(f"  Train pairs: {len(TRAIN_PAIRS)}", flush=True)
print(f"  Example holdouts: {list(HOLDOUT_PAIRS)[:5]}", flush=True)


def encode(words):
    return [WORD2ID[w] for w in words]


def generate_sentence(pairs_allowed):
    """Generate one sentence using only the allowed CHAR-OBJ pairs."""
    template = random.choice(["action", "emotion", "compound"])

    if template == "action":
        # CHAR ACTION the OBJ .
        char = random.choice(CHARS)
        obj = random.choice(OBJECTS)
        if (char, obj) not in pairs_allowed:
            # Resample until valid
            valid = [(c, o) for c, o in pairs_allowed if c in CHARS and o in OBJECTS]
            char, obj = random.choice(valid)
        action = random.choice(ACTIONS)
        return ["<bos>", char, action, "the", obj, ".", "<eos>"]

    elif template == "emotion":
        # CHAR was EMOTION .
        char = random.choice(CHARS)
        emotion = random.choice(EMOTIONS)
        return ["<bos>", char, "was", emotion, ".", "<eos>"]

    else:
        # CHAR and CHAR played with the OBJ .
        c1, c2 = random.sample(CHARS, 2)
        obj = random.choice(OBJECTS)
        # Check both pairs
        if (c1, obj) not in pairs_allowed or (c2, obj) not in pairs_allowed:
            valid_objs = [o for o in OBJECTS if (c1, o) in pairs_allowed and (c2, o) in pairs_allowed]
            if not valid_objs:
                return generate_sentence(pairs_allowed)  # retry
            obj = random.choice(valid_objs)
        return ["<bos>", c1, "and", c2, "played", "with", "the", obj, ".", "<eos>"]


def make_dataset(pairs, n_sentences):
    sentences = []
    for _ in range(n_sentences):
        sent = generate_sentence(pairs)
        sentences.append(encode(sent))
    return sentences


# ── Data ─────────────────────────────────────────────────────────────

print("Generating datasets...", flush=True)
train_sentences = make_dataset(TRAIN_PAIRS, N_SENTENCES)
# Test set: specifically generate sentences with HOLDOUT pairs
test_sentences = []
for c, o in HOLDOUT_PAIRS:
    for action in ACTIONS:
        test_sentences.append(encode(["<bos>", c, action, "the", o, ".", "<eos>"]))

# Also generate some in-distribution test sentences
id_test_sentences = make_dataset(TRAIN_PAIRS, len(test_sentences))

MAX_LEN = max(len(s) for s in train_sentences + test_sentences + id_test_sentences)

def pad(sentences):
    return torch.tensor([s + [PAD] * (MAX_LEN - len(s)) for s in sentences], dtype=torch.long)

train_data = pad(train_sentences)
test_ood = pad(test_sentences)
test_id = pad(id_test_sentences)

print(f"  Train: {len(train_data)} sentences, max_len={MAX_LEN}", flush=True)
print(f"  Test OOD: {len(test_ood)} (held-out CHAR-OBJ pairs)", flush=True)
print(f"  Test ID: {len(test_id)} (in-distribution)", flush=True)


# ── Models ───────────────────────────────────────────────────────────

class SharedMBlock(nn.Module):
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

    def forward(self, x):
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


class LM(nn.Module):
    def __init__(self, block_fn, d, L):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.blocks = nn.ModuleList([block_fn() for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))


def make_sharedm():
    A = nn.Parameter(torch.randn(D_MODEL, D_MODEL) * 0.02)
    B = nn.Parameter(torch.randn(D_MODEL, D_MODEL) * 0.02)
    model = LM(lambda: SharedMBlock(D_MODEL, N_HEADS, A, B), D_MODEL, N_LAYERS)
    model.A = A
    model.B = B
    return model

def make_transformer():
    return LM(lambda: TransformerBlock(D_MODEL, N_HEADS), D_MODEL, N_LAYERS)


# ── Training ─────────────────────────────────────────────────────────

def get_batch(data, batch_size=BATCH_SIZE):
    ix = torch.randint(len(data), (batch_size,))
    batch = data[ix].to(DEVICE)
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


def evaluate_set(model, data):
    """Evaluate loss and per-position accuracy on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE].to(DEVICE)
            x = batch[:, :-1]
            y = batch[:, 1:]
            logits = model(x)

            # Loss (ignore padding)
            mask = (y != PAD)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction='none')
            loss = (loss.reshape_as(y) * mask).sum()
            total_loss += loss.item()

            # Top-5 accuracy for the OBJECT position (position 4 in "CHAR ACTION the OBJ")
            # Position 3 in x (0-indexed after BOS) predicts position 4 in y
            preds = logits[:, 3, :].topk(5, dim=-1).indices  # (batch, 5)
            targets = y[:, 3]  # the OBJ token
            correct = (preds == targets.unsqueeze(1)).any(dim=1)
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    return total_loss / total_tokens, total_correct / len(data)


def train_model(model, name, steps=STEPS):
    model = model.to(DEVICE)
    p = sum(p.numel() for p in model.parameters())
    print(f"\n  {name}: {p:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warmup = 200 if not args.dry_run else 30
    sched = torch.optim.lr_scheduler.LambdaLR(opt,
        lambda s: s/warmup if s < warmup else 0.5*(1+math.cos(math.pi*(s-warmup)/(steps-warmup))))

    t0 = time.time()
    log_every = 50 if args.dry_run else 300
    val_log = []

    for step in range(steps):
        model.train()
        x, y = get_batch(train_data)
        logits = model(x)
        mask = (y != PAD)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction='none')
        loss = (loss.reshape_as(y) * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if (step+1) % log_every == 0 or step == 0:
            id_loss, id_acc = evaluate_set(model, test_id)
            ood_loss, ood_acc = evaluate_set(model, test_ood)
            val_log.append((step+1, id_loss, ood_loss, id_acc, ood_acc))
            print(f"    step {step+1:5d} | ID loss {id_loss:.4f} acc@5 {id_acc:.3f} | "
                  f"OOD loss {ood_loss:.4f} acc@5 {ood_acc:.3f} | {time.time()-t0:.0f}s", flush=True)

    # Final evaluation
    id_loss, id_acc = evaluate_set(model, test_id)
    ood_loss, ood_acc = evaluate_set(model, test_ood)
    print(f"    FINAL | ID loss {id_loss:.4f} acc@5 {id_acc:.3f} | OOD loss {ood_loss:.4f} acc@5 {ood_acc:.3f}", flush=True)

    return {
        "name": name,
        "params": p,
        "id_loss": id_loss,
        "ood_loss": ood_loss,
        "id_acc_top5": id_acc,
        "ood_acc_top5": ood_acc,
        "generalisation_gap": ood_loss - id_loss,
        "val_log": val_log,
        "time": time.time() - t0,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("  LP3: COMPOSITIONAL GENERALISATION", flush=True)
    print(f"  {len(HOLDOUT_PAIRS)} held-out CHAR-OBJ pairs", flush=True)
    print("=" * 60, flush=True)

    results = {}
    for name, make_fn in [("SharedM", make_sharedm), ("Transformer", make_transformer)]:
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}", flush=True)
        torch.manual_seed(42)
        model = make_fn()
        r = train_model(model, name)
        results[name] = r
        del model

    # ── Summary ──────────────────────────────────────────────────────

    print(f"\n{'='*60}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Model':<15s} {'ID loss':>10s} {'OOD loss':>10s} {'Gap':>8s} {'ID acc@5':>10s} {'OOD acc@5':>10s}", flush=True)
    for name, r in results.items():
        print(f"  {name:<15s} {r['id_loss']:>10.4f} {r['ood_loss']:>10.4f} {r['generalisation_gap']:>+7.4f} "
              f"{r['id_acc_top5']:>10.3f} {r['ood_acc_top5']:>10.3f}", flush=True)

    sm = results["SharedM"]
    tf = results["Transformer"]
    print(f"\n  Generalisation gap (OOD - ID loss):", flush=True)
    print(f"    SharedM:     {sm['generalisation_gap']:+.4f}", flush=True)
    print(f"    Transformer: {tf['generalisation_gap']:+.4f}", flush=True)
    if sm['generalisation_gap'] < tf['generalisation_gap']:
        print(f"  ★ SharedM generalises better (smaller gap)", flush=True)
    else:
        print(f"  ✗ Transformer generalises equally or better", flush=True)

    print(f"\n  OOD top-5 accuracy:", flush=True)
    print(f"    SharedM:     {sm['ood_acc_top5']:.1%}", flush=True)
    print(f"    Transformer: {tf['ood_acc_top5']:.1%}", flush=True)
    if sm['ood_acc_top5'] > tf['ood_acc_top5']:
        print(f"  ★ SharedM predicts novel combinations better", flush=True)
    else:
        print(f"  ✗ Transformer predicts novel combinations equally or better", flush=True)

    with open(OUT / "lp3_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {OUT / 'lp3_metrics.json'}", flush=True)

    # Plot training dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, r in results.items():
        color = '#1E88E5' if 'SharedM' in name else '#E53935'
        steps = [v[0] for v in r['val_log']]
        id_losses = [v[1] for v in r['val_log']]
        ood_losses = [v[2] for v in r['val_log']]
        ood_accs = [v[4] for v in r['val_log']]
        axes[0].plot(steps, ood_losses, '-', color=color, linewidth=2, label=f'{name} OOD')
        axes[0].plot(steps, id_losses, '--', color=color, linewidth=1, alpha=0.5, label=f'{name} ID')
        axes[1].plot(steps, ood_accs, 'o-', color=color, linewidth=2, markersize=4, label=name)

    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
    axes[0].set_title("In-distribution vs OOD loss", fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Top-5 accuracy")
    axes[1].set_title("OOD accuracy (novel CHAR-OBJ pairs)", fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "lp3_compositional.png", dpi=150, bbox_inches='tight')
    print(f"  Plot: {OUT / 'lp3_compositional.png'}", flush=True)


if __name__ == "__main__":
    main()
