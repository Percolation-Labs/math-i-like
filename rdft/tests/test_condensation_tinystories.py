"""
Condensation LM on TinyStories
================================

The real test: can a condensation machine learn next-token prediction
on actual English text?

Architecture:
  - Character-level (simple tokenisation, small vocab)
  - Field dim = 128, matching a small GRU/LSTM
  - Deposition: token embeds into field
  - Evolution: field mixes (learned linear map) + decays
  - Nucleation: field -> next token logits
  - Depletion: prediction consumes field state

Compare to:
  - GRU (same hidden dim)
  - LSTM (same hidden dim)

TinyStories is ideal: simple English, short sequences, learnable patterns.
If the condensation machine learns to predict "Once upon a t..." -> "i",
the field is carrying context and the physics computes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import functools

print = functools.partial(print, flush=True)
torch.manual_seed(42)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ════════════════════════════════════════════════════════════════
# DATA: TinyStories, character-level
# ════════════════════════════════════════════════════════════════

def load_tinystories(max_chars=500_000, seq_len=128):
    """Load TinyStories and prepare character-level sequences."""
    print("Loading TinyStories...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Collect text
    text = []
    total = 0
    for example in ds:
        t = example['text']
        text.append(t)
        total += len(t)
        if total > max_chars:
            break

    corpus = "\n".join(text)[:max_chars]
    print(f"  Corpus: {len(corpus):,} chars")

    # Character vocabulary (printable ASCII subset)
    chars = sorted(set(corpus))
    # Keep only common chars, map rare to <unk>
    char_counts = {}
    for c in corpus:
        char_counts[c] = char_counts.get(c, 0) + 1
    common_chars = [c for c, cnt in char_counts.items() if cnt > 50]
    common_chars = sorted(common_chars)

    char2idx = {c: i+1 for i, c in enumerate(common_chars)}
    char2idx['<unk>'] = 0
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(char2idx)
    print(f"  Vocab: {vocab_size} chars")

    # Encode
    encoded = torch.tensor([char2idx.get(c, 0) for c in corpus], dtype=torch.long)

    # Chunk into sequences
    n_seqs = len(encoded) // seq_len
    data = encoded[:n_seqs * seq_len].view(n_seqs, seq_len)

    # Shuffle and split
    perm = torch.randperm(n_seqs)
    data = data[perm]
    n_train = int(0.9 * n_seqs)
    X_train = data[:n_train].to(DEVICE)
    X_test = data[n_train:].to(DEVICE)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Sample: '{''.join(idx2char.get(i.item(), '?') for i in X_train[0, :60])}'")

    return X_train, X_test, vocab_size, idx2char


# ════════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════════

class CondensationLM(nn.Module):
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.3):
        super().__init__()
        self.vocab = vocab
        self.field_dim = field_dim
        self.n_substeps = n_substeps
        self.dt = dt

        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.05))
        self.nucleate = nn.Linear(field_dim, vocab)
        self.deplete = nn.Linear(vocab, field_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def forward(self, x):
        batch, seq_len = x.shape
        phi = torch.zeros(batch, self.field_dim, device=x.device)
        all_logits = []

        for t in range(seq_len):
            phi = phi + self.embed(x[:, t])
            for _ in range(self.n_substeps):
                phi = phi + self.dt * (self.mix(phi) - F.softplus(self.decay) * phi)

            logits = self.nucleate(phi)
            all_logits.append(logits)

            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)

        return torch.stack(all_logits, dim=1)


class CondensationLM_NoDepl(nn.Module):
    """Ablation: no depletion."""
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.3):
        super().__init__()
        self.vocab = vocab
        self.field_dim = field_dim
        self.n_substeps = n_substeps
        self.dt = dt
        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.05))
        self.nucleate = nn.Linear(field_dim, vocab)

    def forward(self, x):
        batch, seq_len = x.shape
        phi = torch.zeros(batch, self.field_dim, device=x.device)
        all_logits = []
        for t in range(seq_len):
            phi = phi + self.embed(x[:, t])
            for _ in range(self.n_substeps):
                phi = phi + self.dt * (self.mix(phi) - F.softplus(self.decay) * phi)
            all_logits.append(self.nucleate(phi))
        return torch.stack(all_logits, dim=1)


class GRULM(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        return self.head(self.gru(self.embed(x))[0])


class LSTMLM(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        return self.head(self.lstm(self.embed(x))[0])


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def train_lm(model, X_tr, X_te, vocab, epochs=30, lr=1e-3, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_tr.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0; nb = 0; total_correct = 0; total_tokens = 0

        for s in range(0, min(n, 2000), bs):  # cap batches per epoch
            idx = perm[s:s+bs]
            xb = X_tr[idx]
            logits = model(xb[:, :-1])
            targets = xb[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(-1) == targets).sum().item()
            total_tokens += targets.numel()
            nb += 1

        model.eval()
        with torch.no_grad():
            te_logits = model(X_te[:200, :-1])
            te_targets = X_te[:200, 1:]
            te_loss = F.cross_entropy(te_logits.reshape(-1, vocab), te_targets.reshape(-1)).item()
            te_acc = (te_logits.argmax(-1) == te_targets).float().mean().item()

        train_acc = total_correct / total_tokens
        print(f"    ep {ep:3d}  train_loss={total_loss/nb:.3f}  "
              f"train_acc={train_acc:.3f}  test_loss={te_loss:.3f}  test_acc={te_acc:.3f}")

    return te_loss, te_acc


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def generate(model, idx2char, seed_text, length=100, vocab_size=None, char2idx=None):
    """Generate text from the model."""
    model.eval()
    if char2idx is None:
        char2idx = {c: i for i, c in idx2char.items()}

    tokens = [char2idx.get(c, 0) for c in seed_text]
    x = torch.tensor([tokens], device=DEVICE)

    generated = list(seed_text)
    with torch.no_grad():
        # Build up context
        logits = model(x)
        # Now generate
        for _ in range(length):
            next_logits = logits[0, -1]  # last position
            # Sample (temperature 0.8)
            probs = F.softmax(next_logits / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            generated.append(idx2char.get(next_tok, '?'))
            # Extend input
            x = torch.cat([x, torch.tensor([[next_tok]], device=DEVICE)], dim=1)
            logits = model(x)

    return ''.join(generated)


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

SEQ_LEN = 128
FIELD_DIM = 128

X_train, X_test, vocab_size, idx2char = load_tinystories(max_chars=500_000, seq_len=SEQ_LEN)
char2idx = {c: i for i, c in idx2char.items()}

print()
print("=" * 70)
print(f"TRAINING: TinyStories char-level, seq_len={SEQ_LEN}, field_dim={FIELD_DIM}")
print("=" * 70)

models = {
    'Condensation LM': CondensationLM(vocab_size, field_dim=FIELD_DIM, n_substeps=2).to(DEVICE),
    'CM (no depletion)': CondensationLM_NoDepl(vocab_size, field_dim=FIELD_DIM, n_substeps=2).to(DEVICE),
    'GRU': GRULM(vocab_size, hidden=FIELD_DIM).to(DEVICE),
    'LSTM': LSTMLM(vocab_size, hidden=FIELD_DIM).to(DEVICE),
}

results = {}
for name, model in models.items():
    print(f"\n  {name} ({nparams(model):,} params):")
    t0 = time.time()
    te_loss, te_acc = train_lm(model, X_train, X_test, vocab_size, epochs=30, lr=1e-3)
    elapsed = time.time() - t0
    results[name] = {'loss': te_loss, 'acc': te_acc, 'params': nparams(model), 'time': elapsed}

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()
print(f"  {'Model':<25} {'Loss':>7}  {'Acc':>6}  {'Params':>10}  {'Time':>6}")
print(f"  {'─'*25} {'─'*7}  {'─'*6}  {'─'*10}  {'─'*6}")
for name, r in results.items():
    print(f"  {name:<25} {r['loss']:7.3f}  {r['acc']:6.3f}  {r['params']:10,}  {r['time']:5.0f}s")

# Depletion helps?
cm_loss = results['Condensation LM']['loss']
nd_loss = results['CM (no depletion)']['loss']
print()
if cm_loss < nd_loss - 0.02:
    print(f"  >> Depletion helps: CM loss {cm_loss:.3f} < no-depl {nd_loss:.3f}")
elif nd_loss < cm_loss - 0.02:
    print(f"  >> Depletion hurts: CM loss {cm_loss:.3f} > no-depl {nd_loss:.3f}")
else:
    print(f"  >> Depletion neutral: CM {cm_loss:.3f} ≈ no-depl {nd_loss:.3f}")

# Generate samples
print()
print("=" * 70)
print("GENERATED TEXT")
print("=" * 70)

for name in ['Condensation LM', 'GRU']:
    model = [m for n, m in models.items() if n == name][0]
    print(f"\n  {name}:")
    for seed in ["Once upon a ", "The little ", "She was "]:
        gen = generate(model, idx2char, seed, length=80, char2idx=char2idx)
        print(f"    '{gen}'")
