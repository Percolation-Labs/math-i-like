"""
C1 Scaling Experiment: Does v2b close the gap with GRU at larger scale?

Keep v2b exactly as-is (the cleanest working physics-native architecture).
Scale field_dim: 64, 128, 256.
Compare to GRU at matched hidden dim.
Measure the loss gap trend.

The conjecture C1:
  gap(k) = L_CLM(k) - L_GRU(k) ~ O(k^(-alpha))  with alpha > 0

If true: the gap narrows with scale. Physics-native memory is viable.
If false: the gap is constant/growing. Gating is fundamentally more expressive.
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


# ────────────────────────────────────────────────────────────
# DATA
# ────────────────────────────────────────────────────────────

def load_tinystories(max_chars=500_000, seq_len=128):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    text = []
    total = 0
    for example in ds:
        text.append(example['text'])
        total += len(example['text'])
        if total > max_chars:
            break
    corpus = "\n".join(text)[:max_chars]
    char_counts = {}
    for c in corpus:
        char_counts[c] = char_counts.get(c, 0) + 1
    common = sorted([c for c, cnt in char_counts.items() if cnt > 50])
    char2idx = {c: i+1 for i, c in enumerate(common)}
    char2idx['<unk>'] = 0
    idx2char = {i: c for c, i in char2idx.items()}
    encoded = torch.tensor([char2idx.get(c, 0) for c in corpus], dtype=torch.long)
    n_seqs = len(encoded) // seq_len
    data = encoded[:n_seqs * seq_len].view(n_seqs, seq_len)
    perm = torch.randperm(n_seqs)
    data = data[perm]
    n_train = int(0.9 * n_seqs)
    return data[:n_train].to(DEVICE), data[n_train:].to(DEVICE), len(char2idx), idx2char


# ────────────────────────────────────────────────────────────
# v2b (unchanged from previous best)
# ────────────────────────────────────────────────────────────

class CondensationLM_v2b(nn.Module):
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.2):
        super().__init__()
        self.vocab = vocab
        self.field_dim = field_dim
        self.n_substeps = n_substeps
        self.dt = dt
        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.1))
        self.nonlin_strength = nn.Parameter(torch.tensor(0.3))
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
                mu = F.softplus(self.decay)
                nl = F.softplus(self.nonlin_strength)
                phi_c = phi.clamp(-3, 3)
                dphi = self.mix(phi) - mu * phi - nl * phi_c * (phi_c**2 - 1)
                phi = phi + self.dt * dphi
                phi = phi.clamp(-5, 5)
            logits = self.nucleate(phi)
            all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)
        return torch.stack(all_logits, dim=1)


class GRULM(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
    def forward(self, x):
        return self.head(self.gru(self.embed(x))[0])


def train_lm(model, X_tr, X_te, vocab, epochs=20, lr=1e-3, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_tr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0; nb = 0
        for s in range(0, min(n, 2000), bs):
            idx = perm[s:s+bs]
            xb = X_tr[idx]
            logits = model(xb[:, :-1])
            targets = xb[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            te_logits = model(X_te[:200, :-1])
            te_targets = X_te[:200, 1:]
            te_loss = F.cross_entropy(te_logits.reshape(-1, vocab), te_targets.reshape(-1)).item()
            te_acc = (te_logits.argmax(-1) == te_targets).float().mean().item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    ep {ep:3d}  train_loss={total_loss/nb:.3f}  test_loss={te_loss:.3f}  test_acc={te_acc:.3f}")
    return te_loss, te_acc


def nparams(m): return sum(p.numel() for p in m.parameters())


# ────────────────────────────────────────────────────────────
# SCALING SWEEP
# ────────────────────────────────────────────────────────────

X_train, X_test, vocab_size, idx2char = load_tinystories(max_chars=500_000, seq_len=128)
print(f"Train: {X_train.shape}, Vocab: {vocab_size}\n")

# Scales to test
scales = [64, 128, 256]

results = []  # list of dicts

for dim in scales:
    print()
    print("=" * 70)
    print(f"SCALE: dim = {dim}")
    print("=" * 70)

    # v2b
    print(f"\n  v2b (field_dim={dim}):")
    torch.manual_seed(42)
    m_cm = CondensationLM_v2b(vocab_size, field_dim=dim, n_substeps=2).to(DEVICE)
    p_cm = nparams(m_cm)
    print(f"    params: {p_cm:,}")
    t0 = time.time()
    cm_loss, cm_acc = train_lm(m_cm, X_train, X_test, vocab_size, epochs=20)
    cm_time = time.time() - t0

    # GRU
    print(f"\n  GRU (hidden={dim}):")
    torch.manual_seed(42)
    m_gru = GRULM(vocab_size, hidden=dim).to(DEVICE)
    p_gru = nparams(m_gru)
    print(f"    params: {p_gru:,}")
    t0 = time.time()
    gru_loss, gru_acc = train_lm(m_gru, X_train, X_test, vocab_size, epochs=20)
    gru_time = time.time() - t0

    gap = cm_loss - gru_loss
    results.append({
        'dim': dim, 'cm_loss': cm_loss, 'cm_acc': cm_acc, 'cm_params': p_cm, 'cm_time': cm_time,
        'gru_loss': gru_loss, 'gru_acc': gru_acc, 'gru_params': p_gru, 'gru_time': gru_time,
        'gap': gap
    })

    print(f"\n  SUMMARY at dim={dim}:")
    print(f"    v2b:  loss={cm_loss:.3f}  acc={cm_acc:.3f}  params={p_cm:,}")
    print(f"    GRU:  loss={gru_loss:.3f}  acc={gru_acc:.3f}  params={p_gru:,}")
    print(f"    Gap:  {gap:.3f}")


# ────────────────────────────────────────────────────────────
# SCALING ANALYSIS
# ────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("C1 SCALING ANALYSIS")
print("=" * 70)
print()
print(f"  {'dim':>4}  {'v2b loss':>10}  {'GRU loss':>10}  {'Gap':>7}  {'v2b params':>12}  {'GRU params':>12}")
print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*12}  {'─'*12}")
for r in results:
    print(f"  {r['dim']:4d}  {r['cm_loss']:10.3f}  {r['gru_loss']:10.3f}  {r['gap']:7.3f}  "
          f"{r['cm_params']:12,}  {r['gru_params']:12,}")

# Is the gap narrowing?
gaps = [r['gap'] for r in results]
dims = [r['dim'] for r in results]

print()
print(f"  Gap trajectory:  dim={dims[0]} -> gap={gaps[0]:.3f}")
for i in range(1, len(results)):
    delta = gaps[i] - gaps[i-1]
    arrow = "↓" if delta < -0.01 else ("↑" if delta > 0.01 else "→")
    print(f"                   dim={dims[i]} -> gap={gaps[i]:.3f}  ({arrow} {delta:+.3f})")

# Log-log fit: log(gap) vs log(dim)
if all(g > 0 for g in gaps):
    log_dims = np.log(dims)
    log_gaps = np.log(gaps)
    slope = np.polyfit(log_dims, log_gaps, 1)[0]
    print(f"\n  Log-log slope (alpha in gap ~ dim^-alpha): {-slope:.3f}")
    print(f"  C1 conjecture: alpha > 0 (gap narrows with scale)")
    if -slope > 0.1:
        print(f"  >> C1 SUPPORTED: gap narrows at rate ~ dim^{-abs(slope):.2f}")
    elif -slope > 0:
        print(f"  >> C1 WEAKLY supported: gap narrows slowly")
    else:
        print(f"  >> C1 FALSIFIED: gap grows with scale")
else:
    # Some gap is negative (v2b beat GRU!)
    if any(g < 0 for g in gaps):
        print(f"\n  >> v2b matched or beat GRU at some scale — C1 strongly supported")

print()
print("  Note: this is 20 epochs per model. Longer training may shift the gap.")
print("  The trend direction is what matters for C1, not absolute values.")
