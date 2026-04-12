"""
Surgical additions to v2b. One change at a time.

v3 taught us: piling on physics mechanisms doesn't help.
Find the SINGLE missing thing that closes the most gap.

Variants tested (all at dim=128, 20 epochs):
  v2b-LN:       v2b + LayerNorm after each substep
  v2b-res:      v2b + residual connections (explicit skip)
  v2b-enz:      v2b + input-conditioned decay rate
                (analog of GRU's reset gate; physics: reaction rates
                 depend on local conditions, like enzymes)
  v2b-MLP:      v2b + small MLP reaction term (vs fixed cubic)
  v2b-substeps: v2b with more substeps (6 vs 2)

Baseline: v2b = 1.536 loss, GRU = 1.340 loss. Gap = 0.196.
Goal: identify which surgical change closes the most gap.
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


# ─── Baseline: v2b ─────────────────────────────────────────

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


# ─── Surgical variants ─────────────────────────────────────

class CondensationLM_v2b_LN(nn.Module):
    """v2b + LayerNorm after each substep. Nothing else changed."""
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
        self.ln = nn.LayerNorm(field_dim)   # NEW
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
                phi = self.ln(phi)   # NEW: LayerNorm after each substep
            logits = self.nucleate(phi)
            all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)
        return torch.stack(all_logits, dim=1)


class CondensationLM_v2b_enz(nn.Module):
    """
    v2b + input-conditioned decay rate.
    The MOST RECENT TOKEN modulates local decay rates.
    Physics: enzyme concentration modulates reaction rates.
    No multiplicative gates on state — but input tunes dynamics.
    """
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
        # NEW: token -> per-dim decay modulation
        self.enzyme = nn.Embedding(vocab, field_dim)
        nn.init.zeros_(self.enzyme.weight)  # start with no modulation
        self.nucleate = nn.Linear(field_dim, vocab)
        self.deplete = nn.Linear(vocab, field_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def forward(self, x):
        batch, seq_len = x.shape
        phi = torch.zeros(batch, self.field_dim, device=x.device)
        all_logits = []
        for t in range(seq_len):
            phi = phi + self.embed(x[:, t])
            # Token-conditioned decay
            decay_mod = torch.sigmoid(self.enzyme(x[:, t]))  # (B, field_dim), in (0, 1)
            for _ in range(self.n_substeps):
                mu_base = F.softplus(self.decay)
                mu_effective = mu_base * (0.2 + 1.6 * decay_mod)  # varies 0.2x..1.8x
                nl = F.softplus(self.nonlin_strength)
                phi_c = phi.clamp(-3, 3)
                dphi = self.mix(phi) - mu_effective * phi - nl * phi_c * (phi_c**2 - 1)
                phi = phi + self.dt * dphi
                phi = phi.clamp(-5, 5)
            logits = self.nucleate(phi)
            all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)
        return torch.stack(all_logits, dim=1)


class CondensationLM_v2b_MLP(nn.Module):
    """v2b with a small MLP reaction term instead of a fixed cubic."""
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.2):
        super().__init__()
        self.vocab = vocab
        self.field_dim = field_dim
        self.n_substeps = n_substeps
        self.dt = dt
        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.1))
        # NEW: small MLP reaction (two-layer with tanh)
        # Forces polynomial nonlinearity but free to be any shape
        self.reaction = nn.Sequential(
            nn.Linear(field_dim, field_dim // 2),
            nn.Tanh(),
            nn.Linear(field_dim // 2, field_dim, bias=False),
        )
        # Init reaction output to ~zero so we start like v1 (linear)
        nn.init.normal_(self.reaction[2].weight, std=0.01)
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
                phi_c = phi.clamp(-3, 3)
                reaction = self.reaction(phi_c)
                dphi = self.mix(phi) - mu * phi + reaction
                phi = phi + self.dt * dphi
                phi = phi.clamp(-5, 5)
            logits = self.nucleate(phi)
            all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)
        return torch.stack(all_logits, dim=1)


class CondensationLM_v2b_substeps(nn.Module):
    """v2b with 6 substeps instead of 2. Let the dynamics settle longer."""
    def __init__(self, vocab, field_dim=128, n_substeps=6, dt=0.1):
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


# ─── RUN ────────────────────────────────────────────────────

X_train, X_test, vocab_size, idx2char = load_tinystories(max_chars=500_000, seq_len=128)
print(f"Train: {X_train.shape}, Vocab: {vocab_size}\n")

DIM = 128
EPOCHS = 20

variants = {
    'v2b (baseline)':          CondensationLM_v2b,
    'v2b + LayerNorm':         CondensationLM_v2b_LN,
    'v2b + enzymatic decay':   CondensationLM_v2b_enz,
    'v2b + MLP reaction':      CondensationLM_v2b_MLP,
    'v2b + 6 substeps':        CondensationLM_v2b_substeps,
}

results = {}
for name, cls in variants.items():
    print()
    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)
    torch.manual_seed(42)
    m = cls(vocab_size, field_dim=DIM).to(DEVICE)
    p = nparams(m)
    print(f"    params: {p:,}")
    t0 = time.time()
    loss, acc = train_lm(m, X_train, X_test, vocab_size, epochs=EPOCHS)
    elapsed = time.time() - t0
    results[name] = {'loss': loss, 'acc': acc, 'params': p, 'time': elapsed}

# GRU reference
print()
print("=" * 70)
print("  GRU reference")
print("=" * 70)
torch.manual_seed(42)
m_gru = GRULM(vocab_size, hidden=DIM).to(DEVICE)
p_gru = nparams(m_gru)
print(f"    params: {p_gru:,}")
t0 = time.time()
gru_loss, gru_acc = train_lm(m_gru, X_train, X_test, vocab_size, epochs=EPOCHS)
results['GRU'] = {'loss': gru_loss, 'acc': gru_acc, 'params': p_gru, 'time': time.time() - t0}

# ─── ANALYSIS ───────────────────────────────────────────────
print()
print("=" * 70)
print("SURGICAL ANALYSIS")
print("=" * 70)
print()
print(f"  {'Variant':<28} {'Loss':>6}  {'Acc':>6}  {'Params':>8}  {'Gap':>6}")
print(f"  {'─'*28} {'─'*6}  {'─'*6}  {'─'*8}  {'─'*6}")

baseline = results['v2b (baseline)']['loss']
for name, r in results.items():
    gap_from_baseline = r['loss'] - baseline
    gap_from_gru = r['loss'] - gru_loss
    marker = ''
    if name == 'v2b (baseline)':
        marker = '← baseline'
    elif name == 'GRU':
        marker = '← target'
    elif gap_from_baseline < -0.03:
        marker = '← IMPROVED'
    print(f"  {name:<28} {r['loss']:6.3f}  {r['acc']:6.3f}  {r['params']:8,}  {gap_from_baseline:+6.3f}  {marker}")

# Which variant closed the most gap to GRU?
best_variant = None
best_improvement = 0
for name, r in results.items():
    if name in ['v2b (baseline)', 'GRU']:
        continue
    improvement = baseline - r['loss']
    if improvement > best_improvement:
        best_improvement = improvement
        best_variant = name

print()
if best_variant:
    r = results[best_variant]
    total_gap = baseline - gru_loss
    closed_pct = best_improvement / total_gap * 100 if total_gap > 0 else 0
    print(f"  Best single addition: {best_variant}")
    print(f"    Improved by {best_improvement:.3f} nats ({closed_pct:.0f}% of gap to GRU)")
    print(f"    Extra params: {results[best_variant]['params'] - baseline_params:,}" if False else f"    Extra params: {results[best_variant]['params'] - results['v2b (baseline)']['params']:,}")
else:
    print("  No variant improved meaningfully over baseline.")

print()
print("  This tells us which ONE physics-native change is doing the work.")
print("  If LN: normalization is essential (engineering).")
print("  If enzymatic: input-conditioned reactions are essential (physics).")
print("  If MLP: the fixed cubic is too simple; generic polynomials help.")
print("  If substeps: the dynamics need more time to settle.")
