"""
Enzymatic++: push input-conditioned physics parameters.

The surgical experiment showed enzymatic decay helps (+0.014 nats).
The mechanism is right: input reshapes local physics, not gates state.
Push it harder: make the input modulate MULTIPLE physics parameters.

Physical picture:
  Each token creates a local 'reaction environment':
    - Enzyme concentration (decay rate)    -> how fast things relax
    - Catalyst (nonlinear strength)        -> how strongly bistable
    - pH / local field shift               -> where the vacua sit

Token-conditioned parameters (all enzymatic):
  mu(x)     : decay rate
  nu(x)     : nonlinear (Allen-Cahn) strength
  shift(x)  : offset of the field (shifts vacuum position)

This is NOT gating — the input doesn't multiply or override the state.
It locally reshapes the potential landscape that the field evolves in.

Baselines:
  v2b               : 1.533  (no enzymes)
  v2b + enz (decay) : 1.519  (best from surgical)
Target:
  GRU               : 1.340
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
        if total > max_chars: break
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
    perm = torch.randperm(n_seqs); data = data[perm]
    n_train = int(0.9 * n_seqs)
    return data[:n_train].to(DEVICE), data[n_train:].to(DEVICE), len(char2idx), idx2char


# ─── Variants ────────────────────────────────────────────────

class v2b(nn.Module):
    """Baseline."""
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.2):
        super().__init__()
        self.vocab = vocab; self.field_dim = field_dim
        self.n_substeps = n_substeps; self.dt = dt
        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.1))
        self.nonlin_strength = nn.Parameter(torch.tensor(0.3))
        self.nucleate = nn.Linear(field_dim, vocab)
        self.deplete = nn.Linear(vocab, field_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def forward(self, x):
        B, T = x.shape
        phi = torch.zeros(B, self.field_dim, device=x.device)
        all_logits = []
        for t in range(T):
            phi = phi + self.embed(x[:, t])
            for _ in range(self.n_substeps):
                mu = F.softplus(self.decay)
                nl = F.softplus(self.nonlin_strength)
                phi_c = phi.clamp(-3, 3)
                dphi = self.mix(phi) - mu * phi - nl * phi_c * (phi_c**2 - 1)
                phi = (phi + self.dt * dphi).clamp(-5, 5)
            logits = self.nucleate(phi); all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)
        return torch.stack(all_logits, dim=1)


class v2b_enz(nn.Module):
    """Enzymatic decay only (from surgical). Current best."""
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.2):
        super().__init__()
        self.vocab = vocab; self.field_dim = field_dim
        self.n_substeps = n_substeps; self.dt = dt
        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.1))
        self.nonlin_strength = nn.Parameter(torch.tensor(0.3))
        self.enz_decay = nn.Embedding(vocab, field_dim)
        nn.init.zeros_(self.enz_decay.weight)
        self.nucleate = nn.Linear(field_dim, vocab)
        self.deplete = nn.Linear(vocab, field_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def forward(self, x):
        B, T = x.shape
        phi = torch.zeros(B, self.field_dim, device=x.device)
        all_logits = []
        for t in range(T):
            phi = phi + self.embed(x[:, t])
            decay_mod = torch.sigmoid(self.enz_decay(x[:, t]))  # (B, fd)
            for _ in range(self.n_substeps):
                mu = F.softplus(self.decay) * (0.2 + 1.6 * decay_mod)
                nl = F.softplus(self.nonlin_strength)
                phi_c = phi.clamp(-3, 3)
                dphi = self.mix(phi) - mu * phi - nl * phi_c * (phi_c**2 - 1)
                phi = (phi + self.dt * dphi).clamp(-5, 5)
            logits = self.nucleate(phi); all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)
        return torch.stack(all_logits, dim=1)


class v2b_enz_plus(nn.Module):
    """
    Enzymatic++: token modulates decay, nonlinear strength, AND vacuum shift.
    The input defines the local reaction environment.
    """
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.2):
        super().__init__()
        self.vocab = vocab; self.field_dim = field_dim
        self.n_substeps = n_substeps; self.dt = dt
        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.1))
        self.nonlin_strength = nn.Parameter(torch.tensor(0.3))

        # Three enzymatic embeddings (same vocab, different meanings)
        self.enz_decay = nn.Embedding(vocab, field_dim)   # modulates mu
        self.enz_nonlin = nn.Embedding(vocab, field_dim)  # modulates nu (per-dim)
        self.enz_shift = nn.Embedding(vocab, field_dim)   # shifts vacuum position

        nn.init.zeros_(self.enz_decay.weight)
        nn.init.zeros_(self.enz_nonlin.weight)
        nn.init.normal_(self.enz_shift.weight, std=0.05)  # small init shifts

        self.nucleate = nn.Linear(field_dim, vocab)
        self.deplete = nn.Linear(vocab, field_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def forward(self, x):
        B, T = x.shape
        phi = torch.zeros(B, self.field_dim, device=x.device)
        all_logits = []
        for t in range(T):
            phi = phi + self.embed(x[:, t])

            # Token-conditioned physics parameters
            decay_mod = torch.sigmoid(self.enz_decay(x[:, t]))    # (B, fd), in (0,1)
            nonlin_mod = torch.sigmoid(self.enz_nonlin(x[:, t]))  # (B, fd), in (0,1)
            shift = self.enz_shift(x[:, t])                       # (B, fd), in R

            for _ in range(self.n_substeps):
                mu = F.softplus(self.decay) * (0.2 + 1.6 * decay_mod)
                nl = F.softplus(self.nonlin_strength) * (0.2 + 1.6 * nonlin_mod)

                # Field centered at the token-conditioned vacuum
                phi_centered = phi - shift
                phi_c = phi_centered.clamp(-3, 3)

                dphi = self.mix(phi) - mu * phi - nl * phi_c * (phi_c**2 - 1)
                phi = (phi + self.dt * dphi).clamp(-5, 5)

            logits = self.nucleate(phi); all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)
        return torch.stack(all_logits, dim=1)


class v2b_enz_full(nn.Module):
    """
    Full enzymatic: also low-rank token modulation of the mix matrix.
    delta_mix(phi) = u(x) (v(x)^T phi)
    """
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.2, rank=4):
        super().__init__()
        self.vocab = vocab; self.field_dim = field_dim
        self.n_substeps = n_substeps; self.dt = dt
        self.rank = rank
        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.1))
        self.nonlin_strength = nn.Parameter(torch.tensor(0.3))

        self.enz_decay = nn.Embedding(vocab, field_dim)
        self.enz_nonlin = nn.Embedding(vocab, field_dim)
        self.enz_shift = nn.Embedding(vocab, field_dim)
        # Low-rank mix modulation: (rank, field_dim) readers and writers
        self.enz_mix_u = nn.Embedding(vocab, rank * field_dim)  # (B, r*fd)
        self.enz_mix_v = nn.Embedding(vocab, rank * field_dim)

        nn.init.zeros_(self.enz_decay.weight)
        nn.init.zeros_(self.enz_nonlin.weight)
        nn.init.normal_(self.enz_shift.weight, std=0.05)
        nn.init.normal_(self.enz_mix_u.weight, std=0.01)
        nn.init.normal_(self.enz_mix_v.weight, std=0.01)

        self.nucleate = nn.Linear(field_dim, vocab)
        self.deplete = nn.Linear(vocab, field_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def forward(self, x):
        B, T = x.shape
        phi = torch.zeros(B, self.field_dim, device=x.device)
        all_logits = []
        for t in range(T):
            phi = phi + self.embed(x[:, t])
            decay_mod = torch.sigmoid(self.enz_decay(x[:, t]))
            nonlin_mod = torch.sigmoid(self.enz_nonlin(x[:, t]))
            shift = self.enz_shift(x[:, t])
            u = self.enz_mix_u(x[:, t]).view(B, self.rank, self.field_dim)
            v = self.enz_mix_v(x[:, t]).view(B, self.rank, self.field_dim)

            for _ in range(self.n_substeps):
                mu = F.softplus(self.decay) * (0.2 + 1.6 * decay_mod)
                nl = F.softplus(self.nonlin_strength) * (0.2 + 1.6 * nonlin_mod)
                phi_centered = phi - shift
                phi_c = phi_centered.clamp(-3, 3)

                # Low-rank additive mix modulation
                # For each sample: scores = v @ phi (B,r), delta = u.T @ scores (B, fd)
                scores = torch.einsum('brf,bf->br', v, phi)  # (B, r)
                delta_mix = torch.einsum('brf,br->bf', u, scores)  # (B, fd)

                dphi = self.mix(phi) + delta_mix - mu * phi - nl * phi_c * (phi_c**2 - 1)
                phi = (phi + self.dt * dphi).clamp(-5, 5)

            logits = self.nucleate(phi); all_logits.append(logits)
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

variants = {
    'v2b (baseline)':       v2b,
    'v2b + enz (decay)':    v2b_enz,
    'v2b + enz++ (3 params)': v2b_enz_plus,
    'v2b + enz full (+mix)':  v2b_enz_full,
}

results = {}
for name, cls in variants.items():
    print()
    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)
    torch.manual_seed(42)
    m = cls(vocab_size, field_dim=128).to(DEVICE)
    p = nparams(m)
    print(f"    params: {p:,}")
    t0 = time.time()
    loss, acc = train_lm(m, X_train, X_test, vocab_size, epochs=20)
    elapsed = time.time() - t0
    results[name] = {'loss': loss, 'acc': acc, 'params': p, 'time': elapsed}

# GRU reference
print()
print("=" * 70)
print("  GRU reference")
print("=" * 70)
torch.manual_seed(42)
m_gru = GRULM(vocab_size, hidden=128).to(DEVICE)
p_gru = nparams(m_gru)
print(f"    params: {p_gru:,}")
gru_loss, gru_acc = train_lm(m_gru, X_train, X_test, vocab_size, epochs=20)
results['GRU'] = {'loss': gru_loss, 'acc': gru_acc, 'params': p_gru, 'time': 0}

# ─── ANALYSIS ───────────────────────────────────────────────
print()
print("=" * 70)
print("ENZYMATIC++ ANALYSIS")
print("=" * 70)
print()
baseline_loss = results['v2b (baseline)']['loss']
gru_loss_v = results['GRU']['loss']
total_gap = baseline_loss - gru_loss_v

print(f"  {'Variant':<28} {'Loss':>6}  {'Acc':>6}  {'Params':>8}  {'% gap closed':>14}")
print(f"  {'─'*28} {'─'*6}  {'─'*6}  {'─'*8}  {'─'*14}")
for name, r in results.items():
    if r['loss'] < baseline_loss:
        pct_closed = (baseline_loss - r['loss']) / total_gap * 100
        pct_str = f"{pct_closed:.0f}%"
    else:
        pct_str = "—"
    marker = ''
    if name == 'v2b (baseline)': marker = '← baseline'
    elif name == 'GRU': marker = '← target'
    print(f"  {name:<28} {r['loss']:6.3f}  {r['acc']:6.3f}  {r['params']:8,}  {pct_str:>14}  {marker}")

# Best variant
best = min([(n, r['loss']) for n, r in results.items() if n not in ('GRU',)], key=lambda x: x[1])
print()
print(f"  Best physics-native: {best[0]}  loss={best[1]:.3f}")
print(f"  Gap to GRU: {best[1] - gru_loss_v:.3f}  ({(best[1] - gru_loss_v) / total_gap * 100:.0f}% of original)")
