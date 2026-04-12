"""
Condensation LM v2: Closing the gap with GRU
==============================================

The linear field leaks. Two physics-native fixes:

V2a: MULTI-TIMESCALE FIELDS
  Instead of one field with decay mu, use K "species" fields with
  different decay rates. Fast fields (large mu) forget quickly.
  Slow fields (small mu) hold long-range context.
  Nucleation reads all species. Depletion affects all species.

  This is Wilson-Cowan with K layers. This is multi-species
  reaction-diffusion. This is NOT gating — the timescales are
  fixed (or learned scalars) per species, not input-dependent.

V2b: NONLINEAR REACTION (Allen-Cahn bistability)
  Replace linear `phi += dt*(mix(phi) - mu*phi)` with
  `phi += dt*(mix(phi) - mu*phi*(phi^2 - 1))`
  The cubic term creates two stable states at phi = ±1.
  The field can LATCH into either state — a physical analogue of
  a memory bit, without any gating mechanism.

V2c: BOTH
  Multi-timescale + nonlinear reaction.

Compare to v1 (linear, single field) and GRU on TinyStories.
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
# DATA (same as TinyStories test)
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════════

class CondensationLM_v1(nn.Module):
    """Baseline: linear, single-timescale field (the original v1)."""
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


class CondensationLM_v2a(nn.Module):
    """
    Multi-timescale fields.
    K species with different decay rates (learned).
    Each species has its own field dimension.
    """
    def __init__(self, vocab, field_dim_per_species=32, n_species=4, n_substeps=2, dt=0.3):
        super().__init__()
        self.vocab = vocab
        self.K = n_species
        self.fd = field_dim_per_species
        self.total_dim = n_species * field_dim_per_species
        self.n_substeps = n_substeps
        self.dt = dt

        # Each species has its own embedding contribution
        self.embed = nn.Embedding(vocab, self.total_dim)

        # Each species has its own mix matrix (block-diagonal is efficient)
        # But simpler: one shared mix on the full dim
        self.mix = nn.Linear(self.total_dim, self.total_dim, bias=False)

        # Per-species decay rates (this is what makes timescales DIFFERENT)
        # Init with a range: fast (mu=0.5) to slow (mu=0.01)
        # span from ~1 step lifetime to ~100 steps
        init_decays = torch.logspace(-2, 0, n_species)  # 0.01 to 1.0
        # Store log-decays, take softplus for positivity
        self.log_decay = nn.Parameter(torch.log(init_decays.clone()))

        self.nucleate = nn.Linear(self.total_dim, vocab)
        self.deplete = nn.Linear(vocab, self.total_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def get_decays(self):
        """Return per-species decay, broadcast to full field dim."""
        decays = F.softplus(self.log_decay)  # (K,)
        # Broadcast: each species applies its decay to its fd-dim block
        return decays.unsqueeze(-1).expand(self.K, self.fd).reshape(-1)  # (total_dim,)

    def forward(self, x, return_dynamics=False):
        batch, seq_len = x.shape
        phi = torch.zeros(batch, self.total_dim, device=x.device)
        all_logits = []
        per_species_norms = []

        for t in range(seq_len):
            phi = phi + self.embed(x[:, t])
            decays = self.get_decays()
            for _ in range(self.n_substeps):
                phi = phi + self.dt * (self.mix(phi) - decays * phi)
            logits = self.nucleate(phi)
            all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)

            if return_dynamics:
                # norm per species
                phi_r = phi.view(batch, self.K, self.fd)
                per_species_norms.append(phi_r.norm(dim=-1).mean(dim=0).detach().cpu())

        out = torch.stack(all_logits, dim=1)
        if return_dynamics:
            return out, per_species_norms
        return out


class CondensationLM_v2b(nn.Module):
    """Nonlinear field (Allen-Cahn bistability). Single timescale."""
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
                # Allen-Cahn: dphi/dt = mix(phi) - mu*phi - nl*phi*(phi^2 - 1)
                # The cubic term creates stable fixed points at phi = +/- 1
                # Clip phi to prevent explosion from the cubic term
                phi_clipped = phi.clamp(-3, 3)
                dphi = self.mix(phi) - mu * phi - nl * phi_clipped * (phi_clipped**2 - 1)
                phi = phi + self.dt * dphi
                phi = phi.clamp(-5, 5)  # safety clip
            logits = self.nucleate(phi)
            all_logits.append(logits)
            pred_soft = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(pred_soft)
        return torch.stack(all_logits, dim=1)


class CondensationLM_v2c(nn.Module):
    """Multi-timescale + nonlinear. Both improvements combined."""
    def __init__(self, vocab, field_dim_per_species=32, n_species=4, n_substeps=2, dt=0.2):
        super().__init__()
        self.vocab = vocab
        self.K = n_species
        self.fd = field_dim_per_species
        self.total_dim = n_species * field_dim_per_species
        self.n_substeps = n_substeps
        self.dt = dt

        self.embed = nn.Embedding(vocab, self.total_dim)
        self.mix = nn.Linear(self.total_dim, self.total_dim, bias=False)
        init_decays = torch.logspace(-2, 0, n_species)
        self.log_decay = nn.Parameter(torch.log(init_decays.clone()))
        self.nonlin_strength = nn.Parameter(torch.tensor(0.3))
        self.nucleate = nn.Linear(self.total_dim, vocab)
        self.deplete = nn.Linear(vocab, self.total_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def get_decays(self):
        decays = F.softplus(self.log_decay)
        return decays.unsqueeze(-1).expand(self.K, self.fd).reshape(-1)

    def forward(self, x):
        batch, seq_len = x.shape
        phi = torch.zeros(batch, self.total_dim, device=x.device)
        all_logits = []
        for t in range(seq_len):
            phi = phi + self.embed(x[:, t])
            decays = self.get_decays()
            for _ in range(self.n_substeps):
                nl = F.softplus(self.nonlin_strength)
                phi_clipped = phi.clamp(-3, 3)
                dphi = self.mix(phi) - decays * phi - nl * phi_clipped * (phi_clipped**2 - 1)
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


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def train_lm(model, X_tr, X_te, vocab, epochs=30, lr=1e-3, bs=64):
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
            opt.zero_grad()
            loss.backward()
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


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def generate(model, idx2char, char2idx, seed, length=80):
    model.eval()
    tokens = [char2idx.get(c, 0) for c in seed]
    x = torch.tensor([tokens], device=DEVICE)
    gen = list(seed)
    with torch.no_grad():
        for _ in range(length):
            logits = model(x)
            next_logits = logits[0, -1]
            probs = F.softmax(next_logits / 0.8, dim=-1)
            nt = torch.multinomial(probs, 1).item()
            gen.append(idx2char.get(nt, '?'))
            x = torch.cat([x, torch.tensor([[nt]], device=DEVICE)], dim=1)
    return ''.join(gen)


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

SEQ_LEN = 128
X_train, X_test, vocab_size, idx2char = load_tinystories(max_chars=500_000, seq_len=SEQ_LEN)
char2idx = {c: i for i, c in idx2char.items()}

print(f"Train: {X_train.shape}, Vocab: {vocab_size}")
print()

models = {
    'v1 (linear, 1-scale)':   CondensationLM_v1(vocab_size, field_dim=128).to(DEVICE),
    'v2a (multi-timescale)':  CondensationLM_v2a(vocab_size, n_species=4, field_dim_per_species=32).to(DEVICE),
    'v2b (nonlinear)':        CondensationLM_v2b(vocab_size, field_dim=128).to(DEVICE),
    'v2c (multi + nonlinear)': CondensationLM_v2c(vocab_size, n_species=4, field_dim_per_species=32).to(DEVICE),
    'GRU':                    GRULM(vocab_size, hidden=128).to(DEVICE),
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
print(f"  {'Model':<28} {'Loss':>7}  {'Acc':>6}  {'Params':>10}  {'Time':>6}")
print(f"  {'─'*28} {'─'*7}  {'─'*6}  {'─'*10}  {'─'*6}")
for name, r in results.items():
    print(f"  {name:<28} {r['loss']:7.3f}  {r['acc']:6.3f}  {r['params']:10,}  {r['time']:5.0f}s")

# Learned timescales from v2a
v2a = [m for n, m in models.items() if 'v2a' in n][0]
decays = F.softplus(v2a.log_decay).detach().cpu().numpy()
print(f"\n  v2a learned timescales (1/decay, lifetime in steps):")
for i, d in enumerate(decays):
    print(f"    species {i}: decay={d:.4f}  lifetime={1/d:.1f} steps")

# Dynamics: per-species field norms
with torch.no_grad():
    _, species_norms = v2a(X_test[:100, :-1], return_dynamics=True)

print(f"\n  v2a per-species field norms (avg over positions 0, 30, 60, 90, 127):")
print(f"  {'Pos':>4}", end='')
for i in range(v2a.K):
    print(f"  {'sp'+str(i):>6}", end='')
print()
for pos in [0, 30, 60, 90, min(len(species_norms)-1, 127)]:
    print(f"  {pos:4d}", end='')
    for i in range(v2a.K):
        print(f"  {species_norms[pos][i]:6.3f}", end='')
    print()

# Generation
print()
print("=" * 70)
print("GENERATED TEXT")
print("=" * 70)
for name in ['v1 (linear, 1-scale)', 'v2a (multi-timescale)', 'v2c (multi + nonlinear)', 'GRU']:
    model = [m for n, m in models.items() if n == name][0]
    print(f"\n  {name}:")
    for seed in ["Once upon a ", "The little "]:
        gen = generate(model, idx2char, char2idx, seed, length=80)
        print(f"    '{gen}'")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
v1_loss = results['v1 (linear, 1-scale)']['loss']
v2a_loss = results['v2a (multi-timescale)']['loss']
v2b_loss = results['v2b (nonlinear)']['loss']
v2c_loss = results['v2c (multi + nonlinear)']['loss']
gru_loss = results['GRU']['loss']

print(f"\n  v1 (baseline):      {v1_loss:.3f}")
print(f"  v2a (multi-scale):  {v2a_loss:.3f}  ({'better' if v2a_loss < v1_loss else 'worse'} than v1 by {abs(v1_loss-v2a_loss):.3f})")
print(f"  v2b (nonlinear):    {v2b_loss:.3f}  ({'better' if v2b_loss < v1_loss else 'worse'} than v1 by {abs(v1_loss-v2b_loss):.3f})")
print(f"  v2c (both):         {v2c_loss:.3f}  ({'better' if v2c_loss < v1_loss else 'worse'} than v1 by {abs(v1_loss-v2c_loss):.3f})")
print(f"  GRU (target):       {gru_loss:.3f}")

best_cm = min(v2a_loss, v2b_loss, v2c_loss)
gap_v1 = v1_loss - gru_loss
gap_best = best_cm - gru_loss
print(f"\n  Gap to GRU:  v1={gap_v1:.3f}  ->  best v2={gap_best:.3f}")
if gap_best < gap_v1 * 0.5:
    print(f"  >> Closed more than half the gap. Physics-native memory works.")
elif gap_best < gap_v1:
    print(f"  >> Partial improvement. The mechanisms help but don't fully close gap.")
else:
    print(f"  >> No improvement over v1. Need different physics.")
