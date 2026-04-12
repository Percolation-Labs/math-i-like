"""
Condensation LM v3: Physics-Native, Engineering-Modern
=========================================================

The tension:
  Modern architectures (transformers) are in a local minimum.
  They work brilliantly but are not on the path to biological elegance.
  To escape, we must be willing to take apparent regressions guided by
  physical principles that have no transformer analogue.

What to IMPORT from 30 years of deep learning:
  ✓ LayerNorm (= gain control, biological)
  ✓ Residual updates (= skip pathways, biological)
  ✓ Orthogonal init (prevents init explosion)
  ✓ AdamW + cosine schedule (just training)
  ✓ Longer dynamics / deeper time (= more brain-like, not less)
  ✓ Sparse activation (= how real neurons fire)

What to REFUSE (transformer-local-minimum machinery):
  ✗ Attention-over-values (violates locality)
  ✗ Softmax everything (unprincipled as physics)
  ✗ Multiplicative gates (no physical analogue)

What physics adds that modern AI lacks:
  • LOCALITY: coupling is sparse, not global
  • CONSERVATION: something is preserved (particles + field budget)
  • SPONTANEOUS SYMMETRY BREAKING: memory as vacuum structure (Allen-Cahn)
  • RG-LIKE HIERARCHY: fast scales inform slow scales
  • MULTI-VACUUM FIELDS: field can latch into multiple stable states,
    not just two (generalized Landau polynomial)

v3 architecture:
  - Field with RESIDUAL + LAYERNORM updates (modern engineering)
  - Local topology (sparse attention, only k neighbours)
  - Ginzburg-Landau reaction: phi + higher-order polynomial
    (multi-vacuum instead of bistable)
  - Multi-species field with CROSS-COUPLING between scales
    (RG-like: fast field modulates slow field's decay)
  - SPARSE particle activation (top-k firing)
  - Deeper dynamics (more substeps, let it settle)

We test at the same 39K scale as v2b and a doubled scale (128K)
to start measuring the C1 scaling conjecture.
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
# DATA
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
# V3 ARCHITECTURE
# ════════════════════════════════════════════════════════════════

class GinzburgLandauReaction(nn.Module):
    """
    Generalised Ginzburg-Landau polynomial: multiple vacua.
    V(phi) = sum_k a_k * phi^{2k}  with alternating signs.

    Derivative (force on field): dV/dphi = sum_k 2k * a_k * phi^{2k-1}

    With learned coefficients that can create:
      - 2 vacua (Allen-Cahn, phi^4 theory):  a_2 * phi^3 term
      - 3 vacua (phi^6 theory):              +a_3 * phi^5 term
      - 4+ vacua (higher polynomials)

    This is the physics origin of memory: multiple stable field values.
    """
    def __init__(self, field_dim, max_order=3):
        super().__init__()
        self.max_order = max_order
        # Learned coefficients for phi^(2k+1) terms in the force
        # Start with Allen-Cahn (phi^3 term dominant)
        coeffs = torch.zeros(max_order)
        coeffs[0] = 0.3  # phi^3 term (Allen-Cahn)
        self.log_coeffs = nn.Parameter(torch.log(F.softplus(coeffs) + 0.01))

    def forward(self, phi):
        """Returns -dV/dphi (the restoring force)."""
        coeffs = F.softplus(self.log_coeffs)  # positive
        # Force = -dV/dphi, with alternating signs to create multi-vacuum
        # F(phi) = -a_1 * phi * (phi^2 - 1) + a_2 * phi^3 * (phi^2 - ...) - ...
        # Simplest: Ginzburg-Landau F(phi) = phi - phi^3 + phi^5/5 - ...
        # We use: F = sum_k (-1)^k * a_k * phi^(2k+1)
        # Clipped to prevent explosion
        phi_c = phi.clamp(-3, 3)
        force = torch.zeros_like(phi)
        for k in range(self.max_order):
            sign = -1 if k % 2 == 0 else 1  # alternating for multi-vacuum
            force = force + sign * coeffs[k] * phi_c ** (2*k + 1)
        return force


class LocalTopology(nn.Module):
    """
    SPARSE LOCAL topology. Not global attention.
    Each channel couples only to its k nearest neighbours (in channel space).
    This is locality as a hard constraint.
    """
    def __init__(self, n_channels, k_neighbours=3):
        super().__init__()
        self.n = n_channels
        self.k = k_neighbours
        # Learned coupling to each of the k nearest neighbours (on left and right)
        # Init as weak local coupling
        self.coupling = nn.Parameter(torch.randn(n_channels, 2*k_neighbours) * 0.1)

    def forward(self, phi_per_channel):
        """
        phi_per_channel: (batch, n_channels, species_dim)
        Returns: topological diffusion term (batch, n_channels, species_dim)
        """
        B, N, D = phi_per_channel.shape
        out = torch.zeros_like(phi_per_channel)
        for offset in range(1, self.k + 1):
            # Right neighbour coupling
            right = torch.roll(phi_per_channel, -offset, dims=1)
            w_r = self.coupling[:, offset - 1].view(1, N, 1)
            out = out + w_r * (right - phi_per_channel)
            # Left neighbour coupling
            left = torch.roll(phi_per_channel, offset, dims=1)
            w_l = self.coupling[:, self.k + offset - 1].view(1, N, 1)
            out = out + w_l * (left - phi_per_channel)
        return out


class CondensationLM_v3(nn.Module):
    """
    Physics-native, engineering-modern condensation language model.

    Key architectural choices:
      1. Multi-species field (K species × D dims each) for RG hierarchy
      2. LOCAL topology between species (sparse, not global attention)
      3. GINZBURG-LANDAU reaction (multiple vacua, not just bistable)
      4. LayerNorm + residual (modern engineering, biologically plausible)
      5. Sparse particle activation (top-k, like real neural firing)
      6. Cross-scale coupling: fast species modulates slow species' decay
    """
    def __init__(self, vocab, n_species=4, species_dim=32, n_substeps=3, dt=0.15,
                 top_k_particles=None, k_neighbours=2):
        super().__init__()
        self.vocab = vocab
        self.K = n_species
        self.D = species_dim
        self.total = n_species * species_dim
        self.n_substeps = n_substeps
        self.dt = dt
        self.top_k = top_k_particles if top_k_particles else vocab

        # Token -> field deposition (one projection per species allows specialisation)
        self.embed = nn.Embedding(vocab, self.total)

        # Per-species decay rates (RG-like hierarchy: logspace timescales)
        init_decays = torch.logspace(-1.5, 0, n_species)
        self.log_decay = nn.Parameter(torch.log(init_decays))

        # LOCAL mixing within each species (not global mix matrix!)
        # Each species has a compact mix matrix
        self.species_mix = nn.ModuleList([
            nn.Linear(species_dim, species_dim, bias=False) for _ in range(n_species)
        ])
        # Orthogonal init for stability
        for m in self.species_mix:
            nn.init.orthogonal_(m.weight)

        # CROSS-SCALE coupling (RG-like): fast species modulates slow species
        self.cross_scale = nn.Parameter(torch.randn(n_species, n_species) * 0.05)

        # LOCAL topology (sparse coupling between species)
        self.local_topo = LocalTopology(n_species, k_neighbours)

        # Ginzburg-Landau reaction (per-species)
        self.reactions = nn.ModuleList([
            GinzburgLandauReaction(species_dim, max_order=2) for _ in range(n_species)
        ])

        # LayerNorm per species (biological gain control)
        self.field_norm = nn.ModuleList([
            nn.LayerNorm(species_dim) for _ in range(n_species)
        ])

        # Nucleation: field -> next token
        self.nucleate = nn.Linear(self.total, vocab)

        # Depletion (sparse: only top-k particles deplete)
        self.deplete = nn.Linear(vocab, self.total, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.01)

    def field_step(self, phi):
        """
        phi: (batch, K, D) — field split by species
        One substep of coupled multi-species dynamics.
        """
        B, K, D = phi.shape

        # Per-species decay (RG-like hierarchy)
        decays = F.softplus(self.log_decay)  # (K,)

        # Cross-scale modulation: species j's activity modulates species i's effective decay
        phi_norms = phi.norm(dim=2, keepdim=False) / np.sqrt(D)  # (B, K)
        # Each species i gets a modulation from other species
        mod = torch.sigmoid(phi_norms @ self.cross_scale.T)  # (B, K)
        effective_decays = decays.unsqueeze(0) * (0.5 + mod)  # (B, K)

        # Within-species mixing (per-species linear)
        mixed = torch.zeros_like(phi)
        for k in range(K):
            mixed[:, k, :] = self.species_mix[k](phi[:, k, :])

        # Nonlinear reaction (per-species Ginzburg-Landau)
        reaction = torch.zeros_like(phi)
        for k in range(K):
            reaction[:, k, :] = self.reactions[k](phi[:, k, :])

        # Topological coupling between species (local, sparse)
        topo_coupling = self.local_topo(phi)

        # Update: dphi/dt = mix + reaction + topo - decay*phi
        dphi = mixed + reaction + topo_coupling - effective_decays.unsqueeze(-1) * phi

        # Residual update with LayerNorm
        phi_new = phi + self.dt * dphi
        # Per-species LayerNorm
        phi_normed = torch.zeros_like(phi_new)
        for k in range(K):
            phi_normed[:, k, :] = self.field_norm[k](phi_new[:, k, :])

        return phi_normed

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        B, T = x.shape
        phi = torch.zeros(B, self.K, self.D, device=x.device)
        all_logits = []

        for t in range(T):
            # Deposit
            deposit = self.embed(x[:, t]).view(B, self.K, self.D)
            phi = phi + deposit

            # Evolve (multiple substeps to let dynamics settle)
            for _ in range(self.n_substeps):
                phi = self.field_step(phi)

            # Nucleate from flat field
            phi_flat = phi.view(B, self.total)
            logits = self.nucleate(phi_flat)
            all_logits.append(logits)

            # SPARSE depletion: only top-k tokens deplete the field
            pred_soft = F.softmax(logits.detach(), dim=-1)
            if self.top_k < self.vocab:
                # Zero out all but top-k
                top_vals, top_idx = pred_soft.topk(self.top_k, dim=-1)
                sparse_pred = torch.zeros_like(pred_soft)
                sparse_pred.scatter_(1, top_idx, top_vals)
                # Renormalize
                sparse_pred = sparse_pred / (sparse_pred.sum(dim=-1, keepdim=True) + 1e-8)
                consumption = self.deplete(sparse_pred).view(B, self.K, self.D)
            else:
                consumption = self.deplete(pred_soft).view(B, self.K, self.D)
            phi = phi - consumption

        return torch.stack(all_logits, dim=1)


# ════════════════════════════════════════════════════════════════
# BASELINES (from v2b and GRU)
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# TRAINING (with cosine schedule + warmup)
# ════════════════════════════════════════════════════════════════

def train_lm(model, X_tr, X_te, vocab, epochs=30, lr=1e-3, bs=64, warmup=3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Cosine schedule with warmup
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

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
        scheduler.step()

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
print(f"Train: {X_train.shape}, Vocab: {vocab_size}\n")

# Match roughly 39K params like v2b
# v3 with 4 species, 32 dim each = 128 total
# Plus extra params for per-species mix, GL reactions, cross-scale, local topo
models = {
    'v2b (previous best)': CondensationLM_v2b(vocab_size, field_dim=128).to(DEVICE),
    'v3 (physics-modern)': CondensationLM_v3(
        vocab_size, n_species=4, species_dim=32, n_substeps=3,
        top_k_particles=20, k_neighbours=2
    ).to(DEVICE),
    'GRU': GRULM(vocab_size, hidden=128).to(DEVICE),
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
print(f"  {'Model':<28} {'Loss':>7}  {'Acc':>6}  {'Params':>8}  {'Time':>6}")
print(f"  {'─'*28} {'─'*7}  {'─'*6}  {'─'*8}  {'─'*6}")
for name, r in results.items():
    print(f"  {name:<28} {r['loss']:7.3f}  {r['acc']:6.3f}  {r['params']:8,}  {r['time']:5.0f}s")

# Analysis: what did v3 learn?
v3 = models['v3 (physics-modern)']
print()
print("=" * 70)
print("WHAT v3 LEARNED")
print("=" * 70)
print()

decays = F.softplus(v3.log_decay).detach().cpu().numpy()
print("  Per-species decay rates (timescale hierarchy):")
for i, d in enumerate(decays):
    print(f"    species {i}: decay={d:.4f}  lifetime={1/d:.1f} steps")

print()
print("  Cross-scale coupling matrix (fast species -> slow species modulation):")
cs = torch.sigmoid(v3.cross_scale).detach().cpu().numpy()
print(f"    diagonal: {np.diag(cs)}")
print(f"    off-diag mean: {(cs.sum() - cs.trace()) / (v3.K * (v3.K - 1)):.3f}")

print()
print("  Ginzburg-Landau reaction coefficients (vacuum structure):")
for k in range(v3.K):
    coeffs = F.softplus(v3.reactions[k].log_coeffs).detach().cpu().numpy()
    print(f"    species {k}: phi^3 coef={coeffs[0]:.3f}, phi^5 coef={coeffs[1]:.3f}")

print()
print("  Local topology (sparse coupling strength):")
topo = v3.local_topo.coupling.data.abs().mean(dim=0).cpu().numpy()
print(f"    mean |coupling|: {topo}")

# Generation
print()
print("=" * 70)
print("GENERATED TEXT")
print("=" * 70)
for name in ['v2b (previous best)', 'v3 (physics-modern)', 'GRU']:
    model = models[name]
    print(f"\n  {name}:")
    for seed in ["Once upon a ", "The little "]:
        gen = generate(model, idx2char, char2idx, seed, length=80)
        print(f"    '{gen}'")

print()
print("=" * 70)
print("SUMMARY vs CONJECTURE C1 (scaling)")
print("=" * 70)
v2b_loss = results['v2b (previous best)']['loss']
v3_loss = results['v3 (physics-modern)']['loss']
gru_loss = results['GRU']['loss']

gap_v2b = v2b_loss - gru_loss
gap_v3 = v3_loss - gru_loss

print(f"\n  v2b:       loss={v2b_loss:.3f}  gap to GRU = {gap_v2b:.3f}")
print(f"  v3:        loss={v3_loss:.3f}  gap to GRU = {gap_v3:.3f}")
print(f"  GRU:       loss={gru_loss:.3f}")
print()
if v3_loss < v2b_loss - 0.01:
    improvement = (v2b_loss - v3_loss) / (v2b_loss - gru_loss) * 100
    print(f"  >> v3 improves on v2b. Closed additional {improvement:.0f}% of gap to GRU.")
if v3_loss < gru_loss + 0.05:
    print(f"  >> v3 is within 0.05 of GRU. Physics-native approach is viable at this scale.")
if v3_loss > v2b_loss:
    print(f"  >> v3 regressed from v2b. The added complexity hurt; simpler was better.")
    print(f"     This is a useful negative result: not every modern import helps.")
