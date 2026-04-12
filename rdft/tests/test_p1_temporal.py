"""
P1: Temporal Sequence Processing with DPFT Dynamics
=====================================================

The DPFT network currently processes spatial grids with fixed input.
This experiment makes it TEMPORAL: the field accumulates across timesteps
as tokens arrive, and particles fire at decision points.

Architecture:
  - Input: sequence of tokens, one per timestep
  - Field phi: persistent state that accumulates across tokens (stigmergy)
    Each token contributes to the field via deposition
    Field diffuses and decays between tokens
  - Particles n: discrete events that fire when field exceeds threshold
    The particle state at the final timestep = the output
  - Topology A: coupling graph between field channels (not spatial sites)
    Defines which internal "feature channels" interact

Task: Delayed Context-Query (from test_thought_cloud.py)
  - Context bit at position 0 sets a rule
  - Query bit at final position is the input
  - Label depends on BOTH (requires accumulated context + token identity)
  - A GRU can do this. Can the DPFT dynamics do it too?

Comparison:
  - Temporal DPFT (field accumulates across time)
  - GRU baseline (same hidden size)
  - LSTM baseline
  - Temporal DPFT without field (particles only, no stigmergy)

This proves the field IS stigmergy: it persists between tokens and
carries context that modulates future particle events.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import functools

print = functools.partial(print, flush=True)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ════════════════════════════════════════════════════════════════
# TEMPORAL DPFT NETWORK
# ════════════════════════════════════════════════════════════════

class TemporalDPFT(nn.Module):
    """
    DPFT network that processes sequences temporally.

    Per-token dynamics:
      1. Token is encoded and deposited into the field
      2. Field evolves (diffusion + decay + particle feedback)
      3. Particles evolve (nucleation from field + branching + death)
      4. Move to next token (field and particles PERSIST — stigmergy)

    The field state between tokens IS the memory.
    No hidden state, no cell state, no gates — just physics.
    """
    def __init__(self, vocab_size=4, n_channels=8, d_attn=4, n_substeps=3, dt=0.2):
        super().__init__()
        self.C = n_channels
        self.n_substeps = n_substeps  # dynamics steps per token
        self.dt = dt

        # Token -> field deposition
        self.token_embed = nn.Embedding(vocab_size, n_channels)

        # Field dynamics (channel-space, not spatial)
        # "Diffusion" here is inter-channel mixing
        self.channel_mix = nn.Linear(n_channels, n_channels, bias=False)
        nn.init.eye_(self.channel_mix.weight)  # init as identity
        self.mu = nn.Parameter(torch.ones(n_channels) * 0.05)  # field decay
        self.alpha = nn.Parameter(torch.ones(n_channels) * 0.2)  # particle -> field
        self.beta = nn.Parameter(torch.ones(n_channels) * 0.1)  # consumption

        # Particle dynamics
        self.birth_proj = nn.Linear(n_channels, n_channels)
        self.death_rate = nn.Parameter(torch.ones(n_channels) * 0.15)
        self.branching = nn.Parameter(torch.ones(n_channels) * 0.05)

        # Output
        self.head = nn.Linear(n_channels * 2, 2)  # read phi + n

    def step(self, phi, n, token_deposit):
        """Run n_substeps of coupled dynamics after depositing a token."""
        # Deposit token into field (stigmergic trace)
        phi = phi + token_deposit

        for _ in range(self.n_substeps):
            # Field: channel mixing (analogue of spatial diffusion)
            mixed = self.channel_mix(phi)
            mu = F.softplus(self.mu)
            alpha = self.alpha
            beta = F.softplus(self.beta)
            dphi = 0.1 * (mixed - phi) - mu * phi + alpha * n - beta * n * phi
            phi = phi + self.dt * dphi

            # Particles: nucleate from field
            birth = torch.sigmoid(self.birth_proj(phi))
            death = torch.sigmoid(self.death_rate)
            branch = F.softplus(self.branching)
            dn = birth * (1.0 + branch * n) - death * n
            n = F.relu(n + self.dt * dn)

        return phi, n

    def forward(self, x, return_trajectory=False):
        """
        x: (batch, seq_len) token ids
        """
        batch, seq_len = x.shape
        phi = torch.zeros(batch, self.C, device=x.device)
        n = torch.zeros(batch, self.C, device=x.device)

        traj_phi = []
        traj_n = []

        for t in range(seq_len):
            deposit = self.token_embed(x[:, t])  # (batch, C)
            phi, n = self.step(phi, n, deposit)

            if return_trajectory:
                traj_phi.append(phi.detach().norm(dim=1).mean().item())
                traj_n.append(n.detach().norm(dim=1).mean().item())

        # Classify from final state (both sectors)
        combined = torch.cat([phi, n], dim=1)
        logits = self.head(combined)

        if return_trajectory:
            return logits, traj_phi, traj_n
        return logits


class TemporalParticleOnly(nn.Module):
    """Ablation: particles evolve but no persistent field (no stigmergy)."""
    def __init__(self, vocab_size=4, n_channels=8, n_substeps=3, dt=0.2):
        super().__init__()
        self.C = n_channels
        self.n_substeps = n_substeps
        self.dt = dt
        self.token_embed = nn.Embedding(vocab_size, n_channels)
        self.birth_rate = nn.Parameter(torch.ones(n_channels) * 0.3)
        self.death_rate = nn.Parameter(torch.ones(n_channels) * 0.15)
        self.head = nn.Linear(n_channels, 2)

    def forward(self, x):
        batch, seq_len = x.shape
        n = torch.zeros(batch, self.C, device=x.device)
        for t in range(seq_len):
            deposit = self.token_embed(x[:, t])
            n = n + deposit  # direct deposit into particles
            for _ in range(self.n_substeps):
                birth = torch.sigmoid(self.birth_rate)
                death = torch.sigmoid(self.death_rate)
                dn = birth * (1.0 + 0.05 * n) - death * n
                n = F.relu(n + self.dt * dn)
        return self.head(n)


class GRUBaseline(nn.Module):
    """GRU with matched hidden size."""
    def __init__(self, vocab_size=4, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.embed(x)
        _, h_final = self.gru(h)
        return self.head(h_final.squeeze(0))


# ════════════════════════════════════════════════════════════════
# TASK: Delayed XOR (from test_thought_cloud.py)
# ════════════════════════════════════════════════════════════════

def make_data(n=3000, seq_len=10):
    VOCAB = 4
    X = torch.randint(0, VOCAB, (n, seq_len))
    X[:, 0] = torch.randint(0, 2, (n,))
    X[:, -1] = torch.randint(0, 2, (n,))
    context = X[:, 0]
    query = X[:, -1]
    y = torch.where(context == 0, query, 1 - query).long()
    return X.to(DEVICE), y.to(DEVICE)


def train(model, X_tr, y_tr, X_te, y_te, epochs=300, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {'acc': [], 'loss': []}
    for ep in range(epochs):
        model.train()
        logits = model(X_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(1) == y_te).float().mean().item()
        hist['acc'].append(acc)
        hist['loss'].append(loss.item())
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"    ep {ep:4d}  loss={loss.item():.4f}  acc={acc:.3f}")
    return hist


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

print()
for seq_len in [10, 20, 40]:
    print("=" * 70)
    print(f"DELAYED XOR — seq_len={seq_len}")
    print("=" * 70)

    torch.manual_seed(42)
    X_tr, y_tr = make_data(3000, seq_len)
    X_te, y_te = make_data(1000, seq_len)

    models = {
        'Temporal DPFT':     TemporalDPFT(n_channels=8, n_substeps=3).to(DEVICE),
        'Particle only':     TemporalParticleOnly(n_channels=8, n_substeps=3).to(DEVICE),
        'GRU':               GRUBaseline(hidden=8).to(DEVICE),
    }

    for name, model in models.items():
        print(f"\n  {name} ({nparams(model)} params):")
        t0 = time.time()
        hist = train(model, X_tr, y_tr, X_te, y_te, epochs=300)
        elapsed = time.time() - t0
        print(f"    Final acc={hist['acc'][-1]:.3f}  time={elapsed:.1f}s")

    # Dynamics trajectory for DPFT
    dpft = [m for n, m in models.items() if 'DPFT' in n][0]
    dpft.eval()
    with torch.no_grad():
        _, phi_traj, n_traj = dpft(X_te[:100], return_trajectory=True)

    print(f"\n  DPFT field trajectory (stigmergic accumulation):")
    print(f"  {'Token':>6}  {'|phi|':>7}  {'|n|':>7}")
    for t in range(min(seq_len, 15)):
        print(f"  {t:6d}  {phi_traj[t]:7.3f}  {n_traj[t]:7.3f}")
    if seq_len > 15:
        print(f"  {'...':>6}")
        print(f"  {seq_len-1:6d}  {phi_traj[-1]:7.3f}  {n_traj[-1]:7.3f}")

    # Split by context
    with torch.no_grad():
        mask0 = X_te[:100, 0] == 0
        mask1 = X_te[:100, 0] == 1
        _, phi0, _ = dpft(X_te[:100][mask0], return_trajectory=True)
        _, phi1, _ = dpft(X_te[:100][mask1], return_trajectory=True)
    sep = abs(phi0[-1] - phi1[-1])
    print(f"\n  Field divergence by context at final token: {sep:.3f}")
    if sep > 0.05:
        print("  Field retains context information across sequence — stigmergy works.")

print()
print("=" * 70)
print("P1 SUMMARY")
print("=" * 70)
print("  If Temporal DPFT matches GRU: the field IS memory (stigmergy).")
print("  If particles-only fails: the field is necessary (not just state).")
print("  If field diverges by context: the accumulated field encodes the rule.")
