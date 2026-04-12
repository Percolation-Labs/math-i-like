"""
Doi-Peliti Field Theory Network (v2)
=====================================

Architecture insight (from discussion):
  - The FIELD is where stigmergy happens: particles deposit into it,
    it persists, it modifies future particle dynamics. This is the
    residual/shared medium. The MSR sector.
  - ATTENTION is topology: it defines WHO couples to WHOM. It is not
    a processing mechanism — it is the coupling GRAPH. The Kirchhoff
    polynomial counts spanning trees of THIS structure.

So the network has:
  1. A continuous FIELD (stigmergic medium) that evolves by diffusion-reaction
  2. ATTENTION that defines the coupling topology between sites
  3. Discrete PARTICLES that nucleate/die based on local field + topology

The field does the slow, persistent, indirect communication (pheromone).
The attention does the fast, direct, structural coupling (who-to-whom).
The particles are the discrete events (decisions, spikes, thoughts).

This is NOT a transformer with a field bolted on.
The attention here is NOT doing content processing — it is a learned
adjacency matrix that replaces the fixed spatial kernel (Laplacian).
The field does the actual computation through its dynamics.
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
# THE DPFT NETWORK v2: Field + Topological Attention
# ════════════════════════════════════════════════════════════════

class DPFTv2(nn.Module):
    """
    Coupled particle-field network with attention as topology.

    Three components:
      1. FIELD (phi): continuous, evolves by diffusion-reaction on a graph
         - The stigmergic medium. Particles deposit, field persists, field modulates.
         - dphi/dt = D * L(phi) - mu*phi + alpha*n - beta*n*phi
         - L is NOT a fixed Laplacian — it is a LEARNED graph Laplacian
           defined by attention weights.

      2. TOPOLOGY (A): attention-derived coupling matrix
         - Each site has a key and query (like attention)
         - A[i,j] = softmax(q_i . k_j / sqrt(d)) = coupling strength from j to i
         - This IS the adjacency matrix. The graph Laplacian is L = D_out - A.
         - The Kirchhoff polynomial of Feynman diagrams on this graph
           counts how information flows.
         - Updated every few timesteps (topology evolves slower than field)

      3. PARTICLES (n): discrete events at each site
         - Nucleate where field exceeds threshold (condensation)
         - Die at a constant rate (evaporation)
         - Branch: existing particles catalyse new ones (autocatalysis)
         - Birth rate depends on LOCAL field (stigmergic signal)
         - Branching rate depends on NEIGHBORS (topological coupling)

    Key difference from a transformer:
      - Transformer: attention computes output directly (weighted sum of values)
      - Here: attention defines the GRAPH on which a PDE runs
      - The PDE (field dynamics) does the computation over time
      - Attention is structure, not processing
    """
    def __init__(self, grid_size=32, n_steps=20, dt=0.1, n_channels=4, d_attn=8):
        super().__init__()
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.dt = dt
        self.C = n_channels
        self.d_attn = d_attn

        # ── Topology: attention as graph structure ──
        # Each site gets a query and key that define coupling
        self.site_query = nn.Linear(n_channels, d_attn)
        self.site_key = nn.Linear(n_channels, d_attn)
        # The "value" in standard attention is replaced by field dynamics
        # There is no V projection — attention only defines topology

        # ── Field dynamics ──
        self.D = nn.Parameter(torch.ones(n_channels) * 0.3)       # diffusion
        self.mu = nn.Parameter(torch.ones(n_channels) * 0.1)      # decay
        self.alpha = nn.Parameter(torch.ones(n_channels) * 0.3)   # deposition
        self.beta = nn.Parameter(torch.ones(n_channels) * 0.1)    # consumption

        # Also keep a LOCAL spatial kernel for nearest-neighbor diffusion
        # (attention handles long-range; this handles local smoothness)
        laplacian_init = torch.tensor([[[1.0, -2.0, 1.0]]]).repeat(n_channels, 1, 1)
        self.local_kernel = nn.Parameter(laplacian_init)
        self.local_weight = nn.Parameter(torch.tensor(0.5))  # balance local vs topological

        # ── Particle dynamics ──
        self.birth_proj = nn.Linear(n_channels, n_channels)
        self.death_rate = nn.Parameter(torch.ones(n_channels) * 0.2)
        self.branching = nn.Parameter(torch.ones(n_channels) * 0.05)
        # Topological branching: neighbors contribute to birth
        self.topo_branch_weight = nn.Parameter(torch.tensor(0.1))

        # ── Input/Output ──
        self.input_encoder = nn.Linear(1, n_channels)
        self.output_head = nn.Linear(n_channels * 2, 1)

    def compute_topology(self, phi):
        """
        Compute the coupling graph from current field state.
        This IS attention, but used as topology, not processing.

        phi: (batch, C, grid) -> A: (batch, grid, grid) coupling matrix
        """
        # Each site's field state determines its query/key
        phi_t = phi.transpose(1, 2)  # (batch, grid, C)
        Q = self.site_query(phi_t)   # (batch, grid, d_attn)
        K = self.site_key(phi_t)     # (batch, grid, d_attn)

        # Attention scores = coupling strengths
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_attn ** 0.5)
        A = torch.softmax(scores, dim=-1)  # (batch, grid, grid)
        # A[b, i, j] = how strongly site j couples to site i
        return A

    def graph_diffusion(self, phi, A):
        """
        Diffusion on the learned graph.
        Standard diffusion: dphi/dt = D * L * phi where L = graph Laplacian
        L = D_out - A (degree matrix minus adjacency)

        phi: (batch, C, grid)
        A:   (batch, grid, grid)
        """
        # For each channel, apply graph Laplacian
        # phi^T A gives "sum of neighbor field values weighted by coupling"
        phi_t = phi.transpose(1, 2)  # (batch, grid, C)
        neighbor_sum = torch.bmm(A, phi_t)  # (batch, grid, C)
        # Laplacian: neighbor_sum - phi (diffusion toward neighbors)
        lap = (neighbor_sum - phi_t).transpose(1, 2)  # (batch, C, grid)
        return lap

    def local_diffusion(self, phi):
        """Standard nearest-neighbor diffusion (spatial smoothness)."""
        return F.conv1d(phi, self.local_kernel, padding=1, groups=self.C)

    def field_step(self, phi, n, source, A):
        """Field dynamics using BOTH local and topological diffusion."""
        # Two diffusion channels:
        # 1. Local (nearest-neighbor, spatial smoothness)
        # 2. Topological (attention-defined graph, long-range coupling)
        D = F.softplus(self.D).view(1, self.C, 1)
        mu = F.softplus(self.mu).view(1, self.C, 1)
        alpha = self.alpha.view(1, self.C, 1)
        beta = F.softplus(self.beta).view(1, self.C, 1)
        w = torch.sigmoid(self.local_weight)

        local_lap = self.local_diffusion(phi)
        topo_lap = self.graph_diffusion(phi, A)

        # Weighted combination of local and topological diffusion
        combined_lap = w * local_lap + (1 - w) * topo_lap

        dphi = D * combined_lap - mu * phi + alpha * n - beta * n * phi + source
        return phi + self.dt * dphi

    def particle_step(self, n, phi, A):
        """Particle dynamics with topological branching."""
        # Local birth: depends on field at this site (stigmergic nucleation)
        phi_local = phi.transpose(1, 2)
        birth = torch.sigmoid(self.birth_proj(phi_local)).transpose(1, 2)

        death = torch.sigmoid(self.death_rate).view(1, self.C, 1)
        branch = F.softplus(self.branching).view(1, self.C, 1)

        # Topological branching: particles at coupled neighbors help nucleation
        # (ants following pheromone trails recruit more ants)
        n_t = n.transpose(1, 2)  # (batch, grid, C)
        neighbor_particles = torch.bmm(A, n_t).transpose(1, 2)  # (batch, C, grid)
        topo_branch = torch.sigmoid(self.topo_branch_weight) * neighbor_particles

        dn = birth * (1.0 + branch * n + topo_branch) - death * n
        return F.relu(n + self.dt * dn)

    def forward(self, x, return_dynamics=False):
        batch = x.shape[0]

        # Encode input into field
        phi = self.input_encoder(x.unsqueeze(-1)).transpose(1, 2)
        n = torch.zeros_like(phi)
        source = phi.clone() * 0.1

        traj_phi = []
        traj_n = []
        traj_topo = []
        topo_update_freq = 4  # topology evolves slower than field

        A = None
        for t in range(self.n_steps):
            # Update topology every few steps (it's the slow variable)
            if t % topo_update_freq == 0:
                A = self.compute_topology(phi)

            phi = self.field_step(phi, n, source * (0.5 ** (t / self.n_steps)), A)
            n = self.particle_step(n, phi, A)

            if return_dynamics:
                traj_phi.append(phi.detach().mean(dim=1).mean(dim=0).cpu())
                traj_n.append(n.detach().mean(dim=1).mean(dim=0).cpu())
                if t % topo_update_freq == 0:
                    traj_topo.append(A.detach().mean(dim=0).cpu())

        combined = torch.cat([n, phi], dim=1).transpose(1, 2)
        output = self.output_head(combined).squeeze(-1)

        if return_dynamics:
            return output, traj_phi, traj_n, traj_topo
        return output


# ════════════════════════════════════════════════════════════════
# ABLATIONS: isolate each component
# ════════════════════════════════════════════════════════════════

class FieldLocalOnly(nn.Module):
    """Field with LOCAL diffusion only (fixed Laplacian, no attention topology)."""
    def __init__(self, grid_size=32, n_steps=20, dt=0.1, n_channels=4):
        super().__init__()
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.dt = dt
        self.C = n_channels

        self.local_kernel = nn.Parameter(torch.tensor([[[1.0, -2.0, 1.0]]]).repeat(n_channels, 1, 1))
        self.D = nn.Parameter(torch.ones(n_channels) * 0.5)
        self.mu = nn.Parameter(torch.ones(n_channels) * 0.1)
        self.alpha = nn.Parameter(torch.ones(n_channels) * 0.3)
        self.beta = nn.Parameter(torch.ones(n_channels) * 0.1)
        self.birth_proj = nn.Linear(n_channels, n_channels)
        self.death_rate = nn.Parameter(torch.ones(n_channels) * 0.2)
        self.branching = nn.Parameter(torch.ones(n_channels) * 0.05)
        self.input_encoder = nn.Linear(1, n_channels)
        self.output_head = nn.Linear(n_channels * 2, 1)

    def forward(self, x):
        phi = self.input_encoder(x.unsqueeze(-1)).transpose(1, 2)
        n = torch.zeros_like(phi)
        source = phi.clone() * 0.1
        D = F.softplus(self.D).view(1, self.C, 1)
        mu = F.softplus(self.mu).view(1, self.C, 1)
        alpha = self.alpha.view(1, self.C, 1)
        beta = F.softplus(self.beta).view(1, self.C, 1)

        for t in range(self.n_steps):
            lap = F.conv1d(phi, self.local_kernel, padding=1, groups=self.C)
            dphi = D * lap - mu * phi + alpha * n - beta * n * phi + source * (0.5 ** (t/self.n_steps))
            phi = phi + self.dt * dphi

            phi_t = phi.transpose(1, 2)
            birth = torch.sigmoid(self.birth_proj(phi_t)).transpose(1, 2)
            death = torch.sigmoid(self.death_rate).view(1, self.C, 1)
            branch = F.softplus(self.branching).view(1, self.C, 1)
            dn = birth * (1.0 + branch * n) - death * n
            n = F.relu(n + self.dt * dn)

        combined = torch.cat([n, phi], dim=1).transpose(1, 2)
        return self.output_head(combined).squeeze(-1)


class TopoOnly(nn.Module):
    """Attention topology with NO field dynamics — just weighted aggregation.
    This IS a standard attention/graph-neural-network. The control."""
    def __init__(self, grid_size=32, n_steps=5, n_channels=4, d_attn=8):
        super().__init__()
        self.C = n_channels
        self.n_steps = n_steps
        self.input_encoder = nn.Linear(1, n_channels)
        self.query = nn.Linear(n_channels, d_attn)
        self.key = nn.Linear(n_channels, d_attn)
        self.value = nn.Linear(n_channels, n_channels)  # HAS values (standard attention)
        self.ffn = nn.Linear(n_channels, n_channels)
        self.output_head = nn.Linear(n_channels, 1)
        self.d_attn = d_attn

    def forward(self, x):
        h = self.input_encoder(x.unsqueeze(-1))  # (batch, grid, C)
        for _ in range(self.n_steps):
            Q = self.query(h)
            K = self.key(h)
            V = self.value(h)
            A = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.d_attn**0.5, dim=-1)
            h = h + torch.bmm(A, V)  # standard attention
            h = h + F.relu(self.ffn(h))
        return self.output_head(h).squeeze(-1)


class MLPBaseline(nn.Module):
    """Standard MLP — no physics, no graph."""
    def __init__(self, grid_size=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(grid_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, grid_size),
        )

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════
# TASKS
# ════════════════════════════════════════════════════════════════

GRID = 32

def task_blur_threshold(n=3000):
    """Spikes -> Gaussian blur -> threshold. Needs local diffusion + nucleation."""
    X = torch.zeros(n, GRID)
    Y = torch.zeros(n, GRID)
    for i in range(n):
        n_spikes = torch.randint(2, 6, (1,)).item()
        positions = torch.randint(0, GRID, (n_spikes,))
        X[i, positions] = torch.rand(n_spikes) * 2
        kernel = torch.exp(-torch.arange(-4, 5).float()**2 / 4.0)
        kernel = kernel / kernel.sum()
        blurred = F.conv1d(X[i:i+1].unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=4).squeeze()
        Y[i] = (blurred > 0.15).float()
    return X.to(DEVICE), Y.to(DEVICE)


def task_nonlocal_spread(n=3000):
    """
    Spikes spread to form clusters — nearby spikes merge influence.
    Requires both local diffusion AND nonlocal coupling (distant spikes
    that happen to have similar heights should create similar patterns).
    This tests whether topological attention adds value over pure diffusion.
    """
    X = torch.zeros(n, GRID)
    Y = torch.zeros(n, GRID)
    for i in range(n):
        n_spikes = torch.randint(3, 7, (1,)).item()
        positions = torch.randint(0, GRID, (n_spikes,))
        heights = torch.randint(1, 4, (n_spikes,)).float()
        X[i, positions] = heights

        # Rule: each spike creates a spread proportional to height
        # AND spikes with the SAME height reinforce each other nonlocally
        profile = torch.zeros(GRID)
        pos_f = torch.arange(GRID).float()
        for s in range(n_spikes):
            p, h = positions[s].item(), heights[s].item()
            # Local spread
            profile += h * 0.3 * torch.exp(-(pos_f - p)**2 / (2 * h))
            # Nonlocal: same-height spikes add a global boost
            same_h = (heights == h).float().sum().item() - 1  # other spikes with same height
            profile += same_h * 0.1 * torch.exp(-(pos_f - p)**2 / (2 * (h + 2)))

        Y[i] = (profile > 0.5).float()
    return X.to(DEVICE), Y.to(DEVICE)


def task_pattern_completion(n=3000):
    """
    Input: partial periodic pattern (some positions masked).
    Target: complete pattern.
    Requires learning the spatial structure (topology) of the pattern
    and using it to fill gaps (stigmergic field propagation).
    """
    X = torch.zeros(n, GRID)
    Y = torch.zeros(n, GRID)
    for i in range(n):
        freq = torch.randint(2, 6, (1,)).item()
        phase = torch.rand(1).item() * 2 * np.pi
        pattern = (torch.sin(torch.arange(GRID).float() * 2 * np.pi * freq / GRID + phase) > 0).float()
        Y[i] = pattern

        # Mask ~40% of positions
        mask = torch.rand(GRID) > 0.4
        X[i] = pattern * mask.float()
    return X.to(DEVICE), Y.to(DEVICE)


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def train_model(model, X_tr, Y_tr, X_te, Y_te, epochs=200, lr=1e-3, batch_size=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_tr.shape[0]
    hist = {'loss': [], 'test_loss': []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0
        nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            pred = model(X_tr[idx])
            loss = F.mse_loss(pred, Y_tr[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            te_loss = F.mse_loss(model(X_te), Y_te).item()

        hist['loss'].append(total_loss / nb)
        hist['test_loss'].append(te_loss)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    ep {epoch:4d}  train={total_loss/nb:.5f}  test={te_loss:.5f}")

    return hist


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ════════════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ════════════════════════════════════════════════════════════════

tasks = {
    'blur+threshold (local)':        task_blur_threshold,
    'nonlocal spread (needs topo)':  task_nonlocal_spread,
    'pattern completion (needs both)': task_pattern_completion,
}

all_results = {}

for task_name, task_fn in tasks.items():
    print()
    print("=" * 70)
    print(f"TASK: {task_name}")
    print("=" * 70)

    torch.manual_seed(42)
    X_tr, Y_tr = task_fn(3000)
    X_te, Y_te = task_fn(500)

    models = {
        'DPFT v2 (field+topo)':  DPFTv2(GRID, n_steps=20, n_channels=4, d_attn=8).to(DEVICE),
        'Field only (local)':    FieldLocalOnly(GRID, n_steps=20, n_channels=4).to(DEVICE),
        'Topo only (attn/GNN)':  TopoOnly(GRID, n_steps=5, n_channels=4, d_attn=8).to(DEVICE),
        'MLP baseline':          MLPBaseline(GRID, hidden=64).to(DEVICE),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  {name} ({nparams(model):,} params):")
        t0 = time.time()
        hist = train_model(model, X_tr, Y_tr, X_te, Y_te, epochs=200, lr=1e-3)
        elapsed = time.time() - t0
        results[name] = {
            'test_loss': hist['test_loss'][-1],
            'params': nparams(model),
            'time': elapsed,
        }

    print(f"\n  {'Model':<25} {'Test MSE':>10}  {'Params':>8}  {'Time':>6}")
    print(f"  {'─'*25} {'─'*10}  {'─'*8}  {'─'*6}")
    for name, r in results.items():
        print(f"  {name:<25} {r['test_loss']:10.5f}  {r['params']:8,}  {r['time']:5.1f}s")

    all_results[task_name] = results


# ════════════════════════════════════════════════════════════════
# TOPOLOGY ANALYSIS: What graph did it learn?
# ════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("TOPOLOGY ANALYSIS: What coupling graph did attention learn?")
print("=" * 70)
print()

# Train DPFT v2 on pattern completion and inspect topology
torch.manual_seed(42)
X_tr, Y_tr = task_pattern_completion(3000)
X_te, Y_te = task_pattern_completion(500)

model = DPFTv2(GRID, n_steps=20, n_channels=4, d_attn=8).to(DEVICE)
train_model(model, X_tr, Y_tr, X_te, Y_te, epochs=200, lr=1e-3)

model.eval()
with torch.no_grad():
    _, traj_phi, traj_n, traj_topo = model(X_te[:50], return_dynamics=True)

# Analyze the coupling matrix
if traj_topo:
    A = traj_topo[-1]  # final topology, (grid, grid)
    print("  Coupling matrix properties:")
    print(f"    Shape: {A.shape}")
    print(f"    Sparsity (fraction < 0.01): {(A < 0.01).float().mean():.3f}")
    print(f"    Max coupling: {A.max():.4f}")
    print(f"    Diagonal dominance: {A.diag().mean():.4f} vs off-diag {(A - A.diag().diag()).abs().mean():.4f}")

    # Is it local (diagonal-heavy) or nonlocal (spread out)?
    for radius in [1, 3, 5, 10]:
        band = torch.zeros_like(A)
        for i in range(GRID):
            for j in range(GRID):
                if abs(i-j) <= radius or abs(i-j) >= GRID-radius:
                    band[i,j] = 1
        band_mass = (A * band).sum() / A.sum()
        print(f"    Mass within radius {radius:2d}: {band_mass:.3f}")

    # Effective degree (how many neighbors each site really couples to)
    # Use entropy of each row as a measure
    row_entropy = -(A * (A + 1e-10).log()).sum(dim=1)
    effective_degree = row_entropy.exp()
    print(f"    Effective degree: {effective_degree.mean():.1f} +/- {effective_degree.std():.1f}")
    print(f"    (Uniform coupling over {GRID} sites would give {GRID:.0f})")
    print(f"    (Pure local would give ~1)")

    print()
    print("  Field-to-particle transfer during dynamics:")
    print(f"  {'Step':>5}  {'|phi|':>8}  {'|n|':>8}  {'ratio':>8}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}")
    for t in range(0, len(traj_phi), 4):
        pn = traj_phi[t].norm().item()
        nn_val = traj_n[t].norm().item()
        print(f"  {t:5d}  {pn:8.4f}  {nn_val:8.4f}  {pn/(nn_val+1e-8):8.2f}")

print()
print("  Learned local vs topological balance:")
w = torch.sigmoid(model.local_weight).item()
print(f"    Local weight: {w:.3f}")
print(f"    Topological weight: {1-w:.3f}")
if w > 0.7:
    print("    Mostly local diffusion — topology adds refinement")
elif w < 0.3:
    print("    Mostly topological — long-range coupling dominates")
else:
    print("    Balanced: both local smoothness and long-range structure matter")

print()
print(f"  Topological branching weight: {torch.sigmoid(model.topo_branch_weight).item():.3f}")
print("  (How much neighbors' particles help local nucleation)")


# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("FINAL SUMMARY: Three Components, Three Roles")
print("=" * 70)
print()
print("  FIELD (phi) = stigmergic medium")
print("    Particles deposit -> field persists -> field modulates future particles")
print("    Information propagates by diffusion (slow, local, persistent)")
print("    This is pheromone. This is supersaturation. This is relational context.")
print()
print("  TOPOLOGY (A) = coupling graph (attention as structure, not processing)")
print("    Defines WHO couples to WHOM — the graph Laplacian for field diffusion")
print("    AND the neighborhood for particle branching")
print("    This is the Kirchhoff polynomial's graph. Spanning trees = information paths.")
print("    Updated slowly (structure changes slower than content)")
print()
print("  PARTICLES (n) = discrete events")
print("    Nucleate from field (condensation), die (evaporation), branch (catalysis)")
print("    The countable things: spikes, decisions, words, thoughts")
print()

for task_name, results in all_results.items():
    dpft = results['DPFT v2 (field+topo)']['test_loss']
    field = results['Field only (local)']['test_loss']
    topo = results['Topo only (attn/GNN)']['test_loss']
    mlp = results['MLP baseline']['test_loss']
    print(f"  {task_name}:")
    print(f"    DPFT v2: {dpft:.5f}  Field: {field:.5f}  Topo: {topo:.5f}  MLP: {mlp:.5f}")
    if dpft < min(field, topo) * 0.95:
        print(f"    -> BOTH components contribute (field+topo < either alone)")
    print()

print("  This architecture computes by running physics, not by stacking layers.")
print("  Attention defines the topology. The field carries the stigmergy.")
print("  Particles are the discrete outputs. 73-200 params. No transformer.")
