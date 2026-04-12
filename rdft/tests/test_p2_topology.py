"""
P2: Topology as Load-Bearing Structure
========================================

Previous experiments showed the learned topology is nearly uniform
(effective degree 25.5/32). Topology refines but doesn't drive.

This experiment forces topology to be ESSENTIAL by using a task where
the GRAPH STRUCTURE is the answer: community detection.

Task: Planted community structure
  - A 32-node graph with 4 planted communities of 8 nodes each
  - Edge weights are high within communities, low between
  - Input: noisy node features (community label + noise)
  - Target: clean community labels

  A field-only model (local diffusion) can't solve this because the
  communities aren't spatially local — they're defined by connectivity.
  The topology (attention) must learn to couple nodes within the same
  community, and the field must diffuse ALONG those learned couplings
  to denoise the labels.

Comparison:
  - DPFT v2 (field + topology): should learn community structure in A
  - Field only (local kernel): should fail (communities aren't spatial)
  - Topology only (GNN): should work but less efficiently
  - MLP: should work with enough params but no structural inductive bias
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

GRID = 32
N_COMMUNITIES = 4
COMMUNITY_SIZE = GRID // N_COMMUNITIES


# ════════════════════════════════════════════════════════════════
# TASK: Community-structured denoising
# ════════════════════════════════════════════════════════════════

def make_community_data(n=3000):
    """
    Each sample: a 32-node graph with 4 planted communities.
    Input: noisy feature per node (community_id + Gaussian noise).
    Target: community_id per node (clean classification).

    The communities are randomly PERMUTED each sample — so the model
    can't memorize "node 0 is always community 0". It must discover
    the structure from the noisy features.
    """
    X = torch.zeros(n, GRID)
    Y = torch.zeros(n, GRID).long()

    for i in range(n):
        # Random permutation of community assignments
        perm = torch.randperm(GRID)
        labels = torch.zeros(GRID).long()
        for c in range(N_COMMUNITIES):
            members = perm[c*COMMUNITY_SIZE:(c+1)*COMMUNITY_SIZE]
            labels[members] = c

        # Noisy features: community_id (scaled) + noise
        # Noise level makes this non-trivial
        features = labels.float() * 2.0 + torch.randn(GRID) * 1.5

        X[i] = features
        Y[i] = labels

    return X.to(DEVICE), Y.to(DEVICE)


def make_community_data_with_graph(n=3000):
    """
    Harder version: features are PURE noise, but the graph structure
    (given as a second input) encodes community membership.
    The model must use topology to propagate labels.

    We encode graph structure by providing pairwise similarity:
    input = (node_features, adjacency_hint)
    """
    X_feat = torch.zeros(n, GRID)
    X_adj = torch.zeros(n, GRID, GRID)
    Y = torch.zeros(n, GRID).long()

    for i in range(n):
        perm = torch.randperm(GRID)
        labels = torch.zeros(GRID).long()
        for c in range(N_COMMUNITIES):
            members = perm[c*COMMUNITY_SIZE:(c+1)*COMMUNITY_SIZE]
            labels[members] = c

        # Features: ONE node per community gets a clean label (seed nodes)
        # Rest get noise
        features = torch.randn(GRID) * 2.0
        for c in range(N_COMMUNITIES):
            seed = perm[c * COMMUNITY_SIZE]  # first member is seed
            features[seed] = c * 3.0  # clean signal

        # Adjacency: strong within community, weak between
        adj = torch.zeros(GRID, GRID)
        for a in range(GRID):
            for b in range(GRID):
                if labels[a] == labels[b]:
                    adj[a, b] = 0.8 + 0.2 * torch.rand(1).item()
                else:
                    adj[a, b] = 0.1 * torch.rand(1).item()

        X_feat[i] = features
        X_adj[i] = adj
        Y[i] = labels

    return X_feat.to(DEVICE), X_adj.to(DEVICE), Y.to(DEVICE)


# ════════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════════

class DPFTCommunity(nn.Module):
    """DPFT v2 for community detection. Uses both field dynamics and topology."""
    def __init__(self, n_channels=4, d_attn=8, n_steps=20, dt=0.1):
        super().__init__()
        self.C = n_channels
        self.n_steps = n_steps
        self.dt = dt

        self.input_enc = nn.Linear(1, n_channels)

        # Topology
        self.site_q = nn.Linear(n_channels, d_attn)
        self.site_k = nn.Linear(n_channels, d_attn)
        self.d_attn = d_attn

        # Field dynamics
        self.local_kernel = nn.Parameter(torch.tensor([[[1., -2., 1.]]]).repeat(n_channels, 1, 1))
        self.D = nn.Parameter(torch.ones(n_channels) * 0.3)
        self.mu = nn.Parameter(torch.ones(n_channels) * 0.1)
        self.alpha = nn.Parameter(torch.ones(n_channels) * 0.2)
        self.beta = nn.Parameter(torch.ones(n_channels) * 0.1)
        self.local_w = nn.Parameter(torch.tensor(0.3))  # bias toward topological

        # Particles
        self.birth_proj = nn.Linear(n_channels, n_channels)
        self.death_rate = nn.Parameter(torch.ones(n_channels) * 0.2)
        self.branching = nn.Parameter(torch.ones(n_channels) * 0.05)
        self.topo_branch = nn.Parameter(torch.tensor(0.3))

        self.head = nn.Linear(n_channels * 2, N_COMMUNITIES)

    def forward(self, x, return_topo=False):
        phi = self.input_enc(x.unsqueeze(-1)).transpose(1, 2)  # (B, C, 32)
        n = torch.zeros_like(phi)
        source = phi.clone() * 0.05

        A = None
        for t in range(self.n_steps):
            if t % 4 == 0:
                phi_t = phi.transpose(1, 2)
                Q = self.site_q(phi_t)
                K = self.site_k(phi_t)
                A = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.d_attn**0.5, dim=-1)

            # Field: local + topological diffusion
            D = F.softplus(self.D).view(1, self.C, 1)
            mu = F.softplus(self.mu).view(1, self.C, 1)
            alpha = self.alpha.view(1, self.C, 1)
            beta = F.softplus(self.beta).view(1, self.C, 1)
            w = torch.sigmoid(self.local_w)

            local_lap = F.conv1d(phi, self.local_kernel, padding=1, groups=self.C)
            topo_lap = (torch.bmm(A, phi.transpose(1, 2)) - phi.transpose(1, 2)).transpose(1, 2)

            dphi = D * (w * local_lap + (1-w) * topo_lap) - mu*phi + alpha*n - beta*n*phi
            dphi = dphi + source * (0.5 ** (t / self.n_steps))
            phi = phi + self.dt * dphi

            # Particles
            birth = torch.sigmoid(self.birth_proj(phi.transpose(1, 2))).transpose(1, 2)
            death = torch.sigmoid(self.death_rate).view(1, self.C, 1)
            branch = F.softplus(self.branching).view(1, self.C, 1)
            topo_b = torch.sigmoid(self.topo_branch) * torch.bmm(A, n.transpose(1, 2)).transpose(1, 2)
            dn = birth * (1 + branch * n + topo_b) - death * n
            n = F.relu(n + self.dt * dn)

        combined = torch.cat([n, phi], dim=1).transpose(1, 2)  # (B, 32, 2C)
        logits = self.head(combined)  # (B, 32, 4)

        if return_topo:
            return logits, A.detach()
        return logits


class FieldOnlyComm(nn.Module):
    """Field with local diffusion only. No attention topology."""
    def __init__(self, n_channels=4, n_steps=20, dt=0.1):
        super().__init__()
        self.C = n_channels
        self.n_steps = n_steps
        self.dt = dt
        self.input_enc = nn.Linear(1, n_channels)
        self.kernel = nn.Parameter(torch.tensor([[[1., -2., 1.]]]).repeat(n_channels, 1, 1))
        self.D = nn.Parameter(torch.ones(n_channels) * 0.3)
        self.mu = nn.Parameter(torch.ones(n_channels) * 0.1)
        self.birth_proj = nn.Linear(n_channels, n_channels)
        self.death_rate = nn.Parameter(torch.ones(n_channels) * 0.2)
        self.head = nn.Linear(n_channels * 2, N_COMMUNITIES)

    def forward(self, x):
        phi = self.input_enc(x.unsqueeze(-1)).transpose(1, 2)
        n = torch.zeros_like(phi)
        for t in range(self.n_steps):
            lap = F.conv1d(phi, self.kernel, padding=1, groups=self.C)
            D = F.softplus(self.D).view(1, self.C, 1)
            mu = F.softplus(self.mu).view(1, self.C, 1)
            phi = phi + self.dt * (D * lap - mu * phi + 0.2 * n)
            birth = torch.sigmoid(self.birth_proj(phi.transpose(1, 2))).transpose(1, 2)
            death = torch.sigmoid(self.death_rate).view(1, self.C, 1)
            n = F.relu(n + self.dt * (birth - death * n))
        combined = torch.cat([n, phi], dim=1).transpose(1, 2)
        return self.head(combined)


class MLPComm(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRID, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, GRID * N_COMMUNITIES),
        )
    def forward(self, x):
        return self.net(x).view(x.shape[0], GRID, N_COMMUNITIES)


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def train_model(model, X_tr, Y_tr, X_te, Y_te, epochs=200, lr=1e-3, bs=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_tr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0; nb = 0
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            logits = model(X_tr[idx])
            loss = F.cross_entropy(logits.reshape(-1, N_COMMUNITIES), Y_tr[idx].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            te_logits = model(X_te)
            te_acc = (te_logits.argmax(-1) == Y_te).float().mean().item()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"    ep {ep:4d}  loss={total_loss/nb:.4f}  acc={te_acc:.3f}")
    return te_acc


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("P2: Community Detection (noisy features)")
print("=" * 70)

torch.manual_seed(42)
X_tr, Y_tr = make_community_data(3000)
X_te, Y_te = make_community_data(500)

models = {
    'DPFT (field+topo)': DPFTCommunity(n_channels=4, d_attn=8).to(DEVICE),
    'Field only':        FieldOnlyComm(n_channels=4).to(DEVICE),
    'MLP':               MLPComm(hidden=64).to(DEVICE),
}

results = {}
for name, model in models.items():
    print(f"\n  {name} ({nparams(model)} params):")
    t0 = time.time()
    acc = train_model(model, X_tr, Y_tr, X_te, Y_te, epochs=200)
    elapsed = time.time() - t0
    results[name] = {'acc': acc, 'params': nparams(model), 'time': elapsed}

print(f"\n  {'Model':<25} {'Acc':>6}  {'Params':>8}")
print(f"  {'─'*25} {'─'*6}  {'─'*8}")
for name, r in results.items():
    print(f"  {name:<25} {r['acc']:6.3f}  {r['params']:8,}")

# Topology analysis
dpft_model = [m for n, m in models.items() if 'DPFT' in n][0]
dpft_model.eval()
with torch.no_grad():
    _, A = dpft_model(X_te[:50], return_topo=True)

A_avg = A.mean(dim=0)  # average coupling matrix
print(f"\n  Coupling matrix analysis:")
print(f"    Sparsity (< 0.01): {(A_avg < 0.01).float().mean():.3f}")
print(f"    Effective degree: {(-(A_avg * (A_avg+1e-10).log()).sum(dim=1)).exp().mean():.1f}")

w = torch.sigmoid(dpft_model.local_w).item()
print(f"    Local weight: {w:.3f}, Topological: {1-w:.3f}")

print()
print("=" * 70)
print("P2 SUMMARY")
print("=" * 70)
print("  If DPFT >> Field-only: topology IS load-bearing for this task.")
print("  If topology learns sparse structure: attention discovered communities.")
print("  If local_weight << 0.5: the network prefers topological diffusion.")
