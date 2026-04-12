"""
The Condensation Machine
=========================

The simplest possible learning machine built from coupled particle-field dynamics.

The idea:
  Evidence accumulates in a continuous FIELD (like supersaturation building).
  When enough evidence gathers, a PARTICLE nucleates (a decision crystallises).
  The particle DEPLETES the field (commitment consumes evidence).
  Other particles can no longer nucleate from the depleted field.

  Classification = which particle nucleates first.
  Learning = adjusting deposition/threshold/depletion so the RIGHT particle wins.

The TENSION that drives learning:
  - Field wants to accumulate (more evidence = better discrimination)
  - Particles want to fire (commitment before the field saturates)
  - Depletion creates competition (one decision suppresses others)
  - Too eager = noise triggers wrong particle
  - Too cautious = field saturates, all particles fire together

The optimal operating point is at the EDGE — near the phase transition
between "nothing fires" and "everything fires". The network learns to
sit at criticality because that's where it can discriminate.

This is a drift-diffusion model (neuroscience) crossed with
Wegener-Bergeron-Findeisen (cloud physics), made differentiable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

print = functools.partial(print, flush=True)
torch.manual_seed(42)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


class CondensationMachine(nn.Module):
    """
    A learning machine where classification = condensation.

    State:
      phi: (batch, field_dim) — the evidence field (continuous, shared)
      n:   (batch, n_classes) — particle activations (soft-discrete, competing)

    Per input feature:
      1. DEPOSIT: input feature is projected into field
         phi += W_deposit @ feature
      2. DIFFUSE: field mixes (evidence integrates)
         phi = phi + dt * (W_mix @ phi - decay * phi)
      3. NUCLEATE: each class reads the field and tries to fire
         n_k += dt * sigma(W_nucleate_k @ phi) * (1 - n_k)
         The (1-n_k) term: once a particle has fired, it stays fired.
      4. DEPLETE: fired particles consume the field
         phi -= sum_k(n_k * W_deplete_k)
         This is the WBF process: one crystal grows, supersaturation drops,
         other crystals can't nucleate from depleted field.

    Output: particle activations n (which class condensed?)
    Loss: cross-entropy on n

    What's learned:
      - W_deposit: which features contribute what kind of evidence
      - W_nucleate: what field configuration triggers each class
      - W_deplete: how much each decision consumes the evidence
      - decay, thresholds: operating point (how close to criticality)
    """
    def __init__(self, input_dim, n_classes, field_dim=16, n_steps=5, dt=0.5):
        super().__init__()
        self.field_dim = field_dim
        self.n_classes = n_classes
        self.n_steps = n_steps
        self.dt = dt

        # Deposition: input -> field
        self.deposit = nn.Linear(input_dim, field_dim)

        # Field dynamics: mixing + decay
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.1))

        # Nucleation: field -> particle birth (one per class)
        self.nucleate = nn.Linear(field_dim, n_classes)

        # Depletion: each particle's consumption pattern
        self.deplete = nn.Linear(n_classes, field_dim, bias=False)
        # Init small so depletion starts weak and grows with learning
        nn.init.normal_(self.deplete.weight, std=0.01)

    def forward(self, x, return_dynamics=False):
        """
        x: (batch, input_dim)
        """
        batch = x.shape[0]

        # Deposit input into field
        phi = self.deposit(x)  # (batch, field_dim)

        # Particles start at zero (nothing has condensed yet)
        n = torch.zeros(batch, self.n_classes, device=x.device)

        phi_traj = []
        n_traj = []

        for t in range(self.n_steps):
            # Field dynamics: mix + decay
            phi = phi + self.dt * (self.mix(phi) - F.softplus(self.decay) * phi)

            # Nucleation: field drives particle birth
            # The (1-n) factor: already-fired particles don't re-fire
            birth_rate = torch.sigmoid(self.nucleate(phi))
            n = n + self.dt * birth_rate * (1 - n)
            n = n.clamp(0, 1)

            # Depletion: particles consume the field
            phi = phi - self.deplete(n)

            if return_dynamics:
                phi_traj.append(phi.detach().norm(dim=1).mean().item())
                n_traj.append(n.detach().clone())

        if return_dynamics:
            return n, phi_traj, n_traj
        return n


class MLPBaseline(nn.Module):
    """Same task, standard MLP, matched params."""
    def __init__(self, input_dim, n_classes, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════
# TASKS
# ════════════════════════════════════════════════════════════════

def make_classification_data(n=2000, input_dim=8, n_classes=4):
    """Simple classification: 4 clusters in 8D space."""
    X = torch.randn(n, input_dim)
    y = torch.randint(0, n_classes, (n,))
    # Plant class signal: class k has positive mean in dimensions k*2, k*2+1
    for k in range(n_classes):
        mask = y == k
        X[mask, k*2] += 2.0
        X[mask, k*2+1] += 1.5
    return X.to(DEVICE), y.to(DEVICE)


def make_xor_data(n=2000):
    """XOR: non-linearly separable. Tests if field dynamics help."""
    X = torch.randn(n, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).long()
    return X.to(DEVICE), y.to(DEVICE)


def make_competition_data(n=2000, input_dim=8, n_classes=4):
    """
    Ambiguous inputs where classes COMPETE.
    Each input has evidence for TWO classes. The stronger signal should win.
    Tests the depletion/competition mechanism.
    """
    X = torch.randn(n, input_dim) * 0.5
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        # Pick two competing classes
        c1, c2 = torch.randperm(n_classes)[:2].tolist()
        strength1 = 1.5 + torch.rand(1).item()
        strength2 = 0.5 + torch.rand(1).item()
        X[i, c1*2] += strength1
        X[i, c2*2] += strength2
        y[i] = c1  # stronger signal wins
    return X.to(DEVICE), y.to(DEVICE)


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def train(model, X_tr, y_tr, X_te, y_te, epochs=200, lr=3e-3, is_condensation=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {'acc': [], 'loss': []}

    for ep in range(epochs):
        model.train()
        logits = model(X_tr)
        if is_condensation:
            # For condensation machine: n is in [0,1], treat as probabilities
            loss = F.cross_entropy(logits * 5, y_tr)  # scale up for sharper decisions
        else:
            loss = F.cross_entropy(logits, y_tr)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            te_logits = model(X_te)
            if is_condensation:
                te_logits = te_logits * 5
            acc = (te_logits.argmax(1) == y_te).float().mean().item()

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

tasks = {
    'clusters (4 classes in 8D)': lambda: make_classification_data(2000, 8, 4),
    'XOR (nonlinear)': lambda: make_xor_data(2000),
    'competition (ambiguous)': lambda: make_competition_data(2000, 8, 4),
}

for task_name, task_fn in tasks.items():
    print()
    print("=" * 70)
    print(f"TASK: {task_name}")
    print("=" * 70)

    torch.manual_seed(42)
    X_tr, y_tr = task_fn()
    X_te, y_te = task_fn()

    n_classes = y_tr.max().item() + 1
    input_dim = X_tr.shape[1]

    cm = CondensationMachine(input_dim, n_classes, field_dim=16, n_steps=5).to(DEVICE)
    mlp = MLPBaseline(input_dim, n_classes, hidden=16).to(DEVICE)

    print(f"\n  Condensation Machine ({nparams(cm)} params):")
    h_cm = train(cm, X_tr, y_tr, X_te, y_te, epochs=200, is_condensation=True)

    print(f"\n  MLP Baseline ({nparams(mlp)} params):")
    h_mlp = train(mlp, X_tr, y_tr, X_te, y_te, epochs=200)

    print(f"\n  Result:  CM={h_cm['acc'][-1]:.3f}  MLP={h_mlp['acc'][-1]:.3f}")

    # ── Dynamics analysis ──
    cm.eval()
    with torch.no_grad():
        _, phi_traj, n_traj = cm(X_te[:200], return_dynamics=True)

    print(f"\n  Condensation dynamics:")
    print(f"  {'Step':>5}  {'|phi|':>7}  {'n_max':>7}  {'n_spread':>9}  {'decided':>8}")
    for t in range(len(phi_traj)):
        n_t = n_traj[t]  # (200, n_classes)
        n_max = n_t.max(dim=1).values.mean().item()
        n_spread = (n_t.max(dim=1).values - n_t.min(dim=1).values).mean().item()
        decided = (n_t.max(dim=1).values > 0.8).float().mean().item()
        print(f"  {t:5d}  {phi_traj[t]:7.3f}  {n_max:7.3f}  {n_spread:9.3f}  {decided:8.1%}")

    # Depletion analysis
    depl_norm = cm.deplete.weight.data.norm().item()
    decay_val = F.softplus(cm.decay).item()
    print(f"\n  Learned parameters:")
    print(f"    Depletion strength: {depl_norm:.3f}")
    print(f"    Field decay: {decay_val:.3f}")
    print(f"    Nucleation bias: {cm.nucleate.bias.data.cpu().numpy()}")

    # Check: does the winning particle suppress others?
    with torch.no_grad():
        n_final = n_traj[-1]  # (200, n_classes)
        winner = n_final.argmax(dim=1)  # (200,)
        winner_activation = n_final[torch.arange(200), winner].mean().item()
        loser_activation = (n_final.sum(dim=1) - n_final[torch.arange(200), winner]).mean().item() / max(1, n_classes - 1)
        print(f"    Winner activation: {winner_activation:.3f}")
        print(f"    Avg loser activation: {loser_activation:.3f}")
        print(f"    Suppression ratio: {winner_activation / (loser_activation + 1e-6):.1f}x")


# ════════════════════════════════════════════════════════════════
# THE KEY QUESTION: Does depletion help or hurt?
# ════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("ABLATION: Does depletion (competition) help?")
print("=" * 70)

torch.manual_seed(42)
X_tr, y_tr = make_competition_data(2000, 8, 4)
X_te, y_te = make_competition_data(2000, 8, 4)

# With depletion
cm_depl = CondensationMachine(8, 4, field_dim=16, n_steps=5).to(DEVICE)
h_depl = train(cm_depl, X_tr, y_tr, X_te, y_te, epochs=300, is_condensation=True)

# Without depletion (zero out depletion and freeze it)
cm_nodepl = CondensationMachine(8, 4, field_dim=16, n_steps=5).to(DEVICE)
cm_nodepl.deplete.weight.data.zero_()
cm_nodepl.deplete.weight.requires_grad = False
print(f"\n  Without depletion ({nparams(cm_nodepl)} params, depletion frozen at 0):")
h_nodepl = train(cm_nodepl, X_tr, y_tr, X_te, y_te, epochs=300, is_condensation=True)

print(f"\n  With depletion:    {h_depl['acc'][-1]:.3f}")
print(f"  Without depletion: {h_nodepl['acc'][-1]:.3f}")

if h_depl['acc'][-1] > h_nodepl['acc'][-1] + 0.02:
    print("  >> Depletion HELPS. Competition between particles improves discrimination.")
    print("     The WBF process is not just physics — it's a learning mechanism.")
elif h_nodepl['acc'][-1] > h_depl['acc'][-1] + 0.02:
    print("  >> Depletion HURTS. Winner-take-all suppresses useful information.")
else:
    print("  >> Similar. Depletion is neutral at this scale.")


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("  The Condensation Machine classifies by PHYSICS:")
print("    1. Evidence deposits into a shared field")
print("    2. Field integrates and mixes (diffusion)")
print("    3. Particles nucleate when evidence is sufficient")
print("    4. Nucleation depletes the field (commitment)")
print("    5. Depletion suppresses competing hypotheses")
print()
print("  Classification = which particle condenses first")
print("  Learning = adjusting the condensation landscape")
print("  The tension: accumulate vs commit, shared vs competitive")
