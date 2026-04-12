"""
Condensation Language Model
=============================

Next-token prediction via condensation physics.

At each position:
  1. Token deposits into field (context accumulates)
  2. Field evolves (mixing, decay)
  3. Particles nucleate to predict next token (condensation = prediction)
  4. Nucleation DEPLETES the field (prediction consumes context)
  5. Next token arrives, deposits into depleted field

Depletion is now ESSENTIAL: each prediction changes the field for
the next one. Without depletion, the field just grows monotonically
and saturates. With depletion, the field is a WORKING MEMORY that
gets consumed as predictions are made and refreshed as tokens arrive.

Data: synthetic sequences with learnable structure.
  - Bigram patterns (A usually follows B)
  - Repetition (ABCABC...)
  - Simple grammar (noun-verb-noun patterns)
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

VOCAB = 8
SEQ_LEN = 16


# ════════════════════════════════════════════════════════════════
# CONDENSATION LM
# ════════════════════════════════════════════════════════════════

class CondensationLM(nn.Module):
    """
    Language model where prediction = condensation.

    The field is the context. Tokens deposit evidence.
    Predictions nucleate from the field. Predictions deplete the field.
    The balance of deposition and depletion IS the memory mechanism.
    """
    def __init__(self, vocab=VOCAB, field_dim=32, n_substeps=3, dt=0.3):
        super().__init__()
        self.vocab = vocab
        self.field_dim = field_dim
        self.n_substeps = n_substeps
        self.dt = dt

        # Token -> field deposition
        self.embed = nn.Embedding(vocab, field_dim)

        # Field dynamics
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.05))

        # Nucleation: field -> next token prediction
        self.nucleate = nn.Linear(field_dim, vocab)

        # Depletion: prediction consumes field
        # Key: this is a LEARNED consumption pattern
        # Each predicted token removes specific evidence from the field
        self.deplete = nn.Linear(vocab, field_dim, bias=False)
        nn.init.normal_(self.deplete.weight, std=0.05)

    def forward(self, x, return_dynamics=False):
        """
        x: (batch, seq_len) token ids
        Returns: (batch, seq_len, vocab) logits for next token
        """
        batch, seq_len = x.shape
        phi = torch.zeros(batch, self.field_dim, device=x.device)

        all_logits = []
        phi_norms = []
        depletions = []

        for t in range(seq_len):
            # 1. DEPOSIT: token adds evidence to field
            deposit = self.embed(x[:, t])
            phi = phi + deposit

            # 2. EVOLVE: field mixes and decays
            for _ in range(self.n_substeps):
                phi = phi + self.dt * (self.mix(phi) - F.softplus(self.decay) * phi)

            # 3. NUCLEATE: predict next token from field
            logits = self.nucleate(phi)  # (batch, vocab)
            all_logits.append(logits)

            # 4. DEPLETE: prediction consumes field
            # Use soft prediction (not argmax) so it's differentiable
            pred_soft = F.softmax(logits.detach(), dim=-1)  # stop grad on prediction
            consumption = self.deplete(pred_soft)
            phi = phi - consumption

            if return_dynamics:
                phi_norms.append(phi.norm(dim=1).mean().item())
                depletions.append(consumption.norm(dim=1).mean().item())

        logits_out = torch.stack(all_logits, dim=1)  # (batch, seq_len, vocab)

        if return_dynamics:
            return logits_out, phi_norms, depletions
        return logits_out


class CondensationLM_NoDepletion(nn.Module):
    """Ablation: same but no depletion. Field only grows."""
    def __init__(self, vocab=VOCAB, field_dim=32, n_substeps=3, dt=0.3):
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


class GRUBaseline(nn.Module):
    """GRU LM baseline."""
    def __init__(self, vocab=VOCAB, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        h = self.embed(x)
        out, _ = self.gru(h)
        return self.head(out)


# ════════════════════════════════════════════════════════════════
# DATA: Sequences with learnable structure
# ════════════════════════════════════════════════════════════════

def make_bigram_data(n=3000, seq_len=SEQ_LEN):
    """
    Sequences generated by a bigram model.
    Each token has a preferred successor.
    0->1, 1->2, 2->3, 3->0 (cycle) with 70% probability,
    random otherwise.
    """
    transition = torch.zeros(VOCAB, VOCAB)
    for i in range(VOCAB):
        transition[i] = 0.3 / (VOCAB - 1)
        transition[i, (i + 1) % VOCAB] = 0.7

    X = torch.zeros(n, seq_len, dtype=torch.long)
    X[:, 0] = torch.randint(0, VOCAB, (n,))
    for t in range(1, seq_len):
        for i in range(n):
            prev = X[i, t-1].item()
            X[i, t] = torch.multinomial(transition[prev], 1)

    return X.to(DEVICE)


def make_repetition_data(n=3000, seq_len=SEQ_LEN):
    """
    Sequences that repeat a short motif.
    Motif of length 3-4, repeated to fill seq_len.
    Next token = motif[position % motif_len].
    """
    X = torch.zeros(n, seq_len, dtype=torch.long)
    for i in range(n):
        motif_len = torch.randint(3, 5, (1,)).item()
        motif = torch.randint(0, VOCAB, (motif_len,))
        for t in range(seq_len):
            X[i, t] = motif[t % motif_len]
    return X.to(DEVICE)


def make_grammar_data(n=3000, seq_len=SEQ_LEN):
    """
    Simple grammar: tokens 0-3 are "nouns", 4-7 are "verbs".
    Pattern: N V N V N V...
    Within nouns: any of 0-3. Within verbs: any of 4-7.
    Next-token prediction must learn: after a noun comes a verb,
    after a verb comes a noun.
    """
    X = torch.zeros(n, seq_len, dtype=torch.long)
    for i in range(n):
        for t in range(seq_len):
            if t % 2 == 0:
                X[i, t] = torch.randint(0, 4, (1,))  # noun
            else:
                X[i, t] = torch.randint(4, 8, (1,))  # verb
    return X.to(DEVICE)


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def train_lm(model, X_tr, X_te, epochs=300, lr=3e-3, bs=256):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_tr.shape[0]
    hist = {'loss': [], 'acc': []}

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0; nb = 0
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            xb = X_tr[idx]
            logits = model(xb[:, :-1])  # predict from tokens 0..T-2
            targets = xb[:, 1:]          # targets are tokens 1..T-1
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            te_logits = model(X_te[:, :-1])
            te_targets = X_te[:, 1:]
            te_loss = F.cross_entropy(te_logits.reshape(-1, VOCAB), te_targets.reshape(-1)).item()
            te_acc = (te_logits.argmax(-1) == te_targets).float().mean().item()

        hist['loss'].append(te_loss)
        hist['acc'].append(te_acc)

        if ep % 50 == 0 or ep == epochs - 1:
            print(f"    ep {ep:4d}  loss={te_loss:.4f}  acc={te_acc:.3f}")

    return hist


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

tasks = {
    'bigram': make_bigram_data,
    'repetition': make_repetition_data,
    'grammar (N-V-N-V)': make_grammar_data,
}

all_results = {}

for task_name, task_fn in tasks.items():
    print()
    print("=" * 70)
    print(f"TASK: {task_name}")
    print("=" * 70)

    torch.manual_seed(42)
    X_tr = task_fn(3000)
    X_te = task_fn(500)

    models = {
        'Condensation LM':       CondensationLM(field_dim=32, n_substeps=3).to(DEVICE),
        'CM (no depletion)':     CondensationLM_NoDepletion(field_dim=32, n_substeps=3).to(DEVICE),
        'GRU':                   GRUBaseline(hidden=32).to(DEVICE),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  {name} ({nparams(model)} params):")
        hist = train_lm(model, X_tr, X_te, epochs=300)
        results[name] = {'acc': hist['acc'][-1], 'loss': hist['loss'][-1], 'params': nparams(model)}

    print(f"\n  {'Model':<25} {'Acc':>6}  {'Loss':>7}  {'Params':>7}")
    print(f"  {'─'*25} {'─'*6}  {'─'*7}  {'─'*7}")
    for name, r in results.items():
        print(f"  {name:<25} {r['acc']:6.3f}  {r['loss']:7.4f}  {r['params']:7d}")

    all_results[task_name] = results

    # Dynamics for condensation LM
    cm = [m for n, m in models.items() if n == 'Condensation LM'][0]
    cm.eval()
    with torch.no_grad():
        logits, phi_norms, depletions = cm(X_te[:100, :-1], return_dynamics=True)

    print(f"\n  Field dynamics (condensation LM):")
    print(f"  {'Pos':>4}  {'|phi|':>7}  {'|depl|':>7}  {'net':>7}")
    for t in range(min(len(phi_norms), 12)):
        net = phi_norms[t] - (phi_norms[t-1] if t > 0 else 0)
        print(f"  {t:4d}  {phi_norms[t]:7.3f}  {depletions[t]:7.3f}  {net:+7.3f}")


# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

for task_name, results in all_results.items():
    cm_acc = results['Condensation LM']['acc']
    no_depl = results['CM (no depletion)']['acc']
    gru_acc = results['GRU']['acc']
    print(f"  {task_name}:")
    print(f"    CM={cm_acc:.3f}  CM(no depl)={no_depl:.3f}  GRU={gru_acc:.3f}")
    if cm_acc > no_depl + 0.02:
        print(f"    >> Depletion helps (+{cm_acc - no_depl:.3f})")
    if cm_acc > gru_acc - 0.03:
        print(f"    >> CM competitive with GRU")
    print()

print("  The condensation LM predicts next tokens by physics:")
print("  tokens deposit -> field accumulates -> prediction nucleates -> field depletes")
print("  If depletion helps: the WBF process is a memory mechanism.")
print("  If CM matches GRU: physics-based context is viable for language.")
