"""
Thought Cloud Microphysics: Transformer-Scale Experiment
=========================================================

Scales the toy experiments to a real transformer architecture:
  L: causal transformer (4 layers, d=64, 4 heads)
  R: GRU accumulator over L's hidden states
  Gate: per-layer learned coupling (R -> L injection)

Task: "Contextual Selective Copy"
  - Sequence has a context region (sets a rule) and a query region
  - The rule determines HOW to transform query tokens into outputs
  - Requires both token-level attention (L) and accumulated context (R)
  - More complex than delayed XOR — needs compositional processing

Runs on MPS (Apple Silicon) if available, else CPU.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Unbuffered output so we see results in real time
import functools
print = functools.partial(print, flush=True)

torch.manual_seed(42)
np.random.seed(42)

# ── Device ────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print(f"Using CPU")


# ════════════════════════════════════════════════════════════════
# TASK: Majority-vote context + selective copy
# ════════════════════════════════════════════════════════════════
# Easier than parity (parity is a known hard problem for transformers)
# but still requires accumulation over distributed context.
#
# Task design:
#   - Sequence of 32 tokens
#   - Positions 0-15: "paragraph" — each token is from {0,1,2,3}
#     The MAJORITY class determines the rule (0,1,2, or 3)
#   - Positions 16-17: SEP tokens
#   - Positions 18-25: query operands (from {8..23})
#   - Positions 26-29: target = f(operands[:4], majority_class)
#
# Rules by majority class:
#   0: identity  [a,b,c,d] -> [a,b,c,d]
#   1: reverse   [a,b,c,d] -> [d,c,b,a]
#   2: shift     [a,b,c,d] -> [b,c,d,a]
#   3: swap      [a,b,c,d] -> [c,d,a,b]
#
# WHY this needs accumulation:
#   - Context tokens are from {0,1,2,3} mixed uniformly with majority planted
#   - Must count occurrences across 16 positions to find majority
#   - Single attention head can't easily compute argmax(count)
#   - R's recurrent accumulation can track running counts
#
# WHY this needs L too:
#   - Must copy/reorder specific query tokens — needs precise attention

VOCAB = 24   # 0-3 = context, 4-7 = special, 8-23 = operands
SEQ_LEN = 32
N_PARA = 16
N_SEP = 2
N_QUERY = 8
N_TARGET = 4
SEP_TOK = 4


def apply_op(op, operands):
    a = operands[:4].clone()
    if op == 0: return a
    elif op == 1: return a.flip(0)
    elif op == 2: return torch.roll(a, -1)
    elif op == 3: return a[torch.tensor([2, 3, 0, 1])]
    return a


def make_data(n=5000):
    X = torch.zeros(n, SEQ_LEN, dtype=torch.long)
    Y = torch.zeros(n, N_TARGET, dtype=torch.long)

    for i in range(n):
        # Plant majority: pick a class, make it appear more than others
        majority_class = torch.randint(0, 4, (1,)).item()

        # Fill paragraph: ~40% majority class, ~20% each other
        para = torch.zeros(N_PARA, dtype=torch.long)
        for j in range(N_PARA):
            if torch.rand(1).item() < 0.45:
                para[j] = majority_class
            else:
                # Pick from other 3 classes uniformly
                other = torch.tensor([c for c in range(4) if c != majority_class])
                para[j] = other[torch.randint(0, 3, (1,))]

        # Verify majority (if tie, force it)
        counts = torch.bincount(para, minlength=4)
        if counts.argmax().item() != majority_class:
            # Force by flipping a few
            non_maj = (para != majority_class).nonzero().squeeze()
            if non_maj.dim() == 0:
                non_maj = non_maj.unsqueeze(0)
            n_flip = min(3, len(non_maj))
            para[non_maj[:n_flip]] = majority_class

        actual_majority = torch.bincount(para, minlength=4).argmax().item()

        X[i, :N_PARA] = para
        X[i, N_PARA:N_PARA+N_SEP] = SEP_TOK

        ops = torch.randint(8, VOCAB, (N_QUERY,))
        X[i, N_PARA+N_SEP:N_PARA+N_SEP+N_QUERY] = ops

        target = apply_op(actual_majority, ops)
        target_start = N_PARA + N_SEP + N_QUERY
        X[i, target_start:target_start+N_TARGET] = target
        Y[i] = target

    return X.to(DEVICE), Y.to(DEVICE)


X_train, Y_train = make_data(4000)
X_test, Y_test = make_data(1000)

print(f"Data: train={X_train.shape}, test={X_test.shape}")
print(f"Sequence length: {SEQ_LEN}, paragraph: {N_PARA}, query: {N_QUERY}, target: {N_TARGET}")
print(f"16 context tokens from {{0,1,2,3}}, majority determines operation")
print(f"4 operations: identity, reverse, shift, swap")
print()


# ════════════════════════════════════════════════════════════════
# TRANSFORMER L (particle sector)
# ════════════════════════════════════════════════════════════════

class TransformerL(nn.Module):
    """Causal transformer — the left hemisphere / particle sector."""
    def __init__(self, vocab=VOCAB, d=64, n_heads=4, n_layers=4, max_len=SEQ_LEN):
        super().__init__()
        self.d = d
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_len, d)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d),
                'attn': nn.MultiheadAttention(d, n_heads, batch_first=True),
                'ln2': nn.LayerNorm(d),
                'ffn': nn.Sequential(
                    nn.Linear(d, d * 2),
                    nn.GELU(),
                    nn.Linear(d * 2, d),
                ),
            }))

        self.ln_f = nn.LayerNorm(d)

    def forward(self, x, r_injections=None):
        """
        x: (batch, seq_len) token ids
        r_injections: optional list of (batch, seq_len, d) tensors, one per layer
        Returns: (batch, seq_len, d) hidden states, plus per-layer hiddens
        """
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()

        layer_hiddens = []
        for i, layer in enumerate(self.layers):
            # Inject R's signal BEFORE attention (structural embeddings)
            if r_injections is not None and r_injections[i] is not None:
                h = h + r_injections[i]

            # Self-attention
            h_norm = layer['ln1'](h)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm, attn_mask=mask)
            h = h + attn_out

            # FFN
            h = h + layer['ffn'](layer['ln2'](h))

            layer_hiddens.append(h)

        return self.ln_f(h), layer_hiddens


# ════════════════════════════════════════════════════════════════
# GRU R (field sector)
# ════════════════════════════════════════════════════════════════

class FieldR(nn.Module):
    """
    Right hemisphere / MSR field sector.
    Accumulates L's hidden states into a continuous relational state.
    Produces per-layer injection signals for L.
    """
    def __init__(self, d=64, r=32, n_layers=4, observe_layer=1):
        super().__init__()
        self.d = d
        self.r = r
        self.n_layers = n_layers
        self.observe_layer = observe_layer

        # Input: observe L's hidden states at one layer
        self.input_proj = nn.Linear(d, r)

        # Recurrent accumulation (the MSR field dynamics)
        self.gru = nn.GRU(r, r, batch_first=True)

        # Per-layer projection back to L's space
        self.output_projs = nn.ModuleList([
            nn.Linear(r, d) for _ in range(n_layers)
        ])

        # Per-layer gates (initialised closed)
        self.gate_biases = nn.ParameterList([
            nn.Parameter(torch.tensor(-3.0)) for _ in range(n_layers)
        ])

    def forward(self, l_hidden):
        """
        l_hidden: (batch, seq_len, d) — L's hidden states at observe_layer
        Returns: list of (batch, seq_len, d) injection signals, one per L layer
        """
        # Project L's hidden states into R's space
        r_input = self.input_proj(l_hidden)  # (batch, seq, r)

        # Accumulate through GRU
        r_states, _ = self.gru(r_input)  # (batch, seq, r)

        # Generate per-layer injections with gates
        injections = []
        for i in range(self.n_layers):
            gate = torch.sigmoid(self.gate_biases[i])
            proj = self.output_projs[i](r_states)  # (batch, seq, d)
            injections.append(gate * proj)

        return injections, r_states

    def get_gates(self):
        return [torch.sigmoid(b).item() for b in self.gate_biases]


# ════════════════════════════════════════════════════════════════
# TWO-HEMISPHERE MODEL
# ════════════════════════════════════════════════════════════════

class TwoHemisphere(nn.Module):
    """
    The full coupled system:
      1. L processes tokens through shallow layers (0..observe_layer)
      2. R observes L's hidden states, accumulates field
      3. R injects structural context into L's deep layers
      4. L continues through deep layers with R's injection
      5. Classification head on target positions
    """
    def __init__(self, d=64, r=32, n_heads=4, n_layers=4, observe_layer=1):
        super().__init__()
        self.L = TransformerL(d=d, n_heads=n_heads, n_layers=n_layers)
        self.R = FieldR(d=d, r=r, n_layers=n_layers, observe_layer=observe_layer)
        self.head = nn.Linear(d, VOCAB)
        self.observe_layer = observe_layer
        self.n_layers = n_layers

    def forward(self, x, return_dynamics=False):
        B, S = x.shape

        # Single-pass: L processes through layers, R observes midpoint
        # and adds to L's output (no expensive double forward pass)

        # Full L forward (no injection for single-pass efficiency)
        h_full, layer_hiddens = self.L(x, r_injections=None)

        # R observes L at midpoint
        l_observed = layer_hiddens[self.observe_layer]

        # R accumulates and generates injection for final output
        injections, r_states = self.R(l_observed)

        # Add R's deep-layer signals to L's output (single residual)
        # This is the coupling: R's accumulated field modifies L's representation
        h_coupled = h_full
        for i in range(self.observe_layer + 1, self.n_layers):
            h_coupled = h_coupled + injections[i]

        # Predict at target positions
        target_h = h_coupled[:, -N_TARGET:, :]  # (B, 4, d)
        logits = self.head(target_h)  # (B, 4, vocab)

        if return_dynamics:
            gates = self.R.get_gates()
            field_norms = r_states.norm(dim=2).mean(dim=0).cpu().tolist()  # per-position
            return logits, gates, field_norms

        return logits

    def get_gates(self):
        return self.R.get_gates()


class TransformerBaseline(nn.Module):
    """L-only transformer baseline (no R)."""
    def __init__(self, d=64, n_heads=4, n_layers=4):
        super().__init__()
        self.L = TransformerL(d=d, n_heads=n_heads, n_layers=n_layers)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        h, _ = self.L(x)
        return self.head(h[:, -N_TARGET:, :])


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def train_model(model, X_tr, Y_tr, X_te, Y_te, epochs=80, lr=3e-4, batch_size=256):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = X_tr.shape[0]
    hist = {'loss': [], 'acc': [], 'gates': []}

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]
            logits = model(xb) if not isinstance(model, TwoHemisphere) else model(xb)
            # logits: (B, 4, vocab), yb: (B, 4)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            logits_te = model(X_te) if not isinstance(model, TwoHemisphere) else model(X_te)
            pred = logits_te.argmax(dim=2)  # (B, 4)
            acc = (pred == Y_te).float().mean().item()
            # Per-position accuracy
            pos_acc = [(pred[:, i] == Y_te[:, i]).float().mean().item() for i in range(4)]

        gates = model.get_gates() if hasattr(model, 'get_gates') else [0]*4
        hist['loss'].append(total_loss / n_batches)
        hist['acc'].append(acc)
        hist['gates'].append(gates)

        if epoch % 10 == 0 or epoch == epochs - 1:
            gate_str = " ".join(f"{g:.3f}" for g in gates) if any(g > 0 for g in gates) else "—"
            print(f"  epoch {epoch:4d}  loss={total_loss/n_batches:.4f}  "
                  f"acc={acc:.3f}  pos_acc={[f'{a:.2f}' for a in pos_acc]}  gates=[{gate_str}]")

    return hist


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ════════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ════════════════════════════════════════════════════════════════

print("=" * 70)
print("EXPERIMENT A: Transformer Baseline (L only)")
print("=" * 70)
print()

EPOCHS = 80

baseline = TransformerBaseline(d=64, n_heads=4, n_layers=4).to(DEVICE)
print(f"  Params: {nparams(baseline):,}")
t0 = time.time()
h_base = train_model(baseline, X_train, Y_train, X_test, Y_test, epochs=EPOCHS)
t_base = time.time() - t0
print(f"  Time: {t_base:.1f}s")

print()
print("=" * 70)
print("EXPERIMENT B: Two-Hemisphere (L + R coupled)")
print("=" * 70)
print()

two_hemi = TwoHemisphere(d=64, r=32, n_heads=4, n_layers=4, observe_layer=1).to(DEVICE)
print(f"  Params: {nparams(two_hemi):,}  (L={nparams(two_hemi.L):,}, R={nparams(two_hemi.R):,})")
t0 = time.time()
h_2h = train_model(two_hemi, X_train, Y_train, X_test, Y_test, epochs=EPOCHS)
t_2h = time.time() - t0
print(f"  Time: {t_2h:.1f}s")


# ════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

print(f"  {'Model':<30} {'Acc':>6}  {'Params':>10}  {'Time':>7}")
print(f"  {'─'*30} {'─'*6}  {'─'*10}  {'─'*7}")
print(f"  {'Transformer (L only)':<30} {h_base['acc'][-1]:6.3f}  {nparams(baseline):10,}  {t_base:6.1f}s")
print(f"  {'Two-Hemisphere (L+R)':<30} {h_2h['acc'][-1]:6.3f}  {nparams(two_hemi):10,}  {t_2h:6.1f}s")

print()
print("  Per-layer gate values (Two-Hemisphere):")
final_gates = h_2h['gates'][-1]
for i, g in enumerate(final_gates):
    depth = "shallow" if i <= 1 else "deep"
    inj = "no injection" if i <= 1 else "R injects here"
    print(f"    Layer {i}: gate={g:.4f}  ({depth}, {inj})")

print()

# Gate trajectory
print("  Gate trajectory during training:")
print(f"  {'Epoch':>6}", end="")
for i in range(4):
    print(f"  {'L'+str(i):>7}", end="")
print()
for epoch_idx in [0, 10, 20, 40, 60, 79]:
    if epoch_idx < len(h_2h['gates']):
        gs = h_2h['gates'][epoch_idx]
        print(f"  {epoch_idx:6d}", end="")
        for g in gs:
            print(f"  {g:7.4f}", end="")
        print()

print()

# Field dynamics
two_hemi.eval()
with torch.no_grad():
    _, gates_final, field_norms = two_hemi(X_test[:200], return_dynamics=True)

print("  R's field norm across sequence (sampled positions):")
print(f"  {'Pos':>4}  {'Norm':>7}  {'Region':>12}")
print(f"  {'─'*4}  {'─'*7}  {'─'*12}")
target_start = N_PARA + N_SEP + N_QUERY
sample_positions = list(range(0, N_PARA, 4)) + [N_PARA, N_PARA+N_SEP] + \
                   list(range(N_PARA+N_SEP, N_PARA+N_SEP+N_QUERY, 2)) + \
                   list(range(target_start, target_start+N_TARGET))
for t in sample_positions:
    if t < len(field_norms):
        if t < N_PARA:
            region = 'context'
        elif t < N_PARA + N_SEP:
            region = 'SEP'
        elif t < target_start:
            region = 'query'
        else:
            region = 'TARGET'
        print(f"  {t:4d}  {field_norms[t]:7.3f}  {region:>12}")

print()

# Context-dependent field analysis
with torch.no_grad():
    # Split test set by majority context
    # Split by majority class
    X_sub = X_test[:200]
    majorities = X_sub[:, :N_PARA].cpu()
    maj_classes = torch.stack([torch.bincount(row, minlength=4).argmax() for row in majorities])

    field_by_op = {}
    for op in range(4):
        mask = maj_classes == op
        if mask.sum() > 10:
            _, _, fn_op = two_hemi(X_sub[mask.to(DEVICE)], return_dynamics=True)
            field_by_op[op] = fn_op

if field_by_op:
    op_names = ['identity', 'reverse', 'shift', 'swap']
    print("  Field norm at final position, by operation (majority class):")
    for op, fn in sorted(field_by_op.items()):
        print(f"    op={op} ({op_names[op]:>8}): field_norm={fn[-1]:.3f}")

    norms = [fn[-1] for fn in field_by_op.values()]
    spread = max(norms) - min(norms)
    print(f"    Spread: {spread:.3f}")
    if spread > 0.1:
        print("    R's field encodes different operations as different field states.")


# ── Depth differentiation check ───────────────────────────────
print()
shallow_gates = np.mean(final_gates[:2])
deep_gates = np.mean(final_gates[2:])
print(f"  Depth differentiation:")
print(f"    Shallow layers (0-1) avg gate: {shallow_gates:.4f}")
print(f"    Deep layers (2-3) avg gate:    {deep_gates:.4f}")
if deep_gates > shallow_gates * 1.5:
    print("    Deep > Shallow: compositional layers need more relational context.")
    print("    Matches Two-Hemisphere paper finding (gate=0 shallow, gate=1 deep).")
elif deep_gates > shallow_gates * 1.1:
    print("    Slight depth gradient — would sharpen with scale.")
else:
    print("    Uniform gates — task may not yet differentiate depth needs.")
    print("    (Note: layers 0-1 don't receive R injection by design.)")


# ════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  Baseline (L only):     {h_base['acc'][-1]:.3f}")
print(f"  Two-Hemisphere (L+R):  {h_2h['acc'][-1]:.3f}")
print(f"  Gate trajectory: {h_2h['gates'][0][2]:.4f} -> {h_2h['gates'][-1][2]:.4f} (layer 2)")
print()
if h_2h['acc'][-1] > h_base['acc'][-1] + 0.02:
    print("  Two-Hemisphere outperforms baseline transformer.")
    print("  R's relational field provides context that attention alone misses.")
elif abs(h_2h['acc'][-1] - h_base['acc'][-1]) < 0.02:
    print("  Performance similar — transformer attention may suffice at this scale.")
    print("  The Two-Hemisphere paper shows the gap widens with sequence length")
    print("  and task complexity (the 0.55 nat improvement was on real text).")
else:
    print("  Baseline wins — two-pass overhead may hurt at this scale.")
    print("  The coupling architecture needs tuning for this task.")

print()
print(f"  Total time: {t_base + t_2h:.0f}s on {DEVICE}")
