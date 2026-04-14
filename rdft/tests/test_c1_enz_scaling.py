"""
Enzymatic full at dim=256: does the gap stay constant with scale?

At dim=128 (enz full vs GRU): gap = 0.055 nats at ~matched params.
At dim=256: v2b gap was 0.236 nats. Does enz full keep it at ~0.05?
If yes, C1 (properly stated: gap at matched params is bounded with scale)
is supported. If the gap widens, gating wins at scale.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, functools

print = functools.partial(print, flush=True)
torch.manual_seed(42)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

def load_tinystories(max_chars=500_000, seq_len=128):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    text = []; total = 0
    for ex in ds:
        text.append(ex['text']); total += len(ex['text'])
        if total > max_chars: break
    corpus = "\n".join(text)[:max_chars]
    cc = {}
    for c in corpus: cc[c] = cc.get(c, 0) + 1
    common = sorted([c for c, n in cc.items() if n > 50])
    c2i = {c: i+1 for i, c in enumerate(common)}; c2i['<unk>'] = 0
    i2c = {i: c for c, i in c2i.items()}
    enc = torch.tensor([c2i.get(c, 0) for c in corpus], dtype=torch.long)
    n = len(enc) // seq_len
    d = enc[:n*seq_len].view(n, seq_len)
    d = d[torch.randperm(n)]
    n_tr = int(0.9 * n)
    return d[:n_tr].to(DEVICE), d[n_tr:].to(DEVICE), len(c2i), i2c


class v2b_enz_full(nn.Module):
    def __init__(self, vocab, field_dim=128, n_substeps=2, dt=0.2, rank=4):
        super().__init__()
        self.vocab = vocab; self.field_dim = field_dim
        self.n_substeps = n_substeps; self.dt = dt; self.rank = rank
        self.embed = nn.Embedding(vocab, field_dim)
        self.mix = nn.Linear(field_dim, field_dim, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.1))
        self.nonlin_strength = nn.Parameter(torch.tensor(0.3))
        self.enz_decay = nn.Embedding(vocab, field_dim)
        self.enz_nonlin = nn.Embedding(vocab, field_dim)
        self.enz_shift = nn.Embedding(vocab, field_dim)
        self.enz_mix_u = nn.Embedding(vocab, rank * field_dim)
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
            dm = torch.sigmoid(self.enz_decay(x[:, t]))
            nm = torch.sigmoid(self.enz_nonlin(x[:, t]))
            sh = self.enz_shift(x[:, t])
            u = self.enz_mix_u(x[:, t]).view(B, self.rank, self.field_dim)
            v = self.enz_mix_v(x[:, t]).view(B, self.rank, self.field_dim)
            for _ in range(self.n_substeps):
                mu = F.softplus(self.decay) * (0.2 + 1.6 * dm)
                nl = F.softplus(self.nonlin_strength) * (0.2 + 1.6 * nm)
                phi_c = (phi - sh).clamp(-3, 3)
                scores = torch.einsum('brf,bf->br', v, phi)
                delta = torch.einsum('brf,br->bf', u, scores)
                dphi = self.mix(phi) + delta - mu * phi - nl * phi_c * (phi_c**2 - 1)
                phi = (phi + self.dt * dphi).clamp(-5, 5)
            logits = self.nucleate(phi); all_logits.append(logits)
            ps = F.softmax(logits.detach(), dim=-1)
            phi = phi - self.deplete(ps)
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
        tot = 0; nb = 0
        for s in range(0, min(n, 2000), bs):
            idx = perm[s:s+bs]; xb = X_tr[idx]
            logits = model(xb[:, :-1]); targets = xb[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            tl = model(X_te[:200, :-1])
            tt = X_te[:200, 1:]
            te_loss = F.cross_entropy(tl.reshape(-1, vocab), tt.reshape(-1)).item()
            te_acc = (tl.argmax(-1) == tt).float().mean().item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    ep {ep:3d}  train={tot/nb:.3f}  test={te_loss:.3f}  acc={te_acc:.3f}")
    return te_loss, te_acc


def nparams(m): return sum(p.numel() for p in m.parameters())

X_tr, X_te, V, i2c = load_tinystories()
print(f"Vocab: {V}\n")

print("=" * 70)
print("enz_full at dim=256")
print("=" * 70)
torch.manual_seed(42)
m = v2b_enz_full(V, field_dim=256).to(DEVICE)
print(f"  params: {nparams(m):,}")
t0 = time.time()
el, ea = train_lm(m, X_tr, X_te, V, epochs=20)
print(f"  time: {time.time()-t0:.0f}s")

print()
print("=" * 70)
print("GRU at dim=256")
print("=" * 70)
torch.manual_seed(42)
g = GRULM(V, hidden=256).to(DEVICE)
print(f"  params: {nparams(g):,}")
t0 = time.time()
gl, ga = train_lm(g, X_tr, X_te, V, epochs=20)
print(f"  time: {time.time()-t0:.0f}s")

print()
print("=" * 70)
print("SCALING RESULTS")
print("=" * 70)
print(f"\n  dim=128 (previous):")
print(f"    enz_full: loss=1.396, params=123,966")
print(f"    GRU:      loss=1.340, params=114,492")
print(f"    gap: 0.056")
print(f"\n  dim=256 (this run):")
print(f"    enz_full: loss={el:.3f}, params={nparams(m):,}")
print(f"    GRU:      loss={gl:.3f}, params={nparams(g):,}")
print(f"    gap: {el - gl:.3f}")
print()
old_gap = 0.056
new_gap = el - gl
if new_gap < old_gap + 0.02:
    print(f"  >> Gap stable or narrowing ({old_gap:.3f} -> {new_gap:.3f})")
    print(f"     Enzymatic physics scales as well as GRU gating.")
elif new_gap > old_gap + 0.05:
    print(f"  >> Gap WIDENING ({old_gap:.3f} -> {new_gap:.3f})")
    print(f"     Gating still more expressive at scale.")
else:
    print(f"  >> Gap roughly constant ({old_gap:.3f} -> {new_gap:.3f})")
