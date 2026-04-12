"""Diagnose whether the social field is doing anything interesting.

Three tests:
1. Weight magnitude: have w_mod/w_deposit moved from initialization?
2. Ablation: zero out field modulation, measure val loss change
3. Field dynamics: capture field state norms across layers/chunks on real text
"""
import os
import sys
import copy
import math
import torch

sys.path.insert(0, os.path.expanduser("~/owt"))
sys.path.insert(0, os.path.expanduser("~/tinystories"))

from data import get_tokenizer, get_dataloaders
from model import GPT

ckpt_path = os.path.expanduser("~/owt/checkpoints/owt_social_latest.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = get_tokenizer()

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
args = ckpt["args"]
step = ckpt.get("step", "?")
print(f"Checkpoint: step {step}, {args['d_model']}d, {args['n_layer']}L, {args['n_head']}H")
print(f"Field: use_field={args['use_field']}, mod_type={args['mod_type']}, evap={args['evap_rate']}")

model = GPT(
    vocab_size=enc.n_vocab,
    d_model=args["d_model"],
    n_head=args["n_head"],
    n_layer=args["n_layer"],
    dropout=0.0,
    max_len=args["context_length"],
    chunk_size=args["chunk_size"],
    evap_rate=args["evap_rate"],
    use_field=args["use_field"],
    mod_type=args["mod_type"],
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# =========================================================================
# TEST 1: Weight magnitudes — have field params moved from init?
# =========================================================================
print("\n" + "=" * 70)
print("TEST 1: Field weight magnitudes (vs init)")
print("=" * 70)

for i, block in enumerate(model.blocks):
    attn = block.attn
    if not attn.use_field:
        print(f"  Layer {i}: field disabled")
        continue

    w_dep_norm = attn.w_deposit.weight.data.norm().item()
    w_dep_std = attn.w_deposit.weight.data.std().item()

    if attn.mod_type == "additive":
        w_mod_norm = attn.w_mod.weight.data.norm().item()
        w_mod_std = attn.w_mod.weight.data.std().item()
        w_mod_max = attn.w_mod.weight.data.abs().max().item()
    else:
        w_mod_norm = w_mod_std = w_mod_max = 0.0

    # Compare w_qkv for scale reference
    w_qkv_std = attn.w_qkv.weight.data.std().item()

    print(f"  Layer {i:2d}:  w_deposit std={w_dep_std:.5f}  |  "
          f"w_mod std={w_mod_std:.5f} max={w_mod_max:.5f}  |  "
          f"w_qkv std={w_qkv_std:.5f}  |  "
          f"mod/qkv ratio={w_mod_std / (w_qkv_std + 1e-10):.3f}")

# =========================================================================
# TEST 2: Ablation — val loss with vs without field modulation
# =========================================================================
print("\n" + "=" * 70)
print("TEST 2: Ablation — val loss with field ON vs OFF")
print("=" * 70)

_, val_loader = get_dataloaders(
    batch_size=8, context_length=args["context_length"],
    num_workers=2, data_dir=os.path.expanduser("~/owt/.data"))

def eval_loss(mdl, n_batches=100):
    mdl.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = mdl(x, y)
            total += loss.item()
            n += 1
    return total / max(n, 1)

# Normal (field ON)
loss_on = eval_loss(model)
ppl_on = math.exp(loss_on)
print(f"  Field ON:  val_loss={loss_on:.4f}  ppl={ppl_on:.2f}")

# Ablate: zero out w_mod weights (field deposits still happen but don't affect keys)
model_ablated = copy.deepcopy(model)
for block in model_ablated.blocks:
    attn = block.attn
    if attn.use_field and attn.mod_type == "additive":
        attn.w_mod.weight.data.zero_()
    elif attn.use_field and attn.mod_type == "gating":
        attn.w_gate.weight.data.zero_()
        attn.w_gate.bias.data.fill_(100.0)  # sigmoid(100) ~ 1.0, no gating

loss_off = eval_loss(model_ablated)
ppl_off = math.exp(loss_off)
delta_ppl = ppl_off - ppl_on
pct = 100 * delta_ppl / ppl_on
print(f"  Field OFF: val_loss={loss_off:.4f}  ppl={ppl_off:.2f}")
print(f"  Delta:     +{delta_ppl:.2f} ppl ({pct:+.1f}%)")
if pct > 1.0:
    print(f"  >>> Field is contributing meaningfully ({pct:.1f}% PPL increase when ablated)")
elif pct > 0.1:
    print(f"  >>> Field has a small but measurable effect ({pct:.1f}%)")
else:
    print(f"  >>> Field effect is negligible so far ({pct:.2f}%)")

del model_ablated
torch.cuda.empty_cache()

# =========================================================================
# TEST 3: Field dynamics — norms across layers and chunks
# =========================================================================
print("\n" + "=" * 70)
print("TEST 3: Field dynamics on sample text")
print("=" * 70)

text = """The development of artificial intelligence has been one of the most \
transformative technological shifts in human history. Beginning with early \
symbolic systems in the 1950s, the field underwent several winters of reduced \
funding and interest before the deep learning revolution of the 2010s brought \
unprecedented breakthroughs in computer vision, natural language processing, \
and game playing. Today, large language models trained on vast corpora of text \
can generate remarkably coherent prose, translate between languages, and even \
write code, raising profound questions about the nature of understanding and \
the future of human labor."""

tokens = enc.encode(text)
idx = torch.tensor([tokens], dtype=torch.long, device=device)
print(f"  Input: {len(tokens)} tokens, {len(tokens) // args['chunk_size']} full chunks")

# Enable capture on all attention layers
for block in model.blocks:
    block.attn._capture = True

with torch.no_grad():
    _ = model(idx)

print(f"\n  {'Layer':>5} | {'Chunk':>5} | {'Field L2':>10} | {'Field Max':>10} | {'Deposit L2':>10}")
print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for li, block in enumerate(model.blocks):
    attn = block.attn
    if not hasattr(attn, '_field_states'):
        continue

    n_states = len(attn._field_states)
    for ci in range(n_states):
        fs = attn._field_states[ci]  # (B, H, d_f)
        fs_norm = fs.norm(dim=-1).mean().item()
        fs_max = fs.abs().max().item()

        if ci < len(attn._deposits):
            dep = attn._deposits[ci]  # (B, C, H, d_f)
            dep_norm = dep.norm(dim=-1).mean().item()
        else:
            dep_norm = 0.0

        print(f"  {li:5d} | {ci:5d} | {fs_norm:10.4f} | {fs_max:10.4f} | {dep_norm:10.4f}")

# Per-layer summary: field magnitude vs key magnitude
print(f"\n  Field modulation strength per layer (field shift vs key norm):")
print(f"  {'Layer':>5} | {'Key L2':>10} | {'Shift L2':>10} | {'Ratio':>8}")
print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

# Rerun to capture key stats
for li, block in enumerate(model.blocks):
    attn = block.attn
    if not attn.use_field or not hasattr(attn, '_field_states'):
        continue

    # Get the last field state (most accumulated)
    last_field = attn._field_states[-1].to(device)  # (B, H, d_f)
    if attn.mod_type == "additive":
        k_shift = attn.w_mod(last_field)  # (B, H, d)
        shift_norm = k_shift.norm(dim=-1).mean().item()
    else:
        shift_norm = 0.0

    # Get typical key norm from a forward pass
    x_sample = model.tok_emb(idx)
    for bi, b in enumerate(model.blocks):
        if bi < li:
            x_sample = b(x_sample)
    x_ln = b.ln1(x_sample)
    qkv = attn.w_qkv(x_ln).reshape(1, len(tokens), 3, attn.n_head, attn.head_dim)
    k_vals = qkv[:, :, 1, :, :]  # (B, T, H, d)
    k_norm = k_vals.norm(dim=-1).mean().item()

    ratio = shift_norm / (k_norm + 1e-10)
    print(f"  {li:5d} | {k_norm:10.4f} | {shift_norm:10.4f} | {ratio:8.4f}")

# Cleanup
for block in model.blocks:
    block.attn._capture = False
    if hasattr(block.attn, '_field_states'):
        del block.attn._field_states
    if hasattr(block.attn, '_deposits'):
        del block.attn._deposits

print("\nDone.")
