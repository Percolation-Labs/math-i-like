"""Probes designed to test what the stigmergic field SHOULD be good at.

Standard perplexity averages over all positions equally. But the field only
modulates keys from chunk 1 onwards (positions 128+), and its theory says it
should help with sequential state accumulation and cross-chunk dependencies.

Three probes:
1. Per-chunk ablation: break val loss by chunk position (0-127, 128-255, ...)
   to see if the field helps more in later chunks
2. Entity tracking: passages where a named entity is introduced early and
   referenced later -- measure prediction accuracy at the reference point
3. State tracking: constructed sequences with cumulative state that must be
   tracked across chunk boundaries
"""
import copy
import math
import os
import sys
import torch
import torch.nn.functional as F

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
CS = args["chunk_size"]  # 128

print(f"Checkpoint: step {step}")
print(f"Chunk size: {CS}, context: {args['context_length']}")

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

model_ablated = copy.deepcopy(model)
for block in model_ablated.blocks:
    attn = block.attn
    if attn.use_field and attn.mod_type == "additive":
        attn.w_mod.weight.data.zero_()

# =========================================================================
# PROBE 1: Per-chunk-position ablation
# =========================================================================
print("\n" + "=" * 70)
print("PROBE 1: Per-chunk ablation — field ON vs OFF by position in sequence")
print("=" * 70)
print(f"Chunk boundaries: 0, {CS}, {2*CS}, {3*CS}, ...")

_, val_loader = get_dataloaders(
    batch_size=8, context_length=args["context_length"],
    num_workers=2, data_dir=os.path.expanduser("~/owt/.data"))

n_chunks = args["context_length"] // CS
chunk_loss_on = [0.0] * n_chunks
chunk_loss_off = [0.0] * n_chunks
chunk_count = [0] * n_chunks
n_batches = 200

with torch.no_grad():
    for i, (x, y) in enumerate(val_loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_on = model(x)
            logits_off = model_ablated(x)

        for ci in range(n_chunks):
            s = ci * CS
            e = min((ci + 1) * CS, args["context_length"])

            loss_on = F.cross_entropy(
                logits_on[:, s:e, :].reshape(-1, logits_on.size(-1)),
                y[:, s:e].reshape(-1), reduction="mean").item()
            loss_off = F.cross_entropy(
                logits_off[:, s:e, :].reshape(-1, logits_off.size(-1)),
                y[:, s:e].reshape(-1), reduction="mean").item()

            chunk_loss_on[ci] += loss_on
            chunk_loss_off[ci] += loss_off
            chunk_count[ci] += 1

print(f"\n{'Chunk':>6} | {'Positions':>12} | {'PPL ON':>8} | {'PPL OFF':>8} | {'ΔPPL':>8} | {'Δ%':>7}")
print(f"{'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

for ci in range(n_chunks):
    avg_on = chunk_loss_on[ci] / chunk_count[ci]
    avg_off = chunk_loss_off[ci] / chunk_count[ci]
    ppl_on = math.exp(avg_on)
    ppl_off = math.exp(avg_off)
    delta = ppl_off - ppl_on
    pct = 100 * delta / ppl_on if ppl_on > 0 else 0

    s = ci * CS
    e = (ci + 1) * CS
    print(f"{ci:6d} | {s:>5}-{e-1:<5} | {ppl_on:8.2f} | {ppl_off:8.2f} | {delta:+8.2f} | {pct:+6.2f}%")

# =========================================================================
# PROBE 2: Entity tracking across chunk boundaries
# =========================================================================
print("\n" + "=" * 70)
print("PROBE 2: Entity tracking — does the field help predict references")
print("         to entities introduced in earlier chunks?")
print("=" * 70)

entity_probes = [
    {
        "name": "Name recall (1 chunk gap)",
        "text": "Dr. Evelyn Hartwick had spent thirty years studying the migration patterns of Arctic terns. Her research station on the coast of Norway was modest but well-equipped. Every spring she would count the birds as they arrived from their incredible journey across hemispheres. One morning, she noticed something unusual in the data. The migration had started two weeks earlier than any previous year on record. She immediately called her colleague in London to discuss the implications. According to",
        "targets": [" Dr", " Evelyn", " Hart"],
    },
    {
        "name": "Location recall (2 chunk gap)",
        "text": "The ancient library of Thessaloniki contained manuscripts dating back to the Byzantine Empire. Scholars from across Europe traveled there to study the collection, which included rare texts on mathematics, philosophy, and natural history. The head librarian, a quiet man named Stavros, had catalogued every item by hand over the course of forty years. When the digitization project finally began, the team from the university set up their equipment in the main reading room. They scanned each page carefully, preserving centuries of knowledge. The project took three years to complete. When it was finally done, the digital archive was hosted on servers in the same city where the originals were kept, in",
        "targets": [" Thess", " the", " Greece"],
    },
    {
        "name": "Character state tracking",
        "text": "Maria left her apartment carrying a red umbrella and a bag of groceries. She walked three blocks to the park, where she sat on a bench and ate an apple from the bag. A sudden gust of wind pulled the umbrella from her hand and sent it tumbling across the grass. She chased it but it blew into the pond. Sighing, she returned to the bench and finished her lunch. When she stood up to leave, she realized she no longer had her",
        "targets": [" umbrella", " red"],
    },
    {
        "name": "Causal chain (long range)",
        "text": "The factory had been dumping chemicals into the river for years before anyone noticed. The first sign was when fish began dying downstream near the village of Millbrook. Local fishermen reported strange discoloration in the water. Environmental inspectors were called in, and after months of testing, they traced the contamination back upstream to the Grayson Chemical plant. The company denied responsibility initially but internal documents later revealed they had known about the leaks since the plant opened. The resulting lawsuit was filed by the residents of the village that had been most affected, which was",
        "targets": [" Mill", " the"],
    },
]

print(f"\nFor each probe, comparing rank of target tokens with field ON vs OFF.\n")

for probe in entity_probes:
    tokens = enc.encode(probe["text"])
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    chunk_of_last = len(tokens) // CS

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_on = model(idx)[:, -1, :]
        logits_off = model_ablated(idx)[:, -1, :]

    probs_on = F.softmax(logits_on.float(), dim=-1)[0]
    probs_off = F.softmax(logits_off.float(), dim=-1)[0]

    print(f"  {probe['name']} ({len(tokens)} tokens, chunk {chunk_of_last}):")

    for target in probe["targets"]:
        tid = enc.encode(target)
        if len(tid) == 0:
            continue
        tid = tid[0]

        rank_on = (probs_on > probs_on[tid]).sum().item() + 1
        rank_off = (probs_off > probs_off[tid]).sum().item() + 1
        prob_on = probs_on[tid].item()
        prob_off = probs_off[tid].item()

        indicator = ""
        if rank_on < rank_off:
            indicator = " ← field helps"
        elif rank_off < rank_on:
            indicator = " ← field hurts"

        print(f"    target '{target}' (token {tid}): "
              f"rank {rank_on:>5} (p={prob_on:.4f}) ON  vs  "
              f"rank {rank_off:>5} (p={prob_off:.4f}) OFF{indicator}")
    print()

# =========================================================================
# PROBE 3: Synthetic state tracking across chunks
# =========================================================================
print("=" * 70)
print("PROBE 3: Counting / state tracking in natural language")
print("=" * 70)

counting_probes = [
    {
        "name": "Enumeration recall",
        "text": "There are five key principles of effective leadership. First, a good leader must communicate clearly with their team. Second, they must be able to delegate tasks appropriately. Third, they should lead by example and demonstrate the values they expect from others. Fourth, they need to be adaptable and open to change when circumstances require it. And fifth, they must",
        "question": "Model should predict something about the 5th principle",
    },
    {
        "name": "Negation state",
        "text": "The committee voted on the proposal. Johnson voted yes. Smith voted no. Williams voted yes. Brown voted yes. Davis voted no. Miller voted yes. Wilson voted no. The final count was five votes to three in favor. However, since the rules required a two-thirds majority, the proposal was",
        "targets": [" rejected", " not", " defeated", " denied"],
    },
    {
        "name": "Temporal sequence",
        "text": "The experiment consisted of five phases. In phase one, the solution was heated to 100 degrees. In phase two, the catalyst was added and the mixture turned blue. In phase three, the temperature was reduced to 50 degrees and the color changed to green. In phase four, a second reagent was introduced, causing the solution to become clear. In the final phase, the solution was cooled to room temperature. At the end of the experiment, the color of the solution was",
        "targets": [" clear", " transparent", " color"],
    },
]

for probe in counting_probes:
    tokens = enc.encode(probe["text"])
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_on = model(idx)[:, -1, :]
        logits_off = model_ablated(idx)[:, -1, :]

    probs_on = F.softmax(logits_on.float(), dim=-1)[0]
    probs_off = F.softmax(logits_off.float(), dim=-1)[0]

    print(f"\n  {probe['name']} ({len(tokens)} tokens, chunk {len(tokens)//CS}):")

    # Show top-5 predictions for both
    top_on = torch.topk(probs_on, 10)
    top_off = torch.topk(probs_off, 10)

    print(f"    Top 5 predictions (field ON):  ", end="")
    for j in range(5):
        tok = enc.decode([top_on.indices[j].item()])
        print(f"'{tok}'({top_on.values[j]:.3f})", end="  ")
    print()

    print(f"    Top 5 predictions (field OFF): ", end="")
    for j in range(5):
        tok = enc.decode([top_off.indices[j].item()])
        print(f"'{tok}'({top_off.values[j]:.3f})", end="  ")
    print()

    if "targets" in probe:
        for target in probe["targets"]:
            tid = enc.encode(target)
            if len(tid) == 0:
                continue
            tid = tid[0]
            rank_on = (probs_on > probs_on[tid]).sum().item() + 1
            rank_off = (probs_off > probs_off[tid]).sum().item() + 1
            prob_on = probs_on[tid].item()
            prob_off = probs_off[tid].item()
            indicator = ""
            if rank_on < rank_off:
                indicator = " ← field helps"
            elif rank_off < rank_on:
                indicator = " ← field hurts"
            print(f"    target '{target}': rank {rank_on:>5} (p={prob_on:.4f}) ON vs "
                  f"rank {rank_off:>5} (p={prob_off:.4f}) OFF{indicator}")

print("\n\nDone.")
