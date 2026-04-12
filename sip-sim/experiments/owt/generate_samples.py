"""Generate sample text from the latest OWT checkpoint.

Runs on the Lightning AI studio -- imports match the studio's train.py.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.expanduser("~/owt"))
sys.path.insert(0, os.path.expanduser("~/tinystories"))

from data import get_tokenizer
from model import GPT

ckpt_path = os.path.expanduser("~/owt/checkpoints/owt_social_latest.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = get_tokenizer()

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
args = ckpt["args"]
print(f"Loaded step {ckpt.get('step', '?')}")

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
print(f"{model.count_parameters()/1e6:.1f}M params, field={args['use_field']}")


def generate(prompt, max_tokens=300):
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
    return enc.decode(out[0].tolist())


prompts = [
    "Once upon a time, in a small village nestled between mountains,",
    "The scientist looked at the data and realized something extraordinary:",
    "She opened the old wooden box and found",
    "The city had changed dramatically since the war ended.",
    "In the depths of the ocean, a strange creature",
    "The recipe for happiness, according to most philosophers,",
    "Breaking news: researchers at MIT have announced",
    "Dear diary, today was the strangest day of my life.",
]

for prompt in prompts:
    text = generate(prompt)
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*70}")
    print(text)
