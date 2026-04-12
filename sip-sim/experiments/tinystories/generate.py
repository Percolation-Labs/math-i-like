"""Generate text from a trained checkpoint.

Usage:
    python generate.py --checkpoint checkpoints/social_final.pt
    python generate.py --checkpoint checkpoints/social_final.pt --prompt "The dog"
    python generate.py --checkpoint checkpoints/social_final.pt --interactive
"""

import argparse

import torch

from sip_sim.neural import get_device, load_model_from_checkpoint


def generate(model, enc, prompt, device, max_tokens=300, temperature=0.8, top_k=50):
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Device: {device}")
    model, ckpt_args, enc = load_model_from_checkpoint(args.checkpoint, device)

    n_params = model.count_parameters()
    field_str = (f"social field (ε={ckpt_args['evap_rate']}, {ckpt_args['mod_type']})"
                 if ckpt_args["use_field"] else "baseline")
    print(f"Loaded {n_params/1e6:.1f}M param model — {field_str}")

    if args.interactive:
        print("\nInteractive mode. Type a prompt, press Enter to generate. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input(">>> ")
                if not prompt.strip():
                    continue
                text = generate(model, enc, prompt, device,
                                args.max_tokens, args.temperature, args.top_k)
                print(text)
                print()
            except KeyboardInterrupt:
                print("\nBye!")
                break
    else:
        text = generate(model, enc, args.prompt, device,
                        args.max_tokens, args.temperature, args.top_k)
        print(text)


if __name__ == "__main__":
    main()
