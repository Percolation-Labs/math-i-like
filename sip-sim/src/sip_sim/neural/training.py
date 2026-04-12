"""Training utilities for GPT models.

Shared helpers used by all training scripts: device selection,
loss estimation, checkpoint management, and learning rate schedules.
"""

import math
import os

import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(model, val_loader, device, max_batches=50, amp_dtype=None):
    """Estimate validation loss over a subset of batches.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    val_loader : DataLoader
        Validation data loader.
    device : torch.device
        Device to run on.
    max_batches : int
        Maximum number of batches to evaluate.
    amp_dtype : torch.dtype or None
        If set, enable AMP autocast with this dtype.
    """
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


@torch.no_grad()
def generate_sample(model, device, enc, prompt="Once upon a time",
                    max_tokens=200, temperature=0.8, top_k=50):
    """Generate text from a prompt using the model."""
    model.eval()
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k)
    model.train()
    return enc.decode(out[0].tolist())


def save_checkpoint(model, step, args, path, optimizer=None):
    """Save a training checkpoint.

    Handles torch.compile'd models by unwrapping _orig_mod.
    """
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "model": raw_model.state_dict(),
        "step": step,
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, device, optimizer=None):
    """Load a checkpoint into model (and optionally optimizer).

    Returns the step number from the checkpoint.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


def load_model_from_checkpoint(path, device):
    """Load a trained model from checkpoint for inference.

    Returns (model, args_dict, enc).
    """
    from sip_sim.neural.model import GPT
    from sip_sim.neural.data import get_tokenizer

    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    enc = get_tokenizer()

    model = GPT(
        vocab_size=enc.n_vocab,
        d_model=args["d_model"],
        n_head=args["n_head"],
        n_layer=args["n_layer"],
        dropout=0.0,  # no dropout at inference
        max_len=args["context_length"],
        chunk_size=args["chunk_size"],
        evap_rate=args.get("evap_rate", 0.05),
        use_field=args["use_field"],
        mod_type=args.get("mod_type", "additive"),
        multi_scale=args.get("multi_scale", False),
        learnable_retain=args.get("learnable_retain", False),
        cross_layer=args.get("cross_layer", False),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, args, enc


def cosine_lr_schedule(warmup_steps, total_steps, min_lr_ratio=0.0):
    """Return a lambda for torch LambdaLR implementing cosine decay with warmup.

    Parameters
    ----------
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total number of training steps.
    min_lr_ratio : float
        Minimum LR as fraction of peak (cosine decays to this).
    """
    def schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule


def setup_training_precision(device):
    """Configure precision settings for training.

    Returns (use_amp, amp_dtype) tuple.
    """
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return use_amp, amp_dtype


def maybe_compile(model, device, enabled=True):
    """Compile model with torch.compile if available and on CUDA."""
    if enabled and hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model with torch.compile...")
        return torch.compile(model)
    return model
