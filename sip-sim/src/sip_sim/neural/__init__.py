"""Neural network modules for stigmergic attention research.

Provides the GPT model with optional social field attention,
along with training utilities and shared components.
"""

from sip_sim.neural.attention import ChunkParallelAttention
from sip_sim.neural.components import RMSNorm, RotaryEmbedding, apply_rope, rotate_half
from sip_sim.neural.data import get_tokenizer
from sip_sim.neural.model import Block, GPT
from sip_sim.neural.training import (
    cosine_lr_schedule,
    estimate_loss,
    generate_sample,
    get_device,
    load_checkpoint,
    load_model_from_checkpoint,
    maybe_compile,
    save_checkpoint,
    setup_training_precision,
)

__all__ = [
    # Model
    "GPT",
    "Block",
    "ChunkParallelAttention",
    # Components
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rope",
    "rotate_half",
    # Training
    "get_device",
    "estimate_loss",
    "generate_sample",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "cosine_lr_schedule",
    "setup_training_precision",
    "maybe_compile",
    # Data
    "get_tokenizer",
]
