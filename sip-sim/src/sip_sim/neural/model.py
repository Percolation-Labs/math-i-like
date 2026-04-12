"""GPT with optional stigmergic (social field) attention.

Default config (~30M params):
    d_model=384, n_head=6, n_layer=6, vocab_size=50257

Set use_field=False for baseline (standard transformer).
Set use_field=True for social field variant.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sip_sim.neural.attention import ChunkParallelAttention
from sip_sim.neural.components import RMSNorm


class Block(nn.Module):
    """Transformer block: attention + MLP with pre-norm."""

    def __init__(self, d_model, n_head, dropout=0.1, chunk_size=128,
                 evap_rate=0.05, use_field=True, mod_type="additive",
                 learnable_retain=False, multi_scale=False, cross_layer=False):
        super().__init__()
        self.cross_layer = cross_layer
        self.ln1 = RMSNorm(d_model)
        self.attn = ChunkParallelAttention(
            d_model, n_head, dropout, chunk_size,
            evap_rate, use_field, mod_type,
            learnable_retain=learnable_retain,
            multi_scale=multi_scale,
            cross_layer=cross_layer)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, field_init=None):
        if self.cross_layer:
            attn_out, field_final = self.attn(self.ln1(x), field_init=field_init)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, field_final
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x


class GPT(nn.Module):
    """GPT with optional social field attention.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (default: GPT-2 BPE = 50257).
    d_model : int
        Hidden dimension.
    n_head : int
        Number of attention heads.
    n_layer : int
        Number of transformer blocks.
    dropout : float
        Dropout rate.
    max_len : int
        Maximum sequence length.
    chunk_size : int
        Chunk size for chunk-parallel attention.
    evap_rate : float
        Field evaporation rate (1 - retain).
    use_field : bool
        Enable social field attention.
    mod_type : str
        Field modulation type: "additive" or "gating".
    gradient_checkpointing : bool
        Enable gradient checkpointing to save memory.
    learnable_retain : bool
        If True, retention rate is learned per head.
    multi_scale : bool
        If True, adds a slow field alongside the fast field.
    cross_layer : bool
        If True, field state flows vertically between layers.
    """

    def __init__(self, vocab_size=50257, d_model=384, n_head=6, n_layer=6,
                 dropout=0.1, max_len=512, chunk_size=128,
                 evap_rate=0.05, use_field=True, mod_type="additive",
                 gradient_checkpointing=False,
                 learnable_retain=False, multi_scale=False, cross_layer=False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.cross_layer = cross_layer

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, chunk_size,
                  evap_rate, use_field, mod_type,
                  learnable_retain=learnable_retain,
                  multi_scale=multi_scale,
                  cross_layer=cross_layer)
            for _ in range(n_layer)
        ])

        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale residual projections
        for block in self.blocks:
            nn.init.normal_(block.attn.w_out.weight, std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(block.mlp[-2].weight, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) target indices or None
        Returns: logits (B, T, V) if no targets, else (loss, logits)
        """
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.drop(x)

        field_state = None
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                if self.cross_layer:
                    x, field_state = torch.utils.checkpoint.checkpoint(
                        block, x, field_state, use_reentrant=False)
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, use_reentrant=False)
            else:
                if self.cross_layer:
                    x, field_state = block(x, field_init=field_state)
                else:
                    x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   targets.view(-1), ignore_index=-1)
            return loss, logits

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max_len
            idx_cond = idx if idx.shape[1] <= self.max_len else idx[:, -self.max_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
