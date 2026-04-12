"""Chunk-parallel GPT with multiscale social field attention.

Self-contained model file for the Lightning AI studio. Combines components,
attention, and model into a single file with full multiscale support.

Backward compatible: setting multi_scale=False reproduces the original
single-field architecture exactly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T, device):
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, freqs):
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Chunk-parallel attention with optional multiscale social field
# ---------------------------------------------------------------------------

class ChunkParallelAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, chunk_size=128,
                 evap_rate=0.05, use_field=True, mod_type="additive",
                 multi_scale=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.use_field = use_field
        self.mod_type = mod_type
        self.multi_scale = multi_scale

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

        if use_field:
            self.field_dim = self.head_dim
            self.evap_rate = evap_rate
            self.retain = 1.0 - evap_rate

            self.w_deposit = nn.Linear(d_model, n_head * self.field_dim, bias=False)
            nn.init.normal_(self.w_deposit.weight, std=0.02)

            if mod_type == "additive":
                self.w_mod = nn.Linear(self.field_dim, self.head_dim, bias=False)
                nn.init.zeros_(self.w_mod.weight)
            elif mod_type == "gating":
                self.w_gate = nn.Linear(self.field_dim, self.head_dim)
                nn.init.zeros_(self.w_gate.weight)
                nn.init.constant_(self.w_gate.bias, 2.0)

            if multi_scale:
                slow_init_logit = math.log(0.995 / 0.005)
                self.slow_retain_logit = nn.Parameter(
                    torch.full((n_head,), slow_init_logit))

                self.w_deposit_slow = nn.Linear(
                    d_model, n_head * self.field_dim, bias=False)
                nn.init.normal_(self.w_deposit_slow.weight, std=0.02)

                self.w_mod_slow = nn.Linear(
                    self.field_dim, self.head_dim, bias=False)
                nn.init.zeros_(self.w_mod_slow.weight)

    def _attend(self, q, k, v):
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale

        Tq, Tk = q.shape[2], k.shape[2]
        offset = Tk - Tq
        mask = torch.ones(Tq, Tk, device=q.device, dtype=torch.bool).triu(diagonal=offset + 1)
        att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        return att @ v

    def _update_field_with_retain(self, field_state, out_chunk, w_deposit, retain):
        B, C, _ = out_chunk.shape
        deposits = w_deposit(out_chunk)
        deposits = deposits.reshape(B, C, self.n_head, self.field_dim)
        device = deposits.device

        if isinstance(retain, (int, float)):
            decay_weights = retain ** torch.arange(
                C - 1, -1, -1, device=device).float()
            weighted = deposits * decay_weights.reshape(1, C, 1, 1)
            field_state = (retain ** C) * field_state + weighted.sum(dim=1)
        else:
            steps = torch.arange(C - 1, -1, -1, device=device).float()
            decay_weights = retain.unsqueeze(0) ** steps.unsqueeze(1)
            weighted = deposits * decay_weights.unsqueeze(0).unsqueeze(-1)
            retain_C = retain ** C
            field_state = retain_C.unsqueeze(0).unsqueeze(-1) * field_state + weighted.sum(dim=1)

        return field_state

    def _update_field(self, field_state, out_chunk):
        return self._update_field_with_retain(
            field_state, out_chunk, self.w_deposit, self.retain)

    def _update_slow_field(self, field_state, out_chunk):
        retain = torch.sigmoid(self.slow_retain_logit)
        return self._update_field_with_retain(
            field_state, out_chunk, self.w_deposit_slow, retain)

    def _modulate_keys(self, k, field_state, slow_field_state=None):
        if self.mod_type == "additive":
            k_shift = self.w_mod(field_state)
            k = k + k_shift.unsqueeze(2)
            if slow_field_state is not None:
                k_shift_slow = self.w_mod_slow(slow_field_state)
                k = k + k_shift_slow.unsqueeze(2)
            return k
        elif self.mod_type == "gating":
            gates = torch.sigmoid(self.w_gate(field_state))
            return k * gates.unsqueeze(2)
        return k

    def forward(self, x):
        B, T, _ = x.shape
        H, d = self.n_head, self.head_dim
        CS = self.chunk_size

        qkv = self.w_qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if not self.use_field:
            out = self._attend(q, k, v)
            out = out.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
            return self.resid_drop(self.w_out(out))

        n_chunks = (T + CS - 1) // CS
        field_state = torch.zeros(B, H, self.field_dim, device=x.device)

        slow_field = None
        if self.multi_scale:
            slow_field = torch.zeros(B, H, self.field_dim, device=x.device)

        outputs = []
        k_cache_list = []
        v_cache_list = []

        for ci in range(n_chunks):
            s = ci * CS
            e = min(s + CS, T)

            q_c = q[:, :, s:e, :]
            k_c = k[:, :, s:e, :]
            v_c = v[:, :, s:e, :]

            if ci > 0:
                k_c = self._modulate_keys(k_c, field_state, slow_field)

            if k_cache_list:
                k_full = torch.cat(k_cache_list + [k_c], dim=2)
                v_full = torch.cat(v_cache_list + [v_c], dim=2)
            else:
                k_full = k_c
                v_full = v_c

            out_c = self._attend(q_c, k_full, v_full)
            outputs.append(out_c)

            out_flat = out_c.permute(0, 2, 1, 3).reshape(B, e - s, self.d_model)
            field_state = self._update_field(field_state, out_flat)

            if self.multi_scale:
                slow_field = self._update_slow_field(slow_field, out_flat)

            k_cache_list.append(k_c)
            v_cache_list.append(v_c)

        output = torch.cat(outputs, dim=2)
        output = output.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        return self.resid_drop(self.w_out(output))


# ---------------------------------------------------------------------------
# Transformer block and full GPT
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, chunk_size=128,
                 evap_rate=0.05, use_field=True, mod_type="additive",
                 multi_scale=False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = ChunkParallelAttention(
            d_model, n_head, dropout, chunk_size,
            evap_rate, use_field, mod_type,
            multi_scale=multi_scale)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=384, n_head=6, n_layer=6,
                 dropout=0.1, max_len=512, chunk_size=128,
                 evap_rate=0.05, use_field=True, mod_type="additive",
                 gradient_checkpointing=False, multi_scale=False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, chunk_size,
                  evap_rate, use_field, mod_type,
                  multi_scale=multi_scale)
            for _ in range(n_layer)
        ])

        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
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
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.drop(x)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False)
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
        for _ in range(max_new_tokens):
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
