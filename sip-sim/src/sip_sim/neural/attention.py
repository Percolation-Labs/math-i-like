"""Chunk-parallel attention with optional stigmergic (social) field.

Key idea: instead of position-by-position sequential field updates (O(T) serial),
process chunks of C tokens in parallel with field updates between chunks (O(T/C) serial).
Within each chunk: standard parallel causal attention.
Between chunks: field state carries information, modulating keys in next chunk.

At inference time, autoregressive generation is already sequential, so the field
update adds zero overhead.

Field variants (controlled by constructor args):
- Fixed retention (default): single evap_rate for all heads
- Learnable retention: per-head sigmoid-bounded retention learned during training
- Multi-scale: fast field (fixed decay) + slow field (learnable decay) per head
- Cross-layer coupling: accept/return field state for vertical field flow
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sip_sim.neural.components import RotaryEmbedding, apply_rope


class ChunkParallelAttention(nn.Module):
    """Causal attention with optional social field, chunk-parallel for training.

    The social field is a vector recurrence per head that modulates key geometry
    between chunks. Within a chunk, standard parallel attention is used.
    Cross-chunk attention uses a KV cache so no information is lost.

    The approximation: all keys within chunk k are modulated by the same field
    state (from end of chunk k-1). This is accurate when chunk_size * evap_rate
    is small (field changes slowly within a chunk).

    Parameters
    ----------
    learnable_retain : bool
        If True, retention rate is a learnable per-head parameter (sigmoid-bounded).
    multi_scale : bool
        If True, adds a slow field alongside the fast field, each with separate
        deposit/modulation projections. The slow field uses learnable retention
        initialised near 1.0 for long-range memory.
    cross_layer : bool
        If True, forward() accepts field_init and returns final field_state
        for vertical coupling between layers.
    """

    def __init__(self, d_model, n_head, dropout=0.1, chunk_size=128,
                 evap_rate=0.05, use_field=True, mod_type="additive",
                 learnable_retain=False, multi_scale=False, cross_layer=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.use_field = use_field
        self.mod_type = mod_type
        self.learnable_retain = learnable_retain
        self.multi_scale = multi_scale
        self.cross_layer = cross_layer

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

        if use_field:
            self.field_dim = self.head_dim
            self.evap_rate = evap_rate

            if learnable_retain:
                # sigmoid(logit) = retain; init so retain ≈ 1 - evap_rate
                init_logit = math.log((1 - evap_rate) / evap_rate)
                self.retain_logit = nn.Parameter(
                    torch.full((n_head,), init_logit))
            else:
                self.retain = 1.0 - evap_rate

            # Deposit and modulation for primary (fast) field
            self.w_deposit = nn.Linear(d_model, n_head * self.field_dim, bias=False)
            nn.init.normal_(self.w_deposit.weight, std=0.02)

            if mod_type == "additive":
                self.w_mod = nn.Linear(self.field_dim, self.head_dim, bias=False)
                nn.init.zeros_(self.w_mod.weight)
            elif mod_type == "gating":
                self.w_gate = nn.Linear(self.field_dim, self.head_dim)
                nn.init.zeros_(self.w_gate.weight)
                nn.init.constant_(self.w_gate.bias, 2.0)

            # Slow field for multi-scale
            if multi_scale:
                slow_init_logit = math.log(0.995 / 0.005)  # retain ≈ 0.995
                self.slow_retain_logit = nn.Parameter(
                    torch.full((n_head,), slow_init_logit))

                self.w_deposit_slow = nn.Linear(
                    d_model, n_head * self.field_dim, bias=False)
                nn.init.normal_(self.w_deposit_slow.weight, std=0.02)

                self.w_mod_slow = nn.Linear(
                    self.field_dim, self.head_dim, bias=False)
                nn.init.zeros_(self.w_mod_slow.weight)

    def _get_retain(self):
        """Get retention rate(s), handling learnable vs fixed."""
        if self.learnable_retain:
            return torch.sigmoid(self.retain_logit)  # (H,)
        return self.retain  # scalar

    def _attend(self, q, k, v):
        """Standard scaled dot-product causal attention."""
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
        """Update field state from chunk outputs with a given retention rate.

        retain: scalar or (H,) tensor
        """
        B, C, _ = out_chunk.shape
        deposits = w_deposit(out_chunk)
        deposits = deposits.reshape(B, C, self.n_head, self.field_dim)

        device = deposits.device

        if isinstance(retain, (int, float)):
            # Scalar retain — same as original
            decay_weights = retain ** torch.arange(
                C - 1, -1, -1, device=device).float()
            weighted = deposits * decay_weights.reshape(1, C, 1, 1)
            field_state = (retain ** C) * field_state + weighted.sum(dim=1)
        else:
            # Per-head retain: (H,) -> need per-head decay weights
            steps = torch.arange(C - 1, -1, -1, device=device).float()  # (C,)
            # retain: (H,) -> decay_weights: (C, H)
            decay_weights = retain.unsqueeze(0) ** steps.unsqueeze(1)
            # deposits: (B, C, H, d_f) * decay_weights: (1, C, H, 1)
            weighted = deposits * decay_weights.unsqueeze(0).unsqueeze(-1)
            retain_C = retain ** C  # (H,)
            field_state = retain_C.unsqueeze(0).unsqueeze(-1) * field_state + weighted.sum(dim=1)

        return field_state

    def _update_field(self, field_state, out_chunk):
        """Update primary field state from chunk outputs."""
        retain = self._get_retain()
        return self._update_field_with_retain(
            field_state, out_chunk, self.w_deposit, retain)

    def _update_slow_field(self, field_state, out_chunk):
        """Update slow field state from chunk outputs."""
        retain = torch.sigmoid(self.slow_retain_logit)
        return self._update_field_with_retain(
            field_state, out_chunk, self.w_deposit_slow, retain)

    def _modulate_keys(self, k, field_state, slow_field_state=None):
        """Apply social field modulation to keys."""
        if self.mod_type == "additive":
            k_shift = self.w_mod(field_state)  # (B, H, d)
            k = k + k_shift.unsqueeze(2)
            if slow_field_state is not None:
                k_shift_slow = self.w_mod_slow(slow_field_state)
                k = k + k_shift_slow.unsqueeze(2)
            return k
        elif self.mod_type == "gating":
            gates = torch.sigmoid(self.w_gate(field_state))
            return k * gates.unsqueeze(2)
        return k

    def forward(self, x, field_init=None):
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, T, d_model)
        field_init : Tensor (B, H, d_f) or None
            Initial field state from previous layer (cross-layer coupling).

        Returns
        -------
        output : Tensor (B, T, d_model)
        field_final : Tensor (B, H, d_f) — only if cross_layer is True
        """
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
            result = self.resid_drop(self.w_out(out))
            if self.cross_layer:
                return result, None
            return result

        # --- Chunk-parallel with social field ---
        n_chunks = (T + CS - 1) // CS

        if field_init is not None and self.cross_layer:
            field_state = field_init
        else:
            field_state = torch.zeros(B, H, self.field_dim, device=x.device)

        slow_field = None
        if self.multi_scale:
            slow_field = torch.zeros(B, H, self.field_dim, device=x.device)

        outputs = []
        k_cache_list = []
        v_cache_list = []

        if getattr(self, '_capture', False):
            self._field_states = [field_state.detach().cpu()]
            self._deposits = []
            if self.multi_scale:
                self._slow_field_states = [slow_field.detach().cpu()]

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

            if getattr(self, '_capture', False):
                deposits = self.w_deposit(out_flat)
                deposits = deposits.reshape(B, e - s, self.n_head, self.field_dim)
                self._deposits.append(deposits.detach().cpu())

            field_state = self._update_field(field_state, out_flat)

            if self.multi_scale:
                slow_field = self._update_slow_field(slow_field, out_flat)

            if getattr(self, '_capture', False):
                self._field_states.append(field_state.detach().cpu())
                if self.multi_scale:
                    self._slow_field_states.append(slow_field.detach().cpu())

            k_cache_list.append(k_c)
            v_cache_list.append(v_c)

        output = torch.cat(outputs, dim=2)
        output = output.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        result = self.resid_drop(self.w_out(output))

        if self.cross_layer:
            return result, field_state
        return result

    @torch.no_grad()
    def generate_step(self, x, field_state=None):
        """Single-step forward for autoregressive generation.

        x: (B, 1, d_model) -- single new token embedding (after norms etc)
        field_state: (B, H, d_f) or None

        Returns: (output, new_field_state)
        Note: caller must manage KV cache externally for full generation.
        """
        raise NotImplementedError("Use GPT.generate() for inference")
