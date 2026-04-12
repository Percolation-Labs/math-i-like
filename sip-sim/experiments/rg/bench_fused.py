"""Benchmark: how fast can we make the position-by-position loop?

Tests:
  1. Current Python loop on CPU
  2. Current Python loop on MPS
  3. torch.jit.script version on CPU
  4. torch.jit.script version on MPS
  5. Tighter loop (pre-allocate, minimize allocations) on CPU
  6. Tighter loop on MPS

Usage:
    cd sip-sim
    uv run python experiments/rg/bench_fused.py
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Current implementation (from run_decoupled.py) ────────────

class FieldAttentionBaseline(nn.Module):
    """Current position-by-position implementation."""

    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_f = self.head_dim

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_deposit = nn.Linear(d_model, n_head * self.d_f, bias=False)
        self.w_mod = nn.Linear(self.d_f, self.head_dim, bias=False)
        self.retain = 0.90

    def forward(self, x):
        B, T, C = x.shape
        H, d_h, d_f = self.n_head, self.head_dim, self.d_f

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        field_state = torch.zeros(B, H, d_f, device=x.device, dtype=x.dtype)
        field_states = []
        output_list = []

        for i in range(T):
            k_i_raw = k[:, :, i:i+1, :]

            if i > 0:
                fh = torch.stack(field_states, dim=2)
                mod = self.w_mod(fh)
                k_prev_mod = k[:, :, :i, :] + mod
                k_mod = torch.cat([k_prev_mod, k_i_raw], dim=2)
            else:
                k_mod = k_i_raw

            q_i = q[:, :, i:i+1, :]
            att_i = (q_i @ k_mod.transpose(-2, -1)) * (d_h ** -0.5)
            att_i = F.softmax(att_i, dim=-1)
            out_i = (att_i @ v[:, :, :i+1, :]).squeeze(2)
            out_flat = out_i.transpose(1, 2).contiguous().view(B, C)
            output_list.append(out_flat)

            deposit = self.w_deposit(out_flat).view(B, H, d_f)
            field_state = self.retain * field_state + deposit
            field_states.append(field_state)

        return torch.stack(output_list, dim=1)


# ── Tighter loop: pre-allocate, avoid stack/cat ───────────────

class FieldAttentionTight(nn.Module):
    """Tighter loop: pre-allocate field history, avoid repeated stack."""

    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_f = self.head_dim
        self.max_len = max_len

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_deposit = nn.Linear(d_model, n_head * self.d_f, bias=False)
        self.w_mod = nn.Linear(self.d_f, self.head_dim, bias=False)
        self.retain = 0.90

    def forward(self, x):
        B, T, C = x.shape
        H, d_h, d_f = self.n_head, self.head_dim, self.d_f
        scale = d_h ** -0.5

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        # Pre-allocate
        out = torch.empty(B, T, C, device=x.device, dtype=x.dtype)
        k_work = k.clone()  # working copy of keys (will be modulated in place)
        field_state = torch.zeros(B, H, d_f, device=x.device, dtype=x.dtype)

        w_mod_weight = self.w_mod.weight  # (d_h, d_f)
        w_dep_weight = self.w_deposit.weight  # (H*d_f, C)

        for i in range(T):
            if i > 0:
                # Modulate key at position i-1 using field state
                # field_state: (B, H, d_f), w_mod: (d_h, d_f) -> (B, H, d_h)
                mod_i = field_state @ w_mod_weight.t()  # (B, H, d_h)
                k_work[:, :, i-1, :] = k[:, :, i-1, :] + mod_i

            q_i = q[:, :, i:i+1, :]  # (B, H, 1, d_h)
            att_i = (q_i @ k_work[:, :, :i+1, :].transpose(-2, -1)) * scale
            att_i = F.softmax(att_i, dim=-1)
            out_i = (att_i @ v[:, :, :i+1, :]).squeeze(2)  # (B, H, d_h)
            out_flat = out_i.transpose(1, 2).contiguous().view(B, C)  # (B, C)
            out[:, i, :] = out_flat

            deposit = (out_flat @ w_dep_weight.t()).view(B, H, d_f)
            field_state = self.retain * field_state + deposit

        return out


# ── JIT-scriptable version ────────────────────────────────────

@torch.jit.script
def field_forward_jit(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    w_mod_weight: torch.Tensor, w_dep_weight: torch.Tensor,
    retain: float, B: int, T: int, C: int, H: int, d_h: int, d_f: int
) -> torch.Tensor:
    scale = d_h ** -0.5
    out = torch.empty(B, T, C, device=q.device, dtype=q.dtype)
    k_work = k.clone()
    field_state = torch.zeros(B, H, d_f, device=q.device, dtype=q.dtype)

    for i in range(T):
        if i > 0:
            mod_i = field_state @ w_mod_weight.t()
            k_work[:, :, i-1, :] = k[:, :, i-1, :] + mod_i

        q_i = q[:, :, i:i+1, :]
        att_i = torch.matmul(q_i, k_work[:, :, :i+1, :].transpose(-2, -1)) * scale
        att_i = torch.softmax(att_i, dim=-1)
        out_i = torch.matmul(att_i, v[:, :, :i+1, :]).squeeze(2)
        out_flat = out_i.transpose(1, 2).contiguous().view(B, C)
        out[:, i, :] = out_flat

        deposit = torch.matmul(out_flat, w_dep_weight.t()).view(B, H, d_f)
        field_state = retain * field_state + deposit

    return out


class FieldAttentionJIT(nn.Module):
    """Wrapper that calls the JIT-scripted forward."""

    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_f = self.head_dim

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_deposit = nn.Linear(d_model, n_head * self.d_f, bias=False)
        self.w_mod = nn.Linear(self.d_f, self.head_dim, bias=False)
        self.retain = 0.90

    def forward(self, x):
        B, T, C = x.shape
        H, d_h, d_f = self.n_head, self.head_dim, self.d_f

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        return field_forward_jit(
            q, k, v, self.w_mod.weight, self.w_deposit.weight,
            self.retain, B, T, C, H, d_h, d_f)


# ── Benchmark ─────────────────────────────────────────────────

def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench_one(model, x, device, n_warmup=3, n_iter=10, label=""):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
            sync(device)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(x)
            sync(device)
        dt = (time.perf_counter() - t0) / n_iter

    print(f"  {label:>30s}: {dt*1000:8.1f} ms/fwd")
    return dt


def run_benchmarks():
    D = 48
    H = 4
    T = 18
    B = 64
    MAX_LEN = 50

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    for device in devices:
        print(f"\n{'=' * 60}")
        print(f"  Device: {device}, B={B}, T={T}, d={D}, H={H}")
        print(f"{'=' * 60}")

        x = torch.randn(B, T, D, device=device)

        # Baseline (current implementation)
        m_base = FieldAttentionBaseline(D, H, MAX_LEN).to(device)
        bench_one(m_base, x, device, label="baseline (current)")

        # Tight loop
        m_tight = FieldAttentionTight(D, H, MAX_LEN).to(device)
        bench_one(m_tight, x, device, label="tight loop")

        # JIT scripted
        m_jit = FieldAttentionJIT(D, H, MAX_LEN).to(device)
        bench_one(m_jit, x, device, label="JIT scripted")

        # torch.compile (if available)
        try:
            m_compiled = torch.compile(
                FieldAttentionTight(D, H, MAX_LEN).to(device),
                mode="reduce-overhead")
            bench_one(m_compiled, x, device, n_warmup=5, label="torch.compile tight")
        except Exception as e:
            print(f"  {'torch.compile tight':>30s}: FAILED ({e})")

    # Longer sequence benchmark
    for T_long in [34, 50]:
        print(f"\n{'─' * 60}")
        print(f"  Longer sequence: T={T_long}")
        print(f"{'─' * 60}")
        for device in devices:
            x_long = torch.randn(B, T_long, D, device=device)
            m_base = FieldAttentionBaseline(D, H, MAX_LEN).to(device)
            m_tight = FieldAttentionTight(D, H, MAX_LEN).to(device)
            m_jit = FieldAttentionJIT(D, H, MAX_LEN).to(device)

            bench_one(m_base, x_long, device, label=f"{device} baseline T={T_long}")
            bench_one(m_tight, x_long, device, label=f"{device} tight T={T_long}")
            bench_one(m_jit, x_long, device, label=f"{device} JIT T={T_long}")

    # Training step benchmark (includes backward pass)
    print(f"\n{'=' * 60}")
    print(f"  Training step (fwd + bwd), T={T}, B={B}")
    print(f"{'=' * 60}")
    for device in devices:
        x = torch.randn(B, T, D, device=device)
        target = torch.randint(0, 16, (B,), device=device)

        for label, ModelClass in [("baseline", FieldAttentionBaseline),
                                   ("tight", FieldAttentionTight)]:
            model = ModelClass(D, H, MAX_LEN).to(device)
            head = nn.Linear(D, 16, bias=False).to(device)
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            # Warmup
            for _ in range(3):
                out = model(x)
                loss = F.cross_entropy(head(out[:, -1, :]), target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sync(device)

            t0 = time.perf_counter()
            for _ in range(10):
                out = model(x)
                loss = F.cross_entropy(head(out[:, -1, :]), target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sync(device)
            dt = (time.perf_counter() - t0) / 10

            print(f"  {device + ' ' + label:>30s}: {dt*1000:8.1f} ms/step")


if __name__ == "__main__":
    run_benchmarks()
