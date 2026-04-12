"""
P3b: Critical Exponent — 1 Channel
====================================

P3 measured beta = 0.935 ≈ 1.0 (mean-field DP) with 4 channels.
The explanation: 4 channels = effective dimension d=4 = d_c,
so mean-field is exact.

Prediction: with 1 channel on a longer grid, d_eff = 1 < d_c = 4,
so beta should shift to the 1D DP value: beta ≈ 0.584.

This is the direct test of the AC pipeline's universality prediction.
Same stoichiometry [0→1, 1→0, 1→2] → same Puiseux exponent p/q=1/2
→ DP universality class → but the EXPONENT depends on dimension.

We also test 2 channels (d_eff=2) to see if beta interpolates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

print = functools.partial(print, flush=True)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


class DPFTCritical(nn.Module):
    def __init__(self, grid_size=128, n_channels=1, n_steps=200, dt=0.02):
        super().__init__()
        self.grid_size = grid_size
        self.C = n_channels
        self.n_steps = n_steps
        self.dt = dt

        self.kernel = nn.Parameter(torch.tensor([[[1., -2., 1.]]]).repeat(n_channels, 1, 1))
        self.D = nn.Parameter(torch.ones(n_channels) * 0.3)
        self.mu = nn.Parameter(torch.ones(n_channels) * 0.1)
        self.beta_consume = nn.Parameter(torch.ones(n_channels) * 0.1)

        self.birth_proj = nn.Linear(n_channels, n_channels)
        self.death_rate = nn.Parameter(torch.ones(n_channels) * 0.3)
        self.branching = nn.Parameter(torch.ones(n_channels) * 0.1)

    def run_dynamics(self, phi_init, alpha_ctrl=1.0, measure_from=0.5):
        phi = phi_init.clone()
        n = torch.zeros_like(phi)
        densities = []
        measure_start = int(self.n_steps * measure_from)

        for t in range(self.n_steps):
            lap = F.conv1d(phi, self.kernel, padding=1, groups=self.C)
            D = F.softplus(self.D).view(1, self.C, 1)
            mu = F.softplus(self.mu).view(1, self.C, 1)
            beta = F.softplus(self.beta_consume).view(1, self.C, 1)
            dphi = D * lap - mu * phi + 0.2 * n - beta * n * phi
            phi = phi + self.dt * dphi

            phi_t = phi.transpose(1, 2)
            birth = alpha_ctrl * torch.sigmoid(self.birth_proj(phi_t)).transpose(1, 2)
            death = torch.sigmoid(self.death_rate).view(1, self.C, 1)
            branch = F.softplus(self.branching).view(1, self.C, 1)
            dn = birth * (1 + branch * n) - death * n
            n = F.relu(n + self.dt * dn)

            if t >= measure_start:
                densities.append(n.mean().item())

        return np.mean(densities) if densities else 0.0


def measure_exponent(n_channels, grid_size, label):
    print(f"\n{'='*70}")
    print(f"  {label}: {n_channels} channel(s), grid={grid_size}")
    print(f"  Prediction: d_eff={n_channels}, d_c=4")
    if n_channels >= 4:
        print(f"  Expected: mean-field beta ≈ 1.0")
    elif n_channels == 1:
        print(f"  Expected: 1D DP beta ≈ 0.58")
    else:
        print(f"  Expected: intermediate beta")
    print(f"{'='*70}")

    model = DPFTCritical(grid_size=grid_size, n_channels=n_channels,
                         n_steps=200, dt=0.02).to(DEVICE)

    # Coarse sweep
    print(f"\n  Coarse sweep...")
    alphas_coarse = np.linspace(0.1, 4.0, 25)
    densities_coarse = []
    n_samples = 100

    for alpha in alphas_coarse:
        model.eval()
        with torch.no_grad():
            phi_init = torch.randn(n_samples, n_channels, grid_size, device=DEVICE) * 0.3 + 0.5
            d = model.run_dynamics(phi_init, alpha_ctrl=alpha)
            densities_coarse.append(d)

    densities_coarse = np.array(densities_coarse)

    # Find critical point from steepest relative change
    log_d = np.log(densities_coarse + 1e-10)
    grad = np.gradient(log_d, alphas_coarse)
    alpha_c = alphas_coarse[np.argmax(grad)]

    print(f"  alpha_c ≈ {alpha_c:.3f}")

    # Fine sweep above critical point
    print(f"  Fine sweep near alpha_c...")
    alphas_fine = np.linspace(alpha_c + 0.02, alpha_c + 2.0, 50)
    densities_fine = []

    for alpha in alphas_fine:
        model.eval()
        with torch.no_grad():
            phi_init = torch.randn(n_samples, n_channels, grid_size, device=DEVICE) * 0.3 + 0.5
            d = model.run_dynamics(phi_init, alpha_ctrl=alpha)
            densities_fine.append(d)

    densities_fine = np.array(densities_fine)

    # Log-log fit
    eps = alphas_fine - alpha_c
    rho = densities_fine
    valid = (eps > 0.05) & (rho > 1e-6)

    if valid.sum() > 5:
        log_eps = np.log(eps[valid])
        log_rho = np.log(rho[valid])

        # Fit in the scaling region (not too far from critical point)
        # Use only the first half to avoid saturation effects
        n_fit = min(valid.sum(), 20)
        coeffs = np.polyfit(log_eps[:n_fit], log_rho[:n_fit], 1)
        beta = coeffs[0]

        # Also try fitting just the closest points (most reliable)
        n_close = min(valid.sum(), 10)
        coeffs_close = np.polyfit(log_eps[:n_close], log_rho[:n_close], 1)
        beta_close = coeffs_close[0]

        print(f"\n  RESULTS:")
        print(f"    beta (full fit, {n_fit} pts):   {beta:.3f}")
        print(f"    beta (close fit, {n_close} pts): {beta_close:.3f}")
        print()
        print(f"    Theory comparison:")
        print(f"      Mean-field DP (d >= 4): beta = 1.000")
        print(f"      1D DP:                  beta = 0.584")
        print(f"      2D DP:                  beta = 0.584  (same as 1D for order param)")
        print(f"      Mean-field generic:     beta = 1.000")

        # Show scaling data
        print(f"\n    Scaling data:")
        print(f"    {'eps':>8}  {'<n>':>12}  {'log(eps)':>10}  {'log(<n>)':>10}")
        for j in range(min(15, n_fit)):
            idx = np.where(valid)[0][j]
            print(f"    {eps[idx]:8.4f}  {rho[idx]:12.6f}  {log_eps[j]:10.4f}  {log_rho[j]:10.4f}")

        return beta, beta_close, alpha_c
    else:
        print(f"  Not enough data for fit. Max density: {densities_fine.max():.6f}")
        return None, None, alpha_c


# ════════════════════════════════════════════════════════════════
# RUN: 1, 2, and 4 channels
# ════════════════════════════════════════════════════════════════

results = {}

for n_ch, grid in [(1, 256), (2, 128), (4, 64)]:
    label = f"{n_ch}-channel (d_eff={n_ch})"
    beta, beta_close, ac = measure_exponent(n_ch, grid, label)
    results[n_ch] = {'beta': beta, 'beta_close': beta_close, 'alpha_c': ac}


# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("CRITICAL EXPONENT vs EFFECTIVE DIMENSION")
print("=" * 70)
print()
print(f"  {'Channels':>9}  {'d_eff':>6}  {'beta':>8}  {'beta(close)':>12}  {'Theory':>12}")
print(f"  {'─'*9}  {'─'*6}  {'─'*8}  {'─'*12}  {'─'*12}")

for n_ch in [1, 2, 4]:
    r = results[n_ch]
    if n_ch >= 4:
        theory = "MF: 1.000"
    elif n_ch == 1:
        theory = "1D: 0.584"
    else:
        theory = "interp?"
    beta_str = f"{r['beta']:.3f}" if r['beta'] else "—"
    close_str = f"{r['beta_close']:.3f}" if r['beta_close'] else "—"
    print(f"  {n_ch:9d}  {n_ch:6d}  {beta_str:>8}  {close_str:>12}  {theory:>12}")

print()
print("  The AC prediction: stoichiometry [0→1, 1→0, 1→2] → DP universality.")
print("  The exponent depends on effective dimension d_eff ~ n_channels.")
print("  If beta decreases from ~1.0 (4ch) toward ~0.58 (1ch),")
print("  the dimensional crossover of DP universality is confirmed")
print("  in a neural network whose forward pass is the coupled dynamics.")
print()

# Check if there's a trend
betas = [results[n]['beta'] for n in [1, 2, 4] if results[n]['beta'] is not None]
channels = [n for n in [1, 2, 4] if results[n]['beta'] is not None]
if len(betas) >= 2:
    if betas[0] < betas[-1] - 0.05:
        print("  >> DIMENSIONAL CROSSOVER DETECTED.")
        print(f"     beta decreases from {betas[-1]:.3f} ({channels[-1]}ch)")
        print(f"     to {betas[0]:.3f} ({channels[0]}ch)")
        print("     Consistent with DP universality below d_c.")
    elif abs(betas[0] - betas[-1]) < 0.05:
        print("  >> No dimensional dependence. All exponents similar.")
        print("     May indicate the network is always in mean-field,")
        print("     or the measurement needs longer equilibration.")
    else:
        print(f"  >> Unexpected trend: beta increases with fewer channels.")
