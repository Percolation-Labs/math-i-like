"""
2D lattice verification of the one-loop KS instability threshold.

Strategy: measure the linear growth rate ω(k) of the lowest Fourier
mode for a range of densities n₀, extract n_c where ω=0, compare
with tree-level and one-loop predictions.

Uses semi-implicit time-stepping for stability.
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import math

# ── Parameters ──────────────────────────────────────────────────────
D_A = 1.0; D_c = 1.0; kappa = 1.0; chi = 0.5; mu_s = 0.5
dt = 0.005

D_tot = D_A + D_c
n_c_tree = D_A * kappa / (chi * mu_s)
oneloop_C = chi * mu_s * D_c**2 / (math.pi * D_tot**3)

print("=" * 70)
print("KELLER-SEGEL LATTICE SIMULATION")
print("=" * 70)
print(f"D_A={D_A}, D_c={D_c}, κ={kappa}, χ={chi}, μ={mu_s}")
print(f"Tree-level n_c = {n_c_tree:.4f}")
print(f"One-loop coefficient C = {oneloop_C:.6f}")
print(f"  n_c(L) = n_c^tree - C·κ/(χμ)·ln(L) = {n_c_tree:.4f} - {oneloop_C*kappa/(chi*mu_s):.4f}·ln(L)")
print()


def measure_growth_rate(L, n0, n_steps=4000, dt=0.005, n_trials=30):
    """
    Evolve linearized KS on L×L torus, measure growth rate of k_min mode.
    Semi-implicit: diffusion treated implicitly, coupling explicitly.
    """
    kx = 2*np.pi*fftfreq(L, d=1.0)
    ky = 2*np.pi*fftfreq(L, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    
    # Semi-implicit operators
    diff_n = 1.0 / (1.0 + dt * D_A * K2)
    diff_c = 1.0 / (1.0 + dt * (D_c * K2 + kappa))
    
    # Steady-state attractant
    c0 = mu_s * n0 / kappa
    
    # k_min shell
    k_min = 2*np.pi/L
    shell = (np.abs(np.sqrt(K2) - k_min) < 0.6*k_min) & (K2 > 0)
    
    rates = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 1000 + L)
        
        # Small perturbation around uniform state
        dn = 0.001 * n0 * rng.randn(L, L)
        dc = np.zeros((L, L))
        
        log_S = []
        for step in range(n_steps):
            # Current concentrations
            n_full = n0 + dn
            c_full = c0 + dc
            
            # Clip for stability
            n_full = np.clip(n_full, 0.01*n0, 10*n0)
            
            # Attractant gradient
            c_hat = fft2(c_full)
            grad_cx = np.real(ifft2(1j*KX*c_hat))
            grad_cy = np.real(ifft2(1j*KY*c_hat))
            
            # Chemotactic flux divergence
            flux_x = n_full * grad_cx
            flux_y = n_full * grad_cy
            div_flux = np.real(ifft2(1j*KX*fft2(flux_x) + 1j*KY*fft2(flux_y)))
            
            # Semi-implicit update for n
            dn_hat = fft2(dn - dt*chi*div_flux + dt*chi*n0*D_c*fft2(np.real(ifft2(-K2*c_hat))).real.__array__())
            # Simpler: just do explicit coupling + implicit diffusion
            rhs_n = dn + dt * (-chi * div_flux + chi * n0 * np.real(ifft2(-K2 * c_hat)))
            # Actually, let me just do the basic update properly
            
            # Explicit coupling terms
            source_n = -chi * div_flux  # chemotaxis
            source_c = mu_s * dn        # secretion
            
            # Semi-implicit step
            dn_hat = fft2(dn + dt * source_n)
            dc_hat = fft2(dc + dt * source_c)
            
            dn_hat *= diff_n  # implicit diffusion for n
            dc_hat *= diff_c  # implicit diffusion + decay for c
            
            dn = np.real(ifft2(dn_hat))
            dc = np.real(ifft2(dc_hat))
            
            # Clip perturbation
            dn = np.clip(dn, -0.5*n0, 0.5*n0)
            dc = np.clip(dc, -5*c0, 5*c0)
            
            # Record structure factor
            if step % 20 == 0:
                S_k = np.abs(fft2(dn))**2 / L**4
                S_shell = S_k[shell].mean() if shell.sum() > 0 else 1e-30
                if S_shell > 1e-30:
                    log_S.append(np.log(S_shell))
        
        # Fit growth rate
        if len(log_S) > 20:
            times = np.arange(len(log_S)) * 20 * dt
            # Use latter half for fit (after transients)
            n_half = len(times) // 2
            if n_half > 5:
                p = np.polyfit(times[n_half:], log_S[n_half:], 1)
                rates.append(p[0] / 2)  # ω = (d/dt log S)/2
    
    if rates:
        return np.mean(rates), np.std(rates)/np.sqrt(len(rates))
    return 0.0, 1.0


# ── Measure for different L ──────────────────────────────────────────
L_values = [16, 24, 32, 48, 64]
print(f"{'L':>5s}  {'n_c(tree)':>10s}  {'n_c(1-loop)':>12s}  {'n_c(sim)':>10s}  {'ln(L)':>6s}")
print("-" * 55)

sim_results = []
for L in L_values:
    ell = math.log(L)
    n_c_1loop = n_c_tree - oneloop_C * kappa/(chi*mu_s) * ell
    
    # Binary search for critical density
    n_lo = 0.5 * n_c_tree
    n_hi = 1.3 * n_c_tree
    
    for bisect in range(10):
        n_mid = (n_lo + n_hi) / 2
        omega, omega_err = measure_growth_rate(L, n_mid, n_steps=3000, n_trials=15)
        if omega > 0.01:  # unstable
            n_hi = n_mid
        elif omega < -0.01:  # stable
            n_lo = n_mid
        else:
            break
    
    n_c_sim = (n_lo + n_hi) / 2
    print(f"{L:5d}  {n_c_tree:10.4f}  {n_c_1loop:12.4f}  {n_c_sim:10.4f}  {ell:6.2f}")
    sim_results.append((L, n_c_sim, n_c_1loop))

# ── Fit and compare ──────────────────────────────────────────────────
print()
if len(sim_results) >= 3:
    Ls = np.array([r[0] for r in sim_results])
    n_sims = np.array([r[1] for r in sim_results])
    n_1loops = np.array([r[2] for r in sim_results])
    log_Ls = np.log(Ls)
    
    # Fit n_c(sim) = A - B·ln(L)
    p_sim = np.polyfit(log_Ls, n_sims, 1)
    B_sim = -p_sim[0]
    B_theory = oneloop_C * kappa / (chi * mu_s)
    
    print("=" * 70)
    print("FIT RESULTS")
    print("=" * 70)
    print(f"  n_c(sim) = {p_sim[1]:.4f} - {B_sim:.4f}·ln(L)")
    print(f"  n_c(1-loop) = {n_c_tree:.4f} - {B_theory:.4f}·ln(L)")
    print()
    print(f"  Slope ratio B_sim/B_theory = {B_sim/B_theory:.3f}")
    print(f"  (1.0 = perfect agreement with one-loop prediction)")
    print()
    if abs(B_sim/B_theory - 1) < 0.3:
        print("  >>> CONSISTENT with one-loop prediction")
    elif B_sim/B_theory > 0:
        print(f"  >>> Qualitative agreement (both predict n_c decreases with L)")
    else:
        print("  >>> Disagreement — check simulation parameters")
