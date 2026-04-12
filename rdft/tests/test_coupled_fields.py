"""
Coupled DP-MSR proof-of-concept: protein aggregation (Knowles et al.)

Derives the half-time scaling t_{1/2} ~ m_0^{-gamma} from the coupled
ODE system (saddle-point equations of the coupled action).

Known results (Knowles et al., PNAS 2009):
  - Primary nucleation + elongation: gamma = n_c / 2
  - Secondary nucleation dominated:  gamma = (n_2 + 1) / 2

We integrate the coupled ODEs, extract t_{1/2}, and fit gamma.
"""
import numpy as np
from scipy.integrate import solve_ivp

# ── Model parameters ────────────────────────────────────────────────
k_n = 1e-4    # primary nucleation rate
k_plus = 1.0  # elongation rate (per fibril end)
k_2 = 0.0     # secondary nucleation rate (off for Case 1)
n_c = 2       # primary nucleus size
n_2 = 2       # secondary nucleus size


def coupled_odes(t, y, m0, k_n, k_plus, k_2, n_c, n_2):
    """
    Saddle-point equations of the coupled DP-MSR action.
    
    DP sector:  dP/dt = k_n * m^n_c + k_2 * m^n_2 * M   (fibril number)
    Coupling:   dM/dt = 2 * k_plus * m * P                (fibril mass)
    MSR sector: m = m0 - M                                (monomer conservation)
    """
    P, M = y
    m = max(m0 - M, 0.0)  # monomer concentration (MSR field)
    
    dP = k_n * m**n_c + k_2 * m**n_2 * M   # DP: nucleation events
    dM = 2.0 * k_plus * m * P               # coupling: growth depletes field
    
    return [dP, dM]


def find_half_time(m0, k_n, k_plus, k_2, n_c, n_2, t_max=1e6):
    """Integrate coupled system and find t where M(t) = m0/2."""
    sol = solve_ivp(
        coupled_odes, [0, t_max], [0.0, 0.0],
        args=(m0, k_n, k_plus, k_2, n_c, n_2),
        method='RK45', dense_output=True,
        rtol=1e-10, atol=1e-12, max_step=t_max/100
    )
    
    # Find t_{1/2} by interpolation
    M_vals = sol.y[1]
    target = m0 / 2.0
    
    for i in range(1, len(M_vals)):
        if M_vals[i] >= target:
            # Linear interpolation
            t1 = sol.t[i-1]
            t2 = sol.t[i]
            M1 = M_vals[i-1]
            M2 = M_vals[i]
            t_half = t1 + (target - M1) * (t2 - t1) / (M2 - M1)
            return t_half
    
    return None  # didn't reach half-time


def fit_scaling(m0_values, t_half_values):
    """Fit log(t_half) = -gamma * log(m0) + const."""
    log_m = np.log(m0_values)
    log_t = np.log(t_half_values)
    # Linear regression
    coeffs = np.polyfit(log_m, log_t, 1)
    gamma = -coeffs[0]
    return gamma


# ── Case 1: Primary nucleation only ────────────────────────────────
print("=" * 65)
print("CASE 1: Primary nucleation + elongation (k_2 = 0)")
print(f"  Prediction (Knowles et al.): gamma = n_c/2 = {n_c/2}")
print("=" * 65)

m0_values = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
t_half_primary = []

for m0 in m0_values:
    t_h = find_half_time(m0, k_n, k_plus, 0.0, n_c, n_2, t_max=1e7)
    t_half_primary.append(t_h)
    print(f"  m0 = {m0:6.1f}  =>  t_half = {t_h:.4f}")

gamma_primary = fit_scaling(m0_values, np.array(t_half_primary))
print(f"\n  Fitted gamma = {gamma_primary:.4f}")
print(f"  Theory gamma = {n_c/2:.4f}")
print(f"  Relative error = {abs(gamma_primary - n_c/2)/(n_c/2)*100:.2f}%")

# ── Case 2: Secondary nucleation dominated ─────────────────────────
print("\n" + "=" * 65)
print("CASE 2: Secondary nucleation dominated")
k_2_val = 1.0
n_2_val = 2
print(f"  Prediction (Knowles et al.): gamma = (n_2+1)/2 = {(n_2_val+1)/2}")
print("=" * 65)

t_half_secondary = []

for m0 in m0_values:
    t_h = find_half_time(m0, k_n, k_plus, k_2_val, n_c, n_2_val, t_max=1e7)
    t_half_secondary.append(t_h)
    print(f"  m0 = {m0:6.1f}  =>  t_half = {t_h:.4f}")

gamma_secondary = fit_scaling(m0_values, np.array(t_half_secondary))
print(f"\n  Fitted gamma = {gamma_secondary:.4f}")
print(f"  Theory gamma = {(n_2_val+1)/2:.4f}")
print(f"  Relative error = {abs(gamma_secondary - (n_2_val+1)/2)/((n_2_val+1)/2)*100:.2f}%")

# ── Case 3: Decoupled test (no monomer depletion) ──────────────────
print("\n" + "=" * 65)
print("CASE 3: DECOUPLED — no monomer depletion (DP sector alone)")
print("  This shows what you get WITHOUT the MSR feedback.")
print("=" * 65)

def decoupled_odes(t, y, m0, k_n, k_plus, n_c):
    """DP sector only: m = m0 always (no depletion)."""
    P, M = y
    dP = k_n * m0**n_c
    dM = 2.0 * k_plus * m0 * P
    return [dP, dM]

# With no depletion, M(t) = k_plus * k_n * m0^(n_c+1) * t^2  (parabolic)
# This NEVER saturates — there is no sigmoid, no half-time in the 
# physical sense.  M grows without bound (unphysical: more mass in 
# fibrils than total monomer).
print("  Analytical solution: M(t) = k_+ * k_n * m0^(n_c+1) * t^2")
print("  -> M grows WITHOUT BOUND (no saturation, no sigmoid)")
print("  -> The 'half-time' from this is UNPHYSICAL because M > m0")
print()

for m0 in [10.0, 50.0]:
    sol = solve_ivp(decoupled_odes, [0, 100], [0, 0],
                    args=(m0, k_n, k_plus, n_c),
                    dense_output=True, max_step=1.0)
    t_check = 50.0
    M_at_t = sol.sol(t_check)[1]
    print(f"  m0={m0}: M(t={t_check}) = {M_at_t:.1f} "
          f"(vs total monomer = {m0}) "
          f"{'*** UNPHYSICAL' if M_at_t > m0 else ''}")

print("\n  CONCLUSION: Without MSR feedback (monomer depletion),")
print("  the DP sector produces unphysical unbounded growth.")
print("  The sigmoid kinetics and t_{1/2} scaling REQUIRE the coupling.")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Primary nucleation:   gamma = {gamma_primary:.4f}  (theory: {n_c/2})")
print(f"  Secondary nucleation: gamma = {gamma_secondary:.4f}  (theory: {(n_2_val+1)/2})")
print("  Decoupled (DP only):  no physical half-time (unbounded growth)")
print()
print("  Both scaling laws require the DP-MSR coupling.")
print("  The product k_n * k_+ appears in t_{1/2} — one factor from")
print("  each sector, exactly as in the Keller-Segel case.")
