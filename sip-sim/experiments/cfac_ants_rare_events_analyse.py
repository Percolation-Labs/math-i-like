"""
Analyse the Gillespie MFPT data and fit the CFAC instanton action.

Data from cfac_ants_rare_events.py (g = 4.5, α = 1):
    V      τ          log τ
    30    17.42      2.858
    50    29.98      3.401
    80    61.03      4.111
   120   177.36      5.178
   180   807.97      6.695

CFAC predicts τ ~ A · V^p · exp(V S*) where S* = 0.0265.
Take the log:  log τ = log A + p log V + V S*
Fit this three-parameter model.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


V = np.array([30, 50, 80, 120, 180])
tau = np.array([17.42, 29.98, 61.03, 177.36, 807.97])
S_cfac = 0.0265

log_tau = np.log(tau)

# Pure linear fit: log τ = a + S V
# (ignores V^p prefactor)
a_lin, S_lin = np.polyfit(V, log_tau, 1)[::-1]
print(f"Linear fit (pure exponential): S = {S_lin:.4f}   "
      f"vs CFAC = {S_cfac:.4f}  (error {100*abs(S_lin-S_cfac)/S_cfac:.1f}%)")

# Three-parameter: log τ = log A + p log V + S V
# Fit as linear regression in [1, log V, V] with targets log τ
X = np.column_stack([np.ones_like(V, dtype=float), np.log(V), V])
coeffs = np.linalg.lstsq(X, log_tau, rcond=None)[0]
log_A, p_fit, S_fit = coeffs
print(f"\n3-param fit (with V^p prefactor): ")
print(f"  p = {p_fit:.3f} (exact theory: p = 3/2 for Kramers-type)")
print(f"  S = {S_fit:.4f}  vs CFAC = {S_cfac:.4f}  "
      f"(error {100*abs(S_fit-S_cfac)/S_cfac:.1f}%)")

# Constrained fit: force p = 3/2 and fit only (log A, S)
Y = log_tau - 1.5 * np.log(V)
a_con, S_con = np.polyfit(V, Y, 1)[::-1]
print(f"\nConstrained (p = 3/2): S = {S_con:.4f}  "
      f"(error {100*abs(S_con-S_cfac)/S_cfac:.1f}%)")

# Asymptotic slope: take pairs (V_i, V_{i+1}) and compute
# Δlog τ / ΔV, which converges to S as V → ∞
print(f"\nLocal slopes Δlogτ/ΔV (should converge to S* = 0.0265):")
for i in range(len(V)-1):
    dlog = log_tau[i+1] - log_tau[i]
    dV = V[i+1] - V[i]
    print(f"  V = {V[i]}→{V[i+1]}:  slope = {dlog/dV:.4f}")

# Plot
figdir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
    'figures_ants'))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                          constrained_layout=True)

ax = axes[0]
ax.semilogy(V, tau, 'bo', ms=11, mfc='none', mew=2, label='Gillespie')
Vgrid = np.linspace(V.min(), V.max(), 100)
# CFAC prediction with fitted overall amplitude
y_cfac = np.exp(log_A + S_cfac * Vgrid)
ax.semilogy(Vgrid, y_cfac, 'k-', lw=2,
            label=rf'CFAC: $\tau \propto e^{{S^* V}}$, $S^* = {S_cfac}$')
ax.set_xlabel(r'Volume $V$')
ax.set_ylabel(r'MFPT $\tau$')
ax.set_title(r'MFPT scales exponentially in $V$')
ax.legend(loc='upper left', frameon=False, fontsize=10)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
# Plot local slopes converging to S*
Vmid = 0.5 * (V[:-1] + V[1:])
local_slopes = np.diff(log_tau) / np.diff(V)
ax.plot(Vmid, local_slopes, 'bo-', ms=9, mfc='none', mew=2,
        label=rf'measured: fit $S = {S_lin:.4f}$')
ax.axhline(S_cfac, color='k', ls='--', lw=2,
           label=rf'CFAC: $S^* = {S_cfac}$')
ax.set_xlabel(r'Volume $V$')
ax.set_ylabel(r'local slope $\Delta\log\tau / \Delta V$')
ax.set_title(rf'Slope $\to S^* = {S_cfac}$  (measured $3.6\%$ error)')
ax.legend(loc='upper right', frameon=False, fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 0.11)

out = os.path.join(figdir, 'canonical_ant_instanton.pdf')
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out}")
