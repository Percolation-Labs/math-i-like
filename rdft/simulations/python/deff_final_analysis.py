#!/usr/bin/env python3
"""Final d_eff analysis: produce deliverable table + plot."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = '/Users/sirsh/code/math/rdft/simulations/results'
with open(os.path.join(OUT_DIR, 'deff_sierpinski_lo.json')) as f:
    data = json.load(f)

res = data['results']
print("Sierpinski gasket L=5, N={}, d_s analytic = {:.4f}".format(
    data['geometry']['N'], data['geometry']['d_s_analytic']))

# Table
print("\n{:>6s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s} {:>6s} {:>6s} {:>6s}".format(
    "delta", "a", "S_1", "Δf", "B_med", "B_mean", "f6", "f8", "f10"))
rows = []
for r in res:
    ffs = [r['per_V'][i]['flip_frac'] for i in range(3)]
    row = (r['delta'], 0.05, r['S1'], r['Delta_f'],
           r['B_median'], r['B_mean'], ffs[0], ffs[1], ffs[2])
    rows.append(row)
    print("{:>6.2f} {:>6.2f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>6.2f} {:>6.2f} {:>6.2f}".format(*row))

Df = np.array([r[3] for r in rows])
Bmed = np.array([r[4] for r in rows])
Bmean = np.array([r[5] for r in rows])
S1 = np.array([r[2] for r in rows])
flipmin = np.array([min(r[6], r[7], r[8]) for r in rows])

def fit_and_se(x, y):
    s, i = np.polyfit(x, y, 1)
    yhat = s*x + i
    resid = y - yhat
    n = len(x)
    s2 = (resid**2).sum() / max(n-2, 1)
    sx2 = ((x-x.mean())**2).sum()
    return s, i, float(np.sqrt(s2/sx2)) if sx2 > 0 else float('nan')

print("\n-- Fits of log(B) vs log(Δf) --")
for label, mask in [
    ('all 5',          np.ones(5, dtype=bool)),
    ('flipmin>0.3',    flipmin > 0.3),
    ('flipmin>=0.6',   flipmin >= 0.6),
    ('flipmin==1.0',   flipmin >= 0.99),
]:
    if mask.sum() >= 2:
        xs = np.log(Df[mask]); ys = np.log(Bmed[mask])
        sm, _, se = fit_and_se(xs, ys)
        sa, _, sea = fit_and_se(xs, np.log(Bmean[mask]))
        print(f"  [{label:<14s}, N={mask.sum()}] "
              f"slope(med)={sm:+.3f}±{se:.3f} → d_eff={1-sm:.2f}±{se:.2f} ; "
              f"slope(mean)={sa:+.3f}±{sea:.3f} → d_eff={1-sa:.2f}±{sea:.2f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
ax = axes[0]
sc = ax.scatter(Df, Bmed, c=flipmin, cmap='viridis', s=70, edgecolor='k', label='B (median)')
ax.scatter(Df, Bmean, marker='^', facecolor='none', edgecolor='red', s=60, label='B (mean)')
# Trend lines: slope -1 (d=2) and d_s=1.365 anchored at δ=1.80
anchor_df, anchor_b = Df[0], Bmed[0]
df_grid = np.logspace(np.log10(Df.min()*0.9), np.log10(Df.max()*1.1), 30)
for d_eff_ref, label, ls in [(1, 'd=1', ':'), (1.3652, 'd_s=1.365', '--'), (2, 'd=2', '-')]:
    b_grid = anchor_b * (anchor_df / df_grid) ** (d_eff_ref - 1)
    ax.plot(df_grid, b_grid, ls, label=label)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\Delta f$')
ax.set_ylabel(r'$B = \partial \log \tau/\partial V_\mathrm{eff}$')
ax.set_title(r'Sierpinski gasket L=5: CNT scaling of barrier')
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3, which='both')
plt.colorbar(sc, ax=ax, label='min flip frac')

ax = axes[1]
for r in res:
    Vs = np.array([p['V_eff'] for p in r['per_V']])
    taus = np.array([p['tau_mean'] for p in r['per_V']])
    ffs = np.array([p['flip_frac'] for p in r['per_V']])
    ax.plot(Vs, taus, 'o-', label=f"δ={r['delta']}, Δf={r['Delta_f']:.2f}")
ax.set_yscale('log')
ax.set_xlabel(r'$V_\mathrm{eff}$')
ax.set_ylabel(r'$\langle \tau \rangle$ (flipped trials)')
ax.set_title(r'Arrhenius: $\log \tau$ vs $V_\mathrm{eff}$')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'deff_sierpinski_scan.pdf')
plt.savefig(plot_path)
plt.savefig(plot_path.replace('.pdf', '.png'), dpi=140)
print(f"\nWrote {plot_path}")
