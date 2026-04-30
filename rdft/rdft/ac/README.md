# `rdft.ac` — Analytic Combinatorics for Doi-Peliti DSEs

This package implements the Coupled Field Analytic Combinatorics (CFAC) framework.
The starting point is that any Doi-Peliti reaction-network DSE has the
Lagrange form

  $G = z\,\phi(G)$

with $\phi$ a polynomial whose coefficients encode the reaction-network
vertex structure.  The package provides tools to:

1. Build $\phi(G)$ from a reaction list (Doi-Peliti vertex extraction).
2. Find the dominant Puiseux branch of $G(z)$ as an algebraic curve.
3. Stratify the coefficient space of $\phi$ by Puiseux order.
4. Tilt the DSE by stochastic-thermodynamics observables (currents,
   entropy production) and read off the SCGF.
5. Compute one-loop bridge functions for $d$-dimensional dressing of
   the mean-field skeleton.

## Module map (tier-structured)

Modules are grouped by **tier** — see each module's docstring for the
`Tier: 1 | 2 | 3` header.  Tier-1 symbols are re-exported at the
`rdft.ac` package level; Tier-2 and Tier-3 modules are importable via
`from rdft.ac.<module> import ...`.

### Tier 1 — core DSE pipeline (6 modules)

These are the fundamental modules every CFAC workflow touches.

| Module | Purpose | Key API |
|---|---|---|
| `dse.py` | DSE construction from a Liouvillian / vertex dict | `combinatorial_dse_kernel`, `coupled_dse`, `ac_full_derivation`, `dse_from_liouvillian` |
| `lagrange.py` | Lagrange equations $G = z\,\phi(G)$, canonical examples | `LagrangeEquation`, `pair_annihilation_dse`, `sir_epidemic` |
| `transfer.py` | Flajolet-Sedgewick transfer theorem | `Singularity`, `from_lagrange` |
| `algebraic.py` | Newton polygon, Puiseux singularities of algebraic curves | `NewtonPolygon`, `AlgebraicSingularity` |
| `stratification.py` | Puiseux strata $\mathcal{C}_k$ in coefficient space | `puiseux_order`, `canonical_family`, `phi_from_reactions`, `is_dyadic` |
| `bridge.py` | One-loop bridge functions, anomalous dimensions | `bridge_scalar`, `bridge_rank_k`, `gamma_k_anomalous`, `tau_dressed`, `one_loop_KS`, `one_loop_On` |

### Tier 2 — documented extensions (15 modules)

Extensions with proofs, tests, and associated papers.  Imported explicitly.

| Module | Purpose | Key API |
|---|---|---|
| `admissible.py` | Stratification for transcendental (non-polynomial) kernels | `admissible_asymptotics`, `find_critical_point` |
| `multivariate.py` | Multivariate ACSV (Pemantle-Wilson) for coupled DSEs | `classify_singular_point`, `find_critical_point_multivariate` |
| `log_corrections.py` | Algebraic-logarithmic asymptotics | `transfer_theorem_log_corrected`, `detect_marginal_tuning` |
| `tilted.py` | Tilted Doi-Peliti DSEs, SCGF, dynamical phase transitions | `scgf`, `tilted_phi`, `gallavotti_cohen_residual` |
| `hopf_flow.py` | Connes-Kreimer Hopf algebra for RG flow | (forest algebra + antipode) |
| `ope.py` | Composite-operator anomalous dimensions via pointed GFs | (OPE coefficients) |
| `eft_matching.py` | Integrate out heavy modes, produce effective light-sector DSE | (EFT Wilson coefficients) |
| `multipoint.py` | Multi-point correlations via n-variable CFAC | (n-point generators) |
| `signed_projector.py` | Microscopic $\mathcal{C}_3$ derivation from conservation Ward identity | (signed projector) |
| `manna_full.py` | Manna/CDP exponents: full CFAC pipeline (counting × bridge × algebra) | (CDP one-loop) |
| `manna_2loop.py` | Two-loop CFAC for CDP/Manna with dimensionless couplings, Padé | (2-loop FRG-analog) |
| `manna_depinning.py` | Le Doussal-Wiese mapping to qEW depinning | (qEW mapping) |
| `conserved.py` (v1) | NESS projection for CDP 2-field DSE (scoping) | `coupled_dse_conserved`, `cdp_one_loop_dressing` |
| `conserved_2.py` (v2) | Soft-mode-induced quartic for CDP activity DSE | `cdp_dse_with_softmode`, `gamma_3_from_softmode` |
| `conserved_3.py` (v3) | Non-local Ward-identity treatment; C_3 demonstration | `find_C3_multicritical_line`, `manna_C3_complete` |

### Tier 3 — research / speculative / failure-mode (18 modules)

Exploratory modules.  Not re-exported.  Some document explicit failure modes
(kept as audit trail for papers).

| Module | Purpose |
|---|---|
| `correspondence.py` | AC ↔ QFT correspondence table generator |
| `lerw.py` | LERW as Kirchhoff enumeration (Prop 1) |
| `lerw_dirichlet.py` | LERW on Z^d boxes with Dirichlet boundary |
| `lerw_exact.py` | Deterministic LERW mean length from Kirchhoff ratio |
| `lerw_extrap.py` | Algebraic extrapolation schemes for d_f from finite-size data |
| `lerw_hierarchical.py` | LERW on Migdal-Kadanoff hierarchical lattices |
| `lerw_scaling.py` | Finite-size scaling of LERW on Z^d tori |
| `lerw_tube.py` | LERW on Z^3 tubes (failure mode: quasi-1D collapse) |
| `replica.py` | Replicated directed polymer as n-species CRN (Prop 3) |
| `replica_closed.py` | Closed-form 2-body binding for KPZ replica on Z |
| `replica_cubic.py` | Cubic-in-n analysis of the KPZ replica rate |
| `replica_transfer.py` | Transfer-matrix evaluation of KPZ n-walker replica CRN |
| `tap_complexity.py` | Planar (genus-0) TAP complexity from semicircle resolvent |
| `matrix_model.py` | Ribbon-graph enumeration for TAP complexity (Prop 2) |
| `network_percolation.py` | AC for percolation on configuration-model networks |
| `sandpile_group.py` | Sandpile-group animal for LERW exponent (failure mode) |
| `topology_sketch.py` | Speculative: CFAC for Gromov-Witten / Donaldson-Thomas |
| `manna_dp_anchor_DEPRECATED.py` | Documented wrong-path for Manna; regression-tested |

## Conventions

### Doi-Peliti vertex extraction

For a reaction $k\,A \to l\,A$ at rate $\lambda$, the generator is
$Q = \lambda\bigl((z+1)^l - (z+1)^k\bigr)\,\partial_z^k$.  The expansion
in $z$ gives vertices indexed by $(j, k)$ with weight
$\lambda(\binom{l}{j} - \binom{k}{j})$.

Vertices with $j+k=2$ (mass-type) modify the bare propagator $G_0$, NOT
$\phi$.  Vertices with $j+k\geq 3$ contribute $g_{j,k}\,G^{j+k-2}$ to
$\phi(G)$.  Use `phi_from_reactions(reactions)` for the full extraction;
this is the right answer.

### Stratification

The locus $\mathcal{C}_k$ of $1/k$-Puiseux dominance is codimension-$(k-2)$
in the coefficient space of $\phi$.  The canonical carrier family
$\phi_{k,\beta}(G) = (1+G)^k + \beta G$ realises every $\mathcal{C}_k$
for $\beta < 0$ with $|\beta|$ above a $k$-dependent threshold (see
Theorem A.2 of `paper/cfac/cfac_theorem.tex`).

### 0-d skeleton vs $d$-dimensional dressing

Every CFAC prediction comes in two layers (Theorem A.2):

  $\tau_k(d) = (1 + 1/k) + \gamma_k(d)$,   $\gamma_k(d) = O(d_c - d)$

The 0-d skeleton ($1+1/k$) is dimension-free and given by the
stratification.  The $d$-dimensional dressing ($\gamma_k(d)$) is
computed via bridge functions.  Both layers are CFAC content and
should always be reported together.  $d$ may be a fractal spectral
dimension $d_s$ — the dressing extends without modification.

## Workflow examples

```python
# 1. Stratify a CRN
from rdft.ac.stratification import phi_from_reactions, puiseux_order

reactions = [(1, 3, 3.0), (2, 1, 2.0), (2, 3, 1.0)]  # A->3A, 2A->A, 2A->3A
phi = phi_from_reactions(reactions)
k_dom, z_star = puiseux_order(phi)
print(f"phi = {phi},  k_dom = {k_dom},  |z*| = {abs(z_star):.4f}")

# 2. Tilt the SCGF
from rdft.ac.tilted import scgf
import numpy as np

s_arr, lam = scgf(reactions, tilt_indices=[0], s_values=np.linspace(-0.4, 0.4, 41))

# 3. d-dimensional dressing
from rdft.ac.bridge import tau_dressed

for d in [6, 4, 3, 2, 1]:
    print(f"d={d}: tau_3(d) = {tau_dressed(3, d):.4f}")
```

## Test coverage

| Test file | What it locks in |
|---|---|
| `tests/test_stratification_module.py` | Stratification API, canonical family, Doi-Peliti extraction, Banderier-Drmota status |
| `tests/test_tilted_module.py` | Tilted phi, SCGF, dynamical phase transitions |
| `tests/test_bridge_rank_k.py` | Rank-$k$ bridge constants, anomalous dimensions, $\tau_{\rm dressed}$ |
| `tests/test_stratification.py` | Theorem A.2 numerical and symbolic checks |
| `tests/test_cube_root_crn.py` | The cube-root CRN worked example (notes the published-claim correction) |
| `tests/test_saw_n_to_zero.py` | SAW one-loop via $n\to 0$ O(n) |

## Probes versus experiments

- `simulations/python/experiments/` — self-contained tests of CFAC predictions
  that produce concrete numerical results plotted in the papers.  Treat each
  as a numerical theorem statement.
- `simulations/python/probes/` — heuristic / scoping calculations used to
  build intuition or estimate orders of magnitude.  May contain back-of-envelope
  formulas; results to be taken qualitatively.

When a probe matures into a real prediction, port its logic into a library
module and write a proper experiment for it.
