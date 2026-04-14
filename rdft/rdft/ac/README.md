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

## Module map

| Module | Purpose | Key API |
|---|---|---|
| `algebraic.py` | Newton polygon, Puiseux singularities of algebraic curves | `AlgebraicSingularity` |
| `lagrange.py` | Lagrange equations $G = z\,\phi(G)$, examples | `LagrangeEquation`, `pair_annihilation_dse` |
| `transfer.py` | Flajolet-Sedgewick transfer theorem | `Singularity`, `from_lagrange` |
| `dse.py` | DSE construction from a Liouvillian / vertex dict | `combinatorial_dse_kernel`, `coupled_dse`, `ac_full_derivation` |
| **`stratification.py`** | Puiseux strata $\mathcal{C}_k$ in coefficient space | `puiseux_order`, `canonical_family`, `phi_from_reactions`, `is_dyadic` |
| **`tilted.py`** | Current-tilted DSEs, SCGF, dynamical phase transitions | `scgf`, `tilted_phi`, `gallavotti_cohen_residual` |
| `bridge.py` | One-loop bridge functions, anomalous dimensions | `bridge_scalar`, `bridge_rank_k`, `gamma_k_anomalous`, `tau_dressed`, `one_loop_KS`, `one_loop_On` |
| `correspondence.py` | CRN-to-field-theory correspondence tables | `Correspondence` |
| `network_percolation.py` | Random-graph percolation generating functions | `erdos_renyi_kernel`, `cluster_size_exponent_theory` |

Bold modules are the new additions for the stratification programme.

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
