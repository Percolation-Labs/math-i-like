# RDFT: Reaction-Diffusion Field Theory

From stoichiometry to critical exponents — via analytic combinatorics.

Enter a stoichiometry matrix. Get the Liouvillian, all Feynman diagrams,
graph polynomials, and critical exponents. Two routes: the **AC route**
(direct, from the singularity of the generating function) and the **RG route**
(scenic, via loop integrals and renormalisation). Both agree.

## Install

```bash
# Python (uv recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"

# Rust simulator (optional — needed for rdft simulate)
cd simulations && cargo build --release && cd ..
```

## CLI

```bash
# From a stoichiometry matrix — the generic interface
rdft stoichiometry '[[2,0]]'                                    # 2A → ∅
rdft stoichiometry '[[1,2],[1,0],[2,1]]' -r beta,epsilon,chi    # Gribov (BRW)
rdft stoichiometry '[[3,0]]'                                    # 3A → ∅ (novel)
rdft stoichiometry '[[1,1,0,0]]'                                # A+B → ∅

# Named processes
rdft analyze gribov
rdft analyze pair_annihilation

# Show interaction vertices (corollas)
rdft corollas gribov

# Full BRW worked example — reproduces Bordeu, Amarteifio+ (2019)
rdft brw

# Run all 15 literature CRNs through the AC pipeline
rdft survey
```

## Python API

```python
# One-liner: stoichiometry → exponents
from rdft import analyze
results = analyze('gribov')

# From a raw matrix
from rdft.core.reaction_network import ReactionNetwork
net = ReactionNetwork.from_stoichiometry(
    [[1,2], [1,0], [2,1]],
    rates=['beta', 'epsilon', 'chi'],
    name='Gribov'
)

# Named factories — 14 processes from the literature
net = ReactionNetwork.pair_annihilation()      # 2A → ∅
net = ReactionNetwork.gribov()                 # A→2A, A→∅, 2A→A  (BRW/DP)
net = ReactionNetwork.triplet_annihilation()   # 3A → ∅  (d_c = 1)
net = ReactionNetwork.k_particle_annihilation(4)  # 4A → ∅  (d_c = 2/3)
net = ReactionNetwork.contact_process()        # A→2A, 2A→∅  (DP class)
net = ReactionNetwork.schlogl_second()         # 2A→3A, A→∅  (DP class)
net = ReactionNetwork.schlogl_first()          # Schlögl I  (Ising class)
net = ReactionNetwork.barw_even()              # A→3A, 2A→∅  (parity-conserving)
net = ReactionNetwork.barw_odd()               # A→2A, 2A→∅  (DP class)
net = ReactionNetwork.pcpd()                   # 2A→3A, 2A→∅  (controversial)
net = ReactionNetwork.lotka_volterra()         # predator-prey (2 species)
net = ReactionNetwork.reversible_annihilation()  # 2A⇌C (2 species)

# Step-by-step access
from rdft.core.generators import Liouvillian
L = Liouvillian(net)
print(L.vertices)                           # interaction corollas

from rdft.ac.dse import ac_full_derivation
import sympy as sp
d = sp.Symbol('d', positive=True)
result = ac_full_derivation(L, d, p=1)
print(result['alpha_p'])                    # d/2 - 1
print(result['d_c'])                        # 4
print(result['singularity_type'])           # square_root
```

## What it computes

```
Stoichiometry matrix  S = [[k₁,l₁], [k₂,l₂], ...]
        ↓
Heisenberg-Weyl generator  Q = λ((z+1)^l - (z+1)^k)∂z^k
        ↓
Interaction vertices (corollas)  {φ̃^m φ^n : coupling}
        ↓                              ↓
   AC ROUTE (direct)            RG ROUTE (scenic)
        ↓                              ↓
   DSE kernel φ(G)              Wick contractions
        ↓                              ↓
   Branch point G*              Feynman diagrams
        ↓                              ↓
   Puiseux exponent p/q         Kirchhoff K(α), Symanzik Ψ
        ↓                              ↓
   Transfer theorem             Parametric integral I(G;d)
        ↓                              ↓
   + Weyl's law                 ε-expansion, Z-factors
        ↓                              ↓
   α_p = (pd - 2)/2            β(λ), fixed point λ*
        ↓                              ↓
        └──────── SAME ANSWER ─────────┘
                      ↓
              d → d_s (spectral dimension)
                      ↓
         Exponents on any graph (Sierpinski, random tree, ...)
```

## Simulation

The Rust simulator runs BRW particle simulations on arbitrary graphs and
compares scaling exponents with theory (Bordeu, Amarteifio+ 2019).

```bash
# From the CLI
rdft simulate --graph lattice:3:50 -n 10000 --tmax 3000
rdft simulate --suite              # full validation: 7 graph types
rdft simulate --graph sierpinski:4 -n 20000 --no-plot -o results.json

# From Python
from rdft import run_ensemble, run_brw
result = run_ensemble(graph="lattice:3:50", realizations=10000, t_max=3000)
results = run_brw()   # full suite

# Plot results
from rdft.simulate import plot_results
plot_results(results)
```

Graph types: `lattice:DIM:SIZE`, `sierpinski:LEVEL`, `tree:N:SEED`, `ba:N:M`

Parallelised across all CPU cores via [rayon](https://docs.rs/rayon).
Full 7-graph suite (~20k realizations each) completes in ~80s on Apple Silicon.

## Tests

```bash
uv run pytest tests/ -v     # 42 Python tests
cd simulations && cargo test # 9 Rust tests
```

## Paper

The companion paper is in `paper/main.tex` (compile with `tectonic main.tex`).
It proves that the AC route derives the BRW scaling exponents without
evaluating any momentum integrals, and validates against simulation data
from Bordeu, Amarteifio et al. (2019) on hypercubic lattices, Sierpinski
carpets, random trees, and scale-free networks.

## Singularity diagnostics

The pipeline includes automatic singularity diagnostics (`diagnose_singularity`)
that detect and explain when the standard AC route cannot be applied:

| Condition | Diagnosis | Example |
|-----------|-----------|---------|
| Quadratic phi, phi''(G*) ≠ 0 | Square-root branch → n^{-3/2} | DP class (Gribov, Contact) |
| All vertices have same n_in | Ward identity: d_c reduced by factor 2 | kA→∅ |
| Leading coeff vanishes at criticality | Branch → pole (parity conservation) | BARW-even (PC class) |
| Cubic phi, phi''(G*) = 0 at special ratio | Cube-root branch → n^{-4/3} | PCPD (at σ/λ ≈ 4.72) |

## References

- Amarteifio (2019) PhD thesis, Imperial College London
- Bordeu, Amarteifio et al. (2019) *Sci. Rep.* 9:15590
- Doi (1976) *J. Phys. A* 9:1465 — second quantisation
- Peliti (1985) *J. Phys. (Paris)* 46:1469 — coherent-state path integral
- Lee (1994) *J. Phys. A* 27:2633 — exact exponents for kA→∅
- Flajolet & Sedgewick (2009) *Analytic Combinatorics* — transfer theorem
- Connes & Kreimer (1998) *Comm. Math. Phys.* 199:203 — Hopf algebra of renormalisation
- Bogner & Weinzierl (2010) arXiv:1002.3458 — parametric representations
- Cardy & Täuber (1996) *PRL* 77:4780 — BARW parity-conserving class
- Hinrichsen (2000) *Adv. Phys.* 49:815 — absorbing state transitions review
- Täuber, Howard & Vollmayr-Lee (2005) *J. Phys. A* 38:R79 — field-theoretic RG for reaction-diffusion
- Schlögl (1972) *Z. Phys.* 253:147 — Schlögl models
