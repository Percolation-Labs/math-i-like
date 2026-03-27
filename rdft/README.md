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

# Named factories
net = ReactionNetwork.pair_annihilation()   # 2A → ∅
net = ReactionNetwork.gribov()              # A→2A, A→∅, 2A→A
net = ReactionNetwork.brw_full()            # full two-species with tracers
net = ReactionNetwork.contact_process()     # directed percolation class

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

## References

- Amarteifio (2019) PhD thesis, Imperial College London
- Bordeu, Amarteifio et al. (2019) *Sci. Rep.* 9:15590
- Doi (1976) *J. Phys. A* 9:1465 — second quantisation
- Peliti (1985) *J. Phys. (Paris)* 46:1469 — coherent-state path integral
- Lee (1994) *J. Phys. A* 27:2633 — exact exponents for kA→∅
- Flajolet & Sedgewick (2009) *Analytic Combinatorics* — transfer theorem
- Connes & Kreimer (1998) *Comm. Math. Phys.* 199:203 — Hopf algebra of renormalisation
- Bogner & Weinzierl (2010) arXiv:1002.3458 — parametric representations
