# 2026-05-07 — More messy diagrammatics (reply to Gunnar)

Response to Gunnar's note of 7 May 2026 (`More messy diagrammatics_ Note 7
May 2026.pdf`), in which he challenges us to extend the $n=2$ rung-count
$\frac{1}{2}(2v_1)^m$ to general $n$-rail diagrams and to fold the loop
integrals into the AC framework.

## TL;DR

The transfer operator for the "1's path through the $n$ rails" is the
weighted graph Laplacian $L = D - A$ on $K_n$ with edge weights $v_{ij}$.
The order-$m$ weighted sum is $W^{(n)}_m = (L^m)_{00}$. For uniform
$v_{ij} \equiv v$ this collapses to

$$W^{(n)}_m = \frac{n-1}{n}(nv)^m,$$

extending Gunnar's $\frac{1}{2}(2v_1)^m$ for $n=2$ to $\frac{2}{3}(3v)^m$,
$\frac{3}{4}(4v)^m$, etc.

The loop integrals factor cleanly: each diagram = (Laplacian path weight) ×
(kinematic kernel for its chunk-length pattern). Two walks with the same
chunk pattern share their kernel, so the kernel library is bounded by the
partition number $p(m+1)$ — a $(\text{very large}) \to (\text{partitions})$
contraction.

## Files

- `note.tex` / `note.pdf` — main response document.
- `poc_laplacian.py` — base verification (numerical + symbolic):
  - Test 1: uniform-$v$ enumeration matches $\frac{n-1}{n}n^m$ for
    $n \in \{2,3,4,5\}$, $m \in \{0,\dots,7\}$.
  - Test 2: symbolic-$v_{ij}$ enumeration at $n=3$ matches $(L^m)_{00}$
    polynomial-by-polynomial up to $m=4$.
  - Test 3: chunk-length-pattern decomposition at $n=3$, $m=5$ — eight
    distinct chunk patterns, signed multiplicities sum to
    $\frac{2}{3} \cdot 3^5 = 162$.
- `verify_shining.py` — verifies every "shining a light" example in §4
  of the note against direct enumeration / SymPy:
  - (1') resolvent $\mathcal G^{(3)}(z) = (1-zv)/(1-3zv)$ and its
    $z$-expansion;
  - (2') first-return $F^{(3)}(z) = 2zv/(1-zv)$;
  - (3') channel marking $v_{01}$ at $m=2$;
  - (4') sub-rail focus, foreground-mixed-background decomposition;
  - (5') forbidden $v_{12}=0$ at $m=3$, plus the killed-terms
    factorisation $v_{12}(v_{01}-v_{02})^2$;
  - (6') Schur reduction including intermediate $\Sigma(z)$ and
    $L^{\text{eff}}_{00}(z)$.

## Build

```
tectonic note.tex
```

## Run the PoC

```
python3 poc_laplacian.py
```
