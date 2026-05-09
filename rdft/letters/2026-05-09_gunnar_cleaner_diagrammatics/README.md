# 2026-05-09 — Cleaner diagrammatic rules (reply to Gunnar)

Response to Gunnar's note `Cleaner diagrammatic rules_ Note 9 May 2026.pdf`.

## TL;DR

The 7 May reply had two errors in §3 (the kernel half):
1. dropped the $V_0$ shift on the excitation rail's propagator;
2. claimed swap-and-return at $m{=}2$ has no loop integral — it does.

The combinatorial half (§2 and §4 of the 7 May note) is independent of
those errors and survives unchanged: $W^{(n)}_m = (L^m)_{00}$, the closed
form $\tfrac{n-1}{n}(nv)^m$, the resolvent $(1-zv)/(1-znv)$, and the
marking / Schur / continued-fraction toolkit.

The reply (a) reproduces Gunnar's atlas faithfully with the corrected
rules, (b) pinpoints where the 7 May misunderstanding sat, (c) addresses
his four asks, and (d) — added per ``how do we avoid embarrassment'' —
runs an independent four-checksum verification harness before sending.

## Findings from the verification harness

**Solid:** walk enumeration matches the Laplacian formula at every $(n,m)$
tested; uniform-$v$ closed form holds for $n\in\{2,3,4\}$,
$m\in\{1,\dots,4\}$; the four $n{=}2$ and $n{=}3,m{=}2$ atlas diagrams
match Gunnar's stated values bit-for-bit; the $n{=}3,m{=}3$ stays-only
kernel matches by independent residue computation.

**Two flagged issues in Gunnar's note (raised politely in the reply):**

1. **Sign on the first $n{=}3,m{=}3$ diagram.** Gunnar writes weight
   $+v_{01}v_{12}v_{20}$, but no $0\!\to\!0$ walk under the strict rule
   ``one of $i,j$ carries the excitation at vertex $v_{ij}$'' produces a
   $+$ sign on the three triangle edges. Either his rule is meant
   loosely (allowing a bath–bath spectator at $V_2$, in which case the
   Laplacian undercounts), or the leading sign is a slip.

2. **Second $n{=}3,m{=}3$ kernel.** His stated kernel
   $(-i\omega + V_0 + 2r)^{-2}$ does not match the integrand he writes
   for that diagram; the integrand actually evaluates to
   $\big[(-i\omega+2r)(-i\omega+V_0+2r)\big]^{-1}$. So either the
   integrand is missing a $V_0$ factor (consistent with our derivation
   that two of three loop legs carry the excitation in the cyclic walk
   and so should both be $V_0$-shifted) or the stated value is wrong.

## Files

- `note.tex` / `note.pdf` — main reply; §6 documents the verification.
- `verify_diagrammatics.py` — four checksums (A) walk enumeration vs
  Laplacian, (B) atlas reconstruction, (C) independent residue
  computation of every kernel, (D) sanity limits ($V_0 \to 0$ and
  uniform $v$).
- `verify_log.txt` — captured output of the verification run.
- `verify_kernel.py` — short standalone version of the $n{=}2,m{=}2$
  kernel and Laplacian checks (kept around for quick re-runs).

## Build

```
tectonic note.tex
```

## Run the checks

```
python3 verify_diagrammatics.py | tee verify_log.txt
```
