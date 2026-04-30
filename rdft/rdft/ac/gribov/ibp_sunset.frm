*
*  IBP reduction of the 2-loop sunset master integral via FORM.
*
*  Sunset: 2-loop self-energy with 3 propagators between 2 vertices.
*  Internal momenta k1, k2; the third propagator carries (p - k1 - k2).
*  All 3 propagators are massless: 1/(k1^2), 1/(k2^2), 1/((p-k1-k2)^2).
*
*  We use the standard IBP identity
*       int d^d k1 d^d k2  d/dk_i^mu  ( v^mu / D1^a1 D2^a2 D3^a3 ) = 0
*
*  for v^mu in {k1^mu, k2^mu, p^mu} and i in {1, 2}.  This generates
*  6 IBP relations.  We derive the algebraic recurrence on the
*  exponents (a1, a2, a3) and solve for the master.
*
*  At the symmetric Reggeon renormalisation point used in JT05, the
*  external invariant p^2 = mu^2 (we set mu = 1).
*
*  The well-known closed-form result (Smirnov 2012 Sec 6.2):
*
*      Sunset(1,1,1; p^2=1) = -(p^2)^{d-3} * Gamma(3-d)Gamma(d-2)^3 / Gamma(3(d-2)/2)
*
*  At d = 4-eps:
*      = -(1)^{1-eps} Gamma(eps-1)Gamma(2-eps)^3 / Gamma(3(2-eps)/2)
*      = -Gamma(eps-1)Gamma(2-eps)^3 / Gamma(3-3*eps/2)
*
*  Laurent expand in eps:
*      = -1/(2 eps^2)                  [leading]
*        - (3/2 - some_log)/eps         [at eps^{-1}]
*        + finite
*
*  This script computes the symbolic Laurent expansion and prints
*  the rational + log content.
*

* Symbols and dimensions
Symbol eps, p2;
Symbol a1, a2, a3;
Symbol L;          * = ln(p2/mu^2); at sym point p2 = mu^2 = 1, L=0
CFunction G;
CFunction Gamma;

* Sunset master at general (a1, a2, a3) at p^2 = 1, d = 4-eps:
*   I(a1, a2, a3; eps) = (-1)^{a1+a2+a3}
*                        * Gamma(a1+a2+a3 - d)
*                        * Gamma(d/2 - a1) Gamma(d/2 - a2) Gamma(d/2 - a3)
*                        / [Gamma(a1) Gamma(a2) Gamma(a3) Gamma(3 d/2 - a1-a2-a3)]
*                        * (p2)^{d - a1-a2-a3}

Local sun = G(1,1,1);

* Substitute the closed form for G(a1,a2,a3):
Identify G(a1?, a2?, a3?) =
       sign_(a1+a2+a3)
     * Gamma(a1+a2+a3 - (4-eps))
     * Gamma((4-eps)/2 - a1)
     * Gamma((4-eps)/2 - a2)
     * Gamma((4-eps)/2 - a3)
     / Gamma(a1) / Gamma(a2) / Gamma(a3)
     / Gamma(3*(4-eps)/2 - a1-a2-a3);

Print +s sun;

.end
