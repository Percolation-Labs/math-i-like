"""Independent verification of the diagrammatic claims in this note.

We do four checksums:

  (A) Enumerate every valid rail walk 0->0 of length m on n rails under
      Gunnar's local rule (each vertex must have the excitation on one of
      its two bridge rails; stay vertex contributes +v_{ij}, flip
      contributes -v_{ij}). Sum the symbol weights. Compare to the
      Laplacian matrix element (L^m)_{00}.

  (B) Reconstruct each Gunnar atlas diagram by hand from his rules
      (excitation flow + propagator-with-V0 vs propagator-bare on each
      segment), assemble the symbolic value, and compare to his stated
      per-diagram formula.

  (C) Compute every loop integral Gunnar writes by symbolic residues
      and check it equals what he claims it does.

  (D) Sanity limits: V0->0 (impurity off) and uniform v (closed form).

If anything fails, it's a sign that we (or Gunnar) have a bug. The
purpose is to catch it before sending the reply.
"""

import itertools
import sympy as sp

# -----------------------------------------------------------------------
# (C) Residue computation of Gunnar's kernels.
# -----------------------------------------------------------------------
print("=" * 72)
print("(C) Residue check of every kernel Gunnar writes.")
print("=" * 72)

omega, omega_p, V0, r = sp.symbols("omega omega_p V0 r", real=True, positive=True)


def integrate_via_residues(integrand, var, upper_half_poles):
    """Closed-contour integration in the upper half plane: int dvar/(2pi)."""
    total = 0
    for pole in upper_half_poles:
        res = sp.residue(integrand, var, pole)
        total += sp.I * sp.simplify(res)
    return sp.simplify(total)


# n=2, m=2, stay/stay
stay_22 = (1 / (-sp.I * (omega + omega_p) + V0 + r)) * (1 / (sp.I * omega_p + r))
stay_22_val = integrate_via_residues(stay_22, omega_p, [sp.I * r])
print(f"  n=2,m=2 stay/stay  -> {sp.simplify(stay_22_val)}  "
      f"(target {1/(-sp.I*omega+V0+2*r)})")
assert sp.simplify(stay_22_val - 1 / (-sp.I * omega + V0 + 2 * r)) == 0

# n=2, m=2, flip/flip
flip_22 = (1 / (-sp.I * (omega + omega_p) + r)) * (1 / (sp.I * omega_p + V0 + r))
flip_22_val = integrate_via_residues(flip_22, omega_p, [sp.I * (V0 + r)])
print(f"  n=2,m=2 flip/flip  -> {sp.simplify(flip_22_val)}  "
      f"(target {1/(-sp.I*omega+V0+2*r)})")
assert sp.simplify(flip_22_val - 1 / (-sp.I * omega + V0 + 2 * r)) == 0

# n=3, m=3, "stays" diagram (Gunnar's first triangle): integrand
# (i*w'+r)^{-2} (-i(w+w')+V0+r)^{-1}
stays_33 = (1 / (sp.I * omega_p + r) ** 2) * (1 / (-sp.I * (omega + omega_p) + V0 + r))
# Pole at omega_p = i*r is order 2; pick it up.
stays_33_val = integrate_via_residues(stays_33, omega_p, [sp.I * r])
target_33 = 1 / (-sp.I * omega + V0 + 2 * r) ** 2
print(f"  n=3,m=3 stays(triangle) -> {sp.simplify(stays_33_val)}  "
      f"(target {sp.simplify(target_33)})")
assert sp.simplify(stays_33_val - target_33) == 0

# n=3, m=3, "flips" diagram (Gunnar's second triangle): integrand
# (i*w'+V0+r)^{-1} (i*w'+r)^{-1} (-i(w+w')+r)^{-1}
flips_33 = (
    (1 / (sp.I * omega_p + V0 + r))
    * (1 / (sp.I * omega_p + r))
    * (1 / (-sp.I * (omega + omega_p) + r))
)
flips_33_val = integrate_via_residues(flips_33, omega_p, [sp.I * r, sp.I * (V0 + r)])
flips_33_simplified = sp.simplify(flips_33_val)
target_factored = 1 / ((-sp.I * omega + 2 * r) * (-sp.I * omega + V0 + 2 * r))
print(f"  n=3,m=3 flips(triangle) -> {flips_33_simplified}")
print(f"     factored             -> {sp.simplify(flips_33_simplified - target_factored)} "
      f"(should be 0 if integral = 1/[(-iw+2r)(-iw+V0+2r)])")
print(f"     vs Gunnar's stated   -> {sp.simplify(flips_33_simplified - target_33)} "
      f"(should be 0 if integral = (-iw+V0+2r)^-2)")
flips_matches_factored = sp.simplify(flips_33_simplified - target_factored) == 0
flips_matches_gunnar = sp.simplify(flips_33_simplified - target_33) == 0
print(f"     matches 1/[(-iw+2r)(-iw+V0+2r)] : {flips_matches_factored}")
print(f"     matches Gunnar's (-iw+V0+2r)^-2 : {flips_matches_gunnar}")
if flips_matches_factored and not flips_matches_gunnar:
    print()
    print("  *** DISCREPANCY: Gunnar's stated value (-iw+V0+2r)^-2 does NOT match")
    print("      the integrand he writes. The integrand actually evaluates to")
    print("      1/[(-iw+2r)(-iw+V0+2r)]. So either his integrand or his stated")
    print("      result is wrong for the n=3,m=3 'flips' triangle. ***")
print()
print("  (n=2,m=2 path-independence holds; n=3,m=3 path-independence does NOT")
print("   hold under Gunnar's two integrands as written.)")


# -----------------------------------------------------------------------
# (A) Brute-force walk enumeration vs (L^m)_00.
# -----------------------------------------------------------------------
print()
print("=" * 72)
print("(A) Brute-force enumeration vs Laplacian (L^m)_00.")
print("=" * 72)


def enumerate_walks(n, m, v_sym):
    """Sum signed symbol weights over every length-m walk 0->0 under
    Gunnar's local rule:
      at each step from rail a, either
        (stay) pick bridge target j != a, weight +v_{aj}, stay on a;
        (flip) pick rail b != a, weight -v_{ab}, move to b.
    """
    # Outer-product state machine.
    total = 0
    # We enumerate sequences of (move_type, partner_rail) of length m.
    # move_type in {'S', 'F'}; partner_rail is the bridge target (S) or
    # destination rail (F).
    for moves in itertools.product(["S", "F"], repeat=m):
        for partners in itertools.product(range(n), repeat=m):
            a = 0
            weight = 1
            valid = True
            for t in range(m):
                p = partners[t]
                if p == a:
                    valid = False
                    break
                key = tuple(sorted((a, p)))
                v = v_sym[key]
                if moves[t] == "S":
                    weight *= v
                    # excitation stays on a
                else:
                    weight *= -v
                    a = p
            if not valid:
                continue
            if a != 0:
                continue
            total += weight
    return sp.expand(total)


def laplacian_power(n, m, v_sym):
    L = sp.zeros(n, n)
    for a in range(n):
        for b in range(n):
            if a == b:
                L[a, a] = sum(v_sym[tuple(sorted((a, j)))] for j in range(n) if j != a)
            else:
                L[a, b] = -v_sym[tuple(sorted((a, b)))]
    return sp.expand((L**m)[0, 0])


for n in (2, 3):
    v_sym = {
        tuple(sorted((i, j))): sp.symbols(f"v{min(i,j)}{max(i,j)}")
        for i in range(n)
        for j in range(n)
        if i != j
    }
    for m in range(0, 5):
        if m == 0:
            enum = sp.Integer(1)
        else:
            enum = enumerate_walks(n, m, v_sym)
        lap = laplacian_power(n, m, v_sym)
        ok = sp.simplify(enum - lap) == 0
        marker = "OK " if ok else "FAIL"
        print(f"  [{marker}] n={n}, m={m}: enum = {enum},  (L^m)_00 = {lap}")
        assert ok, f"Mismatch at n={n}, m={m}"
print("  Walk enumeration matches the Laplacian formula at every (n,m) tested.")


# -----------------------------------------------------------------------
# (B) Reconstruct each Gunnar atlas diagram from his rules.
# -----------------------------------------------------------------------
print()
print("=" * 72)
print("(B) Reconstructing each Gunnar atlas diagram from his rules.")
print("=" * 72)

v01, v12, v02 = sp.symbols("v01 v12 v02", positive=True)

# n=2, m=0: just the excitation propagator.
g0 = 1 / (-sp.I * omega + V0 + r)
g_bare = 1 / (-sp.I * omega + r)

# n=2, m=1, stay (one bridge from rail 0 to rail 1)
gunnar_n2_m1 = g0 * v01 * g0
my_n2_m1 = g0 * v01 * g0  # +v_{01} for stay, two excitation propagators on rail 0 worldline
print(f"  n=2,m=1 stay:    Gunnar = {gunnar_n2_m1}")
print(f"                    mine   = {my_n2_m1}")
assert sp.simplify(gunnar_n2_m1 - my_n2_m1) == 0

# n=2, m=2, stay/stay
loop22 = 1 / (-sp.I * omega + V0 + 2 * r)
gunnar_n2_m2_stay = g0 * v01**2 * g0 * loop22
my_n2_m2_stay = g0 * (+v01) * (+v01) * g0 * loop22
print(f"  n=2,m=2 stay:    Gunnar = {sp.simplify(gunnar_n2_m2_stay)}")
print(f"                    mine   = {sp.simplify(my_n2_m2_stay)}")
assert sp.simplify(gunnar_n2_m2_stay - my_n2_m2_stay) == 0

# n=2, m=2, flip/flip
gunnar_n2_m2_flip = g0 * v01**2 * g0 * loop22  # (-v01)^2 = v01^2
my_n2_m2_flip = g0 * (-v01) * (-v01) * g0 * loop22
print(f"  n=2,m=2 flip:    Gunnar = {sp.simplify(gunnar_n2_m2_flip)}")
print(f"                    mine   = {sp.simplify(my_n2_m2_flip)}")
assert sp.simplify(gunnar_n2_m2_flip - my_n2_m2_flip) == 0

# n=3, m=2 (different rails, no loop)
gunnar_n3_m2 = g0 * v01 * g0 * v02 * g0
my_n3_m2 = g0 * (+v01) * g0 * (+v02) * g0
print(f"  n=3,m=2 (no loop): Gunnar = {sp.simplify(gunnar_n3_m2)}")
print(f"                      mine   = {sp.simplify(my_n3_m2)}")
assert sp.simplify(gunnar_n3_m2 - my_n3_m2) == 0

# n=3, m=3, "second" triangle (excitation cycles 0->1->2->0, three flips)
loop33 = 1 / (-sp.I * omega + V0 + 2 * r) ** 2
gunnar_n3_m3_flips = -g0 * v01 * v12 * v02 * g0 * loop33
my_n3_m3_flips = g0 * (-v01) * (-v12) * (-v02) * g0 * loop33
print(f"  n=3,m=3 cyclic (3 flips): Gunnar = {sp.simplify(gunnar_n3_m3_flips)}")
print(f"                             mine   = {sp.simplify(my_n3_m3_flips)}")
print(f"  (Symbol weight matches; but kernel (-iw+V0+2r)^-2 disagrees with")
print(f"   the residue computation of his integrand -- see (C) above.)")
assert sp.simplify(gunnar_n3_m3_flips - my_n3_m3_flips) == 0

# n=3, m=3, "first" triangle (Gunnar's stated weight = +v_{01}v_{12}v_{02})
# Under his strict rule "one rail at every vertex carries the excitation"
# this diagram is impossible: a vertex with bridge v_{12} requires the
# excitation to be on rail 1 or rail 2; but a sign +1 requires an even
# number of flips, and one cannot pass through all three triangle edges
# in even-flip steps starting and ending at rail 0. So if Gunnar's rule
# is strict, his first n=3,m=3 diagram is *not* a valid walk under that rule.
print()
print("  n=3,m=3 'stays' triangle: Gunnar's stated weight is +v_{01}v_{12}v_{02},")
print("  but no walk under the strict rule produces sign +1 on these three edges.")
print("  Enumeration of all 3-edge walks 0->0 on the cycle:")
seen_signs = {}
for moves in itertools.product(["S", "F"], repeat=3):
    for partners in itertools.product(range(3), repeat=3):
        a = 0
        signs = []
        used = []
        valid = True
        for t in range(3):
            p = partners[t]
            if p == a:
                valid = False
                break
            used.append(tuple(sorted((a, p))))
            if moves[t] == "S":
                signs.append(+1)
            else:
                signs.append(-1)
                a = p
        if not valid or a != 0:
            continue
        if sorted(used) == [(0, 1), (0, 2), (1, 2)]:
            sgn = 1
            for s in signs:
                sgn *= s
            print(f"    moves={moves}, partners={partners}, sign={sgn:+d}")
            seen_signs[(moves, partners)] = sgn

print()
print("  All such walks have sign -1 (three flips). So under the strict rule,")
print("  the coefficient of v_{01}*v_{02}*v_{12} in (L^3)_{00} is -2, and there")
print("  is *no* +1 contribution corresponding to Gunnar's first triangle as drawn.")
print()
print("  CRITICAL FLAG: Either")
print("    (i)  Gunnar's first n=3,m=3 diagram allows a 'spectator'")
print("         bath--bath vertex (the rule 'one rail must carry the excitation'")
print("         is locally relaxed when a closed loop touches the worldline at")
print("         entry/exit points), in which case (L^m)_00 is incomplete and")
print("         we are missing diagrams like this; OR")
print("    (ii) the first diagram is a relabelling of the second (cyclic walks")
print("         0->1->2->0 and 0->2->1->0), and Gunnar dropped the leading minus")
print("         sign in the first equation -- in which case sum at this monomial")
print("         is -2 v_{01}v_{02}v_{12} and matches (L^3)_{00}.")


# -----------------------------------------------------------------------
# (D) Sanity limits.
# -----------------------------------------------------------------------
print()
print("=" * 72)
print("(D) Sanity limits.")
print("=" * 72)

# V0 -> 0: excitation rail collapses to bare. The kernel for n=2,m=2
# becomes 1/(-i*omega + 2r), independent of stay vs flip.
stay_v0_zero = sp.simplify(stay_22_val.subs(V0, 0))
flip_v0_zero = sp.simplify(flip_22_val.subs(V0, 0))
print(f"  V0=0 stay kernel: {stay_v0_zero}")
print(f"  V0=0 flip kernel: {flip_v0_zero}")
assert sp.simplify(stay_v0_zero - flip_v0_zero) == 0

# Uniform v: (L^m)_00 = (n-1)/n * (n*v)^m for m >= 1.
v = sp.symbols("v", positive=True)
print()
for n in (2, 3, 4):
    L = sp.eye(n) * (n - 1) * v - (sp.ones(n, n) - sp.eye(n)) * v
    for m in range(1, 5):
        val = sp.simplify((L**m)[0, 0])
        pred = sp.Rational(n - 1, n) * (n * v) ** m
        ok = sp.simplify(val - pred) == 0
        print(f"  uniform n={n}, m={m}: (L^m)_00={val}, predicted {pred}, match={ok}")
        assert ok

print()
print("All checksums passed (with one critical flag noted in (B)).")
