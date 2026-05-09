"""Verify that the two n=2, m=2 kernels in Gunnar's 9-May note both
evaluate to (-i*omega + V0 + 2*r)^{-1}, i.e. the kinematic kernel is
path-independent (stay/stay vs flip/flip)."""

import sympy as sp

omega, omega_p, V0, r = sp.symbols("omega omega_p V0 r", real=True, positive=True)

# Both integrands as written in Gunnar's note.
stay_integrand = 1 / (-sp.I * (omega + omega_p) + V0 + r) * 1 / (sp.I * omega_p + r)
flip_integrand = 1 / (-sp.I * (omega + omega_p) + r) * 1 / (sp.I * omega_p + V0 + r)

# Close the omega' contour in the upper half plane.
# Stay: pole at omega_p = i*r (UHP); residue gives 1/[i*(-i*omega + V0 + 2r)].
# Flip: pole at omega_p = i*(V0+r) (UHP); residue gives 1/[i*(-i*omega + V0 + 2r)].
# Integral = 2*pi*i * residue / (2*pi)   (with the d omega'/(2*pi) measure).

def upper_half_residue_integral(integrand, var, poles_uhp):
    total = 0
    for pole in poles_uhp:
        res = sp.simplify(sp.residue(integrand, var, pole))
        total += sp.I * res  # 2*pi*i * res / (2*pi)
    return sp.simplify(total)


stay_value = upper_half_residue_integral(stay_integrand, omega_p, [sp.I * r])
flip_value = upper_half_residue_integral(flip_integrand, omega_p, [sp.I * (V0 + r)])

target = 1 / (-sp.I * omega + V0 + 2 * r)

print("Stay/stay integral =", sp.simplify(stay_value))
print("Flip/flip integral =", sp.simplify(flip_value))
print("Target            =", sp.simplify(target))
print()
print("stay - target =", sp.simplify(stay_value - target))
print("flip - target =", sp.simplify(flip_value - target))

assert sp.simplify(stay_value - target) == 0
assert sp.simplify(flip_value - target) == 0
print("\nBoth kernels match (-i*omega + V0 + 2r)^{-1}.  Path-independence confirmed.")


# --- second check: the n=2 walk count is 2^{m-1} for m>=1, all positive. ---
print("\nLaplacian walk-count check at n=2 (uniform v_{01}=v):")
v = sp.symbols("v", positive=True)
L2 = sp.Matrix([[v, -v], [-v, v]])
for m in range(1, 7):
    Lm = L2**m
    val = sp.simplify(Lm[0, 0])
    pred = sp.Rational(1, 2) * (2 * v) ** m
    ok = sp.simplify(val - pred) == 0
    print(f"  m={m}: (L^m)_00 = {val}, predicted {pred}, match = {ok}")


# --- third check: at n=3, m=3, value is 18*v^3 (not zero), so "odd m cancels" is wrong. ---
print("\nLaplacian at n=3, m=3 (uniform v):")
L3 = sp.Matrix([[2*v, -v, -v], [-v, 2*v, -v], [-v, -v, 2*v]])
val = sp.simplify((L3**3)[0, 0])
pred = sp.Rational(2, 3) * (3 * v) ** 3
print(f"  (L^3)_00 = {val}, predicted {pred}, match = {sp.simplify(val - pred) == 0}")
print("  (Odd m does NOT cancel.)")
