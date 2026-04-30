"""Quick test: build resultant, find complex branch points."""
import sys
sys.path.insert(0, '/Users/sirsh/code/math/rdft/paper/wip')
from complex_branch_gap import build_resultant, discriminant_X, complex_branch_points, X, Z, xYZ_solve_at_z
import sympy as sp
import time

for alpha in [0.0, 0.1, 0.3, 0.5, 0.7]:
    t0 = time.time()
    F0, F1, R = build_resultant(alpha)
    deg_x = sp.Poly(R, X).degree()
    deg_z = sp.Poly(R, Z).degree()
    disc = discriminant_X(R)
    zbps = complex_branch_points(disc)
    zbps = [z for z in zbps if abs(z - 1.0) < 20]
    zbps.sort(key=lambda z: abs(z - 1.0))
    print(f"alpha={alpha:.2f} | deg_x(R)={deg_x}, deg_z(R)={deg_z} | "
          f"#bps={len(zbps)}, nearest 3:")
    for z in zbps[:5]:
        mark = " (REAL)" if abs(z.imag) < 1e-6 else ""
        print(f"  z = {z.real:+.5f} {z.imag:+.5f}i  |z-1|={abs(z-1):.4f}{mark}")
    # solutions at z=1
    sols = xYZ_solve_at_z(F0, F1, 1.0 + 0j, R_poly=R)
    print(f"  #(X,Y)-sols at z=1: {len(sols)}")
    sym = [(xr, yr) for (xr, yr) in sols if abs(xr-yr)<1e-3 and abs(xr.imag)<1e-3]
    print(f"  symmetric real sols: {sorted([round(s[0].real,5) for s in sym])}")
    print(f"  time={time.time()-t0:.1f}s")
    print()
