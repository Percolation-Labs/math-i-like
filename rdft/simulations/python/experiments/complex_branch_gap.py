"""
Test whether the branch-gap nucleation formula extends to moderate coupling
on a spatial lattice when the nucleation saddle migrates into the complex plane.

Setup: 3-site coupled ring with site-coupling alpha,
    w+(x_i) = a + lam * [(1-alpha) x_i^2 + alpha * (x_{i-1}^2 + x_{i+1}^2)/2]
    w-(x_i) = delta x_i + d x_i^3
    a=0.1, lam=5, delta=4, d=1.
Symmetric ansatz x_1 = x_2 = Y, x_0 = X.

For each site the DSE reads (cumulant-generating, z-dressed):
    wm(x) - z * wp(x) = 0
which is the usual z-dressed fixed-point equation whose real branch point
in z gives the single-site barrier.

On the lattice we have two coupled polynomial equations in X, Y (with z),
F_0(X, Y; z) = 0  (site 0, neighbors are 1,2 -> Y,Y)
F_1(X, Y; z) = 0  (site 1, neighbors are 0,2 -> X,Y)
(Site 2 is identical to site 1 by the ansatz.)

Eliminate Y via resultant:
    R(X, z) = Res_Y(F_0, F_1)
Branch points in z are zeros of discriminant_X(R).
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad
import mpmath as mp
import time
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

# -------------------- model --------------------
A_ = 0.1
LAM = 5.0
DEL = 4.0
D_ = 1.0

X, Y, Z = sp.symbols('X Y z', complex=True)


def build_resultant(alpha_val):
    """Build F_0, F_1 and R(X, z) = Res_Y(F_0, F_1) for given alpha."""
    alpha = sp.Rational(int(round(alpha_val * 10000)), 10000)
    a = sp.Rational(1, 10)
    lam = sp.Integer(5)
    delta = sp.Integer(4)
    d = sp.Integer(1)

    # Site 0: neighbors 1,2 -> Y,Y, self X
    wp0 = a + lam * ((1 - alpha) * X ** 2 + alpha * (Y ** 2 + Y ** 2) / 2)
    wm0 = delta * X + d * X ** 3
    F0 = sp.expand(wm0 - Z * wp0)

    # Site 1: self Y, neighbors 0,2 -> X,Y
    wp1 = a + lam * ((1 - alpha) * Y ** 2 + alpha * (X ** 2 + Y ** 2) / 2)
    wm1 = delta * Y + d * Y ** 3
    F1 = sp.expand(wm1 - Z * wp1)

    # Resultant in Y -> polynomial in X, z
    R = sp.resultant(F0, F1, Y)
    R = sp.expand(R)
    return F0, F1, R


def discriminant_X(R_poly):
    """Discriminant of R as polynomial in X -> polynomial in z alone.
    First take square-free part in X to avoid the decoupled-limit issue where
    R = F_0^3 (with F_0 not containing Y) has an identically-zero discriminant.
    """
    Rp = sp.Poly(R_poly, X)
    Rsqf = Rp.sqf_part()
    disc = sp.discriminant(Rsqf, X)
    return sp.expand(disc)


def complex_branch_points(disc_poly, z_min=0.01, z_max=50):
    """Find all roots of disc(z) = 0 in the complex z-plane using numpy."""
    dp = sp.Poly(disc_poly, Z)
    # Take square-free part to remove multiplicities
    dp = dp.sqf_part()
    coeffs = [complex(c) for c in dp.all_coeffs()]
    # Normalize
    roots = np.roots(coeffs)
    out = []
    for r in roots:
        rc = complex(r)
        if abs(rc) > 1e-12 and abs(rc) < 1e6:
            out.append(rc)
    return out


def compile_solver(F0, F1, R):
    """Precompile fast numeric routines:
      - R_xcoefs_of_z(z) -> list of complex coefficients of R as poly in X
      - F1Y_coefs_of_xz(x, z) -> complex coefs of polynomial in Y (F_1 at X=x, z fixed)
      - F0_at_xyz(x, y, z) -> complex value of F_0
    Uses sympy.lambdify with numpy backend (no per-call subs)."""
    # coefs of R(X,z) as polynomial in X -> each a polynomial in z
    Rx = sp.Poly(R, X)
    degR = Rx.degree()
    Rx_coeffs_z = [Rx.nth(k) for k in range(degR, -1, -1)]  # highest X first
    Rx_funcs = [sp.lambdify(Z, c, modules='numpy') for c in Rx_coeffs_z]

    # F_1 as poly in Y (coefs depend on X, z)
    F1Y = sp.Poly(F1, Y)
    degY1 = F1Y.degree()
    F1_coeffs_xz = [F1Y.nth(k) for k in range(degY1, -1, -1)]
    F1_funcs = [sp.lambdify((X, Z), c, modules='numpy') for c in F1_coeffs_xz]

    # F_0 as function of (X,Y,Z)
    F0_fn = sp.lambdify((X, Y, Z), F0, modules='numpy')

    def R_xcoefs(zc):
        return [complex(f(zc)) for f in Rx_funcs]

    def F1_ycoefs(xc, zc):
        return [complex(f(xc, zc)) for f in F1_funcs]

    def F0_val(xc, yc, zc):
        return complex(F0_fn(xc, yc, zc))

    return {
        'R_xcoefs': R_xcoefs,
        'F1_ycoefs': F1_ycoefs,
        'F0_val': F0_val,
        'degR': degR,
        'degY1': degY1,
    }


def xYZ_solve_fast(solver, z_val, tol=1e-4):
    """Fast coupled solve using precompiled solver. Returns list of (X,Y)."""
    zc = complex(z_val)
    Rc = solver['R_xcoefs'](zc)
    # normalize leading zero issue
    while len(Rc) > 1 and abs(Rc[0]) < 1e-300:
        Rc = Rc[1:]
    Xroots = np.roots(Rc) if len(Rc) > 1 else []
    out = []
    for xr in Xroots:
        xc = complex(xr)
        Yc = solver['F1_ycoefs'](xc, zc)
        while len(Yc) > 1 and abs(Yc[0]) < 1e-300:
            Yc = Yc[1:]
        if len(Yc) < 2:
            continue
        Yroots = np.roots(Yc)
        for yr in Yroots:
            yc = complex(yr)
            val = solver['F0_val'](xc, yc, zc)
            scale = 1.0 + abs(xc) ** 3 + abs(yc) ** 3
            if abs(val) / scale < tol:
                out.append((xc, yc))
    return out


def xYZ_solve_at_z(F0, F1, z_val, R_poly=None, tol=1e-4, solver=None):
    """Compatibility wrapper: if solver is provided, use the fast path."""
    if solver is not None:
        return xYZ_solve_fast(solver, z_val, tol=tol)
    # slow sympy path (kept for compatibility with earlier debug scripts)
    zc = complex(z_val)
    if R_poly is not None:
        Rx = sp.Poly(R_poly, X)
        coeffs = []
        deg = Rx.degree()
        for k in range(deg, -1, -1):
            c = Rx.nth(k)
            c = complex(c.subs(Z, zc)) if isinstance(c, sp.Expr) else complex(c)
            coeffs.append(c)
        Xroots = np.roots(coeffs)
    else:
        return []
    F0z = F0.subs(Z, zc)
    F1z = F1.subs(Z, zc)
    F1_has_Y = F1.has(Y)
    primary = F1z if F1_has_Y else F0z
    check = F0z if F1_has_Y else F1z
    out = []
    for xr in Xroots:
        xc = complex(xr)
        pY = primary.subs(X, xc)
        polyY = sp.Poly(sp.expand(pY), Y)
        Ycoefs = [complex(c) for c in polyY.all_coeffs()]
        if len(Ycoefs) < 2:
            pY2 = check.subs(X, xc) if check.has(X) else check
            polyY = sp.Poly(sp.expand(pY2), Y)
            Ycoefs = [complex(c) for c in polyY.all_coeffs()]
            Yroots = np.roots(Ycoefs) if len(Ycoefs) >= 2 else []
            for yr in Yroots:
                out.append((xc, complex(yr)))
            continue
        Yroots = np.roots(Ycoefs)
        for yr in Yroots:
            yc = complex(yr)
            val = complex(check.subs({X: xc, Y: yc}))
            scale = 1.0 + abs(xc) ** 3 + abs(yc) ** 3
            if abs(val) / scale < tol:
                out.append((xc, yc))
    return out


# -------------------- real branch-gap (baseline / sanity) --------------------
def real_branches_of_R(R_poly, z_val, n_symm_only=True):
    """Return real X-roots of R(X, z) at real z."""
    if abs(z_val.imag) > 1e-14:
        return None
    Rp = sp.Poly(R_poly.subs(Z, float(z_val.real)), X)
    Xroots = Rp.nroots(n=30)
    real = sorted([complex(r).real for r in Xroots if abs(complex(r).imag) < 1e-6])
    return real


# -------------------- complex-contour branch-gap integral --------------------
def x_branches_coupled(F0, F1, z_val):
    """Return all (X,Y) solutions to coupled system at complex z.
    Select 'stable' (low-mean-field-like) and 'unstable' X branches by tracking
    from z=1 via continuation in a deformation."""
    return xYZ_solve_at_z(F0, F1, z_val)


def track_branches(F0, F1, R, z_start, z_end, x_stab_1, y_stab_1,
                   x_unst_1, y_unst_1, n_steps=400, path='straight',
                   solver=None):
    """Track (X,Y) solutions from z_start to z_end along a path.
    Return arrays of zs, x_stab, y_stab, x_unst, y_unst, with |x_stab - x_unst|
    at each step (useful to detect branch-merging)."""
    if path == 'straight':
        pathfn = lambda s: z_start + (z_end - z_start) * s
        dz_ds = lambda s: (z_end - z_start)
    elif path == 'upper':
        zm = 0.5 * (z_start + z_end) + 1j * 0.5 * abs(z_end - z_start)
        pathfn = lambda s: (1 - s) ** 2 * z_start + 2 * (1 - s) * s * zm + s ** 2 * z_end
        dz_ds = lambda s: 2 * (1 - s) * (zm - z_start) + 2 * s * (z_end - zm)
    elif path == 'lower':
        zm = 0.5 * (z_start + z_end) - 1j * 0.5 * abs(z_end - z_start)
        pathfn = lambda s: (1 - s) ** 2 * z_start + 2 * (1 - s) * s * zm + s ** 2 * z_end
        dz_ds = lambda s: 2 * (1 - s) * (zm - z_start) + 2 * s * (z_end - zm)
    else:
        raise ValueError(path)

    s_vals = np.linspace(0, 1, n_steps + 1)
    z_vals = np.array([pathfn(s) for s in s_vals])

    x_stab = complex(x_stab_1)
    y_stab = complex(y_stab_1)
    x_unst = complex(x_unst_1)
    y_unst = complex(y_unst_1)

    trk = {
        'z': np.zeros(n_steps + 1, dtype=complex),
        'xs': np.zeros(n_steps + 1, dtype=complex),
        'ys': np.zeros(n_steps + 1, dtype=complex),
        'xu': np.zeros(n_steps + 1, dtype=complex),
        'yu': np.zeros(n_steps + 1, dtype=complex),
        'dzds': np.zeros(n_steps + 1, dtype=complex),
    }
    trk['z'][0] = z_vals[0]
    trk['xs'][0] = x_stab
    trk['ys'][0] = y_stab
    trk['xu'][0] = x_unst
    trk['yu'][0] = y_unst
    trk['dzds'][0] = dz_ds(0)

    for k in range(1, n_steps + 1):
        z_k = z_vals[k]
        if solver is not None:
            sols = xYZ_solve_fast(solver, z_k)
        else:
            sols = xYZ_solve_at_z(F0, F1, z_k, R_poly=R)
        if not sols:
            trk['z'][k] = z_k
            trk['xs'][k] = x_stab
            trk['ys'][k] = y_stab
            trk['xu'][k] = x_unst
            trk['yu'][k] = y_unst
            trk['dzds'][k] = dz_ds(s_vals[k])
            continue

        def closest(xtrk, ytrk):
            best = None
            bestd = 1e18
            for (xr, yr) in sols:
                dist = abs(xr - xtrk) ** 2 + abs(yr - ytrk) ** 2
                if dist < bestd:
                    bestd = dist
                    best = (xr, yr)
            return best, bestd

        (x_stab, y_stab), ds1 = closest(x_stab, y_stab)
        (x_unst, y_unst), ds2 = closest(x_unst, y_unst)
        trk['z'][k] = z_k
        trk['xs'][k] = x_stab
        trk['ys'][k] = y_stab
        trk['xu'][k] = x_unst
        trk['yu'][k] = y_unst
        trk['dzds'][k] = dz_ds(s_vals[k])

    return trk, s_vals


def contour_integral_from_tracks(trk, s_vals):
    """Integrate (x_unst - x_stab)/z * dz/ds ds using trapezoidal rule."""
    integ = (trk['xu'] - trk['xs']) / trk['z'] * trk['dzds']
    ds = s_vals[1] - s_vals[0]
    S = np.sum(0.5 * (integ[:-1] + integ[1:])) * ds
    return S


def find_merge_zstar_by_tracking(F0, F1, R, z_start, x_stab_1, y_stab_1,
                                  x_unst_1, y_unst_1,
                                  directions=None, n_steps=600, max_r=6.0,
                                  solver=None):
    """Scan along several ray directions from z_start, tracking two branches
    (stable and unstable), and return the z* at which they first merge
    (min |x_stab - x_unst|)."""
    if directions is None:
        # Try several ray angles
        directions = [np.exp(1j * theta) for theta in
                      np.linspace(-np.pi, np.pi, 32, endpoint=False)]
    candidates = []
    for dir_ in directions:
        z_end = z_start + dir_ * max_r
        # Skip rays that pass too close to z=0 (the 1/z integrand explodes)
        # Distance from z=0 to line from z_start to z_end
        # Parametrize z(t) = z_start + t*(z_end - z_start), t in [0,1].
        # min |z(t)| at t* = -Re(conj(dz)*z_start)/|dz|^2 clipped to [0,1]
        dz = z_end - z_start
        denom = abs(dz) ** 2
        if denom > 0:
            t_star = -(dz.conjugate() * z_start).real / denom
            t_star = max(0.0, min(1.0, t_star))
            min_dist = abs(z_start + t_star * dz)
        else:
            min_dist = abs(z_start)
        if min_dist < 0.1:
            continue
        trk, svs = track_branches(F0, F1, R, z_start, z_end,
                                  x_stab_1, y_stab_1, x_unst_1, y_unst_1,
                                  n_steps=n_steps, path='straight',
                                  solver=solver)
        gap = np.abs(trk['xu'] - trk['xs'])
        k_min = int(np.argmin(gap))
        if k_min == 0:
            continue
        candidates.append((float(gap[k_min]), complex(trk['z'][k_min]),
                           trk, svs, k_min, complex(dir_)))
    # Choose smallest gap (most likely branch-point)
    candidates.sort(key=lambda c: c[0])
    return candidates


def contour_integral(F0, F1, R, z_start, z_end, x_stab_1, x_unst_1,
                     n_steps=400, path='straight'):
    """Numerically integrate Re[ int (x_unst - x_stab) / z  dz ]
    along a contour from z_start to z_end.
    Track branches by continuity: start from known roots at z_start,
    at each step pick the (X, Y) pair closest to previous one.
    x_stab, x_unst are the X-components of the two branches (symmetric-sector).
    """
    # Parameterize path
    if path == 'straight':
        zs = np.linspace(0, 1, n_steps + 1)
        pathfn = lambda s: z_start + (z_end - z_start) * s
        dz_ds = lambda s: (z_end - z_start)
    elif path == 'upper':
        # Go via upper half-plane: semicircle-ish through z_mid with large Im
        zm = 0.5 * (z_start + z_end) + 1j * 0.5 * abs(z_end - z_start)
        # piecewise Bezier via zm
        def pathfn(s):
            # quadratic Bezier: (1-s)^2 z0 + 2(1-s)s zm + s^2 z1
            return (1 - s) ** 2 * z_start + 2 * (1 - s) * s * zm + s ** 2 * z_end
        def dz_ds(s):
            return 2 * (1 - s) * (zm - z_start) + 2 * s * (z_end - zm)
    elif path == 'lower':
        zm = 0.5 * (z_start + z_end) - 1j * 0.5 * abs(z_end - z_start)
        def pathfn(s):
            return (1 - s) ** 2 * z_start + 2 * (1 - s) * s * zm + s ** 2 * z_end
        def dz_ds(s):
            return 2 * (1 - s) * (zm - z_start) + 2 * s * (z_end - zm)
    else:
        raise ValueError(path)

    # Tracked values
    x_stab = complex(x_stab_1)
    x_unst = complex(x_unst_1)
    # We also need Y for tracking; store full (X,Y)
    # Initial Y for each: from symmetric sector Y = X typically near z=1?
    # Actually in the symmetric ansatz x_1=x_2=Y, but X and Y don't have to be equal.
    # Near z=1, the fixed points of coupled system have X = Y = single-site FPs by symmetry.
    y_stab = complex(x_stab_1)  # on sym branch, X = Y near single-site FPs
    y_unst = complex(x_unst_1)

    s_vals = np.linspace(0, 1, n_steps + 1)
    z_vals = np.array([pathfn(s) for s in s_vals])

    # Accumulate integral via trapezoid
    integrand = np.zeros(n_steps + 1, dtype=complex)
    integrand[0] = (x_unst - x_stab) / z_vals[0] * dz_ds(0)

    # Track by re-solving at each z and picking closest
    for k in range(1, n_steps + 1):
        z_k = z_vals[k]
        # solve
        sols = xYZ_solve_at_z(F0, F1, z_k, R_poly=R)
        if not sols:
            # try with more precision
            integrand[k] = integrand[k - 1]
            continue
        # Find closest to (x_stab, y_stab) and (x_unst, y_unst)
        def closest(xtrk, ytrk):
            best = None
            bestd = 1e18
            for (xr, yr) in sols:
                dist = abs(xr - xtrk) ** 2 + abs(yr - ytrk) ** 2
                if dist < bestd:
                    bestd = dist
                    best = (xr, yr)
            return best

        ns = closest(x_stab, y_stab)
        nu = closest(x_unst, y_unst)
        if ns is None or nu is None:
            integrand[k] = integrand[k - 1]
            continue
        x_stab, y_stab = ns
        x_unst, y_unst = nu
        integrand[k] = (x_unst - x_stab) / z_k * dz_ds(s_vals[k])

    # Trapezoidal
    ds = s_vals[1] - s_vals[0]
    S = np.sum(0.5 * (integrand[:-1] + integrand[1:])) * ds
    return S  # complex; take real part


# -------------------- Gillespie MFPT on 3-site ring --------------------
def gillespie_3ring(alpha, V_eff, n_trials=500, t_max=1e9, rng_seed=0):
    """3-site ring Gillespie. Return MFPT (mean time until any site has
    x_i >= 0.7 * x_high), measuring with success rate."""
    rng = np.random.default_rng(rng_seed)
    a, lam, delta, d = A_, LAM, DEL, D_
    V = float(V_eff)
    # single-site fixed points
    coefs = [d, -lam, delta, -a]
    roots = np.roots(coefs)
    real_roots = sorted([float(r.real) for r in roots if abs(r.imag) < 1e-8])
    x_low = real_roots[0]
    x_mid = real_roots[1]
    x_high = real_roots[2]
    thr_n = int(np.ceil(0.7 * x_high * V))

    # ring adjacency
    adj = [[1, 2], [0, 2], [0, 1]]
    N = 3

    times = []
    succ = 0
    for trial in range(n_trials):
        n = np.full(N, max(1, int(round(x_low * V))), dtype=np.int64)
        x = n.astype(float) / V
        def rate_at(i):
            x2i = x[i] * x[i]
            x2nb = (x[adj[i][0]] ** 2 + x[adj[i][1]] ** 2) / 2
            wp = a + lam * ((1 - alpha) * x2i + alpha * x2nb)
            wm = delta * x[i] + d * x[i] ** 3
            return max(V * wp, 0.0), max(V * wm, 0.0)

        wp_arr = np.zeros(N)
        wm_arr = np.zeros(N)
        for i in range(N):
            wp_arr[i], wm_arr[i] = rate_at(i)

        t = 0.0
        ok = False
        steps = 0
        MAX = 4_000_000
        while t < t_max and steps < MAX:
            if (n >= thr_n).any():
                ok = True
                break
            total = wp_arr.sum() + wm_arr.sum()
            if total <= 0:
                break
            dt = rng.exponential(1.0 / total)
            t += dt
            r = rng.random() * total
            cp = np.cumsum(wp_arr)
            if r < cp[-1]:
                i = int(np.searchsorted(cp, r))
                n[i] += 1
                dn = +1
            else:
                r2 = r - cp[-1]
                cm = np.cumsum(wm_arr)
                i = int(np.searchsorted(cm, r2))
                if n[i] == 0:
                    steps += 1
                    continue
                n[i] -= 1
                dn = -1
            x[i] = n[i] / V
            # affected sites: i and its neighbors
            for j in [i] + adj[i]:
                wp_arr[j], wm_arr[j] = rate_at(j)
            steps += 1
        if ok:
            times.append(t)
            succ += 1
    return times, succ, n_trials


def mfpt_slope(alpha, V_list, n_trials, seed_base=1234):
    """Return log(mean MFPT) for each V and linear slope S."""
    logT = []
    for k, V in enumerate(V_list):
        times, succ, tot = gillespie_3ring(alpha, V, n_trials=n_trials,
                                           t_max=1e10, rng_seed=seed_base + k)
        if succ >= 5:
            m = np.mean(times)
            logT.append(np.log(m))
            print(f"    alpha={alpha:.2f} V={V}: {succ}/{tot} succ, "
                  f"<t>={m:.3e}, log<t>={np.log(m):.4f}")
        else:
            logT.append(None)
            print(f"    alpha={alpha:.2f} V={V}: ONLY {succ}/{tot} succ")
    # fit on valid
    Vs_valid = [V for V, l in zip(V_list, logT) if l is not None]
    Ls_valid = [l for l in logT if l is not None]
    if len(Vs_valid) >= 2:
        slope, inter = np.polyfit(Vs_valid, Ls_valid, 1)
    else:
        slope, inter = None, None
    return slope, inter, list(zip(V_list, logT))


# ================================================================
# Main execution
# ================================================================
if __name__ == '__main__':
    t0 = time.time()

    # Sanity: at alpha = 0, R should factor and give S_1 ~ 0.456
    print("=" * 72)
    print("Sanity check: single-site barrier (alpha = 0 decoupled limit)")
    print("=" * 72)
    def wp1(x): return A_ + LAM * x * x
    def wm1(x): return DEL * x + D_ * x ** 3
    coefs = [D_, -LAM, DEL, -A_]
    fps = sorted([r.real for r in np.roots(coefs) if abs(r.imag) < 1e-8])
    x_low, x_mid, x_high = fps
    S1_wkb, _ = quad(lambda x: np.log(wm1(x) / wp1(x)), x_low, x_mid)
    print(f"  fixed points: low={x_low:.4f}, mid={x_mid:.4f}, high={x_high:.4f}")
    print(f"  S_1 (WKB, low->mid) = {S1_wkb:.6f}")

    # Real branch-gap: for cubic d x^3 - lam z x^2 + delta x - a z = 0 in X,
    # discriminant in z. (Shifted by z scaling inside the single-site DSE.)
    # We use the canonical form wm(x) = z * wp(x):
    #   delta x + d x^3 = z (a + lam x^2)
    #   d x^3 - z lam x^2 + delta x - z a = 0
    xs = sp.symbols('xs')
    zs = sp.symbols('zs')
    Pcub = D_ * xs ** 3 - zs * LAM * xs ** 2 + DEL * xs - zs * A_
    disc_cub = sp.discriminant(sp.Poly(Pcub, xs), xs)
    zroots = sp.Poly(sp.expand(disc_cub), zs).nroots(n=30)
    zbps_real = sorted([complex(r).real for r in zroots if abs(complex(r).imag) < 1e-6])
    print(f"  single-site z-branch-points (real): {zbps_real}")

    # branch-gap integral along real z
    def x_roots_of(zz):
        pc = [D_, -zz * LAM, DEL, -zz * A_]
        return np.roots(pc)

    # Pick z* nearest to 1 among real ones; integrate gap from z=1 to z*
    # For reference
    if zbps_real:
        z_star = min(zbps_real, key=lambda z: abs(z - 1.0))
        print(f"  chosen z* = {z_star:.6f}")
    # (we compare Gillespie slope directly against WKB S_1 at alpha=0 in report)

    # -------- Per-alpha: branch points of coupled resultant --------
    # Include alpha=0.0 as sanity check
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.0, 0.1, 0.3, 0.5, 0.7])
    args, _ = parser.parse_known_args()
    ALPHAS = args.alphas
    if args.quick:
        V_LIST = [4, 6]
        N_TRIALS = 40
    else:
        V_LIST = [4, 6, 8, 10]
        N_TRIALS = 500

    summary = []
    for alpha in ALPHAS:
        print()
        print("=" * 72)
        print(f"alpha = {alpha}")
        print("=" * 72)
        t_a = time.time()
        F0, F1, R = build_resultant(alpha)
        print(f"  Resultant R(X,z) degree in X: {sp.Poly(R, X).degree()}")
        print(f"  Resultant R(X,z) degree in z: {sp.Poly(R, Z).degree()}")
        disc = discriminant_X(R)
        print(f"  disc_X(R)(z) built, deg in z: {sp.Poly(disc, Z).degree()}")
        # Compile fast numeric solver for this alpha
        solver = compile_solver(F0, F1, R)

        # Complex branch points
        zbps = complex_branch_points(disc)
        # filter by magnitude to get 'relevant' ones
        zbps = [z for z in zbps if abs(z - 1.0) < 20 and abs(z) > 0.01]
        # sort by distance to z=1
        zbps.sort(key=lambda z: abs(z - 1.0))
        print(f"  Complex z-branch-points of disc_X(R), |z-1|<20:")
        for k, z in enumerate(zbps[:12]):
            mark = " (REAL)" if abs(z.imag) < 1e-6 else ""
            print(f"    z_{k} = {z.real:+.5f} {z.imag:+.5f} i"
                  f"   |z-1|={abs(z-1):.4f}{mark}")

        # Identify low/unstable X-branches at z=1 (physical point):
        # At z=1, system reduces to steady-state: wm = wp at each site
        # Coupled system: F0 = F1 = 0 at z=1
        sols_z1 = xYZ_solve_fast(solver, 1.0 + 0j)
        # On symmetric branch X=Y; get real (X,X) with X>0
        sym_real = []
        for (xr, yr) in sols_z1:
            if abs(xr - yr) < 1e-3 and abs(xr.imag) < 1e-3 and xr.real > -1e-3:
                sym_real.append(xr.real)
        # Cluster with tolerance 1e-3 to dedupe numerical noise
        sym_real.sort()
        clustered = []
        for v in sym_real:
            if not clustered or abs(v - clustered[-1]) > 1e-3:
                clustered.append(v)
        sym_real = clustered
        print(f"  Symmetric real fixed points at z=1: {[round(v,4) for v in sym_real]}")

        if len(sym_real) >= 3:
            x_low_s, x_mid_s, x_high_s = sym_real[0], sym_real[1], sym_real[2]
        elif len(sym_real) >= 2:
            x_low_s, x_mid_s = sym_real[0], sym_real[1]
            x_high_s = sym_real[-1]
        else:
            x_low_s, x_mid_s, x_high_s = x_low, x_mid, x_high
        print(f"  stable={x_low_s:.4f}, unstable={x_mid_s:.4f}, high={x_high_s:.4f}")

        # On the symmetric branch, X = Y at the physical fixed points.
        # Start tracking: (x_stab, y_stab) = (x_low_s, x_low_s),
        #                 (x_unst, y_unst) = (x_mid_s, x_mid_s)
        results_alpha = {'alpha': alpha, 'zbps': zbps, 'paths': {}}

        # Find all asymmetric fixed points at z=1; use each as candidate
        # "unstable saddle" seed and find the z* where it merges with the
        # symmetric-low stable branch
        print(f"  Scanning asymmetric fixed points at z=1 as saddle candidates...")
        # Deduplicate sols_z1 by clustering
        uniq_sols = []
        for (xr, yr) in sols_z1:
            dup = False
            for (xu, yu) in uniq_sols:
                if abs(xr - xu) < 1e-3 and abs(yr - yu) < 1e-3:
                    dup = True
                    break
            if not dup:
                uniq_sols.append((xr, yr))
        # Asymmetric = X != Y
        asym_fps = [(xr, yr) for (xr, yr) in uniq_sols if abs(xr - yr) > 1e-3]
        # Also add symmetric middle (x_mid_s, x_mid_s) for reference
        sym_sad = (complex(x_mid_s), complex(x_mid_s))

        seed_list = [('sym_sad', sym_sad)]
        for i, fp in enumerate(asym_fps):
            label = 'asym_real' if abs(fp[0].imag) < 1e-4 and abs(fp[1].imag) < 1e-4 else 'asym_cplx'
            seed_list.append((f"{label}_{i}", fp))

        # For each seed, find the merge with the symmetric low branch
        best_global = None
        merge_results = []
        for lbl, (x_sad, y_sad) in seed_list:
            cand_list = find_merge_zstar_by_tracking(
                F0, F1, R, 1.0 + 0j,
                complex(x_low_s), complex(x_low_s),
                x_sad, y_sad,
                directions=None, n_steps=200, max_r=4.0,
                solver=solver,
            )
            if not cand_list:
                continue
            gap, z_star, trk, svs, k_min, dirv = cand_list[0]
            # Integral up to merge point
            s_sub = svs[:k_min + 1]
            trk_sub = {k: v[:k_min + 1] for k, v in trk.items()}
            S = contour_integral_from_tracks(trk_sub, s_sub)
            Sr = S.real
            Si = S.imag
            print(f"    seed={lbl:>14s} seed=({x_sad.real:+.3f}{x_sad.imag:+.3f}i, "
                  f"{y_sad.real:+.3f}{y_sad.imag:+.3f}i), "
                  f"z*={z_star.real:+.3f}{z_star.imag:+.3f}i, "
                  f"|z*-1|={abs(z_star-1):.3f}, gap={gap:.2e}, "
                  f"S=|Re|{abs(Sr):.4f}+Im{Si:+.4f}i")
            merge_results.append({
                'label': lbl, 'seed': (x_sad, y_sad), 'z_star': z_star,
                'gap': gap, 'S_re': Sr, 'S_im': Si,
            })

        results_alpha['merge_results'] = merge_results
        # Best = smallest |Re S| > 0.01 with smallest gap, preferring
        # asymmetric (physical) over symmetric
        # Actually "best" here: smallest |Re S|
        viable = [m for m in merge_results if abs(m['S_re']) > 0.05 and m['gap'] < 0.3]
        if viable:
            viable.sort(key=lambda m: abs(m['S_re']))
            best = viable[0]
            print(f"  [best] seed={best['label']}, z*={best['z_star']:+.4f}, "
                  f"S = |Re|={abs(best['S_re']):.4f} + {best['S_im']:+.4f}i")
            results_alpha['z_star_best'] = best['z_star']
            results_alpha['S_best'] = (best['S_re'], best['S_im'])
            results_alpha['merge_gap'] = best['gap']
        else:
            # fallback to smallest |Re S|
            if merge_results:
                merge_results.sort(key=lambda m: abs(m['S_re']))
                best = merge_results[0]
                results_alpha['z_star_best'] = best['z_star']
                results_alpha['S_best'] = (best['S_re'], best['S_im'])
                results_alpha['merge_gap'] = best['gap']

        # (Removed: disc-zero straight-line integration.
        #  The merge-tracking above is the right way to find the saddle.)

        # Gillespie
        print(f"  --- Gillespie MFPT, 3-site ring, alpha={alpha} ---")
        slope, inter, data = mfpt_slope(alpha, V_LIST, n_trials=N_TRIALS, seed_base=2024 + int(alpha * 100))
        if slope is not None:
            print(f"  Gillespie slope S = {slope:.4f}  (intercept = {inter:.4f})")
        else:
            print(f"  Gillespie slope S = (too few successes)")
        results_alpha['S_gillespie'] = slope
        results_alpha['mfpt_data'] = data
        summary.append(results_alpha)
        print(f"  (alpha={alpha} took {time.time()-t_a:.1f}s)")

    # ===================== Final summary =====================
    print()
    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    header = (f"{'alpha':>6s} | {'z*_best (merge)':>28s} | "
              f"{'|Re(S)|':>10s} | {'Im(S)':>10s} | {'gap':>8s} | "
              f"{'S_sym':>8s} | {'S_Gill':>8s} | {'ratio':>8s}")
    print(header)
    print("-" * len(header))
    for r in summary:
        zb = r.get('z_star_best')
        if zb is not None:
            zbest = f"{zb.real:+.4f}{zb.imag:+.4f}i"
            Sbest_r, Sbest_i = r['S_best']
            Sbest_abs = abs(Sbest_r)
            Sbest_s = f"{Sbest_abs:.4f}"
            Sbest_im = f"{Sbest_i:+.4f}"
            gap_s = f"{r.get('merge_gap', 0):.2e}"
        else:
            zbest = "---"
            Sbest_s = "---"
            Sbest_abs = None
            Sbest_im = "---"
            gap_s = "---"

        # Symmetric-sector S: find the sym_sad entry
        S_sym_s = "---"
        for m in r.get('merge_results', []):
            if m['label'] == 'sym_sad':
                S_sym_s = f"{abs(m['S_re']):.4f}"
                break

        SG = r['S_gillespie']
        ratio = (Sbest_abs / SG) if (SG and Sbest_abs is not None) else None
        ratio_s = f"{ratio:.3f}" if ratio is not None else "---"
        SG_s = f"{SG:.4f}" if SG is not None else "---"
        print(f"{r['alpha']:>6.2f} | {zbest:>28s} | {Sbest_s:>10s} | "
              f"{Sbest_im:>10s} | {gap_s:>8s} | "
              f"{S_sym_s:>8s} | {SG_s:>8s} | {ratio_s:>8s}")

    print()
    print(f"total wall time: {time.time()-t0:.1f}s")
