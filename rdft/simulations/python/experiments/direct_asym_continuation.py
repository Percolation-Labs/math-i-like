"""
Focused test: follow the asymmetric real saddle at alpha=0.1 (X,Y)=(1.175, 0.131)
as alpha is increased to 0.3, 0.5, 0.7. This traces the "1-site-nucleated"
saddle into the complex plane (it goes complex at alpha ~ 0.2).

At each alpha, then compute the branch-gap integral from z=1 (where this saddle
sits at the continued (X*, Y*)) to the z* where the saddle collides with
the symmetric-low stable branch (x_low, x_low).

Compare to Gillespie barrier.
"""
import sys, time
sys.path.insert(0, '/Users/sirsh/code/math/rdft/paper/wip')
from complex_branch_gap import (
    build_resultant, compile_solver, xYZ_solve_fast,
    track_branches, contour_integral_from_tracks,
    gillespie_3ring, X, Z
)
import numpy as np


def find_asym_real_fp(sols, X_hint=1.175, Y_hint=0.131):
    """Pick the (X,Y) closest to hint with X != Y."""
    cands = [(xr, yr) for (xr, yr) in sols if abs(xr - yr) > 1e-3]
    if not cands:
        return None
    return min(cands, key=lambda xy: abs(xy[0] - X_hint) + abs(xy[1] - Y_hint))


def continue_asym_fp(alpha_target, alpha_grid=None):
    """Numerically continue the asymmetric 'site-0-excited' FP from
    alpha=0.1 to alpha_target. Return its (X, Y) at z=1, alpha_target."""
    if alpha_grid is None:
        # fine grid, interpolate from 0.0 (well, start at 0.1 where it's real)
        alpha_grid = list(np.linspace(0.1, alpha_target, 20))
    # Seed at alpha=0.1
    F0, F1, R = build_resultant(0.1)
    solver = compile_solver(F0, F1, R)
    sols_0p1 = xYZ_solve_fast(solver, 1.0 + 0j)
    fp = find_asym_real_fp(sols_0p1, 1.175, 0.131)
    if fp is None:
        print("  No real asymmetric FP at alpha=0.1 — aborting continuation.", flush=True)
        return None
    print(f"  alpha=0.10: asym FP = (X, Y) = "
          f"({fp[0].real:+.4f}{fp[0].imag:+.4f}i, "
          f"{fp[1].real:+.4f}{fp[1].imag:+.4f}i)", flush=True)

    cur_x, cur_y = fp
    for a in alpha_grid[1:]:
        F0, F1, R = build_resultant(a)
        solver = compile_solver(F0, F1, R)
        sols = xYZ_solve_fast(solver, 1.0 + 0j)
        # closest to (cur_x, cur_y)
        cands = [(xr, yr) for (xr, yr) in sols if abs(xr - yr) > 1e-3]
        if not cands:
            print(f"  alpha={a:.3f}: no asymmetric solution! aborting", flush=True)
            return None
        nxt = min(cands, key=lambda xy: abs(xy[0] - cur_x) + abs(xy[1] - cur_y))
        cur_x, cur_y = nxt
    print(f"  alpha={alpha_target:.2f}: continued asym FP = "
          f"({cur_x.real:+.4f}{cur_x.imag:+.4f}i, "
          f"{cur_y.real:+.4f}{cur_y.imag:+.4f}i)", flush=True)
    return cur_x, cur_y


def find_merge_single_direction(F0, F1, R, solver, z_start,
                                 x_stab, y_stab, x_unst, y_unst,
                                 max_r=4.0, n_steps=300, n_dirs=24):
    """Scan rays, return best (z*, S, trk) where branches merge."""
    best = None
    for theta in np.linspace(-np.pi, np.pi, n_dirs, endpoint=False):
        dir_ = np.exp(1j * theta)
        z_end = z_start + dir_ * max_r
        # Skip rays that pass too close to z=0
        dz = z_end - z_start
        denom = abs(dz) ** 2
        if denom > 0:
            t_star = -(dz.conjugate() * z_start).real / denom
            t_star = max(0.0, min(1.0, t_star))
            min_dist = abs(z_start + t_star * dz)
            if min_dist < 0.1:
                continue
        trk, svs = track_branches(F0, F1, R, z_start, z_end,
                                  x_stab, y_stab, x_unst, y_unst,
                                  n_steps=n_steps, path='straight',
                                  solver=solver)
        gap = np.abs(trk['xu'] - trk['xs'])
        k_min = int(np.argmin(gap))
        if k_min == 0:
            continue
        g = float(gap[k_min])
        if best is None or g < best[0]:
            # integral up to k_min
            s_sub = svs[:k_min + 1]
            trk_sub = {k: v[:k_min + 1] for k, v in trk.items()}
            S = contour_integral_from_tracks(trk_sub, s_sub)
            best = (g, complex(trk['z'][k_min]), S, trk_sub, s_sub, theta)
    return best


def main():
    # -- Gillespie baseline (more trials for cleaner slopes) --
    ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7]
    V_LIST = [4, 6, 8, 10]
    N_TRIALS = 600
    print("=" * 72, flush=True)
    print("Part 1: Gillespie MFPT slopes on 3-site ring", flush=True)
    print("=" * 72, flush=True)
    gillespie_slopes = {}
    for alpha in ALPHAS:
        logT = []
        Vs_valid = []
        print(f"\nalpha = {alpha}", flush=True)
        for k, V in enumerate(V_LIST):
            t0 = time.time()
            times, succ, tot = gillespie_3ring(alpha, V, n_trials=N_TRIALS,
                                                t_max=1e10,
                                                rng_seed=10000 + int(alpha * 1000) + k)
            if succ >= 10:
                m = np.mean(times)
                lt = np.log(m)
                logT.append(lt)
                Vs_valid.append(V)
                print(f"  V={V}: {succ}/{tot} succ, <t>={m:.3e}, log<t>={lt:.4f} "
                      f"(wall {time.time()-t0:.1f}s)", flush=True)
            else:
                print(f"  V={V}: only {succ}/{tot} succ (skipped)", flush=True)
        if len(Vs_valid) >= 2:
            slope, inter = np.polyfit(Vs_valid, logT, 1)
            print(f"  => slope S = {slope:.4f}, intercept = {inter:.4f}", flush=True)
            gillespie_slopes[alpha] = slope
        else:
            gillespie_slopes[alpha] = None

    # -- Branch-gap analysis --
    print("\n" + "=" * 72, flush=True)
    print("Part 2: Branch-gap analysis (asymmetric saddle continuation)", flush=True)
    print("=" * 72, flush=True)
    branch_results = {}
    # x_low_symmetric at alpha=0 ≈ 0.0258 (and is alpha-independent along X=Y)
    x_low_sym = 0.025830
    x_mid_sym = 0.965874

    for alpha in ALPHAS:
        print(f"\nalpha = {alpha}", flush=True)
        F0, F1, R = build_resultant(alpha)
        solver = compile_solver(F0, F1, R)

        # (A) SYMMETRIC saddle (always real z* = 2.8427 for all alpha)
        print(f"  (A) Symmetric saddle (seed: X=Y=0.9659):", flush=True)
        best_sym = find_merge_single_direction(
            F0, F1, R, solver, 1.0 + 0j,
            x_low_sym, x_low_sym, x_mid_sym, x_mid_sym,
            max_r=4.0, n_steps=250, n_dirs=24,
        )
        if best_sym:
            g, zstar, S, trk, svs, theta = best_sym
            print(f"      z* = {zstar.real:+.4f}{zstar.imag:+.4f}i, "
                  f"gap={g:.2e}, S = |Re|{abs(S.real):.4f} + {S.imag:+.4f}i", flush=True)

        # (B) ASYMMETRIC saddle (continued from alpha=0.1 to alpha)
        if alpha > 0.0:
            print(f"  (B) Asymmetric saddle (continued from alpha=0.1 real FP):", flush=True)
            if alpha <= 0.1:
                # Seed at current alpha, find closest real asym FP
                sols = xYZ_solve_fast(solver, 1.0 + 0j)
                fp = find_asym_real_fp(sols, 1.175, 0.131)
                if fp:
                    x_asym, y_asym = fp
                else:
                    x_asym = y_asym = None
            else:
                cont = continue_asym_fp(alpha)
                if cont:
                    x_asym, y_asym = cont
                else:
                    x_asym = y_asym = None

            if x_asym is not None:
                print(f"      saddle seed at z=1: "
                      f"X={x_asym.real:+.4f}{x_asym.imag:+.4f}i, "
                      f"Y={y_asym.real:+.4f}{y_asym.imag:+.4f}i", flush=True)
                best_asym = find_merge_single_direction(
                    F0, F1, R, solver, 1.0 + 0j,
                    x_low_sym, x_low_sym, x_asym, y_asym,
                    max_r=4.0, n_steps=250, n_dirs=24,
                )
                if best_asym:
                    g, zstar, S, trk, svs, theta = best_asym
                    print(f"      z* = {zstar.real:+.4f}{zstar.imag:+.4f}i, "
                          f"gap={g:.2e}, S = |Re|{abs(S.real):.4f} + {S.imag:+.4f}i", flush=True)
                    branch_results[alpha] = {
                        'asym_z_star': zstar, 'asym_S': S, 'asym_gap': g,
                        'sym_S': best_sym[2] if best_sym else None,
                    }
                    continue
        branch_results[alpha] = {
            'asym_z_star': None, 'asym_S': None, 'asym_gap': None,
            'sym_S': best_sym[2] if best_sym else None,
        }

    # -- Summary --
    print("\n" + "=" * 72, flush=True)
    print("FINAL COMPARISON TABLE", flush=True)
    print("=" * 72, flush=True)
    h = (f"{'alpha':>6s} | {'S_sym':>10s} | {'z*_asym':>24s} | "
         f"{'|Re S_asym|':>12s} | {'Im S_asym':>10s} | "
         f"{'S_Gill':>10s} | {'ratio':>8s}")
    print(h, flush=True)
    print("-" * len(h), flush=True)
    for alpha in ALPHAS:
        br = branch_results.get(alpha, {})
        S_sym = br.get('sym_S')
        S_asym = br.get('asym_S')
        z_star = br.get('asym_z_star')
        SG = gillespie_slopes.get(alpha)
        S_sym_s = f"{abs(S_sym.real):.4f}" if S_sym is not None else "---"
        if z_star is not None:
            zst_s = f"{z_star.real:+.3f}{z_star.imag:+.3f}i"
            S_asym_re = f"{abs(S_asym.real):.4f}"
            S_asym_im = f"{S_asym.imag:+.4f}"
        else:
            zst_s = "---"
            S_asym_re = "---"
            S_asym_im = "---"
        SG_s = f"{SG:.4f}" if SG is not None else "---"
        if (S_asym is not None) and (SG is not None) and abs(SG) > 1e-10:
            ratio = abs(S_asym.real) / SG
            ratio_s = f"{ratio:.3f}"
        else:
            ratio_s = "---"
        print(f"{alpha:>6.2f} | {S_sym_s:>10s} | {zst_s:>24s} | "
              f"{S_asym_re:>12s} | {S_asym_im:>10s} | "
              f"{SG_s:>10s} | {ratio_s:>8s}", flush=True)


if __name__ == '__main__':
    main()
