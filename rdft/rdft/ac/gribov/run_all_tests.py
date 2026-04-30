"""
run_all_tests.py
================

End-to-end test of the CFAC + IBP-solver framework for 2-loop DP.

Exercises:
  1. Lagrange counts from G = z(1+G^2)
  2. Closed-form double-pole identity Z_X^(2,2) = (1/2) a (beta_1 + a)
  3. Closed-form simple-pole BPHZ counterterm
  4. 12 IBP coefficients
  5. Full pipeline reproducing JT05 Eq. (60)
  6. From-scratch IBP solver: Laporta triangulation of sunset family

Run from the repo root:  python -m rdft.ac.gribov.run_all_tests
Or:                       python rdft/ac/gribov/run_all_tests.py
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

# Locate this package's directory and the IBP solver alongside it.
GRIBOV_DIR = Path(__file__).resolve().parent
IBP_DIR = GRIBOV_DIR.parent / 'ibp_solver'
REPO_ROOT = GRIBOV_DIR.parent.parent.parent   # /Users/.../rdft


def run(label, cmd, expect_in_output=None):
    print()
    print('=' * 76)
    print(f' TEST: {label}')
    print('=' * 76)
    # Run from repo root so `from rdft.ac.ibp_solver...` imports work.
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    out = result.stdout + result.stderr
    if expect_in_output:
        for needle in expect_in_output:
            if needle in out:
                print(f'  [PASS] Found: {needle}')
            else:
                print(f'  [FAIL] Missing: {needle}')
                print(out[-500:])
                return False
    if result.returncode != 0:
        print(f'  [FAIL] non-zero exit: {result.returncode}')
        print(out[-500:])
        return False
    return True


def main():
    print('CFAC + IBP framework end-to-end test')

    g = lambda name: str(GRIBOV_DIR / name)
    i = lambda name: str(IBP_DIR / name)

    tests = [
        ('1. Lagrange + 1-loop algebra factors',
         [sys.executable, g('assembly.py')],
         ['MATCH', 'a_psi^(1) = 1/4', 'a_lambda^(1) = 1/8']),

        ('2. Double-pole closed form (Hopf identity)',
         [sys.executable, g('actrick.py')],
         ['1-loop algebra', 'a^psi_nest = (1/4)^2']),

        ('3. Simple-pole BPHZ counterterm',
         [sys.executable, g('simple_poles.py')],
         ['BPHZ', 'primitive']),

        ('4. 12 IBP coefficients',
         [sys.executable, g('ibp_coefficients.py')],
         ['ALL FOUR Z-FACTOR SIMPLE POLES MATCH JT05 Eq. (57): True']),

        ('5. Full pipeline -> JT05 Eq. (60)',
         [sys.executable, g('two_loop.py')],
         ['ALL MATCH:  True']),

        ('6. IBP solver: identity generation',
         [sys.executable, i('sunset_identity.py')],
         ['(d - 3)']),

        ('7. IBP solver: targeted reductions (d-5, d-4 coefficients)',
         [sys.executable, i('test_reduction.py')],
         ['d - 5', 'd - 4']),

        ('8. IBP solver: full Laporta triangulation',
         [sys.executable, i('triangulate.py')],
         ['Iterations:', 'I[1,1,1] is identified as a MASTER']),

        ('9. Causal->Euclidean reduction (thesis 2.5.3.2): bubble + sunset',
         [sys.executable, g('test_causal_reduction.py')],
         ['alpha0 + alpha1', 'alpha0*alpha1 + alpha0*alpha2 + alpha1*alpha2']),

        ('10. Contraction-view verification: code output vs. literal tadpole',
         [sys.executable, g('verify_contraction_view.py')],
         ['Psi_A == Psi_B ?  False', 'DIFFERENT parametric forms']),
    ]

    passes = 0
    for label, cmd, expects in tests:
        if run(label, cmd, expects):
            passes += 1

    print()
    print('=' * 76)
    print(f' SUMMARY: {passes}/{len(tests)} tests passed')
    print('=' * 76)


if __name__ == '__main__':
    main()
