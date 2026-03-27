"""
rdft.simulate
=============
Python interface to the Rust BRW simulator.

Calls the compiled Rust binary (rdft-sim) and returns parsed results.
Build the simulator first: cd simulations && cargo build --release

Usage::

    from rdft.simulate import run_brw, run_ensemble, find_binary

    # Full BRW validation suite
    results = run_brw()

    # Single configuration
    result = run_ensemble(
        graph="lattice:3:50",
        crn="birth_death",
        realizations=10000,
        t_max=5000,
        d_s=3.0,
    )

    # Access exponents
    for ex in result['fitted_exponents']:
        print(f"p={ex['p']}: α_cond={ex['alpha_cond']:.3f}")
"""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional


def find_binary() -> Path:
    """Locate the rdft-sim binary. Builds it if not found."""
    sim_dir = Path(__file__).parent.parent / "simulations"

    # Check release binary
    binary = sim_dir / "target" / "release" / "rdft-sim"
    if binary.exists():
        return binary

    # Check debug binary
    binary_debug = sim_dir / "target" / "debug" / "rdft-sim"
    if binary_debug.exists():
        return binary_debug

    # Check PATH
    which = shutil.which("rdft-sim")
    if which:
        return Path(which)

    # Try to build
    if (sim_dir / "Cargo.toml").exists():
        cargo = shutil.which("cargo")
        if cargo:
            print("Building rdft-sim (first run only)...")
            subprocess.run(
                [cargo, "build", "--release"],
                cwd=sim_dir,
                check=True,
            )
            if binary.exists():
                return binary

    raise FileNotFoundError(
        "rdft-sim binary not found. Build it with:\n"
        "  cd simulations && cargo build --release\n"
        "Or install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    )


def run_ensemble(
    graph: str = "lattice:2:100",
    crn: str = "birth_death",
    realizations: int = 1000,
    t_max: float = 2000.0,
    d_s: float = 2.0,
    max_per_site: int = 0,
) -> dict:
    """
    Run a single simulation ensemble.

    Parameters
    ----------
    graph : str
        Graph specification, e.g. "lattice:3:50", "sierpinski:4",
        "tree:5000:42", "ba:5000:3", "complete:10"
    crn : str
        CRN type: "birth_death", "gribov", "brw", "pair_annihilation", "coagulation"
    realizations : int
        Number of independent realizations (parallelised across CPU cores)
    t_max : float
        Maximum simulation time
    d_s : float
        Spectral dimension (for theory comparison)
    max_per_site : int
        Max particles per site (0=unlimited, 1=instant coalescence)

    Returns
    -------
    dict with keys: graph, crn, d_s, n_realizations, n_survived,
        times, moments, moments_surviving, fitted_exponents, convergence
    """
    binary = find_binary()

    cmd = [
        str(binary),
        "--graph", graph,
        "--crn", crn,
        "--realizations", str(realizations),
        "--tmax", str(t_max),
        "--ds", str(d_s),
        "--max-per-site", str(max_per_site),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def run_brw() -> dict:
    """
    Run the full BRW validation suite (all graph types).

    Returns dict with key 'runs', each containing fitted exponents
    for comparison with Bordeu, Amarteifio et al. (2019).
    """
    binary = find_binary()
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def plot_results(results: dict, show: bool = True, save_dir: Optional[str] = None):
    """
    Generate scaling, convergence, and comparison plots.

    Parameters
    ----------
    results : dict
        Output from run_brw() or run_ensemble()
    show : bool
        Whether to call plt.show()
    save_dir : str, optional
        Directory to save PNG files
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "simulations" / "python"))
    from plot_brw import plot_scaling, plot_convergence, plot_comparison, print_summary

    runs = results.get('runs', [results])
    print_summary(runs)

    if save_dir:
        from pathlib import Path as P
        d = P(save_dir)
        d.mkdir(exist_ok=True)
        plot_scaling(runs, str(d / "scaling.png"))
        plot_convergence(runs, str(d / "convergence.png"))
        plot_comparison(runs, str(d / "comparison.png"))
    else:
        plot_scaling(runs)
        plot_convergence(runs)
        plot_comparison(runs)

    if show:
        import matplotlib.pyplot as plt
        plt.show()
