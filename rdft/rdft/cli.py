"""
rdft.cli
========
Command-line interface for RDFT.

Usage:
    rdft analyze gribov
    rdft analyze --stoichiometry "[[1,2],[1,0],[2,1]]" --rates "beta,epsilon,chi"
    rdft analyze pair_annihilation --show-diagrams
    rdft brw
    rdft corollas gribov
"""

import typer
import json
from typing import Optional

app = typer.Typer(
    name="rdft",
    help="Reaction-Diffusion Field Theory: from stoichiometry to critical exponents",
    add_completion=False,
)


@app.command()
def analyze(
    process: str = typer.Argument(
        ...,
        help="Process name (pair_annihilation, gribov, contact_process, coagulation) "
             "or 'custom' with --stoichiometry"
    ),
    stoichiometry: Optional[str] = typer.Option(
        None, "--stoichiometry", "-s",
        help='Stoichiometry matrix as JSON, e.g. "[[2,0]]" for 2A→∅'
    ),
    rates: Optional[str] = typer.Option(
        None, "--rates", "-r",
        help='Comma-separated rate names, e.g. "beta,epsilon,chi"'
    ),
    max_loops: int = typer.Option(0, "--loops", "-l", help="Max loop order for diagrams"),
    show_diagrams: bool = typer.Option(False, "--diagrams", "-d", help="Show diagram details"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Analyze a reaction-diffusion process via AC and RG routes."""
    import sympy as sp

    if process == 'custom' and stoichiometry:
        S = json.loads(stoichiometry)
        rate_syms = None
        if rates:
            rate_syms = [sp.Symbol(r.strip(), positive=True) for r in rates.split(',')]
        from rdft.core.reaction_network import ReactionNetwork
        net = ReactionNetwork.from_stoichiometry(S, rates=rate_syms, name='Custom')
        from rdft.core.generators import Liouvillian
        from rdft.ac.dse import dse_from_liouvillian, upper_critical_dimension, ac_full_derivation

        L = Liouvillian(net)
        d = sp.Symbol('d', positive=True)

        if not quiet:
            typer.echo(f"\n{'='*55}")
            typer.echo(f"RDFT Analysis: {net.name}")
            typer.echo(f"{'='*55}")
            typer.echo(f"\nStoichiometry: {S}")
            typer.echo(f"Q = {L.total}")
            typer.echo(f"Vertices: {L.vertices}")

            result = ac_full_derivation(L, d, p=1)
            typer.echo(f"\nφ(G) = {result['phi']}")
            typer.echo(f"d_c = {result['d_c']}")
            typer.echo(f"Singularity: {result['singularity_type']}")
            typer.echo(f"α₁ = {result['alpha_p']}  =  {result['alpha_p_formula']}")
    else:
        from rdft.pipeline import analyze as run_analyze
        run_analyze(process, max_loops=max_loops, verbose=not quiet)


@app.command()
def brw():
    """Run the complete BRW worked example (Bordeu+ 2019)."""
    from rdft.pipeline import brw_worked_example
    brw_worked_example()


@app.command()
def corollas(
    process: str = typer.Argument("gribov", help="Process name"),
):
    """Show the interaction vertices (corollas) for a process."""
    from rdft.core.reaction_network import ReactionNetwork
    from rdft.core.generators import Liouvillian

    factories = {
        'gribov': ReactionNetwork.gribov,
        'pair_annihilation': ReactionNetwork.pair_annihilation,
        'coagulation': ReactionNetwork.coagulation,
        'contact_process': ReactionNetwork.contact_process,
        'triplet_annihilation': ReactionNetwork.triplet_annihilation,
        'barw_even': ReactionNetwork.barw_even,
        'barw_odd': ReactionNetwork.barw_odd,
        'pcpd': ReactionNetwork.pcpd,
        'schlogl_first': ReactionNetwork.schlogl_first,
        'schlogl_second': ReactionNetwork.schlogl_second,
        'prion': lambda: ReactionNetwork.prion_propagation(minimal=True),
        'lotka_volterra': ReactionNetwork.lotka_volterra,
        'michaelis_menten': ReactionNetwork.michaelis_menten,
        'reversible_annihilation': ReactionNetwork.reversible_annihilation,
    }

    if process not in factories:
        typer.echo(f"Unknown process: {process}")
        typer.echo(f"Available: {', '.join(factories.keys())}")
        raise typer.Exit(1)

    net = factories[process]()
    L = Liouvillian(net)

    typer.echo(f"\nCorollas for {net.name}:")
    typer.echo(f"Q = {L.total}\n")

    for (m, n), g in sorted(L.vertices.items()):
        legs = m + n
        vtype = "interaction" if legs >= 3 else "propagator"
        typer.echo(f"  φ̃^{m}φ^{n}  coupling={g}  [{vtype}]")


@app.command()
def stoichiometry(
    matrix: str = typer.Argument(..., help='JSON stoichiometry, e.g. "[[2,0]]"'),
    rates: Optional[str] = typer.Option(None, "--rates", "-r"),
    name: str = typer.Option("Custom", "--name", "-n"),
):
    """Analyze a process from its raw stoichiometry matrix."""
    import sympy as sp

    S = json.loads(matrix)
    rate_syms = None
    if rates:
        rate_syms = [sp.Symbol(r.strip(), positive=True) for r in rates.split(',')]

    from rdft.core.reaction_network import ReactionNetwork
    from rdft.core.generators import Liouvillian
    from rdft.ac.dse import ac_full_derivation, upper_critical_dimension

    net = ReactionNetwork.from_stoichiometry(S, rates=rate_syms, name=name)
    L = Liouvillian(net)
    d = sp.Symbol('d', positive=True)

    typer.echo(f"\n{'='*55}")
    typer.echo(f"RDFT: {net.name}")
    typer.echo(f"{'='*55}")
    typer.echo(f"\nReactions:")
    for rxn in net.reactions:
        typer.echo(f"  {rxn}")

    typer.echo(f"\nLiouvillian Q = {L.total}")
    typer.echo(f"\nVertices:")
    for (m, n), g in sorted(L.vertices.items()):
        typer.echo(f"  φ̃^{m}φ^{n}: {g}")

    result = ac_full_derivation(L, d, p=1)
    typer.echo(f"\n--- AC Route ---")
    typer.echo(f"φ(G) = {result['phi']}")
    typer.echo(f"d_c = {result['d_c']}")
    typer.echo(f"Singularity: {result['singularity_type']}")
    typer.echo(f"Puiseux exponent: {result['puiseux_exponent']}")
    typer.echo(f"\nScaling: α_p = {result['alpha_p_formula']}")

    # Show table for p=1..5
    typer.echo(f"\nExponents α_p = (p·d - 2)/2:")
    for p in range(1, 6):
        r = ac_full_derivation(L, d, p=p)
        typer.echo(f"  p={p}: α_{p} = {r['alpha_p']}")


@app.command()
def survey():
    """Run all 16 literature CRNs through the AC pipeline."""
    from rdft.core.reaction_network import ReactionNetwork
    from rdft.core.generators import Liouvillian
    from rdft.ac.dse import (
        combinatorial_dse_kernel, upper_critical_dimension,
        classify_vertices, diagnose_singularity,
    )
    import sympy as sp

    G = sp.Symbol('G')

    factories = {
        '2A→∅':          ReactionNetwork.pair_annihilation,
        '2A→A':          ReactionNetwork.coagulation,
        '3A→∅':          lambda: ReactionNetwork.triplet_annihilation(),
        '4A→∅':          lambda: ReactionNetwork.k_particle_annihilation(4),
        'Gribov':        ReactionNetwork.gribov,
        'Contact':       ReactionNetwork.contact_process,
        'Schlögl II':    ReactionNetwork.schlogl_second,
        'BARW-odd':      ReactionNetwork.barw_odd,
        'BARW-even':     ReactionNetwork.barw_even,
        'PCPD':          ReactionNetwork.pcpd,
        'Schlögl I':     ReactionNetwork.schlogl_first,
        'Lotka-Volterra': ReactionNetwork.lotka_volterra,
        '2A⇌C':          ReactionNetwork.reversible_annihilation,
        'Prion':         lambda: ReactionNetwork.prion_propagation(minimal=True),
        'Michaelis-Menten': ReactionNetwork.michaelis_menten,
    }

    typer.echo(f"\n{'='*70}")
    typer.echo(f"  RDFT: CRN Survey — AC Pipeline on 15 Literature Processes")
    typer.echo(f"{'='*70}\n")

    for name, factory in factories.items():
        net = factory()
        L = Liouvillian(net)
        verts = L.vertices

        # Handle multi-species
        if net.n_species == 1:
            phi = combinatorial_dse_kernel(verts, G)
            d_c = upper_critical_dimension(verts)
            diag = diagnose_singularity(verts)
            stype = diag.get('singularity_type', '?')
            warnings = diag.get('warnings', [])
        else:
            d_c_vals = []
            for key in verts:
                total_legs = sum(sum(pair) for pair in key)
                if total_legs >= 3:
                    d_c_vals.append(sp.Rational(4, total_legs - 2))
            d_c = max(d_c_vals) if d_c_vals else '∞'
            stype = 'multi-species'
            warnings = []

        try:
            deg = sp.Poly(combinatorial_dse_kernel(verts, G), G).degree() if net.n_species == 1 else '?'
        except Exception:
            deg = '?'

        typer.echo(f"  {name:<18s}  d_c={str(d_c):>5s}  deg={str(deg):>2s}  {stype}")
        for w in warnings:
            typer.echo(f"    ⚠ {w}")

    typer.echo(f"\nDone. See `rdft analyze <process>` for detailed analysis of any single CRN.")


@app.command()
def simulate(
    graph: str = typer.Option("lattice:3:50", "--graph", "-g", help="Graph spec, e.g. lattice:3:50, sierpinski:4, tree:5000, ba:5000:3"),
    crn: str = typer.Option("birth_death", "--crn", "-c", help="CRN type: birth_death, gribov, brw, pair_annihilation"),
    realizations: int = typer.Option(5000, "--realizations", "-n", help="Number of realizations"),
    t_max: float = typer.Option(2000.0, "--tmax", "-t", help="Max simulation time"),
    d_s: float = typer.Option(0.0, "--ds", help="Spectral dimension (0 = auto from graph)"),
    suite: bool = typer.Option(False, "--suite", help="Run full BRW validation suite (ignores other options)"),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Generate plots"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save JSON results to file"),
):
    """Run the Rust particle simulator (BRW / general CRN on graphs)."""
    from rdft.simulate import run_brw, run_ensemble, plot_results

    if suite:
        typer.echo("Running full BRW validation suite...")
        results = run_brw()
    else:
        # Auto-detect d_s from graph name if not specified
        if d_s == 0.0:
            ds_map = {
                'lattice:1': 1.0, 'lattice:2': 2.0, 'lattice:3': 3.0,
                'lattice:4': 4.0, 'lattice:5': 5.0,
                'sierpinski': 1.86, 'tree': 1.333, 'ba': 4.0, 'complete': 100.0,
            }
            for prefix, val in ds_map.items():
                if graph.startswith(prefix):
                    d_s = val
                    break
            else:
                d_s = 2.0

        typer.echo(f"Simulating {crn} on {graph} (d_s={d_s}, n={realizations}, t_max={t_max})")
        results = run_ensemble(graph=graph, crn=crn, realizations=realizations, t_max=t_max, d_s=d_s)

    if output:
        import json as _json
        with open(output, 'w') as f:
            _json.dump(results, f, indent=2)
        typer.echo(f"Results saved to {output}")

    # Print summary
    runs = results.get('runs', [results])
    for run in runs:
        exps = run.get('fitted_exponents', [])
        typer.echo(f"\n  {run['graph']} ({run.get('n_survived','?')}/{run.get('n_realizations','?')} survived)")
        for ex in exps:
            ac = ex.get('alpha_cond', ex.get('alpha_surviving', '?'))
            th = ex.get('alpha_theory_cond', ex.get('alpha_theory', '?'))
            typer.echo(f"    p={ex['p']}: α_cond={ac:.3f}  theory={th:.3f}" if isinstance(ac, float) else f"    p={ex['p']}: {ac}")

    if plot:
        plot_results(results, show=True)


def main():
    app()


if __name__ == "__main__":
    main()
