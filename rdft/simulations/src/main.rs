/// rdft-sim: Fast particle simulator for reaction-diffusion systems on graphs.
///
/// Reproduces the BRW scaling exponents from Bordeu, Amarteifio et al. (2019).
///
/// Usage:
///   rdft-sim                              # Run full BRW validation suite
///   rdft-sim --graph lattice:2:50 --crn brw --realizations 1000 --tmax 2000
///   rdft-sim --graph sierpinski:3 --crn gribov --realizations 500
///   rdft-sim --graph ba:1000:3 --crn pair_annihilation --realizations 2000

use std::io::Write;
use std::time::Instant;

use rdft_sim::crn::CRN;
use rdft_sim::engine::{self, SimConfig};
use rdft_sim::graph::Graph;

use serde::Serialize;

#[derive(Serialize)]
struct RunOutput {
    graph: String,
    crn: String,
    n_realizations: u32,
    n_survived: u32,
    times: Vec<f64>,
    /// Unconditional moments ⟨V^p⟩ (averaged over all realizations).
    moments: Vec<Vec<f64>>,
    /// Conditional moments ⟨V^p | survived⟩.
    moments_surviving: Vec<Vec<f64>>,
    fitted_exponents: Vec<ExponentResult>,
    convergence: Vec<Vec<(u32, f64)>>,
    wall_time_secs: f64,
}

#[derive(Serialize)]
struct ExponentResult {
    p: usize,
    alpha_measured: f64,
    alpha_surviving: f64,
    alpha_theory: Option<f64>,
}

#[derive(Serialize)]
struct SuiteOutput {
    runs: Vec<RunOutput>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "--help" {
        print_help();
        return;
    }

    if args.len() > 1 {
        let (graph, crn, n_real, t_max, max_per_site) = parse_args(&args);
        let mut config = SimConfig::brw_default(t_max);
        config.max_per_site = max_per_site;
        let result = run_single(&graph, &crn, &config, n_real, None);
        let json = serde_json::to_string_pretty(&result).unwrap();
        println!("{}", json);
    } else {
        run_brw_suite();
    }
}

fn run_single(
    graph: &Graph,
    crn: &CRN,
    config: &SimConfig,
    n_real: u32,
    theory: Option<&[f64]>,
) -> RunOutput {
    let max_p = 3;
    let start = Instant::now();

    eprint!(
        "  Running {} on {} ({} realizations)... ",
        crn.name, graph.name, n_real
    );
    std::io::stderr().flush().ok();

    let ensemble = engine::run_ensemble(graph, crn, config, n_real, max_p);
    let elapsed = start.elapsed().as_secs_f64();

    eprintln!(
        "done in {:.1}s ({}/{} survived)",
        elapsed, ensemble.n_survived, n_real
    );

    let fitted_exponents: Vec<ExponentResult> = (0..max_p)
        .map(|p| {
            let alpha = engine::fit_power_law(&ensemble.times, &ensemble.moments[p]);
            let alpha_surv =
                engine::fit_power_law(&ensemble.times, &ensemble.moments_surviving[p]);
            ExponentResult {
                p: p + 1,
                alpha_measured: alpha,
                alpha_surviving: alpha_surv,
                alpha_theory: theory.and_then(|t| t.get(p).copied()),
            }
        })
        .collect();

    for ex in &fitted_exponents {
        let theory_str = ex
            .alpha_theory
            .map(|t| format!("{:.3}", t))
            .unwrap_or_else(|| "?".into());
        eprintln!(
            "    p={}: α_all={:.3}  α_surv={:.3}  (theory: {})",
            ex.p, ex.alpha_measured, ex.alpha_surviving, theory_str
        );
    }

    RunOutput {
        graph: graph.name.clone(),
        crn: crn.name.clone(),
        n_realizations: n_real,
        n_survived: ensemble.n_survived,
        times: ensemble.times,
        moments: ensemble.moments,
        moments_surviving: ensemble.moments_surviving,
        fitted_exponents,
        convergence: ensemble.convergence,
        wall_time_secs: elapsed,
    }
}

/// Run the full BRW validation suite from Bordeu, Amarteifio et al. (2019).
fn run_brw_suite() {
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  rdft-sim: BRW Validation Suite                        ║");
    eprintln!("║  Bordeu, Amarteifio et al. (2019) Sci. Rep. 9:15590    ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!();

    let crn = CRN::brw_coalescent();
    let n_real = 2000;

    // Theory: alpha_p = (p*d_s - 2)/2 for d_s < 4, else 2p-1
    let configs: Vec<(Graph, f64, Vec<f64>)> = vec![
        // 1D lattice: d_s = 1, alpha_p = (p-2)/2
        (
            Graph::hypercubic(1, 10000),
            5000.0,
            vec![-0.5, 0.0, 0.5],
        ),
        // 2D lattice: d_s = 2, alpha_p = (2p-2)/2 = p-1
        (
            Graph::hypercubic(2, 200),
            3000.0,
            vec![0.0, 1.0, 2.0],
        ),
        // 3D lattice: d_s = 3, alpha_p = (3p-2)/2
        (
            Graph::hypercubic(3, 50),
            2000.0,
            vec![0.5, 2.0, 3.5],
        ),
        // 5D lattice: d_s = 5 >= 4 → mean-field: alpha_p = 2p-1
        (
            Graph::hypercubic(5, 10),
            1000.0,
            vec![1.0, 3.0, 5.0],
        ),
        // Sierpinski carpet: d_s ~ 1.86
        (
            Graph::sierpinski_carpet(4),
            2000.0,
            vec![-0.07, 0.86, 1.79],
        ),
        // Random tree: d_s ~ 4/3
        (
            Graph::random_tree(10000, 42),
            2000.0,
            vec![-0.33, 0.33, 1.0],
        ),
        // Barabási-Albert: d_s >= 4 → mean-field
        (
            Graph::barabasi_albert(5000, 3, 42),
            1000.0,
            vec![1.0, 3.0, 5.0],
        ),
    ];

    let start = Instant::now();
    let mut runs = Vec::new();

    for (graph, t_max, theory) in &configs {
        let config = SimConfig::brw_default(*t_max);
        let result = run_single(graph, &crn, &config, n_real, Some(theory));
        runs.push(result);
    }

    let total_time = start.elapsed().as_secs_f64();
    eprintln!("\n  Total wall time: {:.1}s", total_time);

    // Summary table
    eprintln!(
        "\n{:<30} {:>3} {:>8} {:>8} {:>8} {:>5}",
        "Graph", "p", "Theory", "All", "Surv.", "Match"
    );
    eprintln!("{}", "-".repeat(67));
    for run in &runs {
        for ex in &run.fitted_exponents {
            let theory_str = ex
                .alpha_theory
                .map(|t| format!("{:.3}", t))
                .unwrap_or("?".into());
            let ok = ex.alpha_theory.map_or("?", |t| {
                if (ex.alpha_measured - t).abs() < 0.3 {
                    "ok"
                } else {
                    "MISS"
                }
            });
            eprintln!(
                "  {:<28} {:>3} {:>8} {:>8.3} {:>8.3} {:>5}",
                run.graph, ex.p, theory_str, ex.alpha_measured, ex.alpha_surviving, ok
            );
        }
    }

    let suite = SuiteOutput { runs };
    let json = serde_json::to_string_pretty(&suite).unwrap();
    println!("{}", json);
}

fn parse_args(args: &[String]) -> (Graph, CRN, u32, f64, u32) {
    let mut graph_str = "lattice:2:100".to_string();
    let mut crn_str = "brw".to_string();
    let mut n_real = 1000u32;
    let mut t_max = 2000.0f64;
    let mut max_per_site = 1u32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--graph" => {
                graph_str = args[i + 1].clone();
                i += 2;
            }
            "--crn" => {
                crn_str = args[i + 1].clone();
                i += 2;
            }
            "--realizations" | "-n" => {
                n_real = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--tmax" => {
                t_max = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--max-per-site" => {
                max_per_site = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                i += 1;
            }
        }
    }

    let graph = parse_graph(&graph_str);
    let crn = parse_crn(&crn_str);
    (graph, crn, n_real, t_max, max_per_site)
}

fn parse_graph(s: &str) -> Graph {
    let parts: Vec<&str> = s.split(':').collect();
    match parts[0] {
        "lattice" => {
            let dim: usize = parts.get(1).unwrap_or(&"2").parse().unwrap();
            let size: usize = parts.get(2).unwrap_or(&"100").parse().unwrap();
            Graph::hypercubic(dim, size)
        }
        "sierpinski" => {
            let level: usize = parts.get(1).unwrap_or(&"3").parse().unwrap();
            Graph::sierpinski_carpet(level)
        }
        "tree" => {
            let n: usize = parts.get(1).unwrap_or(&"1000").parse().unwrap();
            let seed: u64 = parts.get(2).unwrap_or(&"42").parse().unwrap();
            Graph::random_tree(n, seed)
        }
        "ba" => {
            let n: usize = parts.get(1).unwrap_or(&"1000").parse().unwrap();
            let m: usize = parts.get(2).unwrap_or(&"3").parse().unwrap();
            Graph::barabasi_albert(n, m, 42)
        }
        "complete" => {
            let n: usize = parts.get(1).unwrap_or(&"10").parse().unwrap();
            Graph::complete(n)
        }
        _ => {
            eprintln!("Unknown graph type: {}. Using 2D lattice.", parts[0]);
            Graph::hypercubic(2, 100)
        }
    }
}

fn parse_crn(s: &str) -> CRN {
    match s {
        "brw" => CRN::brw_coalescent(),
        "gribov" => CRN::gribov_critical(),
        "pair_annihilation" => CRN::pair_annihilation(0.5),
        "coagulation" => CRN::coagulation(0.5),
        "birth_death" => CRN::birth_death(0.1, 0.1),
        _ => {
            eprintln!("Unknown CRN: {}. Using brw.", s);
            CRN::brw_coalescent()
        }
    }
}

fn print_help() {
    eprintln!(
        r#"rdft-sim: Fast particle simulator for reaction-diffusion systems

USAGE:
  rdft-sim                    Run full BRW validation suite (all graph types)
  rdft-sim [OPTIONS]          Run a single configuration

OPTIONS:
  --graph TYPE:PARAMS     Graph type (default: lattice:2:100)
                          lattice:DIM:SIZE   d-dimensional periodic lattice
                          sierpinski:LEVEL   Sierpinski carpet
                          tree:N:SEED        Random tree
                          ba:N:M             Barabási-Albert network
                          complete:N         Complete graph
  --crn TYPE              CRN type (default: brw)
                          brw                BRW with coalescence (Bordeu+ 2019)
                          gribov             Gribov process (A→2A, A→∅, 2A→A)
                          pair_annihilation  2A → ∅
                          coagulation        2A → A
                          birth_death        A → 2A, A → ∅
  --realizations N        Number of realizations (default: 1000)
  --tmax T                Max simulation time (default: 2000)
  --max-per-site N        Max particles per site (default: 1 for coalescence)
  --help                  This message

EXAMPLES:
  rdft-sim --graph lattice:3:30 --crn brw -n 5000 --tmax 1000
  rdft-sim --graph sierpinski:4 --crn gribov -n 2000 --max-per-site 0
  rdft-sim --graph ba:5000:3 --crn pair_annihilation -n 3000

OUTPUT:
  JSON to stdout, progress to stderr.
  Pipe to file: rdft-sim > results.json
  Plot: python simulations/python/plot_brw.py results.json
"#
    );
}
