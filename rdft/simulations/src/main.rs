/// rdft-sim: Fast particle simulator for reaction-diffusion systems on graphs.
///
/// Reproduces the BRW scaling exponents from Bordeu, Amarteifio et al. (2019).
///
/// The key theoretical prediction (Bordeu+ 2019, Eqs. 2-5):
///   ⟨V^p⟩(t) ~ t^{(p·d_s - 2)/2}    for d_s < d_c = 4
///   ⟨V^p⟩(t) ~ t^{2p-1}              for d_s ≥ d_c = 4 (mean-field)
///
/// where V(t) = number of distinct sites ever visited (volume explored),
/// and the average is UNCONDITIONAL (over all realizations including extinct ones).
///
/// The unconditional scaling arises because:
///   - Surviving walks: V|surv ~ t^{d_s/2} (diffusive spreading)
///   - Survival probability: P_surv ~ 1/t (critical branching)
///   - So: ⟨V^p⟩ ~ (1/t) × t^{p·d_s/2} = t^{p·d_s/2 - 1} = t^{(p·d_s-2)/2}

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
    d_s: f64,
    n_realizations: u32,
    n_survived: u32,
    times: Vec<f64>,
    moments: Vec<Vec<f64>>,
    moments_surviving: Vec<Vec<f64>>,
    fitted_exponents: Vec<ExponentResult>,
    convergence: Vec<Vec<(u32, f64)>>,
    wall_time_secs: f64,
}

#[derive(Serialize)]
struct ExponentResult {
    p: usize,
    /// Unconditional fit: log-log slope of ⟨V^p⟩ vs t.
    alpha_uncond: f64,
    /// Conditional fit: log-log slope of ⟨V^p|survived⟩ vs t.
    alpha_cond: f64,
    /// Theory: (p*d_s - 2)/2 for unconditional.
    alpha_theory_uncond: f64,
    /// Theory: p*d_s/2 for conditional (surviving walks).
    alpha_theory_cond: f64,
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
        let (graph, crn, n_real, t_max, max_per_site, d_s) = parse_args(&args);
        let mut config = SimConfig::brw_default(t_max);
        config.max_per_site = max_per_site;
        // For multi-species CRNs, configure initial species and tracking
        if crn.name.starts_with("Prion") {
            config.initial_species = 1; // seed M (species 1)
            config.track_species = Some(1); // track M population
        } else if crn.name.starts_with("Lotka") {
            config.initial_species = 0; // seed prey A
            // Seed some predators too
        } else if crn.name == "A+B->0" {
            config.initial_species = 0; // seed A; B should be seeded separately
        }
        let result = run_single(&graph, &crn, &config, n_real, d_s);
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
    d_s: f64,
) -> RunOutput {
    let max_p = 3;
    let start = Instant::now();

    eprint!(
        "  Running {} on {} (d_s={:.2}, {} realizations)... ",
        crn.name, graph.name, d_s, n_real
    );
    std::io::stderr().flush().ok();

    let ensemble = engine::run_ensemble(graph, crn, config, n_real, max_p);
    let elapsed = start.elapsed().as_secs_f64();

    eprintln!(
        "done in {:.1}s ({}/{} survived)",
        elapsed, ensemble.n_survived, n_real
    );

    let d_c = 4.0;
    let fitted_exponents: Vec<ExponentResult> = (0..max_p)
        .map(|p_idx| {
            let p = (p_idx + 1) as f64;
            let alpha_uncond = engine::fit_power_law(&ensemble.times, &ensemble.moments[p_idx]);
            let alpha_cond =
                engine::fit_power_law(&ensemble.times, &ensemble.moments_surviving[p_idx]);

            let (alpha_theory_uncond, alpha_theory_cond) = if d_s < d_c {
                ((p * d_s - 2.0) / 2.0, p * d_s / 2.0)
            } else {
                (2.0 * p - 1.0, 2.0 * p) // mean-field
            };

            ExponentResult {
                p: p_idx + 1,
                alpha_uncond,
                alpha_cond,
                alpha_theory_uncond,
                alpha_theory_cond,
            }
        })
        .collect();

    for ex in &fitted_exponents {
        eprintln!(
            "    p={}: α_uncond={:.3} (th {:.3})  α_cond={:.3} (th {:.3})",
            ex.p, ex.alpha_uncond, ex.alpha_theory_uncond, ex.alpha_cond, ex.alpha_theory_cond
        );
    }

    RunOutput {
        graph: graph.name.clone(),
        crn: crn.name.clone(),
        d_s,
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

/// Full BRW validation suite from Bordeu, Amarteifio et al. (2019).
fn run_brw_suite() {
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  rdft-sim: BRW Validation Suite                        ║");
    eprintln!("║  Bordeu, Amarteifio et al. (2019) Sci. Rep. 9:15590    ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Model: Critical birth-death (β=ε=0.1), no coagulation");
    eprintln!("  Observable: V(t) = distinct sites ever visited");
    eprintln!("  Theory: ⟨V^p⟩ ~ t^{{(p·d_s - 2)/2}} [unconditional]");
    eprintln!("          ⟨V^p|surv⟩ ~ t^{{p·d_s/2}}  [conditional]");
    eprintln!();

    let crn = CRN::birth_death(0.1, 0.1);

    // (graph, d_s, t_max, n_realizations)
    let configs: Vec<(Graph, f64, f64, u32)> = vec![
        // 1D: d_s=1
        (Graph::hypercubic(1, 20000), 1.0, 10000.0, 20000),
        // 2D: d_s=2
        (Graph::hypercubic(2, 300), 2.0, 5000.0, 20000),
        // 3D: d_s=3
        (Graph::hypercubic(3, 50), 3.0, 5000.0, 20000),
        // 5D: d_s=5 >= d_c=4 → mean-field
        (Graph::hypercubic(5, 10), 5.0, 2000.0, 20000),
        // Sierpinski: d_s ≈ 1.86
        (Graph::sierpinski_carpet(4), 1.86, 5000.0, 20000),
        // Random tree: d_s ≈ 4/3
        (Graph::random_tree(20000, 42), 1.333, 5000.0, 20000),
        // BA network: d_s ≥ 4 → mean-field
        (Graph::barabasi_albert(5000, 3, 42), 4.0, 2000.0, 10000),
    ];

    let start = Instant::now();
    let mut runs = Vec::new();

    for (graph, d_s, t_max, n_real) in &configs {
        let config = SimConfig::brw_default(*t_max);
        let result = run_single(graph, &crn, &config, *n_real, *d_s);
        runs.push(result);
    }

    let total_time = start.elapsed().as_secs_f64();
    eprintln!("\n  Total wall time: {:.1}s", total_time);

    // Summary: conditional exponents (cleaner signal)
    eprintln!(
        "\n  CONDITIONAL EXPONENTS (surviving walks): α_cond/p → d_s/2"
    );
    eprintln!(
        "  {:<28} {:>5} {:>3} {:>8} {:>8} {:>8} {:>6}",
        "Graph", "d_s", "p", "α_cond", "Theory", "α/p", "Match"
    );
    eprintln!("  {}", "-".repeat(65));
    for run in &runs {
        for ex in &run.fitted_exponents {
            let ratio = ex.alpha_cond / ex.p as f64;
            let ok = if (ex.alpha_cond - ex.alpha_theory_cond).abs()
                < 0.3 * ex.alpha_theory_cond.abs().max(0.5)
            {
                "ok"
            } else {
                "MISS"
            };
            eprintln!(
                "  {:<28} {:>5.2} {:>3} {:>8.3} {:>8.3} {:>8.3} {:>6}",
                run.graph,
                run.d_s,
                ex.p,
                ex.alpha_cond,
                ex.alpha_theory_cond,
                ratio,
                ok
            );
        }
    }

    // Summary: unconditional exponents (BRW paper formula)
    eprintln!(
        "\n  UNCONDITIONAL EXPONENTS (all walks): α = (p·d_s - 2)/2"
    );
    eprintln!(
        "  {:<28} {:>3} {:>8} {:>8} {:>6}",
        "Graph", "p", "Meas.", "Theory", "Match"
    );
    eprintln!("  {}", "-".repeat(55));
    for run in &runs {
        for ex in &run.fitted_exponents {
            let ok = if (ex.alpha_uncond - ex.alpha_theory_uncond).abs() < 0.5 {
                "ok"
            } else {
                "MISS"
            };
            eprintln!(
                "  {:<28} {:>3} {:>8.3} {:>8.3} {:>6}",
                run.graph, ex.p, ex.alpha_uncond, ex.alpha_theory_uncond, ok
            );
        }
    }

    let suite = SuiteOutput { runs };
    let json = serde_json::to_string_pretty(&suite).unwrap();
    println!("{}", json);
}

fn parse_args(args: &[String]) -> (Graph, CRN, u32, f64, u32, f64) {
    let mut graph_str = "lattice:2:100".to_string();
    let mut crn_str = "gribov".to_string();
    let mut n_real = 1000u32;
    let mut t_max = 2000.0f64;
    let mut max_per_site = 0u32;
    let mut d_s = 2.0f64;
    let mut rates_str: Option<String> = None;

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
            "--ds" => {
                d_s = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--rates" => {
                rates_str = Some(args[i + 1].clone());
                i += 2;
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                i += 1;
            }
        }
    }

    let graph = parse_graph(&graph_str);
    let crn = parse_crn_with_rates(&crn_str, rates_str.as_deref());
    (graph, crn, n_real, t_max, max_per_site, d_s)
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

fn parse_crn_with_rates(s: &str, rates: Option<&str>) -> CRN {
    // Parse comma-separated rates if provided
    let r: Vec<f64> = rates
        .map(|r| r.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_default();

    match s {
        "prion" if !r.is_empty() => {
            // --rates beta,lambda,mu_h,mu_m
            CRN::prion(
                r.get(0).copied().unwrap_or(0.175),
                r.get(1).copied().unwrap_or(1.0),
                r.get(2).copied().unwrap_or(0.5),
                r.get(3).copied().unwrap_or(0.1),
            )
        }
        "birth_death" if !r.is_empty() => {
            CRN::birth_death(
                r.get(0).copied().unwrap_or(0.1),
                r.get(1).copied().unwrap_or(0.1),
            )
        }
        _ => parse_crn(s),
    }
}

fn parse_crn(s: &str) -> CRN {
    match s {
        "brw" => CRN::brw_coalescent(),
        "gribov" => CRN::gribov_critical(),
        "pair_annihilation" => CRN::pair_annihilation(0.5),
        "coagulation" => CRN::coagulation(0.5),
        "birth_death" => CRN::birth_death(0.1, 0.1),
        "contact" => CRN::contact_process(0.3, 0.1),
        "triplet" => CRN::triplet_annihilation(0.3),
        "barw_even" => CRN::barw_even(0.3, 0.3),
        "pcpd" => CRN::pcpd(0.3, 0.3),
        "schlogl2" => CRN::schlogl_second(0.3, 0.1),
        "prion" => CRN::prion(0.175, 1.0, 0.5, 0.1),
        "prion_critical" => CRN::prion_critical(1.0, 0.5, 0.1),
        "lotka_volterra" => CRN::lotka_volterra(0.5, 0.3, 0.1),
        "ab_annihilation" => CRN::two_species_annihilation(0.5),
        "michaelis_menten" => CRN::michaelis_menten(1.0, 0.5, 1.0),
        _ => {
            eprintln!("Unknown CRN: {}. Using birth_death.", s);
            CRN::birth_death(0.1, 0.1)
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
  --crn TYPE              CRN type (default: gribov)
                          --- Single species ---
                          brw                BRW with coalescence
                          gribov             Gribov (A->2A, A->0, 2A->A)
                          birth_death        Critical (A->2A, A->0)
                          pair_annihilation  2A -> 0
                          coagulation        2A -> A
                          contact            Contact process (A->2A, 2A->0)
                          triplet            3A -> 0
                          barw_even          A->3A, 2A->0 (parity-conserving)
                          pcpd               2A->3A, 2A->0
                          schlogl2           2A->3A, A->0 (DP class)
                          --- Multi-species ---
                          prion              H+M->2M, 0->H, H->0, M->0
                          prion_critical     Prion at mean-field R0=1
                          lotka_volterra     A->2A, A+B->2B, B->0
                          ab_annihilation    A+B -> 0
                          michaelis_menten   E+S->ES, ES->E+S, ES->E+P
  --ds D                  Spectral dimension (for theory comparison)
  --realizations N        Number of realizations (default: 1000)
  --tmax T                Max simulation time (default: 2000)
  --max-per-site N        Max particles per site (0=unlimited, default: 0)
  --help                  This message

EXAMPLES:
  rdft-sim --graph lattice:3:30 --crn birth_death -n 20000 --tmax 5000 --ds 3
  rdft-sim --graph sierpinski:4 --crn birth_death -n 20000 --ds 1.86
  rdft-sim --graph ba:5000:3 --crn birth_death -n 10000 --ds 4

OUTPUT:
  JSON to stdout, progress to stderr.
  Pipe to file: rdft-sim > results.json
  Plot: python simulations/python/plot_brw.py results.json
"#
    );
}
