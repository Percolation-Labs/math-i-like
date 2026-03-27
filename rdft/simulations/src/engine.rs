/// Core simulation engine: tau-leaping on arbitrary graphs.
///
/// Supports arbitrary multi-species CRNs. Each site stores a vector
/// of per-species particle counts. Reactions are applied via:
///   - Source reactions (no reactants): Poisson production per site
///   - Unary reactions (1 particle): multinomial sampling
///   - Binary same-species (2 of same): Poisson tau-leaping on pairs
///   - Binary cross-species (1+1 different): Poisson on product of counts
///   - Higher-order: Poisson on combinatorial count
///
/// Diffusion is per-species with independent rates.
/// Realizations are embarrassingly parallel via rayon.

use std::collections::HashMap;

use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Binomial, Poisson};
use rayon::prelude::*;
use serde::Serialize;

use crate::crn::CRN;
use crate::graph::Graph;

// ----------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub dt: f64,
    pub t_max: f64,
    pub record_times: Vec<f64>,
    pub origin: u32,
    pub initial_count: u32,
    /// Which species to seed initially (index). Default 0.
    /// For prion: seed species 1 (M) at origin.
    pub initial_species: usize,
    /// Max particles per site per species after reactions (0 = unlimited).
    pub max_per_site: u32,
    /// Global population cap (total across all species).
    pub max_particles: u32,
    /// Which species to track for "population" observable.
    /// None = total all species. Some(i) = species i only.
    pub track_species: Option<usize>,
}

impl SimConfig {
    pub fn brw_default(t_max: f64) -> Self {
        let n_records = 80;
        let log_min = 0.0f64;
        let log_max = t_max.log10();
        let record_times: Vec<f64> = (0..n_records)
            .map(|i| {
                let frac = i as f64 / (n_records - 1) as f64;
                10f64.powf(log_min + frac * (log_max - log_min))
            })
            .collect();

        SimConfig {
            dt: 1.0,
            t_max,
            record_times,
            origin: 0,
            initial_count: 1,
            initial_species: 0,
            max_per_site: 0,
            max_particles: 500_000,
            track_species: None,
        }
    }
}

// ----------------------------------------------------------------
// Output types
// ----------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct Realization {
    pub times: Vec<f64>,
    /// Number of distinct sites ever visited (by any species).
    pub volume: Vec<u32>,
    /// Tracked population at each time.
    pub population: Vec<u32>,
    pub survived: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct EnsembleResult {
    pub times: Vec<f64>,
    pub moments: Vec<Vec<f64>>,
    pub moments_surviving: Vec<Vec<f64>>,
    pub max_p: usize,
    pub n_survived: u32,
    pub n_total: u32,
    pub convergence: Vec<Vec<(u32, f64)>>,
}

// ----------------------------------------------------------------
// Site state: per-species counts
// ----------------------------------------------------------------

type SiteState = Vec<u32>; // counts[species_idx]

fn total_count(state: &SiteState) -> u32 {
    state.iter().sum()
}

fn is_empty(state: &SiteState) -> bool {
    state.iter().all(|&c| c == 0)
}

// ----------------------------------------------------------------
// Reaction classification
// ----------------------------------------------------------------

#[derive(Clone, Debug)]
enum ReactionType {
    /// No reactants needed (source): ∅ → products. Poisson production.
    Source { products: Vec<u32>, rate_dt: f64 },
    /// Exactly one particle of one species consumed.
    Unary { species: usize, products: Vec<u32>, rate_dt: f64 },
    /// Two particles of the SAME species consumed.
    BinarySame { species: usize, products: Vec<u32>, rate_dt: f64 },
    /// One particle each of two DIFFERENT species consumed.
    BinaryCross { sp_a: usize, sp_b: usize, products: Vec<u32>, rate_dt: f64 },
    /// Three particles of same species.
    TernarySame { species: usize, products: Vec<u32>, rate_dt: f64 },
    /// General higher-order (fallback).
    General { reactants: Vec<u32>, products: Vec<u32>, rate_dt: f64 },
}

fn classify_reactions(crn: &CRN, dt: f64) -> Vec<ReactionType> {
    let mut classified = Vec::new();
    for rxn in &crn.reactions {
        let total_r: u32 = rxn.reactants.iter().sum();
        let rate_dt = rxn.rate * dt;

        if total_r == 0 {
            classified.push(ReactionType::Source {
                products: rxn.products.clone(),
                rate_dt,
            });
        } else if total_r == 1 {
            let sp = rxn.reactants.iter().position(|&r| r > 0).unwrap();
            classified.push(ReactionType::Unary {
                species: sp,
                products: rxn.products.clone(),
                rate_dt,
            });
        } else if total_r == 2 {
            let nonzero: Vec<(usize, u32)> = rxn.reactants.iter()
                .enumerate()
                .filter(|(_, &r)| r > 0)
                .map(|(i, &r)| (i, r))
                .collect();
            if nonzero.len() == 1 && nonzero[0].1 == 2 {
                classified.push(ReactionType::BinarySame {
                    species: nonzero[0].0,
                    products: rxn.products.clone(),
                    rate_dt,
                });
            } else if nonzero.len() == 2 {
                classified.push(ReactionType::BinaryCross {
                    sp_a: nonzero[0].0,
                    sp_b: nonzero[1].0,
                    products: rxn.products.clone(),
                    rate_dt,
                });
            } else {
                classified.push(ReactionType::General {
                    reactants: rxn.reactants.clone(),
                    products: rxn.products.clone(),
                    rate_dt,
                });
            }
        } else if total_r == 3 {
            let nonzero: Vec<(usize, u32)> = rxn.reactants.iter()
                .enumerate()
                .filter(|(_, &r)| r > 0)
                .map(|(i, &r)| (i, r))
                .collect();
            if nonzero.len() == 1 && nonzero[0].1 == 3 {
                classified.push(ReactionType::TernarySame {
                    species: nonzero[0].0,
                    products: rxn.products.clone(),
                    rate_dt,
                });
            } else {
                classified.push(ReactionType::General {
                    reactants: rxn.reactants.clone(),
                    products: rxn.products.clone(),
                    rate_dt,
                });
            }
        } else {
            classified.push(ReactionType::General {
                reactants: rxn.reactants.clone(),
                products: rxn.products.clone(),
                rate_dt,
            });
        }
    }
    classified
}

// ----------------------------------------------------------------
// Run one realization
// ----------------------------------------------------------------

pub fn run_realization(graph: &Graph, crn: &CRN, config: &SimConfig, seed: u64) -> Realization {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_sites = graph.num_nodes();
    let n_sp = crn.n_species;
    let rxn_types = classify_reactions(crn, config.dt);

    // Sparse occupation: site -> [count_per_species]
    let mut occupation: HashMap<u32, SiteState> = HashMap::new();
    let mut init_state = vec![0u32; n_sp];
    init_state[config.initial_species] = config.initial_count;
    occupation.insert(config.origin, init_state);

    let mut visited = vec![false; n_sites];
    visited[config.origin as usize] = true;
    let mut n_visited: u32 = 1;

    let n_rec = config.record_times.len();
    let mut times = Vec::with_capacity(n_rec);
    let mut volumes = Vec::with_capacity(n_rec);
    let mut populations = Vec::with_capacity(n_rec);

    let mut t: f64 = 0.0;
    let mut next_rec: usize = 0;

    loop {
        // Record observables
        while next_rec < n_rec && t >= config.record_times[next_rec] {
            let pop: u32 = match config.track_species {
                Some(sp) => occupation.values().map(|s| s[sp]).sum(),
                None => occupation.values().map(|s| total_count(s)).sum(),
            };
            times.push(config.record_times[next_rec]);
            volumes.push(n_visited);
            populations.push(pop);
            next_rec += 1;
        }

        if t >= config.t_max || next_rec >= n_rec {
            break;
        }

        // --- Step 1: Reactions at each site ---
        let sites: Vec<(u32, SiteState)> = occupation.drain().collect();

        // First: apply source reactions (production at ALL sites, not just occupied)
        // For efficiency, only apply at sites that exist + a Poisson number of new sites
        let mut new_occupation: HashMap<u32, SiteState> = HashMap::with_capacity(sites.len() * 2);

        // Source reactions: produce at each existing site
        for rxn in &rxn_types {
            if let ReactionType::Source { products, rate_dt } = rxn {
                for &(site, _) in &sites {
                    let produced: Vec<u32> = products.iter()
                        .map(|&p| if p > 0 { sample_poisson(&mut rng, *rate_dt * p as f64) } else { 0 })
                        .collect();
                    if produced.iter().any(|&p| p > 0) {
                        let entry = new_occupation.entry(site).or_insert_with(|| vec![0; n_sp]);
                        for (i, &p) in produced.iter().enumerate() {
                            entry[i] += p;
                        }
                    }
                }
            }
        }

        // Non-source reactions at occupied sites
        for (site, mut state) in sites {
            // Apply non-source reactions
            for rxn in &rxn_types {
                match rxn {
                    ReactionType::Source { .. } => {} // already handled
                    ReactionType::Unary { species, products, rate_dt } => {
                        let n = state[*species];
                        if n > 0 {
                            let firings = sample_binomial(&mut rng, n, rate_dt.min(1.0));
                            state[*species] -= firings;
                            for (i, &p) in products.iter().enumerate() {
                                state[i] += firings * p;
                            }
                        }
                    }
                    ReactionType::BinarySame { species, products, rate_dt } => {
                        let n = state[*species];
                        if n >= 2 {
                            let pairs = (n as f64) * (n as f64 - 1.0) / 2.0;
                            let mean = rate_dt * pairs;
                            let firings = sample_poisson(&mut rng, mean).min(n / 2);
                            state[*species] -= firings * 2;
                            for (i, &p) in products.iter().enumerate() {
                                state[i] += firings * p;
                            }
                        }
                    }
                    ReactionType::BinaryCross { sp_a, sp_b, products, rate_dt } => {
                        let na = state[*sp_a];
                        let nb = state[*sp_b];
                        if na > 0 && nb > 0 {
                            let combos = na as f64 * nb as f64;
                            let mean = rate_dt * combos;
                            let firings = sample_poisson(&mut rng, mean)
                                .min(na).min(nb);
                            state[*sp_a] -= firings;
                            state[*sp_b] -= firings;
                            for (i, &p) in products.iter().enumerate() {
                                state[i] += firings * p;
                            }
                        }
                    }
                    ReactionType::TernarySame { species, products, rate_dt } => {
                        let n = state[*species];
                        if n >= 3 {
                            let triples = (n as f64) * (n as f64 - 1.0) * (n as f64 - 2.0) / 6.0;
                            let mean = rate_dt * triples;
                            let firings = sample_poisson(&mut rng, mean).min(n / 3);
                            state[*species] -= firings * 3;
                            for (i, &p) in products.iter().enumerate() {
                                state[i] += firings * p;
                            }
                        }
                    }
                    ReactionType::General { reactants, products, rate_dt } => {
                        // Check all reactants available
                        let feasible = reactants.iter().enumerate()
                            .all(|(i, &r)| state[i] >= r);
                        if feasible {
                            // Combinatorial count of reactant tuples
                            let mut combos = 1.0f64;
                            for (i, &r) in reactants.iter().enumerate() {
                                let n = state[i] as f64;
                                // C(n, r) ≈ n^r / r! for tau-leaping
                                for j in 0..r {
                                    combos *= (n - j as f64) / (j as f64 + 1.0);
                                }
                            }
                            let mean = rate_dt * combos;
                            let max_firings = reactants.iter().enumerate()
                                .filter(|(_, &r)| r > 0)
                                .map(|(i, &r)| state[i] / r)
                                .min()
                                .unwrap_or(0);
                            let firings = sample_poisson(&mut rng, mean).min(max_firings);
                            for (i, &r) in reactants.iter().enumerate() {
                                state[i] -= firings * r;
                            }
                            for (i, &p) in products.iter().enumerate() {
                                state[i] += firings * p;
                            }
                        }
                    }
                }
            }

            // Per-site cap
            if config.max_per_site > 0 {
                for c in state.iter_mut() {
                    if *c > config.max_per_site {
                        *c = config.max_per_site;
                    }
                }
            }

            // Merge with source-produced particles
            if !is_empty(&state) {
                let entry = new_occupation.entry(site).or_insert_with(|| vec![0; n_sp]);
                for (i, &c) in state.iter().enumerate() {
                    entry[i] += c;
                }
            }
        }

        occupation = new_occupation;
        // Remove empty sites
        occupation.retain(|_, s| !is_empty(s));

        // --- Step 2: Diffusion (per species) ---
        let mut diff_occupation: HashMap<u32, SiteState> =
            HashMap::with_capacity(occupation.len() * 2);

        for (&site, state) in &occupation {
            let nbrs = graph.neighbors(site);
            let deg = nbrs.len();
            if deg == 0 {
                let entry = diff_occupation.entry(site).or_insert_with(|| vec![0; n_sp]);
                for (i, &c) in state.iter().enumerate() {
                    entry[i] += c;
                }
                continue;
            }

            for sp in 0..n_sp {
                let count = state[sp];
                if count == 0 { continue; }

                let d_rate = crn.diffusion_rates[sp];
                let hopping = if d_rate >= 1.0 {
                    count
                } else if d_rate <= 0.0 {
                    0
                } else {
                    sample_binomial(&mut rng, count, d_rate)
                };
                let staying = count - hopping;

                if staying > 0 {
                    let entry = diff_occupation.entry(site).or_insert_with(|| vec![0; n_sp]);
                    entry[sp] += staying;
                }

                // Distribute hoppers among neighbors (multinomial)
                let mut remaining = hopping;
                for (i, &nbr) in nbrs.iter().enumerate() {
                    if remaining == 0 { break; }
                    if i == deg - 1 {
                        let entry = diff_occupation.entry(nbr).or_insert_with(|| vec![0; n_sp]);
                        entry[sp] += remaining;
                    } else {
                        let slots_left = (deg - i) as f64;
                        let going = sample_binomial(&mut rng, remaining, 1.0 / slots_left);
                        if going > 0 {
                            let entry = diff_occupation.entry(nbr).or_insert_with(|| vec![0; n_sp]);
                            entry[sp] += going;
                        }
                        remaining -= going;
                    }
                }
            }
        }

        occupation = diff_occupation;
        occupation.retain(|_, s| !is_empty(s));

        // Coalescence cap after diffusion
        if config.max_per_site > 0 {
            for state in occupation.values_mut() {
                for c in state.iter_mut() {
                    if *c > config.max_per_site {
                        *c = config.max_per_site;
                    }
                }
            }
        }

        // --- Mark visited sites ---
        for &site in occupation.keys() {
            let s = site as usize;
            if !visited[s] {
                visited[s] = true;
                n_visited += 1;
            }
        }

        // Global population cap
        let grand_total: u32 = occupation.values().map(|s| total_count(s)).sum();
        if grand_total > config.max_particles {
            let keep_frac = config.max_particles as f64 / grand_total as f64;
            let sites: Vec<(u32, SiteState)> = occupation.drain().collect();
            for (site, state) in sites {
                let new_state: SiteState = state.iter()
                    .map(|&c| sample_binomial(&mut rng, c, keep_frac))
                    .collect();
                if !is_empty(&new_state) {
                    occupation.insert(site, new_state);
                }
            }
        }

        t += config.dt;

        // Check extinction of tracked species
        let tracked_alive = match config.track_species {
            Some(sp) => occupation.values().any(|s| s[sp] > 0),
            None => !occupation.is_empty(),
        };
        if !tracked_alive {
            while next_rec < n_rec {
                times.push(config.record_times[next_rec]);
                volumes.push(n_visited);
                populations.push(0);
                next_rec += 1;
            }
            break;
        }
    }

    let survived = match config.track_species {
        Some(sp) => occupation.values().any(|s| s[sp] > 0),
        None => !occupation.is_empty(),
    };

    Realization {
        times,
        volume: volumes,
        population: populations,
        survived,
    }
}

// ----------------------------------------------------------------
// Distribution samplers
// ----------------------------------------------------------------

#[inline]
fn sample_binomial(rng: &mut impl Rng, n: u32, p: f64) -> u32 {
    if n == 0 || p <= 0.0 { return 0; }
    if p >= 1.0 { return n; }
    if let Ok(dist) = Binomial::new(n as u64, p) {
        dist.sample(rng) as u32
    } else {
        0
    }
}

#[inline]
fn sample_poisson(rng: &mut impl Rng, mean: f64) -> u32 {
    if mean <= 0.0 { return 0; }
    if mean > 1e6 {
        let std = mean.sqrt();
        return (mean + std * rng.gen::<f64>()).max(0.0) as u32;
    }
    if let Ok(dist) = Poisson::new(mean) {
        let val: f64 = dist.sample(rng);
        val as u32
    } else {
        0
    }
}

// ----------------------------------------------------------------
// Ensemble run (parallel via rayon)
// ----------------------------------------------------------------

pub fn run_ensemble(
    graph: &Graph,
    crn: &CRN,
    config: &SimConfig,
    n_realizations: u32,
    max_p: usize,
) -> EnsembleResult {
    let realizations: Vec<Realization> = (0..n_realizations)
        .into_par_iter()
        .map(|i| {
            let seed =
                (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            run_realization(graph, crn, config, seed)
        })
        .collect();

    let n_rec = config.record_times.len();
    let n_survived = realizations.iter().filter(|r| r.survived).count() as u32;
    let n_total = n_realizations;

    // Unconditional moments
    let mut moments = vec![vec![0.0f64; n_rec]; max_p];
    let n_all = n_total.max(1) as f64;
    for real in &realizations {
        for t_idx in 0..real.volume.len().min(n_rec) {
            let v = real.volume[t_idx] as f64;
            let mut vp = v;
            for p in 0..max_p {
                moments[p][t_idx] += vp / n_all;
                vp *= v;
            }
        }
    }

    // Conditional moments (surviving)
    let surviving: Vec<&Realization> = realizations.iter().filter(|r| r.survived).collect();
    let n_surv = surviving.len().max(1) as f64;
    let mut moments_surviving = vec![vec![0.0f64; n_rec]; max_p];
    for real in &surviving {
        for t_idx in 0..real.volume.len().min(n_rec) {
            let v = real.volume[t_idx] as f64;
            let mut vp = v;
            for p in 0..max_p {
                moments_surviving[p][t_idx] += vp / n_surv;
                vp *= v;
            }
        }
    }

    // Convergence data
    let batch_size = (n_realizations / 40).max(1);
    let mut convergence = vec![Vec::new(); max_p];
    let mut running_sums = vec![vec![0.0f64; n_rec]; max_p];
    let mut n_so_far = 0u32;

    for real in &realizations {
        n_so_far += 1;
        for t_idx in 0..real.volume.len().min(n_rec) {
            let v = real.volume[t_idx] as f64;
            let mut vp = v;
            for p in 0..max_p {
                running_sums[p][t_idx] += vp;
                vp *= v;
            }
        }

        if n_so_far % batch_size == 0 || n_so_far == n_total {
            let ns = n_so_far as f64;
            for p in 0..max_p {
                let means: Vec<f64> = running_sums[p].iter().map(|&s| s / ns).collect();
                let alpha = fit_power_law(&config.record_times, &means);
                convergence[p].push((n_so_far, alpha));
            }
        }
    }

    EnsembleResult {
        times: config.record_times.clone(),
        moments,
        moments_surviving,
        max_p,
        n_survived,
        n_total,
        convergence,
    }
}

// ----------------------------------------------------------------
// Power-law fitting
// ----------------------------------------------------------------

pub fn fit_power_law(times: &[f64], values: &[f64]) -> f64 {
    fit_power_law_window(times, values, 0.55, 0.92)
}

pub fn fit_power_law_window(times: &[f64], values: &[f64], frac_start: f64, frac_end: f64) -> f64 {
    let n = times.len();
    if n < 4 { return f64::NAN; }

    let start = (n as f64 * frac_start) as usize;
    let end = (n as f64 * frac_end).ceil() as usize;
    let end = end.min(n);
    if end <= start + 2 { return f64::NAN; }

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut count = 0.0f64;

    for i in start..end {
        if times[i] > 1.0 && values[i] > 0.0 {
            let x = times[i].ln();
            let y = values[i].ln();
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1.0;
        }
    }

    if count < 3.0 { return f64::NAN; }
    (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crn::CRN;
    use crate::graph::Graph;

    #[test]
    fn test_single_realization_runs() {
        let g = Graph::hypercubic(2, 100);
        let crn = CRN::gribov_critical();
        let config = SimConfig::brw_default(100.0);
        let real = run_realization(&g, &crn, &config, 42);
        assert!(!real.times.is_empty());
        assert_eq!(real.times.len(), real.volume.len());
    }

    #[test]
    fn test_prion_realization_runs() {
        let g = Graph::hypercubic(1, 100);
        let crn = CRN::prion(0.5, 1.0, 0.5, 0.1);
        let mut config = SimConfig::brw_default(50.0);
        config.initial_species = 1; // seed M
        config.track_species = Some(1); // track M
        let real = run_realization(&g, &crn, &config, 42);
        assert!(!real.times.is_empty());
    }

    #[test]
    fn test_multispecies_ensemble() {
        let g = Graph::hypercubic(1, 50);
        let crn = CRN::prion(0.5, 1.0, 0.5, 0.1);
        let mut config = SimConfig::brw_default(20.0);
        config.initial_species = 1;
        config.track_species = Some(1);
        let result = run_ensemble(&g, &crn, &config, 50, 2);
        assert_eq!(result.moments.len(), 2);
    }

    #[test]
    fn test_power_law_fit() {
        let times: Vec<f64> = (1..100).map(|i| i as f64).collect();
        let values: Vec<f64> = times.iter().map(|t| t.sqrt()).collect();
        let alpha = fit_power_law(&times, &values);
        assert!((alpha - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_ensemble_small() {
        let g = Graph::hypercubic(2, 50);
        let crn = CRN::gribov_critical();
        let config = SimConfig::brw_default(50.0);
        let result = run_ensemble(&g, &crn, &config, 50, 2);
        assert_eq!(result.moments.len(), 2);
        assert_eq!(result.times.len(), result.moments[0].len());
    }
}
