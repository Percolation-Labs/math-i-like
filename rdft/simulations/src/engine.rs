/// Core simulation engine: tau-leaping on arbitrary graphs.
///
/// Algorithm per timestep:
///   1. Reactions — at each occupied site:
///      - Unary reactions applied simultaneously via multinomial sampling
///        (avoids sequential bias: (1+β)(1-ε) ≠ 1 even when β=ε).
///      - Binary (k=2): Poisson tau-leaping.
///   2. Coalescence cap — if max_per_site > 0, truncate each site.
///   3. Diffusion — each particle hops to a random neighbor.
///      Multinomial O(degree) per site, not O(n_particles).
///   4. Book-keeping — update visited set, record observables.
///
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
    /// Max particles per site after reactions (0 = unlimited).
    /// Set to 1 for BRW with instant coalescence.
    pub max_per_site: u32,
    /// Global population cap.
    pub max_particles: u32,
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
            max_per_site: 0, // no cap — let reactions handle dynamics
            max_particles: 500_000,
        }
    }
}

// ----------------------------------------------------------------
// Output types
// ----------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct Realization {
    pub times: Vec<f64>,
    /// Number of distinct sites ever visited.
    pub volume: Vec<u32>,
    /// Current total particle count.
    pub population: Vec<u32>,
    pub survived: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct EnsembleResult {
    pub times: Vec<f64>,
    /// moments[p-1][t_idx] = ⟨V^p⟩ unconditional (all realizations).
    pub moments: Vec<Vec<f64>>,
    /// moments_surviving[p-1][t_idx] = ⟨V^p | survived⟩
    pub moments_surviving: Vec<Vec<f64>>,
    pub max_p: usize,
    pub n_survived: u32,
    pub n_total: u32,
    /// convergence[p-1] = vec of (n_so_far, alpha_estimate).
    pub convergence: Vec<Vec<(u32, f64)>>,
}

// ----------------------------------------------------------------
// Precomputed reaction classification
// ----------------------------------------------------------------

struct ReactionSets {
    /// Unary reactions (k=1) with their output particle count and probability.
    unary: Vec<(u32, f64)>, // (l, rate*dt)
    /// Binary reactions (k=2).
    binary: Vec<(u32, f64)>, // (l, rate*dt)
}

impl ReactionSets {
    fn from_crn(crn: &CRN, dt: f64) -> Self {
        let mut unary = Vec::new();
        let mut binary = Vec::new();
        for rxn in &crn.reactions {
            match rxn.k {
                1 => unary.push((rxn.l, rxn.rate * dt)),
                2 => binary.push((rxn.l, rxn.rate * dt)),
                _ => {} // higher order ignored
            }
        }
        ReactionSets { unary, binary }
    }
}

// ----------------------------------------------------------------
// Run one realization
// ----------------------------------------------------------------

pub fn run_realization(graph: &Graph, crn: &CRN, config: &SimConfig, seed: u64) -> Realization {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_sites = graph.num_nodes();
    let rxn_sets = ReactionSets::from_crn(crn, config.dt);

    // Sparse occupation: site -> particle count
    let mut occupation: HashMap<u32, u32> = HashMap::new();
    occupation.insert(config.origin, config.initial_count);

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
            let pop: u32 = occupation.values().sum();
            times.push(config.record_times[next_rec]);
            volumes.push(n_visited);
            populations.push(pop);
            next_rec += 1;
        }

        if t >= config.t_max || next_rec >= n_rec {
            break;
        }

        // --- Step 1: Reactions at each site ---
        let sites: Vec<(u32, u32)> = occupation.drain().collect();
        for (site, mut n) in sites {
            if n == 0 {
                continue;
            }

            // Unary reactions (applied SIMULTANEOUSLY via multinomial)
            n = apply_unary_multinomial(&mut rng, n, &rxn_sets.unary);

            // Binary reactions (tau-leaping)
            for &(l, rate_dt) in &rxn_sets.binary {
                if n < 2 {
                    break;
                }
                n = apply_binary(&mut rng, n, l, rate_dt);
            }

            // Per-site cap (coalescence)
            if config.max_per_site > 0 && n > config.max_per_site {
                n = config.max_per_site;
            }

            if n > 0 {
                occupation.insert(site, n);
            }
        }

        // --- Step 2: Diffusion ---
        if crn.diffusion_rate > 0.0 {
            let mut new_occ: HashMap<u32, u32> =
                HashMap::with_capacity(occupation.len() * 2);

            for (&site, &count) in &occupation {
                if count == 0 {
                    continue;
                }
                let nbrs = graph.neighbors(site);
                let deg = nbrs.len();
                if deg == 0 {
                    *new_occ.entry(site).or_insert(0) += count;
                    continue;
                }

                let hopping = if crn.diffusion_rate >= 1.0 {
                    count
                } else {
                    sample_binomial(&mut rng, count, crn.diffusion_rate)
                };
                let staying = count - hopping;
                if staying > 0 {
                    *new_occ.entry(site).or_insert(0) += staying;
                }

                // Multinomial: distribute hopping particles among neighbors
                let mut remaining = hopping;
                for (i, &nbr) in nbrs.iter().enumerate() {
                    if remaining == 0 {
                        break;
                    }
                    if i == deg - 1 {
                        *new_occ.entry(nbr).or_insert(0) += remaining;
                    } else {
                        let slots_left = (deg - i) as f64;
                        let going = sample_binomial(&mut rng, remaining, 1.0 / slots_left);
                        if going > 0 {
                            *new_occ.entry(nbr).or_insert(0) += going;
                        }
                        remaining -= going;
                    }
                }
            }

            occupation = new_occ;

            // Apply coalescence after diffusion too
            if config.max_per_site > 0 {
                for val in occupation.values_mut() {
                    if *val > config.max_per_site {
                        *val = config.max_per_site;
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
        let total: u32 = occupation.values().sum();
        if total > config.max_particles {
            let keep_frac = config.max_particles as f64 / total as f64;
            let sites: Vec<(u32, u32)> = occupation.drain().collect();
            for (site, n) in sites {
                let kept = sample_binomial(&mut rng, n, keep_frac);
                if kept > 0 {
                    occupation.insert(site, kept);
                }
            }
        }

        t += config.dt;

        if occupation.is_empty() {
            while next_rec < n_rec {
                times.push(config.record_times[next_rec]);
                volumes.push(n_visited);
                populations.push(0);
                next_rec += 1;
            }
            break;
        }
    }

    let survived = !occupation.is_empty();
    Realization {
        times,
        volume: volumes,
        population: populations,
        survived,
    }
}

// ----------------------------------------------------------------
// Unary reactions: simultaneous multinomial
// ----------------------------------------------------------------
// For each particle, exactly one outcome occurs:
//   - Reaction i fires: particle → l_i offspring, with prob p_i
//   - No reaction: particle survives as-is, with prob 1 - Σp_i
// This avoids the sequential bias where (1+β)(1-ε) ≠ 1 for β=ε.

#[inline]
fn apply_unary_multinomial(rng: &mut impl Rng, n: u32, reactions: &[(u32, f64)]) -> u32 {
    if n == 0 || reactions.is_empty() {
        return n;
    }

    let probs: Vec<f64> = reactions.iter().map(|&(_, p)| p.min(1.0)).collect();
    let p_sum: f64 = probs.iter().sum();

    if p_sum > 1.0 {
        // Rates too high for this dt — rescale
        let scale = 0.99 / p_sum;
        return apply_unary_multinomial_inner(
            rng,
            n,
            reactions,
            &probs.iter().map(|p| p * scale).collect::<Vec<_>>(),
        );
    }

    apply_unary_multinomial_inner(rng, n, reactions, &probs)
}

#[inline]
fn apply_unary_multinomial_inner(
    rng: &mut impl Rng,
    n: u32,
    reactions: &[(u32, f64)],
    probs: &[f64],
) -> u32 {
    // Sequential multinomial decomposition
    let mut remaining = n;
    let mut total_offspring: u32 = 0;
    let mut p_used: f64 = 0.0;

    for (i, &(l, _)) in reactions.iter().enumerate() {
        if remaining == 0 {
            break;
        }
        // Conditional probability for this reaction
        let p_cond = if 1.0 - p_used > 1e-12 {
            (probs[i] / (1.0 - p_used)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let n_i = sample_binomial(rng, remaining, p_cond);
        total_offspring += n_i * l; // l offspring per firing
        remaining -= n_i;
        p_used += probs[i];
    }

    // Remaining particles had no reaction → survive as 1 each
    total_offspring += remaining;
    total_offspring
}

// ----------------------------------------------------------------
// Binary reactions: Poisson tau-leaping
// ----------------------------------------------------------------

#[inline]
fn apply_binary(rng: &mut impl Rng, n: u32, l: u32, rate_dt: f64) -> u32 {
    if n < 2 || rate_dt <= 0.0 {
        return n;
    }
    let pairs = (n as f64) * (n as f64 - 1.0) / 2.0;
    let mean = rate_dt * pairs;
    let firings = sample_poisson(rng, mean).min(n / 2);
    // Each firing: consume 2 particles, produce l
    let removed = firings * 2;
    let added = firings * l;
    n.saturating_sub(removed) + added
}

// ----------------------------------------------------------------
// Distribution samplers
// ----------------------------------------------------------------

#[inline]
fn sample_binomial(rng: &mut impl Rng, n: u32, p: f64) -> u32 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    if let Ok(dist) = Binomial::new(n as u64, p) {
        dist.sample(rng) as u32
    } else {
        0
    }
}

#[inline]
fn sample_poisson(rng: &mut impl Rng, mean: f64) -> u32 {
    if mean <= 0.0 {
        return 0;
    }
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

/// Fit log(y) = alpha * log(t) + const in the late-time regime.
/// Uses the last 60% of the data (in log-space) to avoid early transients.
pub fn fit_power_law(times: &[f64], values: &[f64]) -> f64 {
    fit_power_law_window(times, values, 0.55, 0.92)
}

/// Fit with explicit start/end fractions of the data range.
pub fn fit_power_law_window(times: &[f64], values: &[f64], frac_start: f64, frac_end: f64) -> f64 {
    let n = times.len();
    if n < 4 {
        return f64::NAN;
    }

    let start = (n as f64 * frac_start) as usize;
    let end = (n as f64 * frac_end).ceil() as usize;
    let end = end.min(n);
    if end <= start + 2 {
        return f64::NAN;
    }

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

    if count < 3.0 {
        return f64::NAN;
    }

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
    fn test_critical_mean_offspring() {
        // Verify that unary multinomial preserves mean for β=ε
        let mut rng = SmallRng::seed_from_u64(12345);
        let reactions = vec![(2u32, 0.3f64), (0u32, 0.3f64)]; // branch, death
        let mut total = 0u64;
        let n_trials = 100_000;
        let n_init = 100u32;
        for _ in 0..n_trials {
            total += apply_unary_multinomial(&mut rng, n_init, &reactions) as u64;
        }
        let mean = total as f64 / n_trials as f64;
        // Should be very close to n_init (critical)
        assert!(
            (mean - n_init as f64).abs() < 1.0,
            "Mean offspring {} should be close to {}",
            mean,
            n_init
        );
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
