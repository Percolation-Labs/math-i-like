/// Core simulation engine: tau-leaping on arbitrary graphs.
///
/// Algorithm per timestep:
///   1. Diffusion — each particle hops to a random neighbor.
///      For a site with n particles and k neighbors we sample a
///      multinomial (sequential binomial decomposition) in O(k), not O(n).
///   2. Reactions — at each occupied site, tau-leaping:
///      - Unary (k=1):  firings ~ Binomial(n, rate*dt)
///      - Binary (k=2): firings ~ Poisson(rate * C(n,2) * dt), capped at n/2
///   3. Coalescence cap — if max_per_site is set, truncate each site.
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
    /// Timestep size.
    pub dt: f64,
    /// Maximum simulation time.
    pub t_max: f64,
    /// Times at which to record observables (sorted).
    pub record_times: Vec<f64>,
    /// Starting site for the initial particle(s).
    pub origin: u32,
    /// Number of particles placed at origin at t=0.
    pub initial_count: u32,
    /// Maximum particles per site after reactions (0 = unlimited).
    /// Set to 1 for BRW with instant coalescence.
    pub max_per_site: u32,
    /// Maximum total particles (prevents runaway).
    pub max_particles: u32,
}

impl SimConfig {
    /// Sensible defaults for BRW simulations with coalescence.
    pub fn brw_default(t_max: f64) -> Self {
        // Log-spaced record times from t=1 to t_max
        let n_records = 60;
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
            max_per_site: 1, // instant coalescence for BRW
            max_particles: 1_000_000,
        }
    }

    /// Config for general CRN (no per-site cap).
    pub fn general(t_max: f64, dt: f64) -> Self {
        let n_records = 60;
        let log_min = dt.log10();
        let log_max = t_max.log10();
        let record_times: Vec<f64> = (0..n_records)
            .map(|i| {
                let frac = i as f64 / (n_records - 1) as f64;
                10f64.powf(log_min + frac * (log_max - log_min))
            })
            .collect();

        SimConfig {
            dt,
            t_max,
            record_times,
            origin: 0,
            initial_count: 1,
            max_per_site: 0, // no cap
            max_particles: 500_000,
        }
    }
}

// ----------------------------------------------------------------
// Single realization output
// ----------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct Realization {
    pub times: Vec<f64>,
    /// Number of distinct sites ever visited.
    pub volume: Vec<u32>,
    /// Current number of particles.
    pub population: Vec<u32>,
    pub survived: bool,
}

// ----------------------------------------------------------------
// Ensemble output
// ----------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct EnsembleResult {
    pub times: Vec<f64>,
    /// moments[p-1][t_idx] = ⟨V^p⟩ at that time (unconditional average over all realizations).
    pub moments: Vec<Vec<f64>>,
    /// moments_surviving[p-1][t_idx] = ⟨V^p | survived⟩
    pub moments_surviving: Vec<Vec<f64>>,
    pub max_p: usize,
    pub n_survived: u32,
    pub n_total: u32,
    /// convergence[p-1] = vec of (n_so_far, alpha_estimate) for surviving-walk average.
    pub convergence: Vec<Vec<(u32, f64)>>,
}

// ----------------------------------------------------------------
// Run one realization
// ----------------------------------------------------------------

pub fn run_realization(graph: &Graph, crn: &CRN, config: &SimConfig, seed: u64) -> Realization {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_sites = graph.num_nodes();

    // Sparse occupation: site -> particle count
    let mut occupation: HashMap<u32, u32> = HashMap::new();
    occupation.insert(config.origin, config.initial_count);

    // Visited tracking
    let mut visited = vec![false; n_sites];
    visited[config.origin as usize] = true;
    let mut n_visited: u32 = 1;

    // Output arrays
    let n_rec = config.record_times.len();
    let mut times = Vec::with_capacity(n_rec);
    let mut volumes = Vec::with_capacity(n_rec);
    let mut populations = Vec::with_capacity(n_rec);

    let mut t: f64 = 0.0;
    let mut next_rec: usize = 0;

    loop {
        // Record observables at scheduled times
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

        // --- Step 1: Reactions (before diffusion for BRW-style) ---
        if !crn.reactions.is_empty() {
            let sites: Vec<(u32, u32)> = occupation.drain().collect();
            for (site, mut n) in sites {
                for rxn in &crn.reactions {
                    if n == 0 {
                        break;
                    }
                    n = apply_reaction(&mut rng, n, rxn.k, rxn.l, rxn.rate, config.dt);
                }
                if config.max_per_site > 0 && n > config.max_per_site {
                    n = config.max_per_site;
                }
                if n > 0 {
                    occupation.insert(site, n);
                }
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

                // Distribute hopping particles among neighbors (multinomial via sequential binomial)
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

            // Apply per-site cap after diffusion too (coalescence on contact)
            if config.max_per_site > 0 {
                for val in occupation.values_mut() {
                    if *val > config.max_per_site {
                        *val = config.max_per_site;
                    }
                }
            }
        }

        // --- Mark newly visited sites ---
        for &site in occupation.keys() {
            let s = site as usize;
            if !visited[s] {
                visited[s] = true;
                n_visited += 1;
            }
        }

        // Cap total particles
        let total: u32 = occupation.values().sum();
        if total > config.max_particles {
            // Randomly cull to limit
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

        // Early exit if population extinct
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
// Reaction application (tau-leaping)
// ----------------------------------------------------------------

#[inline]
fn apply_reaction(rng: &mut impl Rng, n: u32, k: u32, l: u32, rate: f64, dt: f64) -> u32 {
    if n < k || rate <= 0.0 {
        return n;
    }

    let firings = if k == 1 {
        let p = (rate * dt).min(1.0);
        sample_binomial(rng, n, p)
    } else if k == 2 {
        let pairs = (n as f64) * (n as f64 - 1.0) / 2.0;
        let mean = rate * pairs * dt;
        let f = sample_poisson(rng, mean);
        f.min(n / 2)
    } else {
        let mut combos = 1u64;
        for i in 0..k {
            combos = combos * (n as u64 - i as u64) / (i as u64 + 1);
        }
        let mean = rate * combos as f64 * dt;
        let f = sample_poisson(rng, mean);
        f.min(n / k)
    };

    let removed = firings * k;
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
        let val = mean + std * rng.gen::<f64>();
        return val.max(0.0) as u32;
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

    // Unconditional moments (all realizations)
    let mut moments = vec![vec![0.0f64; n_rec]; max_p];
    let n_all = n_total as f64;
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

    // Conditional moments (surviving realizations only)
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

    // Convergence: running exponent estimate as a function of N
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
// Power-law fitting: log(y) = alpha * log(t) + const
// ----------------------------------------------------------------

pub fn fit_power_law(times: &[f64], values: &[f64]) -> f64 {
    let n = times.len();
    if n < 4 {
        return f64::NAN;
    }

    // Use middle 60% to avoid lattice artifacts and finite-size effects
    let start = n / 5;
    let end = 4 * n / 5;
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
        let g = Graph::hypercubic(1, 1000);
        let crn = CRN::brw_coalescent();
        let config = SimConfig::brw_default(100.0);
        let real = run_realization(&g, &crn, &config, 42);
        assert!(!real.times.is_empty());
        assert_eq!(real.times.len(), real.volume.len());
        // With coalescent BRW, the walk should survive most of the time
        assert!(real.volume.last().copied().unwrap_or(0) > 1);
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
        let crn = CRN::brw_coalescent();
        let config = SimConfig::brw_default(50.0);
        let result = run_ensemble(&g, &crn, &config, 20, 2);
        assert_eq!(result.moments.len(), 2);
        assert_eq!(result.times.len(), result.moments[0].len());
    }
}
