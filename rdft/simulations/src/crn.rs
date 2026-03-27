/// Chemical Reaction Network specification for multi-species systems.
///
/// Each reaction specifies per-species reactant and product counts.
/// Single-species reactions are the special case with n_species=1.
///
/// Examples:
///   H + M → 2M:  reactants=[1,1], products=[0,2]
///   ∅ → H:       reactants=[0,0], products=[1,0]
///   2A → ∅:      reactants=[2],   products=[0]

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reaction {
    /// Per-species reactant counts consumed per firing.
    pub reactants: Vec<u32>,
    /// Per-species product counts created per firing.
    pub products: Vec<u32>,
    /// Rate constant.
    pub rate: f64,
    pub name: String,
}

impl Reaction {
    /// Total number of reactant particles.
    pub fn total_reactants(&self) -> u32 {
        self.reactants.iter().sum()
    }

    /// Single-species convenience: k reactants → l products.
    pub fn single(k: u32, l: u32, rate: f64, name: &str) -> Self {
        Reaction {
            reactants: vec![k],
            products: vec![l],
            rate,
            name: name.into(),
        }
    }

    /// Two-species convenience.
    pub fn two_species(k: [u32; 2], l: [u32; 2], rate: f64, name: &str) -> Self {
        Reaction {
            reactants: k.to_vec(),
            products: l.to_vec(),
            rate,
            name: name.into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CRN {
    pub reactions: Vec<Reaction>,
    pub n_species: usize,
    pub species_names: Vec<String>,
    /// Per-species diffusion rates.
    pub diffusion_rates: Vec<f64>,
    pub name: String,
}

impl CRN {
    // ----------------------------------------------------------------
    // Single-species factories (backward compatible)
    // ----------------------------------------------------------------

    /// Gribov process (BRW): A->2A (beta), A->0 (epsilon), 2A->A (chi).
    pub fn gribov(beta: f64, epsilon: f64, chi: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction::single(1, 2, beta, "A->2A"),
                Reaction::single(1, 0, epsilon, "A->0"),
                Reaction::single(2, 1, chi, "2A->A"),
            ],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "Gribov (BRW)".into(),
        }
    }

    pub fn gribov_critical() -> Self {
        Self::gribov(0.3, 0.3, 0.1)
    }

    pub fn brw_coalescent() -> Self {
        CRN {
            reactions: vec![Reaction::single(1, 2, 0.5, "A->2A")],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "BRW (coalescent)".into(),
        }
    }

    pub fn pair_annihilation(rate: f64) -> Self {
        CRN {
            reactions: vec![Reaction::single(2, 0, rate, "2A->0")],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "Pair annihilation".into(),
        }
    }

    pub fn coagulation(rate: f64) -> Self {
        CRN {
            reactions: vec![Reaction::single(2, 1, rate, "2A->A")],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "Coagulation".into(),
        }
    }

    pub fn birth_death(birth: f64, death: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction::single(1, 2, birth, "A->2A"),
                Reaction::single(1, 0, death, "A->0"),
            ],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "Birth-death".into(),
        }
    }

    pub fn contact_process(birth: f64, annihilation: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction::single(1, 2, birth, "A->2A"),
                Reaction::single(2, 0, annihilation, "2A->0"),
            ],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "Contact process".into(),
        }
    }

    pub fn triplet_annihilation(rate: f64) -> Self {
        CRN {
            reactions: vec![Reaction::single(3, 0, rate, "3A->0")],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "Triplet annihilation".into(),
        }
    }

    // ----------------------------------------------------------------
    // Multi-species factories
    // ----------------------------------------------------------------

    /// Prion propagation: H + M → 2M, ∅ → H, H → ∅, M → ∅.
    /// Species 0 = H (healthy), Species 1 = M (misfolded).
    pub fn prion(beta: f64, lam: f64, mu_h: f64, mu_m: f64) -> Self {
        CRN {
            reactions: vec![
                // H + M → 2M: consume 1H + 1M, produce 0H + 2M
                Reaction::two_species([1, 1], [0, 2], beta, "H+M->2M"),
                // ∅ → H: consume nothing, produce 1H
                Reaction::two_species([0, 0], [1, 0], lam, "0->H"),
                // H → ∅: consume 1H
                Reaction::two_species([1, 0], [0, 0], mu_h, "H->0"),
                // M → ∅: consume 1M
                Reaction::two_species([0, 1], [0, 0], mu_m, "M->0"),
            ],
            n_species: 2,
            species_names: vec!["H".into(), "M".into()],
            diffusion_rates: vec![1.0, 1.0],
            name: format!("Prion (β={beta}, λ={lam}, μH={mu_h}, μM={mu_m})"),
        }
    }

    /// Prion at mean-field criticality: R₀ = β·λ/(μ_H·μ_M) = 1.
    pub fn prion_critical(lam: f64, mu_h: f64, mu_m: f64) -> Self {
        let beta = mu_h * mu_m / lam;
        Self::prion(beta, lam, mu_h, mu_m)
    }

    /// Lotka-Volterra: A→2A, A+B→2B, B→∅.
    /// Species 0 = A (prey), Species 1 = B (predator).
    pub fn lotka_volterra(sigma: f64, lambda: f64, mu: f64) -> Self {
        CRN {
            reactions: vec![
                // A → 2A: prey reproduction
                Reaction::two_species([1, 0], [2, 0], sigma, "A->2A"),
                // A + B → 2B: predation
                Reaction::two_species([1, 1], [0, 2], lambda, "A+B->2B"),
                // B → ∅: predator death
                Reaction::two_species([0, 1], [0, 0], mu, "B->0"),
            ],
            n_species: 2,
            species_names: vec!["A".into(), "B".into()],
            diffusion_rates: vec![1.0, 1.0],
            name: "Lotka-Volterra".into(),
        }
    }

    /// Two-species annihilation: A + B → ∅.
    pub fn two_species_annihilation(rate: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction::two_species([1, 1], [0, 0], rate, "A+B->0"),
            ],
            n_species: 2,
            species_names: vec!["A".into(), "B".into()],
            diffusion_rates: vec![1.0, 1.0],
            name: "A+B->0".into(),
        }
    }

    /// BARW-even: A→3A, 2A→∅ (parity-conserving).
    pub fn barw_even(sigma: f64, lambda: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction::single(1, 3, sigma, "A->3A"),
                Reaction::single(2, 0, lambda, "2A->0"),
            ],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "BARW-even (PC)".into(),
        }
    }

    /// PCPD: 2A→3A, 2A→∅.
    pub fn pcpd(sigma: f64, lambda: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction::single(2, 3, sigma, "2A->3A"),
                Reaction::single(2, 0, lambda, "2A->0"),
            ],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "PCPD".into(),
        }
    }

    /// Schlögl II: 2A→3A, A→∅ (DP class).
    pub fn schlogl_second(sigma: f64, mu: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction::single(2, 3, sigma, "2A->3A"),
                Reaction::single(1, 0, mu, "A->0"),
            ],
            n_species: 1,
            species_names: vec!["A".into()],
            diffusion_rates: vec![1.0],
            name: "Schlogl II".into(),
        }
    }

    /// Michaelis-Menten: E+S→ES, ES→E+S, ES→E+P.
    /// Species: 0=E, 1=S, 2=ES, 3=P.
    pub fn michaelis_menten(k1: f64, km1: f64, k2: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction { reactants: vec![1,1,0,0], products: vec![0,0,1,0], rate: k1, name: "E+S->ES".into() },
                Reaction { reactants: vec![0,0,1,0], products: vec![1,1,0,0], rate: km1, name: "ES->E+S".into() },
                Reaction { reactants: vec![0,0,1,0], products: vec![1,0,0,1], rate: k2, name: "ES->E+P".into() },
            ],
            n_species: 4,
            species_names: vec!["E".into(), "S".into(), "ES".into(), "P".into()],
            diffusion_rates: vec![1.0, 1.0, 0.5, 1.0],
            name: "Michaelis-Menten".into(),
        }
    }

    /// Helper: is this a single-species CRN?
    pub fn is_single_species(&self) -> bool {
        self.n_species == 1
    }

    /// Build from a list of (reactants, products, rate, name) tuples.
    pub fn custom(
        reactions: Vec<(Vec<u32>, Vec<u32>, f64, &str)>,
        species_names: Vec<&str>,
        diffusion_rates: Vec<f64>,
    ) -> Self {
        let n_species = species_names.len();
        CRN {
            reactions: reactions
                .into_iter()
                .map(|(r, p, rate, name)| {
                    assert_eq!(r.len(), n_species);
                    assert_eq!(p.len(), n_species);
                    Reaction { reactants: r, products: p, rate, name: name.into() }
                })
                .collect(),
            n_species,
            species_names: species_names.into_iter().map(|s| s.into()).collect(),
            diffusion_rates,
            name: "Custom CRN".into(),
        }
    }
}
