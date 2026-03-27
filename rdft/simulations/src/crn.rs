/// Chemical Reaction Network specification for single-species systems.
///
/// Each reaction is kA -> lA with a rate constant.  The simulator
/// applies reactions via tau-leaping: at each site with n particles,
/// the number of firings is sampled from the appropriate distribution.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reaction {
    /// Number of reactant particles consumed per firing
    pub k: u32,
    /// Number of product particles created per firing
    pub l: u32,
    /// Rate constant (probability per particle per dt for k=1,
    /// or per pair per dt for k=2)
    pub rate: f64,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CRN {
    pub reactions: Vec<Reaction>,
    /// Diffusion rate (probability of hopping per dt; 1.0 = always hop)
    pub diffusion_rate: f64,
    pub name: String,
}

impl CRN {
    /// Gribov process (BRW): A->2A (beta), A->0 (epsilon), 2A->A (chi).
    /// At criticality: beta = epsilon.
    pub fn gribov(beta: f64, epsilon: f64, chi: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction { k: 1, l: 2, rate: beta,   name: "A->2A".into() },
                Reaction { k: 1, l: 0, rate: epsilon, name: "A->0".into() },
                Reaction { k: 2, l: 1, rate: chi,     name: "2A->A".into() },
            ],
            diffusion_rate: 1.0,
            name: "Gribov (BRW)".into(),
        }
    }

    /// Critical BRW with default parameters.
    pub fn gribov_critical() -> Self {
        Self::gribov(0.3, 0.1, 0.05)
    }

    /// BRW with instant coalescence (max 1 particle per site).
    /// Each particle branches with probability p_branch per step.
    /// Coalescence is handled by the engine's max_per_site=1.
    /// This is the standard model for Bordeu+ (2019) scaling.
    pub fn brw_coalescent() -> Self {
        CRN {
            reactions: vec![
                Reaction { k: 1, l: 2, rate: 0.5, name: "A->2A".into() },
            ],
            diffusion_rate: 1.0,
            name: "BRW (coalescent)".into(),
        }
    }

    /// Pair annihilation: 2A -> 0.
    pub fn pair_annihilation(rate: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction { k: 2, l: 0, rate, name: "2A->0".into() },
            ],
            diffusion_rate: 1.0,
            name: "Pair annihilation".into(),
        }
    }

    /// Coagulation: 2A -> A.
    pub fn coagulation(rate: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction { k: 2, l: 1, rate, name: "2A->A".into() },
            ],
            diffusion_rate: 1.0,
            name: "Coagulation".into(),
        }
    }

    /// Birth-death: A->2A, A->0.
    pub fn birth_death(birth: f64, death: f64) -> Self {
        CRN {
            reactions: vec![
                Reaction { k: 1, l: 2, rate: birth, name: "A->2A".into() },
                Reaction { k: 1, l: 0, rate: death, name: "A->0".into() },
            ],
            diffusion_rate: 1.0,
            name: "Birth-death".into(),
        }
    }

    /// Build from a list of (k, l, rate, name) tuples.
    pub fn custom(reactions: Vec<(u32, u32, f64, &str)>, diffusion_rate: f64) -> Self {
        CRN {
            reactions: reactions
                .into_iter()
                .map(|(k, l, rate, name)| Reaction {
                    k, l, rate, name: name.into(),
                })
                .collect(),
            diffusion_rate,
            name: "Custom CRN".into(),
        }
    }
}
