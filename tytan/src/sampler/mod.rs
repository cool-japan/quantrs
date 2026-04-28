//! Samplers for solving QUBO/HOBO problems.
//!
//! QuantRS2 Tytan provides five pure-Rust metaheuristic samplers for solving
//! QUBO (Quadratic Unconstrained Binary Optimization) and HOBO (Higher-Order
//! Binary Optimization) problems. All samplers implement the [`Sampler`] trait,
//! which exposes a uniform `run_qubo` / `run_hobo` interface.
//!
//! The [`energy`] submodule provides SIMD-accelerated QUBO energy evaluation
//! functions that serve as the shared inner loop for all samplers.
//!
//! # Sampler Comparison
//!
//! | Sampler | Type | Strengths | Typical problem size | Citation |
//! |---------|------|-----------|----------------------|----------|
//! | [`SASampler`] | Local search | Versatile, easy to tune | Small–large | Kirkpatrick 1983 |
//! | [`GASampler`] | Evolutionary | Population diversity | Medium | Holland 1975 |
//! | [`TabuSampler`] | Tabu search | Structured combinatorial | Small–medium | Glover 1989 |
//! | [`SBSampler`] | Bifurcation dynamics | Large dense QUBO | Large | Goto 2019 |
//! | [`PopulationAnnealingSampler`] | Population annealing | Ensemble, near-ground-state | Medium–large | Hukushima 2003 |
//!
//! # Choosing a Sampler
//!
//! - **Start with [`SASampler`]** for most problems. It is robust, has sensible
//!   defaults, and handles both small and large instances.
//! - **Use [`TabuSampler`]** for scheduling, routing, or other problems that have
//!   a structured neighbourhood where recently-visited moves should be forbidden.
//! - **Use [`SBSampler`]** for large, dense QUBO matrices (n ≥ 200) where SA is
//!   too slow. The ballistic (`SBVariant::Ballistic`) variant converges faster;
//!   discrete (`SBVariant::Discrete`) gives crisper spin assignments.
//! - **Use [`PopulationAnnealingSampler`]** when you need an ensemble of
//!   near-ground-state solutions or when the problem has a complex low-energy
//!   landscape and you want thermodynamic statistics.
//! - **Use [`GASampler`]** when the problem structure maps naturally to bitstring
//!   chromosomes and crossover is likely to be productive (e.g., scheduling
//!   problems with independent sub-problems).
//!
//! # Quick Example
//!
//! The following example solves a 3-variable Max-Cut QUBO (K3 graph) with the
//! Simulated Annealing sampler:
//!
//! ```
//! use quantrs2_tytan::sampler::{SASampler, Sampler};
//! use scirs2_core::ndarray::Array;
//! use std::collections::HashMap;
//!
//! // K3 Max-Cut: minimise -x0 - x1 - x2 + 2*x0*x1 + 2*x0*x2 + 2*x1*x2
//! let mut q = Array::<f64, _>::zeros((3, 3));
//! q[[0, 0]] = -1.0;
//! q[[1, 1]] = -1.0;
//! q[[2, 2]] = -1.0;
//! q[[0, 1]] = 2.0;
//! q[[0, 2]] = 2.0;
//! q[[1, 2]] = 2.0;
//!
//! let mut var_map = HashMap::new();
//! var_map.insert("x0".to_string(), 0);
//! var_map.insert("x1".to_string(), 1);
//! var_map.insert("x2".to_string(), 2);
//!
//! let sampler = SASampler::new(Some(42));
//! let samples = sampler.run_qubo(&(q, var_map), 5).expect("SA sampler failed");
//! assert!(!samples.is_empty());
//! // Best solution is at index 0 (sorted by energy ascending)
//! println!("Best energy: {}", samples[0].energy);
//! ```

pub mod energy;
pub mod errors;
pub mod genetic_algorithm;
pub mod gpu;
pub mod hardware;
pub mod population_annealing;
pub mod simulated_annealing;
pub mod simulated_bifurcation;
pub mod tabu_search;

use scirs2_core::ndarray::Array;
use std::collections::HashMap;

pub use errors::{SamplerError, SamplerResult};

/// A sample result from a QUBO/HOBO problem
///
/// This struct represents a single sample result from a QUBO/HOBO
/// problem, including the variable assignments, energy, and occurrence count.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// The variable assignments (name -> value mapping)
    pub assignments: HashMap<String, bool>,
    /// The energy (objective function value)
    pub energy: f64,
    /// The number of times this solution appeared
    pub occurrences: usize,
}

/// Trait for samplers that can solve QUBO/HOBO problems
pub trait Sampler {
    /// Run the sampler on a QUBO problem
    ///
    /// # Arguments
    ///
    /// * `qubo` - The QUBO problem to solve: (matrix, variable mapping)
    /// * `shots` - The number of samples to take
    ///
    /// # Returns
    ///
    /// A vector of sample results, sorted by energy (best solutions first)
    fn run_qubo(
        &self,
        qubo: &(
            Array<f64, scirs2_core::ndarray::Ix2>,
            HashMap<String, usize>,
        ),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>>;

    /// Run the sampler on a HOBO problem
    ///
    /// # Arguments
    ///
    /// * `hobo` - The HOBO problem to solve: (tensor, variable mapping)
    /// * `shots` - The number of samples to take
    ///
    /// # Returns
    ///
    /// A vector of sample results, sorted by energy (best solutions first)
    fn run_hobo(
        &self,
        hobo: &(
            Array<f64, scirs2_core::ndarray::IxDyn>,
            HashMap<String, usize>,
        ),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>>;
}

/// Evaluate energy for a QUBO problem given a binary state vector
///
/// # Arguments
///
/// * `state` - Binary state vector (solution)
/// * `h_vector` - Linear coefficients (diagonal)
/// * `j_matrix` - Quadratic coefficients (n_vars x n_vars matrix in flattened form)
/// * `n_vars` - Number of variables
///
/// # Returns
///
/// The energy (objective function value) for the given state
#[allow(dead_code)]
pub(crate) fn evaluate_qubo_energy(
    state: &[bool],
    h_vector: &[f64],
    j_matrix: &[f64],
    n_vars: usize,
) -> f64 {
    let mut energy = 0.0;

    // Linear terms
    for i in 0..n_vars {
        if state[i] {
            energy += h_vector[i];
        }
    }

    // Quadratic terms
    for i in 0..n_vars {
        if state[i] {
            for j in 0..n_vars {
                if state[j] {
                    energy += j_matrix[i * n_vars + j];
                }
            }
        }
    }

    energy
}

// Re-export main sampler implementations
pub use genetic_algorithm::GASampler;
pub use gpu::ArminSampler;
pub use hardware::{DWaveSampler, MIKASAmpler};
pub use population_annealing::PopulationAnnealingSampler;
pub use simulated_annealing::SASampler;
pub use simulated_bifurcation::{SBSampler, SBVariant};
pub use tabu_search::TabuSampler;

// Re-export energy module public interface
pub use energy::{
    build_dense_q, build_dense_q_from_array, compute_influence, compute_influence_simd,
    energy_delta, energy_delta_from_array, energy_delta_simd, energy_full, energy_full_from_array,
    energy_full_simd, update_influence, update_influence_simd,
};
