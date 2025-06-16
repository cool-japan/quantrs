//! Samplers for solving QUBO/HOBO problems.
//!
//! This module provides various samplers (solvers) for QUBO and HOBO
//! problems, including simulated annealing, genetic algorithms, and
//! specialized hardware samplers.

pub mod errors;
pub mod genetic_algorithm;
pub mod gpu;
pub mod hardware;
pub mod simulated_annealing;

use ndarray::Array;
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
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
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
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
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
    let energy = 0.0;

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
pub use simulated_annealing::SASampler;
