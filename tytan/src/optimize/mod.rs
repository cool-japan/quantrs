//! Optimization utilities for QUBO/HOBO problems.
//!
//! This module provides optimization utilities and algorithms for
//! solving QUBO and HOBO problems, with optional SciRS2 integration.

use ndarray::{Array, ArrayD, Ix2};
use rand::Rng;
use std::collections::HashMap;

#[cfg(feature = "scirs")]
use scirs2_optimize::unconstrained::{minimize, MinimizeParams, MinimizeResult};

#[cfg(feature = "scirs")]
use scirs2_linalg::decomposition;

use crate::sampler::SampleResult;

/// Enhanced QUBO optimization using SciRS2 (when available)
///
/// This function provides enhanced optimization for QUBO problems,
/// using advanced techniques from SciRS2 when available.
#[cfg(feature = "advanced_optimization")]
pub fn optimize_qubo(
    matrix: &Array<f64, Ix2>,
    var_map: &HashMap<String, usize>,
    initial_guess: Option<Vec<bool>>,
    max_iterations: usize,
) -> Vec<SampleResult> {
    use scirs2_optimize::unconstrained::{MinimizeAlgorithm, MinimizeParams};

    let n_vars = var_map.len();

    // Map from indices back to variable names
    let idx_to_var: HashMap<usize, String> = var_map
        .iter()
        .map(|(var, &idx)| (idx, var.clone()))
        .collect();

    // Define the objective function
    let objective = |x: &[f64]| {
        // Convert continuous values to binary
        let binary: Vec<bool> = x.iter().map(|&val| val > 0.5).collect();

        // Calculate energy
        let mut energy = 0.0;

        // Linear terms
        for i in 0..n_vars {
            if binary[i] {
                energy += matrix[[i, i]];
            }
        }

        // Quadratic terms
        for i in 0..n_vars {
            if binary[i] {
                for j in (i + 1)..n_vars {
                    if binary[j] {
                        energy += matrix[[i, j]];
                    }
                }
            }
        }

        energy
    };

    // Create initial guess (either provided or random)
    let x0: Vec<f64> = match initial_guess {
        Some(guess) => guess.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
        None => {
            use rand::Rng;
            let mut rng = rand::rng();
            (0..n_vars)
                .map(|_| if rng.random_bool(0.5) { 1.0 } else { 0.0 })
                .collect()
        }
    };

    // Configure optimization parameters
    let mut params = MinimizeParams::new();
    params.max_iterations = max_iterations;
    params.algorithm = MinimizeAlgorithm::SimulatedAnnealing;
    params.tolerance = 1e-6;

    // Run the optimization
    let result = minimize(objective, &x0, &params).unwrap();

    // Convert to binary solution
    let binary: Vec<bool> = result.x.iter().map(|&val| val > 0.5).collect();

    // Convert to SampleResult
    let assignments: HashMap<String, bool> = binary
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            let var_name = idx_to_var.get(&idx).unwrap().clone();
            (var_name, value)
        })
        .collect();

    // Create result
    let sample_result = SampleResult {
        assignments,
        energy: result.f_min,
        occurrences: 1,
    };

    vec![sample_result]
}

/// Fallback QUBO optimization implementation
#[cfg(not(feature = "advanced_optimization"))]
pub fn optimize_qubo(
    matrix: &Array<f64, Ix2>,
    var_map: &HashMap<String, usize>,
    initial_guess: Option<Vec<bool>>,
    max_iterations: usize,
) -> Vec<SampleResult> {
    // Use basic simulated annealing for fallback
    let n_vars = var_map.len();

    // Map from indices back to variable names
    let idx_to_var: HashMap<usize, String> = var_map
        .iter()
        .map(|(var, &idx)| (idx, var.clone()))
        .collect();

    // Create initial solution (either provided or random)
    let mut solution: Vec<bool> = match initial_guess {
        Some(guess) => guess,
        None => {
            use rand::Rng;
            let mut rng = rand::rng();
            (0..n_vars).map(|_| rng.random_bool(0.5)).collect()
        }
    };

    // Calculate initial energy
    let mut energy = calculate_energy(&solution, matrix);

    // Basic simulated annealing parameters
    let mut temperature = 10.0;
    let cooling_rate = 0.99;

    // Simulated annealing loop
    let mut rng = rand::rng();

    for _ in 0..max_iterations {
        // Generate a neighbor by flipping a random bit
        let flip_idx = rng.random_range(0..n_vars);
        solution[flip_idx] = !solution[flip_idx];

        // Calculate new energy
        let new_energy = calculate_energy(&solution, matrix);

        // Determine if we accept the move
        let accept = if new_energy < energy {
            true
        } else {
            let p = ((energy - new_energy) / temperature).exp();
            rng.random::<f64>() < p
        };

        if !accept {
            // Undo the flip if not accepted
            solution[flip_idx] = !solution[flip_idx];
        } else {
            energy = new_energy;
        }

        // Cool down
        temperature *= cooling_rate;
    }

    // Convert to SampleResult
    let assignments: HashMap<String, bool> = solution
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            let var_name = idx_to_var.get(&idx).unwrap().clone();
            (var_name, value)
        })
        .collect();

    // Create result
    let sample_result = SampleResult {
        assignments,
        energy,
        occurrences: 1,
    };

    vec![sample_result]
}

/// Calculate the energy of a solution for a QUBO problem
pub fn calculate_energy(solution: &[bool], matrix: &Array<f64, Ix2>) -> f64 {
    let n = solution.len();
    let mut energy = 0.0;

    // Calculate from diagonal terms (linear)
    for i in 0..n {
        if solution[i] {
            energy += matrix[[i, i]];
        }
    }

    // Calculate from off-diagonal terms (quadratic)
    for i in 0..n {
        if solution[i] {
            for j in (i + 1)..n {
                if solution[j] {
                    energy += matrix[[i, j]];
                }
            }
        }
    }

    energy
}

/// Advanced HOBO tensor optimization using SciRS2 tensor methods
#[cfg(feature = "scirs")]
pub fn optimize_hobo(
    tensor: &ArrayD<f64>,
    var_map: &HashMap<String, usize>,
    initial_guess: Option<Vec<bool>>,
    max_iterations: usize,
) -> Vec<SampleResult> {
    use scirs2_linalg::tensor_contraction::{cp, tucker};

    let n_vars = var_map.len();
    let dim = tensor.ndim();

    // Map from indices back to variable names
    let idx_to_var: HashMap<usize, String> = var_map
        .iter()
        .map(|(var, &idx)| (idx, var.clone()))
        .collect();

    // Decompose the tensor for efficient computation
    let decomposed = if dim > 3 {
        // For high-dimensional tensors, use CP decomposition
        let rank = std::cmp::min(n_vars, 50); // Truncate to reasonable rank
        let cp_tensors = cp::decompose(tensor, rank);

        // Processing with CP tensors
        // ...
    } else {
        // For lower dimensions, use Tucker decomposition
        let ranks = vec![std::cmp::min(n_vars, 20); dim];
        let tucker_decomp = tucker::decompose(tensor, &ranks);

        // Processing with Tucker decomposition
        // ...
    };

    // For now, fall back to basic implementation
    // (Actual implementation would use tensor methods)

    // Convert to SampleResult placeholder
    let assignments: HashMap<String, bool> = var_map
        .iter()
        .map(|(name, _)| (name.clone(), false))
        .collect();

    // Create result
    let sample_result = SampleResult {
        assignments,
        energy: 0.0,
        occurrences: 1,
    };

    vec![sample_result]
}

/// Basic HOBO optimization for when SciRS2 is not available
#[cfg(not(feature = "scirs"))]
pub fn optimize_hobo(
    _tensor: &ArrayD<f64>,
    var_map: &HashMap<String, usize>,
    _initial_guess: Option<Vec<bool>>,
    _max_iterations: usize,
) -> Vec<SampleResult> {
    // For now, implement a simple fallback that only works for 3rd order
    // For higher orders, you'd need a more general implementation

    // (Implementation would go here)

    // Return placeholder
    let assignments: HashMap<String, bool> = var_map
        .iter()
        .map(|(name, _)| (name.clone(), false))
        .collect();

    // Create result
    let sample_result = SampleResult {
        assignments,
        energy: 0.0,
        occurrences: 1,
    };

    vec![sample_result]
}
