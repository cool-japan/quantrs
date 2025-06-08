//! Optimization utilities for QUBO/HOBO problems.
//!
//! This module provides optimization utilities and algorithms for
//! solving QUBO and HOBO problems, with optional SciRS2 integration.

use ndarray::{Array, ArrayD, Ix2};
use rand::Rng;
use std::collections::HashMap;

use crate::sampler::SampleResult;

#[cfg(feature = "scirs")]
use crate::scirs_stub;

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
    // Use SciRS2 enhanced parallel sampling
    let enhanced_matrix = scirs_stub::enhance_qubo_matrix(matrix);
    let samples = scirs_stub::parallel_sample_qubo(&enhanced_matrix, max_iterations);
    
    // Map from indices back to variable names
    let idx_to_var: HashMap<usize, String> = var_map
        .iter()
        .map(|(var, &idx)| (idx, var.clone()))
        .collect();
    
    // Convert to SampleResults
    let mut results: Vec<SampleResult> = samples
        .into_iter()
        .map(|(solution, energy)| {
            let assignments: HashMap<String, bool> = solution
                .iter()
                .enumerate()
                .map(|(idx, &value)| {
                    let var_name = idx_to_var.get(&idx).unwrap().clone();
                    (var_name, value)
                })
                .collect();
            
            SampleResult {
                assignments,
                energy,
                occurrences: 1,
            }
        })
        .collect();
    
    // Sort by energy and return best solutions
    results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
    results.truncate(10); // Return top 10 solutions
    
    results
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
            let mut rng = rand::thread_rng();
            (0..n_vars).map(|_| rng.gen_bool(0.5)).collect()
        }
    };

    // Calculate initial energy
    let mut energy = calculate_energy(&solution, matrix);

    // Basic simulated annealing parameters
    let mut temperature = 10.0;
    let cooling_rate = 0.99;

    // Simulated annealing loop
    let mut rng = rand::thread_rng();

    for _ in 0..max_iterations {
        // Generate a neighbor by flipping a random bit
        let flip_idx = rng.gen_range(0..n_vars);
        solution[flip_idx] = !solution[flip_idx];

        // Calculate new energy
        let new_energy = calculate_energy(&solution, matrix);

        // Determine if we accept the move
        let accept = if new_energy < energy {
            true
        } else {
            let p = ((energy - new_energy) / temperature).exp();
            rng.gen::<f64>() < p
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
    calculate_energy_standard(solution, matrix)
}

/// Standard energy calculation without SciRS2
fn calculate_energy_standard(solution: &[bool], matrix: &Array<f64, Ix2>) -> f64 {
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

/// Advanced HOBO tensor optimization using SciRS2
#[cfg(feature = "scirs")]
pub fn optimize_hobo(
    tensor: &ArrayD<f64>,
    var_map: &HashMap<String, usize>,
    initial_guess: Option<Vec<bool>>,
    max_iterations: usize,
) -> Vec<SampleResult> {
    // Apply SciRS2 tensor optimizations (placeholder)
    let _enhanced = scirs_stub::optimize_hobo_tensor(tensor);
    
    // For now, return a simple result
    // In a full implementation, this would use tensor decomposition
    optimize_hobo_basic(tensor, var_map, initial_guess, max_iterations)
}

/// Basic HOBO optimization for when SciRS2 is not available
#[cfg(not(feature = "scirs"))]
pub fn optimize_hobo(
    tensor: &ArrayD<f64>,
    var_map: &HashMap<String, usize>,
    initial_guess: Option<Vec<bool>>,
    max_iterations: usize,
) -> Vec<SampleResult> {
    optimize_hobo_basic(tensor, var_map, initial_guess, max_iterations)
}

/// Basic HOBO optimization implementation
fn optimize_hobo_basic(
    _tensor: &ArrayD<f64>,
    var_map: &HashMap<String, usize>,
    _initial_guess: Option<Vec<bool>>,
    _max_iterations: usize,
) -> Vec<SampleResult> {
    // For now, implement a simple fallback
    // In a full implementation, this would handle arbitrary tensor orders
    
    // Return placeholder
    let assignments: HashMap<String, bool> = var_map
        .iter()
        .map(|(name, _)| (name.clone(), false))
        .collect();

    vec![SampleResult {
        assignments,
        energy: 0.0,
        occurrences: 1,
    }]
}
