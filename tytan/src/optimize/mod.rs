//! Optimization utilities for QUBO/HOBO problems.
//!
//! This module provides optimization utilities and algorithms for
//! solving QUBO and HOBO problems, with optional SciRS2 integration.

use scirs2_core::ndarray::{Array, ArrayD, Dimension, Ix2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{SeedableRng, StdRng};
use std::collections::HashMap;

use crate::sampler::SampleResult;

#[cfg(feature = "scirs")]
use crate::scirs_stub;

/// Enhanced QUBO optimization using SciRS2's parallel execution primitives
///
/// This performs genuine multi-start simulated annealing: the same
/// Metropolis-criterion local search as the non-`advanced_optimization`
/// fallback below, but run as several independent, differently-seeded
/// restarts in parallel via `scirs2_core::parallel_ops`. Running more
/// independent restarts explores more of the search space and is a real,
/// honest enhancement over a single-threaded run — unlike the previous
/// implementation, which despite its "Enhanced ... advanced techniques from
/// SciRS2" doc comment only ever performed uniform random sampling with no
/// annealing at all (worse than the plain fallback below for the same
/// wall-clock budget).
#[cfg(feature = "advanced_optimization")]
pub fn optimize_qubo(
    matrix: &Array<f64, Ix2>,
    var_map: &HashMap<String, usize>,
    initial_guess: Option<Vec<bool>>,
    max_iterations: usize,
) -> Vec<SampleResult> {
    use scirs2_core::parallel_ops::*;

    let n_vars = var_map.len();
    if n_vars == 0 {
        return Vec::new();
    }

    let idx_to_var: HashMap<usize, String> = var_map
        .iter()
        .map(|(var, &idx)| (idx, var.clone()))
        .collect();

    let num_runs = scirs2_core::parallel_ops::current_num_threads().max(4);
    let sweeps = max_iterations.max(1);

    let seeds: Vec<u64> = {
        let mut seeder = thread_rng();
        (0..num_runs)
            .map(|i| seeder.random::<u64>().wrapping_add(i as u64))
            .collect()
    };

    let mut runs: Vec<(Vec<bool>, f64)> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);

            let mut solution: Vec<bool> = initial_guess
                .clone()
                .unwrap_or_else(|| (0..n_vars).map(|_| rng.random_bool()).collect());
            let mut energy = calculate_energy(&solution, matrix);
            let mut best_solution = solution.clone();
            let mut best_energy = energy;

            let mut temperature = 10.0_f64;
            let cooling_rate = 0.99_f64;

            for _ in 0..sweeps {
                let flip_idx = rng.random_range(0..n_vars);
                solution[flip_idx] = !solution[flip_idx];
                let new_energy = calculate_energy(&solution, matrix);

                let accept = new_energy < energy || {
                    let p = ((energy - new_energy) / temperature).exp();
                    rng.random::<f64>() < p
                };

                if accept {
                    energy = new_energy;
                    if energy < best_energy {
                        best_energy = energy;
                        best_solution = solution.clone();
                    }
                } else {
                    solution[flip_idx] = !solution[flip_idx];
                }

                temperature *= cooling_rate;
            }

            (best_solution, best_energy)
        })
        .collect();

    // Sort by energy, deduplicate identical solutions across restarts, and
    // return the best (up to 10) distinct solutions found.
    runs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    runs.dedup_by(|a, b| a.0 == b.0);
    runs.truncate(10);

    runs.into_iter()
        .map(|(solution, energy)| {
            let assignments: HashMap<String, bool> = solution
                .iter()
                .enumerate()
                .filter_map(|(idx, &value)| {
                    idx_to_var
                        .get(&idx)
                        .map(|var_name| (var_name.clone(), value))
                })
                .collect();

            SampleResult {
                assignments,
                energy,
                occurrences: 1,
            }
        })
        .collect()
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
    let mut solution: Vec<bool> = if let Some(guess) = initial_guess {
        guess
    } else {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();
        (0..n_vars).map(|_| rng.random_bool(0.5)).collect()
    };

    // Calculate initial energy
    let mut energy = calculate_energy(&solution, matrix);

    // Basic simulated annealing parameters
    let mut temperature = 10.0;
    let cooling_rate = 0.99;

    // Simulated annealing loop
    let mut rng = thread_rng();

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

        if accept {
            energy = new_energy;
        } else {
            // Undo the flip if not accepted
            solution[flip_idx] = !solution[flip_idx];
        }

        // Cool down
        temperature *= cooling_rate;
    }

    // Convert to SampleResult
    let assignments: HashMap<String, bool> = solution
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| {
            idx_to_var
                .get(&idx)
                .map(|var_name| (var_name.clone(), value))
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

/// Compute HOBO energy for an arbitrary-rank tensor and a binary assignment.
///
/// `E = Σ_{i₁,…,iₐ} T[i₁,…,iₐ] · x[i₁] · … · x[iₐ]`
///
/// Indices that exceed the length of `state` are treated as `false`.
fn hobo_energy(tensor: &ArrayD<f64>, state: &[bool]) -> f64 {
    let mut energy = 0.0_f64;
    for (idx, &coeff) in tensor.indexed_iter() {
        if coeff.abs() < 1e-14 {
            continue;
        }
        // All indices in this multi-index entry must refer to `true` variables.
        let all_active = idx
            .slice()
            .iter()
            .all(|&i| state.get(i).copied().unwrap_or(false));
        if all_active {
            energy += coeff;
        }
    }
    energy
}

/// Basic HOBO optimization using simulated annealing.
///
/// This performs sweep-based SA over all variables, accepting up-hill moves
/// probabilistically (Metropolis criterion).  The best configuration seen
/// across all sweeps is returned.
fn optimize_hobo_basic(
    tensor: &ArrayD<f64>,
    var_map: &HashMap<String, usize>,
    initial_guess: Option<Vec<bool>>,
    max_iterations: usize,
) -> Vec<SampleResult> {
    let n = var_map.len();
    if n == 0 {
        return vec![];
    }

    let mut rng = StdRng::seed_from_u64(42);

    // Initialise the binary state vector (indexed by var_map position).
    let mut state: Vec<bool> = match initial_guess {
        Some(guess) if guess.len() == n => guess,
        _ => (0..n).map(|_| rng.random_bool()).collect(),
    };

    let mut energy = hobo_energy(tensor, &state);
    let mut best_state = state.clone();
    let mut best_energy = energy;

    // Simulated-annealing schedule: start at a temperature proportional to
    // the largest coefficient magnitude so that the initial acceptance ratio
    // is reasonable, and cool geometrically.
    let max_coeff = tensor
        .iter()
        .fold(0.0_f64, |m, &v| if v.abs() > m { v.abs() } else { m });
    let initial_temperature = if max_coeff > 0.0 { max_coeff } else { 1.0 };
    let cooling_rate = 0.99_f64;
    let mut temperature = initial_temperature;

    let num_sweeps = max_iterations.max(1);
    for _sweep in 0..num_sweeps {
        // One sweep: try flipping each variable once in order.
        for i in 0..n {
            state[i] = !state[i];
            let new_energy = hobo_energy(tensor, &state);
            let delta = new_energy - energy;

            // Accept if the move decreases energy, or with Boltzmann probability.
            let accept = delta < 0.0
                || (temperature > 1e-14 && rng.random::<f64>() < (-delta / temperature).exp());

            if accept {
                energy = new_energy;
                if energy < best_energy {
                    best_energy = energy;
                    best_state = state.clone();
                }
            } else {
                // Revert the flip.
                state[i] = !state[i];
            }
        }
        temperature *= cooling_rate;
    }

    // Build the assignment map from the best state found.
    let assignments: HashMap<String, bool> = var_map
        .iter()
        .filter_map(|(name, &idx)| best_state.get(idx).map(|&v| (name.clone(), v)))
        .collect();

    vec![SampleResult {
        assignments,
        energy: best_energy,
        occurrences: 1,
    }]
}

#[cfg(all(test, feature = "advanced_optimization"))]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_optimize_qubo_advanced_finds_real_optimum() {
        // Minimizing QUBO: x_i = 1 is favorable for every i (diagonal = -1,
        // no interaction terms), so the true optimum is all-ones with
        // energy = -n. The old fabricated implementation only ever
        // performed uniform random sampling with no annealing, so it had no
        // reliable way to consistently land on the true optimum.
        let n = 6;
        let matrix = Array2::from_shape_fn((n, n), |(i, j)| if i == j { -1.0 } else { 0.0 });

        let mut var_map = HashMap::new();
        for i in 0..n {
            var_map.insert(format!("x{i}"), i);
        }

        let results = optimize_qubo(&matrix, &var_map, None, 200);
        assert!(!results.is_empty());

        let best = results
            .iter()
            .fold(f64::INFINITY, |acc, r| acc.min(r.energy));
        assert!(
            (best - (-(n as f64))).abs() < 1e-9,
            "expected best energy {}, got {best}",
            -(n as f64)
        );

        // Results must actually be sorted best-first and distinct.
        for pair in results.windows(2) {
            assert!(pair[0].energy <= pair[1].energy);
        }
    }
}
