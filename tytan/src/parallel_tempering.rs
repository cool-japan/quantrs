//! Parallel tempering implementation for enhanced sampling.
//!
//! This module provides parallel tempering algorithms for better
//! exploration of the solution space in quantum annealing problems.

use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
use ndarray::{Array, Ix2, IxDyn};
use rand::prelude::*;
use std::collections::HashMap;

/// Parallel tempering sampler that runs multiple chains at different temperatures
pub struct ParallelTemperingSampler {
    /// Number of parallel chains
    num_chains: usize,
    /// Temperature schedule for the chains
    temperatures: Vec<f64>,
    /// Number of sweeps per chain
    sweeps: usize,
    /// Random number generator
    rng: StdRng,
}

impl ParallelTemperingSampler {
    /// Create a new parallel tempering sampler
    pub fn new(num_chains: Option<usize>, sweeps: Option<usize>) -> Self {
        let num_chains = num_chains.unwrap_or(8);
        let sweeps = sweeps.unwrap_or(1000);

        // Create geometric temperature schedule
        let temperatures = (0..num_chains)
            .map(|i| 1.0 * (10.0_f64).powf(i as f64 / (num_chains - 1) as f64))
            .collect();

        Self {
            num_chains,
            temperatures,
            sweeps,
            rng: StdRng::from_seed([42; 32]),
        }
    }

    /// Set custom temperature schedule
    pub fn with_temperatures(mut self, temperatures: Vec<f64>) -> Self {
        self.num_chains = temperatures.len();
        self.temperatures = temperatures;
        self
    }

    /// Run parallel tempering on a QUBO problem
    fn run_parallel_tempering(
        &mut self,
        matrix: &Array<f64, Ix2>,
        var_map: &HashMap<String, usize>,
        num_reads: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        let n = matrix.nrows();
        let mut best_solutions = Vec::new();

        for _ in 0..num_reads {
            // Initialize chains with random states
            let mut chains: Vec<Vec<i32>> = (0..self.num_chains)
                .map(|_| {
                    (0..n)
                        .map(|_| if self.rng.gen::<f64>() < 0.5 { 0 } else { 1 })
                        .collect()
                })
                .collect();

            // Run parallel tempering
            for _ in 0..self.sweeps {
                // Update each chain
                for (chain_idx, chain) in chains.iter_mut().enumerate() {
                    let temperature = self.temperatures[chain_idx];
                    self.metropolis_update(chain, matrix, temperature);
                }

                // Attempt replica exchanges
                for i in 0..(self.num_chains - 1) {
                    if self.rng.gen::<f64>() < 0.1 {
                        // 10% exchange probability
                        let energy_i = self.calculate_energy(&chains[i], matrix);
                        let energy_j = self.calculate_energy(&chains[i + 1], matrix);

                        let delta = (energy_j - energy_i)
                            * (1.0 / self.temperatures[i] - 1.0 / self.temperatures[i + 1]);

                        if delta <= 0.0 || self.rng.gen::<f64>() < (-delta).exp() {
                            chains.swap(i, i + 1);
                        }
                    }
                }
            }

            // Extract best solution (from lowest temperature chain)
            let best_chain = &chains[0];
            let energy = self.calculate_energy(best_chain, matrix);

            let mut assignments = HashMap::new();
            for (var_name, &idx) in var_map {
                assignments.insert(var_name.clone(), best_chain[idx] == 1);
            }

            best_solutions.push(SampleResult {
                assignments,
                energy,
                occurrences: 1,
            });
        }

        Ok(best_solutions)
    }

    /// Perform a Metropolis update on a chain
    fn metropolis_update(
        &mut self,
        chain: &mut Vec<i32>,
        matrix: &Array<f64, Ix2>,
        temperature: f64,
    ) {
        let n = chain.len();

        for _ in 0..n {
            let idx = self.rng.gen_range(0..n);
            let old_value = chain[idx];
            let new_value = 1 - old_value;

            let old_energy = self.calculate_energy(chain, matrix);
            chain[idx] = new_value;
            let new_energy = self.calculate_energy(chain, matrix);
            chain[idx] = old_value;

            let delta_energy = new_energy - old_energy;

            if delta_energy <= 0.0 || self.rng.gen::<f64>() < (-delta_energy / temperature).exp() {
                chain[idx] = new_value;
            }
        }
    }

    /// Calculate energy of a configuration
    fn calculate_energy(&self, config: &[i32], matrix: &Array<f64, Ix2>) -> f64 {
        let mut energy = 0.0;
        let n = config.len();

        for i in 0..n {
            for j in 0..n {
                energy += matrix[[i, j]] * config[i] as f64 * config[j] as f64;
            }
        }

        energy
    }
}

impl Sampler for ParallelTemperingSampler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        num_reads: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        let (matrix, var_map) = qubo;
        let mut sampler = Self {
            num_chains: self.num_chains,
            temperatures: self.temperatures.clone(),
            sweeps: self.sweeps,
            rng: StdRng::from_seed([42; 32]),
        };
        sampler.run_parallel_tempering(matrix, var_map, num_reads)
    }

    fn run_hobo(
        &self,
        hobo: &(Array<f64, IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Convert dynamic array to 2D for QUBO processing
        let (matrix_dyn, var_map) = hobo;

        if matrix_dyn.ndim() != 2 {
            return Err(SamplerError::InvalidParameter(
                "HOBO matrix must be 2D for parallel tempering".into(),
            ));
        }

        let matrix_2d = matrix_dyn
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| SamplerError::InvalidParameter("Failed to convert matrix to 2D".into()))?;

        let mut sampler = Self {
            num_chains: self.num_chains,
            temperatures: self.temperatures.clone(),
            sweeps: self.sweeps,
            rng: StdRng::from_seed([42; 32]),
        };
        sampler.run_parallel_tempering(&matrix_2d, var_map, shots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_parallel_tempering_basic() {
        let mut matrix = Array2::<f64>::zeros((2, 2));
        matrix[[0, 0]] = -1.0;
        matrix[[1, 1]] = -1.0;
        matrix[[0, 1]] = 2.0;
        matrix[[1, 0]] = 2.0;

        let mut var_map = HashMap::new();
        var_map.insert("x".to_string(), 0);
        var_map.insert("y".to_string(), 1);

        let sampler = ParallelTemperingSampler::new(Some(4), Some(100));
        let result = sampler.run_qubo(&(matrix, var_map), 10);

        assert!(result.is_ok());
        let solutions = result.unwrap();
        assert_eq!(solutions.len(), 10);
    }
}
