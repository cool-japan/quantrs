//! D-Wave Quantum Annealer Sampler Implementation

use ndarray::{Array, Ix2};
use std::collections::HashMap;

use quantrs2_anneal::QuboModel;

use super::super::{SampleResult, Sampler, SamplerError, SamplerResult};

/// D-Wave Quantum Annealer Sampler
///
/// This sampler connects to D-Wave's quantum annealing hardware
/// to solve QUBO problems. It requires an API key and Internet access.
pub struct DWaveSampler {
    /// D-Wave API key
    #[allow(dead_code)]
    api_key: String,
}

impl DWaveSampler {
    /// Create a new D-Wave sampler
    ///
    /// # Arguments
    ///
    /// * `api_key` - The D-Wave API key
    #[must_use]
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }
}

impl Sampler for DWaveSampler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Extract matrix and variable mapping
        let (matrix, var_map) = qubo;

        // Get the problem dimension
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Convert ndarray to a QuboModel
        let mut qubo_model = QuboModel::new(n_vars);

        // Set linear and quadratic terms
        for i in 0..n_vars {
            if matrix[[i, i]] != 0.0 {
                qubo_model.set_linear(i, matrix[[i, i]])?;
            }

            for j in (i + 1)..n_vars {
                if matrix[[i, j]] != 0.0 {
                    qubo_model.set_quadratic(i, j, matrix[[i, j]])?;
                }
            }
        }

        // Initialize the D-Wave client
        #[cfg(feature = "dwave")]
        {
            use quantrs2_anneal::dwave::{DWaveClient, DWaveParams};

            // Create D-Wave client
            let dwave_client = DWaveClient::new(&self.api_key)?;

            // Configure submission parameters
            let mut params = DWaveParams::default();
            params.num_reads = shots;

            // Submit the problem to D-Wave
            let dwave_result = dwave_client.solve_qubo(&qubo_model, params)?;

            // Convert to our result format
            let mut results = Vec::new();

            // Process each solution
            for solution in dwave_result.solutions {
                // Convert binary array to HashMap
                let assignments: HashMap<String, bool> = solution
                    .binary_vars
                    .iter()
                    .enumerate()
                    .map(|(idx, &value)| {
                        let var_name = idx_to_var.get(&idx).unwrap().clone();
                        (var_name, value)
                    })
                    .collect();

                // Create a result
                let result = SampleResult {
                    assignments,
                    energy: solution.energy,
                    occurrences: solution.occurrences,
                };

                results.push(result);
            }

            Ok(results)
        }

        #[cfg(not(feature = "dwave"))]
        {
            Err(SamplerError::DWaveUnavailable(
                "D-Wave support not enabled. Rebuild with '--features dwave'".to_string(),
            ))
        }
    }

    fn run_hobo(
        &self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // For HOBO problems, we need to first convert to QUBO if possible
        if hobo.0.ndim() <= 2 {
            // If it's already 2D, just forward to run_qubo
            let qubo = (
                hobo.0.clone().into_dimensionality::<Ix2>().unwrap(),
                hobo.1.clone(),
            );
            self.run_qubo(&qubo, shots)
        } else {
            // D-Wave doesn't directly support higher-order problems
            // We could implement automatic quadratization here, but for now return an error
            Err(SamplerError::InvalidParameter(
                "D-Wave doesn't support HOBO problems directly. Use a quadratization technique first.".to_string()
            ))
        }
    }
}
