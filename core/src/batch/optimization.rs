//! Batch optimization for parameterized quantum circuits using SciRS2

use super::execution::{BatchCircuit, BatchCircuitExecutor};
use super::BatchStateVector;
use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
// use scirs2_core::parallel_ops::*;
use crate::optimization_stubs::{minimize, Method, OptimizeResult, Options};
use crate::parallel_ops_stubs::*;
// use scirs2_core::optimization::{minimize, Method, OptimizeResult, Options};
use std::sync::Arc;

// Import SciRS2 optimization
// extern crate scirs2_optimize;
// use scirs2_optimize::unconstrained::{minimize, Method, OptimizeResult, Options};

/// Batch optimizer for parameterized quantum circuits
pub struct BatchParameterOptimizer {
    /// Circuit executor
    executor: BatchCircuitExecutor,
    /// Optimization configuration
    config: OptimizationConfig,
    /// Cache for gradient computations
    gradient_cache: Option<GradientCache>,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Use parallel gradient computation
    pub parallel_gradients: bool,
    /// Optimization method
    pub method: Method,
    /// Enable gradient caching
    pub enable_cache: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 0.1,
            parallel_gradients: true,
            method: Method::BFGS,
            enable_cache: true,
        }
    }
}

/// Gradient cache for repeated computations
#[derive(Debug, Clone)]
struct GradientCache {
    /// Cached gradients
    gradients: Vec<Array1<f64>>,
    /// Parameter values for cached gradients
    parameters: Vec<Vec<f64>>,
    /// Maximum cache size
    max_size: usize,
}

impl BatchParameterOptimizer {
    /// Create a new batch parameter optimizer
    pub const fn new(executor: BatchCircuitExecutor, config: OptimizationConfig) -> Self {
        let gradient_cache = if config.enable_cache {
            Some(GradientCache {
                gradients: Vec::new(),
                parameters: Vec::new(),
                max_size: 100,
            })
        } else {
            None
        };

        Self {
            executor,
            config,
            gradient_cache,
        }
    }

    /// Optimize parameters for a batch of circuits
    pub fn optimize_batch(
        &mut self,
        circuit_fn: impl Fn(&[f64]) -> QuantRS2Result<BatchCircuit> + Sync + Send + Clone + 'static,
        initial_params: &[f64],
        cost_fn: impl Fn(&BatchStateVector) -> f64 + Sync + Send + Clone + 'static,
        initial_states: &BatchStateVector,
    ) -> QuantRS2Result<OptimizeResult<f64>> {
        let _num_params = initial_params.len();

        // Define objective function
        let executor = Arc::new(self.executor.clone());
        let states = Arc::new(initial_states.clone());
        let circuit_fn = Arc::new(circuit_fn);
        let cost_fn = Arc::new(cost_fn);

        let objective = {
            let executor = executor;
            let states = states;
            let circuit_fn = circuit_fn;
            let cost_fn = cost_fn;

            move |params: &scirs2_core::ndarray::ArrayView1<f64>| -> f64 {
                let params_slice = match params.as_slice() {
                    Some(slice) => slice,
                    None => return f64::INFINITY,
                };
                let circuit = match (*circuit_fn)(params_slice) {
                    Ok(c) => c,
                    Err(_) => return f64::INFINITY,
                };

                let mut batch_copy = (*states).clone();
                match executor.execute_batch(&circuit, &mut batch_copy) {
                    Ok(_) => (*cost_fn)(&batch_copy),
                    Err(_) => f64::INFINITY,
                }
            }
        };

        // Create options
        let options = Options {
            max_iter: self.config.max_iterations,
            ftol: self.config.tolerance,
            gtol: self.config.tolerance,
            ..Default::default()
        };

        // Run optimization using SciRS2
        let initial_array = Array1::from_vec(initial_params.to_vec());
        let result = minimize(objective, &initial_array, self.config.method, Some(options));

        match result {
            Ok(opt_result) => Ok(opt_result),
            Err(e) => Err(QuantRS2Error::InvalidInput(format!(
                "Optimization failed: {e:?}"
            ))),
        }
    }

    /// Compute gradients using parameter shift rule
    pub fn compute_gradients_batch(
        &mut self,
        circuit_fn: impl Fn(&[f64]) -> QuantRS2Result<BatchCircuit> + Sync + Send,
        params: &[f64],
        cost_fn: impl Fn(&BatchStateVector) -> f64 + Sync + Send,
        initial_states: &BatchStateVector,
        shift: f64,
    ) -> QuantRS2Result<Vec<f64>> {
        // Check cache
        if let Some(cache) = &self.gradient_cache {
            for (i, cached_params) in cache.parameters.iter().enumerate() {
                if params
                    .iter()
                    .zip(cached_params)
                    .all(|(a, b)| (a - b).abs() < 1e-10)
                {
                    return Ok(cache.gradients[i].to_vec());
                }
            }
        }

        let num_params = params.len();

        if self.config.parallel_gradients {
            // Compute gradients in parallel
            // Clone executor for parallel use
            let executor = self.executor.clone();
            let gradients: Vec<f64> = (0..num_params)
                .into_par_iter()
                .map(|i| {
                    compute_single_gradient_static(
                        &executor,
                        &circuit_fn,
                        params,
                        i,
                        &cost_fn,
                        initial_states,
                        shift,
                    )
                    .unwrap_or(0.0)
                })
                .collect();

            // Update cache
            if let Some(cache) = &mut self.gradient_cache {
                if cache.gradients.len() >= cache.max_size {
                    cache.gradients.remove(0);
                    cache.parameters.remove(0);
                }
                cache.gradients.push(Array1::from_vec(gradients.clone()));
                cache.parameters.push(params.to_vec());
            }

            Ok(gradients)
        } else {
            // Sequential gradient computation
            let mut gradients = vec![0.0; num_params];

            for i in 0..num_params {
                gradients[i] = self.compute_single_gradient(
                    &circuit_fn,
                    params,
                    i,
                    &cost_fn,
                    initial_states,
                    shift,
                )?;
            }

            Ok(gradients)
        }
    }

    /// Compute gradient for a single parameter
    fn compute_single_gradient(
        &self,
        circuit_fn: impl Fn(&[f64]) -> QuantRS2Result<BatchCircuit>,
        params: &[f64],
        param_idx: usize,
        cost_fn: impl Fn(&BatchStateVector) -> f64,
        initial_states: &BatchStateVector,
        shift: f64,
    ) -> QuantRS2Result<f64> {
        compute_single_gradient_static(
            &self.executor,
            &circuit_fn,
            params,
            param_idx,
            &cost_fn,
            initial_states,
            shift,
        )
    }

    /// Optimize multiple parameter sets in parallel
    pub fn optimize_parallel_batch(
        &mut self,
        circuit_fn: impl Fn(&[f64]) -> QuantRS2Result<BatchCircuit> + Sync + Send + Clone + 'static,
        initial_param_sets: &[Vec<f64>],
        cost_fn: impl Fn(&BatchStateVector) -> f64 + Sync + Send + Clone + 'static,
        initial_states: &BatchStateVector,
    ) -> QuantRS2Result<Vec<OptimizeResult<f64>>> {
        let results: Vec<_> = initial_param_sets
            .par_iter()
            .map(|params| {
                let mut optimizer = self.clone();
                optimizer.optimize_batch(
                    circuit_fn.clone(),
                    params,
                    cost_fn.clone(),
                    initial_states,
                )
            })
            .collect::<QuantRS2Result<Vec<_>>>()?;

        Ok(results)
    }
}

impl Clone for BatchParameterOptimizer {
    fn clone(&self) -> Self {
        Self {
            executor: self.executor.clone(),
            config: self.config.clone(),
            gradient_cache: self.gradient_cache.clone(),
        }
    }
}

impl Clone for BatchCircuitExecutor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            gpu_backend: self.gpu_backend.clone(),
            // thread_pool: None, // Don't clone thread pool - removed per CORE_USAGE_POLICY
        }
    }
}

/// Static function to compute gradient for a single parameter
fn compute_single_gradient_static(
    executor: &BatchCircuitExecutor,
    circuit_fn: &impl Fn(&[f64]) -> QuantRS2Result<BatchCircuit>,
    params: &[f64],
    param_idx: usize,
    cost_fn: &impl Fn(&BatchStateVector) -> f64,
    initial_states: &BatchStateVector,
    shift: f64,
) -> QuantRS2Result<f64> {
    // Parameter shift rule: df/dθ = (f(θ+π/2) - f(θ-π/2)) / 2
    let mut params_plus = params.to_vec();
    let mut params_minus = params.to_vec();

    params_plus[param_idx] += shift;
    params_minus[param_idx] -= shift;

    // Evaluate at shifted parameters
    let circuit_plus = circuit_fn(&params_plus)?;
    let circuit_minus = circuit_fn(&params_minus)?;

    let mut states_plus = initial_states.clone();
    let mut states_minus = initial_states.clone();

    let result_plus = executor.execute_batch(&circuit_plus, &mut states_plus);
    let result_minus = executor.execute_batch(&circuit_minus, &mut states_minus);

    result_plus?;
    result_minus?;

    let cost_plus = cost_fn(&states_plus);
    let cost_minus = cost_fn(&states_minus);

    Ok((cost_plus - cost_minus) / (2.0 * shift))
}

/// Batch VQE (Variational Quantum Eigensolver) optimization
pub struct BatchVQE {
    /// Parameter optimizer
    optimizer: BatchParameterOptimizer,
    /// Hamiltonian to minimize
    hamiltonian: Array2<Complex64>,
}

impl BatchVQE {
    /// Create a new batch VQE optimizer
    pub const fn new(
        executor: BatchCircuitExecutor,
        hamiltonian: Array2<Complex64>,
        config: OptimizationConfig,
    ) -> Self {
        Self {
            optimizer: BatchParameterOptimizer::new(executor, config),
            hamiltonian,
        }
    }

    /// Run VQE optimization
    pub fn optimize(
        &mut self,
        ansatz_fn: impl Fn(&[f64]) -> QuantRS2Result<BatchCircuit> + Sync + Send + Clone + 'static,
        initial_params: &[f64],
        num_samples: usize,
        n_qubits: usize,
    ) -> QuantRS2Result<VQEResult> {
        // Create batch of initial states
        let batch = BatchStateVector::new(num_samples, n_qubits, Default::default())?;

        // Define cost function (energy expectation)
        let hamiltonian = self.hamiltonian.clone();
        let cost_fn = move |states: &BatchStateVector| -> f64 {
            let mut total_energy = 0.0;

            for i in 0..states.batch_size() {
                if let Ok(state) = states.get_state(i) {
                    let energy = compute_energy(&state, &hamiltonian);
                    total_energy += energy;
                }
            }

            total_energy / states.batch_size() as f64
        };

        // Run optimization
        let result = self
            .optimizer
            .optimize_batch(ansatz_fn, initial_params, cost_fn, &batch)?;

        Ok(VQEResult {
            optimal_params: result.x.to_vec(),
            ground_state_energy: result.fun,
            iterations: result.iterations,
            converged: result.success,
        })
    }
}

/// VQE optimization result
#[derive(Debug, Clone)]
pub struct VQEResult {
    /// Optimal parameters
    pub optimal_params: Vec<f64>,
    /// Ground state energy
    pub ground_state_energy: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
}

/// Compute energy expectation value
fn compute_energy(state: &Array1<Complex64>, hamiltonian: &Array2<Complex64>) -> f64 {
    let temp = hamiltonian.dot(state);
    let energy = state
        .iter()
        .zip(temp.iter())
        .map(|(a, b)| a.conj() * b)
        .sum::<Complex64>();

    energy.re
}

/// Batch QAOA (Quantum Approximate Optimization Algorithm)
pub struct BatchQAOA {
    /// Parameter optimizer
    optimizer: BatchParameterOptimizer,
    /// Problem Hamiltonian
    cost_hamiltonian: Array2<Complex64>,
    /// Mixer Hamiltonian
    mixer_hamiltonian: Array2<Complex64>,
    /// Number of QAOA layers
    p: usize,
}

impl BatchQAOA {
    /// Create a new batch QAOA optimizer
    pub const fn new(
        executor: BatchCircuitExecutor,
        cost_hamiltonian: Array2<Complex64>,
        mixer_hamiltonian: Array2<Complex64>,
        p: usize,
        config: OptimizationConfig,
    ) -> Self {
        Self {
            optimizer: BatchParameterOptimizer::new(executor, config),
            cost_hamiltonian,
            mixer_hamiltonian,
            p,
        }
    }

    /// Run QAOA optimization
    pub fn optimize(
        &mut self,
        initial_params: &[f64],
        num_samples: usize,
        n_qubits: usize,
    ) -> QuantRS2Result<QAOAResult> {
        if initial_params.len() != 2 * self.p {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} parameters, got {}",
                2 * self.p,
                initial_params.len()
            )));
        }

        // Create QAOA circuit constructor
        // let _p = self.p;
        let _cost_ham = self.cost_hamiltonian.clone();
        let _mixer_ham = self.mixer_hamiltonian.clone();

        let qaoa_circuit = move |_params: &[f64]| -> QuantRS2Result<BatchCircuit> {
            // This is a placeholder - actual QAOA circuit construction would go here
            let circuit = BatchCircuit::new(n_qubits);
            // Add QAOA layers based on params, cost_ham, mixer_ham
            Ok(circuit)
        };

        // Create batch of superposition initial states
        let batch = BatchStateVector::new(num_samples, n_qubits, Default::default())?;
        // Initialize to uniform superposition (would apply Hadamards)

        // Define cost function
        let cost_hamiltonian = self.cost_hamiltonian.clone();
        let cost_fn = move |states: &BatchStateVector| -> f64 {
            let mut total_cost = 0.0;

            for i in 0..states.batch_size() {
                if let Ok(state) = states.get_state(i) {
                    let cost = compute_energy(&state, &cost_hamiltonian);
                    total_cost += cost;
                }
            }

            total_cost / states.batch_size() as f64
        };

        // Run optimization
        let result =
            self.optimizer
                .optimize_batch(qaoa_circuit, initial_params, cost_fn, &batch)?;

        Ok(QAOAResult {
            optimal_params: result.x.to_vec(),
            optimal_cost: result.fun,
            iterations: result.iterations,
            converged: result.success,
        })
    }
}

/// QAOA optimization result
#[derive(Debug, Clone)]
pub struct QAOAResult {
    /// Optimal parameters (beta and gamma values)
    pub optimal_params: Vec<f64>,
    /// Optimal cost value
    pub optimal_cost: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::Hadamard;
    use crate::qubit::QubitId;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gradient_computation() {
        let config = Default::default();
        let executor = BatchCircuitExecutor::new(config)
            .expect("BatchCircuitExecutor creation should succeed");
        let mut optimizer = BatchParameterOptimizer::new(executor, Default::default());

        // Simple circuit function
        let circuit_fn = |_params: &[f64]| -> QuantRS2Result<BatchCircuit> {
            let mut circuit = BatchCircuit::new(1);
            // Add parameterized rotation based on params[0]
            circuit.add_gate(Box::new(Hadamard { target: QubitId(0) }))?;
            Ok(circuit)
        };

        // Simple cost function
        let cost_fn = |_states: &BatchStateVector| -> f64 { 1.0 };

        let batch = BatchStateVector::new(1, 1, Default::default())
            .expect("BatchStateVector creation should succeed");
        let params = vec![0.5];

        let gradients = optimizer
            .compute_gradients_batch(circuit_fn, &params, cost_fn, &batch, 0.01)
            .expect("Gradient computation should succeed");

        assert_eq!(gradients.len(), 1);
    }

    #[test]
    fn test_vqe_setup() {
        let executor = BatchCircuitExecutor::new(Default::default())
            .expect("BatchCircuitExecutor creation should succeed");

        // Simple 2x2 Hamiltonian
        let hamiltonian = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ];

        let vqe = BatchVQE::new(executor, hamiltonian, Default::default());

        // Just test creation
        assert_eq!(vqe.hamiltonian.shape(), &[2, 2]);
    }
}
