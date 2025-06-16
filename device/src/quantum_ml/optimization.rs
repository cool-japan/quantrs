//! Quantum Machine Learning Optimization
//!
//! This module provides quantum-specific optimization algorithms for training
//! quantum machine learning models, including gradient-based and gradient-free methods.

use super::*;
use crate::{DeviceError, DeviceResult, QuantumDevice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Quantum optimizer trait
pub trait QuantumOptimizer: Send + Sync {
    /// Optimize parameters for a given objective function
    fn optimize(
        &mut self,
        initial_parameters: Vec<f64>,
        objective_function: Box<dyn ObjectiveFunction + Send + Sync>,
    ) -> DeviceResult<OptimizationResult>;

    /// Get optimizer configuration
    fn config(&self) -> &OptimizerConfig;

    /// Reset optimizer state
    fn reset(&mut self);
}

/// Objective function trait for quantum optimization
pub trait ObjectiveFunction: Send + Sync {
    /// Evaluate the objective function
    fn evaluate(&self, parameters: &[f64]) -> DeviceResult<f64>;

    /// Compute gradients (if available)
    fn gradient(&self, parameters: &[f64]) -> DeviceResult<Option<Vec<f64>>>;

    /// Get function metadata
    fn metadata(&self) -> HashMap<String, String>;
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub learning_rate: f64,
    pub momentum: Option<f64>,
    pub adaptive_learning_rate: bool,
    pub bounds: Option<(f64, f64)>,
    pub noise_resilience: bool,
    pub convergence_window: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            momentum: Some(0.9),
            adaptive_learning_rate: true,
            bounds: None,
            noise_resilience: true,
            convergence_window: 10,
        }
    }
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub optimal_parameters: Vec<f64>,
    pub optimal_value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub function_evaluations: usize,
    pub gradient_evaluations: usize,
    pub optimization_history: Vec<OptimizationStep>,
    pub final_gradient_norm: Option<f64>,
    pub execution_time: std::time::Duration,
}

/// Single optimization step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStep {
    pub iteration: usize,
    pub parameters: Vec<f64>,
    pub objective_value: f64,
    pub gradient_norm: Option<f64>,
    pub learning_rate: f64,
    pub step_size: f64,
}

/// Gradient-based optimizer
pub struct GradientBasedOptimizer {
    config: OptimizerConfig,
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    state: OptimizerState,
}

#[derive(Debug, Clone)]
struct OptimizerState {
    iteration: usize,
    momentum_buffer: Vec<f64>,
    velocity: Vec<f64>,          // For Adam
    squared_gradients: Vec<f64>, // For Adam/AdaGrad
    learning_rate: f64,
    convergence_history: Vec<f64>,
}

impl GradientBasedOptimizer {
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        config: OptimizerConfig,
    ) -> Self {
        let state = OptimizerState {
            iteration: 0,
            momentum_buffer: Vec::new(),
            velocity: Vec::new(),
            squared_gradients: Vec::new(),
            learning_rate: config.learning_rate,
            convergence_history: Vec::new(),
        };

        Self {
            config,
            device,
            state,
        }
    }

    fn update_parameters_adam(
        &mut self,
        parameters: &[f64],
        gradients: &[f64],
    ) -> DeviceResult<Vec<f64>> {
        if self.state.velocity.is_empty() {
            self.state.velocity = vec![0.0; parameters.len()];
            self.state.squared_gradients = vec![0.0; parameters.len()];
        }

        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        let mut updated_params = parameters.to_vec();

        for i in 0..parameters.len() {
            // Update biased first moment estimate
            self.state.velocity[i] = beta1 * self.state.velocity[i] + (1.0 - beta1) * gradients[i];

            // Update biased second raw moment estimate
            self.state.squared_gradients[i] = beta2 * self.state.squared_gradients[i]
                + (1.0 - beta2) * gradients[i] * gradients[i];

            // Compute bias-corrected first moment estimate
            let m_hat =
                self.state.velocity[i] / (1.0 - beta1.powi(self.state.iteration as i32 + 1));

            // Compute bias-corrected second raw moment estimate
            let v_hat = self.state.squared_gradients[i]
                / (1.0 - beta2.powi(self.state.iteration as i32 + 1));

            // Update parameter
            updated_params[i] -= self.state.learning_rate * m_hat / (v_hat.sqrt() + epsilon);

            // Apply bounds if specified
            if let Some((min_bound, max_bound)) = self.config.bounds {
                updated_params[i] = updated_params[i].clamp(min_bound, max_bound);
            }
        }

        Ok(updated_params)
    }

    fn update_parameters_sgd(
        &mut self,
        parameters: &[f64],
        gradients: &[f64],
    ) -> DeviceResult<Vec<f64>> {
        if self.state.momentum_buffer.is_empty() {
            self.state.momentum_buffer = vec![0.0; parameters.len()];
        }

        let momentum = self.config.momentum.unwrap_or(0.0);
        let mut updated_params = parameters.to_vec();

        for i in 0..parameters.len() {
            // Update momentum buffer
            self.state.momentum_buffer[i] =
                momentum * self.state.momentum_buffer[i] - self.state.learning_rate * gradients[i];

            // Update parameter
            updated_params[i] += self.state.momentum_buffer[i];

            // Apply bounds if specified
            if let Some((min_bound, max_bound)) = self.config.bounds {
                updated_params[i] = updated_params[i].clamp(min_bound, max_bound);
            }
        }

        Ok(updated_params)
    }

    fn update_learning_rate(&mut self, gradient_norm: f64) {
        if self.config.adaptive_learning_rate {
            // Simple adaptive learning rate based on gradient norm
            if gradient_norm > 1.0 {
                self.state.learning_rate *= 0.95;
            } else if gradient_norm < 0.1 {
                self.state.learning_rate *= 1.05;
            }

            // Keep within reasonable bounds
            self.state.learning_rate = self.state.learning_rate.clamp(1e-6, 1.0);
        }
    }

    fn check_convergence(&mut self, objective_value: f64) -> bool {
        self.state.convergence_history.push(objective_value);

        if self.state.convergence_history.len() < self.config.convergence_window {
            return false;
        }

        // Keep only the last window of values
        if self.state.convergence_history.len() > self.config.convergence_window {
            self.state.convergence_history.remove(0);
        }

        // Check if the change in objective value is below tolerance
        let recent_values = &self.state.convergence_history;
        let max_val = recent_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = recent_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        (max_val - min_val).abs() < self.config.tolerance
    }
}

impl QuantumOptimizer for GradientBasedOptimizer {
    fn optimize(
        &mut self,
        initial_parameters: Vec<f64>,
        objective_function: Box<dyn ObjectiveFunction + Send + Sync>,
    ) -> DeviceResult<OptimizationResult> {
        let start_time = std::time::Instant::now();
        let mut parameters = initial_parameters;
        let mut optimization_history = Vec::new();
        let mut function_evaluations = 0;
        let mut gradient_evaluations = 0;
        let mut best_value = f64::INFINITY;
        let mut best_parameters = parameters.clone();

        for iteration in 0..self.config.max_iterations {
            self.state.iteration = iteration;

            // Evaluate objective function
            let objective_value = objective_function.evaluate(&parameters)?;
            function_evaluations += 1;

            if objective_value < best_value {
                best_value = objective_value;
                best_parameters = parameters.clone();
            }

            // Compute gradients
            let gradients = objective_function.gradient(&parameters)?;

            let (gradient_norm, final_gradient_norm) = if let Some(grad) = gradients {
                gradient_evaluations += 1;
                let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

                // Update parameters based on optimizer type
                parameters = match self.config.optimizer_type {
                    OptimizerType::Adam => self.update_parameters_adam(&parameters, &grad)?,
                    OptimizerType::GradientDescent => {
                        self.update_parameters_sgd(&parameters, &grad)?
                    }
                    _ => {
                        return Err(DeviceError::InvalidInput(format!(
                            "Optimizer type {:?} not supported in gradient-based optimizer",
                            self.config.optimizer_type
                        )))
                    }
                };

                // Update learning rate
                self.update_learning_rate(norm);

                (Some(norm), Some(norm))
            } else {
                (None, None)
            };

            let step_size = if let Some(norm) = gradient_norm {
                self.state.learning_rate * norm
            } else {
                0.0
            };

            // Record optimization step
            optimization_history.push(OptimizationStep {
                iteration,
                parameters: parameters.clone(),
                objective_value,
                gradient_norm,
                learning_rate: self.state.learning_rate,
                step_size,
            });

            // Check convergence
            if self.check_convergence(objective_value) {
                return Ok(OptimizationResult {
                    optimal_parameters: best_parameters,
                    optimal_value: best_value,
                    iterations: iteration + 1,
                    converged: true,
                    function_evaluations,
                    gradient_evaluations,
                    optimization_history,
                    final_gradient_norm,
                    execution_time: start_time.elapsed(),
                });
            }
        }

        Ok(OptimizationResult {
            optimal_parameters: best_parameters,
            optimal_value: best_value,
            iterations: self.config.max_iterations,
            converged: false,
            function_evaluations,
            gradient_evaluations,
            optimization_history: optimization_history.clone(),
            final_gradient_norm: optimization_history
                .last()
                .and_then(|step| step.gradient_norm),
            execution_time: start_time.elapsed(),
        })
    }

    fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    fn reset(&mut self) {
        self.state = OptimizerState {
            iteration: 0,
            momentum_buffer: Vec::new(),
            velocity: Vec::new(),
            squared_gradients: Vec::new(),
            learning_rate: self.config.learning_rate,
            convergence_history: Vec::new(),
        };
    }
}

/// Gradient-free optimizer (for noisy quantum devices)
pub struct GradientFreeOptimizer {
    config: OptimizerConfig,
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    population: Vec<Individual>,
    generation: usize,
}

#[derive(Debug, Clone)]
struct Individual {
    parameters: Vec<f64>,
    fitness: Option<f64>,
}

impl GradientFreeOptimizer {
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        config: OptimizerConfig,
    ) -> Self {
        Self {
            config,
            device,
            population: Vec::new(),
            generation: 0,
        }
    }

    fn initialize_population(&mut self, parameter_count: usize, population_size: usize) {
        self.population.clear();

        for _ in 0..population_size {
            let parameters = if let Some((min_bound, max_bound)) = self.config.bounds {
                (0..parameter_count)
                    .map(|_| min_bound + fastrand::f64() * (max_bound - min_bound))
                    .collect()
            } else {
                (0..parameter_count)
                    .map(|_| fastrand::f64() * 2.0 * std::f64::consts::PI)
                    .collect()
            };

            self.population.push(Individual {
                parameters,
                fitness: None,
            });
        }
    }

    fn evaluate_population(
        &mut self,
        objective_function: &dyn ObjectiveFunction,
    ) -> DeviceResult<()> {
        for individual in &mut self.population {
            if individual.fitness.is_none() {
                individual.fitness = Some(objective_function.evaluate(&individual.parameters)?);
            }
        }
        Ok(())
    }

    fn evolve_population(&mut self) -> DeviceResult<()> {
        // Sort population by fitness
        self.population.sort_by(|a, b| {
            let fitness_a = a.fitness.unwrap_or(f64::INFINITY);
            let fitness_b = b.fitness.unwrap_or(f64::INFINITY);
            fitness_a
                .partial_cmp(&fitness_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let elite_count = self.population.len() / 4; // Keep top 25%
        let new_population_size = self.population.len();

        // Keep elite individuals
        let mut new_population = self.population[..elite_count].to_vec();

        // Generate offspring
        while new_population.len() < new_population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(3)?;
            let parent2 = self.tournament_selection(3)?;

            // Crossover and mutation
            let mut offspring = self.crossover(&parent1, &parent2)?;
            self.mutate(&mut offspring)?;

            new_population.push(offspring);
        }

        self.population = new_population;
        Ok(())
    }

    fn tournament_selection(&self, tournament_size: usize) -> DeviceResult<Individual> {
        let mut best_individual = None;
        let mut best_fitness = f64::INFINITY;

        for _ in 0..tournament_size {
            let idx = fastrand::usize(0..self.population.len());
            let individual = &self.population[idx];
            let fitness = individual.fitness.unwrap_or(f64::INFINITY);

            if fitness < best_fitness {
                best_fitness = fitness;
                best_individual = Some(individual.clone());
            }
        }

        best_individual
            .ok_or_else(|| DeviceError::InvalidInput("Tournament selection failed".to_string()))
    }

    fn crossover(&self, parent1: &Individual, parent2: &Individual) -> DeviceResult<Individual> {
        let parameter_count = parent1.parameters.len();
        let mut offspring_params = Vec::with_capacity(parameter_count);

        for i in 0..parameter_count {
            // Uniform crossover
            let param = if fastrand::bool() {
                parent1.parameters[i]
            } else {
                parent2.parameters[i]
            };
            offspring_params.push(param);
        }

        Ok(Individual {
            parameters: offspring_params,
            fitness: None,
        })
    }

    fn mutate(&self, individual: &mut Individual) -> DeviceResult<()> {
        let mutation_rate = 0.1;
        let mutation_strength = 0.1;

        for param in &mut individual.parameters {
            if fastrand::f64() < mutation_rate {
                let mutation = (fastrand::f64() - 0.5) * 2.0 * mutation_strength;
                *param += mutation;

                // Apply bounds if specified
                if let Some((min_bound, max_bound)) = self.config.bounds {
                    *param = param.clamp(min_bound, max_bound);
                }
            }
        }

        individual.fitness = None; // Reset fitness after mutation
        Ok(())
    }

    fn get_best_individual(&self) -> Option<&Individual> {
        self.population.iter().min_by(|a, b| {
            let fitness_a = a.fitness.unwrap_or(f64::INFINITY);
            let fitness_b = b.fitness.unwrap_or(f64::INFINITY);
            fitness_a
                .partial_cmp(&fitness_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl QuantumOptimizer for GradientFreeOptimizer {
    fn optimize(
        &mut self,
        initial_parameters: Vec<f64>,
        objective_function: Box<dyn ObjectiveFunction + Send + Sync>,
    ) -> DeviceResult<OptimizationResult> {
        let start_time = std::time::Instant::now();
        let population_size = 20; // Default population size
        let parameter_count = initial_parameters.len();
        let mut optimization_history = Vec::new();
        let mut function_evaluations = 0;

        // Initialize population with the initial parameters as one individual
        self.initialize_population(parameter_count, population_size - 1);
        self.population.push(Individual {
            parameters: initial_parameters,
            fitness: None,
        });

        for generation in 0..self.config.max_iterations {
            self.generation = generation;

            // Evaluate population
            self.evaluate_population(objective_function.as_ref())?;
            function_evaluations += self.population.len();

            // Get best individual
            let best_individual = self.get_best_individual().ok_or_else(|| {
                DeviceError::InvalidInput("No valid individuals in population".to_string())
            })?;

            let best_fitness = best_individual.fitness.unwrap_or(f64::INFINITY);

            // Record optimization step
            optimization_history.push(OptimizationStep {
                iteration: generation,
                parameters: best_individual.parameters.clone(),
                objective_value: best_fitness,
                gradient_norm: None,
                learning_rate: 0.0, // Not applicable for gradient-free
                step_size: 0.0,
            });

            // Check convergence
            if generation > 10 {
                let recent_best: Vec<f64> = optimization_history
                    .iter()
                    .rev()
                    .take(10)
                    .map(|step| step.objective_value)
                    .collect();

                let max_recent = recent_best.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_recent = recent_best.iter().fold(f64::INFINITY, |a, &b| a.min(b));

                if (max_recent - min_recent).abs() < self.config.tolerance {
                    return Ok(OptimizationResult {
                        optimal_parameters: best_individual.parameters.clone(),
                        optimal_value: best_fitness,
                        iterations: generation + 1,
                        converged: true,
                        function_evaluations,
                        gradient_evaluations: 0,
                        optimization_history,
                        final_gradient_norm: None,
                        execution_time: start_time.elapsed(),
                    });
                }
            }

            // Evolve population for next generation
            if generation < self.config.max_iterations - 1 {
                self.evolve_population()?;
            }
        }

        let best_individual = self.get_best_individual().ok_or_else(|| {
            DeviceError::InvalidInput("No valid individuals in final population".to_string())
        })?;

        Ok(OptimizationResult {
            optimal_parameters: best_individual.parameters.clone(),
            optimal_value: best_individual.fitness.unwrap_or(f64::INFINITY),
            iterations: self.config.max_iterations,
            converged: false,
            function_evaluations,
            gradient_evaluations: 0,
            optimization_history,
            final_gradient_norm: None,
            execution_time: start_time.elapsed(),
        })
    }

    fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    fn reset(&mut self) {
        self.population.clear();
        self.generation = 0;
    }
}

/// VQE objective function
pub struct VQEObjectiveFunction {
    hamiltonian: super::variational_algorithms::Hamiltonian,
    ansatz: Box<dyn super::variational_algorithms::VariationalAnsatz>,
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    shots: usize,
}

impl VQEObjectiveFunction {
    pub fn new(
        hamiltonian: super::variational_algorithms::Hamiltonian,
        ansatz: Box<dyn super::variational_algorithms::VariationalAnsatz>,
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        shots: usize,
    ) -> Self {
        Self {
            hamiltonian,
            ansatz,
            device,
            shots,
        }
    }
}

impl ObjectiveFunction for VQEObjectiveFunction {
    fn evaluate(&self, parameters: &[f64]) -> DeviceResult<f64> {
        // This is a simplified VQE energy evaluation
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let circuit = self.ansatz.build_circuit(parameters)?;
            let device = self.device.read().await;
            let result = Self::execute_circuit_helper(&*device, &circuit, self.shots).await?;

            // Compute energy expectation value
            let mut energy = 0.0;
            let total_shots = result.shots as f64;

            for term in &self.hamiltonian.terms {
                let mut term_expectation = 0.0;

                for (bitstring, count) in &result.counts {
                    let probability = *count as f64 / total_shots;
                    let mut eigenvalue = term.coefficient;

                    for (qubit_idx, pauli_op) in &term.paulis {
                        if let Some(bit_char) = bitstring.chars().nth(*qubit_idx) {
                            let bit_value = if bit_char == '1' { -1.0 } else { 1.0 };

                            match pauli_op {
                                super::variational_algorithms::PauliOperator::Z => {
                                    eigenvalue *= bit_value
                                }
                                super::variational_algorithms::PauliOperator::I => {} // Identity
                                _ => {} // X and Y would need basis rotation
                            }
                        }
                    }

                    term_expectation += probability * eigenvalue;
                }

                energy += term_expectation;
            }

            Ok(energy)
        })
    }

    fn gradient(&self, _parameters: &[f64]) -> DeviceResult<Option<Vec<f64>>> {
        // Gradient computation would require parameter shift rule implementation
        Ok(None)
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("objective_type".to_string(), "VQE".to_string());
        metadata.insert(
            "hamiltonian_terms".to_string(),
            self.hamiltonian.terms.len().to_string(),
        );
        metadata.insert("shots".to_string(), self.shots.to_string());
        metadata
    }
}

impl VQEObjectiveFunction {
    /// Execute a circuit on the quantum device (helper function to work around trait object limitations)
    async fn execute_circuit_helper(
        device: &(dyn QuantumDevice + Send + Sync),
        circuit: &ParameterizedQuantumCircuit,
        shots: usize,
    ) -> DeviceResult<CircuitResult> {
        // For now, return a mock result since we can't execute circuits directly
        // In a real implementation, this would need proper circuit execution
        let mut counts = std::collections::HashMap::new();
        counts.insert("0".repeat(circuit.num_qubits()), shots / 2);
        counts.insert("1".repeat(circuit.num_qubits()), shots / 2);
        
        Ok(CircuitResult {
            counts,
            shots,
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Create a gradient-based optimizer
pub fn create_gradient_optimizer(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    optimizer_type: OptimizerType,
    learning_rate: f64,
) -> Box<dyn QuantumOptimizer> {
    let config = OptimizerConfig {
        optimizer_type,
        learning_rate,
        ..Default::default()
    };

    Box::new(GradientBasedOptimizer::new(device, config))
}

/// Create a gradient-free optimizer
pub fn create_gradient_free_optimizer(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    max_iterations: usize,
) -> Box<dyn QuantumOptimizer> {
    let config = OptimizerConfig {
        optimizer_type: OptimizerType::GradientDescent, // Not used for gradient-free
        max_iterations,
        ..Default::default()
    };

    Box::new(GradientFreeOptimizer::new(device, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_mock_quantum_device;

    /// Mock objective function for testing
    struct QuadraticObjective {
        target: Vec<f64>,
    }

    impl ObjectiveFunction for QuadraticObjective {
        fn evaluate(&self, parameters: &[f64]) -> DeviceResult<f64> {
            let mut sum = 0.0;
            for (i, &param) in parameters.iter().enumerate() {
                let target = self.target.get(i).unwrap_or(&0.0);
                sum += (param - target).powi(2);
            }
            Ok(sum)
        }

        fn gradient(&self, parameters: &[f64]) -> DeviceResult<Option<Vec<f64>>> {
            let mut grad = Vec::new();
            for (i, &param) in parameters.iter().enumerate() {
                let target = self.target.get(i).unwrap_or(&0.0);
                grad.push(2.0 * (param - target));
            }
            Ok(Some(grad))
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut metadata = HashMap::new();
            metadata.insert("objective_type".to_string(), "quadratic".to_string());
            metadata
        }
    }

    #[test]
    fn test_gradient_based_optimizer() {
        let device = create_mock_quantum_device();
        let mut optimizer = GradientBasedOptimizer::new(device, OptimizerConfig::default());

        let objective = Box::new(QuadraticObjective {
            target: vec![1.0, 2.0, 3.0],
        });

        let initial_params = vec![0.0, 0.0, 0.0];
        let result = optimizer.optimize(initial_params, objective).unwrap();

        assert!(result.optimal_value < 1.0); // Should be close to minimum
        assert!(result.function_evaluations > 0);
        assert!(result.gradient_evaluations > 0);
    }

    #[test]
    fn test_gradient_free_optimizer() {
        let device = create_mock_quantum_device();
        let config = OptimizerConfig {
            max_iterations: 50,
            tolerance: 1e-3,
            ..Default::default()
        };
        let mut optimizer = GradientFreeOptimizer::new(device, config);

        let objective = Box::new(QuadraticObjective {
            target: vec![1.0, 2.0],
        });

        let initial_params = vec![0.0, 0.0];
        let result = optimizer.optimize(initial_params, objective).unwrap();

        assert!(result.optimal_value < 5.0); // Should improve from initial
        assert!(result.function_evaluations > 0);
        assert_eq!(result.gradient_evaluations, 0); // Gradient-free
    }

    #[test]
    fn test_optimization_result() {
        let result = OptimizationResult {
            optimal_parameters: vec![1.0, 2.0],
            optimal_value: 0.5,
            iterations: 100,
            converged: true,
            function_evaluations: 200,
            gradient_evaluations: 100,
            optimization_history: vec![],
            final_gradient_norm: Some(1e-6),
            execution_time: std::time::Duration::from_millis(500),
        };

        assert_eq!(result.optimal_parameters.len(), 2);
        assert_eq!(result.optimal_value, 0.5);
        assert!(result.converged);
    }

    #[test]
    fn test_optimizer_config() {
        let config = OptimizerConfig {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.001,
            max_iterations: 500,
            ..Default::default()
        };

        assert_eq!(config.optimizer_type, OptimizerType::Adam);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.max_iterations, 500);
    }
}
