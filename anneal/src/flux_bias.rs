//! Flux bias optimization for D-Wave quantum annealing
//!
//! This module implements flux bias optimization techniques for D-Wave quantum
//! annealers. Flux bias allows fine-tuning the effective bias on qubits by
//! adjusting the flux through superconducting loops, which can improve solution
//! quality and compensate for hardware calibration errors.

use crate::embedding::Embedding;
use crate::ising::{IsingError, IsingModel, IsingResult, QuboModel};
use std::collections::HashMap;

/// Configuration for flux bias optimization
#[derive(Debug, Clone)]
pub struct FluxBiasConfig {
    /// Initial flux bias values (default 0.0)
    pub initial_flux_bias: f64,
    /// Maximum flux bias magnitude
    pub max_flux_bias: f64,
    /// Step size for flux bias adjustment
    pub step_size: f64,
    /// Number of samples for evaluation
    pub num_samples: usize,
    /// Use gradient-based optimization
    pub use_gradients: bool,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Regularization parameter to prevent large flux biases
    pub regularization: f64,
}

impl Default for FluxBiasConfig {
    fn default() -> Self {
        Self {
            initial_flux_bias: 0.0,
            max_flux_bias: 0.1, // D-Wave typical range is -0.2 to 0.2
            step_size: 0.01,
            num_samples: 100,
            use_gradients: true,
            learning_rate: 0.01,
            regularization: 0.001,
        }
    }
}

/// Results from flux bias optimization
#[derive(Debug, Clone)]
pub struct FluxBiasResult {
    /// Optimized flux bias values for each qubit
    pub flux_biases: HashMap<usize, f64>,
    /// Energy improvement achieved
    pub energy_improvement: f64,
    /// Number of optimization iterations
    pub iterations: usize,
    /// Final solution quality metric
    pub solution_quality: f64,
    /// Calibration corrections applied
    pub calibration_corrections: HashMap<usize, f64>,
}

/// Flux bias optimizer for D-Wave systems
pub struct FluxBiasOptimizer {
    config: FluxBiasConfig,
    /// Hardware calibration data (if available)
    calibration_data: Option<CalibrationData>,
    /// History of flux bias adjustments
    adjustment_history: Vec<HashMap<usize, f64>>,
}

/// Hardware calibration data
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Nominal bias values from calibration
    pub nominal_biases: HashMap<usize, f64>,
    /// Bias error estimates
    pub bias_errors: HashMap<usize, f64>,
    /// Coupling error estimates
    pub coupling_errors: HashMap<(usize, usize), f64>,
    /// Temperature estimates for each qubit
    pub qubit_temperatures: HashMap<usize, f64>,
}

impl FluxBiasOptimizer {
    /// Create a new flux bias optimizer
    #[must_use]
    pub const fn new(config: FluxBiasConfig) -> Self {
        Self {
            config,
            calibration_data: None,
            adjustment_history: Vec::new(),
        }
    }

    /// Set hardware calibration data
    pub fn set_calibration_data(&mut self, data: CalibrationData) {
        self.calibration_data = Some(data);
    }

    /// Optimize flux biases for an Ising model
    pub fn optimize_ising(
        &mut self,
        model: &IsingModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
    ) -> IsingResult<FluxBiasResult> {
        let mut result = FluxBiasResult {
            flux_biases: HashMap::new(),
            energy_improvement: 0.0,
            iterations: 0,
            solution_quality: 0.0,
            calibration_corrections: HashMap::new(),
        };

        // Initialize flux biases
        let hardware_qubits = self.get_hardware_qubits(embedding);
        for qubit in hardware_qubits {
            result
                .flux_biases
                .insert(qubit, self.config.initial_flux_bias);
        }

        // Apply calibration corrections if available
        if let Some(calibration) = &self.calibration_data {
            self.apply_calibration_corrections(&mut result.flux_biases, calibration);
            result
                .calibration_corrections
                .clone_from(&result.flux_biases);
        }

        // Compute initial energy
        let initial_energy = self.compute_average_energy(model, samples)?;

        if self.config.use_gradients {
            // Gradient-based optimization
            result = self.optimize_with_gradients(model, embedding, samples, result)?;
        } else {
            // Grid search optimization
            result = self.optimize_grid_search(model, embedding, samples, result)?;
        }

        // Compute final energy and improvement
        let final_energy =
            self.compute_average_energy_with_flux(model, samples, &result.flux_biases)?;
        result.energy_improvement = initial_energy - final_energy;
        result.solution_quality = self.compute_solution_quality(samples, embedding);

        Ok(result)
    }

    /// Optimize flux biases for a QUBO model
    pub fn optimize_qubo(
        &mut self,
        model: &QuboModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
    ) -> IsingResult<FluxBiasResult> {
        // Convert QUBO to Ising
        let (ising, _offset) = model.to_ising();
        self.optimize_ising(&ising, embedding, samples)
    }

    /// Get all hardware qubits used in the embedding
    fn get_hardware_qubits(&self, embedding: &Embedding) -> Vec<usize> {
        let mut hardware_qubits = Vec::new();
        for chain in embedding.chains.values() {
            for &qubit in chain {
                if !hardware_qubits.contains(&qubit) {
                    hardware_qubits.push(qubit);
                }
            }
        }
        hardware_qubits.sort_unstable();
        hardware_qubits
    }

    /// Apply calibration corrections to flux biases
    fn apply_calibration_corrections(
        &self,
        flux_biases: &mut HashMap<usize, f64>,
        calibration: &CalibrationData,
    ) {
        for (qubit, flux_bias) in flux_biases.iter_mut() {
            // Compensate for known bias errors
            if let Some(&error) = calibration.bias_errors.get(qubit) {
                *flux_bias -= error * 0.5; // Partial compensation
            }

            // Adjust for temperature variations
            if let Some(&temp) = calibration.qubit_temperatures.get(qubit) {
                let temp_correction = (temp - 15.0) * 0.001; // 15mK nominal
                *flux_bias += temp_correction;
            }
        }
    }

    /// Optimize using gradient descent
    fn optimize_with_gradients(
        &mut self,
        model: &IsingModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
        mut result: FluxBiasResult,
    ) -> IsingResult<FluxBiasResult> {
        let max_iterations = 50;
        let tolerance = 1e-6;

        for iteration in 0..max_iterations {
            result.iterations = iteration + 1;

            // Compute gradients
            let gradients =
                self.compute_gradients(model, embedding, samples, &result.flux_biases)?;

            // Update flux biases
            let mut converged = true;
            for (qubit, gradient) in gradients {
                if let Some(flux_bias) = result.flux_biases.get_mut(&qubit) {
                    let update = -self.config.learning_rate * gradient;

                    if update.abs() > tolerance {
                        converged = false;
                    }

                    // Update with bounds
                    *flux_bias = (*flux_bias + update)
                        .max(-self.config.max_flux_bias)
                        .min(self.config.max_flux_bias);
                }
            }

            // Store history
            self.adjustment_history.push(result.flux_biases.clone());

            // Check convergence
            if converged {
                break;
            }
        }

        Ok(result)
    }

    /// Optimize using grid search
    fn optimize_grid_search(
        &self,
        model: &IsingModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
        mut result: FluxBiasResult,
    ) -> IsingResult<FluxBiasResult> {
        let hardware_qubits = self.get_hardware_qubits(embedding);
        let mut best_energy = f64::INFINITY;
        let mut best_flux_biases = result.flux_biases.clone();

        // Simple coordinate descent
        for _ in 0..10 {
            // Max iterations
            result.iterations += 1;
            let mut improved = false;

            for &qubit in &hardware_qubits {
                let current_flux = result.flux_biases.get(&qubit).copied().unwrap_or(0.0);

                // Try different flux bias values
                for delta in [-self.config.step_size, 0.0, self.config.step_size] {
                    let new_flux = (current_flux + delta)
                        .max(-self.config.max_flux_bias)
                        .min(self.config.max_flux_bias);

                    result.flux_biases.insert(qubit, new_flux);

                    let energy =
                        self.compute_average_energy_with_flux(model, samples, &result.flux_biases)?;

                    if energy < best_energy {
                        best_energy = energy;
                        best_flux_biases.clone_from(&result.flux_biases);
                        improved = true;
                    }
                }

                // Restore best flux bias for this qubit
                if let Some(&best_flux) = best_flux_biases.get(&qubit) {
                    result.flux_biases.insert(qubit, best_flux);
                }
            }

            if !improved {
                break;
            }
        }

        result.flux_biases = best_flux_biases;
        Ok(result)
    }

    /// Compute gradients of energy with respect to flux biases
    fn compute_gradients(
        &self,
        model: &IsingModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
        flux_biases: &HashMap<usize, f64>,
    ) -> IsingResult<HashMap<usize, f64>> {
        let mut gradients = HashMap::new();
        let epsilon = 0.001;

        for (qubit, &current_flux) in flux_biases {
            // Forward difference
            let mut flux_plus = flux_biases.clone();
            flux_plus.insert(*qubit, current_flux + epsilon);
            let energy_plus = self.compute_average_energy_with_flux(model, samples, &flux_plus)?;

            // Backward difference
            let mut flux_minus = flux_biases.clone();
            flux_minus.insert(*qubit, current_flux - epsilon);
            let energy_minus =
                self.compute_average_energy_with_flux(model, samples, &flux_minus)?;

            // Gradient with regularization
            let gradient = self
                .config
                .regularization
                .mul_add(current_flux, (energy_plus - energy_minus) / (2.0 * epsilon));

            gradients.insert(*qubit, gradient);
        }

        Ok(gradients)
    }

    /// Compute average energy of samples
    fn compute_average_energy(&self, model: &IsingModel, samples: &[Vec<i8>]) -> IsingResult<f64> {
        let mut total_energy = 0.0;
        let mut valid_samples = 0;

        for sample in samples {
            match model.energy(sample) {
                Ok(energy) => {
                    total_energy += energy;
                    valid_samples += 1;
                }
                Err(_) => continue,
            }
        }

        if valid_samples == 0 {
            return Err(IsingError::InvalidValue("No valid samples".to_string()));
        }

        Ok(total_energy / f64::from(valid_samples))
    }

    /// Compute average energy with flux bias adjustments
    fn compute_average_energy_with_flux(
        &self,
        model: &IsingModel,
        samples: &[Vec<i8>],
        flux_biases: &HashMap<usize, f64>,
    ) -> IsingResult<f64> {
        let mut total_energy = 0.0;
        let mut valid_samples = 0;

        for sample in samples {
            match model.energy(sample) {
                Ok(mut energy) => {
                    // Add flux bias contributions
                    for (qubit, &flux_bias) in flux_biases {
                        if *qubit < sample.len() {
                            energy += flux_bias * f64::from(sample[*qubit]);
                        }
                    }
                    total_energy += energy;
                    valid_samples += 1;
                }
                Err(_) => continue,
            }
        }

        if valid_samples == 0 {
            return Err(IsingError::InvalidValue("No valid samples".to_string()));
        }

        Ok(total_energy / f64::from(valid_samples))
    }

    /// Compute solution quality metric
    fn compute_solution_quality(&self, samples: &[Vec<i8>], embedding: &Embedding) -> f64 {
        let mut chain_satisfaction = 0.0;
        let mut total_chains = 0;

        for sample in samples {
            for chain in embedding.chains.values() {
                if chain.len() > 1 {
                    total_chains += 1;
                    let first_val = sample[chain[0]];
                    let satisfied = chain[1..].iter().all(|&q| sample[q] == first_val);
                    if satisfied {
                        chain_satisfaction += 1.0;
                    }
                }
            }
        }

        if total_chains > 0 {
            chain_satisfaction / f64::from(total_chains)
        } else {
            1.0
        }
    }
}

/// Advanced flux bias optimizer with machine learning integration
pub struct MLFluxBiasOptimizer {
    base_optimizer: FluxBiasOptimizer,
    /// Learned flux bias patterns
    learned_patterns: HashMap<String, Vec<f64>>,
    /// Pattern recognition threshold
    pattern_threshold: f64,
}

impl MLFluxBiasOptimizer {
    /// Create a new ML-enhanced flux bias optimizer
    #[must_use]
    pub fn new(config: FluxBiasConfig) -> Self {
        Self {
            base_optimizer: FluxBiasOptimizer::new(config),
            learned_patterns: HashMap::new(),
            pattern_threshold: 0.8,
        }
    }

    /// Learn flux bias patterns from successful optimizations
    pub fn learn_pattern(&mut self, problem_type: &str, flux_biases: &HashMap<usize, f64>) {
        let pattern: Vec<f64> = flux_biases.values().copied().collect();
        self.learned_patterns
            .insert(problem_type.to_string(), pattern);
    }

    /// Apply learned patterns to new problems
    #[must_use]
    pub fn apply_learned_patterns(
        &self,
        problem_type: &str,
        num_qubits: usize,
    ) -> Option<HashMap<usize, f64>> {
        if let Some(pattern) = self.learned_patterns.get(problem_type) {
            let mut flux_biases = HashMap::new();
            for (i, &value) in pattern.iter().take(num_qubits).enumerate() {
                flux_biases.insert(i, value);
            }
            Some(flux_biases)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flux_bias_optimizer_creation() {
        let config = FluxBiasConfig::default();
        let optimizer = FluxBiasOptimizer::new(config);
        assert!(optimizer.adjustment_history.is_empty());
    }

    #[test]
    fn test_calibration_data() {
        let mut calibration = CalibrationData {
            nominal_biases: HashMap::new(),
            bias_errors: HashMap::new(),
            coupling_errors: HashMap::new(),
            qubit_temperatures: HashMap::new(),
        };

        calibration.bias_errors.insert(0, 0.01);
        calibration.qubit_temperatures.insert(0, 16.0); // 16mK

        assert_eq!(calibration.bias_errors.get(&0), Some(&0.01));
    }

    #[test]
    fn test_flux_bias_bounds() {
        let config = FluxBiasConfig {
            max_flux_bias: 0.1,
            ..Default::default()
        };

        assert!(config.max_flux_bias > 0.0);
        assert!(config.max_flux_bias < 1.0); // Reasonable range for D-Wave
    }
}
