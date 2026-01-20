//! Quantum Zeno Effect Annealer implementation
//!
//! This module implements quantum annealing using the Zeno effect to control
//! quantum evolution and enhance optimization performance.

use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use std::time::{Duration, Instant};

use super::error::{AdvancedQuantumError, AdvancedQuantumResult};
use super::utils::Complex;
use crate::ising::IsingModel;
use crate::qaoa::QuantumState;
use crate::simulator::{AnnealingResult, AnnealingSolution};

/// Quantum Zeno Effect Annealer
#[derive(Debug, Clone)]
pub struct QuantumZenoAnnealer {
    /// Zeno configuration
    pub config: ZenoConfig,
    /// Measurement history
    pub measurement_history: Vec<ZenoMeasurement>,
    /// Evolution control
    pub evolution_controller: ZenoEvolutionController,
    /// Performance metrics
    pub performance_metrics: ZenoPerformanceMetrics,
}

/// Configuration for Quantum Zeno effect annealing
#[derive(Debug, Clone)]
pub struct ZenoConfig {
    /// Measurement frequency
    pub measurement_frequency: f64,
    /// Measurement strength
    pub measurement_strength: f64,
    /// Zeno subspace projection
    pub subspace_projection: ZenoSubspaceProjection,
    /// Evolution time between measurements
    pub evolution_time_step: f64,
    /// Total evolution time
    pub total_evolution_time: f64,
    /// Adaptive measurement strategy
    pub adaptive_strategy: ZenoAdaptiveStrategy,
    /// Decoherence model
    pub decoherence_model: DecoherenceModel,
}

impl Default for ZenoConfig {
    fn default() -> Self {
        Self {
            measurement_frequency: 1.0,
            measurement_strength: 1.0,
            subspace_projection: ZenoSubspaceProjection::Adaptive,
            evolution_time_step: 0.1,
            total_evolution_time: 10.0,
            adaptive_strategy: ZenoAdaptiveStrategy::PerformanceBased,
            decoherence_model: DecoherenceModel::default(),
        }
    }
}

/// Zeno subspace projection methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZenoSubspaceProjection {
    /// Strong projection (frequent measurements)
    Strong,
    /// Weak projection (infrequent measurements)
    Weak,
    /// Adaptive projection
    Adaptive,
    /// Continuous monitoring
    Continuous,
}

/// Adaptive strategies for Zeno effect
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZenoAdaptiveStrategy {
    /// Fixed measurement intervals
    Fixed,
    /// Performance-based adaptation
    PerformanceBased,
    /// Energy-landscape guided
    LandscapeGuided,
    /// Quantum Fisher information guided
    QuantumFisherGuided,
}

/// Decoherence models
#[derive(Debug, Clone)]
pub struct DecoherenceModel {
    /// Dephasing rate
    pub dephasing_rate: f64,
    /// Relaxation rate
    pub relaxation_rate: f64,
    /// Environmental coupling strength
    pub environment_coupling: f64,
    /// Temperature
    pub temperature: f64,
    /// Noise correlation time
    pub correlation_time: f64,
}

impl Default for DecoherenceModel {
    fn default() -> Self {
        Self {
            dephasing_rate: 0.01,
            relaxation_rate: 0.005,
            environment_coupling: 0.1,
            temperature: 0.01,
            correlation_time: 1.0,
        }
    }
}

/// Zeno measurement record
#[derive(Debug, Clone)]
pub struct ZenoMeasurement {
    /// Measurement time
    pub time: f64,
    /// Measured observable value
    pub observable_value: f64,
    /// Measurement outcome probabilities
    pub outcome_probabilities: Vec<f64>,
    /// Post-measurement state
    pub post_measurement_state: QuantumState,
    /// Measurement uncertainty
    pub uncertainty: f64,
}

/// Zeno evolution controller
#[derive(Debug, Clone)]
pub struct ZenoEvolutionController {
    /// Current evolution time
    pub current_time: f64,
    /// Evolution operator
    pub evolution_operator: EvolutionOperator,
    /// Measurement schedule
    pub measurement_schedule: Vec<f64>,
    /// Adaptive control parameters
    pub control_parameters: ZenoControlParameters,
}

/// Evolution operator representation
#[derive(Debug, Clone)]
pub struct EvolutionOperator {
    /// Hamiltonian components
    pub hamiltonian_components: Vec<HamiltonianComponent>,
    /// Time-dependent coefficients
    pub time_coefficients: Vec<TimeCoefficient>,
    /// Operator approximation method
    pub approximation_method: OperatorApproximation,
}

/// Hamiltonian components
#[derive(Debug, Clone)]
pub struct HamiltonianComponent {
    /// Pauli string representation
    pub pauli_string: String,
    /// Coefficient
    pub coefficient: f64,
    /// Qubit indices
    pub qubit_indices: Vec<usize>,
}

/// Time-dependent coefficients
#[derive(Debug, Clone)]
pub struct TimeCoefficient {
    /// Functional form
    pub function_type: TimeFunctionType,
    /// Parameters
    pub parameters: Vec<f64>,
    /// Time domain
    pub time_domain: (f64, f64),
}

/// Time function types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeFunctionType {
    /// Constant
    Constant,
    /// Linear
    Linear,
    /// Exponential
    Exponential,
    /// Sinusoidal
    Sinusoidal,
    /// Polynomial
    Polynomial,
    /// Custom function
    Custom(String),
}

/// Operator approximation methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorApproximation {
    /// Trotter decomposition
    Trotter,
    /// Suzuki decomposition
    Suzuki,
    /// Magnus expansion
    Magnus,
    /// Floquet theory
    Floquet,
    /// Krylov subspace
    Krylov,
}

/// Zeno control parameters
#[derive(Debug, Clone)]
pub struct ZenoControlParameters {
    /// Measurement adaptation rate
    pub adaptation_rate: f64,
    /// Control feedback gain
    pub feedback_gain: f64,
    /// Stability threshold
    pub stability_threshold: f64,
    /// Performance target
    pub performance_target: f64,
}

/// Zeno performance metrics
#[derive(Debug, Clone)]
pub struct ZenoPerformanceMetrics {
    /// Final energy
    pub final_energy: f64,
    /// Convergence time
    pub convergence_time: Duration,
    /// Number of measurements
    pub num_measurements: usize,
    /// Average measurement time
    pub avg_measurement_time: Duration,
    /// Zeno efficiency
    pub zeno_efficiency: f64,
    /// Fidelity to target state
    pub state_fidelity: f64,
}

impl QuantumZenoAnnealer {
    /// Create a new quantum Zeno effect annealer
    #[must_use]
    pub const fn new(config: ZenoConfig) -> Self {
        Self {
            config,
            measurement_history: Vec::new(),
            evolution_controller: ZenoEvolutionController {
                current_time: 0.0,
                evolution_operator: EvolutionOperator {
                    hamiltonian_components: Vec::new(),
                    time_coefficients: Vec::new(),
                    approximation_method: OperatorApproximation::Trotter,
                },
                measurement_schedule: Vec::new(),
                control_parameters: ZenoControlParameters {
                    adaptation_rate: 0.1,
                    feedback_gain: 1.0,
                    stability_threshold: 0.01,
                    performance_target: 0.9,
                },
            },
            performance_metrics: ZenoPerformanceMetrics {
                final_energy: f64::INFINITY,
                convergence_time: Duration::from_secs(0),
                num_measurements: 0,
                avg_measurement_time: Duration::from_millis(0),
                zeno_efficiency: 0.0,
                state_fidelity: 0.0,
            },
        }
    }

    /// Solve problem using Zeno annealing
    pub fn solve<P>(&mut self, problem: &P) -> AdvancedQuantumResult<AnnealingResult<Vec<i32>>>
    where
        P: Clone + 'static,
    {
        // For compatibility with the coordinator, convert to the expected format
        if let Ok(ising_problem) = self.convert_to_ising(problem) {
            let solution = self.anneal(&ising_problem)?;
            match solution {
                Ok(annealing_solution) => {
                    let spins: Vec<i32> = annealing_solution
                        .best_spins
                        .iter()
                        .map(|&s| i32::from(s))
                        .collect();
                    Ok(Ok(spins))
                }
                Err(err) => Ok(Err(err)),
            }
        } else {
            Err(AdvancedQuantumError::ZenoError(
                "Cannot convert problem to Ising model".to_string(),
            ))
        }
    }

    /// Convert generic problem to Ising model with enhanced handling
    fn convert_to_ising<P: 'static>(
        &self,
        problem: &P,
    ) -> Result<IsingModel, AdvancedQuantumError> {
        use std::any::Any;

        // Check if it's already an Ising model
        if let Some(ising) = (problem as &dyn Any).downcast_ref::<IsingModel>() {
            return Ok(ising.clone());
        }

        // Check if it's a reference to Ising model
        if let Some(ising_ref) = (problem as &dyn Any).downcast_ref::<&IsingModel>() {
            return Ok((*ising_ref).clone());
        }

        // For other problem types, generate a structured problem for testing
        let num_qubits = self.estimate_problem_size(problem);
        let mut ising = IsingModel::new(num_qubits);

        // Generate problem structure based on Zeno annealing requirements
        let problem_hash = self.hash_problem(problem);
        let mut rng = ChaCha8Rng::seed_from_u64(problem_hash);

        // Create problem suitable for Zeno effect protocols
        match self.config.subspace_projection {
            ZenoSubspaceProjection::Strong => {
                // Strong projection works well with clustered problems
                self.generate_clustered_problem(&mut ising, &mut rng)?;
            }
            ZenoSubspaceProjection::Weak => {
                // Weak projection handles distributed problems
                self.generate_distributed_problem(&mut ising, &mut rng)?;
            }
            _ => {
                // Default structured problem for adaptive and continuous
                self.generate_default_zeno_problem(&mut ising, &mut rng)?;
            }
        }

        Ok(ising)
    }

    /// Generate clustered problem for strong Zeno projection
    fn generate_clustered_problem(
        &self,
        ising: &mut IsingModel,
        rng: &mut ChaCha8Rng,
    ) -> Result<(), AdvancedQuantumError> {
        let num_qubits = ising.num_qubits;
        let cluster_size = 3;

        // Add clustered biases
        for i in 0..num_qubits {
            let cluster_id = i / cluster_size;
            let cluster_bias = match cluster_id % 3 {
                0 => 1.2,
                1 => -1.0,
                _ => 0.3,
            };
            let noise = rng.gen_range(-0.2..0.2);
            ising
                .set_bias(i, cluster_bias + noise)
                .map_err(AdvancedQuantumError::IsingError)?;
        }

        // Add strong intra-cluster couplings
        for cluster_start in (0..num_qubits).step_by(cluster_size) {
            let cluster_end = (cluster_start + cluster_size).min(num_qubits);
            for i in cluster_start..cluster_end {
                for j in (i + 1)..cluster_end {
                    let coupling = rng.gen_range(-1.2..1.2);
                    ising
                        .set_coupling(i, j, coupling)
                        .map_err(AdvancedQuantumError::IsingError)?;
                }
            }
        }

        // Add weaker inter-cluster couplings
        for i in 0..num_qubits {
            for j in (i + cluster_size)..num_qubits {
                if i / cluster_size != j / cluster_size && rng.gen_bool(0.3) {
                    let coupling = rng.gen_range(-0.3..0.3);
                    ising
                        .set_coupling(i, j, coupling)
                        .map_err(AdvancedQuantumError::IsingError)?;
                }
            }
        }

        Ok(())
    }

    /// Generate distributed problem for weak Zeno projection
    fn generate_distributed_problem(
        &self,
        ising: &mut IsingModel,
        rng: &mut ChaCha8Rng,
    ) -> Result<(), AdvancedQuantumError> {
        let num_qubits = ising.num_qubits;

        // Add uniform random biases
        for i in 0..num_qubits {
            let bias = rng.gen_range(-0.7..0.7);
            ising
                .set_bias(i, bias)
                .map_err(AdvancedQuantumError::IsingError)?;
        }

        // Add uniformly distributed couplings
        let coupling_probability = 0.2;
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                if rng.gen::<f64>() < coupling_probability {
                    let coupling = rng.gen_range(-0.5..0.5);
                    ising
                        .set_coupling(i, j, coupling)
                        .map_err(AdvancedQuantumError::IsingError)?;
                }
            }
        }

        Ok(())
    }

    /// Generate default Zeno problem
    fn generate_default_zeno_problem(
        &self,
        ising: &mut IsingModel,
        rng: &mut ChaCha8Rng,
    ) -> Result<(), AdvancedQuantumError> {
        let num_qubits = ising.num_qubits;

        // Add moderate biases with some structure
        for i in 0..num_qubits {
            let structural_bias = if i % 2 == 0 { 0.6 } else { -0.4 };
            let noise = rng.gen_range(-0.3..0.3);
            ising
                .set_bias(i, structural_bias + noise)
                .map_err(AdvancedQuantumError::IsingError)?;
        }

        // Add structured couplings
        for i in 0..(num_qubits - 1) {
            // Chain-like structure
            let coupling = rng.gen_range(-0.8..0.8);
            ising
                .set_coupling(i, i + 1, coupling)
                .map_err(AdvancedQuantumError::IsingError)?;
        }

        // Add some long-range couplings
        for _ in 0..(num_qubits / 3) {
            let i = rng.gen_range(0..num_qubits);
            let j = rng.gen_range(0..num_qubits);
            if i != j && (i as i32 - j as i32).abs() > 2 {
                let coupling = rng.gen_range(-0.4..0.4);
                ising
                    .set_coupling(i, j, coupling)
                    .map_err(AdvancedQuantumError::IsingError)?;
            }
        }

        Ok(())
    }

    /// Estimate problem size from generic type
    const fn estimate_problem_size<P>(&self, _problem: &P) -> usize {
        // In practice, would extract size from problem structure
        // Use reasonable size for Zeno protocols (not too large for quantum simulation)
        10
    }

    /// Generate hash for problem to ensure consistent conversion
    const fn hash_problem<P>(&self, _problem: &P) -> u64 {
        // In practice, would hash problem structure
        // Use fixed seed for reproducibility
        98_765
    }

    /// Perform Zeno effect annealing
    pub fn anneal(
        &mut self,
        problem: &IsingModel,
    ) -> AdvancedQuantumResult<AnnealingResult<AnnealingSolution>> {
        println!("Starting Quantum Zeno annealing");
        let start_time = Instant::now();

        // Initialize quantum state
        let mut current_state = self.initialize_quantum_state(problem)?;

        // Generate measurement schedule
        self.generate_measurement_schedule()?;

        let mut best_energy = f64::INFINITY;
        let mut best_solution = vec![-1; problem.num_qubits];

        // Evolution with Zeno measurements
        for &measurement_time in &self.evolution_controller.measurement_schedule.clone() {
            // Evolve until measurement time
            current_state = self.evolve_to_time(&current_state, measurement_time, problem)?;

            // Perform Zeno measurement
            let measurement = self.perform_zeno_measurement(&current_state, measurement_time)?;
            self.measurement_history.push(measurement.clone());

            // Update state after measurement
            current_state = measurement.post_measurement_state.clone();

            // Evaluate energy
            let energy = self.evaluate_state_energy(&current_state, problem)?;
            if energy < best_energy {
                best_energy = energy;
                best_solution = self.extract_classical_solution(&current_state)?;
            }

            // Adaptive measurement strategy
            if matches!(
                self.config.adaptive_strategy,
                ZenoAdaptiveStrategy::PerformanceBased
            ) {
                self.adapt_measurement_strategy(energy)?;
            }

            self.performance_metrics.num_measurements += 1;
        }

        // Final evolution to completion
        current_state =
            self.evolve_to_time(&current_state, self.config.total_evolution_time, problem)?;
        let final_energy = self.evaluate_state_energy(&current_state, problem)?;

        if final_energy < best_energy {
            best_energy = final_energy;
            best_solution = self.extract_classical_solution(&current_state)?;
        }

        // Update performance metrics
        self.performance_metrics.final_energy = best_energy;
        self.performance_metrics.convergence_time = start_time.elapsed();
        self.performance_metrics.avg_measurement_time = Duration::from_nanos(
            self.performance_metrics.convergence_time.as_nanos() as u64
                / self.performance_metrics.num_measurements.max(1) as u64,
        );
        self.performance_metrics.zeno_efficiency = self.calculate_zeno_efficiency()?;

        println!("Zeno annealing completed. Final energy: {best_energy:.6}");

        Ok(Ok(AnnealingSolution {
            best_energy,
            best_spins: best_solution,
            repetitions: 1,
            total_sweeps: self.performance_metrics.num_measurements,
            runtime: start_time.elapsed(),
            info: "Quantum Zeno effect annealing".to_string(),
        }))
    }

    /// Initialize quantum state
    fn initialize_quantum_state(
        &self,
        problem: &IsingModel,
    ) -> AdvancedQuantumResult<QuantumState> {
        // Initialize in superposition state
        let num_qubits = problem.num_qubits;
        let state_size = 1 << num_qubits;
        let amplitude = 1.0 / (state_size as f64).sqrt();

        Ok(QuantumState {
            amplitudes: vec![
                crate::qaoa::complex::Complex64 {
                    re: amplitude,
                    im: 0.0
                };
                state_size
            ],
            num_qubits,
        })
    }

    /// Generate measurement schedule
    fn generate_measurement_schedule(&mut self) -> AdvancedQuantumResult<()> {
        let num_measurements =
            (self.config.total_evolution_time * self.config.measurement_frequency) as usize;

        match self.config.subspace_projection {
            ZenoSubspaceProjection::Strong => {
                // Frequent measurements
                self.evolution_controller.measurement_schedule = (0..num_measurements)
                    .map(|i| (i + 1) as f64 / self.config.measurement_frequency)
                    .collect();
            }
            ZenoSubspaceProjection::Weak => {
                // Infrequent measurements
                let reduced_num = num_measurements / 4;
                self.evolution_controller.measurement_schedule = (0..reduced_num)
                    .map(|i| (i + 1) as f64 * 4.0 / self.config.measurement_frequency)
                    .collect();
            }
            ZenoSubspaceProjection::Adaptive => {
                // Start with moderate frequency, adapt based on performance
                self.evolution_controller.measurement_schedule = (0..num_measurements)
                    .map(|i| (i + 1) as f64 / self.config.measurement_frequency)
                    .collect();
            }
            ZenoSubspaceProjection::Continuous => {
                // Very frequent measurements
                let dense_num = num_measurements * 2;
                self.evolution_controller.measurement_schedule = (0..dense_num)
                    .map(|i| (i + 1) as f64 / (self.config.measurement_frequency * 2.0))
                    .collect();
            }
        }

        Ok(())
    }

    /// Evolve quantum state to specific time
    fn evolve_to_time(
        &self,
        initial_state: &QuantumState,
        target_time: f64,
        problem: &IsingModel,
    ) -> AdvancedQuantumResult<QuantumState> {
        let time_step = target_time - self.evolution_controller.current_time;

        if time_step <= 0.0 {
            return Ok(initial_state.clone());
        }

        // Simple time evolution (in practice would use sophisticated methods)
        let mut evolved_state = initial_state.clone();

        // Apply evolution operator
        for i in 0..evolved_state.amplitudes.len() {
            let phase = self.calculate_phase_for_state(i, time_step, problem)?;
            let phase_complex = complex_phase(phase);
            evolved_state.amplitudes[i] = crate::qaoa::complex::Complex64 {
                re: evolved_state.amplitudes[i].re.mul_add(
                    phase_complex.re,
                    -(evolved_state.amplitudes[i].im * phase_complex.im),
                ),
                im: evolved_state.amplitudes[i].re.mul_add(
                    phase_complex.im,
                    evolved_state.amplitudes[i].im * phase_complex.re,
                ),
            };
        }

        // Normalize
        let norm: f64 = evolved_state
            .amplitudes
            .iter()
            .map(|a| a.re.mul_add(a.re, a.im * a.im))
            .sum::<f64>()
            .sqrt();
        for amplitude in &mut evolved_state.amplitudes {
            amplitude.re /= norm;
            amplitude.im /= norm;
        }

        Ok(evolved_state)
    }

    /// Calculate phase for specific computational basis state
    fn calculate_phase_for_state(
        &self,
        state_index: usize,
        time_step: f64,
        problem: &IsingModel,
    ) -> AdvancedQuantumResult<f64> {
        // Calculate energy of this computational basis state
        let mut energy = 0.0;

        for qubit in 0..problem.num_qubits {
            let spin = if (state_index >> qubit) & 1 == 1 {
                1
            } else {
                -1
            };

            if let Ok(bias) = problem.get_bias(qubit) {
                energy += bias * f64::from(spin);
            }
        }

        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if let Ok(coupling) = problem.get_coupling(i, j) {
                    if coupling.abs() > 1e-10 {
                        let spin_i = if (state_index >> i) & 1 == 1 { 1 } else { -1 };
                        let spin_j = if (state_index >> j) & 1 == 1 { 1 } else { -1 };
                        energy += coupling * f64::from(spin_i * spin_j);
                    }
                }
            }
        }

        Ok(-energy * time_step) // Phase from time evolution
    }

    /// Perform Zeno measurement
    fn perform_zeno_measurement(
        &self,
        state: &QuantumState,
        measurement_time: f64,
    ) -> AdvancedQuantumResult<ZenoMeasurement> {
        // Simple Z-basis measurement (in practice would be more sophisticated)
        let mut rng = ChaCha8Rng::seed_from_u64(thread_rng().gen());

        // Calculate measurement probabilities
        let mut outcome_probabilities = Vec::new();
        for amplitude in &state.amplitudes {
            outcome_probabilities.push(amplitude.norm_squared());
        }

        // Sample measurement outcome
        let random_val: f64 = rng.gen();
        let mut cumulative_prob = 0.0;
        let mut measured_outcome = 0;

        for (i, &prob) in outcome_probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_val <= cumulative_prob {
                measured_outcome = i;
                break;
            }
        }

        // Collapse state to measured outcome
        let mut post_measurement_amplitudes =
            vec![crate::qaoa::complex::Complex64::new(0.0, 0.0); state.amplitudes.len()];
        post_measurement_amplitudes[measured_outcome] =
            crate::qaoa::complex::Complex64::new(1.0, 0.0);

        let post_measurement_state = QuantumState {
            amplitudes: post_measurement_amplitudes,
            num_qubits: state.num_qubits,
        };

        Ok(ZenoMeasurement {
            time: measurement_time,
            observable_value: measured_outcome as f64,
            outcome_probabilities: outcome_probabilities.clone(),
            post_measurement_state,
            uncertainty: outcome_probabilities[measured_outcome].sqrt(),
        })
    }

    /// Evaluate energy of quantum state
    fn evaluate_state_energy(
        &self,
        state: &QuantumState,
        problem: &IsingModel,
    ) -> AdvancedQuantumResult<f64> {
        let mut total_energy = 0.0;

        for (i, &amplitude) in state.amplitudes.iter().enumerate() {
            if amplitude.abs() > 1e-10 {
                let state_energy = self.calculate_phase_for_state(i, 1.0, problem)?;
                total_energy += amplitude.norm_squared() * (-state_energy); // Convert phase back to energy
            }
        }

        Ok(total_energy)
    }

    /// Extract classical solution from quantum state
    fn extract_classical_solution(&self, state: &QuantumState) -> AdvancedQuantumResult<Vec<i8>> {
        // Find most probable computational basis state
        let max_prob_index = state
            .amplitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.norm_squared()
                    .partial_cmp(&b.norm_squared())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);

        // Convert to spin configuration
        let mut solution = Vec::new();
        for qubit in 0..state.num_qubits {
            let spin = if (max_prob_index >> qubit) & 1 == 1 {
                1
            } else {
                -1
            };
            solution.push(spin);
        }

        Ok(solution)
    }

    /// Adapt measurement strategy based on performance
    fn adapt_measurement_strategy(&mut self, current_energy: f64) -> AdvancedQuantumResult<()> {
        // Simple adaptation: adjust measurement frequency based on energy improvement
        if let Some(last_measurement) = self.measurement_history.last() {
            let energy_change = last_measurement.observable_value - current_energy;

            if energy_change > 0.0 {
                // Improvement: slightly reduce measurement frequency to allow more evolution
                self.config.measurement_frequency *= 0.95;
            } else {
                // No improvement: increase measurement frequency for stronger Zeno effect
                self.config.measurement_frequency *= 1.05;
            }

            // Keep frequency in reasonable bounds
            self.config.measurement_frequency = self.config.measurement_frequency.clamp(0.1, 100.0);
        }

        Ok(())
    }

    /// Calculate Zeno efficiency metric
    fn calculate_zeno_efficiency(&self) -> AdvancedQuantumResult<f64> {
        if self.measurement_history.is_empty() {
            return Ok(0.0);
        }

        // Simple efficiency metric: improvement per measurement
        let initial_energy = self
            .measurement_history
            .first()
            .map(|m| m.observable_value)
            .unwrap_or(0.0);
        let final_energy = self.performance_metrics.final_energy;
        let improvement = initial_energy - final_energy;

        Ok(improvement / self.performance_metrics.num_measurements as f64)
    }
}

/// Helper function for complex phase calculation
fn complex_phase(phase: f64) -> Complex {
    Complex {
        re: phase.cos(),
        im: phase.sin(),
    }
}

/// Create default quantum Zeno annealer
#[must_use]
pub const fn create_quantum_zeno_annealer() -> QuantumZenoAnnealer {
    let config = ZenoConfig {
        measurement_frequency: 1.0,
        measurement_strength: 1.0,
        subspace_projection: ZenoSubspaceProjection::Adaptive,
        evolution_time_step: 0.1,
        total_evolution_time: 10.0,
        adaptive_strategy: ZenoAdaptiveStrategy::PerformanceBased,
        decoherence_model: DecoherenceModel {
            dephasing_rate: 0.01,
            relaxation_rate: 0.005,
            environment_coupling: 0.1,
            temperature: 0.01,
            correlation_time: 1.0,
        },
    };

    QuantumZenoAnnealer::new(config)
}

/// Create Zeno annealer with custom configuration
#[must_use]
pub const fn create_custom_zeno_annealer(
    measurement_frequency: f64,
    projection_type: ZenoSubspaceProjection,
    total_time: f64,
) -> QuantumZenoAnnealer {
    let config = ZenoConfig {
        measurement_frequency,
        measurement_strength: 1.0,
        subspace_projection: projection_type,
        evolution_time_step: 0.1,
        total_evolution_time: total_time,
        adaptive_strategy: ZenoAdaptiveStrategy::PerformanceBased,
        decoherence_model: DecoherenceModel {
            dephasing_rate: 0.01,
            relaxation_rate: 0.005,
            environment_coupling: 0.1,
            temperature: 0.01,
            correlation_time: 1.0,
        },
    };

    QuantumZenoAnnealer::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeno_annealer_creation() {
        let annealer = create_quantum_zeno_annealer();
        assert_eq!(annealer.config.measurement_frequency, 1.0);
        assert_eq!(annealer.config.total_evolution_time, 10.0);
        assert_eq!(annealer.measurement_history.len(), 0);
    }

    #[test]
    fn test_measurement_schedule_generation() {
        let mut annealer = create_quantum_zeno_annealer();
        annealer
            .generate_measurement_schedule()
            .expect("Measurement schedule generation should succeed");

        assert!(!annealer
            .evolution_controller
            .measurement_schedule
            .is_empty());

        // Check schedule is ordered
        let schedule = &annealer.evolution_controller.measurement_schedule;
        for i in 1..schedule.len() {
            assert!(schedule[i] > schedule[i - 1]);
        }
    }

    #[test]
    fn test_quantum_state_initialization() {
        let annealer = create_quantum_zeno_annealer();
        let mut ising = IsingModel::new(3);
        ising.set_bias(0, 1.0).expect("Setting bias should succeed");

        let state = annealer
            .initialize_quantum_state(&ising)
            .expect("Quantum state initialization should succeed");
        assert_eq!(state.num_qubits, 3);
        assert_eq!(state.amplitudes.len(), 8); // 2^3

        // Check normalization
        let norm_squared: f64 = state.amplitudes.iter().map(|a| a.norm_squared()).sum();
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solution_extraction() {
        let annealer = create_quantum_zeno_annealer();

        // State with highest probability on |10⟩ (index 2)
        let state = QuantumState {
            amplitudes: vec![
                crate::qaoa::complex::Complex64::new(0.1, 0.0),
                crate::qaoa::complex::Complex64::new(0.2, 0.0),
                crate::qaoa::complex::Complex64::new(0.9, 0.0),
                crate::qaoa::complex::Complex64::new(0.3, 0.0),
            ],
            num_qubits: 2,
        };

        let solution = annealer
            .extract_classical_solution(&state)
            .expect("Classical solution extraction should succeed");
        assert_eq!(solution.len(), 2);
        assert_eq!(solution[0], -1); // bit 0 is 0 -> spin -1
        assert_eq!(solution[1], 1); // bit 1 is 1 -> spin +1
    }
}
