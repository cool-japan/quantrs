//! Simulated quantum annealing simulator
//!
//! This module provides a simulator for quantum annealing, which can be used
//! to solve optimization problems formulated as Ising models or QUBO problems.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};

/// Errors that can occur during simulated annealing
#[derive(Error, Debug)]
pub enum AnnealingError {
    /// Error in the underlying Ising model
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Error when the annealing schedule is invalid
    #[error("Invalid annealing schedule: {0}")]
    InvalidSchedule(String),

    /// Error when the annealing parameters are invalid
    #[error("Invalid annealing parameter: {0}")]
    InvalidParameter(String),

    /// Error when the annealing process times out
    #[error("Annealing timeout after {0:?}")]
    Timeout(Duration),
}

/// Result type for annealing operations
pub type AnnealingResult<T> = Result<T, AnnealingError>;

/// Transverse field strength schedule for quantum annealing
///
/// The transverse field represents the quantum tunneling term in the Hamiltonian,
/// which decreases over time during the annealing process.
#[derive(Debug, Clone)]
pub enum TransverseFieldSchedule {
    /// Linear schedule: A(t) = A_0 * (1 - t/t_f)
    Linear,

    /// Exponential schedule: A(t) = A_0 * exp(-alpha * t/t_f)
    Exponential(f64), // alpha parameter

    /// Custom schedule function: A(t, t_f) -> A
    Custom(fn(f64, f64) -> f64),
}

impl TransverseFieldSchedule {
    /// Calculate the transverse field strength at time t
    pub fn calculate(&self, t: f64, t_f: f64, a_0: f64) -> f64 {
        match self {
            Self::Linear => a_0 * (1.0 - t / t_f),
            Self::Exponential(alpha) => a_0 * (-alpha * t / t_f).exp(),
            Self::Custom(func) => func(t, t_f),
        }
    }
}

/// Temperature schedule for simulated quantum annealing
///
/// The temperature controls the probability of accepting non-improving moves,
/// and typically decreases over time during the annealing process.
#[derive(Debug, Clone)]
pub enum TemperatureSchedule {
    /// Linear schedule: T(t) = T_0 * (1 - t/t_f)
    Linear,

    /// Exponential schedule: T(t) = T_0 * exp(-alpha * t/t_f)
    Exponential(f64), // alpha parameter

    /// Geometric schedule: T(t) = T_0 * alpha^(t/delta_t)
    Geometric(f64, f64), // alpha and delta_t parameters

    /// Custom schedule function: T(t, t_f) -> T
    Custom(fn(f64, f64) -> f64),
}

impl TemperatureSchedule {
    /// Calculate the temperature at time t
    pub fn calculate(&self, t: f64, t_f: f64, t_0: f64) -> f64 {
        match self {
            Self::Linear => t_0 * (1.0 - t / t_f),
            Self::Exponential(alpha) => t_0 * (-alpha * t / t_f).exp(),
            Self::Geometric(alpha, delta_t) => t_0 * alpha.powf(t / delta_t),
            Self::Custom(func) => func(t, t_f),
        }
    }
}

/// Parameters for simulated quantum annealing
#[derive(Debug, Clone)]
pub struct AnnealingParams {
    /// Initial transverse field strength
    pub initial_transverse_field: f64,

    /// Transverse field schedule
    pub transverse_field_schedule: TransverseFieldSchedule,

    /// Initial temperature
    pub initial_temperature: f64,

    /// Temperature schedule
    pub temperature_schedule: TemperatureSchedule,

    /// Number of Monte Carlo steps
    pub num_sweeps: usize,

    /// Number of spins to update per sweep
    pub updates_per_sweep: Option<usize>,

    /// Number of repetitions/restarts
    pub num_repetitions: usize,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Maximum runtime in seconds
    pub timeout: Option<f64>,

    /// Number of Trotter slices for quantum annealing
    pub trotter_slices: usize,
}

impl AnnealingParams {
    /// Create new annealing parameters with default values
    pub fn new() -> Self {
        Self {
            initial_transverse_field: 2.0,
            transverse_field_schedule: TransverseFieldSchedule::Linear,
            initial_temperature: 2.0,
            temperature_schedule: TemperatureSchedule::Exponential(3.0),
            num_sweeps: 1000,
            updates_per_sweep: None,
            num_repetitions: 10,
            seed: None,
            timeout: Some(60.0), // 60 seconds default timeout
            trotter_slices: 20,
        }
    }

    /// Validate the annealing parameters
    pub fn validate(&self) -> AnnealingResult<()> {
        // Check transverse field
        if self.initial_transverse_field <= 0.0 || !self.initial_transverse_field.is_finite() {
            return Err(AnnealingError::InvalidParameter(format!(
                "Initial transverse field must be positive and finite, got {}",
                self.initial_transverse_field
            )));
        }

        // Check temperature
        if self.initial_temperature <= 0.0 || !self.initial_temperature.is_finite() {
            return Err(AnnealingError::InvalidParameter(format!(
                "Initial temperature must be positive and finite, got {}",
                self.initial_temperature
            )));
        }

        // Check sweeps
        if self.num_sweeps == 0 {
            return Err(AnnealingError::InvalidParameter(
                "Number of sweeps must be positive".to_string(),
            ));
        }

        // Check repetitions
        if self.num_repetitions == 0 {
            return Err(AnnealingError::InvalidParameter(
                "Number of repetitions must be positive".to_string(),
            ));
        }

        // Check timeout
        if let Some(timeout) = self.timeout {
            if timeout <= 0.0 || !timeout.is_finite() {
                return Err(AnnealingError::InvalidParameter(format!(
                    "Timeout must be positive and finite, got {}",
                    timeout
                )));
            }
        }

        // Check Trotter slices
        if self.trotter_slices == 0 {
            return Err(AnnealingError::InvalidParameter(
                "Number of Trotter slices must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for AnnealingParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a simulated quantum annealing run
#[derive(Debug, Clone)]
pub struct AnnealingSolution {
    /// Best spin configuration found
    pub best_spins: Vec<i8>,

    /// Energy of the best configuration
    pub best_energy: f64,

    /// Number of repetitions performed
    pub repetitions: usize,

    /// Total number of sweeps performed
    pub total_sweeps: usize,

    /// Time taken for the annealing process
    pub runtime: Duration,

    /// Information about the annealing process
    pub info: String,
}

/// Simulated quantum annealing solver
///
/// This uses path integral Monte Carlo to simulate quantum annealing,
/// which can be used to find low-energy states of Ising models.
#[derive(Debug, Clone)]
pub struct QuantumAnnealingSimulator {
    /// Parameters for the annealing process
    params: AnnealingParams,
}

impl QuantumAnnealingSimulator {
    /// Create a new quantum annealing simulator with the given parameters
    pub fn new(params: AnnealingParams) -> AnnealingResult<Self> {
        // Validate parameters
        params.validate()?;

        Ok(Self { params })
    }

    /// Create a new quantum annealing simulator with default parameters
    pub fn with_default_params() -> AnnealingResult<Self> {
        Self::new(AnnealingParams::default())
    }
}

impl Default for QuantumAnnealingSimulator {
    fn default() -> Self {
        Self::with_default_params().expect("Default parameters should be valid")
    }
}

impl QuantumAnnealingSimulator {
    /// Find the ground state of an Ising model using simulated quantum annealing
    pub fn solve(&self, model: &IsingModel) -> AnnealingResult<AnnealingSolution> {
        // Start timer
        let start_time = Instant::now();

        // Create random number generator
        let mut rng = match self.params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };

        // Initialize best result
        let num_qubits = model.num_qubits;
        let mut best_spins = vec![1; num_qubits]; // Start with all spins up
        let mut best_energy = match model.energy(&best_spins) {
            Ok(energy) => energy,
            Err(err) => return Err(AnnealingError::IsingError(err)),
        };

        // Track sweeps and repetitions
        let mut total_sweeps = 0;
        let mut completed_repetitions = 0;

        // Determine updates per sweep
        let updates_per_sweep = self.params.updates_per_sweep.unwrap_or(num_qubits);

        // Prepare for quantum annealing with path integral Monte Carlo
        let trotter_slices = self.params.trotter_slices;

        // Perform multiple repetitions
        for _ in 0..self.params.num_repetitions {
            // Initialize random spin configuration for each Trotter slice
            let mut trotter_spins = vec![vec![0; num_qubits]; trotter_slices];
            for slice in &mut trotter_spins {
                for spin in slice.iter_mut() {
                    *spin = if rng.random_bool(0.5) { 1 } else { -1 };
                }
            }

            // Perform simulated quantum annealing
            for sweep in 0..self.params.num_sweeps {
                // Check timeout
                if let Some(timeout) = self.params.timeout {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    if elapsed > timeout {
                        return Err(AnnealingError::Timeout(Duration::from_secs_f64(elapsed)));
                    }
                }

                // Calculate current normalized time
                let t = sweep as f64 / self.params.num_sweeps as f64;
                let t_f = 1.0; // Final time (normalized)

                // Calculate current transverse field and temperature
                let transverse_field = self.params.transverse_field_schedule.calculate(
                    t,
                    t_f,
                    self.params.initial_transverse_field,
                );
                let temperature = self.params.temperature_schedule.calculate(
                    t,
                    t_f,
                    self.params.initial_temperature,
                );

                // Calculate coupling between Trotter slices (J_perp)
                let j_perp = -0.5
                    * temperature
                    * (trotter_slices as f64)
                    * (transverse_field / temperature).ln_1p().abs();

                // Perform Monte Carlo updates
                for _ in 0..updates_per_sweep {
                    // Choose a random qubit and Trotter slice
                    let qubit = rng.random_range(0..num_qubits);
                    let slice = rng.random_range(0..trotter_slices);

                    // Calculate energy change from flipping the spin
                    let current_spin = trotter_spins[slice][qubit];
                    let new_spin = -current_spin;

                    // Temporary change to calculate energy difference
                    trotter_spins[slice][qubit] = new_spin;

                    // Calculate energy of this Trotter slice
                    let mut delta_e = match model.energy(&trotter_spins[slice]) {
                        Ok(energy_new) => {
                            // Revert change to calculate original energy
                            trotter_spins[slice][qubit] = current_spin;
                            let energy_old = model.energy(&trotter_spins[slice])?;
                            energy_new - energy_old
                        }
                        Err(err) => return Err(AnnealingError::IsingError(err)),
                    };

                    // Add contribution from neighboring Trotter slices
                    let prev_slice = (slice + trotter_slices - 1) % trotter_slices;
                    let next_slice = (slice + 1) % trotter_slices;

                    // Convert spins to f64 for the calculations
                    let new_spin_f64 = new_spin as f64;
                    let current_spin_f64 = current_spin as f64;
                    let neighbor_sum = (trotter_spins[prev_slice][qubit]
                        + trotter_spins[next_slice][qubit])
                        as f64;

                    delta_e += j_perp * new_spin_f64 * neighbor_sum;
                    delta_e -= j_perp * current_spin_f64 * neighbor_sum;

                    // Metropolis acceptance criterion
                    let accept = delta_e <= 0.0 || {
                        let p = (-delta_e / temperature).exp();
                        rng.random_range(0.0..1.0) < p
                    };

                    // Apply the spin flip if accepted
                    if accept {
                        trotter_spins[slice][qubit] = new_spin;
                    }
                }

                // Increment sweep counter
                total_sweeps += 1;
            }

            // After annealing, compute the average spin configuration
            let mut avg_spins = vec![0; num_qubits];
            for qubit in 0..num_qubits {
                let sum: i32 = trotter_spins.iter().map(|slice| slice[qubit] as i32).sum();
                avg_spins[qubit] = if sum >= 0 { 1 } else { -1 };
            }

            // Check if this is a better solution
            match model.energy(&avg_spins) {
                Ok(energy) => {
                    if energy < best_energy {
                        best_energy = energy;
                        best_spins = avg_spins;
                    }
                }
                Err(err) => return Err(AnnealingError::IsingError(err)),
            }

            // Increment repetition counter
            completed_repetitions += 1;
        }

        // Calculate runtime
        let runtime = start_time.elapsed();

        // Build result
        Ok(AnnealingSolution {
            best_spins,
            best_energy,
            repetitions: completed_repetitions,
            total_sweeps,
            runtime,
            info: format!(
                "Performed {} repetitions with {} total sweeps in {:?}",
                completed_repetitions, total_sweeps, runtime
            ),
        })
    }
}

/// Classical simulated annealing solver
///
/// This uses Metropolis-Hastings algorithm for simulated annealing,
/// which can be used to find low-energy states of Ising models.
#[derive(Debug, Clone)]
pub struct ClassicalAnnealingSimulator {
    /// Parameters for the annealing process
    params: AnnealingParams,
}

impl ClassicalAnnealingSimulator {
    /// Create a new classical annealing simulator with the given parameters
    pub fn new(params: AnnealingParams) -> AnnealingResult<Self> {
        // Validate parameters
        params.validate()?;

        Ok(Self { params })
    }

    /// Create a new classical annealing simulator with default parameters
    pub fn with_default_params() -> AnnealingResult<Self> {
        Self::new(AnnealingParams::default())
    }
}

impl Default for ClassicalAnnealingSimulator {
    fn default() -> Self {
        Self::with_default_params().expect("Default parameters should be valid")
    }
}

impl ClassicalAnnealingSimulator {
    /// Find the ground state of an Ising model using classical simulated annealing
    pub fn solve(&self, model: &IsingModel) -> AnnealingResult<AnnealingSolution> {
        // Start timer
        let start_time = Instant::now();

        // Create random number generator
        let mut rng = match self.params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };

        // Initialize best result
        let num_qubits = model.num_qubits;
        let mut best_spins = vec![1; num_qubits]; // Start with all spins up
        let mut best_energy = match model.energy(&best_spins) {
            Ok(energy) => energy,
            Err(err) => return Err(AnnealingError::IsingError(err)),
        };

        // Track sweeps and repetitions
        let mut total_sweeps = 0;
        let mut completed_repetitions = 0;

        // Determine updates per sweep
        let updates_per_sweep = self.params.updates_per_sweep.unwrap_or(num_qubits);

        // Perform multiple repetitions
        for _ in 0..self.params.num_repetitions {
            // Initialize random spin configuration
            let mut spins = vec![0; num_qubits];
            for spin in &mut spins {
                *spin = if rng.random_bool(0.5) { 1 } else { -1 };
            }

            // Calculate initial energy
            let mut current_energy = match model.energy(&spins) {
                Ok(energy) => energy,
                Err(err) => return Err(AnnealingError::IsingError(err)),
            };

            // Perform simulated annealing
            for sweep in 0..self.params.num_sweeps {
                // Check timeout
                if let Some(timeout) = self.params.timeout {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    if elapsed > timeout {
                        return Err(AnnealingError::Timeout(Duration::from_secs_f64(elapsed)));
                    }
                }

                // Calculate current normalized time
                let t = sweep as f64 / self.params.num_sweeps as f64;
                let t_f = 1.0; // Final time (normalized)

                // Calculate current temperature
                let temperature = self.params.temperature_schedule.calculate(
                    t,
                    t_f,
                    self.params.initial_temperature,
                );

                // Perform Monte Carlo updates
                for _ in 0..updates_per_sweep {
                    // Choose a random qubit
                    let qubit = rng.random_range(0..num_qubits);

                    // Calculate energy change from flipping the spin
                    let current_spin = spins[qubit];
                    let new_spin = -current_spin;

                    // Temporary change to calculate energy difference
                    spins[qubit] = new_spin;

                    // Calculate new energy
                    let new_energy = match model.energy(&spins) {
                        Ok(energy) => energy,
                        Err(err) => return Err(AnnealingError::IsingError(err)),
                    };

                    let delta_e = new_energy - current_energy;

                    // Metropolis acceptance criterion
                    let accept = delta_e <= 0.0 || {
                        let p = (-delta_e / temperature).exp();
                        rng.random_range(0.0..1.0) < p
                    };

                    // Apply the spin flip if accepted
                    if accept {
                        current_energy = new_energy;
                    } else {
                        // Revert the change
                        spins[qubit] = current_spin;
                    }
                }

                // Increment sweep counter
                total_sweeps += 1;
            }

            // Check if this is a better solution
            if current_energy < best_energy {
                best_energy = current_energy;
                best_spins = spins.clone();
            }

            // Increment repetition counter
            completed_repetitions += 1;
        }

        // Calculate runtime
        let runtime = start_time.elapsed();

        // Build result
        Ok(AnnealingSolution {
            best_spins,
            best_energy,
            repetitions: completed_repetitions,
            total_sweeps,
            runtime,
            info: format!(
                "Performed {} repetitions with {} total sweeps in {:?}",
                completed_repetitions, total_sweeps, runtime
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::ising::QuboModel;

    #[test]
    fn test_annealing_params() {
        // Create default parameters
        let params = AnnealingParams::default();

        // Validate parameters
        assert!(params.validate().is_ok());

        // Test invalid parameters
        let mut invalid_params = params.clone();
        invalid_params.initial_temperature = 0.0;
        assert!(invalid_params.validate().is_err());

        let mut invalid_params = params.clone();
        invalid_params.num_sweeps = 0;
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_classical_annealing_simple() {
        // Create a simple 2-qubit ferromagnetic Ising model
        let mut model = IsingModel::new(2);
        model.set_coupling(0, 1, -1.0).unwrap(); // Ferromagnetic coupling

        // Create annealing simulator with fixed seed for reproducibility
        let mut params = AnnealingParams::default();
        params.seed = Some(42);
        params.num_sweeps = 100;
        params.num_repetitions = 5;

        let simulator = ClassicalAnnealingSimulator::new(params).unwrap();

        // Solve the model
        let result = simulator.solve(&model).unwrap();

        // Check that we found the ground state (all spins aligned)
        assert_eq!(result.best_spins.len(), 2);
        assert!(
            (result.best_spins[0] == 1 && result.best_spins[1] == 1)
                || (result.best_spins[0] == -1 && result.best_spins[1] == -1)
        );

        // Check energy
        assert_eq!(result.best_energy, -1.0);
    }

    #[test]
    fn test_quantum_annealing_simple() {
        // Create a simple 2-qubit ferromagnetic Ising model
        let mut model = IsingModel::new(2);
        model.set_coupling(0, 1, -1.0).unwrap(); // Ferromagnetic coupling

        // Create annealing simulator with fixed seed for reproducibility
        let mut params = AnnealingParams::default();
        params.seed = Some(42);
        params.num_sweeps = 100;
        params.num_repetitions = 5;
        params.trotter_slices = 10;

        let simulator = QuantumAnnealingSimulator::new(params).unwrap();

        // Solve the model
        let result = simulator.solve(&model).unwrap();

        // Check that we found the ground state (all spins aligned)
        assert_eq!(result.best_spins.len(), 2);
        assert!(
            (result.best_spins[0] == 1 && result.best_spins[1] == 1)
                || (result.best_spins[0] == -1 && result.best_spins[1] == -1)
        );

        // Check energy
        assert_eq!(result.best_energy, -1.0);
    }

    #[test]
    fn test_classical_annealing_frustrated() {
        // Create a 3-qubit frustrated Ising model
        let mut model = IsingModel::new(3);
        model.set_coupling(0, 1, -1.0).unwrap(); // Ferromagnetic coupling
        model.set_coupling(1, 2, -1.0).unwrap(); // Ferromagnetic coupling
        model.set_coupling(0, 2, 1.0).unwrap(); // Antiferromagnetic coupling

        // Create annealing simulator with fixed seed for reproducibility
        let mut params = AnnealingParams::default();
        params.seed = Some(42);
        params.num_sweeps = 200;
        params.num_repetitions = 10;

        let simulator = ClassicalAnnealingSimulator::new(params).unwrap();

        // Solve the model
        let result = simulator.solve(&model).unwrap();

        // Check energy (should be -1.0 for the ground state)
        assert!(result.best_energy <= -1.0);
    }
}
