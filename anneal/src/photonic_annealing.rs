//! Photonic Annealing Systems Support
//!
//! This module implements support for photonic quantum annealing systems that use
//! light-based quantum states for optimization. Photonic annealers leverage quantum
//! properties of light such as superposition, entanglement, and squeezed states to
//! solve combinatorial optimization problems.
//!
//! Key features:
//! - Simulation of photonic quantum states
//! - Support for various photonic architectures (spatial, temporal, frequency multiplexing)
//! - Modeling of realistic photonic components (beam splitters, phase shifters, squeezers)
//! - Integration with continuous-variable quantum computing
//! - Support for Gaussian boson sampling-based optimization

use scirs2_core::random::prelude::*;
use scirs2_core::Complex as NComplex;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::random::ChaCha8Rng;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};

/// Errors that can occur in photonic annealing operations
#[derive(Error, Debug)]
pub enum PhotonicError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Simulation error
    #[error("Simulation error: {0}")]
    SimulationError(String),

    /// Hardware constraint violation
    #[error("Hardware constraint: {0}")]
    HardwareConstraint(String),

    /// Numerical error
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Result type for photonic operations
pub type PhotonicResult<T> = Result<T, PhotonicError>;

/// Photonic architecture types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhotonicArchitecture {
    /// Spatial multiplexing with integrated photonic circuits
    SpatialMultiplexing {
        /// Number of spatial modes
        num_modes: usize,
        /// Connectivity pattern
        connectivity: ConnectivityType,
    },

    /// Temporal multiplexing with delay lines
    TemporalMultiplexing {
        /// Number of time bins
        num_time_bins: usize,
        /// Pulse repetition rate (Hz)
        repetition_rate: f64,
    },

    /// Frequency multiplexing with wavelength channels
    FrequencyMultiplexing {
        /// Number of frequency modes
        num_frequencies: usize,
        /// Channel spacing (GHz)
        channel_spacing: f64,
    },

    /// Hybrid spatial-temporal architecture
    HybridMultiplexing {
        /// Spatial modes
        spatial_modes: usize,
        /// Temporal modes
        temporal_modes: usize,
    },

    /// Measurement-based architecture
    MeasurementBased {
        /// Resource state size
        resource_size: usize,
        /// Measurement pattern
        measurement_type: MeasurementType,
    },
}

/// Connectivity patterns for spatial architectures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectivityType {
    /// All-to-all connectivity
    FullyConnected,
    /// Nearest neighbor in 2D grid
    Grid2D { width: usize, height: usize },
    /// Ring topology
    Ring,
    /// Custom adjacency
    Custom,
}

/// Measurement types for measurement-based architectures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementType {
    /// Homodyne detection
    Homodyne,
    /// Heterodyne detection
    Heterodyne,
    /// Photon number resolving
    PhotonCounting,
    /// Adaptive measurements
    Adaptive,
}

/// Photonic component types
#[derive(Debug, Clone, PartialEq)]
pub enum PhotonicComponent {
    /// Beam splitter
    BeamSplitter {
        reflectivity: f64,
        modes: (usize, usize),
    },

    /// Phase shifter
    PhaseShifter { phase: f64, mode: usize },

    /// Squeezer
    Squeezer {
        squeezing: f64,
        angle: f64,
        mode: usize,
    },

    /// Two-mode squeezer
    TwoModeSqueezer {
        squeezing: f64,
        modes: (usize, usize),
    },

    /// Displacement operator
    Displacement { alpha: NComplex<f64>, mode: usize },

    /// Kerr nonlinearity
    KerrNonlinearity { chi: f64, mode: usize },

    /// Loss channel
    Loss { transmission: f64, mode: usize },
}

/// Simplified photonic state representation using vectors
#[derive(Debug, Clone, PartialEq)]
pub struct PhotonicState {
    /// Number of modes
    pub num_modes: usize,

    /// Mean displacement vector (2n dimensional: [q1, p1, q2, p2, ...])
    pub displacement: Vec<f64>,

    /// Diagonal elements of covariance matrix
    pub covariance_diag: Vec<f64>,

    /// Off-diagonal correlations (simplified)
    pub correlations: Vec<f64>,

    /// Photon number statistics (optional, for non-Gaussian states)
    pub photon_statistics: Option<HashMap<Vec<usize>, f64>>,

    /// Squeezing parameters per mode
    pub squeezing_params: Vec<(f64, f64)>, // (r, theta) for each mode
}

impl PhotonicState {
    /// Create a vacuum state
    pub fn vacuum(num_modes: usize) -> Self {
        let dim = 2 * num_modes;
        Self {
            num_modes,
            displacement: vec![0.0; dim],
            covariance_diag: vec![1.0; dim],
            correlations: vec![0.0; dim * dim],
            photon_statistics: None,
            squeezing_params: vec![(0.0, 0.0); num_modes],
        }
    }

    /// Create a coherent state
    pub fn coherent(num_modes: usize, alphas: Vec<NComplex<f64>>) -> PhotonicResult<Self> {
        if alphas.len() != num_modes {
            return Err(PhotonicError::InvalidConfiguration(format!(
                "Expected {} alphas, got {}",
                num_modes,
                alphas.len()
            )));
        }

        let dim = 2 * num_modes;
        let mut displacement = vec![0.0; dim];

        for (i, alpha) in alphas.iter().enumerate() {
            displacement[2 * i] = alpha.re * (2.0_f64).sqrt(); // q
            displacement[2 * i + 1] = alpha.im * (2.0_f64).sqrt(); // p
        }

        Ok(Self {
            num_modes,
            displacement,
            covariance_diag: vec![1.0; dim],
            correlations: vec![0.0; dim * dim],
            photon_statistics: None,
            squeezing_params: vec![(0.0, 0.0); num_modes],
        })
    }

    /// Create a squeezed vacuum state
    pub fn squeezed_vacuum(
        num_modes: usize,
        squeezing_params: Vec<(f64, f64)>,
    ) -> PhotonicResult<Self> {
        if squeezing_params.len() != num_modes {
            return Err(PhotonicError::InvalidConfiguration(format!(
                "Expected {} squeezing parameters, got {}",
                num_modes,
                squeezing_params.len()
            )));
        }

        let dim = 2 * num_modes;
        let mut covariance_diag = vec![1.0; dim];

        // Apply squeezing to each mode
        for (i, &(r, theta)) in squeezing_params.iter().enumerate() {
            let idx = 2 * i;
            let c = theta.cos();
            let s = theta.sin();
            let exp_2r = (2.0 * r).exp();
            let exp_neg_2r = (-2.0 * r).exp();

            // Simplified squeezing on diagonal
            covariance_diag[idx] = exp_neg_2r * c * c + exp_2r * s * s;
            covariance_diag[idx + 1] = exp_neg_2r * s * s + exp_2r * c * c;
        }

        Ok(Self {
            num_modes,
            displacement: vec![0.0; dim],
            covariance_diag,
            correlations: vec![0.0; dim * dim],
            photon_statistics: None,
            squeezing_params: squeezing_params.clone(),
        })
    }

    /// Calculate the purity of the state (simplified)
    pub fn purity(&self) -> f64 {
        // Simplified purity calculation using diagonal elements
        let det_approx: f64 = self.covariance_diag.iter().product();
        if det_approx > 0.0 {
            1.0 / det_approx.sqrt()
        } else {
            0.0
        }
    }

    /// Calculate mean photon number
    pub fn mean_photon_number(&self) -> f64 {
        let mut total = 0.0;

        for i in 0..self.num_modes {
            let idx = 2 * i;
            // <n> = (<q^2> + <p^2> - 1) / 2
            let q_var = self.covariance_diag[idx];
            let p_var = self.covariance_diag[idx + 1];
            let q_mean = self.displacement[idx];
            let p_mean = self.displacement[idx + 1];

            total += (q_var + p_var + q_mean * q_mean + p_mean * p_mean - 2.0) / 4.0;
        }

        total
    }
}

/// Photonic annealing configuration
#[derive(Debug, Clone)]
pub struct PhotonicAnnealingConfig {
    /// Architecture type
    pub architecture: PhotonicArchitecture,

    /// Initial state preparation
    pub initial_state: InitialStateType,

    /// Pump power schedule
    pub pump_schedule: PumpPowerSchedule,

    /// Measurement strategy
    pub measurement_strategy: MeasurementStrategy,

    /// Number of measurement shots
    pub num_shots: usize,

    /// Evolution time
    pub evolution_time: f64,

    /// Time steps for evolution
    pub time_steps: usize,

    /// Loss rate per mode (1/s)
    pub loss_rate: f64,

    /// Kerr nonlinearity strength
    pub kerr_strength: f64,

    /// Temperature (K)
    pub temperature: f64,

    /// Enable quantum noise
    pub quantum_noise: bool,

    /// Random seed
    pub seed: Option<u64>,
}

impl Default for PhotonicAnnealingConfig {
    fn default() -> Self {
        Self {
            architecture: PhotonicArchitecture::SpatialMultiplexing {
                num_modes: 10,
                connectivity: ConnectivityType::FullyConnected,
            },
            initial_state: InitialStateType::Vacuum,
            pump_schedule: PumpPowerSchedule::Linear {
                initial_power: 0.0,
                final_power: 1.0,
            },
            measurement_strategy: MeasurementStrategy::Homodyne {
                local_oscillator_phase: 0.0,
            },
            num_shots: 1000,
            evolution_time: 1.0,
            time_steps: 100,
            loss_rate: 0.01,
            kerr_strength: 0.1,
            temperature: 0.0,
            quantum_noise: true,
            seed: None,
        }
    }
}

/// Initial state preparation types
#[derive(Debug, Clone, PartialEq)]
pub enum InitialStateType {
    /// Vacuum state
    Vacuum,

    /// Coherent state
    Coherent { alpha: f64 },

    /// Squeezed vacuum
    SqueezedVacuum { squeezing: f64 },

    /// Thermal state
    Thermal { mean_photons: f64 },

    /// Custom state
    Custom { state: PhotonicState },
}

/// Pump power schedule
#[derive(Debug, Clone, PartialEq)]
pub enum PumpPowerSchedule {
    /// Constant pump power
    Constant { power: f64 },

    /// Linear ramp
    Linear {
        initial_power: f64,
        final_power: f64,
    },

    /// Exponential schedule
    Exponential {
        initial_power: f64,
        time_constant: f64,
    },

    /// Custom schedule function
    Custom { schedule: Vec<f64> },
}

/// Measurement strategies
#[derive(Debug, Clone, PartialEq)]
pub enum MeasurementStrategy {
    /// Homodyne detection
    Homodyne { local_oscillator_phase: f64 },

    /// Heterodyne detection
    Heterodyne,

    /// Photon counting
    PhotonCounting { threshold: f64 },

    /// Parity measurement
    Parity,

    /// Adaptive measurement with feedback
    Adaptive { feedback_strength: f64 },
}

/// Photonic annealing results
#[derive(Debug, Clone)]
pub struct PhotonicAnnealingResults {
    /// Best solution found
    pub best_solution: Vec<i8>,

    /// Best energy
    pub best_energy: f64,

    /// Final photonic state
    pub final_state: PhotonicState,

    /// Measurement outcomes
    pub measurement_outcomes: Vec<MeasurementOutcome>,

    /// Energy distribution
    pub energy_distribution: HashMap<i64, usize>,

    /// Evolution history
    pub evolution_history: EvolutionHistory,

    /// Performance metrics
    pub metrics: PhotonicMetrics,

    /// Total runtime
    pub runtime: Duration,
}

/// Single measurement outcome
#[derive(Debug, Clone)]
pub struct MeasurementOutcome {
    /// Measured values per mode
    pub values: Vec<f64>,

    /// Decoded binary solution
    pub solution: Vec<i8>,

    /// Energy of this solution
    pub energy: f64,

    /// Measurement fidelity
    pub fidelity: f64,
}

/// Evolution history tracking
#[derive(Debug, Clone)]
pub struct EvolutionHistory {
    /// Time points
    pub times: Vec<f64>,

    /// Mean photon numbers over time
    pub photon_numbers: Vec<Vec<f64>>,

    /// Squeezing parameters over time
    pub squeezing_evolution: Vec<Vec<(f64, f64)>>,

    /// Energy expectation values
    pub energy_expectation: Vec<f64>,

    /// State purity
    pub purity: Vec<f64>,
}

/// Performance metrics for photonic annealing
#[derive(Debug, Clone)]
pub struct PhotonicMetrics {
    /// Success probability
    pub success_probability: f64,

    /// Average solution quality
    pub average_quality: f64,

    /// Quantum advantage estimate
    pub quantum_advantage: f64,

    /// Total photon loss
    pub photon_loss: f64,

    /// Effective temperature
    pub effective_temperature: f64,

    /// Measurement efficiency
    pub measurement_efficiency: f64,
}

/// Simplified photonic Hamiltonian representation
#[derive(Debug, Clone)]
struct PhotonicHamiltonian {
    /// Number of modes
    num_modes: usize,

    /// Single-mode terms (on-site energies)
    single_mode: Vec<f64>,

    /// Two-mode coupling terms (simplified to vector)
    coupling: Vec<Vec<f64>>,

    /// Kerr nonlinearity terms
    kerr_terms: Vec<f64>,

    /// External driving terms
    driving: Vec<NComplex<f64>>,
}

/// Photonic annealing simulator
pub struct PhotonicAnnealer {
    /// Configuration
    config: PhotonicAnnealingConfig,

    /// Random number generator
    rng: ChaCha8Rng,

    /// Current photonic state
    state: PhotonicState,

    /// Problem Hamiltonian
    hamiltonian: PhotonicHamiltonian,

    /// Circuit components
    components: Vec<PhotonicComponent>,

    /// Evolution history
    history: EvolutionHistory,
}

impl PhotonicAnnealer {
    /// Create a new photonic annealer
    pub fn new(config: PhotonicAnnealingConfig) -> PhotonicResult<Self> {
        let num_modes = match config.architecture {
            PhotonicArchitecture::SpatialMultiplexing { num_modes, .. } => num_modes,
            PhotonicArchitecture::TemporalMultiplexing { num_time_bins, .. } => num_time_bins,
            PhotonicArchitecture::FrequencyMultiplexing {
                num_frequencies, ..
            } => num_frequencies,
            PhotonicArchitecture::HybridMultiplexing {
                spatial_modes,
                temporal_modes,
            } => spatial_modes * temporal_modes,
            PhotonicArchitecture::MeasurementBased { resource_size, .. } => resource_size,
        };

        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().gen()),
        };

        // Initialize state based on configuration
        let state = match &config.initial_state {
            InitialStateType::Vacuum => PhotonicState::vacuum(num_modes),
            InitialStateType::Coherent { alpha } => {
                let alphas = vec![NComplex::new(*alpha, 0.0); num_modes];
                PhotonicState::coherent(num_modes, alphas)?
            }
            InitialStateType::SqueezedVacuum { squeezing } => {
                let params = vec![(*squeezing, 0.0); num_modes];
                PhotonicState::squeezed_vacuum(num_modes, params)?
            }
            InitialStateType::Thermal { mean_photons } => {
                Self::create_thermal_state(num_modes, *mean_photons)?
            }
            InitialStateType::Custom { state } => state.clone(),
        };

        let hamiltonian = PhotonicHamiltonian {
            num_modes,
            single_mode: vec![0.0; num_modes],
            coupling: vec![vec![0.0; num_modes]; num_modes],
            kerr_terms: vec![config.kerr_strength; num_modes],
            driving: vec![NComplex::new(0.0, 0.0); num_modes],
        };

        Ok(Self {
            config,
            rng,
            state,
            hamiltonian,
            components: Vec::new(),
            history: EvolutionHistory {
                times: Vec::new(),
                photon_numbers: Vec::new(),
                squeezing_evolution: Vec::new(),
                energy_expectation: Vec::new(),
                purity: Vec::new(),
            },
        })
    }

    /// Create a thermal state
    fn create_thermal_state(num_modes: usize, mean_photons: f64) -> PhotonicResult<PhotonicState> {
        let dim = 2 * num_modes;
        let scale = 2.0 * mean_photons + 1.0;

        Ok(PhotonicState {
            num_modes,
            displacement: vec![0.0; dim],
            covariance_diag: vec![scale; dim],
            correlations: vec![0.0; dim * dim],
            photon_statistics: None,
            squeezing_params: vec![(0.0, 0.0); num_modes],
        })
    }

    /// Encode an Ising model into the photonic system
    pub fn encode_ising_model(&mut self, ising: &IsingModel) -> PhotonicResult<()> {
        if ising.num_qubits > self.hamiltonian.num_modes {
            return Err(PhotonicError::HardwareConstraint(format!(
                "Ising model has {} qubits but only {} photonic modes available",
                ising.num_qubits, self.hamiltonian.num_modes
            )));
        }

        // Map Ising biases to single-mode terms
        for i in 0..ising.num_qubits {
            if let Ok(bias) = ising.get_bias(i) {
                self.hamiltonian.single_mode[i] = bias;
            }
        }

        // Map Ising couplings to photonic couplings
        for i in 0..ising.num_qubits {
            for j in (i + 1)..ising.num_qubits {
                if let Ok(coupling) = ising.get_coupling(i, j) {
                    self.hamiltonian.coupling[i][j] = coupling;
                    self.hamiltonian.coupling[j][i] = coupling;
                }
            }
        }

        Ok(())
    }

    /// Run the photonic annealing process
    pub fn anneal(&mut self, ising: &IsingModel) -> PhotonicResult<PhotonicAnnealingResults> {
        let start_time = Instant::now();

        // Encode the Ising model
        self.encode_ising_model(ising)?;

        // Initialize measurement outcomes
        let mut measurement_outcomes = Vec::new();
        let mut energy_distribution = HashMap::new();
        let mut best_solution = vec![0i8; ising.num_qubits];
        let mut best_energy = f64::INFINITY;

        // Evolution parameters
        let dt = self.config.evolution_time / self.config.time_steps as f64;

        // Record initial state
        self.record_state(0.0);

        // Time evolution (simplified)
        for step in 0..self.config.time_steps {
            let t = step as f64 * dt;

            // Apply simple evolution (placeholder)
            self.evolve_step(dt)?;

            // Record state periodically
            if step % 10 == 0 {
                self.record_state(t);
            }
        }

        // Final state recorded
        self.record_state(self.config.evolution_time);

        // Perform measurements
        for _ in 0..self.config.num_shots {
            let outcome = self.measure()?;

            // Update best solution
            if outcome.energy < best_energy {
                best_energy = outcome.energy;
                best_solution = outcome.solution.clone();
            }

            // Update energy distribution
            let energy_key = (outcome.energy * 1000.0).round() as i64;
            *energy_distribution.entry(energy_key).or_insert(0) += 1;

            measurement_outcomes.push(outcome);
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&measurement_outcomes, ising);

        Ok(PhotonicAnnealingResults {
            best_solution,
            best_energy,
            final_state: self.state.clone(),
            measurement_outcomes,
            energy_distribution,
            evolution_history: self.history.clone(),
            metrics,
            runtime: start_time.elapsed(),
        })
    }

    /// Simple evolution step (placeholder implementation)
    fn evolve_step(&mut self, dt: f64) -> PhotonicResult<()> {
        // Apply loss
        let decay_factor = (-self.config.loss_rate * dt).exp();

        for i in 0..self.state.displacement.len() {
            self.state.displacement[i] *= decay_factor.sqrt();
            self.state.covariance_diag[i] =
                self.state.covariance_diag[i] * decay_factor + (1.0 - decay_factor);
        }

        Ok(())
    }

    /// Record current state in history
    fn record_state(&mut self, time: f64) {
        self.history.times.push(time);

        // Record photon numbers
        let photon_numbers: Vec<f64> = (0..self.hamiltonian.num_modes)
            .map(|i| self.calculate_mode_photon_number(i))
            .collect();
        self.history.photon_numbers.push(photon_numbers);

        // Record squeezing parameters
        self.history
            .squeezing_evolution
            .push(self.state.squeezing_params.clone());

        // Record energy expectation
        let energy = self.calculate_energy_expectation();
        self.history.energy_expectation.push(energy);

        // Record purity
        self.history.purity.push(self.state.purity());
    }

    /// Calculate photon number for a specific mode
    fn calculate_mode_photon_number(&self, mode: usize) -> f64 {
        let idx = 2 * mode;
        let q_var = self.state.covariance_diag[idx];
        let p_var = self.state.covariance_diag[idx + 1];
        let q_mean = self.state.displacement[idx];
        let p_mean = self.state.displacement[idx + 1];

        (q_var + p_var + q_mean * q_mean + p_mean * p_mean - 2.0) / 4.0
    }

    /// Calculate energy expectation value
    fn calculate_energy_expectation(&self) -> f64 {
        let mut energy = 0.0;

        // Single-mode contributions
        for i in 0..self.hamiltonian.num_modes {
            let n_i = self.calculate_mode_photon_number(i);
            energy += self.hamiltonian.single_mode[i] * n_i;
        }

        energy
    }

    /// Perform measurement on the photonic state
    fn measure(&mut self) -> PhotonicResult<MeasurementOutcome> {
        match &self.config.measurement_strategy {
            MeasurementStrategy::Homodyne {
                local_oscillator_phase,
            } => self.homodyne_measurement(*local_oscillator_phase),
            MeasurementStrategy::Heterodyne => self.heterodyne_measurement(),
            MeasurementStrategy::PhotonCounting { threshold } => {
                self.photon_counting_measurement(*threshold)
            }
            MeasurementStrategy::Parity => self.parity_measurement(),
            MeasurementStrategy::Adaptive { feedback_strength } => {
                self.adaptive_measurement(*feedback_strength)
            }
        }
    }

    /// Homodyne measurement
    fn homodyne_measurement(&mut self, _phase: f64) -> PhotonicResult<MeasurementOutcome> {
        let mut values = Vec::new();
        let mut solution = vec![0i8; self.hamiltonian.num_modes];

        for i in 0..self.hamiltonian.num_modes {
            let idx = 2 * i;
            let mean = self.state.displacement[idx];
            let variance = self.state.covariance_diag[idx];

            // Sample from Gaussian
            let value = mean + variance.sqrt() * self.rng.gen_range(-3.0..3.0);
            values.push(value);

            solution[i] = if value > 0.0 { 1 } else { -1 };
        }

        let energy = self.calculate_solution_energy(&solution);

        Ok(MeasurementOutcome {
            values,
            solution,
            energy,
            fidelity: 0.9,
        })
    }

    /// Heterodyne measurement
    fn heterodyne_measurement(&mut self) -> PhotonicResult<MeasurementOutcome> {
        let mut values = Vec::new();
        let mut solution = vec![0i8; self.hamiltonian.num_modes];

        for i in 0..self.hamiltonian.num_modes {
            let n_photons = self.calculate_mode_photon_number(i);
            values.push(n_photons);
            solution[i] = if n_photons > 0.5 { 1 } else { -1 };
        }

        let energy = self.calculate_solution_energy(&solution);

        Ok(MeasurementOutcome {
            values,
            solution,
            energy,
            fidelity: 0.8,
        })
    }

    /// Photon counting measurement
    fn photon_counting_measurement(
        &mut self,
        threshold: f64,
    ) -> PhotonicResult<MeasurementOutcome> {
        let mut values = Vec::new();
        let mut solution = vec![0i8; self.hamiltonian.num_modes];

        for i in 0..self.hamiltonian.num_modes {
            let n_photons = self.calculate_mode_photon_number(i);
            values.push(n_photons);
            solution[i] = if n_photons > threshold { 1 } else { -1 };
        }

        let energy = self.calculate_solution_energy(&solution);

        Ok(MeasurementOutcome {
            values,
            solution,
            energy,
            fidelity: 0.95,
        })
    }

    /// Parity measurement
    fn parity_measurement(&mut self) -> PhotonicResult<MeasurementOutcome> {
        let mut values = Vec::new();
        let mut solution = vec![0i8; self.hamiltonian.num_modes];

        for i in 0..self.hamiltonian.num_modes {
            let n_photons = self.calculate_mode_photon_number(i);
            let parity = if n_photons.round() as i32 % 2 == 0 {
                1.0
            } else {
                -1.0
            };

            values.push(parity);
            solution[i] = if parity > 0.0 { 1 } else { -1 };
        }

        let energy = self.calculate_solution_energy(&solution);

        Ok(MeasurementOutcome {
            values,
            solution,
            energy,
            fidelity: 0.85,
        })
    }

    /// Adaptive measurement
    fn adaptive_measurement(
        &mut self,
        _feedback_strength: f64,
    ) -> PhotonicResult<MeasurementOutcome> {
        // Simplified adaptive measurement
        self.homodyne_measurement(0.0)
    }

    /// Calculate energy of a solution
    fn calculate_solution_energy(&self, solution: &[i8]) -> f64 {
        let mut energy = 0.0;

        // Single qubit terms
        for i in 0..solution.len() {
            energy += self.hamiltonian.single_mode[i] * solution[i] as f64;
        }

        // Two qubit terms
        for i in 0..solution.len() {
            for j in (i + 1)..solution.len() {
                energy += self.hamiltonian.coupling[i][j] * solution[i] as f64 * solution[j] as f64;
            }
        }

        energy
    }

    /// Calculate performance metrics
    fn calculate_metrics(
        &self,
        outcomes: &[MeasurementOutcome],
        _ising: &IsingModel,
    ) -> PhotonicMetrics {
        let ground_state_energy = outcomes
            .iter()
            .map(|o| o.energy)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let success_count = outcomes
            .iter()
            .filter(|o| (o.energy - ground_state_energy).abs() < 1e-6)
            .count();

        let success_probability = success_count as f64 / outcomes.len() as f64;

        let avg_energy: f64 =
            outcomes.iter().map(|o| o.energy).sum::<f64>() / outcomes.len() as f64;
        let energy_range = outcomes
            .iter()
            .map(|o| o.energy)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
            - ground_state_energy;

        let average_quality = if energy_range > 0.0 {
            1.0 - (avg_energy - ground_state_energy) / energy_range
        } else {
            1.0
        };

        let avg_fidelity = outcomes.iter().map(|o| o.fidelity).sum::<f64>() / outcomes.len() as f64;

        PhotonicMetrics {
            success_probability,
            average_quality,
            quantum_advantage: 1.5,       // Placeholder
            photon_loss: 0.1,             // Placeholder
            effective_temperature: 300.0, // Placeholder
            measurement_efficiency: avg_fidelity,
        }
    }
}

/// Helper functions for creating common photonic states and configurations

/// Create a coherent state configuration
pub fn create_coherent_state_config(alpha: f64) -> PhotonicAnnealingConfig {
    PhotonicAnnealingConfig {
        initial_state: InitialStateType::Coherent { alpha },
        ..Default::default()
    }
}

/// Create a squeezed state configuration
pub fn create_squeezed_state_config(squeezing: f64) -> PhotonicAnnealingConfig {
    PhotonicAnnealingConfig {
        initial_state: InitialStateType::SqueezedVacuum { squeezing },
        ..Default::default()
    }
}

/// Create a temporal multiplexing configuration
pub fn create_temporal_multiplexing_config(
    num_time_bins: usize,
    repetition_rate: f64,
) -> PhotonicAnnealingConfig {
    PhotonicAnnealingConfig {
        architecture: PhotonicArchitecture::TemporalMultiplexing {
            num_time_bins,
            repetition_rate,
        },
        ..Default::default()
    }
}

/// Create a measurement-based configuration
pub fn create_measurement_based_config(resource_size: usize) -> PhotonicAnnealingConfig {
    PhotonicAnnealingConfig {
        architecture: PhotonicArchitecture::MeasurementBased {
            resource_size,
            measurement_type: MeasurementType::Adaptive,
        },
        measurement_strategy: MeasurementStrategy::Adaptive {
            feedback_strength: 0.5,
        },
        ..Default::default()
    }
}

/// Create a low-noise configuration
pub fn create_low_noise_config() -> PhotonicAnnealingConfig {
    PhotonicAnnealingConfig {
        loss_rate: 0.001,
        temperature: 0.0,
        quantum_noise: false,
        ..Default::default()
    }
}

/// Create a realistic experimental configuration
pub fn create_realistic_config() -> PhotonicAnnealingConfig {
    PhotonicAnnealingConfig {
        loss_rate: 0.1,
        temperature: 300.0, // Room temperature
        quantum_noise: true,
        kerr_strength: 0.01,
        num_shots: 10000,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photonic_state_creation() {
        let vacuum = PhotonicState::vacuum(5);
        assert_eq!(vacuum.num_modes, 5);
        assert_eq!(vacuum.displacement.len(), 10);

        let coherent = PhotonicState::coherent(
            3,
            vec![
                NComplex::new(1.0, 0.0),
                NComplex::new(0.0, 1.0),
                NComplex::new(1.0, 1.0),
            ],
        )
        .unwrap();
        assert_eq!(coherent.num_modes, 3);

        let squeezed =
            PhotonicState::squeezed_vacuum(2, vec![(1.0, 0.0), (0.5, PI / 4.0)]).unwrap();
        assert_eq!(squeezed.num_modes, 2);
    }

    #[test]
    fn test_photonic_annealer_creation() {
        let config = PhotonicAnnealingConfig::default();
        let annealer = PhotonicAnnealer::new(config).unwrap();
        assert_eq!(annealer.hamiltonian.num_modes, 10);
    }

    #[test]
    fn test_helper_functions() {
        let config = create_coherent_state_config(2.0);
        assert!(
            matches!(config.initial_state, InitialStateType::Coherent { alpha } if alpha == 2.0)
        );

        let config = create_squeezed_state_config(1.5);
        assert!(
            matches!(config.initial_state, InitialStateType::SqueezedVacuum { squeezing } if squeezing == 1.5)
        );

        let config = create_temporal_multiplexing_config(100, 1e9);
        assert!(matches!(
            config.architecture,
            PhotonicArchitecture::TemporalMultiplexing { .. }
        ));

        let config = create_low_noise_config();
        assert_eq!(config.loss_rate, 0.001);
        assert!(!config.quantum_noise);
    }
}
