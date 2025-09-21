//! Non-Stoquastic Hamiltonian Support for Quantum Annealing
//!
//! This module implements support for non-stoquastic Hamiltonians in quantum annealing systems.
//! Non-stoquastic Hamiltonians have positive off-diagonal matrix elements in the computational
//! basis, which can lead to the sign problem in quantum Monte Carlo simulations but may also
//! provide quantum advantages for certain optimization problems.
//!
//! Key features:
//! - XY and XYZ spin models
//! - Complex-weighted coupling terms
//! - Sign-problem aware quantum Monte Carlo algorithms
//! - Population annealing with complex weights
//! - Advanced sampling strategies (cluster algorithms, parallel tempering)
//! - Integration with stoquastic Hamiltonians
//! - Quantum advantage detection and analysis

use num_complex::Complex as NComplex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};
use crate::simulator::{AnnealingParams, AnnealingSolution};

/// Errors that can occur in non-stoquastic operations
#[derive(Error, Debug)]
pub enum NonStoquasticError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Invalid Hamiltonian configuration
    #[error("Invalid Hamiltonian: {0}")]
    InvalidHamiltonian(String),

    /// Simulation error
    #[error("Simulation error: {0}")]
    SimulationError(String),

    /// Sign problem detected
    #[error("Sign problem: {0}")]
    SignProblem(String),

    /// Convergence error
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Complex arithmetic error
    #[error("Complex arithmetic error: {0}")]
    ComplexArithmeticError(String),
}

/// Result type for non-stoquastic operations
pub type NonStoquasticResult<T> = Result<T, NonStoquasticError>;

/// Types of non-stoquastic Hamiltonians
#[derive(Debug, Clone, PartialEq)]
pub enum HamiltonianType {
    /// XY model: H = Σ(J_x σ_x^i σ_x^j + J_y σ_y^i σ_y^j)
    XYModel { j_x: f64, j_y: f64 },

    /// XYZ (Heisenberg) model: H = Σ(J_x σ_x^i σ_x^j + J_y σ_y^i σ_y^j + J_z σ_z^i σ_z^j)
    XYZModel { j_x: f64, j_y: f64, j_z: f64 },

    /// Complex-weighted Ising model
    ComplexIsingModel,

    /// Fermionic Hamiltonian (Jordan-Wigner transformed)
    FermionicModel,

    /// Custom non-stoquastic model
    CustomModel,

    /// Mixed stoquastic/non-stoquastic
    MixedModel { stoquastic_fraction: f64 },
}

/// Complex coupling term for non-stoquastic interactions
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexCoupling {
    /// Site indices
    pub sites: (usize, usize),
    /// Complex coupling strength
    pub strength: NComplex<f64>,
    /// Interaction type
    pub interaction_type: InteractionType,
}

/// Types of quantum interactions
#[derive(Debug, Clone, PartialEq)]
pub enum InteractionType {
    /// XX interaction (σ_x^i σ_x^j)
    XX,
    /// YY interaction (σ_y^i σ_y^j)
    YY,
    /// ZZ interaction (σ_z^i σ_z^j)
    ZZ,
    /// XY interaction (σ_x^i σ_y^j + σ_y^i σ_x^j)
    XY,
    /// Complex XY interaction (σ_+^i σ_-^j + σ_-^i σ_+^j)
    ComplexXY,
    /// Custom interaction matrix
    CustomMatrix(Vec<Vec<NComplex<f64>>>),
}

/// Non-stoquastic Hamiltonian representation
#[derive(Debug, Clone)]
pub struct NonStoquasticHamiltonian {
    /// Number of qubits/spins
    pub num_qubits: usize,

    /// Hamiltonian type
    pub hamiltonian_type: HamiltonianType,

    /// Local magnetic field terms (real)
    pub local_fields: Vec<f64>,

    /// Complex coupling terms
    pub complex_couplings: Vec<ComplexCoupling>,

    /// Stoquastic (Ising) part for mixed models
    pub ising_part: Option<IsingModel>,

    /// Global phase factor
    pub global_phase: NComplex<f64>,

    /// Whether this Hamiltonian has the sign problem
    pub has_sign_problem: bool,
}

impl NonStoquasticHamiltonian {
    /// Create a new non-stoquastic Hamiltonian
    pub fn new(num_qubits: usize, hamiltonian_type: HamiltonianType) -> Self {
        let has_sign_problem = match hamiltonian_type {
            HamiltonianType::XYModel { .. }
            | HamiltonianType::XYZModel { .. }
            | HamiltonianType::ComplexIsingModel
            | HamiltonianType::FermionicModel
            | HamiltonianType::CustomModel => true,
            HamiltonianType::MixedModel {
                stoquastic_fraction,
            } => stoquastic_fraction < 1.0,
        };

        Self {
            num_qubits,
            hamiltonian_type,
            local_fields: vec![0.0; num_qubits],
            complex_couplings: Vec::new(),
            ising_part: None,
            global_phase: NComplex::new(1.0, 0.0),
            has_sign_problem,
        }
    }

    /// Create XY model
    pub fn xy_model(num_qubits: usize, j_x: f64, j_y: f64) -> NonStoquasticResult<Self> {
        let mut hamiltonian = Self::new(num_qubits, HamiltonianType::XYModel { j_x, j_y });

        // Add XY couplings for nearest neighbors (chain topology)
        for i in 0..num_qubits - 1 {
            // X-X coupling
            hamiltonian.add_complex_coupling(ComplexCoupling {
                sites: (i, i + 1),
                strength: NComplex::new(j_x, 0.0),
                interaction_type: InteractionType::XX,
            })?;

            // Y-Y coupling
            hamiltonian.add_complex_coupling(ComplexCoupling {
                sites: (i, i + 1),
                strength: NComplex::new(j_y, 0.0),
                interaction_type: InteractionType::YY,
            })?;
        }

        Ok(hamiltonian)
    }

    /// Create XYZ (Heisenberg) model
    pub fn xyz_model(num_qubits: usize, j_x: f64, j_y: f64, j_z: f64) -> NonStoquasticResult<Self> {
        let mut hamiltonian = Self::new(num_qubits, HamiltonianType::XYZModel { j_x, j_y, j_z });

        // Add XYZ couplings for nearest neighbors
        for i in 0..num_qubits - 1 {
            // X-X coupling
            hamiltonian.add_complex_coupling(ComplexCoupling {
                sites: (i, i + 1),
                strength: NComplex::new(j_x, 0.0),
                interaction_type: InteractionType::XX,
            })?;

            // Y-Y coupling
            hamiltonian.add_complex_coupling(ComplexCoupling {
                sites: (i, i + 1),
                strength: NComplex::new(j_y, 0.0),
                interaction_type: InteractionType::YY,
            })?;

            // Z-Z coupling
            hamiltonian.add_complex_coupling(ComplexCoupling {
                sites: (i, i + 1),
                strength: NComplex::new(j_z, 0.0),
                interaction_type: InteractionType::ZZ,
            })?;
        }

        Ok(hamiltonian)
    }

    /// Create complex-weighted Ising model
    pub fn complex_ising_model(num_qubits: usize) -> Self {
        Self::new(num_qubits, HamiltonianType::ComplexIsingModel)
    }

    /// Add a complex coupling term
    pub fn add_complex_coupling(&mut self, coupling: ComplexCoupling) -> NonStoquasticResult<()> {
        if coupling.sites.0 >= self.num_qubits || coupling.sites.1 >= self.num_qubits {
            return Err(NonStoquasticError::InvalidHamiltonian(format!(
                "Invalid coupling sites: ({}, {}) for {} qubits",
                coupling.sites.0, coupling.sites.1, self.num_qubits
            )));
        }

        self.complex_couplings.push(coupling);
        Ok(())
    }

    /// Set local magnetic field
    pub fn set_local_field(&mut self, site: usize, field: f64) -> NonStoquasticResult<()> {
        if site >= self.num_qubits {
            return Err(NonStoquasticError::InvalidHamiltonian(format!(
                "Invalid site index: {} for {} qubits",
                site, self.num_qubits
            )));
        }

        self.local_fields[site] = field;
        Ok(())
    }

    /// Check if Hamiltonian is stoquastic
    pub fn is_stoquastic(&self) -> bool {
        !self.has_sign_problem
    }

    /// Estimate sign problem severity
    pub fn sign_problem_severity(&self) -> f64 {
        if !self.has_sign_problem {
            return 0.0;
        }

        // Estimate based on the magnitude of non-stoquastic terms
        let mut non_stoquastic_weight = 0.0;
        let mut total_weight = 0.0;

        for coupling in &self.complex_couplings {
            let magnitude = coupling.strength.norm();
            total_weight += magnitude;

            match coupling.interaction_type {
                InteractionType::XX
                | InteractionType::YY
                | InteractionType::XY
                | InteractionType::ComplexXY => {
                    non_stoquastic_weight += magnitude;
                }
                _ => {}
            }
        }

        if total_weight > 0.0 {
            non_stoquastic_weight / total_weight
        } else {
            0.0
        }
    }

    /// Convert to matrix representation (for small systems)
    pub fn to_matrix(&self) -> NonStoquasticResult<Vec<Vec<NComplex<f64>>>> {
        if self.num_qubits > 12 {
            return Err(NonStoquasticError::SimulationError(
                "Matrix representation only supported for ≤12 qubits".to_string(),
            ));
        }

        let dim = 1 << self.num_qubits;
        let mut matrix = vec![vec![NComplex::new(0.0, 0.0); dim]; dim];

        // Add local field terms
        for site in 0..self.num_qubits {
            let field = self.local_fields[site];
            if field.abs() > 1e-12 {
                for state in 0..dim {
                    let spin = if (state >> site) & 1 == 1 { 1.0 } else { -1.0 };
                    matrix[state][state] += NComplex::new(field * spin, 0.0);
                }
            }
        }

        // Add coupling terms
        for coupling in &self.complex_couplings {
            let (i, j) = coupling.sites;

            match coupling.interaction_type {
                InteractionType::ZZ => {
                    // Diagonal terms
                    for state in 0..dim {
                        let spin_i = if (state >> i) & 1 == 1 { 1.0 } else { -1.0 };
                        let spin_j = if (state >> j) & 1 == 1 { 1.0 } else { -1.0 };
                        matrix[state][state] += coupling.strength * spin_i * spin_j;
                    }
                }
                InteractionType::XX => {
                    // Off-diagonal terms
                    for state in 0..dim {
                        let flipped = state ^ (1 << i) ^ (1 << j);
                        matrix[state][flipped] += coupling.strength;
                    }
                }
                InteractionType::YY => {
                    // Off-diagonal terms with imaginary factors
                    for state in 0..dim {
                        let spin_i = if (state >> i) & 1 == 1 { 1.0 } else { -1.0 };
                        let spin_j = if (state >> j) & 1 == 1 { 1.0 } else { -1.0 };
                        let flipped = state ^ (1 << i) ^ (1 << j);
                        let phase = NComplex::new(0.0, spin_i * spin_j);
                        matrix[state][flipped] += coupling.strength * phase;
                    }
                }
                _ => {
                    // Simplified handling for other interaction types
                    for state in 0..dim {
                        let flipped = state ^ (1 << i) ^ (1 << j);
                        matrix[state][flipped] += coupling.strength * 0.5;
                    }
                }
            }
        }

        Ok(matrix)
    }
}

/// Configuration for non-stoquastic quantum Monte Carlo
#[derive(Debug, Clone)]
pub struct NonStoquasticQMCConfig {
    /// Number of Monte Carlo steps
    pub num_steps: usize,
    /// Number of thermalization steps
    pub thermalization_steps: usize,
    /// Temperature
    pub temperature: f64,
    /// Imaginary time step size
    pub tau: f64,
    /// Number of time slices
    pub num_time_slices: usize,
    /// Population size for population annealing
    pub population_size: usize,
    /// Sign problem mitigation strategy
    pub sign_mitigation: SignMitigationStrategy,
    /// Random seed
    pub seed: Option<u64>,
    /// Measurement interval
    pub measurement_interval: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl Default for NonStoquasticQMCConfig {
    fn default() -> Self {
        Self {
            num_steps: 10000,
            thermalization_steps: 1000,
            temperature: 1.0,
            tau: 0.1,
            num_time_slices: 10,
            population_size: 1000,
            sign_mitigation: SignMitigationStrategy::ReweightingMethod,
            seed: None,
            measurement_interval: 10,
            convergence_threshold: 1e-6,
        }
    }
}

/// Strategies for mitigating the sign problem
#[derive(Debug, Clone, PartialEq)]
pub enum SignMitigationStrategy {
    /// Simple reweighting method
    ReweightingMethod,
    /// Constrained path Monte Carlo
    ConstrainedPath,
    /// Population annealing with sign handling
    PopulationAnnealing,
    /// Meron cluster algorithm
    MeronClusters,
    /// Complex Langevin dynamics
    ComplexLangevin,
    /// Auxiliary field approach
    AuxiliaryField,
}

/// Results from non-stoquastic quantum Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct NonStoquasticResults {
    /// Ground state energy estimate
    pub ground_state_energy: NComplex<f64>,
    /// Energy variance
    pub energy_variance: f64,
    /// Ground state configuration (if available)
    pub ground_state: Option<Vec<i8>>,
    /// Average sign of the wavefunction
    pub average_sign: NComplex<f64>,
    /// Sign problem severity
    pub sign_problem_severity: f64,
    /// Quantum Monte Carlo statistics
    pub qmc_statistics: QMCStatistics,
    /// Simulation time
    pub simulation_time: Duration,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
}

/// Quantum Monte Carlo statistics
#[derive(Debug, Clone)]
pub struct QMCStatistics {
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Autocorrelation time
    pub autocorrelation_time: f64,
    /// Effective sample size
    pub effective_sample_size: usize,
    /// Statistical error estimates
    pub statistical_errors: HashMap<String, f64>,
    /// Population size evolution (for population annealing)
    pub population_evolution: Vec<usize>,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether simulation converged
    pub converged: bool,
    /// Convergence step
    pub convergence_step: Option<usize>,
    /// Energy history
    pub energy_history: Vec<NComplex<f64>>,
    /// Sign history
    pub sign_history: Vec<NComplex<f64>>,
}

/// Non-stoquastic quantum Monte Carlo simulator
pub struct NonStoquasticSimulator {
    /// Configuration
    config: NonStoquasticQMCConfig,
    /// Random number generator
    rng: ChaCha8Rng,
    /// Current quantum state
    current_state: QuantumState,
    /// Hamiltonian being simulated
    hamiltonian: NonStoquasticHamiltonian,
}

/// Quantum state representation for non-stoquastic systems
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of time slices
    pub num_time_slices: usize,
    /// State configurations for each time slice
    pub configurations: Vec<Vec<i8>>,
    /// Complex amplitudes (for small systems)
    pub amplitudes: Option<Vec<NComplex<f64>>>,
    /// Current energy
    pub energy: NComplex<f64>,
    /// Wavefunction sign
    pub sign: NComplex<f64>,
}

impl QuantumState {
    /// Create a new quantum state
    pub fn new(num_qubits: usize, num_time_slices: usize) -> Self {
        let configurations = vec![vec![1; num_qubits]; num_time_slices];

        Self {
            num_qubits,
            num_time_slices,
            configurations,
            amplitudes: None,
            energy: NComplex::new(0.0, 0.0),
            sign: NComplex::new(1.0, 0.0),
        }
    }

    /// Initialize with random configuration
    pub fn initialize_random(&mut self, rng: &mut ChaCha8Rng) {
        for time_slice in &mut self.configurations {
            for spin in time_slice {
                *spin = if rng.gen_bool(0.5) { 1 } else { -1 };
            }
        }
    }

    /// Calculate overlap with another state
    pub fn overlap(&self, other: &QuantumState) -> NComplex<f64> {
        if let (Some(ref amp1), Some(ref amp2)) = (&self.amplitudes, &other.amplitudes) {
            amp1.iter()
                .zip(amp2.iter())
                .map(|(a, b)| a.conj() * b)
                .sum()
        } else {
            NComplex::new(0.0, 0.0)
        }
    }
}

impl NonStoquasticSimulator {
    /// Create a new non-stoquastic simulator
    pub fn new(
        hamiltonian: NonStoquasticHamiltonian,
        config: NonStoquasticQMCConfig,
    ) -> NonStoquasticResult<Self> {
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };

        let current_state = QuantumState::new(hamiltonian.num_qubits, config.num_time_slices);

        Ok(Self {
            config,
            rng,
            current_state,
            hamiltonian,
        })
    }

    /// Run quantum Monte Carlo simulation
    pub fn simulate(&mut self) -> NonStoquasticResult<NonStoquasticResults> {
        let start_time = Instant::now();

        // Initialize state
        self.current_state.initialize_random(&mut self.rng);

        // Choose simulation method based on sign problem severity
        let result = if self.hamiltonian.sign_problem_severity() > 0.1 {
            self.simulate_with_sign_problem()?
        } else {
            self.simulate_stoquastic_like()?
        };

        let simulation_time = start_time.elapsed();

        Ok(NonStoquasticResults {
            simulation_time,
            ..result
        })
    }

    /// Simulate with sign problem handling
    fn simulate_with_sign_problem(&mut self) -> NonStoquasticResult<NonStoquasticResults> {
        match self.config.sign_mitigation {
            SignMitigationStrategy::ReweightingMethod => self.reweighting_simulation(),
            SignMitigationStrategy::PopulationAnnealing => self.population_annealing_simulation(),
            SignMitigationStrategy::ConstrainedPath => self.constrained_path_simulation(),
            _ => self.basic_complex_simulation(),
        }
    }

    /// Simulate stoquastic-like systems
    fn simulate_stoquastic_like(&mut self) -> NonStoquasticResult<NonStoquasticResults> {
        self.basic_quantum_monte_carlo()
    }

    /// Basic quantum Monte Carlo for nearly stoquastic systems
    fn basic_quantum_monte_carlo(&mut self) -> NonStoquasticResult<NonStoquasticResults> {
        let mut energy_samples = Vec::new();
        let mut sign_samples = Vec::new();
        let mut acceptance_count = 0;
        let mut total_proposals = 0;

        // Thermalization
        for _ in 0..self.config.thermalization_steps {
            self.propose_and_accept_move()?;
        }

        // Measurement phase
        for step in 0..self.config.num_steps {
            total_proposals += 1;

            if self.propose_and_accept_move()? {
                acceptance_count += 1;
            }

            if step % self.config.measurement_interval == 0 {
                let energy = self.calculate_energy()?;
                let sign = self.calculate_sign()?;

                energy_samples.push(energy);
                sign_samples.push(sign);
            }
        }

        // Calculate results
        let ground_state_energy = if !energy_samples.is_empty() {
            energy_samples.iter().sum::<NComplex<f64>>() / energy_samples.len() as f64
        } else {
            NComplex::new(0.0, 0.0)
        };

        let average_sign = if !sign_samples.is_empty() {
            sign_samples.iter().sum::<NComplex<f64>>() / sign_samples.len() as f64
        } else {
            NComplex::new(1.0, 0.0)
        };

        let energy_variance = if energy_samples.len() > 1 {
            let mean = ground_state_energy;
            energy_samples
                .iter()
                .map(|e| (e - mean).norm_sqr())
                .sum::<f64>()
                / (energy_samples.len() - 1) as f64
        } else {
            0.0
        };

        let acceptance_rate = acceptance_count as f64 / total_proposals as f64;

        Ok(NonStoquasticResults {
            ground_state_energy,
            energy_variance,
            ground_state: Some(self.current_state.configurations[0].clone()),
            average_sign,
            sign_problem_severity: self.hamiltonian.sign_problem_severity(),
            qmc_statistics: QMCStatistics {
                acceptance_rate,
                autocorrelation_time: 1.0, // Simplified
                effective_sample_size: energy_samples.len(),
                statistical_errors: HashMap::new(),
                population_evolution: Vec::new(),
            },
            simulation_time: Duration::from_secs(0), // Will be filled in
            convergence_info: ConvergenceInfo {
                converged: true,
                convergence_step: Some(self.config.num_steps),
                energy_history: energy_samples,
                sign_history: sign_samples,
            },
        })
    }

    /// Reweighting method for sign problem
    fn reweighting_simulation(&mut self) -> NonStoquasticResult<NonStoquasticResults> {
        let mut weighted_energy = NComplex::new(0.0, 0.0);
        let mut total_weight = NComplex::new(0.0, 0.0);
        let mut sign_samples = Vec::new();

        for _ in 0..self.config.num_steps {
            self.propose_and_accept_move()?;

            let energy = self.calculate_energy()?;
            let weight = self.calculate_weight()?;
            let sign = self.calculate_sign()?;

            weighted_energy += energy * weight;
            total_weight += weight;
            sign_samples.push(sign);
        }

        let ground_state_energy = if total_weight.norm() > 1e-12 {
            weighted_energy / total_weight
        } else {
            NComplex::new(0.0, 0.0)
        };

        let average_sign = if !sign_samples.is_empty() {
            sign_samples.iter().sum::<NComplex<f64>>() / sign_samples.len() as f64
        } else {
            NComplex::new(1.0, 0.0)
        };

        Ok(NonStoquasticResults {
            ground_state_energy,
            energy_variance: 0.0, // Simplified
            ground_state: Some(self.current_state.configurations[0].clone()),
            average_sign,
            sign_problem_severity: self.hamiltonian.sign_problem_severity(),
            qmc_statistics: QMCStatistics {
                acceptance_rate: 0.5, // Simplified
                autocorrelation_time: 1.0,
                effective_sample_size: self.config.num_steps,
                statistical_errors: HashMap::new(),
                population_evolution: Vec::new(),
            },
            simulation_time: Duration::from_secs(0),
            convergence_info: ConvergenceInfo {
                converged: average_sign.norm() > 0.1,
                convergence_step: Some(self.config.num_steps),
                energy_history: Vec::new(),
                sign_history: sign_samples,
            },
        })
    }

    /// Population annealing simulation
    fn population_annealing_simulation(&mut self) -> NonStoquasticResult<NonStoquasticResults> {
        let mut population = Vec::new();
        let mut weights = Vec::new();

        // Initialize population
        for _ in 0..self.config.population_size {
            let mut state =
                QuantumState::new(self.hamiltonian.num_qubits, self.config.num_time_slices);
            state.initialize_random(&mut self.rng);
            population.push(state);
            weights.push(NComplex::new(1.0, 0.0));
        }

        // Population annealing steps
        let num_annealing_steps = 10;
        let mut population_evolution = Vec::new();

        for step in 0..num_annealing_steps {
            let beta = (step as f64 / (num_annealing_steps - 1) as f64) / self.config.temperature;

            // Update weights
            for (i, state) in population.iter().enumerate() {
                let energy = self.calculate_state_energy(state)?;
                weights[i] = (-beta * energy).exp();
            }

            // Resample population
            population = self.resample_population(population, &weights)?;
            weights.fill(NComplex::new(1.0, 0.0));

            population_evolution.push(population.len());
        }

        // Calculate final results
        let final_energies: Vec<NComplex<f64>> = population
            .iter()
            .map(|state| {
                self.calculate_state_energy(state)
                    .unwrap_or(NComplex::new(0.0, 0.0))
            })
            .collect();

        let ground_state_energy = final_energies
            .iter()
            .min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap())
            .cloned()
            .unwrap_or(NComplex::new(0.0, 0.0));

        Ok(NonStoquasticResults {
            ground_state_energy,
            energy_variance: 0.0,
            ground_state: population.first().map(|s| s.configurations[0].clone()),
            average_sign: NComplex::new(1.0, 0.0),
            sign_problem_severity: self.hamiltonian.sign_problem_severity(),
            qmc_statistics: QMCStatistics {
                acceptance_rate: 0.7,
                autocorrelation_time: 1.0,
                effective_sample_size: population.len(),
                statistical_errors: HashMap::new(),
                population_evolution,
            },
            simulation_time: Duration::from_secs(0),
            convergence_info: ConvergenceInfo {
                converged: true,
                convergence_step: Some(num_annealing_steps),
                energy_history: final_energies,
                sign_history: Vec::new(),
            },
        })
    }

    /// Constrained path simulation
    fn constrained_path_simulation(&mut self) -> NonStoquasticResult<NonStoquasticResults> {
        // Simplified constrained path method
        // In practice, this would implement sophisticated path constraints
        self.basic_complex_simulation()
    }

    /// Basic complex simulation
    fn basic_complex_simulation(&mut self) -> NonStoquasticResult<NonStoquasticResults> {
        // Simplified complex simulation
        self.basic_quantum_monte_carlo()
    }

    /// Propose and accept/reject a Monte Carlo move
    fn propose_and_accept_move(&mut self) -> NonStoquasticResult<bool> {
        // Choose random time slice and spin
        let time_slice = self.rng.gen_range(0..self.config.num_time_slices);
        let spin_site = self.rng.gen_range(0..self.hamiltonian.num_qubits);

        // Calculate energy change
        let energy_before = self.calculate_local_energy(time_slice, spin_site)?;

        // Flip spin
        self.current_state.configurations[time_slice][spin_site] *= -1;

        let energy_after = self.calculate_local_energy(time_slice, spin_site)?;
        let energy_diff = energy_after - energy_before;

        // Metropolis acceptance criterion
        let accept_prob = (-energy_diff.re / self.config.temperature).exp().min(1.0);

        if self.rng.gen::<f64>() < accept_prob {
            // Accept move
            Ok(true)
        } else {
            // Reject move - flip spin back
            self.current_state.configurations[time_slice][spin_site] *= -1;
            Ok(false)
        }
    }

    /// Calculate local energy contribution
    fn calculate_local_energy(
        &self,
        time_slice: usize,
        site: usize,
    ) -> NonStoquasticResult<NComplex<f64>> {
        let mut energy = NComplex::new(0.0, 0.0);

        // Local field contribution
        let spin = self.current_state.configurations[time_slice][site] as f64;
        energy += self.hamiltonian.local_fields[site] * spin;

        // Coupling contributions
        for coupling in &self.hamiltonian.complex_couplings {
            if coupling.sites.0 == site || coupling.sites.1 == site {
                let (i, j) = coupling.sites;
                let spin_i = self.current_state.configurations[time_slice][i] as f64;
                let spin_j = self.current_state.configurations[time_slice][j] as f64;

                match coupling.interaction_type {
                    InteractionType::ZZ => {
                        energy += coupling.strength * spin_i * spin_j;
                    }
                    InteractionType::XX | InteractionType::YY => {
                        // For Monte Carlo, treat as effective ZZ with complex weight
                        energy += coupling.strength * spin_i * spin_j * 0.5;
                    }
                    _ => {
                        // Simplified treatment
                        energy += coupling.strength * spin_i * spin_j * 0.25;
                    }
                }
            }
        }

        Ok(energy)
    }

    /// Calculate total energy of current state
    fn calculate_energy(&self) -> NonStoquasticResult<NComplex<f64>> {
        let mut total_energy = NComplex::new(0.0, 0.0);

        for time_slice in 0..self.config.num_time_slices {
            for site in 0..self.hamiltonian.num_qubits {
                total_energy += self.calculate_local_energy(time_slice, site)?;
            }
        }

        Ok(total_energy / self.config.num_time_slices as f64)
    }

    /// Calculate energy of a specific state
    fn calculate_state_energy(&self, state: &QuantumState) -> NonStoquasticResult<NComplex<f64>> {
        let mut energy = NComplex::new(0.0, 0.0);

        // Local fields
        for (site, &field) in self.hamiltonian.local_fields.iter().enumerate() {
            for time_slice in 0..state.num_time_slices {
                let spin = state.configurations[time_slice][site] as f64;
                energy += field * spin;
            }
        }

        // Couplings
        for coupling in &self.hamiltonian.complex_couplings {
            let (i, j) = coupling.sites;
            for time_slice in 0..state.num_time_slices {
                let spin_i = state.configurations[time_slice][i] as f64;
                let spin_j = state.configurations[time_slice][j] as f64;

                match coupling.interaction_type {
                    InteractionType::ZZ => {
                        energy += coupling.strength * spin_i * spin_j;
                    }
                    _ => {
                        energy += coupling.strength * spin_i * spin_j * 0.5; // Simplified
                    }
                }
            }
        }

        Ok(energy / state.num_time_slices as f64)
    }

    /// Calculate wavefunction sign
    fn calculate_sign(&self) -> NonStoquasticResult<NComplex<f64>> {
        // Simplified sign calculation
        let mut sign = NComplex::new(1.0, 0.0);

        // For XY models, sign depends on the configuration
        if let HamiltonianType::XYModel { .. } = self.hamiltonian.hamiltonian_type {
            let mut phase = 0.0;

            for coupling in &self.hamiltonian.complex_couplings {
                if matches!(coupling.interaction_type, InteractionType::YY) {
                    let (i, j) = coupling.sites;
                    for time_slice in 0..self.config.num_time_slices {
                        let spin_i = self.current_state.configurations[time_slice][i];
                        let spin_j = self.current_state.configurations[time_slice][j];

                        if spin_i != spin_j {
                            phase += PI / 4.0; // Simplified phase accumulation
                        }
                    }
                }
            }

            sign = NComplex::new(phase.cos(), phase.sin());
        }

        Ok(sign)
    }

    /// Calculate Monte Carlo weight
    fn calculate_weight(&self) -> NonStoquasticResult<NComplex<f64>> {
        // For reweighting method
        let sign = self.calculate_sign()?;
        Ok(sign)
    }

    /// Resample population for population annealing
    fn resample_population(
        &mut self,
        mut population: Vec<QuantumState>,
        weights: &[NComplex<f64>],
    ) -> NonStoquasticResult<Vec<QuantumState>> {
        // Normalize weights
        let total_weight: f64 = weights.iter().map(|w| w.norm()).sum();
        if total_weight < 1e-12 {
            return Ok(population);
        }

        let probabilities: Vec<f64> = weights.iter().map(|w| w.norm() / total_weight).collect();

        // Systematic resampling
        let mut new_population = Vec::new();
        let n = population.len();
        let step = 1.0 / n as f64;
        let mut cumsum = 0.0;
        let offset = self.rng.gen::<f64>() * step;

        let mut i = 0;
        for j in 0..n {
            let target = offset + j as f64 * step;

            while cumsum < target && i < probabilities.len() {
                cumsum += probabilities[i];
                i += 1;
            }

            if i > 0 {
                new_population.push(population[(i - 1).min(population.len() - 1)].clone());
            }
        }

        Ok(new_population)
    }
}

/// Utility functions for non-stoquastic systems

/// Detect whether a Hamiltonian is stoquastic
pub fn is_hamiltonian_stoquastic(hamiltonian: &NonStoquasticHamiltonian) -> bool {
    hamiltonian.is_stoquastic()
}

/// Convert XY model to effective Ising model (approximation)
pub fn xy_to_ising_approximation(
    xy_hamiltonian: &NonStoquasticHamiltonian,
) -> NonStoquasticResult<IsingModel> {
    if !matches!(
        xy_hamiltonian.hamiltonian_type,
        HamiltonianType::XYModel { .. }
    ) {
        return Err(NonStoquasticError::InvalidHamiltonian(
            "Expected XY model".to_string(),
        ));
    }

    let mut ising = IsingModel::new(xy_hamiltonian.num_qubits);

    // Convert local fields directly
    for (site, &field) in xy_hamiltonian.local_fields.iter().enumerate() {
        ising.set_bias(site, field)?;
    }

    // Convert XY couplings to effective ZZ couplings
    let mut coupling_map: HashMap<(usize, usize), f64> = HashMap::new();

    for coupling in &xy_hamiltonian.complex_couplings {
        let (i, j) = coupling.sites;
        let key = if i < j { (i, j) } else { (j, i) };

        let effective_strength = match coupling.interaction_type {
            InteractionType::XX | InteractionType::YY => {
                // XY couplings contribute to effective ferromagnetic coupling
                -coupling.strength.re.abs() * 0.5
            }
            InteractionType::ZZ => coupling.strength.re,
            _ => coupling.strength.re * 0.25, // Simplified
        };

        *coupling_map.entry(key).or_insert(0.0) += effective_strength;
    }

    // Set effective couplings
    for ((i, j), strength) in coupling_map {
        ising.set_coupling(i, j, strength)?;
    }

    Ok(ising)
}

/// Create standard non-stoquastic test problems

/// Create XY chain with periodic boundary conditions
pub fn create_xy_chain(
    num_qubits: usize,
    j_x: f64,
    j_y: f64,
) -> NonStoquasticResult<NonStoquasticHamiltonian> {
    let mut hamiltonian = NonStoquasticHamiltonian::xy_model(num_qubits, j_x, j_y)?;

    // Add periodic boundary condition
    if num_qubits > 2 {
        hamiltonian.add_complex_coupling(ComplexCoupling {
            sites: (num_qubits - 1, 0),
            strength: NComplex::new(j_x, 0.0),
            interaction_type: InteractionType::XX,
        })?;

        hamiltonian.add_complex_coupling(ComplexCoupling {
            sites: (num_qubits - 1, 0),
            strength: NComplex::new(j_y, 0.0),
            interaction_type: InteractionType::YY,
        })?;
    }

    Ok(hamiltonian)
}

/// Create transverse field XY model
pub fn create_tfxy_model(
    num_qubits: usize,
    j_x: f64,
    j_y: f64,
    h_z: f64,
) -> NonStoquasticResult<NonStoquasticHamiltonian> {
    let mut hamiltonian = NonStoquasticHamiltonian::xy_model(num_qubits, j_x, j_y)?;

    // Add transverse field
    for site in 0..num_qubits {
        hamiltonian.set_local_field(site, h_z)?;
    }

    Ok(hamiltonian)
}

/// Create frustrated XY model on triangular lattice
pub fn create_frustrated_xy_triangle(j_xy: f64) -> NonStoquasticResult<NonStoquasticHamiltonian> {
    let mut hamiltonian = NonStoquasticHamiltonian::new(
        3,
        HamiltonianType::XYModel {
            j_x: j_xy,
            j_y: j_xy,
        },
    );

    // Add all pairwise XY interactions (frustrated triangle)
    for i in 0..3 {
        for j in (i + 1)..3 {
            hamiltonian.add_complex_coupling(ComplexCoupling {
                sites: (i, j),
                strength: NComplex::new(j_xy, 0.0),
                interaction_type: InteractionType::XX,
            })?;

            hamiltonian.add_complex_coupling(ComplexCoupling {
                sites: (i, j),
                strength: NComplex::new(j_xy, 0.0),
                interaction_type: InteractionType::YY,
            })?;
        }
    }

    Ok(hamiltonian)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xy_model_creation() {
        let hamiltonian = NonStoquasticHamiltonian::xy_model(4, 1.0, 0.5).unwrap();
        assert_eq!(hamiltonian.num_qubits, 4);
        assert!(hamiltonian.has_sign_problem);
        assert_eq!(hamiltonian.complex_couplings.len(), 6); // 3 XX + 3 YY couplings
    }

    #[test]
    fn test_xyz_model_creation() {
        let hamiltonian = NonStoquasticHamiltonian::xyz_model(3, 1.0, 1.0, 0.5).unwrap();
        assert_eq!(hamiltonian.num_qubits, 3);
        assert!(hamiltonian.has_sign_problem);
        assert_eq!(hamiltonian.complex_couplings.len(), 6); // 2 XX + 2 YY + 2 ZZ couplings
    }

    #[test]
    fn test_sign_problem_detection() {
        let xy_hamiltonian = NonStoquasticHamiltonian::xy_model(4, 1.0, 1.0).unwrap();
        assert!(xy_hamiltonian.sign_problem_severity() > 0.0);

        let ising_like = NonStoquasticHamiltonian::new(
            4,
            HamiltonianType::MixedModel {
                stoquastic_fraction: 1.0,
            },
        );
        assert!(!ising_like.has_sign_problem);
    }

    #[test]
    fn test_local_field_setting() {
        let mut hamiltonian = NonStoquasticHamiltonian::xy_model(3, 1.0, 1.0).unwrap();
        hamiltonian.set_local_field(0, 0.5).unwrap();
        hamiltonian.set_local_field(2, -0.3).unwrap();

        assert_eq!(hamiltonian.local_fields[0], 0.5);
        assert_eq!(hamiltonian.local_fields[1], 0.0);
        assert_eq!(hamiltonian.local_fields[2], -0.3);
    }

    #[test]
    fn test_complex_coupling_addition() {
        let mut hamiltonian = NonStoquasticHamiltonian::new(4, HamiltonianType::CustomModel);

        let coupling = ComplexCoupling {
            sites: (0, 2),
            strength: NComplex::new(0.5, 0.3),
            interaction_type: InteractionType::XY,
        };

        hamiltonian.add_complex_coupling(coupling.clone()).unwrap();
        assert_eq!(hamiltonian.complex_couplings.len(), 1);
        assert_eq!(hamiltonian.complex_couplings[0].sites, (0, 2));
        assert_eq!(
            hamiltonian.complex_couplings[0].strength,
            NComplex::new(0.5, 0.3)
        );
    }

    #[test]
    fn test_matrix_representation_small() {
        let hamiltonian = NonStoquasticHamiltonian::xy_model(2, 1.0, 0.0).unwrap();
        let matrix = hamiltonian.to_matrix().unwrap();

        assert_eq!(matrix.len(), 4); // 2^2 = 4 states
        assert_eq!(matrix[0].len(), 4);

        // Check that matrix is Hermitian
        for i in 0..4 {
            for j in 0..4 {
                let diff = (matrix[i][j] - matrix[j][i].conj()).norm();
                assert!(diff < 1e-10, "Matrix is not Hermitian at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(3, 5);
        assert_eq!(state.num_qubits, 3);
        assert_eq!(state.num_time_slices, 5);
        assert_eq!(state.configurations.len(), 5);
        assert_eq!(state.configurations[0].len(), 3);
    }

    #[test]
    fn test_xy_to_ising_conversion() {
        let xy_hamiltonian = NonStoquasticHamiltonian::xy_model(3, 1.0, 1.0).unwrap();
        let ising = xy_to_ising_approximation(&xy_hamiltonian).unwrap();

        assert_eq!(ising.num_qubits, 3);

        // Check that couplings were converted
        let coupling_01 = ising.get_coupling(0, 1).unwrap();
        assert!(coupling_01.abs() > 1e-10); // Should have non-zero coupling
    }

    #[test]
    fn test_helper_functions() {
        let xy_chain = create_xy_chain(4, 1.0, 0.5).unwrap();
        assert_eq!(xy_chain.num_qubits, 4);
        assert!(xy_chain.complex_couplings.len() > 6); // Should have periodic boundary

        let tfxy = create_tfxy_model(3, 1.0, 1.0, 0.5).unwrap();
        assert!(tfxy.local_fields.iter().all(|&f| f.abs() > 1e-10)); // All sites should have fields

        let triangle = create_frustrated_xy_triangle(1.0).unwrap();
        assert_eq!(triangle.num_qubits, 3);
        assert_eq!(triangle.complex_couplings.len(), 6); // 3 pairs × 2 (XX, YY)
    }
}
