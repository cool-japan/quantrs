//! Quantum annealing simulation with realistic noise models and hardware constraints.
//!
//! This module implements comprehensive quantum annealing simulation that accurately
//! models real quantum annealing hardware including thermal noise, decoherence,
//! control errors, and hardware topology constraints. It supports various
//! optimization problems (QUBO, Ising, etc.) and provides realistic simulation
//! of quantum annealing devices like D-Wave systems.

use crate::prelude::SimulatorError;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops::*;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::device_noise_models::{DeviceNoiseSimulator, DeviceTopology};
use crate::error::Result;
use crate::scirs2_integration::SciRS2Backend;

/// Quantum annealing configuration
#[derive(Debug, Clone)]
pub struct QuantumAnnealingConfig {
    /// Total annealing time (μs)
    pub annealing_time: f64,
    /// Number of time steps
    pub time_steps: usize,
    /// Annealing schedule type
    pub schedule_type: AnnealingScheduleType,
    /// Problem formulation
    pub problem_type: ProblemType,
    /// Hardware topology
    pub topology: AnnealingTopology,
    /// Temperature (K)
    pub temperature: f64,
    /// Enable realistic noise models
    pub enable_noise: bool,
    /// Enable thermal fluctuations
    pub enable_thermal_fluctuations: bool,
    /// Enable control errors
    pub enable_control_errors: bool,
    /// Enable gauge transformations
    pub enable_gauge_transformations: bool,
    /// Post-processing configuration
    pub post_processing: PostProcessingConfig,
}

impl Default for QuantumAnnealingConfig {
    fn default() -> Self {
        Self {
            annealing_time: 20.0, // 20 μs (typical D-Wave)
            time_steps: 2000,
            schedule_type: AnnealingScheduleType::DWave,
            problem_type: ProblemType::Ising,
            topology: AnnealingTopology::Chimera(16),
            temperature: 0.015, // 15 mK
            enable_noise: true,
            enable_thermal_fluctuations: true,
            enable_control_errors: true,
            enable_gauge_transformations: true,
            post_processing: PostProcessingConfig::default(),
        }
    }
}

/// Annealing schedule types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnnealingScheduleType {
    /// Linear schedule
    Linear,
    /// D-Wave like schedule with pause
    DWave,
    /// Optimized schedule for specific problems
    Optimized,
    /// Custom schedule with pause features
    CustomPause {
        pause_start: f64,
        pause_duration: f64,
    },
    /// Non-monotonic schedule
    NonMonotonic,
    /// Reverse annealing
    Reverse { reinitialize_point: f64 },
}

/// Problem types for quantum annealing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemType {
    /// Ising model
    Ising,
    /// Quadratic Unconstrained Binary Optimization
    QUBO,
    /// Maximum Cut
    MaxCut,
    /// Graph Coloring
    GraphColoring,
    /// Traveling Salesman Problem
    TSP,
    /// Number Partitioning
    NumberPartitioning,
    /// Custom optimization problem
    Custom(String),
}

/// Annealing hardware topologies
#[derive(Debug, Clone, PartialEq)]
pub enum AnnealingTopology {
    /// D-Wave Chimera topology
    Chimera(usize), // Parameter is the size
    /// D-Wave Pegasus topology
    Pegasus(usize),
    /// D-Wave Zephyr topology
    Zephyr(usize),
    /// Complete graph
    Complete(usize),
    /// Custom topology
    Custom(DeviceTopology),
}

/// Post-processing configuration
#[derive(Debug, Clone)]
pub struct PostProcessingConfig {
    /// Enable spin reversal transformations
    pub enable_spin_reversal: bool,
    /// Enable local search optimization
    pub enable_local_search: bool,
    /// Maximum local search iterations
    pub max_local_search_iterations: usize,
    /// Enable majority vote post-processing
    pub enable_majority_vote: bool,
    /// Number of reads for majority vote
    pub majority_vote_reads: usize,
    /// Enable energy-based filtering
    pub enable_energy_filtering: bool,
}

impl Default for PostProcessingConfig {
    fn default() -> Self {
        Self {
            enable_spin_reversal: true,
            enable_local_search: true,
            max_local_search_iterations: 100,
            enable_majority_vote: true,
            majority_vote_reads: 1000,
            enable_energy_filtering: true,
        }
    }
}

/// Ising problem representation
#[derive(Debug, Clone)]
pub struct IsingProblem {
    /// Number of spins
    pub num_spins: usize,
    /// Linear coefficients (h_i)
    pub h: Array1<f64>,
    /// Quadratic coefficients (J_{ij})
    pub j: Array2<f64>,
    /// Offset constant
    pub offset: f64,
    /// Problem metadata
    pub metadata: ProblemMetadata,
}

/// QUBO problem representation
#[derive(Debug, Clone)]
pub struct QUBOProblem {
    /// Number of variables
    pub num_variables: usize,
    /// QUBO matrix (Q_{ij})
    pub q: Array2<f64>,
    /// Linear coefficients
    pub linear: Array1<f64>,
    /// Offset constant
    pub offset: f64,
    /// Problem metadata
    pub metadata: ProblemMetadata,
}

/// Problem metadata
#[derive(Debug, Clone, Default)]
pub struct ProblemMetadata {
    /// Problem name
    pub name: Option<String>,
    /// Problem description
    pub description: Option<String>,
    /// Known optimal energy
    pub optimal_energy: Option<f64>,
    /// Problem difficulty estimate
    pub difficulty_score: Option<f64>,
    /// Variable labels
    pub variable_labels: Vec<String>,
}

impl IsingProblem {
    /// Create new Ising problem
    pub fn new(num_spins: usize) -> Self {
        Self {
            num_spins,
            h: Array1::zeros(num_spins),
            j: Array2::zeros((num_spins, num_spins)),
            offset: 0.0,
            metadata: ProblemMetadata::default(),
        }
    }

    /// Set linear coefficient
    pub fn set_h(&mut self, i: usize, value: f64) {
        if i < self.num_spins {
            self.h[i] = value;
        }
    }

    /// Set quadratic coefficient
    pub fn set_j(&mut self, i: usize, j: usize, value: f64) {
        if i < self.num_spins && j < self.num_spins {
            self.j[[i, j]] = value;
            self.j[[j, i]] = value; // Ensure symmetry
        }
    }

    /// Calculate energy for a given configuration
    pub fn calculate_energy(&self, configuration: &[i8]) -> f64 {
        if configuration.len() != self.num_spins {
            return f64::INFINITY;
        }

        let mut energy = self.offset;

        // Linear terms
        for i in 0..self.num_spins {
            energy += self.h[i] * configuration[i] as f64;
        }

        // Quadratic terms
        for i in 0..self.num_spins {
            for j in i + 1..self.num_spins {
                energy += self.j[[i, j]] * configuration[i] as f64 * configuration[j] as f64;
            }
        }

        energy
    }

    /// Convert to QUBO problem
    pub fn to_qubo(&self) -> QUBOProblem {
        let num_vars = self.num_spins;
        let mut q = Array2::zeros((num_vars, num_vars));
        let mut linear = Array1::zeros(num_vars);
        let mut offset = self.offset;

        // Convert Ising to QUBO: s_i = 2x_i - 1
        // H_Ising = sum_i h_i s_i + sum_{i<j} J_{ij} s_i s_j
        // H_QUBO = sum_i q_i x_i + sum_{i<j} q_{ij} x_i x_j + const

        for i in 0..num_vars {
            // Linear terms: h_i s_i = h_i (2x_i - 1) = 2h_i x_i - h_i
            linear[i] += 2.0 * self.h[i];
            offset -= self.h[i];

            for j in i + 1..num_vars {
                // Quadratic terms: J_{ij} s_i s_j = J_{ij} (2x_i - 1)(2x_j - 1)
                // = 4 J_{ij} x_i x_j - 2 J_{ij} x_i - 2 J_{ij} x_j + J_{ij}
                q[[i, j]] += 4.0 * self.j[[i, j]];
                linear[i] -= 2.0 * self.j[[i, j]];
                linear[j] -= 2.0 * self.j[[i, j]];
                offset += self.j[[i, j]];
            }
        }

        QUBOProblem {
            num_variables: num_vars,
            q,
            linear,
            offset,
            metadata: self.metadata.clone(),
        }
    }

    /// Find ground state using brute force (for small problems)
    pub fn find_ground_state_brute_force(&self) -> (Vec<i8>, f64) {
        assert!(
            (self.num_spins <= 20),
            "Brute force search only supported for <= 20 spins"
        );

        let mut best_config = vec![-1; self.num_spins];
        let mut best_energy = f64::INFINITY;

        for state in 0..(1 << self.num_spins) {
            let mut config = vec![-1; self.num_spins];
            for i in 0..self.num_spins {
                if (state >> i) & 1 == 1 {
                    config[i] = 1;
                }
            }

            let energy = self.calculate_energy(&config);
            if energy < best_energy {
                best_energy = energy;
                best_config = config;
            }
        }

        (best_config, best_energy)
    }
}

impl QUBOProblem {
    /// Create new QUBO problem
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            q: Array2::zeros((num_variables, num_variables)),
            linear: Array1::zeros(num_variables),
            offset: 0.0,
            metadata: ProblemMetadata::default(),
        }
    }

    /// Calculate energy for a given binary configuration
    pub fn calculate_energy(&self, configuration: &[u8]) -> f64 {
        if configuration.len() != self.num_variables {
            return f64::INFINITY;
        }

        let mut energy = self.offset;

        // Linear terms
        for i in 0..self.num_variables {
            energy += self.linear[i] * configuration[i] as f64;
        }

        // Quadratic terms
        for i in 0..self.num_variables {
            for j in 0..self.num_variables {
                if i != j {
                    energy += self.q[[i, j]] * configuration[i] as f64 * configuration[j] as f64;
                }
            }
        }

        energy
    }

    /// Convert to Ising problem
    pub fn to_ising(&self) -> IsingProblem {
        let num_spins = self.num_variables;
        let mut h = Array1::zeros(num_spins);
        let mut j = Array2::zeros((num_spins, num_spins));
        let mut offset = self.offset;

        // Convert QUBO to Ising: x_i = (s_i + 1)/2
        for i in 0..num_spins {
            h[i] = self.linear[i] / 2.0;
            offset += self.linear[i] / 2.0;

            for k in 0..num_spins {
                if k != i {
                    h[i] += self.q[[i, k]] / 4.0;
                    offset += self.q[[i, k]] / 4.0;
                }
            }
        }

        for i in 0..num_spins {
            for k in i + 1..num_spins {
                j[[i, k]] = self.q[[i, k]] / 4.0;
            }
        }

        IsingProblem {
            num_spins,
            h,
            j,
            offset,
            metadata: self.metadata.clone(),
        }
    }
}

/// Quantum annealing simulator
pub struct QuantumAnnealingSimulator {
    /// Configuration
    config: QuantumAnnealingConfig,
    /// Current problem
    current_problem: Option<IsingProblem>,
    /// Device noise simulator
    noise_simulator: Option<DeviceNoiseSimulator>,
    /// SciRS2 backend for optimization
    backend: Option<SciRS2Backend>,
    /// Annealing history
    annealing_history: Vec<AnnealingSnapshot>,
    /// Final solutions
    solutions: Vec<AnnealingSolution>,
    /// Statistics
    stats: AnnealingStats,
}

/// Annealing snapshot
#[derive(Debug, Clone)]
pub struct AnnealingSnapshot {
    /// Time parameter
    pub time: f64,
    /// Annealing parameter s(t)
    pub s: f64,
    /// Transverse field strength
    pub transverse_field: f64,
    /// Longitudinal field strength
    pub longitudinal_field: f64,
    /// Current quantum state (if tracking)
    pub quantum_state: Option<Array1<Complex64>>,
    /// Classical state probabilities
    pub classical_probabilities: Option<Array1<f64>>,
    /// Energy expectation value
    pub energy_expectation: f64,
    /// Temperature effects
    pub temperature_factor: f64,
}

/// Annealing solution
#[derive(Debug, Clone)]
pub struct AnnealingSolution {
    /// Solution configuration
    pub configuration: Vec<i8>,
    /// Solution energy
    pub energy: f64,
    /// Solution probability
    pub probability: f64,
    /// Number of occurrences
    pub num_occurrences: usize,
    /// Solution rank
    pub rank: usize,
}

/// Annealing simulation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnnealingStats {
    /// Total annealing time
    pub total_annealing_time_ms: f64,
    /// Number of annealing runs
    pub num_annealing_runs: usize,
    /// Number of solutions found
    pub num_solutions_found: usize,
    /// Best energy found
    pub best_energy_found: f64,
    /// Success probability (if ground state known)
    pub success_probability: f64,
    /// Time to solution statistics
    pub time_to_solution: TimeToSolutionStats,
    /// Noise statistics
    pub noise_stats: NoiseStats,
}

/// Time to solution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeToSolutionStats {
    /// Median time to solution
    pub median_tts: f64,
    /// 99th percentile time to solution
    pub percentile_99_tts: f64,
    /// Success rate
    pub success_rate: f64,
}

/// Noise statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoiseStats {
    /// Thermal excitation events
    pub thermal_excitations: usize,
    /// Control error events
    pub control_errors: usize,
    /// Decoherence events
    pub decoherence_events: usize,
}

impl QuantumAnnealingSimulator {
    /// Create new quantum annealing simulator
    pub fn new(config: QuantumAnnealingConfig) -> Result<Self> {
        Ok(Self {
            config,
            current_problem: None,
            noise_simulator: None,
            backend: None,
            annealing_history: Vec::new(),
            solutions: Vec::new(),
            stats: AnnealingStats::default(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Set problem to solve
    pub fn set_problem(&mut self, problem: IsingProblem) -> Result<()> {
        // Validate problem size against topology
        let max_spins = match &self.config.topology {
            AnnealingTopology::Chimera(size) => size * size * 8,
            AnnealingTopology::Pegasus(size) => size * (size - 1) * 12,
            AnnealingTopology::Zephyr(size) => size * size * 8,
            AnnealingTopology::Complete(size) => *size,
            AnnealingTopology::Custom(topology) => topology.num_qubits,
        };

        if problem.num_spins > max_spins {
            return Err(SimulatorError::InvalidInput(format!(
                "Problem size {} exceeds topology limit {}",
                problem.num_spins, max_spins
            )));
        }

        self.current_problem = Some(problem);
        Ok(())
    }

    /// Run quantum annealing
    pub fn anneal(&mut self, num_reads: usize) -> Result<AnnealingResult> {
        let problem = self
            .current_problem
            .as_ref()
            .ok_or_else(|| SimulatorError::InvalidInput("No problem set".to_string()))?;

        let start_time = std::time::Instant::now();
        self.solutions.clear();

        for read in 0..num_reads {
            let read_start = std::time::Instant::now();

            // Run single annealing cycle
            let solution = self.single_anneal(read)?;
            self.solutions.push(solution);

            if read % 100 == 0 {
                println!(
                    "Completed read {}/{}, time={:.2}ms",
                    read,
                    num_reads,
                    read_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        // Post-process solutions
        if self.config.post_processing.enable_majority_vote {
            self.apply_majority_vote_post_processing()?;
        }

        if self.config.post_processing.enable_local_search {
            self.apply_local_search_post_processing()?;
        }

        // Sort solutions by energy
        self.solutions
            .sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        // Rank solutions
        for (rank, solution) in self.solutions.iter_mut().enumerate() {
            solution.rank = rank;
        }

        // Compute statistics
        self.compute_annealing_statistics()?;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_annealing_time_ms += total_time;
        self.stats.num_annealing_runs += num_reads;

        Ok(AnnealingResult {
            solutions: self.solutions.clone(),
            best_energy: self.solutions.first().map_or(f64::INFINITY, |s| s.energy),
            annealing_history: self.annealing_history.clone(),
            total_time_ms: total_time,
            success_probability: self.stats.success_probability,
            time_to_solution: self.stats.time_to_solution.clone(),
        })
    }

    /// Run single annealing cycle
    fn single_anneal(&mut self, read_id: usize) -> Result<AnnealingSolution> {
        let problem_num_spins = self.current_problem.as_ref().unwrap().num_spins;

        // Initialize quantum state in superposition
        let state_size = 1 << problem_num_spins.min(20); // Limit for memory
        let mut quantum_state = if problem_num_spins <= 20 {
            let mut state = Array1::zeros(state_size);
            // Initialize in equal superposition
            let amplitude = (1.0 / state_size as f64).sqrt();
            state.fill(Complex64::new(amplitude, 0.0));
            Some(state)
        } else {
            None // Use classical approximation for large problems
        };

        let dt = self.config.annealing_time / self.config.time_steps as f64;
        self.annealing_history.clear();

        // Annealing evolution
        for step in 0..=self.config.time_steps {
            let t = step as f64 * dt;
            let s = self.schedule_function(t);

            // Calculate field strengths
            let (transverse_field, longitudinal_field) = self.calculate_field_strengths(s);

            // Apply quantum evolution
            if let Some(ref mut state) = quantum_state {
                self.apply_quantum_evolution(state, transverse_field, longitudinal_field, dt)?;

                // Apply noise if enabled
                if self.config.enable_noise {
                    self.apply_annealing_noise(state, dt)?;
                }
            }

            // Take snapshot
            if step % (self.config.time_steps / 100) == 0 {
                let snapshot = self.take_annealing_snapshot(
                    t,
                    s,
                    transverse_field,
                    longitudinal_field,
                    &quantum_state,
                )?;
                self.annealing_history.push(snapshot);
            }
        }

        // Final measurement/sampling
        let final_configuration = if let Some(ref state) = quantum_state {
            self.measure_final_state(state)?
        } else {
            // Get the problem again for classical sampling
            let problem = self.current_problem.as_ref().unwrap();
            self.classical_sampling(problem)?
        };

        let energy = self
            .current_problem
            .as_ref()
            .unwrap()
            .calculate_energy(&final_configuration);

        Ok(AnnealingSolution {
            configuration: final_configuration,
            energy,
            probability: 1.0 / (self.config.time_steps as f64), // Will be updated later
            num_occurrences: 1,
            rank: 0,
        })
    }

    /// Calculate annealing schedule s(t)
    fn schedule_function(&self, t: f64) -> f64 {
        let normalized_t = t / self.config.annealing_time;

        match self.config.schedule_type {
            AnnealingScheduleType::Linear => normalized_t,
            AnnealingScheduleType::DWave => {
                // D-Wave like schedule with slower start and end
                if normalized_t < 0.1 {
                    5.0 * normalized_t * normalized_t
                } else if normalized_t < 0.9 {
                    0.05 + 0.9 * (normalized_t - 0.1) / 0.8
                } else {
                    0.05f64.mul_add(
                        1.0 - (1.0 - normalized_t) * (1.0 - normalized_t) / 0.01,
                        0.95,
                    )
                }
            }
            AnnealingScheduleType::Optimized => {
                // Optimized schedule based on problem characteristics
                self.optimized_schedule(normalized_t)
            }
            AnnealingScheduleType::CustomPause {
                pause_start,
                pause_duration,
            } => {
                if normalized_t >= pause_start && normalized_t <= pause_start + pause_duration {
                    pause_start // Pause at this value
                } else if normalized_t > pause_start + pause_duration {
                    (normalized_t - pause_duration - pause_start) / (1.0 - pause_duration)
                } else {
                    normalized_t / pause_start
                }
            }
            AnnealingScheduleType::NonMonotonic => {
                // Non-monotonic schedule with oscillations
                (0.1 * (10.0 * std::f64::consts::PI * normalized_t).sin())
                    .mul_add(1.0 - normalized_t, normalized_t)
            }
            AnnealingScheduleType::Reverse { reinitialize_point } => {
                if normalized_t < reinitialize_point {
                    1.0 // Start at problem Hamiltonian
                } else {
                    1.0 - (normalized_t - reinitialize_point) / (1.0 - reinitialize_point)
                }
            }
        }
    }

    /// Optimized schedule function
    fn optimized_schedule(&self, t: f64) -> f64 {
        // Simple optimization: slower evolution near avoided crossings
        // This would be problem-specific in practice
        if t < 0.3 {
            t * t / 0.09 * 0.3
        } else if t < 0.7 {
            0.3 + (t - 0.3) * 0.4 / 0.4
        } else {
            ((t - 0.7) * (t - 0.7) / 0.09).mul_add(0.3, 0.7)
        }
    }

    /// Calculate transverse and longitudinal field strengths
    fn calculate_field_strengths(&self, s: f64) -> (f64, f64) {
        // Standard quantum annealing: H(s) = -A(s) ∑_i σ_x^i + B(s) H_problem
        let a_s = (1.0 - s) * 1.0; // Transverse field strength
        let b_s = s * 1.0; // Longitudinal field strength
        (a_s, b_s)
    }

    /// Apply quantum evolution for one time step
    fn apply_quantum_evolution(
        &mut self,
        state: &mut Array1<Complex64>,
        transverse_field: f64,
        longitudinal_field: f64,
        dt: f64,
    ) -> Result<()> {
        let problem = self.current_problem.as_ref().unwrap();
        let num_spins = problem.num_spins;

        // Build total Hamiltonian matrix
        let hamiltonian = self.build_annealing_hamiltonian(transverse_field, longitudinal_field)?;

        // Apply time evolution: |ψ(t+dt)⟩ = exp(-i H dt / ℏ) |ψ(t)⟩
        let evolution_operator = self.compute_evolution_operator(&hamiltonian, dt)?;
        *state = evolution_operator.dot(state);

        // Renormalize to handle numerical errors
        let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-15 {
            state.mapv_inplace(|x| x / norm);
        }

        Ok(())
    }

    /// Build full annealing Hamiltonian
    fn build_annealing_hamiltonian(
        &self,
        transverse_field: f64,
        longitudinal_field: f64,
    ) -> Result<Array2<Complex64>> {
        let problem = self.current_problem.as_ref().unwrap();
        let num_spins = problem.num_spins;
        let dim = 1 << num_spins;
        let mut hamiltonian = Array2::zeros((dim, dim));

        // Transverse field terms: -A(s) ∑_i σ_x^i
        for spin in 0..num_spins {
            let sigma_x = self.build_sigma_x(spin, num_spins);
            hamiltonian = hamiltonian - sigma_x.mapv(|x| x * transverse_field);
        }

        // Longitudinal field terms: B(s) H_problem
        let problem_hamiltonian = self.build_problem_hamiltonian()?;
        hamiltonian = hamiltonian + problem_hamiltonian.mapv(|x| x * longitudinal_field);

        Ok(hamiltonian)
    }

    /// Build Pauli-X operator for specific spin
    fn build_sigma_x(&self, target_spin: usize, num_spins: usize) -> Array2<Complex64> {
        let dim = 1 << num_spins;
        let mut sigma_x = Array2::zeros((dim, dim));

        for i in 0..dim {
            let j = i ^ (1 << target_spin); // Flip the target spin
            sigma_x[[i, j]] = Complex64::new(1.0, 0.0);
        }

        sigma_x
    }

    /// Build problem Hamiltonian (Ising model)
    fn build_problem_hamiltonian(&self) -> Result<Array2<Complex64>> {
        let problem = self.current_problem.as_ref().unwrap();
        let num_spins = problem.num_spins;
        let dim = 1 << num_spins;
        let mut hamiltonian = Array2::zeros((dim, dim));

        // Linear terms: ∑_i h_i σ_z^i
        for i in 0..num_spins {
            let sigma_z = self.build_sigma_z(i, num_spins);
            hamiltonian = hamiltonian + sigma_z.mapv(|x| x * problem.h[i]);
        }

        // Quadratic terms: ∑_{i<j} J_{ij} σ_z^i σ_z^j
        for i in 0..num_spins {
            for j in i + 1..num_spins {
                if problem.j[[i, j]] != 0.0 {
                    let sigma_z_i = self.build_sigma_z(i, num_spins);
                    let sigma_z_j = self.build_sigma_z(j, num_spins);
                    let sigma_z_ij = sigma_z_i.dot(&sigma_z_j);
                    hamiltonian = hamiltonian + sigma_z_ij.mapv(|x| x * problem.j[[i, j]]);
                }
            }
        }

        // Add offset as identity matrix
        for i in 0..dim {
            hamiltonian[[i, i]] += Complex64::new(problem.offset, 0.0);
        }

        Ok(hamiltonian)
    }

    /// Build Pauli-Z operator for specific spin
    fn build_sigma_z(&self, target_spin: usize, num_spins: usize) -> Array2<Complex64> {
        let dim = 1 << num_spins;
        let mut sigma_z = Array2::zeros((dim, dim));

        for i in 0..dim {
            let sign = if (i >> target_spin) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            sigma_z[[i, i]] = Complex64::new(sign, 0.0);
        }

        sigma_z
    }

    /// Compute time evolution operator
    fn compute_evolution_operator(
        &self,
        hamiltonian: &Array2<Complex64>,
        dt: f64,
    ) -> Result<Array2<Complex64>> {
        // Use matrix exponentiation for small systems
        self.matrix_exponential(hamiltonian, -Complex64::new(0.0, dt))
    }

    /// Matrix exponential implementation
    fn matrix_exponential(
        &self,
        matrix: &Array2<Complex64>,
        factor: Complex64,
    ) -> Result<Array2<Complex64>> {
        let dim = matrix.dim().0;
        let scaled_matrix = matrix.mapv(|x| x * factor);

        let mut result = Array2::eye(dim);
        let mut term = Array2::eye(dim);

        for n in 1..=15 {
            // Limit series expansion
            term = term.dot(&scaled_matrix) / (n as f64);
            let term_norm: f64 = term.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();

            result += &term;

            if term_norm < 1e-12 {
                break;
            }
        }

        Ok(result)
    }

    /// Apply various noise sources during annealing
    fn apply_annealing_noise(&mut self, state: &mut Array1<Complex64>, dt: f64) -> Result<()> {
        if self.config.enable_thermal_fluctuations {
            self.apply_thermal_noise(state, dt)?;
            self.stats.noise_stats.thermal_excitations += 1;
        }

        if self.config.enable_control_errors {
            self.apply_control_error_noise(state, dt)?;
            self.stats.noise_stats.control_errors += 1;
        }

        // Decoherence
        self.apply_decoherence_noise(state, dt)?;
        self.stats.noise_stats.decoherence_events += 1;

        Ok(())
    }

    /// Apply thermal noise
    fn apply_thermal_noise(&self, state: &mut Array1<Complex64>, dt: f64) -> Result<()> {
        // Thermal fluctuations cause random phase evolution
        let kb_t = 1.38e-23 * self.config.temperature; // Boltzmann constant times temperature
        let thermal_energy = kb_t * dt * 1e6; // Convert to relevant energy scale

        for amplitude in state.iter_mut() {
            let thermal_phase = fastrand::f64() * thermal_energy * 2.0 * std::f64::consts::PI;
            *amplitude *= Complex64::new(0.0, thermal_phase).exp();
        }

        Ok(())
    }

    /// Apply control error noise
    fn apply_control_error_noise(&self, state: &mut Array1<Complex64>, dt: f64) -> Result<()> {
        // Control errors cause imperfect implementation of the intended Hamiltonian
        let error_strength = 0.01; // 1% control errors

        // Apply random single-qubit rotations to simulate control errors
        let problem = self.current_problem.as_ref().unwrap();
        for spin in 0..problem.num_spins.min(10) {
            // Limit for performance
            if fastrand::f64() < error_strength * dt {
                let error_angle = fastrand::f64() * 0.1; // Small random rotation
                self.apply_single_spin_rotation(state, spin, error_angle)?;
            }
        }

        Ok(())
    }

    /// Apply decoherence noise
    fn apply_decoherence_noise(&self, state: &mut Array1<Complex64>, dt: f64) -> Result<()> {
        let decoherence_rate = 1e-3; // Typical decoherence rate
        let decoherence_prob = decoherence_rate * dt;

        for amplitude in state.iter_mut() {
            if fastrand::f64() < decoherence_prob {
                // Apply random dephasing
                let phase = fastrand::f64() * 2.0 * std::f64::consts::PI;
                *amplitude *= Complex64::new(0.0, phase).exp();
            }
        }

        Ok(())
    }

    /// Apply single spin rotation
    fn apply_single_spin_rotation(
        &self,
        state: &mut Array1<Complex64>,
        spin: usize,
        angle: f64,
    ) -> Result<()> {
        let problem = self.current_problem.as_ref().unwrap();
        let spin_mask = 1 << spin;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..state.len() {
            if i & spin_mask == 0 {
                let j = i | spin_mask;
                if j < state.len() {
                    let amp_0 = state[i];
                    let amp_1 = state[j];

                    state[i] = cos_half * amp_0 - Complex64::new(0.0, sin_half) * amp_1;
                    state[j] = cos_half * amp_1 - Complex64::new(0.0, sin_half) * amp_0;
                }
            }
        }

        Ok(())
    }

    /// Take annealing snapshot
    fn take_annealing_snapshot(
        &self,
        time: f64,
        s: f64,
        transverse_field: f64,
        longitudinal_field: f64,
        quantum_state: &Option<Array1<Complex64>>,
    ) -> Result<AnnealingSnapshot> {
        let energy_expectation = if let Some(state) = quantum_state {
            self.calculate_energy_expectation(state)?
        } else {
            0.0
        };

        let temperature_factor = (-1.0 / (1.38e-23 * self.config.temperature)).exp();

        Ok(AnnealingSnapshot {
            time,
            s,
            transverse_field,
            longitudinal_field,
            quantum_state: quantum_state.clone(),
            classical_probabilities: None,
            energy_expectation,
            temperature_factor,
        })
    }

    /// Calculate energy expectation value
    fn calculate_energy_expectation(&self, state: &Array1<Complex64>) -> Result<f64> {
        let problem = self.current_problem.as_ref().unwrap();
        let mut expectation = 0.0;

        for (i, &amplitude) in state.iter().enumerate() {
            let prob = amplitude.norm_sqr();

            // Convert state index to spin configuration
            let mut config = vec![-1; problem.num_spins];
            for spin in 0..problem.num_spins {
                if (i >> spin) & 1 == 1 {
                    config[spin] = 1;
                }
            }

            let energy = problem.calculate_energy(&config);
            expectation += prob * energy;
        }

        Ok(expectation)
    }

    /// Measure final quantum state
    fn measure_final_state(&self, state: &Array1<Complex64>) -> Result<Vec<i8>> {
        let problem = self.current_problem.as_ref().unwrap();

        // Sample from the quantum state probability distribution
        let probabilities: Vec<f64> = state.iter().map(|x| x.norm_sqr()).collect();
        let random_val = fastrand::f64();

        let mut cumulative_prob = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_val < cumulative_prob {
                // Convert state index to spin configuration
                let mut config = vec![-1; problem.num_spins];
                for spin in 0..problem.num_spins {
                    if (i >> spin) & 1 == 1 {
                        config[spin] = 1;
                    }
                }
                return Ok(config);
            }
        }

        // Fallback to ground state
        Ok(vec![-1; problem.num_spins])
    }

    /// Classical sampling for large problems
    fn classical_sampling(&self, problem: &IsingProblem) -> Result<Vec<i8>> {
        // Use simulated annealing or other classical heuristics
        let mut config: Vec<i8> = (0..problem.num_spins)
            .map(|_| if fastrand::f64() > 0.5 { 1 } else { -1 })
            .collect();

        // Simple local search
        for _ in 0..1000 {
            let spin_to_flip = fastrand::usize(0..problem.num_spins);
            let old_energy = problem.calculate_energy(&config);

            config[spin_to_flip] *= -1;
            let new_energy = problem.calculate_energy(&config);

            if new_energy > old_energy {
                config[spin_to_flip] *= -1; // Revert if energy increased
            }
        }

        Ok(config)
    }

    /// Apply majority vote post-processing
    fn apply_majority_vote_post_processing(&mut self) -> Result<()> {
        if self.solutions.is_empty() {
            return Ok(());
        }

        // Group solutions by configuration
        let mut config_groups: HashMap<Vec<i8>, Vec<usize>> = HashMap::new();
        for (i, solution) in self.solutions.iter().enumerate() {
            config_groups
                .entry(solution.configuration.clone())
                .or_default()
                .push(i);
        }

        // Update occurrence counts
        for (config, indices) in config_groups {
            let num_occurrences = indices.len();
            for &idx in &indices {
                self.solutions[idx].num_occurrences = num_occurrences;
            }
        }

        Ok(())
    }

    /// Apply local search post-processing
    fn apply_local_search_post_processing(&mut self) -> Result<()> {
        let problem = self.current_problem.as_ref().unwrap();

        for solution in &mut self.solutions {
            let mut improved_config = solution.configuration.clone();
            let mut improved_energy = solution.energy;

            for _ in 0..self.config.post_processing.max_local_search_iterations {
                let mut found_improvement = false;

                for spin in 0..problem.num_spins {
                    // Try flipping this spin
                    improved_config[spin] *= -1;
                    let new_energy = problem.calculate_energy(&improved_config);

                    if new_energy < improved_energy {
                        improved_energy = new_energy;
                        found_improvement = true;
                        break;
                    }
                    improved_config[spin] *= -1; // Revert
                }

                if !found_improvement {
                    break;
                }
            }

            // Update solution if improved
            if improved_energy < solution.energy {
                solution.configuration = improved_config;
                solution.energy = improved_energy;
            }
        }

        Ok(())
    }

    /// Compute annealing statistics
    fn compute_annealing_statistics(&mut self) -> Result<()> {
        if self.solutions.is_empty() {
            return Ok(());
        }

        self.stats.num_solutions_found = self.solutions.len();
        self.stats.best_energy_found = self
            .solutions
            .iter()
            .map(|s| s.energy)
            .fold(f64::INFINITY, f64::min);

        // Calculate success probability if ground state energy is known
        if let Some(optimal_energy) = self
            .current_problem
            .as_ref()
            .and_then(|p| p.metadata.optimal_energy)
        {
            let tolerance = 1e-6;
            let successful_solutions = self
                .solutions
                .iter()
                .filter(|s| (s.energy - optimal_energy).abs() < tolerance)
                .count();
            self.stats.success_probability =
                successful_solutions as f64 / self.solutions.len() as f64;
        }

        Ok(())
    }

    /// Get annealing statistics
    pub const fn get_stats(&self) -> &AnnealingStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = AnnealingStats::default();
    }
}

/// Annealing result
#[derive(Debug, Clone)]
pub struct AnnealingResult {
    /// All solutions found
    pub solutions: Vec<AnnealingSolution>,
    /// Best energy found
    pub best_energy: f64,
    /// Annealing evolution history
    pub annealing_history: Vec<AnnealingSnapshot>,
    /// Total computation time
    pub total_time_ms: f64,
    /// Success probability
    pub success_probability: f64,
    /// Time to solution statistics
    pub time_to_solution: TimeToSolutionStats,
}

/// Quantum annealing utilities
pub struct QuantumAnnealingUtils;

impl QuantumAnnealingUtils {
    /// Create Max-Cut Ising problem
    pub fn create_max_cut_problem(graph_edges: &[(usize, usize)], weights: &[f64]) -> IsingProblem {
        let num_vertices = graph_edges
            .iter()
            .flat_map(|&(u, v)| [u, v])
            .max()
            .unwrap_or(0)
            + 1;

        let mut problem = IsingProblem::new(num_vertices);
        problem.metadata.name = Some("Max-Cut".to_string());

        for (i, &(u, v)) in graph_edges.iter().enumerate() {
            let weight = weights.get(i).copied().unwrap_or(1.0);
            // Max-Cut: maximize ∑ w_{ij} (1 - s_i s_j) / 2
            // Equivalent to minimizing ∑ w_{ij} (s_i s_j - 1) / 2
            problem.set_j(u, v, weight / 2.0);
            problem.offset -= weight / 2.0;
        }

        problem
    }

    /// Create number partitioning problem
    pub fn create_number_partitioning_problem(numbers: &[f64]) -> IsingProblem {
        let n = numbers.len();
        let mut problem = IsingProblem::new(n);
        problem.metadata.name = Some("Number Partitioning".to_string());

        // Minimize (∑_i n_i s_i)^2 = ∑_i n_i^2 + 2 ∑_{i<j} n_i n_j s_i s_j
        for i in 0..n {
            problem.offset += numbers[i] * numbers[i];
            for j in i + 1..n {
                problem.set_j(i, j, 2.0 * numbers[i] * numbers[j]);
            }
        }

        problem
    }

    /// Create random Ising problem
    pub fn create_random_ising_problem(
        num_spins: usize,
        h_range: f64,
        j_range: f64,
    ) -> IsingProblem {
        let mut problem = IsingProblem::new(num_spins);
        problem.metadata.name = Some("Random Ising".to_string());

        // Random linear coefficients
        for i in 0..num_spins {
            problem.set_h(i, (fastrand::f64() - 0.5) * 2.0 * h_range);
        }

        // Random quadratic coefficients
        for i in 0..num_spins {
            for j in i + 1..num_spins {
                if fastrand::f64() < 0.5 {
                    // 50% sparsity
                    problem.set_j(i, j, (fastrand::f64() - 0.5) * 2.0 * j_range);
                }
            }
        }

        problem
    }

    /// Benchmark quantum annealing
    pub fn benchmark_quantum_annealing() -> Result<AnnealingBenchmarkResults> {
        let mut results = AnnealingBenchmarkResults::default();

        let problem_sizes = vec![8, 12, 16];
        let annealing_times = vec![1.0, 10.0, 100.0]; // μs

        for &size in &problem_sizes {
            for &time in &annealing_times {
                // Create random problem
                let problem = Self::create_random_ising_problem(size, 1.0, 1.0);

                let config = QuantumAnnealingConfig {
                    annealing_time: time,
                    time_steps: (time * 100.0) as usize,
                    topology: AnnealingTopology::Complete(size),
                    ..Default::default()
                };

                let mut simulator = QuantumAnnealingSimulator::new(config)?;
                simulator.set_problem(problem)?;

                let start = std::time::Instant::now();
                let result = simulator.anneal(100)?;
                let execution_time = start.elapsed().as_secs_f64() * 1000.0;

                results
                    .execution_times
                    .push((format!("{size}spins_{time}us"), execution_time));
                results
                    .best_energies
                    .push((format!("{size}spins_{time}us"), result.best_energy));
            }
        }

        Ok(results)
    }
}

/// Annealing benchmark results
#[derive(Debug, Clone, Default)]
pub struct AnnealingBenchmarkResults {
    /// Execution times by configuration
    pub execution_times: Vec<(String, f64)>,
    /// Best energies found
    pub best_energies: Vec<(String, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ising_problem_creation() {
        let mut problem = IsingProblem::new(3);
        problem.set_h(0, 0.5);
        problem.set_j(0, 1, -1.0);

        assert_eq!(problem.num_spins, 3);
        assert_eq!(problem.h[0], 0.5);
        assert_eq!(problem.j[[0, 1]], -1.0);
        assert_eq!(problem.j[[1, 0]], -1.0);
    }

    #[test]
    fn test_ising_energy_calculation() {
        let mut problem = IsingProblem::new(2);
        problem.set_h(0, 1.0);
        problem.set_h(1, -0.5);
        problem.set_j(0, 1, 2.0);

        let config = vec![1, -1];
        let energy = problem.calculate_energy(&config);
        // E = h_0 * s_0 + h_1 * s_1 + J_{01} * s_0 * s_1
        // E = 1.0 * 1 + (-0.5) * (-1) + 2.0 * 1 * (-1)
        // E = 1.0 + 0.5 - 2.0 = -0.5
        assert_abs_diff_eq!(energy, -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_ising_to_qubo_conversion() {
        let mut ising = IsingProblem::new(2);
        ising.set_h(0, 1.0);
        ising.set_j(0, 1, -1.0);

        let qubo = ising.to_qubo();
        assert_eq!(qubo.num_variables, 2);

        // Test energy equivalence for a configuration
        let ising_config = vec![1, -1];
        let qubo_config = vec![1, 0]; // s=1 -> x=1, s=-1 -> x=0

        let ising_energy = ising.calculate_energy(&ising_config);
        let qubo_energy = qubo.calculate_energy(&qubo_config);
        assert_abs_diff_eq!(ising_energy, qubo_energy, epsilon = 1e-10);
    }

    #[test]
    fn test_quantum_annealing_simulator_creation() {
        let config = QuantumAnnealingConfig::default();
        let simulator = QuantumAnnealingSimulator::new(config);
        assert!(simulator.is_ok());
    }

    #[test]
    fn test_schedule_functions() {
        let config = QuantumAnnealingConfig {
            annealing_time: 10.0,
            schedule_type: AnnealingScheduleType::Linear,
            ..Default::default()
        };
        let simulator = QuantumAnnealingSimulator::new(config).unwrap();

        assert_abs_diff_eq!(simulator.schedule_function(0.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(simulator.schedule_function(5.0), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(simulator.schedule_function(10.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_cut_problem_creation() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let weights = vec![1.0, 1.0, 1.0];

        let problem = QuantumAnnealingUtils::create_max_cut_problem(&edges, &weights);
        assert_eq!(problem.num_spins, 3);
        assert!(problem.metadata.name.as_ref().unwrap().contains("Max-Cut"));
    }

    #[test]
    fn test_number_partitioning_problem() {
        let numbers = vec![3.0, 1.0, 1.0, 2.0, 2.0, 1.0];
        let problem = QuantumAnnealingUtils::create_number_partitioning_problem(&numbers);

        assert_eq!(problem.num_spins, 6);
        assert!(problem
            .metadata
            .name
            .as_ref()
            .unwrap()
            .contains("Number Partitioning"));
    }

    #[test]
    fn test_small_problem_annealing() {
        let problem = QuantumAnnealingUtils::create_random_ising_problem(3, 1.0, 1.0);

        let config = QuantumAnnealingConfig {
            annealing_time: 1.0,
            time_steps: 100,
            topology: AnnealingTopology::Complete(3),
            enable_noise: false, // Disable for deterministic test
            ..Default::default()
        };

        let mut simulator = QuantumAnnealingSimulator::new(config).unwrap();
        simulator.set_problem(problem).unwrap();

        let result = simulator.anneal(10);
        assert!(result.is_ok());

        let annealing_result = result.unwrap();
        assert_eq!(annealing_result.solutions.len(), 10);
        assert!(!annealing_result.annealing_history.is_empty());
    }

    #[test]
    fn test_field_strength_calculation() {
        let config = QuantumAnnealingConfig::default();
        let simulator = QuantumAnnealingSimulator::new(config).unwrap();

        let (transverse, longitudinal) = simulator.calculate_field_strengths(0.0);
        assert_abs_diff_eq!(transverse, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(longitudinal, 0.0, epsilon = 1e-10);

        let (transverse, longitudinal) = simulator.calculate_field_strengths(1.0);
        assert_abs_diff_eq!(transverse, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(longitudinal, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_annealing_topologies() {
        let topologies = vec![
            AnnealingTopology::Chimera(4),
            AnnealingTopology::Pegasus(3),
            AnnealingTopology::Complete(5),
        ];

        for topology in topologies {
            let config = QuantumAnnealingConfig {
                topology,
                ..Default::default()
            };
            let simulator = QuantumAnnealingSimulator::new(config);
            assert!(simulator.is_ok());
        }
    }

    #[test]
    fn test_ising_ground_state_brute_force() {
        // Simple 2-spin ferromagnetic Ising model
        let mut problem = IsingProblem::new(2);
        problem.set_j(0, 1, -1.0); // Ferromagnetic coupling

        let (ground_state, ground_energy) = problem.find_ground_state_brute_force();

        // Ground states should be [1, 1] or [-1, -1] with energy -1
        assert!(ground_state == vec![1, 1] || ground_state == vec![-1, -1]);
        assert_abs_diff_eq!(ground_energy, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_post_processing_config() {
        let config = PostProcessingConfig::default();
        assert!(config.enable_spin_reversal);
        assert!(config.enable_local_search);
        assert!(config.enable_majority_vote);
        assert_eq!(config.majority_vote_reads, 1000);
    }
}
