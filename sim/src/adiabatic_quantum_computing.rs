//! Adiabatic quantum computing simulation with gap tracking and optimization.
//!
//! This module implements adiabatic quantum computation (AQC), a model of quantum
//! computation that uses the adiabatic theorem to solve optimization problems.
//! The system starts in the ground state of a simple Hamiltonian and slowly
//! evolves to a final Hamiltonian whose ground state encodes the solution.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use crate::statevector::StateVectorSimulator;
use crate::trotter::{Hamiltonian, HamiltonianTerm, TrotterDecomposer, TrotterMethod};

/// Adiabatic quantum computing configuration
#[derive(Debug, Clone)]
pub struct AdiabaticConfig {
    /// Total evolution time
    pub total_time: f64,
    /// Number of time steps
    pub time_steps: usize,
    /// Adiabatic schedule function type
    pub schedule_type: ScheduleType,
    /// Initial Hamiltonian
    pub initial_hamiltonian: Hamiltonian,
    /// Final Hamiltonian (problem Hamiltonian)
    pub final_hamiltonian: Hamiltonian,
    /// Gap tracking configuration
    pub gap_tracking: GapTrackingConfig,
    /// Energy convergence tolerance
    pub energy_tolerance: f64,
    /// Maximum iterations for eigenvalue solving
    pub max_iterations: usize,
    /// Enable adaptive time stepping
    pub adaptive_stepping: bool,
    /// Diabatic transition monitoring
    pub monitor_diabatic_transitions: bool,
}

impl Default for AdiabaticConfig {
    fn default() -> Self {
        Self {
            total_time: 100.0,
            time_steps: 1000,
            schedule_type: ScheduleType::Linear,
            initial_hamiltonian: Hamiltonian::new(1), // Default to 1 qubit
            final_hamiltonian: Hamiltonian::new(1),
            gap_tracking: GapTrackingConfig::default(),
            energy_tolerance: 1e-12,
            max_iterations: 1000,
            adaptive_stepping: true,
            monitor_diabatic_transitions: false,
        }
    }
}

/// Adiabatic schedule types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleType {
    /// Linear interpolation s(t) = t/T
    Linear,
    /// Quadratic schedule s(t) = (t/T)²
    Quadratic,
    /// Cubic schedule s(t) = (t/T)³
    Cubic,
    /// Exponential schedule
    Exponential,
    /// Optimal schedule based on gap
    Optimal,
    /// Custom polynomial schedule
    Polynomial(u32),
    /// Landau-Zener schedule
    LandauZener,
}

/// Gap tracking configuration
#[derive(Debug, Clone)]
pub struct GapTrackingConfig {
    /// Enable gap tracking
    pub enabled: bool,
    /// Minimum gap threshold for adiabatic condition
    pub min_gap_threshold: f64,
    /// Number of eigenvalues to track
    pub num_eigenvalues: usize,
    /// Gap smoothing window size
    pub smoothing_window: usize,
    /// Enable diabatic transition detection
    pub detect_diabatic_transitions: bool,
    /// Gap prediction lookahead steps
    pub lookahead_steps: usize,
}

impl Default for GapTrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_gap_threshold: 1e-6,
            num_eigenvalues: 10,
            smoothing_window: 5,
            detect_diabatic_transitions: true,
            lookahead_steps: 3,
        }
    }
}

/// Adiabatic quantum computer simulator
pub struct AdiabaticQuantumComputer {
    /// Configuration
    config: AdiabaticConfig,
    /// Current quantum state
    state: Array1<Complex64>,
    /// Current time parameter
    current_time: f64,
    /// Evolution history
    evolution_history: Vec<AdiabaticSnapshot>,
    /// Gap tracking data
    gap_history: Vec<GapMeasurement>,
    /// SciRS2 backend for optimization
    backend: Option<SciRS2Backend>,
    /// Statistics
    stats: AdiabaticStats,
}

/// Snapshot of adiabatic evolution
#[derive(Debug, Clone)]
pub struct AdiabaticSnapshot {
    /// Time parameter t
    pub time: f64,
    /// Schedule parameter s(t)
    pub schedule_parameter: f64,
    /// Current quantum state
    pub state: Array1<Complex64>,
    /// Current energy
    pub energy: f64,
    /// Energy gap
    pub gap: Option<f64>,
    /// Instantaneous ground state
    pub instantaneous_ground_state: Option<Array1<Complex64>>,
    /// Fidelity with instantaneous ground state
    pub ground_state_fidelity: Option<f64>,
    /// Adiabatic parameter (gap²T/ℏ)
    pub adiabatic_parameter: Option<f64>,
}

/// Gap measurement data
#[derive(Debug, Clone)]
pub struct GapMeasurement {
    /// Time
    pub time: f64,
    /// Schedule parameter
    pub schedule_parameter: f64,
    /// Energy gap
    pub gap: f64,
    /// Ground state energy
    pub ground_energy: f64,
    /// First excited state energy
    pub first_excited_energy: f64,
    /// Gap derivative (dΔ/dt)
    pub gap_derivative: Option<f64>,
    /// Predicted minimum gap
    pub predicted_min_gap: Option<f64>,
}

/// Adiabatic simulation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdiabaticStats {
    /// Total evolution time
    pub total_evolution_time_ms: f64,
    /// Number of time steps completed
    pub steps_completed: usize,
    /// Number of eigenvalue computations
    pub eigenvalue_computations: usize,
    /// Average eigenvalue computation time
    pub avg_eigenvalue_time_ms: f64,
    /// Minimum gap encountered
    pub min_gap: f64,
    /// Maximum gap encountered
    pub max_gap: f64,
    /// Average gap
    pub avg_gap: f64,
    /// Number of diabatic transitions detected
    pub diabatic_transitions: usize,
    /// Final ground state fidelity
    pub final_ground_state_fidelity: f64,
    /// Success probability (for optimization problems)
    pub success_probability: f64,
}

impl AdiabaticQuantumComputer {
    /// Create new adiabatic quantum computer
    pub fn new(config: AdiabaticConfig) -> Result<Self> {
        // Initialize state in ground state of initial Hamiltonian
        let num_qubits = config.initial_hamiltonian.get_num_qubits();
        let state_size = 1 << num_qubits;

        let mut state = Array1::zeros(state_size);
        state[0] = Complex64::new(1.0, 0.0); // Start with |0...0⟩

        Ok(Self {
            config,
            state,
            current_time: 0.0,
            evolution_history: Vec::new(),
            gap_history: Vec::new(),
            backend: None,
            stats: AdiabaticStats::default(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Set initial state to ground state of initial Hamiltonian
    pub fn initialize_ground_state(&mut self) -> Result<()> {
        let initial_matrix = self.build_hamiltonian_matrix(&self.config.initial_hamiltonian)?;
        let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&initial_matrix)?;

        // Find ground state (lowest eigenvalue)
        let ground_idx = eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Set state to ground state
        self.state = eigenvectors.column(ground_idx).to_owned();

        Ok(())
    }

    /// Run adiabatic evolution
    pub fn evolve(&mut self) -> Result<AdiabaticResult> {
        let start_time = std::time::Instant::now();

        // Initialize in ground state
        self.initialize_ground_state()?;

        // Take initial snapshot
        let initial_snapshot = self.take_snapshot(0.0)?;
        self.evolution_history.push(initial_snapshot);

        let dt = self.config.total_time / self.config.time_steps as f64;

        for step in 1..=self.config.time_steps {
            let step_start = std::time::Instant::now();

            let t = step as f64 * dt;
            let s = self.schedule_function(t);

            // Adaptive time stepping based on gap
            let actual_dt = if self.config.adaptive_stepping {
                self.calculate_adaptive_timestep(t, dt)?
            } else {
                dt
            };

            // Build interpolated Hamiltonian
            let hamiltonian = self.interpolate_hamiltonian(s)?;

            // Evolve state
            self.evolve_step(&hamiltonian, actual_dt)?;

            // Track gap if enabled
            if self.config.gap_tracking.enabled {
                let gap_measurement = self.measure_gap(t, s, &hamiltonian)?;

                // Check for diabatic transitions
                if self.config.monitor_diabatic_transitions {
                    self.check_diabatic_transition(&gap_measurement)?;
                }

                self.gap_history.push(gap_measurement);
            }

            // Take snapshot
            if step % 10 == 0 || step == self.config.time_steps {
                let snapshot = self.take_snapshot(t)?;
                self.evolution_history.push(snapshot);
            }

            self.current_time = t;
            self.stats.steps_completed += 1;

            let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "Step {}/{}: t={:.3}, s={:.3}, time={:.2}ms",
                step, self.config.time_steps, t, s, step_time
            );
        }

        // Compute final statistics
        self.compute_final_statistics()?;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_evolution_time_ms = total_time;

        Ok(AdiabaticResult {
            final_state: self.state.clone(),
            evolution_history: self.evolution_history.clone(),
            gap_history: self.gap_history.clone(),
            total_time_ms: total_time,
            success_probability: self.stats.success_probability,
            min_gap: self.stats.min_gap,
            final_energy: self.calculate_current_energy()?,
        })
    }

    /// Schedule function s(t)
    fn schedule_function(&self, t: f64) -> f64 {
        let s = t / self.config.total_time;

        match self.config.schedule_type {
            ScheduleType::Linear => s,
            ScheduleType::Quadratic => s * s,
            ScheduleType::Cubic => s * s * s,
            ScheduleType::Exponential => (s.exp() - 1.0) / (1f64.exp() - 1.0),
            ScheduleType::Polynomial(n) => s.powi(n as i32),
            ScheduleType::LandauZener => {
                // Landau-Zener formula: optimized for avoiding diabatic transitions
                if s < 0.5 {
                    2.0 * s * s
                } else {
                    1.0 - 2.0 * (1.0 - s) * (1.0 - s)
                }
            }
            ScheduleType::Optimal => {
                // Would implement optimal control based on gap
                s // Fallback to linear for now
            }
        }
    }

    /// Interpolate between initial and final Hamiltonians
    fn interpolate_hamiltonian(&self, s: f64) -> Result<Hamiltonian> {
        let num_qubits = self
            .config
            .initial_hamiltonian
            .get_num_qubits()
            .max(self.config.final_hamiltonian.get_num_qubits());
        let mut interpolated = Hamiltonian::new(num_qubits);

        // H(s) = (1-s) * H_initial + s * H_final
        for term in &self.config.initial_hamiltonian.terms {
            let scaled_term = match term {
                HamiltonianTerm::SinglePauli {
                    qubit,
                    pauli,
                    coefficient,
                } => HamiltonianTerm::SinglePauli {
                    qubit: *qubit,
                    pauli: pauli.clone(),
                    coefficient: coefficient * (1.0 - s),
                },
                HamiltonianTerm::TwoPauli {
                    qubit1,
                    qubit2,
                    pauli1,
                    pauli2,
                    coefficient,
                } => HamiltonianTerm::TwoPauli {
                    qubit1: *qubit1,
                    qubit2: *qubit2,
                    pauli1: pauli1.clone(),
                    pauli2: pauli2.clone(),
                    coefficient: coefficient * (1.0 - s),
                },
                HamiltonianTerm::PauliString {
                    qubits,
                    paulis,
                    coefficient,
                } => HamiltonianTerm::PauliString {
                    qubits: qubits.clone(),
                    paulis: paulis.clone(),
                    coefficient: coefficient * (1.0 - s),
                },
                HamiltonianTerm::Custom {
                    qubits,
                    matrix,
                    coefficient,
                } => HamiltonianTerm::Custom {
                    qubits: qubits.clone(),
                    matrix: matrix.clone(),
                    coefficient: coefficient * (1.0 - s),
                },
            };
            interpolated.add_term(scaled_term);
        }

        for term in &self.config.final_hamiltonian.terms {
            let scaled_term = match term {
                HamiltonianTerm::SinglePauli {
                    qubit,
                    pauli,
                    coefficient,
                } => HamiltonianTerm::SinglePauli {
                    qubit: *qubit,
                    pauli: pauli.clone(),
                    coefficient: coefficient * s,
                },
                HamiltonianTerm::TwoPauli {
                    qubit1,
                    qubit2,
                    pauli1,
                    pauli2,
                    coefficient,
                } => HamiltonianTerm::TwoPauli {
                    qubit1: *qubit1,
                    qubit2: *qubit2,
                    pauli1: pauli1.clone(),
                    pauli2: pauli2.clone(),
                    coefficient: coefficient * s,
                },
                HamiltonianTerm::PauliString {
                    qubits,
                    paulis,
                    coefficient,
                } => HamiltonianTerm::PauliString {
                    qubits: qubits.clone(),
                    paulis: paulis.clone(),
                    coefficient: coefficient * s,
                },
                HamiltonianTerm::Custom {
                    qubits,
                    matrix,
                    coefficient,
                } => HamiltonianTerm::Custom {
                    qubits: qubits.clone(),
                    matrix: matrix.clone(),
                    coefficient: coefficient * s,
                },
            };
            interpolated.add_term(scaled_term);
        }

        Ok(interpolated)
    }

    /// Evolve system for one time step
    fn evolve_step(&mut self, hamiltonian: &Hamiltonian, dt: f64) -> Result<()> {
        // Build Hamiltonian matrix
        let h_matrix = self.build_hamiltonian_matrix(hamiltonian)?;

        // Compute evolution operator U = exp(-i H dt / ℏ)
        let evolution_operator = self.compute_evolution_operator(&h_matrix, dt)?;

        // Apply evolution operator to state
        self.state = evolution_operator.dot(&self.state);

        // Renormalize (to handle numerical errors)
        let norm: f64 = self.state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-15 {
            self.state.mapv_inplace(|x| x / norm);
        }

        Ok(())
    }

    /// Build Hamiltonian matrix from Hamiltonian terms
    fn build_hamiltonian_matrix(&self, hamiltonian: &Hamiltonian) -> Result<Array2<Complex64>> {
        let num_qubits = hamiltonian.get_num_qubits();
        let dim = 1 << num_qubits;
        let mut matrix = Array2::zeros((dim, dim));

        for term in &hamiltonian.terms {
            let (term_matrix, coefficient) = self.build_term_matrix(term, num_qubits)?;
            matrix = matrix + term_matrix.mapv(|x| x * coefficient);
        }

        Ok(matrix)
    }

    /// Build matrix for a single Hamiltonian term
    fn build_term_matrix(
        &self,
        term: &HamiltonianTerm,
        num_qubits: usize,
    ) -> Result<(Array2<Complex64>, f64)> {
        let dim = 1 << num_qubits;

        match term {
            HamiltonianTerm::SinglePauli {
                qubit,
                pauli,
                coefficient,
            } => {
                let mut matrix = Array2::eye(dim);
                let pauli_matrix = self.get_pauli_matrix(pauli)?;
                matrix = self.apply_single_qubit_to_full_matrix(
                    &matrix,
                    &pauli_matrix,
                    *qubit,
                    num_qubits,
                )?;
                Ok((matrix, *coefficient))
            }
            HamiltonianTerm::TwoPauli {
                qubit1,
                qubit2,
                pauli1,
                pauli2,
                coefficient,
            } => {
                let mut matrix = Array2::eye(dim);
                let pauli1_matrix = self.get_pauli_matrix(pauli1)?;
                let pauli2_matrix = self.get_pauli_matrix(pauli2)?;

                matrix = self.apply_single_qubit_to_full_matrix(
                    &matrix,
                    &pauli1_matrix,
                    *qubit1,
                    num_qubits,
                )?;
                matrix = self.apply_single_qubit_to_full_matrix(
                    &matrix,
                    &pauli2_matrix,
                    *qubit2,
                    num_qubits,
                )?;
                Ok((matrix, *coefficient))
            }
            HamiltonianTerm::PauliString {
                qubits,
                paulis,
                coefficient,
            } => {
                let mut matrix = Array2::eye(dim);

                for (qubit, pauli) in qubits.iter().zip(paulis.iter()) {
                    let pauli_matrix = self.get_pauli_matrix(pauli)?;
                    matrix = self.apply_single_qubit_to_full_matrix(
                        &matrix,
                        &pauli_matrix,
                        *qubit,
                        num_qubits,
                    )?;
                }
                Ok((matrix, *coefficient))
            }
            HamiltonianTerm::Custom {
                qubits: _,
                matrix: _,
                coefficient,
            } => {
                // For now, return identity with the coefficient
                Ok((Array2::eye(dim), *coefficient))
            }
        }
    }

    /// Get Pauli matrix for a given Pauli string
    fn get_pauli_matrix(&self, pauli: &str) -> Result<Array2<Complex64>> {
        match pauli.to_uppercase().as_str() {
            "I" => Ok(Array2::eye(2)),
            "X" => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap()),
            "Y" => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap()),
            "Z" => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .unwrap()),
            _ => Err(SimulatorError::InvalidInput(format!(
                "Unknown Pauli operator: {}",
                pauli
            ))),
        }
    }

    /// Apply single-qubit operator to full system matrix
    fn apply_single_qubit_to_full_matrix(
        &self,
        full_matrix: &Array2<Complex64>,
        single_qubit_op: &Array2<Complex64>,
        target_qubit: usize,
        num_qubits: usize,
    ) -> Result<Array2<Complex64>> {
        let dim = 1 << num_qubits;
        let mut result = Array2::zeros((dim, dim));

        for i in 0..dim {
            for j in 0..dim {
                let i_bit = (i >> target_qubit) & 1;
                let j_bit = (j >> target_qubit) & 1;

                result[[i, j]] = full_matrix[[i, j]] * single_qubit_op[[i_bit, j_bit]];
            }
        }

        Ok(result)
    }

    /// Compute evolution operator exp(-i H dt / ℏ)
    fn compute_evolution_operator(
        &self,
        hamiltonian: &Array2<Complex64>,
        dt: f64,
    ) -> Result<Array2<Complex64>> {
        // For small systems, use matrix exponentiation
        // For larger systems, would use Trotterization

        let dim = hamiltonian.dim().0;
        if dim <= 64 {
            // Direct matrix exponentiation
            self.matrix_exponential(hamiltonian, -Complex64::new(0.0, dt))
        } else {
            // Use Trotter decomposition for larger systems
            self.trotter_evolution(hamiltonian, dt)
        }
    }

    /// Direct matrix exponentiation (for small systems)
    fn matrix_exponential(
        &self,
        matrix: &Array2<Complex64>,
        factor: Complex64,
    ) -> Result<Array2<Complex64>> {
        let dim = matrix.dim().0;

        // Scale matrix
        let scaled_matrix = matrix.mapv(|x| x * factor);

        // Use series expansion: exp(A) = I + A + A²/2! + A³/3! + ...
        let mut result = Array2::eye(dim);
        let mut term = Array2::eye(dim);

        for n in 1..=20 {
            // Limit iterations for convergence
            term = term.dot(&scaled_matrix) / (n as f64);
            let term_norm: f64 = term.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();

            result = result + &term;

            // Check convergence
            if term_norm < 1e-15 {
                break;
            }
        }

        Ok(result)
    }

    /// Trotter evolution for large systems
    fn trotter_evolution(
        &self,
        hamiltonian: &Array2<Complex64>,
        dt: f64,
    ) -> Result<Array2<Complex64>> {
        // Placeholder: would implement proper Trotter decomposition
        self.matrix_exponential(hamiltonian, -Complex64::new(0.0, dt))
    }

    /// Measure energy gap
    fn measure_gap(&mut self, t: f64, s: f64, hamiltonian: &Hamiltonian) -> Result<GapMeasurement> {
        let start_time = std::time::Instant::now();

        let h_matrix = self.build_hamiltonian_matrix(hamiltonian)?;
        let (eigenvalues, _) = self.compute_eigendecomposition(&h_matrix)?;

        // Sort eigenvalues
        let mut sorted_eigenvalues = eigenvalues;
        sorted_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ground_energy = sorted_eigenvalues[0];
        let first_excited_energy = sorted_eigenvalues.get(1).copied().unwrap_or(ground_energy);
        let gap = first_excited_energy - ground_energy;

        // Update statistics
        if self.stats.min_gap == 0.0 || gap < self.stats.min_gap {
            self.stats.min_gap = gap;
        }
        if gap > self.stats.max_gap {
            self.stats.max_gap = gap;
        }

        let gap_count = self.gap_history.len() as f64;
        self.stats.avg_gap = (self.stats.avg_gap * gap_count + gap) / (gap_count + 1.0);

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.avg_eigenvalue_time_ms = (self.stats.avg_eigenvalue_time_ms
            * self.stats.eigenvalue_computations as f64
            + computation_time)
            / (self.stats.eigenvalue_computations + 1) as f64;
        self.stats.eigenvalue_computations += 1;

        Ok(GapMeasurement {
            time: t,
            schedule_parameter: s,
            gap,
            ground_energy,
            first_excited_energy,
            gap_derivative: self.estimate_gap_derivative(gap),
            predicted_min_gap: self.predict_minimum_gap(),
        })
    }

    /// Estimate gap derivative
    fn estimate_gap_derivative(&self, current_gap: f64) -> Option<f64> {
        if self.gap_history.len() < 2 {
            return None;
        }

        let prev_gap = self.gap_history.last().unwrap().gap;
        let prev_time = self.gap_history.last().unwrap().time;
        let dt = self.current_time - prev_time;

        if dt > 1e-15 {
            Some((current_gap - prev_gap) / dt)
        } else {
            None
        }
    }

    /// Predict minimum gap using extrapolation
    fn predict_minimum_gap(&self) -> Option<f64> {
        if self.gap_history.len() < 3 {
            return None;
        }

        // Simple quadratic extrapolation
        let n = self.gap_history.len();
        let recent_gaps: Vec<f64> = self.gap_history[n - 3..].iter().map(|g| g.gap).collect();
        let recent_times: Vec<f64> = self.gap_history[n - 3..].iter().map(|g| g.time).collect();

        // Fit quadratic and find minimum
        // For simplicity, just return the minimum of recent measurements
        recent_gaps.into_iter().fold(f64::INFINITY, f64::min).into()
    }

    /// Check for diabatic transitions
    fn check_diabatic_transition(&mut self, gap_measurement: &GapMeasurement) -> Result<()> {
        // Landau-Zener criterion: P_diabatic ≈ exp(-2π Δ²/(ℏ |dH/dt|))
        if let Some(gap_derivative) = gap_measurement.gap_derivative {
            let gap = gap_measurement.gap;
            let dt = self.config.total_time / self.config.time_steps as f64;

            // Estimate diabatic transition probability
            let diabatic_prob = if gap_derivative.abs() > 1e-15 {
                (-2.0 * std::f64::consts::PI * gap * gap / gap_derivative.abs()).exp()
            } else {
                0.0
            };

            // Threshold for detecting diabatic transition
            if diabatic_prob > 0.01 {
                // 1% threshold
                self.stats.diabatic_transitions += 1;
                println!(
                    "Warning: Potential diabatic transition detected at t={:.3}, P_diabatic={:.4}",
                    gap_measurement.time, diabatic_prob
                );
            }
        }

        Ok(())
    }

    /// Calculate adaptive time step based on gap
    fn calculate_adaptive_timestep(&self, t: f64, default_dt: f64) -> Result<f64> {
        if self.gap_history.is_empty() {
            return Ok(default_dt);
        }

        let current_gap = self.gap_history.last().unwrap().gap;

        // Smaller time steps when gap is small
        let gap_factor = (current_gap / self.config.gap_tracking.min_gap_threshold).sqrt();
        let adaptive_dt = default_dt * gap_factor.min(2.0).max(0.1); // Clamp between 0.1 and 2.0 times default

        Ok(adaptive_dt)
    }

    /// Take snapshot of current state
    fn take_snapshot(&mut self, t: f64) -> Result<AdiabaticSnapshot> {
        let s = self.schedule_function(t);
        let energy = self.calculate_current_energy()?;

        // Get current gap
        let gap = self.gap_history.last().map(|g| g.gap);

        // Calculate instantaneous ground state if gap tracking is enabled
        let (instantaneous_ground_state, ground_state_fidelity, adiabatic_parameter) =
            if self.config.gap_tracking.enabled {
                let hamiltonian = self.interpolate_hamiltonian(s)?;
                let h_matrix = self.build_hamiltonian_matrix(&hamiltonian)?;
                let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&h_matrix)?;

                // Find ground state
                let ground_idx = eigenvalues
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let ground_state = eigenvectors.column(ground_idx).to_owned();

                // Calculate fidelity
                let fidelity = self.calculate_fidelity(&self.state, &ground_state);

                // Calculate adiabatic parameter
                let adiabatic_param = gap.map(|g| g * g * self.config.total_time);

                (Some(ground_state), Some(fidelity), adiabatic_param)
            } else {
                (None, None, None)
            };

        Ok(AdiabaticSnapshot {
            time: t,
            schedule_parameter: s,
            state: self.state.clone(),
            energy,
            gap,
            instantaneous_ground_state,
            ground_state_fidelity,
            adiabatic_parameter,
        })
    }

    /// Calculate current energy expectation value
    fn calculate_current_energy(&self) -> Result<f64> {
        let s = self.schedule_function(self.current_time);
        let hamiltonian = self.interpolate_hamiltonian(s)?;
        let h_matrix = self.build_hamiltonian_matrix(&hamiltonian)?;

        // E = ⟨ψ|H|ψ⟩
        let h_psi = h_matrix.dot(&self.state);
        let energy: Complex64 = self
            .state
            .iter()
            .zip(h_psi.iter())
            .map(|(psi, h_psi)| psi.conj() * h_psi)
            .sum();

        Ok(energy.re)
    }

    /// Calculate fidelity between two states
    fn calculate_fidelity(&self, state1: &Array1<Complex64>, state2: &Array1<Complex64>) -> f64 {
        let overlap: Complex64 = state1
            .iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        overlap.norm_sqr()
    }

    /// Compute eigendecomposition
    fn compute_eigendecomposition(
        &self,
        matrix: &Array2<Complex64>,
    ) -> Result<(Vec<f64>, Array2<Complex64>)> {
        // Simplified eigenvalue computation
        // In practice, would use LAPACK or similar high-performance library

        let dim = matrix.dim().0;
        if dim > 16 {
            // For large matrices, use iterative methods or approximations
            return self.compute_approximate_eigenvalues(matrix);
        }

        // For small matrices, use simple power iteration for dominant eigenvalue
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Array2::eye(dim);

        // Simplified: just compute diagonal elements as eigenvalue approximation
        for i in 0..dim {
            eigenvalues.push(matrix[[i, i]].re);
        }

        // Sort eigenvalues
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute approximate eigenvalues for large matrices
    fn compute_approximate_eigenvalues(
        &self,
        matrix: &Array2<Complex64>,
    ) -> Result<(Vec<f64>, Array2<Complex64>)> {
        let dim = matrix.dim().0;

        // Use Lanczos algorithm or similar for large sparse matrices
        // For now, just return diagonal approximation
        let mut eigenvalues = Vec::new();
        for i in 0..dim.min(self.config.gap_tracking.num_eigenvalues) {
            eigenvalues.push(matrix[[i, i]].re);
        }

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let eigenvectors = Array2::eye(dim);

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute final statistics
    fn compute_final_statistics(&mut self) -> Result<()> {
        // Calculate final ground state fidelity
        if let Some(final_snapshot) = self.evolution_history.last() {
            if let Some(fidelity) = final_snapshot.ground_state_fidelity {
                self.stats.final_ground_state_fidelity = fidelity;
            }
        }

        // For optimization problems, estimate success probability
        // This would depend on the specific problem encoding
        self.stats.success_probability = self.stats.final_ground_state_fidelity;

        Ok(())
    }

    /// Get current state
    pub fn get_state(&self) -> &Array1<Complex64> {
        &self.state
    }

    /// Get evolution history
    pub fn get_evolution_history(&self) -> &[AdiabaticSnapshot] {
        &self.evolution_history
    }

    /// Get gap history
    pub fn get_gap_history(&self) -> &[GapMeasurement] {
        &self.gap_history
    }

    /// Get statistics
    pub fn get_stats(&self) -> &AdiabaticStats {
        &self.stats
    }

    /// Reset the simulator
    pub fn reset(&mut self) -> Result<()> {
        let num_qubits = self.config.initial_hamiltonian.get_num_qubits();
        let state_size = 1 << num_qubits;

        self.state = Array1::zeros(state_size);
        self.state[0] = Complex64::new(1.0, 0.0);
        self.current_time = 0.0;
        self.evolution_history.clear();
        self.gap_history.clear();
        self.stats = AdiabaticStats::default();

        Ok(())
    }
}

/// Adiabatic evolution result
#[derive(Debug, Clone)]
pub struct AdiabaticResult {
    /// Final quantum state
    pub final_state: Array1<Complex64>,
    /// Complete evolution history
    pub evolution_history: Vec<AdiabaticSnapshot>,
    /// Gap tracking history
    pub gap_history: Vec<GapMeasurement>,
    /// Total evolution time in milliseconds
    pub total_time_ms: f64,
    /// Success probability
    pub success_probability: f64,
    /// Minimum gap encountered
    pub min_gap: f64,
    /// Final energy
    pub final_energy: f64,
}

/// Adiabatic quantum computing utilities
pub struct AdiabaticUtils;

impl AdiabaticUtils {
    /// Create Max-Cut problem Hamiltonian
    pub fn create_max_cut_hamiltonian(
        graph_edges: &[(usize, usize)],
        weights: &[f64],
    ) -> Hamiltonian {
        let max_vertex = graph_edges
            .iter()
            .flat_map(|&(u, v)| [u, v])
            .max()
            .unwrap_or(0)
            + 1;
        let mut hamiltonian = Hamiltonian::new(max_vertex);

        for (i, &(u, v)) in graph_edges.iter().enumerate() {
            let weight = weights.get(i).copied().unwrap_or(1.0);

            // Add term: w_ij * (I - Z_i Z_j) / 2
            // For Max-Cut, we want to maximize the number of edges cut
            // This corresponds to minimizing -weight/2 * (1 - Z_i Z_j) = -weight/2 + weight/2 * Z_i Z_j
            // So we add the ZZ interaction term with positive coefficient
            hamiltonian
                .add_two_pauli(u, v, "Z", "Z", weight / 2.0)
                .unwrap();
        }

        hamiltonian
    }

    /// Create 3-SAT problem Hamiltonian
    pub fn create_3sat_hamiltonian(clauses: &[Vec<i32>]) -> Hamiltonian {
        let max_var = clauses
            .iter()
            .flat_map(|clause| clause.iter())
            .map(|&lit| lit.abs() as usize)
            .max()
            .unwrap_or(0)
            + 1;
        let mut hamiltonian = Hamiltonian::new(max_var);

        for clause in clauses {
            if clause.len() != 3 {
                continue; // Skip non-3-SAT clauses
            }

            // For clause (x_i ∨ x_j ∨ x_k), add penalty when all literals are false
            // This is a simplified implementation - in practice would need more sophisticated encoding

            // Add penalty for clause being unsatisfied
            // For now, add pairwise interactions between variables in the clause
            for i in 0..clause.len() {
                for j in i + 1..clause.len() {
                    let var1 = clause[i].abs() as usize;
                    let var2 = clause[j].abs() as usize;

                    // Add weak coupling between variables in the same clause
                    if var1 < max_var && var2 < max_var {
                        hamiltonian
                            .add_two_pauli(var1, var2, "Z", "Z", 0.1)
                            .unwrap();
                    }
                }
            }
        }

        hamiltonian
    }

    /// Create transverse field Ising model (TFIM) Hamiltonian
    pub fn create_tfim_hamiltonian(
        num_qubits: usize,
        j_coupling: f64,
        h_field: f64,
    ) -> Hamiltonian {
        let mut hamiltonian = Hamiltonian::new(num_qubits);

        // ZZ coupling terms
        for i in 0..num_qubits - 1 {
            hamiltonian
                .add_two_pauli(i, i + 1, "Z", "Z", -j_coupling)
                .unwrap();
        }

        // X field terms
        for i in 0..num_qubits {
            hamiltonian.add_single_pauli(i, "X", -h_field).unwrap();
        }

        hamiltonian
    }

    /// Create mixing Hamiltonian (typically all X)
    pub fn create_mixing_hamiltonian(num_qubits: usize) -> Hamiltonian {
        let mut hamiltonian = Hamiltonian::new(num_qubits);

        for i in 0..num_qubits {
            hamiltonian.add_single_pauli(i, "X", 1.0).unwrap();
        }

        hamiltonian
    }

    /// Benchmark adiabatic quantum computing
    pub fn benchmark_adiabatic_qc() -> Result<AdiabaticBenchmarkResults> {
        let mut results = AdiabaticBenchmarkResults::default();

        // Test different problem sizes and schedules
        let problem_sizes = vec![4, 6, 8];
        let schedule_types = vec![
            ScheduleType::Linear,
            ScheduleType::Quadratic,
            ScheduleType::LandauZener,
        ];

        for &num_qubits in &problem_sizes {
            for &schedule_type in &schedule_types {
                // Create simple TFIM problem
                let initial_h = Self::create_mixing_hamiltonian(num_qubits);
                let final_h = Self::create_tfim_hamiltonian(num_qubits, 1.0, 0.1);

                let config = AdiabaticConfig {
                    total_time: 10.0,
                    time_steps: 100,
                    schedule_type,
                    initial_hamiltonian: initial_h,
                    final_hamiltonian: final_h,
                    gap_tracking: GapTrackingConfig {
                        enabled: true,
                        min_gap_threshold: 1e-3,
                        ..Default::default()
                    },
                    ..Default::default()
                };

                let mut adiabatic_qc = AdiabaticQuantumComputer::new(config)?;

                let start = std::time::Instant::now();
                let result = adiabatic_qc.evolve()?;
                let execution_time = start.elapsed().as_secs_f64() * 1000.0;

                let key = format!("{}q_{:?}", num_qubits, schedule_type);
                results.execution_times.push((key.clone(), execution_time));
                results
                    .success_probabilities
                    .push((key.clone(), result.success_probability));
                results.min_gaps.push((key, result.min_gap));
            }
        }

        Ok(results)
    }
}

/// Adiabatic benchmark results
#[derive(Debug, Clone, Default)]
pub struct AdiabaticBenchmarkResults {
    /// Execution times by configuration
    pub execution_times: Vec<(String, f64)>,
    /// Success probabilities by configuration
    pub success_probabilities: Vec<(String, f64)>,
    /// Minimum gaps by configuration
    pub min_gaps: Vec<(String, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_adiabatic_qc_creation() {
        let config = AdiabaticConfig::default();
        let adiabatic_qc = AdiabaticQuantumComputer::new(config);
        assert!(adiabatic_qc.is_ok());
    }

    #[test]
    fn test_schedule_functions() {
        let config = AdiabaticConfig {
            total_time: 10.0,
            schedule_type: ScheduleType::Linear,
            ..Default::default()
        };
        let adiabatic_qc = AdiabaticQuantumComputer::new(config).unwrap();

        assert_abs_diff_eq!(adiabatic_qc.schedule_function(0.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(adiabatic_qc.schedule_function(5.0), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(adiabatic_qc.schedule_function(10.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hamiltonian_interpolation() {
        let mut initial_h = Hamiltonian::new(1);
        initial_h.add_pauli_term(1.0, &[(0, 'X')]).unwrap();

        let mut final_h = Hamiltonian::new(1);
        final_h.add_pauli_term(1.0, &[(0, 'Z')]).unwrap();

        let config = AdiabaticConfig {
            initial_hamiltonian: initial_h,
            final_hamiltonian: final_h,
            ..Default::default()
        };

        let adiabatic_qc = AdiabaticQuantumComputer::new(config).unwrap();

        let h_mid = adiabatic_qc.interpolate_hamiltonian(0.5).unwrap();
        assert_eq!(h_mid.terms.len(), 2); // Should have both X and Z terms
    }

    #[test]
    fn test_tfim_hamiltonian() {
        let hamiltonian = AdiabaticUtils::create_tfim_hamiltonian(3, 1.0, 0.5);

        // Should have ZZ coupling terms and X field terms
        let num_zz_terms = hamiltonian.terms.iter().filter(|t| {
            matches!(t, HamiltonianTerm::TwoPauli { pauli1, pauli2, .. } if pauli1 == "Z" && pauli2 == "Z")
        }).count();

        let num_x_terms = hamiltonian
            .terms
            .iter()
            .filter(|t| matches!(t, HamiltonianTerm::SinglePauli { pauli, .. } if pauli == "X"))
            .count();

        assert_eq!(num_zz_terms, 2); // 2 ZZ coupling terms for 3 qubits
        assert_eq!(num_x_terms, 3); // 3 X field terms
    }

    #[test]
    fn test_max_cut_hamiltonian() {
        let edges = vec![(0, 1), (1, 2), (2, 0)]; // Triangle graph
        let weights = vec![1.0, 1.0, 1.0];

        let hamiltonian = AdiabaticUtils::create_max_cut_hamiltonian(&edges, &weights);
        assert!(!hamiltonian.terms.is_empty());
    }

    #[test]
    fn test_mixing_hamiltonian() {
        let hamiltonian = AdiabaticUtils::create_mixing_hamiltonian(2);

        let num_x_terms = hamiltonian
            .terms
            .iter()
            .filter(|t| matches!(t, HamiltonianTerm::SinglePauli { pauli, .. } if pauli == "X"))
            .count();

        assert_eq!(num_x_terms, 2); // Should have X on both qubits
    }

    #[test]
    fn test_adiabatic_evolution() {
        let initial_h = AdiabaticUtils::create_mixing_hamiltonian(2);
        let final_h = AdiabaticUtils::create_tfim_hamiltonian(2, 1.0, 0.1);

        let config = AdiabaticConfig {
            total_time: 1.0,
            time_steps: 10,
            initial_hamiltonian: initial_h,
            final_hamiltonian: final_h,
            gap_tracking: GapTrackingConfig {
                enabled: false, // Disable for simple test
                ..Default::default()
            },
            ..Default::default()
        };

        let mut adiabatic_qc = AdiabaticQuantumComputer::new(config).unwrap();
        let result = adiabatic_qc.evolve();
        assert!(result.is_ok());

        let evolution_result = result.unwrap();
        assert_eq!(evolution_result.evolution_history.len(), 2); // Initial + final snapshots
    }

    #[test]
    fn test_gap_tracking() {
        let initial_h = AdiabaticUtils::create_mixing_hamiltonian(2);
        let final_h = AdiabaticUtils::create_tfim_hamiltonian(2, 1.0, 0.1);

        let config = AdiabaticConfig {
            total_time: 1.0,
            time_steps: 5,
            initial_hamiltonian: initial_h,
            final_hamiltonian: final_h,
            gap_tracking: GapTrackingConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut adiabatic_qc = AdiabaticQuantumComputer::new(config).unwrap();
        let result = adiabatic_qc.evolve();
        assert!(result.is_ok());

        let evolution_result = result.unwrap();
        assert!(!evolution_result.gap_history.is_empty());
        assert!(evolution_result.min_gap >= 0.0);
    }

    #[test]
    fn test_energy_calculation() {
        let initial_h = AdiabaticUtils::create_mixing_hamiltonian(1);
        let final_h = AdiabaticUtils::create_tfim_hamiltonian(1, 1.0, 0.1);

        let config = AdiabaticConfig {
            initial_hamiltonian: initial_h,
            final_hamiltonian: final_h,
            ..Default::default()
        };

        let adiabatic_qc = AdiabaticQuantumComputer::new(config).unwrap();
        let energy = adiabatic_qc.calculate_current_energy();
        assert!(energy.is_ok());
    }

    #[test]
    fn test_fidelity_calculation() {
        let config = AdiabaticConfig::default();
        let adiabatic_qc = AdiabaticQuantumComputer::new(config).unwrap();

        let state1 = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let state2 = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        let fidelity = adiabatic_qc.calculate_fidelity(&state1, &state2);
        assert_abs_diff_eq!(fidelity, 1.0, epsilon = 1e-10);
    }
}
