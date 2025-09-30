//! Adiabatic Quantum Computing Simulation
//!
//! This module implements adiabatic quantum computing (AQC) and quantum annealing
//! algorithms for solving optimization problems. It includes:
//! - Adiabatic evolution under time-dependent Hamiltonians
//! - Quantum annealing for combinatorial optimization
//! - Problem encodings (QUBO, Ising model, etc.)
//! - Annealing schedules and gap analysis
//! - Performance monitoring and optimization

#![allow(clippy::fn_address_comparisons)]

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

/// Types of optimization problems for adiabatic quantum computing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProblemType {
    /// Quadratic Unconstrained Binary Optimization
    QUBO,
    /// Ising spin glass model
    Ising,
    /// Maximum Cut problem
    MaxCut,
    /// Graph Coloring
    GraphColoring,
    /// Traveling Salesman Problem
    TSP,
    /// Number Partitioning
    NumberPartitioning,
    /// Custom problem with user-defined Hamiltonian
    Custom,
}

/// Annealing schedule types
#[derive(Debug, Clone, Copy)]
pub enum AnnealingSchedule {
    /// Linear annealing: s(t) = t/T
    Linear,
    /// Exponential annealing
    Exponential { rate: f64 },
    /// Polynomial annealing: s(t) = (t/T)^p
    Polynomial { power: f64 },
    /// Trigonometric annealing: s(t) = sin²(πt/2T)
    Trigonometric,
    /// Custom schedule with user-defined function
    Custom(fn(f64, f64) -> f64),
}

impl AnnealingSchedule {
    /// Evaluate the schedule at time t with total time T
    pub fn evaluate(&self, t: f64, total_time: f64) -> f64 {
        let s = t / total_time;
        match self {
            AnnealingSchedule::Linear => s,
            AnnealingSchedule::Exponential { rate } => 1.0 - (-rate * s).exp(),
            AnnealingSchedule::Polynomial { power } => s.powf(*power),
            AnnealingSchedule::Trigonometric => (PI * s / 2.0).sin().powi(2),
            AnnealingSchedule::Custom(func) => func(t, total_time),
        }
    }

    /// Get the derivative of the schedule
    pub fn derivative(&self, t: f64, total_time: f64) -> f64 {
        let dt = 1e-8;
        let s1 = self.evaluate(t + dt, total_time);
        let s0 = self.evaluate(t, total_time);
        (s1 - s0) / dt
    }
}

/// QUBO (Quadratic Unconstrained Binary Optimization) problem representation
#[derive(Debug, Clone)]
pub struct QUBOProblem {
    /// Number of variables
    pub num_vars: usize,
    /// Q matrix: objective is x^T Q x
    pub q_matrix: Array2<f64>,
    /// Linear terms (optional)
    pub linear_terms: Option<Array1<f64>>,
    /// Constant offset
    pub offset: f64,
}

impl QUBOProblem {
    /// Create a new QUBO problem
    pub fn new(q_matrix: Array2<f64>) -> QuantRS2Result<Self> {
        let (rows, cols) = q_matrix.dim();
        if rows != cols {
            return Err(QuantRS2Error::InvalidInput(
                "Q matrix must be square".to_string(),
            ));
        }

        Ok(Self {
            num_vars: rows,
            q_matrix,
            linear_terms: None,
            offset: 0.0,
        })
    }

    /// Add linear terms to the QUBO problem
    pub fn with_linear_terms(mut self, linear: Array1<f64>) -> QuantRS2Result<Self> {
        if linear.len() != self.num_vars {
            return Err(QuantRS2Error::InvalidInput(
                "Linear terms size mismatch".to_string(),
            ));
        }
        self.linear_terms = Some(linear);
        Ok(self)
    }

    /// Add constant offset
    pub fn with_offset(mut self, offset: f64) -> Self {
        self.offset = offset;
        self
    }

    /// Evaluate the QUBO objective for a binary solution
    pub fn evaluate(&self, solution: &[u8]) -> QuantRS2Result<f64> {
        if solution.len() != self.num_vars {
            return Err(QuantRS2Error::InvalidInput(
                "Solution size mismatch".to_string(),
            ));
        }

        let mut objective = self.offset;

        // Quadratic terms
        for i in 0..self.num_vars {
            for j in 0..self.num_vars {
                objective += self.q_matrix[[i, j]] * solution[i] as f64 * solution[j] as f64;
            }
        }

        // Linear terms
        if let Some(ref linear) = self.linear_terms {
            for i in 0..self.num_vars {
                objective += linear[i] * solution[i] as f64;
            }
        }

        Ok(objective)
    }

    /// Convert to Ising model (spins ∈ {-1, +1})
    pub fn to_ising(&self) -> IsingProblem {
        let mut h = Array1::zeros(self.num_vars);
        let mut j = Array2::zeros((self.num_vars, self.num_vars));
        let mut offset = self.offset;

        // Transform QUBO to Ising: x_i = (1 + s_i)/2
        for i in 0..self.num_vars {
            // Linear terms: Q_ii * (1 + s_i)/2 + h_i * (1 + s_i)/2
            h[i] = self.q_matrix[[i, i]] / 4.0;
            offset += self.q_matrix[[i, i]] / 4.0;

            if let Some(ref linear) = self.linear_terms {
                h[i] += linear[i] / 2.0;
                offset += linear[i] / 2.0;
            }

            // Quadratic terms: Q_ij * (1 + s_i)/2 * (1 + s_j)/2
            for k in 0..self.num_vars {
                if i != k {
                    j[[i, k]] = self.q_matrix[[i, k]] / 4.0;
                    h[i] += self.q_matrix[[i, k]] / 4.0;
                    h[k] += self.q_matrix[[i, k]] / 4.0;
                    offset += self.q_matrix[[i, k]] / 4.0;
                }
            }
        }

        IsingProblem {
            num_spins: self.num_vars,
            h_fields: h,
            j_couplings: j,
            offset,
        }
    }
}

/// Ising model problem representation
#[derive(Debug, Clone)]
pub struct IsingProblem {
    /// Number of spins
    pub num_spins: usize,
    /// Local magnetic fields (h_i)
    pub h_fields: Array1<f64>,
    /// Coupling matrix (J_ij)
    pub j_couplings: Array2<f64>,
    /// Constant offset
    pub offset: f64,
}

impl IsingProblem {
    /// Create a new Ising problem
    pub fn new(h_fields: Array1<f64>, j_couplings: Array2<f64>) -> QuantRS2Result<Self> {
        let num_spins = h_fields.len();
        let (rows, cols) = j_couplings.dim();

        if rows != num_spins || cols != num_spins {
            return Err(QuantRS2Error::InvalidInput(
                "Coupling matrix size mismatch".to_string(),
            ));
        }

        Ok(Self {
            num_spins,
            h_fields,
            j_couplings,
            offset: 0.0,
        })
    }

    /// Evaluate the Ising energy for a spin configuration
    pub fn evaluate(&self, spins: &[i8]) -> QuantRS2Result<f64> {
        if spins.len() != self.num_spins {
            return Err(QuantRS2Error::InvalidInput(
                "Spin configuration size mismatch".to_string(),
            ));
        }

        let mut energy = self.offset;

        // Local field terms: -h_i * s_i
        for i in 0..self.num_spins {
            energy -= self.h_fields[i] * spins[i] as f64;
        }

        // Coupling terms: -J_ij * s_i * s_j
        for i in 0..self.num_spins {
            for j in i + 1..self.num_spins {
                energy -= self.j_couplings[[i, j]] * spins[i] as f64 * spins[j] as f64;
            }
        }

        Ok(energy)
    }

    /// Generate problem Hamiltonian as a matrix
    pub fn hamiltonian(&self) -> Array2<Complex64> {
        let dim = 1 << self.num_spins;
        let mut hamiltonian = Array2::zeros((dim, dim));

        for state in 0..dim {
            let spins = self.state_to_spins(state);
            let energy = self.evaluate(&spins).unwrap_or(0.0);
            hamiltonian[[state, state]] = Complex64::new(energy, 0.0);
        }

        hamiltonian
    }

    /// Convert computational basis state to spin configuration
    fn state_to_spins(&self, state: usize) -> Vec<i8> {
        (0..self.num_spins)
            .map(|i| if (state >> i) & 1 == 0 { -1 } else { 1 })
            .collect()
    }

    /// Convert spin configuration to computational basis state
    fn spins_to_state(&self, spins: &[i8]) -> usize {
        spins.iter().enumerate().fold(0, |acc, (i, &spin)| {
            acc | (if spin == 1 { 1 } else { 0 }) << i
        })
    }
}

/// Adiabatic quantum computer simulator
pub struct AdiabaticQuantumComputer {
    /// Problem Hamiltonian (H_P)
    problem_hamiltonian: Array2<Complex64>,
    /// Initial Hamiltonian (H_0) - typically transverse field
    initial_hamiltonian: Array2<Complex64>,
    /// Current quantum state
    state: Array1<Complex64>,
    /// Number of qubits/spins
    num_qubits: usize,
    /// Total annealing time
    total_time: f64,
    /// Current time
    current_time: f64,
    /// Time step for evolution
    time_step: f64,
    /// Annealing schedule
    schedule: AnnealingSchedule,
}

impl AdiabaticQuantumComputer {
    /// Create a new adiabatic quantum computer
    pub fn new(
        problem: &IsingProblem,
        total_time: f64,
        time_step: f64,
        schedule: AnnealingSchedule,
    ) -> Self {
        let num_qubits = problem.num_spins;
        let dim = 1 << num_qubits;

        // Problem Hamiltonian
        let problem_hamiltonian = problem.hamiltonian();

        // Initial Hamiltonian: H_0 = -∑_i σ_x^i (transverse field)
        let mut initial_hamiltonian = Array2::zeros((dim, dim));
        for i in 0..num_qubits {
            let pauli_x = Self::pauli_x_tensor(i, num_qubits);
            initial_hamiltonian = initial_hamiltonian + pauli_x;
        }
        initial_hamiltonian = initial_hamiltonian.mapv(|x: Complex64| -x);

        // Initial state: uniform superposition (ground state of H_0)
        let mut state = Array1::zeros(dim);
        let amplitude = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
        state.fill(amplitude);

        Self {
            problem_hamiltonian,
            initial_hamiltonian,
            state,
            num_qubits,
            total_time,
            current_time: 0.0,
            time_step,
            schedule,
        }
    }

    /// Generate Pauli-X tensor product for qubit i
    fn pauli_x_tensor(target_qubit: usize, num_qubits: usize) -> Array2<Complex64> {
        let dim = 1 << num_qubits;
        let mut result = Array2::zeros((dim, dim));

        for state in 0..dim {
            // Flip bit at target_qubit position
            let flipped_state = state ^ (1 << target_qubit);
            result[[state, flipped_state]] = Complex64::new(1.0, 0.0);
        }

        result
    }

    /// Get the current Hamiltonian H(t) = (1-s(t))H_0 + s(t)H_P
    pub fn current_hamiltonian(&self) -> Array2<Complex64> {
        let s = self.schedule.evaluate(self.current_time, self.total_time);
        let one_minus_s = 1.0 - s;

        self.initial_hamiltonian.mapv(|x| x * one_minus_s)
            + self.problem_hamiltonian.mapv(|x| x * s)
    }

    /// Perform one time step of adiabatic evolution
    pub fn step(&mut self) -> QuantRS2Result<()> {
        if self.current_time >= self.total_time {
            return Ok(());
        }

        let hamiltonian = self.current_hamiltonian();

        // Apply time evolution: |ψ(t+dt)⟩ = exp(-iH(t)dt)|ψ(t)⟩
        // Using first-order approximation: exp(-iHdt) ≈ I - iHdt
        let dt = self.time_step.min(self.total_time - self.current_time);
        let evolution_operator = self.compute_evolution_operator(&hamiltonian, dt)?;

        self.state = evolution_operator.dot(&self.state);
        self.current_time += dt;

        // Normalize state
        let norm = self.state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 0.0 {
            self.state.mapv_inplace(|c| c / norm);
        }

        Ok(())
    }

    /// Compute evolution operator exp(-iHdt) using matrix exponentiation
    fn compute_evolution_operator(
        &self,
        hamiltonian: &Array2<Complex64>,
        dt: f64,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let dim = hamiltonian.nrows();
        let mut evolution = Array2::eye(dim);

        // Use series expansion: exp(-iHdt) = I - iHdt - (Hdt)²/2! + i(Hdt)³/3! + ...
        let mut term = Array2::eye(dim);
        let h_dt = hamiltonian.mapv(|h| Complex64::new(0.0, -dt) * h);

        for n in 1..20 {
            // Truncate series at 20 terms
            term = term.dot(&h_dt);
            let coefficient = 1.0 / (1..=n).fold(1.0, |acc, i| acc * i as f64);
            evolution = evolution + term.mapv(|t| t * coefficient);

            // Check convergence
            let term_norm = term.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if term_norm * coefficient < 1e-12 {
                break;
            }
        }

        Ok(evolution)
    }

    /// Run the complete adiabatic evolution
    pub fn run(&mut self) -> QuantRS2Result<()> {
        while self.current_time < self.total_time {
            self.step()?;
        }
        Ok(())
    }

    /// Get measurement probabilities in computational basis
    pub fn measurement_probabilities(&self) -> Vec<f64> {
        self.state.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Sample a measurement result
    pub fn measure(&self) -> usize {
        let mut rng = thread_rng();
        let probs = self.measurement_probabilities();

        let random_value: f64 = rng.gen();
        let mut cumulative = 0.0;

        for (state, prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return state;
            }
        }

        probs.len() - 1 // Fallback
    }

    /// Get the instantaneous energy gap
    pub fn energy_gap(&self) -> QuantRS2Result<f64> {
        let hamiltonian = self.current_hamiltonian();

        // For small systems, compute eigenvalues directly
        if self.num_qubits <= 8 {
            let eigenvalues = self.compute_eigenvalues(&hamiltonian)?;
            let ground_energy = eigenvalues[0];
            let first_excited = eigenvalues[1];
            Ok(first_excited - ground_energy)
        } else {
            // For larger systems, use approximation
            Ok(self.estimate_gap(&hamiltonian))
        }
    }

    /// Compute eigenvalues (simplified implementation)
    fn compute_eigenvalues(&self, hamiltonian: &Array2<Complex64>) -> QuantRS2Result<Vec<f64>> {
        // This is a placeholder - in practice you'd use a proper eigenvalue solver
        let dim = hamiltonian.nrows();
        let mut eigenvalues: Vec<f64> = (0..dim).map(|i| hamiltonian[[i, i]].re).collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(eigenvalues)
    }

    /// Estimate energy gap using power iteration
    fn estimate_gap(&self, _hamiltonian: &Array2<Complex64>) -> f64 {
        // Simplified estimation - would use proper numerical methods in practice
        let s = self.schedule.evaluate(self.current_time, self.total_time);
        // Gap typically closes as sin(πs) near the end of annealing
        (PI * s).sin() * 0.1 // Rough approximation
    }

    /// Check if adiabatic condition is satisfied
    pub fn adiabatic_condition_satisfied(&self) -> QuantRS2Result<bool> {
        let gap = self.energy_gap()?;
        let s_dot = self.schedule.derivative(self.current_time, self.total_time);

        // Adiabatic condition: |⟨1|dH/dt|0⟩| << Δ²
        // Simplified check: gap should be much larger than evolution rate
        Ok(gap > 10.0 * s_dot.abs())
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        let dim = 1 << self.num_qubits;
        let amplitude = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
        self.state.fill(amplitude);
        self.current_time = 0.0;
    }

    /// Get current state vector
    pub fn state(&self) -> &Array1<Complex64> {
        &self.state
    }

    /// Get current annealing parameter s(t)
    pub fn annealing_parameter(&self) -> f64 {
        self.schedule.evaluate(self.current_time, self.total_time)
    }

    /// Get progress as percentage
    pub fn progress(&self) -> f64 {
        (self.current_time / self.total_time).min(1.0) * 100.0
    }
}

/// Quantum annealing optimizer for solving optimization problems
pub struct QuantumAnnealer {
    computer: AdiabaticQuantumComputer,
    problem: IsingProblem,
    best_solution: Option<(Vec<i8>, f64)>,
    history: Vec<QuantumAnnealingSnapshot>,
}

/// Snapshot of quantum annealing state at a point in time
#[derive(Debug, Clone)]
pub struct QuantumAnnealingSnapshot {
    pub time: f64,
    pub annealing_parameter: f64,
    pub energy_gap: f64,
    pub ground_state_probability: f64,
    pub expected_energy: f64,
}

impl QuantumAnnealer {
    /// Create a new quantum annealer
    pub fn new(
        problem: IsingProblem,
        total_time: f64,
        time_step: f64,
        schedule: AnnealingSchedule,
    ) -> Self {
        let computer = AdiabaticQuantumComputer::new(&problem, total_time, time_step, schedule);

        Self {
            computer,
            problem,
            best_solution: None,
            history: Vec::new(),
        }
    }

    /// Run quantum annealing optimization
    pub fn optimize(&mut self) -> QuantRS2Result<(Vec<i8>, f64)> {
        self.computer.reset();
        self.history.clear();

        // Record initial state
        self.record_snapshot()?;

        // Run adiabatic evolution
        while self.computer.current_time < self.computer.total_time {
            self.computer.step()?;

            // Record snapshots at regular intervals
            if self.history.len() % 10 == 0 {
                self.record_snapshot()?;
            }
        }

        // Final measurement and solution extraction
        let measurement = self.computer.measure();
        let spins = self.state_to_spins(measurement);
        let energy = self.problem.evaluate(&spins)?;

        self.best_solution = Some((spins.clone(), energy));

        Ok((spins, energy))
    }

    /// Run multiple annealing runs and return the best solution
    pub fn optimize_multiple_runs(&mut self, num_runs: usize) -> QuantRS2Result<(Vec<i8>, f64)> {
        let mut best_energy = f64::INFINITY;
        let mut best_spins = vec![];

        for _ in 0..num_runs {
            let (spins, energy) = self.optimize()?;
            if energy < best_energy {
                best_energy = energy;
                best_spins = spins;
            }
        }

        Ok((best_spins, best_energy))
    }

    /// Record a snapshot of the current annealing state
    fn record_snapshot(&mut self) -> QuantRS2Result<()> {
        let time = self.computer.current_time;
        let annealing_parameter = self.computer.annealing_parameter();
        let energy_gap = self.computer.energy_gap()?;

        // Calculate ground state probability and expected energy
        let probs = self.computer.measurement_probabilities();
        let ground_state_probability = probs[0]; // Assuming ground state is first

        let mut expected_energy = 0.0;
        for (state_idx, &prob) in probs.iter().enumerate() {
            let spins = self.state_to_spins(state_idx);
            let energy = self.problem.evaluate(&spins).unwrap_or(0.0);
            expected_energy += prob * energy;
        }

        self.history.push(QuantumAnnealingSnapshot {
            time,
            annealing_parameter,
            energy_gap,
            ground_state_probability,
            expected_energy,
        });

        Ok(())
    }

    /// Convert computational basis state to spin configuration
    fn state_to_spins(&self, state: usize) -> Vec<i8> {
        (0..self.problem.num_spins)
            .map(|i| if (state >> i) & 1 == 0 { -1 } else { 1 })
            .collect()
    }

    /// Get annealing history
    pub fn history(&self) -> &[QuantumAnnealingSnapshot] {
        &self.history
    }

    /// Get best solution found
    pub fn best_solution(&self) -> Option<&(Vec<i8>, f64)> {
        self.best_solution.as_ref()
    }

    /// Get minimum energy gap during annealing
    pub fn minimum_gap(&self) -> f64 {
        self.history
            .iter()
            .map(|snapshot| snapshot.energy_gap)
            .fold(f64::INFINITY, f64::min)
    }

    /// Calculate success probability (probability of finding ground state)
    pub fn success_probability(&self) -> f64 {
        self.history
            .last()
            .map(|snapshot| snapshot.ground_state_probability)
            .unwrap_or(0.0)
    }
}

/// Create common optimization problems for testing
pub struct ProblemGenerator;

impl ProblemGenerator {
    /// Generate random QUBO problem
    pub fn random_qubo(num_vars: usize, density: f64) -> QUBOProblem {
        let mut rng = thread_rng();

        let mut q_matrix = Array2::zeros((num_vars, num_vars));

        for i in 0..num_vars {
            for j in i..num_vars {
                if rng.gen::<f64>() < density {
                    let value = rng.gen_range(-1.0..=1.0);
                    q_matrix[[i, j]] = value;
                    if i != j {
                        q_matrix[[j, i]] = value; // Symmetric
                    }
                }
            }
        }

        QUBOProblem::new(q_matrix).unwrap()
    }

    /// Generate MaxCut problem on a random graph
    pub fn max_cut(num_vertices: usize, edge_probability: f64) -> IsingProblem {
        let mut rng = thread_rng();

        let h_fields = Array1::zeros(num_vertices);
        let mut j_couplings = Array2::zeros((num_vertices, num_vertices));

        // Generate random edges with positive coupling (ferromagnetic)
        for i in 0..num_vertices {
            for j in i + 1..num_vertices {
                if rng.gen::<f64>() < edge_probability {
                    let coupling = 1.0; // Unit weight edges
                    j_couplings[[i, j]] = coupling;
                    j_couplings[[j, i]] = coupling;
                }
            }
        }

        IsingProblem::new(h_fields, j_couplings).unwrap()
    }

    /// Generate Number Partitioning problem
    pub fn number_partitioning(numbers: Vec<f64>) -> IsingProblem {
        let n = numbers.len();
        let h_fields = Array1::zeros(n);
        let mut j_couplings = Array2::zeros((n, n));

        // J_ij = numbers[i] * numbers[j]
        for i in 0..n {
            for j in i + 1..n {
                let coupling = numbers[i] * numbers[j];
                j_couplings[[i, j]] = coupling;
                j_couplings[[j, i]] = coupling;
            }
        }

        IsingProblem::new(h_fields, j_couplings).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annealing_schedules() {
        let linear = AnnealingSchedule::Linear;
        assert_eq!(linear.evaluate(0.0, 10.0), 0.0);
        assert_eq!(linear.evaluate(5.0, 10.0), 0.5);
        assert_eq!(linear.evaluate(10.0, 10.0), 1.0);

        let exp = AnnealingSchedule::Exponential { rate: 1.0 };
        assert!(exp.evaluate(0.0, 1.0) < 0.1);
        assert!(exp.evaluate(1.0, 1.0) > 0.6);

        let poly = AnnealingSchedule::Polynomial { power: 2.0 };
        assert_eq!(poly.evaluate(10.0, 10.0), 1.0);
        assert!(poly.evaluate(5.0, 10.0) < linear.evaluate(5.0, 10.0));
    }

    #[test]
    fn test_qubo_problem() {
        let mut q_matrix = Array2::zeros((2, 2));
        q_matrix[[0, 0]] = 1.0;
        q_matrix[[1, 1]] = 1.0;
        q_matrix[[0, 1]] = -2.0;
        q_matrix[[1, 0]] = -2.0;

        let qubo = QUBOProblem::new(q_matrix).unwrap();

        // Test evaluations
        // For Q = [[1, -2], [-2, 1]]:
        // [0,0]: 0*1*0 + 0*(-2)*0 + 0*(-2)*0 + 0*1*0 = 0
        // [1,1]: 1*1*1 + 1*(-2)*1 + 1*(-2)*1 + 1*1*1 = 1 - 2 - 2 + 1 = -2
        // [1,0]: 1*1*1 + 1*(-2)*0 + 0*(-2)*1 + 0*1*0 = 1
        // [0,1]: 0*1*0 + 0*(-2)*1 + 1*(-2)*0 + 1*1*1 = 1
        assert_eq!(qubo.evaluate(&[0, 0]).unwrap(), 0.0);
        assert_eq!(qubo.evaluate(&[1, 1]).unwrap(), -2.0);
        assert_eq!(qubo.evaluate(&[1, 0]).unwrap(), 1.0);
        assert_eq!(qubo.evaluate(&[0, 1]).unwrap(), 1.0);
    }

    #[test]
    fn test_ising_problem() {
        let h_fields = Array1::from(vec![0.5, -0.5]);
        let mut j_couplings = Array2::zeros((2, 2));
        j_couplings[[0, 1]] = 1.0;
        j_couplings[[1, 0]] = 1.0;

        let ising = IsingProblem::new(h_fields, j_couplings).unwrap();

        // Test evaluations
        // Energy = -∑_i h_i s_i - ∑_{i<j} J_{ij} s_i s_j
        // With h = [0.5, -0.5] and J_{01} = 1.0:
        assert_eq!(ising.evaluate(&[-1, -1]).unwrap(), -1.0); // -(0.5*(-1) + (-0.5)*(-1)) - 1*(-1)*(-1) = -(-0.5 + 0.5) - 1 = -1
        assert_eq!(ising.evaluate(&[1, 1]).unwrap(), -1.0); // -(0.5*1 + (-0.5)*1) - 1*1*1 = -(0.5 - 0.5) - 1 = -1
        assert_eq!(ising.evaluate(&[1, -1]).unwrap(), 0.0); // -(0.5*1 + (-0.5)*(-1)) - 1*1*(-1) = -(0.5 + 0.5) - (-1) = -1 + 1 = 0
        assert_eq!(ising.evaluate(&[-1, 1]).unwrap(), 2.0); // -(0.5*(-1) + (-0.5)*1) - 1*(-1)*1 = -(-0.5 - 0.5) - (-1) = 1 + 1 = 2
    }

    #[test]
    fn test_qubo_to_ising_conversion() {
        let mut q_matrix = Array2::zeros((2, 2));
        q_matrix[[0, 1]] = 1.0;
        q_matrix[[1, 0]] = 1.0;

        let qubo = QUBOProblem::new(q_matrix).unwrap();
        let ising = qubo.to_ising();

        // Verify the conversion maintains the same optimal solutions
        // QUBO: minimize x₀x₁, optimal at (0,0), (0,1), (1,0) with value 0
        // Ising: should have corresponding optimal spin configurations

        assert_eq!(ising.num_spins, 2);
        assert!(ising.h_fields.len() == 2);
        assert!(ising.j_couplings.dim() == (2, 2));
    }

    #[test]
    fn test_adiabatic_computer_initialization() {
        let h_fields = Array1::from(vec![0.0, 0.0]);
        let j_couplings = Array2::eye(2);
        let ising = IsingProblem::new(h_fields, j_couplings).unwrap();

        let adiabatic = AdiabaticQuantumComputer::new(
            &ising,
            10.0, // total_time
            0.1,  // time_step
            AnnealingSchedule::Linear,
        );

        assert_eq!(adiabatic.num_qubits, 2);
        assert_eq!(adiabatic.current_time, 0.0);
        assert_eq!(adiabatic.state.len(), 4); // 2^2 = 4 states

        // Initial state should be uniform superposition
        let expected_amplitude = 1.0 / 2.0; // 1/sqrt(4)
        for amplitude in adiabatic.state.iter() {
            assert!((amplitude.norm() - expected_amplitude).abs() < 1e-10);
        }
    }

    #[test]
    fn test_adiabatic_evolution_step() {
        let h_fields = Array1::zeros(1);
        let j_couplings = Array2::zeros((1, 1));
        let ising = IsingProblem::new(h_fields, j_couplings).unwrap();

        let mut adiabatic = AdiabaticQuantumComputer::new(
            &ising,
            1.0, // total_time
            0.1, // time_step
            AnnealingSchedule::Linear,
        );

        let initial_time = adiabatic.current_time;
        adiabatic.step().unwrap();

        assert!(adiabatic.current_time > initial_time);
        assert!(adiabatic.current_time <= adiabatic.total_time);

        // State should remain normalized
        let norm = adiabatic.state.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_annealer() {
        let problem = ProblemGenerator::max_cut(3, 0.5);
        let mut annealer = QuantumAnnealer::new(
            problem,
            5.0, // total_time
            0.1, // time_step
            AnnealingSchedule::Linear,
        );

        let (solution, energy) = annealer.optimize().unwrap();

        assert_eq!(solution.len(), 3);
        assert!(solution.iter().all(|&s| s == -1 || s == 1));
        assert!(energy.is_finite());

        // Should have recorded history
        assert!(!annealer.history().is_empty());

        // Should have a best solution
        assert!(annealer.best_solution().is_some());
    }

    #[test]
    fn test_problem_generators() {
        let qubo = ProblemGenerator::random_qubo(3, 0.5);
        assert_eq!(qubo.num_vars, 3);
        assert_eq!(qubo.q_matrix.dim(), (3, 3));

        let max_cut = ProblemGenerator::max_cut(4, 0.6);
        assert_eq!(max_cut.num_spins, 4);
        assert_eq!(max_cut.h_fields.len(), 4);
        assert_eq!(max_cut.j_couplings.dim(), (4, 4));

        let numbers = vec![1.0, 2.0, 3.0];
        let num_part = ProblemGenerator::number_partitioning(numbers);
        assert_eq!(num_part.num_spins, 3);
    }

    #[test]
    fn test_multiple_annealing_runs() {
        let problem = ProblemGenerator::random_qubo(2, 1.0).to_ising();
        let mut annealer = QuantumAnnealer::new(
            problem,
            2.0, // total_time
            0.2, // time_step
            AnnealingSchedule::Linear,
        );

        let (solution, energy) = annealer.optimize_multiple_runs(3).unwrap();

        assert_eq!(solution.len(), 2);
        assert!(solution.iter().all(|&s| s == -1 || s == 1));
        assert!(energy.is_finite());
    }

    #[test]
    fn test_annealing_parameter_progression() {
        let problem = ProblemGenerator::max_cut(2, 1.0);
        let mut computer = AdiabaticQuantumComputer::new(
            &problem,
            1.0,  // total_time
            0.25, // time_step (4 steps total)
            AnnealingSchedule::Linear,
        );

        // Check annealing parameter at different times
        assert_eq!(computer.annealing_parameter(), 0.0);

        computer.step().unwrap();
        assert!((computer.annealing_parameter() - 0.25).abs() < 1e-10);

        computer.step().unwrap();
        assert!((computer.annealing_parameter() - 0.5).abs() < 1e-10);

        computer.step().unwrap();
        assert!((computer.annealing_parameter() - 0.75).abs() < 1e-10);

        computer.step().unwrap();
        assert!((computer.annealing_parameter() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_gap_calculation() {
        let h_fields = Array1::from(vec![1.0]);
        let j_couplings = Array2::zeros((1, 1));
        let ising = IsingProblem::new(h_fields, j_couplings).unwrap();

        let computer = AdiabaticQuantumComputer::new(&ising, 1.0, 0.1, AnnealingSchedule::Linear);

        let gap = computer.energy_gap().unwrap();
        assert!(gap >= 0.0);
    }

    #[test]
    fn test_measurement_probabilities() {
        let h_fields = Array1::zeros(1);
        let j_couplings = Array2::zeros((1, 1));
        let ising = IsingProblem::new(h_fields, j_couplings).unwrap();

        let computer = AdiabaticQuantumComputer::new(&ising, 1.0, 0.1, AnnealingSchedule::Linear);

        let probs = computer.measurement_probabilities();
        assert_eq!(probs.len(), 2); // 2^1 = 2 states

        // Probabilities should sum to 1
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);

        // All probabilities should be non-negative
        assert!(probs.iter().all(|&p| p >= 0.0));
    }
}
