//! Quantum walk-based optimization algorithms
//!
//! This module implements quantum walk algorithms for solving optimization
//! problems, providing an alternative approach to traditional annealing
//! that leverages quantum interference effects.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use num_complex::Complex64;

use crate::ising::{IsingError, IsingModel};
use crate::simulator::AnnealingSolution;

/// Errors that can occur during quantum walk optimization
#[derive(Error, Debug)]
pub enum QuantumWalkError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),
    
    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Evolution error
    #[error("Evolution error: {0}")]
    EvolutionError(String),
    
    /// Measurement error
    #[error("Measurement error: {0}")]
    MeasurementError(String),
}

/// Result type for quantum walk operations
pub type QuantumWalkResult<T> = Result<T, QuantumWalkError>;

/// Quantum walk algorithm type
#[derive(Debug, Clone)]
pub enum QuantumWalkAlgorithm {
    /// Continuous-time quantum walk (CTQW)
    ContinuousTime {
        evolution_time: f64,
        time_steps: usize,
    },
    
    /// Discrete-time quantum walk (DTQW)
    DiscreteTime {
        coin_operator: CoinOperator,
        steps: usize,
    },
    
    /// Adiabatic quantum walk optimization (AQWO)
    Adiabatic {
        initial_hamiltonian: AdiabaticHamiltonian,
        final_hamiltonian: AdiabaticHamiltonian,
        evolution_time: f64,
        time_steps: usize,
    },
    
    /// Quantum approximate optimization algorithm (QAOA) with walk
    QaoaWalk {
        layers: usize,
        beta_schedule: Vec<f64>,
        gamma_schedule: Vec<f64>,
    },
}

/// Coin operator for discrete-time quantum walks
#[derive(Debug, Clone)]
pub enum CoinOperator {
    /// Hadamard coin
    Hadamard,
    
    /// Grover coin
    Grover,
    
    /// Custom unitary coin
    Custom(Vec<Vec<Complex64>>),
}

/// Hamiltonian specification for adiabatic evolution
#[derive(Debug, Clone)]
pub enum AdiabaticHamiltonian {
    /// Mixing Hamiltonian (typically sum of X operators)
    Mixing,
    
    /// Problem Hamiltonian (from Ising model)
    Problem,
    
    /// Custom Hamiltonian
    Custom(Vec<Vec<Complex64>>),
}

/// Configuration for quantum walk optimization
#[derive(Debug, Clone)]
pub struct QuantumWalkConfig {
    /// Algorithm to use
    pub algorithm: QuantumWalkAlgorithm,
    
    /// Number of measurements/samples
    pub num_measurements: usize,
    
    /// Random seed
    pub seed: Option<u64>,
    
    /// Maximum evolution time
    pub max_evolution_time: f64,
    
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    
    /// Enable amplitude amplification
    pub amplitude_amplification: bool,
    
    /// Number of amplitude amplification iterations
    pub amplification_iterations: usize,
    
    /// Timeout for optimization
    pub timeout: Option<Duration>,
}

impl Default for QuantumWalkConfig {
    fn default() -> Self {
        Self {
            algorithm: QuantumWalkAlgorithm::ContinuousTime {
                evolution_time: 1.0,
                time_steps: 100,
            },
            num_measurements: 1000,
            seed: None,
            max_evolution_time: 10.0,
            convergence_tolerance: 1e-6,
            amplitude_amplification: false,
            amplification_iterations: 3,
            timeout: Some(Duration::from_secs(300)),
        }
    }
}

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Amplitudes for each computational basis state
    pub amplitudes: Vec<Complex64>,
    
    /// Number of qubits
    pub num_qubits: usize,
}

impl QuantumState {
    /// Create a new quantum state
    pub fn new(num_qubits: usize) -> Self {
        let num_states = 2_usize.pow(num_qubits as u32);
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); num_states];
        
        // Initialize to |0...0⟩ state
        amplitudes[0] = Complex64::new(1.0, 0.0);
        
        Self {
            amplitudes,
            num_qubits,
        }
    }
    
    /// Create uniform superposition state
    pub fn uniform_superposition(num_qubits: usize) -> Self {
        let num_states = 2_usize.pow(num_qubits as u32);
        let amplitude = Complex64::new(1.0 / (num_states as f64).sqrt(), 0.0);
        let amplitudes = vec![amplitude; num_states];
        
        Self {
            amplitudes,
            num_qubits,
        }
    }
    
    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm_squared: f64 = self.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        if norm_squared > 0.0 {
            let norm = norm_squared.sqrt();
            for amp in &mut self.amplitudes {
                *amp /= norm;
            }
        }
    }
    
    /// Get probability of measuring state i
    pub fn probability(&self, state_index: usize) -> f64 {
        if state_index < self.amplitudes.len() {
            self.amplitudes[state_index].norm_sqr()
        } else {
            0.0
        }
    }
    
    /// Convert state index to bit string
    pub fn state_to_bits(&self, state_index: usize) -> Vec<i8> {
        let mut bits = vec![0; self.num_qubits];
        let mut index = state_index;
        
        for i in 0..self.num_qubits {
            bits[self.num_qubits - 1 - i] = if (index & 1) == 1 { 1 } else { -1 };
            index >>= 1;
        }
        
        bits
    }
    
    /// Convert bit string to state index
    pub fn bits_to_state(&self, bits: &[i8]) -> usize {
        let mut index = 0;
        for (i, &bit) in bits.iter().enumerate() {
            if bit > 0 {
                index |= 1 << (self.num_qubits - 1 - i);
            }
        }
        index
    }
}

/// Quantum walk optimizer
pub struct QuantumWalkOptimizer {
    /// Configuration
    config: QuantumWalkConfig,
    
    /// Random number generator
    rng: ChaCha8Rng,
}

impl QuantumWalkOptimizer {
    /// Create a new quantum walk optimizer
    pub fn new(config: QuantumWalkConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };
        
        Self { config, rng }
    }
    
    /// Solve an Ising model using quantum walk optimization
    pub fn solve(&mut self, model: &IsingModel) -> QuantumWalkResult<AnnealingSolution> {
        let start_time = Instant::now();
        
        // Initialize quantum state
        let mut state = QuantumState::uniform_superposition(model.num_qubits);
        
        // Evolve according to algorithm
        let algorithm = self.config.algorithm.clone();
        match algorithm {
            QuantumWalkAlgorithm::ContinuousTime { evolution_time, time_steps } => {
                self.continuous_time_evolution(model, &mut state, evolution_time, time_steps)?;
            }
            
            QuantumWalkAlgorithm::DiscreteTime { coin_operator, steps } => {
                self.discrete_time_evolution(model, &mut state, &coin_operator, steps)?;
            }
            
            QuantumWalkAlgorithm::Adiabatic { initial_hamiltonian, final_hamiltonian, evolution_time, time_steps } => {
                self.adiabatic_evolution(model, &mut state, &initial_hamiltonian, &final_hamiltonian, evolution_time, time_steps)?;
            }
            
            QuantumWalkAlgorithm::QaoaWalk { layers, beta_schedule, gamma_schedule } => {
                self.qaoa_walk_evolution(model, &mut state, layers, &beta_schedule, &gamma_schedule)?;
            }
        }
        
        // Apply amplitude amplification if enabled
        if self.config.amplitude_amplification {
            self.amplitude_amplification(model, &mut state)?;
        }
        
        // Perform measurements and find best solution
        let (best_spins, best_energy) = self.measure_and_optimize(model, &state)?;
        
        let runtime = start_time.elapsed();
        
        Ok(AnnealingSolution {
            best_spins,
            best_energy,
            repetitions: 1,
            total_sweeps: self.config.num_measurements,
            runtime,
            info: format!(
                "Quantum walk optimization with {:?} in {:?}",
                self.config.algorithm, runtime
            ),
        })
    }
    
    /// Continuous-time quantum walk evolution
    fn continuous_time_evolution(
        &mut self,
        model: &IsingModel,
        state: &mut QuantumState,
        evolution_time: f64,
        time_steps: usize,
    ) -> QuantumWalkResult<()> {
        let dt = evolution_time / time_steps as f64;
        
        // Build Hamiltonian matrix
        let hamiltonian = self.build_problem_hamiltonian(model)?;
        
        // Time evolution using Trotter decomposition
        for _ in 0..time_steps {
            self.apply_hamiltonian_evolution(state, &hamiltonian, dt)?;
            
            // Add some decoherence/noise
            self.apply_decoherence(state, dt * 0.01);
        }
        
        Ok(())
    }
    
    /// Discrete-time quantum walk evolution
    fn discrete_time_evolution(
        &mut self,
        model: &IsingModel,
        state: &mut QuantumState,
        coin_operator: &CoinOperator,
        steps: usize,
    ) -> QuantumWalkResult<()> {
        for _ in 0..steps {
            // Apply coin operator
            self.apply_coin_operator(state, coin_operator)?;
            
            // Apply shift operator based on problem structure
            self.apply_shift_operator(state, model)?;
            
            // Apply conditional phase based on energy
            self.apply_conditional_phase(state, model)?;
        }
        
        Ok(())
    }
    
    /// Adiabatic quantum walk evolution
    fn adiabatic_evolution(
        &mut self,
        model: &IsingModel,
        state: &mut QuantumState,
        initial_ham: &AdiabaticHamiltonian,
        final_ham: &AdiabaticHamiltonian,
        evolution_time: f64,
        time_steps: usize,
    ) -> QuantumWalkResult<()> {
        let dt = evolution_time / time_steps as f64;
        
        let initial_hamiltonian = self.build_hamiltonian(model, initial_ham)?;
        let final_hamiltonian = self.build_hamiltonian(model, final_ham)?;
        
        for step in 0..time_steps {
            let s = step as f64 / time_steps as f64;
            
            // Interpolate Hamiltonian: H(s) = (1-s)H_initial + s*H_final
            let mut current_hamiltonian = Vec::new();
            for i in 0..initial_hamiltonian.len() {
                let mut row = Vec::new();
                for j in 0..initial_hamiltonian[i].len() {
                    let interpolated = (1.0 - s) * initial_hamiltonian[i][j] + s * final_hamiltonian[i][j];
                    row.push(interpolated);
                }
                current_hamiltonian.push(row);
            }
            
            // Evolve for time dt
            self.apply_hamiltonian_evolution(state, &current_hamiltonian, dt)?;
        }
        
        Ok(())
    }
    
    /// QAOA with quantum walk evolution
    fn qaoa_walk_evolution(
        &mut self,
        model: &IsingModel,
        state: &mut QuantumState,
        layers: usize,
        beta_schedule: &[f64],
        gamma_schedule: &[f64],
    ) -> QuantumWalkResult<()> {
        let problem_hamiltonian = self.build_problem_hamiltonian(model)?;
        let mixing_hamiltonian = self.build_mixing_hamiltonian(model.num_qubits)?;
        
        for layer in 0..layers {
            let gamma = if layer < gamma_schedule.len() {
                gamma_schedule[layer]
            } else {
                gamma_schedule.last().copied().unwrap_or(0.5)
            };
            
            let beta = if layer < beta_schedule.len() {
                beta_schedule[layer]
            } else {
                beta_schedule.last().copied().unwrap_or(0.5)
            };
            
            // Apply problem Hamiltonian evolution
            self.apply_hamiltonian_evolution(state, &problem_hamiltonian, gamma)?;
            
            // Apply mixing Hamiltonian evolution
            self.apply_hamiltonian_evolution(state, &mixing_hamiltonian, beta)?;
        }
        
        Ok(())
    }
    
    /// Build problem Hamiltonian from Ising model
    fn build_problem_hamiltonian(&self, model: &IsingModel) -> QuantumWalkResult<Vec<Vec<Complex64>>> {
        let num_states = 2_usize.pow(model.num_qubits as u32);
        let mut hamiltonian = vec![vec![Complex64::new(0.0, 0.0); num_states]; num_states];
        
        // Add diagonal terms (Z operators for biases and ZZ for couplings)
        for state_idx in 0..num_states {
            let mut energy = 0.0;
            let bits = self.index_to_bits(state_idx, model.num_qubits);
            
            // Add bias terms
            for (qubit, bias) in model.biases().iter() {
                energy += bias * bits[*qubit] as f64;
            }
            
            // Add coupling terms
            for coupling in model.couplings() {
                energy += coupling.strength * bits[coupling.i] as f64 * bits[coupling.j] as f64;
            }
            
            hamiltonian[state_idx][state_idx] = Complex64::new(energy, 0.0);
        }
        
        Ok(hamiltonian)
    }
    
    /// Build mixing Hamiltonian (sum of X operators)
    fn build_mixing_hamiltonian(&self, num_qubits: usize) -> QuantumWalkResult<Vec<Vec<Complex64>>> {
        let num_states = 2_usize.pow(num_qubits as u32);
        let mut hamiltonian = vec![vec![Complex64::new(0.0, 0.0); num_states]; num_states];
        
        // Add X operators for each qubit
        for qubit in 0..num_qubits {
            for state_idx in 0..num_states {
                let flipped_idx = state_idx ^ (1 << (num_qubits - 1 - qubit));
                hamiltonian[state_idx][flipped_idx] += Complex64::new(1.0, 0.0);
            }
        }
        
        Ok(hamiltonian)
    }
    
    /// Build Hamiltonian from specification
    fn build_hamiltonian(&self, model: &IsingModel, ham_type: &AdiabaticHamiltonian) -> QuantumWalkResult<Vec<Vec<Complex64>>> {
        match ham_type {
            AdiabaticHamiltonian::Mixing => self.build_mixing_hamiltonian(model.num_qubits),
            AdiabaticHamiltonian::Problem => self.build_problem_hamiltonian(model),
            AdiabaticHamiltonian::Custom(matrix) => Ok(matrix.clone()),
        }
    }
    
    /// Apply Hamiltonian evolution for time dt
    fn apply_hamiltonian_evolution(
        &self,
        state: &mut QuantumState,
        hamiltonian: &[Vec<Complex64>],
        dt: f64,
    ) -> QuantumWalkResult<()> {
        let num_states = state.amplitudes.len();
        let mut new_amplitudes = vec![Complex64::new(0.0, 0.0); num_states];
        
        // Simple first-order evolution: |ψ(t+dt)⟩ = exp(-i*H*dt)|ψ(t)⟩
        // Approximated as: |ψ(t+dt)⟩ ≈ (I - i*H*dt)|ψ(t)⟩
        for i in 0..num_states {
            new_amplitudes[i] = state.amplitudes[i];  // Identity term
            
            for j in 0..num_states {
                let evolution_term = -Complex64::i() * hamiltonian[i][j] * dt;
                new_amplitudes[i] += evolution_term * state.amplitudes[j];
            }
        }
        
        state.amplitudes = new_amplitudes;
        state.normalize();
        
        Ok(())
    }
    
    /// Apply coin operator to the state
    fn apply_coin_operator(&self, state: &mut QuantumState, coin: &CoinOperator) -> QuantumWalkResult<()> {
        match coin {
            CoinOperator::Hadamard => {
                // Apply Hadamard to each qubit
                for qubit in 0..state.num_qubits {
                    self.apply_single_qubit_gate(state, qubit, &[[1.0, 1.0], [1.0, -1.0]], 1.0 / 2.0_f64.sqrt())?;
                }
            }
            
            CoinOperator::Grover => {
                // Apply Grover coin (more complex, simplified here)
                for qubit in 0..state.num_qubits {
                    self.apply_single_qubit_gate(state, qubit, &[[0.0, 1.0], [1.0, 0.0]], 1.0)?;
                }
            }
            
            CoinOperator::Custom(_matrix) => {
                // Custom coin operator implementation would go here
                return Err(QuantumWalkError::EvolutionError(
                    "Custom coin operators not yet implemented".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    /// Apply single-qubit gate
    fn apply_single_qubit_gate(
        &self,
        state: &mut QuantumState,
        qubit: usize,
        gate: &[[f64; 2]; 2],
        normalization: f64,
    ) -> QuantumWalkResult<()> {
        let num_states = state.amplitudes.len();
        let mut new_amplitudes = vec![Complex64::new(0.0, 0.0); num_states];
        
        for state_idx in 0..num_states {
            let bit = (state_idx >> (state.num_qubits - 1 - qubit)) & 1;
            let flipped_idx = state_idx ^ (1 << (state.num_qubits - 1 - qubit));
            
            // Apply gate matrix
            new_amplitudes[state_idx] += Complex64::new(gate[bit][bit] * normalization, 0.0) * state.amplitudes[state_idx];
            new_amplitudes[flipped_idx] += Complex64::new(gate[1-bit][bit] * normalization, 0.0) * state.amplitudes[state_idx];
        }
        
        state.amplitudes = new_amplitudes;
        state.normalize();
        
        Ok(())
    }
    
    /// Apply shift operator based on problem graph
    fn apply_shift_operator(&self, _state: &mut QuantumState, _model: &IsingModel) -> QuantumWalkResult<()> {
        // Simplified implementation - would normally implement graph-based shifts
        Ok(())
    }
    
    /// Apply conditional phase based on energy
    fn apply_conditional_phase(&self, state: &mut QuantumState, model: &IsingModel) -> QuantumWalkResult<()> {
        for (state_idx, amplitude) in state.amplitudes.iter_mut().enumerate() {
            let bits = self.index_to_bits(state_idx, model.num_qubits);
            let energy = model.energy(&bits).map_err(QuantumWalkError::IsingError)?;
            
            // Apply phase proportional to energy
            let phase = -energy * 0.1;  // Small phase to avoid numerical issues
            *amplitude *= Complex64::new(phase.cos(), phase.sin());
        }
        
        Ok(())
    }
    
    /// Apply decoherence/noise
    fn apply_decoherence(&mut self, state: &mut QuantumState, strength: f64) {
        for amplitude in &mut state.amplitudes {
            let noise_real = (self.rng.gen::<f64>() - 0.5) * strength;
            let noise_imag = (self.rng.gen::<f64>() - 0.5) * strength;
            *amplitude += Complex64::new(noise_real, noise_imag);
        }
        
        state.normalize();
    }
    
    /// Amplitude amplification to boost good solutions
    fn amplitude_amplification(&mut self, model: &IsingModel, state: &mut QuantumState) -> QuantumWalkResult<()> {
        // Find mean energy
        let mut mean_energy = 0.0;
        for (state_idx, amplitude) in state.amplitudes.iter().enumerate() {
            let bits = self.index_to_bits(state_idx, model.num_qubits);
            let energy = model.energy(&bits).map_err(QuantumWalkError::IsingError)?;
            mean_energy += amplitude.norm_sqr() * energy;
        }
        
        for _ in 0..self.config.amplification_iterations {
            // Reflect about good solutions (below mean energy)
            for (state_idx, amplitude) in state.amplitudes.iter_mut().enumerate() {
                let bits = self.index_to_bits(state_idx, model.num_qubits);
                let energy = model.energy(&bits).map_err(QuantumWalkError::IsingError)?;
                
                if energy < mean_energy {
                    *amplitude *= -1.0;  // Flip sign for good solutions
                }
            }
            
            // Reflect about average (inversion about average)
            let avg_amplitude: Complex64 = state.amplitudes.iter().sum::<Complex64>() / state.amplitudes.len() as f64;
            for amplitude in &mut state.amplitudes {
                *amplitude = 2.0 * avg_amplitude - *amplitude;
            }
            
            state.normalize();
        }
        
        Ok(())
    }
    
    /// Measure the state and find the best solution
    fn measure_and_optimize(&mut self, model: &IsingModel, state: &QuantumState) -> QuantumWalkResult<(Vec<i8>, f64)> {
        let mut best_spins = vec![1; model.num_qubits];
        let mut best_energy = f64::INFINITY;
        
        // Sample from the quantum state
        for _ in 0..self.config.num_measurements {
            let measured_state = self.sample_state(state);
            let spins = self.index_to_bits(measured_state, model.num_qubits);
            let energy = model.energy(&spins).map_err(QuantumWalkError::IsingError)?;
            
            if energy < best_energy {
                best_energy = energy;
                best_spins = spins;
            }
        }
        
        Ok((best_spins, best_energy))
    }
    
    /// Sample a computational basis state from the quantum state
    fn sample_state(&mut self, state: &QuantumState) -> usize {
        let random_value = self.rng.gen::<f64>();
        let mut cumulative_prob = 0.0;
        
        for (state_idx, amplitude) in state.amplitudes.iter().enumerate() {
            cumulative_prob += amplitude.norm_sqr();
            if random_value <= cumulative_prob {
                return state_idx;
            }
        }
        
        // Fallback (shouldn't reach here with proper normalization)
        state.amplitudes.len() - 1
    }
    
    /// Convert state index to bit string
    fn index_to_bits(&self, state_index: usize, num_qubits: usize) -> Vec<i8> {
        let mut bits = vec![0; num_qubits];
        let mut index = state_index;
        
        for i in 0..num_qubits {
            bits[num_qubits - 1 - i] = if (index & 1) == 1 { 1 } else { -1 };
            index >>= 1;
        }
        
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(2);
        assert_eq!(state.num_qubits, 2);
        assert_eq!(state.amplitudes.len(), 4);
        assert_eq!(state.amplitudes[0], Complex64::new(1.0, 0.0));
    }
    
    #[test]
    fn test_uniform_superposition() {
        let state = QuantumState::uniform_superposition(2);
        let expected_amplitude = Complex64::new(0.5, 0.0);
        
        for amplitude in &state.amplitudes {
            assert!((amplitude - expected_amplitude).norm() < 1e-10);
        }
    }
    
    #[test]
    fn test_state_normalization() {
        let mut state = QuantumState::new(2);
        state.amplitudes[0] = Complex64::new(2.0, 0.0);
        state.amplitudes[1] = Complex64::new(1.0, 0.0);
        
        state.normalize();
        
        let norm_squared: f64 = state.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_bits_conversion() {
        let state = QuantumState::new(3);
        let bits = state.state_to_bits(5);  // 5 = 101 in binary
        assert_eq!(bits, vec![1, -1, 1]);
        
        let index = state.bits_to_state(&bits);
        assert_eq!(index, 5);
    }
    
    #[test]
    fn test_quantum_walk_config() {
        let config = QuantumWalkConfig::default();
        assert!(matches!(config.algorithm, QuantumWalkAlgorithm::ContinuousTime { .. }));
        assert_eq!(config.num_measurements, 1000);
    }
    
    #[test]
    fn test_simple_optimization() {
        let mut model = IsingModel::new(2);
        model.set_coupling(0, 1, -1.0).unwrap();
        
        let config = QuantumWalkConfig {
            algorithm: QuantumWalkAlgorithm::ContinuousTime {
                evolution_time: 0.5,
                time_steps: 50,
            },
            num_measurements: 100,
            seed: Some(42),
            ..Default::default()
        };
        
        let mut optimizer = QuantumWalkOptimizer::new(config);
        let result = optimizer.solve(&model).unwrap();
        
        assert_eq!(result.best_spins.len(), 2);
        assert!(result.best_energy <= 0.0);  // Should find a good solution
    }
}