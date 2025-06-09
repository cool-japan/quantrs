//! Quantum Volume calculation and benchmarking protocol.
//!
//! This module implements the quantum volume protocol for benchmarking quantum
//! computers, including random circuit generation, ideal simulation, heavy output
//! probability calculation, and quantum volume determination.

use std::collections::HashMap;
use num_complex::Complex64;
use ndarray::{Array1, Array2, ArrayView1};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::error::{SimulatorError, Result};
use crate::statevector::StateVectorSimulator;
use crate::scirs2_integration::SciRS2Backend;
use crate::shot_sampling::{QuantumSampler, SamplingConfig};

/// Quantum Volume test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumVolumeResult {
    /// Quantum volume achieved
    pub quantum_volume: u64,
    /// Width tested (number of qubits)
    pub width: usize,
    /// Depth tested
    pub depth: usize,
    /// Number of circuits tested
    pub num_circuits: usize,
    /// Success probability threshold
    pub threshold: f64,
    /// Actual success probability achieved
    pub success_probability: f64,
    /// Heavy output probabilities for each circuit
    pub heavy_output_probs: Vec<f64>,
    /// Statistical confidence
    pub confidence: f64,
    /// Passed the quantum volume test
    pub passed: bool,
    /// Execution statistics
    pub stats: QVStats,
}

/// Statistics for quantum volume calculation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QVStats {
    /// Total execution time (seconds)
    pub total_time_s: f64,
    /// Classical simulation time (seconds)
    pub classical_sim_time_s: f64,
    /// Circuit generation time (seconds)
    pub circuit_gen_time_s: f64,
    /// Heavy output calculation time (seconds)
    pub heavy_output_time_s: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Number of gates in total
    pub total_gates: usize,
    /// Average circuit fidelity
    pub avg_fidelity: f64,
}

/// Single quantum volume circuit
#[derive(Debug, Clone)]
pub struct QVCircuit {
    /// Circuit width (number of qubits)
    pub width: usize,
    /// Circuit depth
    pub depth: usize,
    /// Random SU(4) gates and their locations
    pub gates: Vec<QVGate>,
    /// Permutation applied at the end
    pub permutation: Vec<usize>,
    /// Ideal output amplitudes
    pub ideal_amplitudes: Array1<Complex64>,
    /// Heavy output threshold
    pub heavy_threshold: f64,
    /// Heavy outputs (above threshold)
    pub heavy_outputs: Vec<usize>,
}

/// Quantum volume gate (random SU(4))
#[derive(Debug, Clone)]
pub struct QVGate {
    /// Target qubits (always 2 for SU(4))
    pub qubits: [usize; 2],
    /// Gate matrix (4x4 unitary)
    pub matrix: Array2<Complex64>,
    /// Layer index
    pub layer: usize,
}

/// Quantum Volume calculator
pub struct QuantumVolumeCalculator {
    /// Random number generator
    rng: ChaCha8Rng,
    /// SciRS2 backend for optimization
    backend: Option<SciRS2Backend>,
    /// QV protocol parameters
    params: QVParams,
}

/// Parameters for quantum volume protocol
#[derive(Debug, Clone)]
pub struct QVParams {
    /// Number of circuits to test per width
    pub circuits_per_width: usize,
    /// Heavy output probability threshold (default: 2/3)
    pub heavy_output_threshold: f64,
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Maximum width to test
    pub max_width: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Use parallel execution
    pub parallel: bool,
}

impl Default for QVParams {
    fn default() -> Self {
        Self {
            circuits_per_width: 100,
            heavy_output_threshold: 2.0 / 3.0,
            confidence_level: 0.95,
            max_width: 8,
            seed: None,
            parallel: true,
        }
    }
}

impl QuantumVolumeCalculator {
    /// Create new quantum volume calculator
    pub fn new(params: QVParams) -> Self {
        let rng = if let Some(seed) = params.seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_rng(&mut rand::thread_rng())
        };
        
        Self {
            rng,
            backend: None,
            params,
        }
    }
    
    /// Initialize with SciRS2 backend
    pub fn with_scirs2_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new()?);
        Ok(self)
    }
    
    /// Calculate quantum volume for a given width
    pub fn calculate_quantum_volume(&mut self, width: usize) -> Result<QuantumVolumeResult> {
        let start_time = std::time::Instant::now();
        let depth = width; // Square circuits
        
        let mut stats = QVStats::default();
        let mut heavy_output_probs = Vec::with_capacity(self.params.circuits_per_width);
        
        // Generate and test circuits
        for circuit_idx in 0..self.params.circuits_per_width {
            let circuit_start = std::time::Instant::now();
            
            // Generate random quantum volume circuit
            let circuit = self.generate_qv_circuit(width, depth)?;
            stats.circuit_gen_time_s += circuit_start.elapsed().as_secs_f64();
            
            // Calculate heavy output probability
            let heavy_prob_start = std::time::Instant::now();
            let heavy_prob = self.calculate_heavy_output_probability(&circuit)?;
            stats.heavy_output_time_s += heavy_prob_start.elapsed().as_secs_f64();
            
            heavy_output_probs.push(heavy_prob);
            stats.total_gates += circuit.gates.len();
            
            // Progress tracking
            if circuit_idx % 10 == 0 && circuit_idx > 0 {
                println!("Processed {}/{} circuits for width {}", circuit_idx, self.params.circuits_per_width, width);
            }
        }
        
        // Calculate success probability
        let success_count = heavy_output_probs.iter()
            .filter(|&&prob| prob >= self.params.heavy_output_threshold)
            .count();
        let success_probability = success_count as f64 / heavy_output_probs.len() as f64;
        
        // Statistical test
        let required_success_prob = self.get_required_success_probability(width)?;
        let passed = success_probability >= required_success_prob;
        
        // Calculate quantum volume
        let quantum_volume = if passed { 1 << width } else { 0 };
        
        // Confidence calculation
        let confidence = self.calculate_confidence(&heavy_output_probs, required_success_prob)?;
        
        stats.total_time_s = start_time.elapsed().as_secs_f64();
        stats.memory_usage_bytes = self.estimate_memory_usage(width);
        stats.avg_fidelity = heavy_output_probs.iter().sum::<f64>() / heavy_output_probs.len() as f64;
        
        Ok(QuantumVolumeResult {
            quantum_volume,
            width,
            depth,
            num_circuits: self.params.circuits_per_width,
            threshold: required_success_prob,
            success_probability,
            heavy_output_probs,
            confidence,
            passed,
            stats,
        })
    }
    
    /// Generate random quantum volume circuit
    fn generate_qv_circuit(&mut self, width: usize, depth: usize) -> Result<QVCircuit> {
        if width < 2 {
            return Err(SimulatorError::InvalidInput(
                "Quantum volume requires at least 2 qubits".to_string()
            ));
        }
        
        let mut gates = Vec::new();
        let mut available_qubits: Vec<usize> = (0..width).collect();
        
        // Generate random SU(4) gates for each layer
        for layer in 0..depth {
            // Randomly partition qubits into pairs
            available_qubits.shuffle(&mut self.rng);
            
            // Add SU(4) gates for each pair
            for pair_idx in 0..(width / 2) {
                let qubit1 = available_qubits[2 * pair_idx];
                let qubit2 = available_qubits[2 * pair_idx + 1];
                
                let gate = QVGate {
                    qubits: [qubit1, qubit2],
                    matrix: self.generate_random_su4()?,
                    layer,
                };
                
                gates.push(gate);
            }
        }
        
        // Generate random permutation
        let mut permutation: Vec<usize> = (0..width).collect();
        permutation.shuffle(&mut self.rng);
        
        // Simulate ideal circuit to get amplitudes
        let ideal_amplitudes = self.simulate_ideal_circuit(width, &gates, &permutation)?;
        
        // Calculate heavy output threshold and heavy outputs
        let heavy_threshold = self.calculate_heavy_threshold(&ideal_amplitudes);
        let heavy_outputs = self.find_heavy_outputs(&ideal_amplitudes, heavy_threshold);
        
        Ok(QVCircuit {
            width,
            depth,
            gates,
            permutation,
            ideal_amplitudes,
            heavy_threshold,
            heavy_outputs,
        })
    }
    
    /// Generate random SU(4) matrix
    fn generate_random_su4(&mut self) -> Result<Array2<Complex64>> {
        // Generate random 4x4 unitary matrix
        // Using Hurwitz parametrization for SU(4)
        
        // Generate 15 random parameters (SU(4) has 15 real parameters)
        let mut params = Vec::with_capacity(15);
        for _ in 0..15 {
            params.push(self.rng.gen::<f64>() * 2.0 * std::f64::consts::PI);
        }
        
        // Construct SU(4) matrix using parametrization
        // This is a simplified construction - full implementation would use
        // proper SU(4) parametrization
        let mut matrix = Array2::zeros((4, 4));
        
        // Simplified random unitary construction
        for i in 0..4 {
            for j in 0..4 {
                let real = self.rng.gen::<f64>() - 0.5;
                let imag = self.rng.gen::<f64>() - 0.5;
                matrix[[i, j]] = Complex64::new(real, imag);
            }
        }
        
        // Gram-Schmidt orthogonalization to make it unitary
        self.gram_schmidt_orthogonalize(&mut matrix)?;
        
        Ok(matrix)
    }
    
    /// Gram-Schmidt orthogonalization for making matrix unitary
    fn gram_schmidt_orthogonalize(&self, matrix: &mut Array2<Complex64>) -> Result<()> {
        let n = matrix.nrows();
        
        // Orthogonalize columns
        for j in 0..n {
            // Normalize column j
            let mut norm = 0.0;
            for i in 0..n {
                norm += matrix[[i, j]].norm_sqr();
            }
            norm = norm.sqrt();
            
            if norm > 1e-12 {
                for i in 0..n {
                    matrix[[i, j]] /= norm;
                }
            }
            
            // Orthogonalize remaining columns against column j
            for k in (j+1)..n {
                let mut dot_product = Complex64::new(0.0, 0.0);
                for i in 0..n {
                    dot_product += matrix[[i, j]].conj() * matrix[[i, k]];
                }
                
                let column_j_values: Vec<Complex64> = (0..n).map(|i| matrix[[i, j]]).collect();
                for i in 0..n {
                    matrix[[i, k]] -= dot_product * column_j_values[i];
                }
            }
        }
        
        Ok(())
    }
    
    /// Simulate ideal quantum volume circuit
    fn simulate_ideal_circuit(
        &self,
        width: usize,
        gates: &[QVGate],
        permutation: &[usize],
    ) -> Result<Array1<Complex64>> {
        let dim = 1 << width;
        let mut state = Array1::zeros(dim);
        state[0] = Complex64::new(1.0, 0.0); // |0...0⟩
        
        // Apply gates layer by layer
        for layer in 0..gates.iter().map(|g| g.layer).max().unwrap_or(0) + 1 {
            let layer_gates: Vec<&QVGate> = gates.iter().filter(|g| g.layer == layer).collect();
            
            // Apply all gates in this layer (they commute since they act on disjoint qubits)
            for gate in layer_gates {
                self.apply_two_qubit_gate(&mut state, width, &gate.matrix, gate.qubits[0], gate.qubits[1])?;
            }
        }
        
        // Apply final permutation
        state = self.apply_permutation(&state, width, permutation)?;
        
        Ok(state)
    }
    
    /// Apply two-qubit gate to state vector
    fn apply_two_qubit_gate(
        &self,
        state: &mut Array1<Complex64>,
        width: usize,
        gate_matrix: &Array2<Complex64>,
        qubit1: usize,
        qubit2: usize,
    ) -> Result<()> {
        let dim = 1 << width;
        let mut new_state = Array1::zeros(dim);
        
        for basis_state in 0..dim {
            let bit1 = (basis_state >> (width - 1 - qubit1)) & 1;
            let bit2 = (basis_state >> (width - 1 - qubit2)) & 1;
            let two_qubit_state = (bit1 << 1) | bit2;
            
            for target_two_qubit in 0..4 {
                let amplitude = gate_matrix[[target_two_qubit, two_qubit_state]];
                
                if amplitude.norm() > 1e-12 {
                    let target_bit1 = (target_two_qubit >> 1) & 1;
                    let target_bit2 = target_two_qubit & 1;
                    
                    let mut target_basis = basis_state;
                    
                    // Update qubit1
                    if target_bit1 == 1 {
                        target_basis |= 1 << (width - 1 - qubit1);
                    } else {
                        target_basis &= !(1 << (width - 1 - qubit1));
                    }
                    
                    // Update qubit2
                    if target_bit2 == 1 {
                        target_basis |= 1 << (width - 1 - qubit2);
                    } else {
                        target_basis &= !(1 << (width - 1 - qubit2));
                    }
                    
                    new_state[target_basis] += amplitude * state[basis_state];
                }
            }
        }
        
        *state = new_state;
        Ok(())
    }
    
    /// Apply permutation to state vector
    fn apply_permutation(
        &self,
        state: &Array1<Complex64>,
        width: usize,
        permutation: &[usize],
    ) -> Result<Array1<Complex64>> {
        let dim = 1 << width;
        let mut new_state = Array1::zeros(dim);
        
        for basis_state in 0..dim {
            let mut permuted_state = 0;
            
            for (new_pos, &old_pos) in permutation.iter().enumerate() {
                let bit = (basis_state >> (width - 1 - old_pos)) & 1;
                permuted_state |= bit << (width - 1 - new_pos);
            }
            
            new_state[permuted_state] = state[basis_state];
        }
        
        Ok(new_state)
    }
    
    /// Calculate heavy output threshold (median probability)
    fn calculate_heavy_threshold(&self, amplitudes: &Array1<Complex64>) -> f64 {
        let mut probabilities: Vec<f64> = amplitudes.iter().map(|amp| amp.norm_sqr()).collect();
        probabilities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median_idx = probabilities.len() / 2;
        probabilities[median_idx]
    }
    
    /// Find heavy outputs (above threshold)
    fn find_heavy_outputs(&self, amplitudes: &Array1<Complex64>, threshold: f64) -> Vec<usize> {
        amplitudes.iter().enumerate()
            .filter_map(|(idx, amp)| {
                if amp.norm_sqr() > threshold {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Calculate heavy output probability for a circuit
    fn calculate_heavy_output_probability(&self, circuit: &QVCircuit) -> Result<f64> {
        // Sum probabilities of all heavy outputs
        let heavy_prob: f64 = circuit.heavy_outputs.iter()
            .map(|&idx| circuit.ideal_amplitudes[idx].norm_sqr())
            .sum();
        
        Ok(heavy_prob)
    }
    
    /// Get required success probability for statistical test
    fn get_required_success_probability(&self, width: usize) -> Result<f64> {
        // For quantum volume, we typically require 2/3 probability with high confidence
        // The exact threshold depends on the number of circuits and confidence level
        
        let baseline_threshold = 2.0 / 3.0;
        
        // Adjust for statistical fluctuations based on number of circuits
        let n = self.params.circuits_per_width as f64;
        let confidence = self.params.confidence_level;
        
        // Use normal approximation for large n
        let z_score = match confidence {
            x if x >= 0.99 => 2.576,
            x if x >= 0.95 => 1.96,
            x if x >= 0.90 => 1.645,
            _ => 1.96,
        };
        
        let standard_error = (baseline_threshold * (1.0 - baseline_threshold) / n).sqrt();
        let adjusted_threshold = baseline_threshold - z_score * standard_error;
        
        Ok(adjusted_threshold.max(0.5)) // At least better than random
    }
    
    /// Calculate confidence in the result
    fn calculate_confidence(&self, heavy_probs: &[f64], threshold: f64) -> Result<f64> {
        let n = heavy_probs.len() as f64;
        let success_count = heavy_probs.iter().filter(|&&p| p >= threshold).count() as f64;
        let success_rate = success_count / n;
        
        // Binomial confidence interval
        let p = success_rate;
        let standard_error = (p * (1.0 - p) / n).sqrt();
        
        // Return confidence that we're above threshold
        if success_rate > threshold {
            let z_score = (success_rate - threshold) / standard_error;
            Ok(self.normal_cdf(z_score))
        } else {
            Ok(0.0)
        }
    }
    
    /// Normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }
    
    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }
    
    /// Estimate memory usage
    fn estimate_memory_usage(&self, width: usize) -> usize {
        let state_vector_size = (1 << width) * std::mem::size_of::<Complex64>();
        let circuit_storage = 1000 * std::mem::size_of::<QVGate>(); // Rough estimate
        state_vector_size + circuit_storage
    }
    
    /// Find maximum quantum volume by testing increasing widths
    pub fn find_maximum_quantum_volume(&mut self) -> Result<Vec<QuantumVolumeResult>> {
        let mut results = Vec::new();
        
        for width in 2..=self.params.max_width {
            println!("Testing quantum volume for width {}", width);
            
            let result = self.calculate_quantum_volume(width)?;
            
            println!("Width {}: QV = {}, Success Prob = {:.3}, Passed = {}", 
                     width, result.quantum_volume, result.success_probability, result.passed);
            
            results.push(result.clone());
            
            // If we fail, we've found the maximum quantum volume
            if !result.passed {
                break;
            }
        }
        
        Ok(results)
    }
}

/// Utility trait for shuffling
trait SliceMutExt<T> {
    fn shuffle(&mut self, rng: &mut impl Rng);
}

impl<T> SliceMutExt<T> for [T] {
    fn shuffle(&mut self, rng: &mut impl Rng) {
        for i in (1..self.len()).rev() {
            let j = rng.gen_range(0..=i);
            self.swap(i, j);
        }
    }
}

/// Benchmark quantum volume calculation
pub fn benchmark_quantum_volume(max_width: usize) -> Result<Vec<QuantumVolumeResult>> {
    let params = QVParams {
        circuits_per_width: 20, // Reduced for benchmarking
        max_width,
        ..Default::default()
    };
    
    let mut calculator = QuantumVolumeCalculator::new(params);
    calculator.find_maximum_quantum_volume()
}

/// Calculate quantum volume for specific parameters
pub fn calculate_quantum_volume_with_params(
    width: usize,
    circuits: usize,
    seed: Option<u64>,
) -> Result<QuantumVolumeResult> {
    let params = QVParams {
        circuits_per_width: circuits,
        max_width: width,
        seed,
        ..Default::default()
    };
    
    let mut calculator = QuantumVolumeCalculator::new(params);
    calculator.calculate_quantum_volume(width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qv_calculator_creation() {
        let params = QVParams::default();
        let calculator = QuantumVolumeCalculator::new(params);
        assert!(calculator.params.circuits_per_width > 0);
    }

    #[test]
    fn test_random_su4_generation() {
        let params = QVParams::default();
        let mut calculator = QuantumVolumeCalculator::new(params);
        
        let matrix = calculator.generate_random_su4().unwrap();
        assert_eq!(matrix.shape(), [4, 4]);
        
        // Check that it's approximately unitary (U†U ≈ I)
        let matrix_dagger = matrix.t().mapv(|x| x.conj());
        let product = matrix_dagger.dot(&matrix);
        
        // Check diagonal elements are close to 1
        for i in 0..4 {
            assert!((product[[i, i]].re - 1.0).abs() < 1e-1);
            assert!(product[[i, i]].im.abs() < 1e-1);
        }
    }

    #[test]
    fn test_qv_circuit_generation() {
        let params = QVParams::default();
        let mut calculator = QuantumVolumeCalculator::new(params);
        
        let circuit = calculator.generate_qv_circuit(4, 4).unwrap();
        assert_eq!(circuit.width, 4);
        assert_eq!(circuit.depth, 4);
        assert!(!circuit.gates.is_empty());
        assert_eq!(circuit.permutation.len(), 4);
    }

    #[test]
    fn test_heavy_output_calculation() {
        let amplitudes = Array1::from_vec(vec![
            Complex64::new(0.6, 0.0),  // High probability
            Complex64::new(0.3, 0.0),  // Medium probability
            Complex64::new(0.1, 0.0),  // Low probability
            Complex64::new(0.05, 0.0), // Very low probability
        ]);
        
        let params = QVParams::default();
        let calculator = QuantumVolumeCalculator::new(params);
        
        let threshold = calculator.calculate_heavy_threshold(&amplitudes);
        let heavy_outputs = calculator.find_heavy_outputs(&amplitudes, threshold);
        
        assert!(threshold > 0.0);
        assert!(!heavy_outputs.is_empty());
    }

    #[test]
    fn test_small_quantum_volume() {
        let result = calculate_quantum_volume_with_params(3, 10, Some(42)).unwrap();
        
        assert_eq!(result.width, 3);
        assert_eq!(result.depth, 3);
        assert_eq!(result.num_circuits, 10);
        assert!(result.quantum_volume >= 0);
        assert!(!result.heavy_output_probs.is_empty());
    }

    #[test]
    fn test_permutation_application() {
        let params = QVParams::default();
        let calculator = QuantumVolumeCalculator::new(params);
        
        let state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        
        let permutation = vec![1, 0]; // Swap qubits
        let permuted = calculator.apply_permutation(&state, 2, &permutation).unwrap();
        
        // |00⟩ should become |00⟩ (no change for this state)
        assert!((permuted[0].re - 1.0).abs() < 1e-10);
    }
}