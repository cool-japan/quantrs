//! Concatenated Quantum Error Correction with Hierarchical Decoding
//!
//! This module implements concatenated quantum error correction codes that provide
//! enhanced error protection through multiple levels of encoding. Concatenated codes
//! work by applying error correction codes recursively: the physical qubits of an
//! outer code are themselves logical qubits encoded using an inner code.
//!
//! This implementation features:
//! - Multiple concatenation levels for exponential error reduction
//! - Hierarchical decoding with error propagation tracking
//! - Adaptive thresholds based on error rates
//! - Resource-efficient syndrome processing
//! - Support for heterogeneous inner and outer codes

use scirs2_core::ndarray::Array1;
use scirs2_core::parallel_ops::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::Result;
// Remove the invalid imports - we'll define our own implementations

/// Concatenation level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcatenationLevel {
    /// Level identifier (0 = innermost, higher = outer levels)
    pub level: usize,
    /// Code distance at this level
    pub distance: usize,
    /// Number of physical qubits per logical qubit at this level
    pub code_rate: usize,
}

/// Concatenated code specification
#[derive(Debug)]
pub struct ConcatenatedCodeConfig {
    /// Concatenation levels from inner to outer
    pub levels: Vec<ConcatenationLevel>,
    /// Error correction codes used at each level
    pub codes_per_level: Vec<Box<dyn ErrorCorrectionCode>>,
    /// Decoding method
    pub decoding_method: HierarchicalDecodingMethod,
    /// Error rate threshold for adaptive decoding
    pub error_threshold: f64,
    /// Enable parallel decoding at each level
    pub parallel_decoding: bool,
    /// Maximum decoding iterations
    pub max_decoding_iterations: usize,
}

/// Error correction code trait for concatenation
pub trait ErrorCorrectionCode: Send + Sync + std::fmt::Debug {
    /// Get code parameters
    fn get_parameters(&self) -> CodeParameters;

    /// Encode logical qubits
    fn encode(&self, logical_state: &Array1<Complex64>) -> Result<Array1<Complex64>>;

    /// Decode with syndrome extraction
    fn decode(&self, encoded_state: &Array1<Complex64>) -> Result<DecodingResult>;

    /// Generate syndrome extraction circuit
    fn syndrome_circuit(&self, num_qubits: usize) -> Result<InterfaceCircuit>;

    /// Apply error correction based on syndrome
    fn correct_errors(&self, state: &mut Array1<Complex64>, syndrome: &[bool]) -> Result<()>;
}

/// Code parameters
#[derive(Debug, Clone, Copy)]
pub struct CodeParameters {
    /// Number of logical qubits
    pub n_logical: usize,
    /// Number of physical qubits
    pub n_physical: usize,
    /// Code distance
    pub distance: usize,
    /// Error correction capability
    pub t: usize, // Can correct up to t errors
}

/// Hierarchical decoding method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchicalDecodingMethod {
    /// Sequential decoding from inner to outer levels
    Sequential,
    /// Parallel decoding across levels
    Parallel,
    /// Adaptive decoding based on error patterns
    Adaptive,
    /// Belief propagation between levels
    BeliefPropagation,
}

/// Decoding result with error information
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// Corrected state
    pub corrected_state: Array1<Complex64>,
    /// Syndrome measurements
    pub syndrome: Vec<bool>,
    /// Error pattern detected
    pub error_pattern: Vec<ErrorType>,
    /// Decoding confidence
    pub confidence: f64,
    /// Number of errors corrected
    pub errors_corrected: usize,
    /// Decoding successful
    pub success: bool,
}

/// Types of quantum errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorType {
    /// No error
    Identity,
    /// Bit flip (X error)
    BitFlip,
    /// Phase flip (Z error)
    PhaseFlip,
    /// Bit-phase flip (Y error)
    BitPhaseFlip,
}

/// Concatenated error correction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatenatedCorrectionResult {
    /// Final corrected state
    pub final_state: Array1<Complex64>,
    /// Results from each concatenation level
    pub level_results: Vec<LevelDecodingResult>,
    /// Overall decoding statistics
    pub stats: ConcatenationStats,
    /// Total execution time
    pub execution_time_ms: f64,
    /// Success probability estimate
    pub success_probability: f64,
}

/// Decoding result for a single level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelDecodingResult {
    /// Concatenation level
    pub level: usize,
    /// Syndrome measurements
    pub syndromes: Vec<Vec<bool>>,
    /// Errors corrected at this level
    pub errors_corrected: usize,
    /// Error patterns detected
    pub error_patterns: Vec<String>,
    /// Decoding confidence
    pub confidence: f64,
    /// Processing time for this level
    pub processing_time_ms: f64,
}

/// Concatenation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcatenationStats {
    /// Total physical qubits used
    pub physical_qubits: usize,
    /// Total logical qubits encoded
    pub logical_qubits: usize,
    /// Overall code distance
    pub effective_distance: usize,
    /// Total syndrome measurements
    pub syndrome_measurements: usize,
    /// Total errors corrected
    pub total_errors_corrected: usize,
    /// Memory overhead
    pub memory_overhead_factor: f64,
    /// Circuit depth overhead
    pub circuit_depth_overhead: usize,
    /// Decoding iterations performed
    pub decoding_iterations: usize,
}

/// Concatenated quantum error correction implementation
pub struct ConcatenatedErrorCorrection {
    /// Configuration
    config: ConcatenatedCodeConfig,
    /// Circuit interface for compilation
    circuit_interface: CircuitInterface,
    /// Syndrome history for adaptive decoding
    syndrome_history: VecDeque<Vec<Vec<bool>>>,
    /// Error rate tracking
    error_rates: HashMap<usize, f64>,
    /// Statistics
    stats: ConcatenationStats,
}

impl ConcatenatedErrorCorrection {
    /// Create new concatenated error correction instance
    pub fn new(config: ConcatenatedCodeConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;
        let syndrome_history = VecDeque::with_capacity(100);
        let error_rates = HashMap::new();

        Ok(Self {
            config,
            circuit_interface,
            syndrome_history,
            error_rates,
            stats: ConcatenationStats::default(),
        })
    }

    /// Encode logical state using concatenated codes
    pub fn encode_concatenated(
        &mut self,
        logical_state: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        let mut current_state = logical_state.clone();

        // Apply encoding at each concatenation level
        for (level, code) in self.config.codes_per_level.iter().enumerate() {
            current_state = code.encode(&current_state)?;

            let params = code.get_parameters();
            self.stats.physical_qubits = params.n_physical;
            self.stats.logical_qubits = params.n_logical;

            // Update effective distance (minimum over all levels)
            if level == 0 {
                self.stats.effective_distance = params.distance;
            } else {
                self.stats.effective_distance = self.stats.effective_distance.min(params.distance);
            }
        }

        Ok(current_state)
    }

    /// Decode using hierarchical error correction
    pub fn decode_hierarchical(
        &mut self,
        encoded_state: &Array1<Complex64>,
    ) -> Result<ConcatenatedCorrectionResult> {
        let start_time = std::time::Instant::now();

        let result = match self.config.decoding_method {
            HierarchicalDecodingMethod::Sequential => self.decode_sequential(encoded_state)?,
            HierarchicalDecodingMethod::Parallel => self.decode_parallel(encoded_state)?,
            HierarchicalDecodingMethod::Adaptive => self.decode_adaptive(encoded_state)?,
            HierarchicalDecodingMethod::BeliefPropagation => {
                self.decode_belief_propagation(encoded_state)?
            }
        };

        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Update syndrome history
        let all_syndromes: Vec<Vec<bool>> = result
            .level_results
            .iter()
            .flat_map(|r| r.syndromes.iter().cloned())
            .collect();
        self.syndrome_history.push_back(all_syndromes);
        if self.syndrome_history.len() > 100 {
            self.syndrome_history.pop_front();
        }

        // Calculate success probability
        let success_probability = self.estimate_success_probability(&result);

        Ok(ConcatenatedCorrectionResult {
            final_state: result.final_state,
            level_results: result.level_results,
            stats: self.stats.clone(),
            execution_time_ms,
            success_probability,
        })
    }

    /// Sequential decoding from outer to inner levels
    fn decode_sequential(
        &mut self,
        encoded_state: &Array1<Complex64>,
    ) -> Result<ConcatenatedCorrectionResult> {
        let mut current_state = encoded_state.clone();
        let mut level_results = Vec::new();

        // Decode from outermost to innermost level
        for (level, code) in self.config.codes_per_level.iter().enumerate().rev() {
            let level_start = std::time::Instant::now();

            let decoding_result = code.decode(&current_state)?;
            current_state = decoding_result.corrected_state;

            // Convert error pattern to strings for serialization
            let error_patterns: Vec<String> = decoding_result
                .error_pattern
                .iter()
                .map(|e| format!("{e:?}"))
                .collect();

            let level_result = LevelDecodingResult {
                level,
                syndromes: vec![decoding_result.syndrome],
                errors_corrected: decoding_result.errors_corrected,
                error_patterns,
                confidence: decoding_result.confidence,
                processing_time_ms: level_start.elapsed().as_secs_f64() * 1000.0,
            };

            level_results.push(level_result);
            self.stats.total_errors_corrected += decoding_result.errors_corrected;
            self.stats.decoding_iterations += 1;
        }

        // Reverse to get correct order (inner to outer)
        level_results.reverse();

        Ok(ConcatenatedCorrectionResult {
            final_state: current_state,
            level_results,
            stats: self.stats.clone(),
            execution_time_ms: 0.0,   // Will be set by caller
            success_probability: 0.0, // Will be calculated by caller
        })
    }

    /// Parallel decoding across levels
    fn decode_parallel(
        &mut self,
        encoded_state: &Array1<Complex64>,
    ) -> Result<ConcatenatedCorrectionResult> {
        if !self.config.parallel_decoding {
            return self.decode_sequential(encoded_state);
        }

        // For true parallel decoding, we need to carefully manage state sharing
        // This is a simplified implementation
        let num_levels = self.config.codes_per_level.len();
        let mut level_results = Vec::with_capacity(num_levels);

        // Process levels in parallel where possible
        let results: Vec<_> = (0..num_levels)
            .into_par_iter()
            .map(|level| {
                let level_start = std::time::Instant::now();

                // For parallel processing, we simulate on a copy of the state
                let mut state_copy = encoded_state.clone();

                let decoding_result = self.config.codes_per_level[level]
                    .decode(&state_copy)
                    .unwrap_or_else(|_| DecodingResult {
                        corrected_state: state_copy,
                        syndrome: vec![false],
                        error_pattern: vec![ErrorType::Identity],
                        confidence: 0.0,
                        errors_corrected: 0,
                        success: false,
                    });

                let error_patterns: Vec<String> = decoding_result
                    .error_pattern
                    .iter()
                    .map(|e| format!("{e:?}"))
                    .collect();

                LevelDecodingResult {
                    level,
                    syndromes: vec![decoding_result.syndrome],
                    errors_corrected: decoding_result.errors_corrected,
                    error_patterns,
                    confidence: decoding_result.confidence,
                    processing_time_ms: level_start.elapsed().as_secs_f64() * 1000.0,
                }
            })
            .collect();

        level_results.extend(results);

        // For simplicity, use the final state from sequential decoding
        let sequential_result = self.decode_sequential(encoded_state)?;

        Ok(ConcatenatedCorrectionResult {
            final_state: sequential_result.final_state,
            level_results,
            stats: self.stats.clone(),
            execution_time_ms: 0.0,
            success_probability: 0.0,
        })
    }

    /// Adaptive decoding based on error patterns
    fn decode_adaptive(
        &mut self,
        encoded_state: &Array1<Complex64>,
    ) -> Result<ConcatenatedCorrectionResult> {
        // Start with sequential decoding
        let mut result = self.decode_sequential(encoded_state)?;

        // Analyze error patterns to decide if additional iterations are needed
        let error_rate = self.calculate_current_error_rate(&result.level_results);

        if error_rate > self.config.error_threshold {
            // High error rate detected - try alternative decoding strategies
            for iteration in 1..self.config.max_decoding_iterations {
                let alternative_result = if iteration % 2 == 1 {
                    self.decode_parallel(encoded_state)?
                } else {
                    self.decode_sequential(encoded_state)?
                };

                let alt_error_rate =
                    self.calculate_current_error_rate(&alternative_result.level_results);
                if alt_error_rate < error_rate {
                    result = alternative_result;
                    break;
                }

                self.stats.decoding_iterations += 1;
            }
        }

        Ok(result)
    }

    /// Belief propagation decoding between levels
    fn decode_belief_propagation(
        &mut self,
        encoded_state: &Array1<Complex64>,
    ) -> Result<ConcatenatedCorrectionResult> {
        // Simplified belief propagation - in practice this would be much more complex
        let mut current_state = encoded_state.clone();
        let mut level_results = Vec::new();

        // Initialize belief messages
        let num_levels = self.config.codes_per_level.len();
        let mut beliefs: Vec<f64> = vec![1.0; num_levels];

        for iteration in 0..self.config.max_decoding_iterations.min(5) {
            for (level, code) in self.config.codes_per_level.iter().enumerate() {
                let level_start = std::time::Instant::now();

                // Decode with current beliefs
                let decoding_result = code.decode(&current_state)?;

                // Update beliefs based on decoding confidence
                beliefs[level] = beliefs[level].mul_add(0.9, decoding_result.confidence * 0.1);

                current_state = decoding_result.corrected_state;

                let error_patterns: Vec<String> = decoding_result
                    .error_pattern
                    .iter()
                    .map(|e| format!("{e:?}"))
                    .collect();

                let level_result = LevelDecodingResult {
                    level,
                    syndromes: vec![decoding_result.syndrome],
                    errors_corrected: decoding_result.errors_corrected,
                    error_patterns,
                    confidence: beliefs[level],
                    processing_time_ms: level_start.elapsed().as_secs_f64() * 1000.0,
                };

                if iteration == 0 || level_results.len() <= level {
                    level_results.push(level_result);
                } else {
                    level_results[level] = level_result;
                }

                self.stats.total_errors_corrected += decoding_result.errors_corrected;
            }

            // Check convergence
            let avg_confidence: f64 = beliefs.iter().sum::<f64>() / beliefs.len() as f64;
            if avg_confidence > 0.95 {
                break;
            }

            self.stats.decoding_iterations += 1;
        }

        Ok(ConcatenatedCorrectionResult {
            final_state: current_state,
            level_results,
            stats: self.stats.clone(),
            execution_time_ms: 0.0,
            success_probability: 0.0,
        })
    }

    /// Calculate current error rate from level results
    fn calculate_current_error_rate(&self, level_results: &[LevelDecodingResult]) -> f64 {
        if level_results.is_empty() {
            return 0.0;
        }

        let total_errors: usize = level_results.iter().map(|r| r.errors_corrected).sum();

        let total_qubits = self.stats.physical_qubits.max(1);
        total_errors as f64 / total_qubits as f64
    }

    /// Estimate success probability based on decoding results
    fn estimate_success_probability(&self, result: &ConcatenatedCorrectionResult) -> f64 {
        if result.level_results.is_empty() {
            return 1.0;
        }

        // Product of confidences across all levels
        let confidence_product: f64 = result.level_results.iter().map(|r| r.confidence).product();

        // Adjust based on error rate
        let error_rate = self.calculate_current_error_rate(&result.level_results);
        let error_penalty = (-error_rate * 10.0).exp();

        (confidence_product * error_penalty).min(1.0).max(0.0)
    }

    /// Get current statistics
    #[must_use]
    pub const fn get_stats(&self) -> &ConcatenationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ConcatenationStats::default();
        self.syndrome_history.clear();
        self.error_rates.clear();
    }
}

/// Implementation of error correction codes for concatenation
/// Simple bit flip code implementation
#[derive(Debug, Clone)]
pub struct BitFlipCode;

impl Default for BitFlipCode {
    fn default() -> Self {
        Self::new()
    }
}

impl BitFlipCode {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Wrapper for bit flip code
#[derive(Debug)]
pub struct ConcatenatedBitFlipCode {
    inner_code: BitFlipCode,
}

impl Default for ConcatenatedBitFlipCode {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcatenatedBitFlipCode {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner_code: BitFlipCode::new(),
        }
    }
}

impl ErrorCorrectionCode for ConcatenatedBitFlipCode {
    fn get_parameters(&self) -> CodeParameters {
        CodeParameters {
            n_logical: 1,
            n_physical: 3,
            distance: 3,
            t: 1,
        }
    }

    fn encode(&self, logical_state: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        // Simulate bit flip encoding
        let n_logical = logical_state.len();
        let n_physical = n_logical * 3;

        let mut encoded = Array1::zeros(n_physical);

        // Triple each logical qubit
        for i in 0..n_logical {
            let amp = logical_state[i];
            encoded[i * 3] = amp;
            encoded[i * 3 + 1] = amp;
            encoded[i * 3 + 2] = amp;
        }

        Ok(encoded)
    }

    fn decode(&self, encoded_state: &Array1<Complex64>) -> Result<DecodingResult> {
        let n_physical = encoded_state.len();
        let n_logical = n_physical / 3;

        let mut corrected_state = Array1::zeros(n_logical);
        let mut syndrome = Vec::new();
        let mut error_pattern = Vec::new();
        let mut errors_corrected = 0;

        for i in 0..n_logical {
            let block_start = i * 3;
            let a0 = encoded_state[block_start];
            let a1 = encoded_state[block_start + 1];
            let a2 = encoded_state[block_start + 2];

            // Majority vote (simplified)
            let distances = [(a0 - a1).norm(), (a1 - a2).norm(), (a0 - a2).norm()];

            let min_dist_idx = distances
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            match min_dist_idx {
                0 => {
                    // a0 and a1 are closest
                    corrected_state[i] = (a0 + a1) / 2.0;
                    if (a2 - a0).norm() > 1e-10 {
                        syndrome.push(true);
                        error_pattern.push(ErrorType::BitFlip);
                        errors_corrected += 1;
                    } else {
                        syndrome.push(false);
                        error_pattern.push(ErrorType::Identity);
                    }
                }
                1 => {
                    // a1 and a2 are closest
                    corrected_state[i] = (a1 + a2) / 2.0;
                    if (a0 - a1).norm() > 1e-10 {
                        syndrome.push(true);
                        error_pattern.push(ErrorType::BitFlip);
                        errors_corrected += 1;
                    } else {
                        syndrome.push(false);
                        error_pattern.push(ErrorType::Identity);
                    }
                }
                2 => {
                    // a0 and a2 are closest
                    corrected_state[i] = (a0 + a2) / 2.0;
                    if (a1 - a0).norm() > 1e-10 {
                        syndrome.push(true);
                        error_pattern.push(ErrorType::BitFlip);
                        errors_corrected += 1;
                    } else {
                        syndrome.push(false);
                        error_pattern.push(ErrorType::Identity);
                    }
                }
                _ => unreachable!(),
            }
        }

        let confidence = 1.0 - (errors_corrected as f64 / n_logical as f64);

        Ok(DecodingResult {
            corrected_state,
            syndrome,
            error_pattern,
            confidence,
            errors_corrected,
            success: errors_corrected <= n_logical,
        })
    }

    fn syndrome_circuit(&self, num_qubits: usize) -> Result<InterfaceCircuit> {
        let mut circuit = InterfaceCircuit::new(num_qubits + 2, 2);

        // Simple syndrome extraction for bit flip code
        for i in (0..num_qubits).step_by(3) {
            if i + 2 < num_qubits {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![i, num_qubits],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![i + 1, num_qubits],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![i + 1, num_qubits + 1],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![i + 2, num_qubits + 1],
                ));
            }
        }

        Ok(circuit)
    }

    fn correct_errors(&self, state: &mut Array1<Complex64>, syndrome: &[bool]) -> Result<()> {
        // Apply corrections based on syndrome
        for (i, &has_error) in syndrome.iter().enumerate() {
            if has_error && i * 3 + 2 < state.len() {
                // Apply majority vote correction
                let block_start = i * 3;
                let majority =
                    (state[block_start] + state[block_start + 1] + state[block_start + 2]) / 3.0;
                state[block_start] = majority;
                state[block_start + 1] = majority;
                state[block_start + 2] = majority;
            }
        }
        Ok(())
    }
}

/// Create concatenated error correction with predefined configuration
pub fn create_standard_concatenated_code(levels: usize) -> Result<ConcatenatedErrorCorrection> {
    let mut concatenation_levels = Vec::new();
    let mut codes_per_level: Vec<Box<dyn ErrorCorrectionCode>> = Vec::new();

    for level in 0..levels {
        concatenation_levels.push(ConcatenationLevel {
            level,
            distance: 3,
            code_rate: 3,
        });

        codes_per_level.push(Box::new(ConcatenatedBitFlipCode::new()));
    }

    let config = ConcatenatedCodeConfig {
        levels: concatenation_levels,
        codes_per_level,
        decoding_method: HierarchicalDecodingMethod::Sequential,
        error_threshold: 0.1,
        parallel_decoding: true,
        max_decoding_iterations: 10,
    };

    ConcatenatedErrorCorrection::new(config)
}

/// Benchmark concatenated error correction performance
pub fn benchmark_concatenated_error_correction() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // Test different concatenation levels
    let levels = vec![1, 2, 3];

    for &level in &levels {
        let start = std::time::Instant::now();

        let mut concatenated = create_standard_concatenated_code(level)?;

        // Create test logical state
        let logical_state = Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);

        // Encode
        let encoded = concatenated.encode_concatenated(&logical_state)?;

        // Simulate some errors by adding noise
        let mut noisy_encoded = encoded.clone();
        for i in 0..noisy_encoded.len().min(5) {
            noisy_encoded[i] += Complex64::new(0.01 * fastrand::f64(), 0.01 * fastrand::f64());
        }

        // Decode
        let _result = concatenated.decode_hierarchical(&noisy_encoded)?;

        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("level_{level}"), time);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_concatenated_code_creation() {
        let concatenated = create_standard_concatenated_code(2);
        assert!(concatenated.is_ok());
    }

    #[test]
    fn test_bit_flip_code_parameters() {
        let code = ConcatenatedBitFlipCode::new();
        let params = code.get_parameters();

        assert_eq!(params.n_logical, 1);
        assert_eq!(params.n_physical, 3);
        assert_eq!(params.distance, 3);
        assert_eq!(params.t, 1);
    }

    #[test]
    fn test_bit_flip_encoding() {
        let code = ConcatenatedBitFlipCode::new();
        let logical_state =
            Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        let encoded = code
            .encode(&logical_state)
            .expect("Encoding should succeed in test");
        assert_eq!(encoded.len(), 6); // 2 logical -> 6 physical

        // Check triplication
        assert!((encoded[0] - logical_state[0]).norm() < 1e-10);
        assert!((encoded[1] - logical_state[0]).norm() < 1e-10);
        assert!((encoded[2] - logical_state[0]).norm() < 1e-10);
    }

    #[test]
    fn test_concatenated_encoding_decoding() {
        let mut concatenated = create_standard_concatenated_code(1)
            .expect("Concatenated code creation should succeed in test");

        let logical_state = Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);

        let encoded = concatenated
            .encode_concatenated(&logical_state)
            .expect("Encoding should succeed in test");
        assert!(encoded.len() >= logical_state.len());

        let result = concatenated
            .decode_hierarchical(&encoded)
            .expect("Decoding should succeed in test");
        assert!(!result.level_results.is_empty());
        assert!(result.success_probability >= 0.0);
    }

    #[test]
    fn test_syndrome_circuit_creation() {
        let code = ConcatenatedBitFlipCode::new();
        let circuit = code
            .syndrome_circuit(6)
            .expect("Syndrome circuit creation should succeed in test");

        assert_eq!(circuit.num_qubits, 8); // 6 data + 2 syndrome qubits
        assert!(!circuit.gates.is_empty());
    }

    #[test]
    fn test_decoding_methods() {
        let mut concatenated = create_standard_concatenated_code(1)
            .expect("Concatenated code creation should succeed in test");

        let logical_state = Array1::from_vec(vec![Complex64::new(1.0, 0.0)]);
        let encoded = concatenated
            .encode_concatenated(&logical_state)
            .expect("Encoding should succeed in test");

        // Test sequential decoding
        concatenated.config.decoding_method = HierarchicalDecodingMethod::Sequential;
        let seq_result = concatenated
            .decode_hierarchical(&encoded)
            .expect("Sequential decoding should succeed in test");
        assert!(!seq_result.level_results.is_empty());

        // Test adaptive decoding
        concatenated.config.decoding_method = HierarchicalDecodingMethod::Adaptive;
        let adapt_result = concatenated
            .decode_hierarchical(&encoded)
            .expect("Adaptive decoding should succeed in test");
        assert!(!adapt_result.level_results.is_empty());
    }

    #[test]
    fn test_error_rate_calculation() {
        let concatenated = create_standard_concatenated_code(1)
            .expect("Concatenated code creation should succeed in test");

        let level_results = vec![LevelDecodingResult {
            level: 0,
            syndromes: vec![vec![true, false]],
            errors_corrected: 1,
            error_patterns: vec!["BitFlip".to_string()],
            confidence: 0.9,
            processing_time_ms: 1.0,
        }];

        let error_rate = concatenated.calculate_current_error_rate(&level_results);
        assert!(error_rate > 0.0);
    }
}
