//! Error Syndrome Detection and Correction
//!
//! This module implements quantum error syndrome detection and correction algorithms
//! specifically designed for quantum annealing systems. It provides functionality for:
//! - Real-time syndrome detection during annealing
//! - Error classification and pattern recognition
//! - Correction protocol execution
//! - Integration with annealing schedules

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::ising::{IsingModel};
use crate::simulator::AnnealingResult;
use crate::qaoa::QuantumState;
use super::config::{QECResult, QuantumErrorCorrectionError};
use super::codes::{ErrorCorrectionCode, CodeParameters};
use super::logical_operations::LogicalOperation;

/// Syndrome detection and correction engine
#[derive(Debug, Clone)]
pub struct SyndromeDetector {
    /// Error correction code being used
    pub code: ErrorCorrectionCode,
    /// Code parameters
    pub parameters: CodeParameters,
    /// Stabilizer generators for syndrome detection
    pub stabilizer_generators: Array2<u8>,
    /// Parity check matrix
    pub parity_check_matrix: Array2<u8>,
    /// Correction lookup table
    pub correction_table: HashMap<Vec<u8>, CorrectionOperation>,
    /// Detection statistics
    pub detection_stats: DetectionStatistics,
    /// Configuration parameters
    pub config: SyndromeDetectorConfig,
}

/// Configuration for syndrome detection
#[derive(Debug, Clone)]
pub struct SyndromeDetectorConfig {
    /// Syndrome measurement frequency (Hz)
    pub measurement_frequency: f64,
    /// Error probability threshold for correction
    pub correction_threshold: f64,
    /// Maximum syndrome history to keep
    pub max_history_length: usize,
    /// Enable adaptive correction timing
    pub adaptive_timing: bool,
    /// Noise model parameters
    pub noise_model: NoiseModel,
    /// Decoding algorithm
    pub decoder: DecodingAlgorithm,
}

/// Noise model for syndrome detection
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Single-qubit depolarizing error rate
    pub single_qubit_error_rate: f64,
    /// Two-qubit gate error rate
    pub two_qubit_error_rate: f64,
    /// Measurement error rate
    pub measurement_error_rate: f64,
    /// Coherence time (microseconds)
    pub coherence_time: f64,
    /// Temperature effects
    pub thermal_noise_rate: f64,
}

/// Decoding algorithms for error correction
#[derive(Debug, Clone, PartialEq)]
pub enum DecodingAlgorithm {
    /// Minimum weight perfect matching
    MinimumWeight,
    /// Maximum likelihood decoding
    MaximumLikelihood,
    /// Neural network decoder
    NeuralNetwork,
    /// Belief propagation
    BeliefPropagation,
    /// Lookup table decoder
    LookupTable,
    /// Union-Find decoder
    UnionFind,
}

/// Correction operation to apply
#[derive(Debug, Clone)]
pub struct CorrectionOperation {
    /// Pauli corrections to apply
    pub pauli_corrections: Vec<PauliCorrection>,
    /// Confidence level of correction
    pub confidence: f64,
    /// Estimated success probability
    pub success_probability: f64,
    /// Required resources
    pub required_resources: CorrectionResources,
}

/// Individual Pauli correction
#[derive(Debug, Clone)]
pub struct PauliCorrection {
    /// Qubit index
    pub qubit: usize,
    /// Pauli operation (I, X, Y, Z)
    pub operation: PauliType,
    /// Correction weight
    pub weight: f64,
}

/// Pauli operation types
#[derive(Debug, Clone, PartialEq)]
pub enum PauliType {
    I, // Identity
    X, // Pauli-X
    Y, // Pauli-Y
    Z, // Pauli-Z
}

/// Resources required for correction
#[derive(Debug, Clone)]
pub struct CorrectionResources {
    /// Number of syndrome measurements
    pub num_measurements: usize,
    /// Number of correction gates
    pub num_correction_gates: usize,
    /// Estimated time overhead (microseconds)
    pub time_overhead: f64,
    /// Ancilla qubits required
    pub ancilla_qubits: usize,
}

/// Detection and correction statistics
#[derive(Debug, Clone)]
pub struct DetectionStatistics {
    /// Total number of syndrome measurements
    pub total_measurements: usize,
    /// Number of non-trivial syndromes detected
    pub errors_detected: usize,
    /// Number of successful corrections
    pub successful_corrections: usize,
    /// Number of failed corrections
    pub failed_corrections: usize,
    /// Average syndrome detection time
    pub avg_detection_time: Duration,
    /// Average correction time
    pub avg_correction_time: Duration,
    /// Syndrome history
    pub syndrome_history: Vec<SyndromeRecord>,
}

/// Individual syndrome measurement record
#[derive(Debug, Clone)]
pub struct SyndromeRecord {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Measured syndrome
    pub syndrome: Vec<u8>,
    /// Error locations (if decoded)
    pub error_locations: Option<Vec<usize>>,
    /// Correction applied
    pub correction_applied: Option<CorrectionOperation>,
    /// Measurement confidence
    pub confidence: f64,
}

/// Result of syndrome detection and correction
#[derive(Debug, Clone)]
pub struct SyndromeResult {
    /// Detected syndrome
    pub syndrome: Vec<u8>,
    /// Recommended correction
    pub correction: Option<CorrectionOperation>,
    /// Detection confidence
    pub confidence: f64,
    /// Error locations
    pub error_locations: Vec<usize>,
    /// Detection metadata
    pub metadata: SyndromeMetadata,
}

/// Metadata for syndrome detection
#[derive(Debug, Clone)]
pub struct SyndromeMetadata {
    /// Time taken for detection
    pub detection_time: Duration,
    /// Number of measurements used
    pub num_measurements: usize,
    /// Decoder used
    pub decoder_used: DecodingAlgorithm,
    /// Noise level estimate
    pub estimated_noise_level: f64,
}

impl SyndromeDetector {
    /// Create new syndrome detector
    pub fn new(
        code: ErrorCorrectionCode,
        parameters: CodeParameters,
        config: SyndromeDetectorConfig,
    ) -> QECResult<Self> {
        let stabilizer_generators = Self::generate_stabilizers(&code, &parameters)?;
        let parity_check_matrix = Self::generate_parity_check_matrix(&code, &parameters)?;
        let correction_table = Self::build_correction_table(&code, &parameters, &config.decoder)?;

        Ok(Self {
            code,
            parameters,
            stabilizer_generators,
            parity_check_matrix,
            correction_table,
            detection_stats: DetectionStatistics::new(),
            config,
        })
    }

    /// Detect syndrome in quantum state
    pub fn detect_syndrome(&mut self, state: &QuantumState) -> QECResult<SyndromeResult> {
        let start_time = Instant::now();

        // Perform syndrome measurements
        let syndrome = self.measure_syndrome(state)?;
        
        // Decode errors if syndrome is non-trivial
        let (correction, error_locations) = if self.is_trivial_syndrome(&syndrome) {
            (None, Vec::new())
        } else {
            let correction = self.decode_syndrome(&syndrome)?;
            let error_locations = self.extract_error_locations(&correction);
            (Some(correction), error_locations)
        };

        // Calculate confidence based on syndrome stability and noise model
        let confidence = self.calculate_detection_confidence(&syndrome, &error_locations)?;

        // Update statistics
        self.update_detection_stats(&syndrome, &correction, start_time.elapsed());

        let metadata = SyndromeMetadata {
            detection_time: start_time.elapsed(),
            num_measurements: self.stabilizer_generators.nrows(),
            decoder_used: self.config.decoder.clone(),
            estimated_noise_level: self.estimate_noise_level(&syndrome),
        };

        Ok(SyndromeResult {
            syndrome,
            correction,
            confidence,
            error_locations,
            metadata,
        })
    }

    /// Apply correction to quantum state
    pub fn apply_correction(
        &mut self,
        state: &mut QuantumState,
        correction: &CorrectionOperation,
    ) -> QECResult<()> {
        let start_time = Instant::now();

        for pauli_correction in &correction.pauli_corrections {
            self.apply_pauli_correction(state, pauli_correction)?;
        }

        // Update correction statistics
        self.detection_stats.successful_corrections += 1;
        self.detection_stats.avg_correction_time = 
            Self::update_average_time(
                self.detection_stats.avg_correction_time,
                start_time.elapsed(),
                self.detection_stats.successful_corrections,
            );

        Ok(())
    }

    /// Perform full detection and correction cycle
    pub fn detect_and_correct(&mut self, state: &mut QuantumState) -> QECResult<SyndromeResult> {
        let syndrome_result = self.detect_syndrome(state)?;

        if let Some(ref correction) = syndrome_result.correction {
            if syndrome_result.confidence >= self.config.correction_threshold {
                self.apply_correction(state, correction)?;
            }
        }

        Ok(syndrome_result)
    }

    /// Generate stabilizer generators for the error correction code
    fn generate_stabilizers(
        code: &ErrorCorrectionCode,
        parameters: &CodeParameters,
    ) -> QECResult<Array2<u8>> {
        match code {
            ErrorCorrectionCode::SurfaceCode => {
                Self::generate_surface_code_stabilizers(parameters)
            }
            ErrorCorrectionCode::RepetitionCode => {
                Self::generate_repetition_code_stabilizers(parameters)
            }
            ErrorCorrectionCode::SteaneCode => {
                Self::generate_steane_code_stabilizers(parameters)
            }
            ErrorCorrectionCode::ShorCode => {
                Self::generate_shor_code_stabilizers(parameters)
            }
            _ => {
                Err(QuantumErrorCorrectionError::CodeError(
                    format!("Stabilizer generation not implemented for code: {:?}", code)
                ))
            }
        }
    }

    /// Generate surface code stabilizers
    fn generate_surface_code_stabilizers(parameters: &CodeParameters) -> QECResult<Array2<u8>> {
        let d = parameters.distance;
        let num_qubits = parameters.num_physical_qubits;
        let num_stabilizers = num_qubits - parameters.num_logical_qubits;

        let mut stabilizers = Array2::zeros((num_stabilizers, 2 * num_qubits));

        // Generate X-type and Z-type stabilizers for surface code
        let mut stabilizer_idx = 0;

        // X-type stabilizers (plaquette checks)
        for row in 0..(d-1) {
            for col in 0..(d-1) {
                if (row + col) % 2 == 0 { // X-type plaquettes
                    let qubits = Self::get_plaquette_qubits(row, col, d);
                    for &qubit in &qubits {
                        if qubit < num_qubits {
                            stabilizers[[stabilizer_idx, qubit]] = 1; // X part
                        }
                    }
                    stabilizer_idx += 1;
                }
            }
        }

        // Z-type stabilizers (vertex checks)
        for row in 0..d {
            for col in 0..d {
                if (row + col) % 2 == 1 { // Z-type vertices
                    let qubits = Self::get_vertex_qubits(row, col, d);
                    for &qubit in &qubits {
                        if qubit < num_qubits {
                            stabilizers[[stabilizer_idx, num_qubits + qubit]] = 1; // Z part
                        }
                    }
                    stabilizer_idx += 1;
                    if stabilizer_idx >= num_stabilizers { break; }
                }
            }
            if stabilizer_idx >= num_stabilizers { break; }
        }

        Ok(stabilizers)
    }

    /// Generate repetition code stabilizers
    fn generate_repetition_code_stabilizers(parameters: &CodeParameters) -> QECResult<Array2<u8>> {
        let n = parameters.num_physical_qubits;
        let num_stabilizers = n - 1;

        let mut stabilizers = Array2::zeros((num_stabilizers, 2 * n));

        // Generate ZZ stabilizers for repetition code
        for i in 0..num_stabilizers {
            stabilizers[[i, n + i]] = 1;     // Z on qubit i
            stabilizers[[i, n + i + 1]] = 1; // Z on qubit i+1
        }

        Ok(stabilizers)
    }

    /// Generate Steane code stabilizers  
    fn generate_steane_code_stabilizers(parameters: &CodeParameters) -> QECResult<Array2<u8>> {
        let n = 7; // Steane code is always 7 qubits
        let mut stabilizers = Array2::zeros((6, 14)); // 6 stabilizers for [[7,1,3]] code

        // X-type stabilizers
        stabilizers[[0, 0]] = 1; stabilizers[[0, 2]] = 1; stabilizers[[0, 4]] = 1; stabilizers[[0, 6]] = 1;
        stabilizers[[1, 1]] = 1; stabilizers[[1, 2]] = 1; stabilizers[[1, 5]] = 1; stabilizers[[1, 6]] = 1;
        stabilizers[[2, 3]] = 1; stabilizers[[2, 4]] = 1; stabilizers[[2, 5]] = 1; stabilizers[[2, 6]] = 1;

        // Z-type stabilizers
        stabilizers[[3, 7]] = 1; stabilizers[[3, 9]] = 1; stabilizers[[3, 11]] = 1; stabilizers[[3, 13]] = 1;
        stabilizers[[4, 8]] = 1; stabilizers[[4, 9]] = 1; stabilizers[[4, 12]] = 1; stabilizers[[4, 13]] = 1;
        stabilizers[[5, 10]] = 1; stabilizers[[5, 11]] = 1; stabilizers[[5, 12]] = 1; stabilizers[[5, 13]] = 1;

        Ok(stabilizers)
    }

    /// Generate Shor code stabilizers
    fn generate_shor_code_stabilizers(parameters: &CodeParameters) -> QECResult<Array2<u8>> {
        let n = 9; // Shor code is always 9 qubits
        let mut stabilizers = Array2::zeros((8, 18)); // 8 stabilizers for [[9,1,3]] code

        // X-type stabilizers (phase error detection)
        stabilizers[[0, 0]] = 1; stabilizers[[0, 1]] = 1; stabilizers[[0, 2]] = 1;
        stabilizers[[1, 3]] = 1; stabilizers[[1, 4]] = 1; stabilizers[[1, 5]] = 1;
        stabilizers[[2, 6]] = 1; stabilizers[[2, 7]] = 1; stabilizers[[2, 8]] = 1;

        // Z-type stabilizers (bit error detection)
        stabilizers[[3, 9]] = 1; stabilizers[[3, 10]] = 1;
        stabilizers[[4, 10]] = 1; stabilizers[[4, 11]] = 1;
        stabilizers[[5, 12]] = 1; stabilizers[[5, 13]] = 1;
        stabilizers[[6, 13]] = 1; stabilizers[[6, 14]] = 1;
        stabilizers[[7, 15]] = 1; stabilizers[[7, 16]] = 1;

        Ok(stabilizers)
    }

    /// Generate parity check matrix
    fn generate_parity_check_matrix(
        code: &ErrorCorrectionCode,
        parameters: &CodeParameters,
    ) -> QECResult<Array2<u8>> {
        // For most stabilizer codes, parity check matrix is derived from stabilizers
        let stabilizers = Self::generate_stabilizers(code, parameters)?;
        Ok(stabilizers.clone()) // Simplified - in practice would extract proper parity check
    }

    /// Build correction lookup table
    fn build_correction_table(
        code: &ErrorCorrectionCode,
        parameters: &CodeParameters,
        decoder: &DecodingAlgorithm,
    ) -> QECResult<HashMap<Vec<u8>, CorrectionOperation>> {
        let mut table = HashMap::new();

        match decoder {
            DecodingAlgorithm::LookupTable => {
                Self::build_lookup_table_decoder(code, parameters, &mut table)?;
            }
            DecodingAlgorithm::MinimumWeight => {
                // For minimum weight, we'll compute corrections on-demand
                // Just add trivial syndrome entry
                table.insert(vec![0; parameters.num_physical_qubits - parameters.num_logical_qubits], 
                           CorrectionOperation::identity());
            }
            _ => {
                // Other decoders compute corrections algorithmically
                table.insert(vec![0; parameters.num_physical_qubits - parameters.num_logical_qubits], 
                           CorrectionOperation::identity());
            }
        }

        Ok(table)
    }

    /// Build lookup table for small codes
    fn build_lookup_table_decoder(
        code: &ErrorCorrectionCode,
        parameters: &CodeParameters,
        table: &mut HashMap<Vec<u8>, CorrectionOperation>,
    ) -> QECResult<()> {
        let num_syndrome_bits = parameters.num_physical_qubits - parameters.num_logical_qubits;
        
        // For small codes, enumerate all possible syndromes
        if num_syndrome_bits <= 10 { // Limit to reasonable table size
            for syndrome_int in 0..(1 << num_syndrome_bits) {
                let syndrome = Self::int_to_syndrome(syndrome_int, num_syndrome_bits);
                let correction = Self::compute_minimum_weight_correction(&syndrome, parameters)?;
                table.insert(syndrome, correction);
            }
        }

        Ok(())
    }

    /// Convert integer to syndrome vector
    fn int_to_syndrome(syndrome_int: usize, num_bits: usize) -> Vec<u8> {
        (0..num_bits)
            .map(|i| ((syndrome_int >> i) & 1) as u8)
            .collect()
    }

    /// Compute minimum weight correction for syndrome
    fn compute_minimum_weight_correction(
        syndrome: &[u8],
        parameters: &CodeParameters,
    ) -> QECResult<CorrectionOperation> {
        // Simplified minimum weight decoding
        let mut corrections = Vec::new();
        
        // Find non-zero syndrome bits and create corresponding corrections
        for (i, &bit) in syndrome.iter().enumerate() {
            if bit == 1 {
                corrections.push(PauliCorrection {
                    qubit: i % parameters.num_physical_qubits,
                    operation: if i < parameters.num_physical_qubits {
                        PauliType::X
                    } else {
                        PauliType::Z
                    },
                    weight: 1.0,
                });
            }
        }

        Ok(CorrectionOperation {
            pauli_corrections: corrections,
            confidence: 0.8, // Default confidence
            success_probability: 0.9,
            required_resources: CorrectionResources {
                num_measurements: syndrome.len(),
                num_correction_gates: corrections.len(),
                time_overhead: 10.0 * corrections.len() as f64,
                ancilla_qubits: 0,
            },
        })
    }

    /// Measure syndrome from quantum state
    fn measure_syndrome(&self, state: &QuantumState) -> QECResult<Vec<u8>> {
        let num_stabilizers = self.stabilizer_generators.nrows();
        let mut syndrome = vec![0u8; num_stabilizers];

        // Simulate syndrome measurements
        let mut rng = ChaCha8Rng::from_rng(rand::thread_rng()).unwrap();
        
        for i in 0..num_stabilizers {
            // In a real implementation, this would measure the stabilizer
            // For simulation, we'll generate syndrome based on noise model
            let error_prob = self.config.noise_model.measurement_error_rate;
            syndrome[i] = if rng.gen::<f64>() < error_prob { 1 } else { 0 };
        }

        Ok(syndrome)
    }

    /// Check if syndrome is trivial (all zeros)
    fn is_trivial_syndrome(&self, syndrome: &[u8]) -> bool {
        syndrome.iter().all(|&bit| bit == 0)
    }

    /// Decode syndrome to find correction
    fn decode_syndrome(&self, syndrome: &[u8]) -> QECResult<CorrectionOperation> {
        match self.config.decoder {
            DecodingAlgorithm::LookupTable => {
                if let Some(correction) = self.correction_table.get(syndrome) {
                    Ok(correction.clone())
                } else {
                    Self::compute_minimum_weight_correction(syndrome, &self.parameters)
                }
            }
            DecodingAlgorithm::MinimumWeight => {
                Self::compute_minimum_weight_correction(syndrome, &self.parameters)
            }
            _ => {
                // Fallback to minimum weight for other decoders
                Self::compute_minimum_weight_correction(syndrome, &self.parameters)
            }
        }
    }

    /// Extract error locations from correction
    fn extract_error_locations(&self, correction: &CorrectionOperation) -> Vec<usize> {
        correction.pauli_corrections
            .iter()
            .map(|pc| pc.qubit)
            .collect()
    }

    /// Calculate detection confidence
    fn calculate_detection_confidence(
        &self,
        syndrome: &[u8],
        error_locations: &[usize],
    ) -> QECResult<f64> {
        // Base confidence on syndrome weight and consistency
        let syndrome_weight = syndrome.iter().map(|&b| b as usize).sum::<usize>();
        let max_weight = syndrome.len();
        
        // Lower confidence for higher syndrome weights (more errors)
        let weight_factor = 1.0 - (syndrome_weight as f64 / max_weight as f64);
        
        // Factor in noise model
        let noise_factor = 1.0 - self.config.noise_model.measurement_error_rate;
        
        Ok(weight_factor * noise_factor * 0.9) // Cap at 0.9
    }

    /// Estimate noise level from syndrome
    fn estimate_noise_level(&self, syndrome: &[u8]) -> f64 {
        let syndrome_weight = syndrome.iter().map(|&b| b as f64).sum::<f64>();
        syndrome_weight / syndrome.len() as f64
    }

    /// Apply Pauli correction to state
    fn apply_pauli_correction(
        &self,
        state: &mut QuantumState,
        correction: &PauliCorrection,
    ) -> QECResult<()> {
        // This would apply the actual Pauli operation in a real implementation
        // For now, we'll just log the correction
        println!("Applying {:?} correction to qubit {}", correction.operation, correction.qubit);
        Ok(())
    }

    /// Update detection statistics
    fn update_detection_stats(
        &mut self,
        syndrome: &[u8],
        correction: &Option<CorrectionOperation>,
        detection_time: Duration,
    ) {
        self.detection_stats.total_measurements += 1;
        
        if !self.is_trivial_syndrome(syndrome) {
            self.detection_stats.errors_detected += 1;
        }

        self.detection_stats.avg_detection_time = Self::update_average_time(
            self.detection_stats.avg_detection_time,
            detection_time,
            self.detection_stats.total_measurements,
        );

        // Record syndrome
        let record = SyndromeRecord {
            timestamp: Instant::now(),
            syndrome: syndrome.to_vec(),
            error_locations: correction.as_ref().map(|c| self.extract_error_locations(c)),
            correction_applied: correction.clone(),
            confidence: 0.8, // Would be calculated properly
        };

        self.detection_stats.syndrome_history.push(record);

        // Limit history size
        if self.detection_stats.syndrome_history.len() > self.config.max_history_length {
            self.detection_stats.syndrome_history.remove(0);
        }
    }

    /// Update average time calculation
    fn update_average_time(current_avg: Duration, new_time: Duration, count: usize) -> Duration {
        let current_total = current_avg * (count - 1) as u32;
        let new_total = current_total + new_time;
        new_total / count as u32
    }

    /// Get plaquette qubits for surface code
    fn get_plaquette_qubits(row: usize, col: usize, d: usize) -> Vec<usize> {
        let mut qubits = Vec::new();
        
        // Add qubits around the plaquette
        if row > 0 { qubits.push(row * d + col); }
        if col > 0 { qubits.push(row * d + col - 1); }
        if row < d - 1 { qubits.push((row + 1) * d + col); }
        if col < d - 1 { qubits.push(row * d + col + 1); }

        qubits
    }

    /// Get vertex qubits for surface code
    fn get_vertex_qubits(row: usize, col: usize, d: usize) -> Vec<usize> {
        // Similar to plaquette but for vertex checks
        Self::get_plaquette_qubits(row, col, d)
    }
}

impl CorrectionOperation {
    /// Create identity correction (no operation)
    pub fn identity() -> Self {
        Self {
            pauli_corrections: Vec::new(),
            confidence: 1.0,
            success_probability: 1.0,
            required_resources: CorrectionResources {
                num_measurements: 0,
                num_correction_gates: 0,
                time_overhead: 0.0,
                ancilla_qubits: 0,
            },
        }
    }
}

impl DetectionStatistics {
    /// Create new detection statistics
    pub fn new() -> Self {
        Self {
            total_measurements: 0,
            errors_detected: 0,
            successful_corrections: 0,
            failed_corrections: 0,
            avg_detection_time: Duration::from_nanos(0),
            avg_correction_time: Duration::from_nanos(0),
            syndrome_history: Vec::new(),
        }
    }

    /// Get error detection rate
    pub fn error_detection_rate(&self) -> f64 {
        if self.total_measurements == 0 {
            0.0
        } else {
            self.errors_detected as f64 / self.total_measurements as f64
        }
    }

    /// Get correction success rate
    pub fn correction_success_rate(&self) -> f64 {
        let total_corrections = self.successful_corrections + self.failed_corrections;
        if total_corrections == 0 {
            0.0
        } else {
            self.successful_corrections as f64 / total_corrections as f64
        }
    }
}

impl Default for SyndromeDetectorConfig {
    fn default() -> Self {
        Self {
            measurement_frequency: 1000.0, // 1 kHz
            correction_threshold: 0.7,
            max_history_length: 1000,
            adaptive_timing: true,
            noise_model: NoiseModel::default(),
            decoder: DecodingAlgorithm::MinimumWeight,
        }
    }
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            single_qubit_error_rate: 0.001,
            two_qubit_error_rate: 0.01,
            measurement_error_rate: 0.02,
            coherence_time: 100.0, // microseconds
            thermal_noise_rate: 0.0001,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_error_correction::codes::*;

    #[test]
    fn test_syndrome_detector_creation() {
        let code = ErrorCorrectionCode::RepetitionCode;
        let parameters = CodeParameters {
            distance: 3,
            num_logical_qubits: 1,
            num_physical_qubits: 3,
            num_ancilla_qubits: 2,
            code_rate: 1.0/3.0,
            threshold_probability: 0.1,
        };
        let config = SyndromeDetectorConfig::default();

        let detector = SyndromeDetector::new(code, parameters, config);
        assert!(detector.is_ok());
    }

    #[test]
    fn test_repetition_code_stabilizers() {
        let parameters = CodeParameters {
            distance: 3,
            num_logical_qubits: 1,
            num_physical_qubits: 3,
            num_ancilla_qubits: 2,
            code_rate: 1.0/3.0,
            threshold_probability: 0.1,
        };

        let stabilizers = SyndromeDetector::generate_repetition_code_stabilizers(&parameters).unwrap();
        assert_eq!(stabilizers.nrows(), 2); // n-1 stabilizers
        assert_eq!(stabilizers.ncols(), 6); // 2n columns (X and Z parts)
    }

    #[test]
    fn test_trivial_syndrome_detection() {
        let detector = create_test_detector();
        let syndrome = vec![0, 0, 0];
        assert!(detector.is_trivial_syndrome(&syndrome));

        let non_trivial = vec![1, 0, 1];
        assert!(!detector.is_trivial_syndrome(&non_trivial));
    }

    #[test]
    fn test_minimum_weight_correction() {
        let parameters = CodeParameters {
            distance: 3,
            num_logical_qubits: 1,
            num_physical_qubits: 3,
            num_ancilla_qubits: 2,
            code_rate: 1.0/3.0,
            threshold_probability: 0.1,
        };

        let syndrome = vec![1, 0];
        let correction = SyndromeDetector::compute_minimum_weight_correction(&syndrome, &parameters).unwrap();
        assert_eq!(correction.pauli_corrections.len(), 1);
        assert_eq!(correction.pauli_corrections[0].qubit, 0);
    }

    fn create_test_detector() -> SyndromeDetector {
        let code = ErrorCorrectionCode::RepetitionCode;
        let parameters = CodeParameters {
            distance: 3,
            num_logical_qubits: 1,
            num_physical_qubits: 3,
            num_ancilla_qubits: 2,
            code_rate: 1.0/3.0,
            threshold_probability: 0.1,
        };
        let config = SyndromeDetectorConfig::default();

        SyndromeDetector::new(code, parameters, config).unwrap()
    }
}