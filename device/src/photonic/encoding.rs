//! Quantum Information Encoding for Photonic Systems
//!
//! This module implements various encoding schemes for quantum information in photonic systems,
//! including error correction codes, concatenated codes, and hardware-specific encodings.

use std::collections::HashMap;
use std::f64::consts::PI;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{PhotonicSystemType, PhotonicMode};
use super::continuous_variable::{Complex, GaussianState};
use super::gate_based::{PhotonicQubitEncoding, PhotonicQubitState};
use crate::DeviceResult;

/// Errors for quantum encoding operations
#[derive(Error, Debug)]
pub enum EncodingError {
    #[error("Invalid encoding parameters: {0}")]
    InvalidParameters(String),
    #[error("Encoding not supported: {0}")]
    UnsupportedEncoding(String),
    #[error("Decoding failed: {0}")]
    DecodingFailed(String),
    #[error("Insufficient physical qubits: {0}")]
    InsufficientQubits(String),
    #[error("Code distance too small: {0}")]
    CodeDistanceTooSmall(usize),
}

type EncodingResult<T> = Result<T, EncodingError>;

/// Quantum error correction code types for photonic systems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhotonicQECCode {
    /// Repetition code
    Repetition { distance: usize },
    /// Surface code (adapted for photonic systems)
    Surface { distance: usize },
    /// Color code
    Color { distance: usize },
    /// Concatenated code
    Concatenated { 
        inner_code: Box<PhotonicQECCode>,
        outer_code: Box<PhotonicQECCode>,
        levels: usize,
    },
    /// CV quantum error correction
    ContinuousVariable {
        code_type: CVCodeType,
        parameters: CVCodeParameters,
    },
    /// Bosonic codes
    Bosonic {
        code_type: BosonicCodeType,
        cutoff: usize,
    },
}

/// Continuous variable QEC code types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CVCodeType {
    /// Gottesman-Kitaev-Preskill (GKP) code
    GKP,
    /// Cat codes
    Cat,
    /// Binomial codes
    Binomial,
    /// Coherent state codes
    CoherentState,
}

/// Parameters for CV QEC codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVCodeParameters {
    /// Squeezing parameter
    pub squeezing: f64,
    /// Displacement parameter
    pub displacement: Complex,
    /// Loss tolerance
    pub loss_tolerance: f64,
    /// Code rate
    pub code_rate: f64,
}

/// Bosonic code types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BosonicCodeType {
    /// Fock state codes
    Fock { basis_states: Vec<usize> },
    /// Coherent state codes
    Coherent { alpha_values: Vec<Complex> },
    /// Squeezed state codes
    Squeezed { squeezing_params: Vec<(f64, f64)> },
}

/// Logical qubit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalQubit {
    /// Encoding scheme used
    pub encoding: PhotonicQECCode,
    /// Physical qubits/modes involved
    pub physical_qubits: Vec<usize>,
    /// Logical state parameters
    pub logical_state: LogicalState,
    /// Error syndrome information
    pub syndrome: Option<ErrorSyndrome>,
}

/// Logical quantum state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalState {
    /// Amplitude for logical |0⟩
    pub amplitude_0: f64,
    /// Amplitude for logical |1⟩
    pub amplitude_1: f64,
    /// Relative phase
    pub phase: f64,
    /// State fidelity
    pub fidelity: f64,
}

impl LogicalState {
    /// Create logical |0⟩ state
    pub fn zero() -> Self {
        Self {
            amplitude_0: 1.0,
            amplitude_1: 0.0,
            phase: 0.0,
            fidelity: 1.0,
        }
    }

    /// Create logical |1⟩ state
    pub fn one() -> Self {
        Self {
            amplitude_0: 0.0,
            amplitude_1: 1.0,
            phase: 0.0,
            fidelity: 1.0,
        }
    }

    /// Create logical |+⟩ state
    pub fn plus() -> Self {
        Self {
            amplitude_0: 1.0 / (2.0_f64).sqrt(),
            amplitude_1: 1.0 / (2.0_f64).sqrt(),
            phase: 0.0,
            fidelity: 1.0,
        }
    }

    /// Create arbitrary logical state
    pub fn arbitrary(amplitude_0: f64, amplitude_1: f64, phase: f64) -> EncodingResult<Self> {
        let norm_squared = amplitude_0 * amplitude_0 + amplitude_1 * amplitude_1;
        if (norm_squared - 1.0).abs() > 1e-6 {
            return Err(EncodingError::InvalidParameters(
                "State amplitudes not normalized".to_string()
            ));
        }

        Ok(Self {
            amplitude_0,
            amplitude_1,
            phase,
            fidelity: 1.0,
        })
    }

    /// Get probability of measuring logical |0⟩
    pub fn prob_zero(&self) -> f64 {
        self.amplitude_0 * self.amplitude_0
    }

    /// Get probability of measuring logical |1⟩
    pub fn prob_one(&self) -> f64 {
        self.amplitude_1 * self.amplitude_1
    }
}

/// Error syndrome for quantum error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSyndrome {
    /// Syndrome bits
    pub syndrome_bits: Vec<bool>,
    /// Error type detected
    pub error_type: Option<ErrorType>,
    /// Error location (if identifiable)
    pub error_location: Option<Vec<usize>>,
    /// Correction applied
    pub correction_applied: Option<Correction>,
}

/// Types of quantum errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorType {
    /// Bit flip error (X error)
    BitFlip,
    /// Phase flip error (Z error)
    PhaseFlip,
    /// Combined error (Y error)
    Combined,
    /// Amplitude damping
    AmplitudeDamping,
    /// Phase damping
    PhaseDamping,
    /// Photon loss
    PhotonLoss,
    /// Thermal noise
    ThermalNoise,
}

/// Quantum error correction operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    /// Type of correction
    pub correction_type: ErrorType,
    /// Qubits to apply correction to
    pub target_qubits: Vec<usize>,
    /// Correction parameters
    pub parameters: Vec<f64>,
}

/// Photonic encoding engine
pub struct PhotonicEncoder {
    /// Available encoding schemes
    pub available_codes: Vec<PhotonicQECCode>,
    /// Encoding performance cache
    pub performance_cache: HashMap<String, EncodingPerformance>,
}

/// Performance metrics for encoding schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingPerformance {
    /// Code distance
    pub distance: usize,
    /// Code rate (logical/physical)
    pub rate: f64,
    /// Threshold error rate
    pub threshold: f64,
    /// Resource overhead
    pub overhead: usize,
    /// Encoding fidelity
    pub encoding_fidelity: f64,
    /// Decoding fidelity
    pub decoding_fidelity: f64,
}

impl PhotonicEncoder {
    pub fn new() -> Self {
        let available_codes = vec![
            PhotonicQECCode::Repetition { distance: 3 },
            PhotonicQECCode::Repetition { distance: 5 },
            PhotonicQECCode::Surface { distance: 3 },
            PhotonicQECCode::Surface { distance: 5 },
            PhotonicQECCode::ContinuousVariable {
                code_type: CVCodeType::GKP,
                parameters: CVCodeParameters {
                    squeezing: 10.0,
                    displacement: Complex::new(0.0, 0.0),
                    loss_tolerance: 0.1,
                    code_rate: 1.0,
                },
            },
            PhotonicQECCode::Bosonic {
                code_type: BosonicCodeType::Coherent {
                    alpha_values: vec![
                        Complex::new(2.0, 0.0),
                        Complex::new(-2.0, 0.0),
                    ],
                },
                cutoff: 10,
            },
        ];

        Self {
            available_codes,
            performance_cache: HashMap::new(),
        }
    }

    /// Encode a logical qubit using the specified code
    pub fn encode_logical_qubit(
        &self,
        logical_state: LogicalState,
        code: &PhotonicQECCode,
        physical_qubits: Vec<usize>,
    ) -> EncodingResult<LogicalQubit> {
        // Validate resources
        let required_qubits = self.get_required_physical_qubits(code)?;
        if physical_qubits.len() < required_qubits {
            return Err(EncodingError::InsufficientQubits(
                format!("Need {} qubits, got {}", required_qubits, physical_qubits.len())
            ));
        }

        match code {
            PhotonicQECCode::Repetition { distance } => {
                self.encode_repetition_code(logical_state, *distance, physical_qubits)
            }
            PhotonicQECCode::Surface { distance } => {
                self.encode_surface_code(logical_state, *distance, physical_qubits)
            }
            PhotonicQECCode::ContinuousVariable { code_type, parameters } => {
                self.encode_cv_code(logical_state, code_type, parameters, physical_qubits)
            }
            PhotonicQECCode::Bosonic { code_type, cutoff } => {
                self.encode_bosonic_code(logical_state, code_type, *cutoff, physical_qubits)
            }
            PhotonicQECCode::Concatenated { inner_code, outer_code, levels } => {
                self.encode_concatenated_code(logical_state, inner_code, outer_code, *levels, physical_qubits)
            }
            _ => Err(EncodingError::UnsupportedEncoding(
                format!("Code {:?} not yet implemented", code)
            )),
        }
    }

    /// Encode using repetition code
    fn encode_repetition_code(
        &self,
        logical_state: LogicalState,
        distance: usize,
        physical_qubits: Vec<usize>,
    ) -> EncodingResult<LogicalQubit> {
        if distance % 2 == 0 {
            return Err(EncodingError::CodeDistanceTooSmall(distance));
        }

        // Repetition code: encode logical state into multiple physical qubits
        let encoding_fidelity = 0.99; // High fidelity for repetition codes
        
        let logical_state_encoded = LogicalState {
            fidelity: logical_state.fidelity * encoding_fidelity,
            ..logical_state
        };

        Ok(LogicalQubit {
            encoding: PhotonicQECCode::Repetition { distance },
            physical_qubits: physical_qubits[0..distance].to_vec(),
            logical_state: logical_state_encoded,
            syndrome: None,
        })
    }

    /// Encode using surface code
    fn encode_surface_code(
        &self,
        logical_state: LogicalState,
        distance: usize,
        physical_qubits: Vec<usize>,
    ) -> EncodingResult<LogicalQubit> {
        if distance < 3 {
            return Err(EncodingError::CodeDistanceTooSmall(distance));
        }

        // Surface code requires d² data qubits + (d²-1) syndrome qubits
        let required_qubits = 2 * distance * distance - 1;
        let encoding_fidelity = 0.95; // Lower than repetition due to complexity
        
        let logical_state_encoded = LogicalState {
            fidelity: logical_state.fidelity * encoding_fidelity,
            ..logical_state
        };

        Ok(LogicalQubit {
            encoding: PhotonicQECCode::Surface { distance },
            physical_qubits: physical_qubits[0..required_qubits].to_vec(),
            logical_state: logical_state_encoded,
            syndrome: None,
        })
    }

    /// Encode using continuous variable code
    fn encode_cv_code(
        &self,
        logical_state: LogicalState,
        code_type: &CVCodeType,
        parameters: &CVCodeParameters,
        physical_modes: Vec<usize>,
    ) -> EncodingResult<LogicalQubit> {
        match code_type {
            CVCodeType::GKP => {
                // GKP encoding: map logical qubit to position/momentum eigenstate grid
                let encoding_fidelity = 0.85; // Lower due to finite squeezing
                
                let logical_state_encoded = LogicalState {
                    fidelity: logical_state.fidelity * encoding_fidelity,
                    ..logical_state
                };

                Ok(LogicalQubit {
                    encoding: PhotonicQECCode::ContinuousVariable {
                        code_type: code_type.clone(),
                        parameters: parameters.clone(),
                    },
                    physical_qubits: physical_modes[0..1].to_vec(), // Single mode for GKP
                    logical_state: logical_state_encoded,
                    syndrome: None,
                })
            }
            CVCodeType::Cat => {
                // Cat code encoding: superposition of coherent states
                let encoding_fidelity = 0.90;
                
                let logical_state_encoded = LogicalState {
                    fidelity: logical_state.fidelity * encoding_fidelity,
                    ..logical_state
                };

                Ok(LogicalQubit {
                    encoding: PhotonicQECCode::ContinuousVariable {
                        code_type: code_type.clone(),
                        parameters: parameters.clone(),
                    },
                    physical_qubits: physical_modes[0..1].to_vec(),
                    logical_state: logical_state_encoded,
                    syndrome: None,
                })
            }
            _ => Err(EncodingError::UnsupportedEncoding(
                format!("CV code {:?} not implemented", code_type)
            )),
        }
    }

    /// Encode using bosonic code
    fn encode_bosonic_code(
        &self,
        logical_state: LogicalState,
        code_type: &BosonicCodeType,
        cutoff: usize,
        physical_modes: Vec<usize>,
    ) -> EncodingResult<LogicalQubit> {
        match code_type {
            BosonicCodeType::Coherent { alpha_values } => {
                if alpha_values.len() < 2 {
                    return Err(EncodingError::InvalidParameters(
                        "Coherent state code requires at least 2 alpha values".to_string()
                    ));
                }

                let encoding_fidelity = 0.88;
                
                let logical_state_encoded = LogicalState {
                    fidelity: logical_state.fidelity * encoding_fidelity,
                    ..logical_state
                };

                Ok(LogicalQubit {
                    encoding: PhotonicQECCode::Bosonic {
                        code_type: code_type.clone(),
                        cutoff,
                    },
                    physical_qubits: physical_modes[0..1].to_vec(),
                    logical_state: logical_state_encoded,
                    syndrome: None,
                })
            }
            _ => Err(EncodingError::UnsupportedEncoding(
                format!("Bosonic code {:?} not implemented", code_type)
            )),
        }
    }

    /// Encode using concatenated code
    fn encode_concatenated_code(
        &self,
        logical_state: LogicalState,
        inner_code: &PhotonicQECCode,
        outer_code: &PhotonicQECCode,
        levels: usize,
        physical_qubits: Vec<usize>,
    ) -> EncodingResult<LogicalQubit> {
        if levels == 0 {
            return Err(EncodingError::InvalidParameters(
                "Concatenation levels must be > 0".to_string()
            ));
        }

        // Recursive encoding: first inner code, then outer code
        let inner_required = self.get_required_physical_qubits(inner_code)?;
        let outer_required = self.get_required_physical_qubits(outer_code)?;
        let total_required = inner_required * outer_required;

        if physical_qubits.len() < total_required {
            return Err(EncodingError::InsufficientQubits(
                format!("Concatenated code needs {} qubits", total_required)
            ));
        }

        // Encoding fidelity decreases with concatenation complexity
        let encoding_fidelity = 0.80;
        
        let logical_state_encoded = LogicalState {
            fidelity: logical_state.fidelity * encoding_fidelity,
            ..logical_state
        };

        Ok(LogicalQubit {
            encoding: PhotonicQECCode::Concatenated {
                inner_code: inner_code.clone(),
                outer_code: outer_code.clone(),
                levels,
            },
            physical_qubits: physical_qubits[0..total_required].to_vec(),
            logical_state: logical_state_encoded,
            syndrome: None,
        })
    }

    /// Decode a logical qubit back to logical state
    pub fn decode_logical_qubit(&self, logical_qubit: &LogicalQubit) -> EncodingResult<LogicalState> {
        // Apply error correction based on syndrome
        let mut corrected_state = logical_qubit.logical_state.clone();
        
        if let Some(syndrome) = &logical_qubit.syndrome {
            corrected_state = self.apply_error_correction(corrected_state, syndrome)?;
        }

        // Apply decoding fidelity loss
        let decoding_fidelity = self.get_decoding_fidelity(&logical_qubit.encoding)?;
        corrected_state.fidelity *= decoding_fidelity;

        Ok(corrected_state)
    }

    /// Apply error correction based on syndrome
    fn apply_error_correction(
        &self,
        logical_state: LogicalState,
        syndrome: &ErrorSyndrome,
    ) -> EncodingResult<LogicalState> {
        let mut corrected_state = logical_state;

        if let Some(correction) = &syndrome.correction_applied {
            match correction.correction_type {
                ErrorType::BitFlip => {
                    // Apply bit flip correction
                    std::mem::swap(&mut corrected_state.amplitude_0, &mut corrected_state.amplitude_1);
                }
                ErrorType::PhaseFlip => {
                    // Apply phase flip correction
                    corrected_state.phase += PI;
                }
                ErrorType::PhotonLoss => {
                    // Reduce fidelity due to photon loss
                    corrected_state.fidelity *= 0.9;
                }
                _ => {
                    // General error correction reduces fidelity
                    corrected_state.fidelity *= 0.95;
                }
            }
        }

        Ok(corrected_state)
    }

    /// Get required number of physical qubits for a code
    pub fn get_required_physical_qubits(&self, code: &PhotonicQECCode) -> EncodingResult<usize> {
        match code {
            PhotonicQECCode::Repetition { distance } => Ok(*distance),
            PhotonicQECCode::Surface { distance } => Ok(2 * distance * distance - 1),
            PhotonicQECCode::Color { distance } => Ok(3 * distance * distance / 2),
            PhotonicQECCode::ContinuousVariable { .. } => Ok(1), // Single mode
            PhotonicQECCode::Bosonic { .. } => Ok(1), // Single mode
            PhotonicQECCode::Concatenated { inner_code, outer_code, .. } => {
                let inner_qubits = self.get_required_physical_qubits(inner_code)?;
                let outer_qubits = self.get_required_physical_qubits(outer_code)?;
                Ok(inner_qubits * outer_qubits)
            }
        }
    }

    /// Get decoding fidelity for a code
    fn get_decoding_fidelity(&self, code: &PhotonicQECCode) -> EncodingResult<f64> {
        match code {
            PhotonicQECCode::Repetition { .. } => Ok(0.995),
            PhotonicQECCode::Surface { .. } => Ok(0.99),
            PhotonicQECCode::ContinuousVariable { code_type, .. } => {
                match code_type {
                    CVCodeType::GKP => Ok(0.90),
                    CVCodeType::Cat => Ok(0.92),
                    _ => Ok(0.85),
                }
            }
            PhotonicQECCode::Bosonic { .. } => Ok(0.88),
            PhotonicQECCode::Concatenated { .. } => Ok(0.85),
            _ => Ok(0.80),
        }
    }

    /// Calculate error syndrome for a logical qubit
    pub fn calculate_syndrome(&self, logical_qubit: &LogicalQubit) -> EncodingResult<ErrorSyndrome> {
        match &logical_qubit.encoding {
            PhotonicQECCode::Repetition { distance } => {
                self.calculate_repetition_syndrome(logical_qubit, *distance)
            }
            PhotonicQECCode::Surface { distance } => {
                self.calculate_surface_syndrome(logical_qubit, *distance)
            }
            _ => {
                // Default syndrome calculation
                Ok(ErrorSyndrome {
                    syndrome_bits: vec![false; 4], // Placeholder
                    error_type: None,
                    error_location: None,
                    correction_applied: None,
                })
            }
        }
    }

    /// Calculate syndrome for repetition code
    fn calculate_repetition_syndrome(
        &self,
        logical_qubit: &LogicalQubit,
        distance: usize,
    ) -> EncodingResult<ErrorSyndrome> {
        // Simplified syndrome calculation for repetition code
        let syndrome_bits = vec![false; distance - 1]; // d-1 syndrome bits
        
        // Detect majority disagreement (simplified)
        let error_detected = logical_qubit.logical_state.fidelity < 0.95;
        
        let error_type = if error_detected {
            Some(ErrorType::BitFlip)
        } else {
            None
        };

        Ok(ErrorSyndrome {
            syndrome_bits,
            error_type,
            error_location: None,
            correction_applied: None,
        })
    }

    /// Calculate syndrome for surface code
    fn calculate_surface_syndrome(
        &self,
        logical_qubit: &LogicalQubit,
        distance: usize,
    ) -> EncodingResult<ErrorSyndrome> {
        // Surface code has (d²-1) syndrome bits
        let num_syndromes = distance * distance - 1;
        let syndrome_bits = vec![false; num_syndromes];
        
        // Simplified error detection
        let error_detected = logical_qubit.logical_state.fidelity < 0.90;
        
        let error_type = if error_detected {
            Some(ErrorType::Combined) // Surface code corrects both X and Z errors
        } else {
            None
        };

        Ok(ErrorSyndrome {
            syndrome_bits,
            error_type,
            error_location: None,
            correction_applied: None,
        })
    }

    /// Get performance metrics for a code
    pub fn get_performance(&self, code: &PhotonicQECCode) -> EncodingResult<EncodingPerformance> {
        let cache_key = format!("{:?}", code);
        
        if let Some(performance) = self.performance_cache.get(&cache_key) {
            return Ok(performance.clone());
        }

        // Calculate performance metrics
        let performance = match code {
            PhotonicQECCode::Repetition { distance } => EncodingPerformance {
                distance: *distance,
                rate: 1.0 / (*distance as f64),
                threshold: 0.5, // 50% for repetition code
                overhead: *distance,
                encoding_fidelity: 0.99,
                decoding_fidelity: 0.995,
            },
            PhotonicQECCode::Surface { distance } => EncodingPerformance {
                distance: *distance,
                rate: 1.0 / ((2 * distance * distance - 1) as f64),
                threshold: 0.11, // ~11% for surface code
                overhead: 2 * distance * distance - 1,
                encoding_fidelity: 0.95,
                decoding_fidelity: 0.99,
            },
            _ => EncodingPerformance {
                distance: 3,
                rate: 0.5,
                threshold: 0.1,
                overhead: 10,
                encoding_fidelity: 0.90,
                decoding_fidelity: 0.85,
            },
        };

        Ok(performance)
    }
}

impl Default for PhotonicEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logical_state_creation() {
        let zero = LogicalState::zero();
        assert_eq!(zero.prob_zero(), 1.0);
        assert_eq!(zero.prob_one(), 0.0);

        let one = LogicalState::one();
        assert_eq!(one.prob_zero(), 0.0);
        assert_eq!(one.prob_one(), 1.0);

        let plus = LogicalState::plus();
        assert!((plus.prob_zero() - 0.5).abs() < 1e-10);
        assert!((plus.prob_one() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = PhotonicEncoder::new();
        assert!(!encoder.available_codes.is_empty());
    }

    #[test]
    fn test_repetition_code_encoding() {
        let encoder = PhotonicEncoder::new();
        let logical_state = LogicalState::zero();
        let code = PhotonicQECCode::Repetition { distance: 3 };
        let physical_qubits = vec![0, 1, 2];

        let encoded = encoder.encode_logical_qubit(logical_state, &code, physical_qubits).unwrap();
        assert_eq!(encoded.physical_qubits.len(), 3);
        assert!(matches!(encoded.encoding, PhotonicQECCode::Repetition { distance: 3 }));
    }

    #[test]
    fn test_surface_code_encoding() {
        let encoder = PhotonicEncoder::new();
        let logical_state = LogicalState::plus();
        let code = PhotonicQECCode::Surface { distance: 3 };
        let physical_qubits: Vec<usize> = (0..17).collect(); // 2*3²-1 = 17 qubits

        let encoded = encoder.encode_logical_qubit(logical_state, &code, physical_qubits).unwrap();
        assert_eq!(encoded.physical_qubits.len(), 17);
    }

    #[test]
    fn test_cv_code_encoding() {
        let encoder = PhotonicEncoder::new();
        let logical_state = LogicalState::zero();
        let code = PhotonicQECCode::ContinuousVariable {
            code_type: CVCodeType::GKP,
            parameters: CVCodeParameters {
                squeezing: 10.0,
                displacement: Complex::new(0.0, 0.0),
                loss_tolerance: 0.1,
                code_rate: 1.0,
            },
        };
        let physical_modes = vec![0];

        let encoded = encoder.encode_logical_qubit(logical_state, &code, physical_modes).unwrap();
        assert_eq!(encoded.physical_qubits.len(), 1);
    }

    #[test]
    fn test_performance_calculation() {
        let encoder = PhotonicEncoder::new();
        let code = PhotonicQECCode::Repetition { distance: 5 };
        
        let performance = encoder.get_performance(&code).unwrap();
        assert_eq!(performance.distance, 5);
        assert_eq!(performance.rate, 0.2); // 1/5
        assert_eq!(performance.overhead, 5);
    }

    #[test]
    fn test_syndrome_calculation() {
        let encoder = PhotonicEncoder::new();
        let logical_state = LogicalState::zero();
        let code = PhotonicQECCode::Repetition { distance: 3 };
        let encoded = encoder.encode_logical_qubit(
            logical_state, 
            &code, 
            vec![0, 1, 2]
        ).unwrap();
        
        let syndrome = encoder.calculate_syndrome(&encoded).unwrap();
        assert_eq!(syndrome.syndrome_bits.len(), 2); // d-1 = 2
    }

    #[test]
    fn test_decoding() {
        let encoder = PhotonicEncoder::new();
        let original_state = LogicalState::plus();
        let code = PhotonicQECCode::Repetition { distance: 3 };
        let encoded = encoder.encode_logical_qubit(
            original_state.clone(), 
            &code, 
            vec![0, 1, 2]
        ).unwrap();
        
        let decoded = encoder.decode_logical_qubit(&encoded).unwrap();
        assert!((decoded.prob_zero() - original_state.prob_zero()).abs() < 0.1);
        assert!((decoded.prob_one() - original_state.prob_one()).abs() < 0.1);
    }
}