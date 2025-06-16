//! Error correction for continuous variable quantum systems
//!
//! This module implements error correction schemes specifically designed for
//! CV quantum systems, including GKP codes and other continuous variable codes.

use super::{CVDeviceConfig, Complex, GaussianState};
use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Types of CV error correction codes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CVErrorCorrectionCode {
    /// Gottesman-Kitaev-Preskill (GKP) codes
    GKP {
        /// Spacing parameter (Δ)
        spacing: f64,
        /// Number of logical qubits
        logical_qubits: usize,
    },
    /// Coherent state codes
    CoherentState {
        /// Alphabet size
        alphabet_size: usize,
        /// Coherent state amplitudes
        amplitudes: Vec<Complex>,
    },
    /// Squeeze-stabilizer codes
    SqueezeStabilizer {
        /// Stabilizer generators
        stabilizers: Vec<CVStabilizer>,
    },
    /// Concatenated CV codes
    Concatenated {
        /// Inner code
        inner_code: Box<CVErrorCorrectionCode>,
        /// Outer code
        outer_code: Box<CVErrorCorrectionCode>,
    },
}

/// CV stabilizer for squeeze-stabilizer codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVStabilizer {
    /// Quadrature operators (coefficient, mode, quadrature_type)
    pub operators: Vec<(f64, usize, QuadratureType)>,
    /// Eigenvalue
    pub eigenvalue: f64,
}

/// Types of quadrature operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuadratureType {
    /// Position quadrature (x)
    Position,
    /// Momentum quadrature (p)
    Momentum,
}

/// Configuration for CV error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVErrorCorrectionConfig {
    /// Error correction code type
    pub code_type: CVErrorCorrectionCode,
    /// Error model parameters
    pub error_model: CVErrorModel,
    /// Syndrome detection threshold
    pub syndrome_threshold: f64,
    /// Maximum correction attempts
    pub max_correction_attempts: usize,
    /// Enable real-time correction
    pub real_time_correction: bool,
    /// Decoder configuration
    pub decoder_config: CVDecoderConfig,
}

/// CV error model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVErrorModel {
    /// Displacement error standard deviation
    pub displacement_std: f64,
    /// Phase error standard deviation
    pub phase_std: f64,
    /// Loss probability
    pub loss_probability: f64,
    /// Thermal photon number
    pub thermal_photons: f64,
    /// Detector efficiency
    pub detector_efficiency: f64,
}

/// Decoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVDecoderConfig {
    /// Decoder type
    pub decoder_type: CVDecoderType,
    /// Maximum likelihood threshold
    pub ml_threshold: f64,
    /// Lookup table size (for discrete decoders)
    pub lookup_table_size: usize,
    /// Enable machine learning enhancement
    pub enable_ml_enhancement: bool,
}

/// Types of CV decoders
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CVDecoderType {
    /// Maximum likelihood decoder
    MaximumLikelihood,
    /// Minimum distance decoder
    MinimumDistance,
    /// Neural network decoder
    NeuralNetwork,
    /// Lookup table decoder
    LookupTable,
}

impl Default for CVErrorCorrectionConfig {
    fn default() -> Self {
        Self {
            code_type: CVErrorCorrectionCode::GKP {
                spacing: (PI).sqrt(),
                logical_qubits: 1,
            },
            error_model: CVErrorModel::default(),
            syndrome_threshold: 0.1,
            max_correction_attempts: 3,
            real_time_correction: true,
            decoder_config: CVDecoderConfig::default(),
        }
    }
}

impl Default for CVErrorModel {
    fn default() -> Self {
        Self {
            displacement_std: 0.1,
            phase_std: 0.05,
            loss_probability: 0.01,
            thermal_photons: 0.1,
            detector_efficiency: 0.95,
        }
    }
}

impl Default for CVDecoderConfig {
    fn default() -> Self {
        Self {
            decoder_type: CVDecoderType::MaximumLikelihood,
            ml_threshold: 0.8,
            lookup_table_size: 10000,
            enable_ml_enhancement: false,
        }
    }
}

/// CV error correction system
pub struct CVErrorCorrector {
    /// Configuration
    config: CVErrorCorrectionConfig,
    /// Current logical state
    logical_state: Option<CVLogicalState>,
    /// Syndrome measurement history
    syndrome_history: Vec<CVSyndrome>,
    /// Correction statistics
    correction_stats: CorrectionStatistics,
}

/// CV logical state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVLogicalState {
    /// Physical modes representing the logical state
    pub physical_modes: GaussianState,
    /// Logical information
    pub logical_info: Vec<LogicalQubitInfo>,
    /// Code parameters
    pub code_parameters: CodeParameters,
}

/// Information about a logical qubit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalQubitInfo {
    /// Logical qubit ID
    pub qubit_id: usize,
    /// Physical modes involved
    pub physical_modes: Vec<usize>,
    /// Current logical operators
    pub logical_operators: LogicalOperators,
}

/// Logical operators for CV codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalOperators {
    /// Logical X operator
    pub logical_x: CVOperator,
    /// Logical Z operator
    pub logical_z: CVOperator,
    /// Logical Y operator (derived)
    pub logical_y: CVOperator,
}

/// CV operator representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVOperator {
    /// Displacement components
    pub displacements: Vec<Complex>,
    /// Squeezing operations
    pub squeezings: Vec<(f64, f64)>, // (parameter, phase)
    /// Mode coupling operations
    pub couplings: Vec<ModeCoupling>,
}

/// Mode coupling operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeCoupling {
    /// Modes involved
    pub modes: (usize, usize),
    /// Coupling strength
    pub strength: f64,
    /// Coupling type
    pub coupling_type: CouplingType,
}

/// Types of mode coupling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CouplingType {
    /// Beamsplitter coupling
    Beamsplitter,
    /// Two-mode squeezing
    TwoModeSqueezing,
    /// Cross-Kerr interaction
    CrossKerr,
}

/// Code parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeParameters {
    /// Code distance
    pub distance: usize,
    /// Number of physical modes
    pub num_physical_modes: usize,
    /// Number of logical qubits
    pub num_logical_qubits: usize,
    /// Error threshold
    pub error_threshold: f64,
}

/// CV syndrome measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVSyndrome {
    /// Syndrome ID
    pub syndrome_id: usize,
    /// Measurement results
    pub measurements: Vec<SyndromeMeasurement>,
    /// Timestamp
    pub timestamp: f64,
    /// Confidence level
    pub confidence: f64,
}

/// Individual syndrome measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndromeMeasurement {
    /// Stabilizer ID
    pub stabilizer_id: usize,
    /// Measurement outcome
    pub outcome: f64,
    /// Expected value
    pub expected_value: f64,
    /// Measurement uncertainty
    pub uncertainty: f64,
}

/// Correction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionStatistics {
    /// Total syndrome measurements
    pub total_syndromes: usize,
    /// Successful corrections
    pub successful_corrections: usize,
    /// Failed corrections
    pub failed_corrections: usize,
    /// Average correction fidelity
    pub average_fidelity: f64,
    /// Logical error rate
    pub logical_error_rate: f64,
}

impl Default for CorrectionStatistics {
    fn default() -> Self {
        Self {
            total_syndromes: 0,
            successful_corrections: 0,
            failed_corrections: 0,
            average_fidelity: 0.0,
            logical_error_rate: 0.0,
        }
    }
}

impl CVErrorCorrector {
    /// Create a new CV error corrector
    pub fn new(config: CVErrorCorrectionConfig) -> Self {
        Self {
            config,
            logical_state: None,
            syndrome_history: Vec::new(),
            correction_stats: CorrectionStatistics::default(),
        }
    }

    /// Initialize logical state
    pub async fn initialize_logical_state(
        &mut self,
        initial_state: GaussianState,
    ) -> DeviceResult<CVLogicalState> {
        println!("Initializing CV logical state...");

        let logical_state = match &self.config.code_type {
            CVErrorCorrectionCode::GKP {
                spacing,
                logical_qubits,
            } => {
                self.initialize_gkp_state(initial_state, *spacing, *logical_qubits)
                    .await?
            }
            CVErrorCorrectionCode::CoherentState {
                alphabet_size,
                amplitudes,
            } => {
                self.initialize_coherent_state_code(initial_state, *alphabet_size, amplitudes)
                    .await?
            }
            _ => {
                return Err(DeviceError::UnsupportedOperation(
                    "Code type not yet implemented".to_string(),
                ));
            }
        };

        self.logical_state = Some(logical_state.clone());
        println!("Logical state initialized successfully");
        Ok(logical_state)
    }

    /// Initialize GKP logical state
    async fn initialize_gkp_state(
        &self,
        mut physical_state: GaussianState,
        spacing: f64,
        num_logical_qubits: usize,
    ) -> DeviceResult<CVLogicalState> {
        // GKP codes encode logical qubits in the infinite-dimensional Hilbert space
        // of a harmonic oscillator using a discrete lattice in phase space

        let num_physical_modes = physical_state.num_modes;

        // Apply GKP state preparation operations
        for mode in 0..num_physical_modes.min(num_logical_qubits) {
            // Apply periodic squeezing to create GKP-like state
            for i in 0..10 {
                let phase = 2.0 * PI * i as f64 / 10.0;
                let squeezing_param = 0.5 * (spacing / PI.sqrt()).ln();
                physical_state.apply_squeezing(mode, squeezing_param, phase)?;
            }
        }

        // Build logical operators for GKP codes
        let mut logical_info = Vec::new();
        for qubit_id in 0..num_logical_qubits {
            let logical_operators = self.build_gkp_logical_operators(qubit_id, spacing);
            logical_info.push(LogicalQubitInfo {
                qubit_id,
                physical_modes: vec![qubit_id], // One mode per logical qubit for single-mode GKP
                logical_operators,
            });
        }

        let code_parameters = CodeParameters {
            distance: 1, // Single-mode GKP has distance 1
            num_physical_modes,
            num_logical_qubits,
            error_threshold: 0.5 * spacing,
        };

        Ok(CVLogicalState {
            physical_modes: physical_state,
            logical_info,
            code_parameters,
        })
    }

    /// Build GKP logical operators
    fn build_gkp_logical_operators(&self, qubit_id: usize, spacing: f64) -> LogicalOperators {
        // GKP logical X: displacement by spacing in position
        let logical_x = CVOperator {
            displacements: vec![Complex::new(spacing, 0.0)],
            squeezings: Vec::new(),
            couplings: Vec::new(),
        };

        // GKP logical Z: displacement by spacing in momentum
        let logical_z = CVOperator {
            displacements: vec![Complex::new(0.0, spacing)],
            squeezings: Vec::new(),
            couplings: Vec::new(),
        };

        // GKP logical Y: combination of X and Z
        let logical_y = CVOperator {
            displacements: vec![Complex::new(
                spacing / (2.0_f64).sqrt(),
                spacing / (2.0_f64).sqrt(),
            )],
            squeezings: Vec::new(),
            couplings: Vec::new(),
        };

        LogicalOperators {
            logical_x,
            logical_z,
            logical_y,
        }
    }

    /// Initialize coherent state code
    async fn initialize_coherent_state_code(
        &self,
        physical_state: GaussianState,
        alphabet_size: usize,
        amplitudes: &[Complex],
    ) -> DeviceResult<CVLogicalState> {
        if amplitudes.len() != alphabet_size {
            return Err(DeviceError::InvalidInput(
                "Number of amplitudes must match alphabet size".to_string(),
            ));
        }

        // For coherent state codes, we prepare superpositions of coherent states
        let num_physical_modes = physical_state.num_modes;
        let num_logical_qubits = 1; // Simplified: one logical qubit per alphabet

        let logical_info = vec![LogicalQubitInfo {
            qubit_id: 0,
            physical_modes: (0..num_physical_modes).collect(),
            logical_operators: self.build_coherent_state_logical_operators(amplitudes),
        }];

        let code_parameters = CodeParameters {
            distance: alphabet_size / 2, // Approximate distance
            num_physical_modes,
            num_logical_qubits,
            error_threshold: amplitudes.iter().map(|a| a.magnitude()).sum::<f64>()
                / alphabet_size as f64
                * 0.5,
        };

        Ok(CVLogicalState {
            physical_modes: physical_state,
            logical_info,
            code_parameters,
        })
    }

    /// Build coherent state logical operators
    fn build_coherent_state_logical_operators(&self, amplitudes: &[Complex]) -> LogicalOperators {
        // Simplified logical operators for coherent state codes
        let avg_amplitude = amplitudes.iter().fold(Complex::zero(), |acc, &a| acc + a)
            * (1.0 / amplitudes.len() as f64);

        LogicalOperators {
            logical_x: CVOperator {
                displacements: vec![avg_amplitude],
                squeezings: Vec::new(),
                couplings: Vec::new(),
            },
            logical_z: CVOperator {
                displacements: vec![Complex::new(0.0, avg_amplitude.magnitude())],
                squeezings: Vec::new(),
                couplings: Vec::new(),
            },
            logical_y: CVOperator {
                displacements: vec![Complex::new(avg_amplitude.real, avg_amplitude.magnitude())],
                squeezings: Vec::new(),
                couplings: Vec::new(),
            },
        }
    }

    /// Perform syndrome measurement
    pub async fn measure_syndrome(&mut self) -> DeviceResult<CVSyndrome> {
        if self.logical_state.is_none() {
            return Err(DeviceError::InvalidInput(
                "No logical state initialized".to_string(),
            ));
        }

        let syndrome_id = self.syndrome_history.len();
        let mut measurements = Vec::new();

        // Measure stabilizers based on code type
        match &self.config.code_type {
            CVErrorCorrectionCode::GKP { spacing, .. } => {
                measurements = self.measure_gkp_stabilizers(*spacing).await?;
            }
            CVErrorCorrectionCode::CoherentState { amplitudes, .. } => {
                measurements = self.measure_coherent_state_stabilizers(amplitudes).await?;
            }
            _ => {
                return Err(DeviceError::UnsupportedOperation(
                    "Syndrome measurement not implemented for this code type".to_string(),
                ));
            }
        }

        // Calculate confidence based on measurement uncertainties
        let confidence = measurements
            .iter()
            .map(|m| 1.0 / (1.0 + m.uncertainty))
            .sum::<f64>()
            / measurements.len() as f64;

        let syndrome = CVSyndrome {
            syndrome_id,
            measurements,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            confidence,
        };

        self.syndrome_history.push(syndrome.clone());
        self.correction_stats.total_syndromes += 1;

        Ok(syndrome)
    }

    /// Measure GKP stabilizers
    async fn measure_gkp_stabilizers(
        &self,
        spacing: f64,
    ) -> DeviceResult<Vec<SyndromeMeasurement>> {
        let logical_state = self.logical_state.as_ref().unwrap();
        let mut measurements = Vec::new();

        // GKP stabilizers are periodic functions in phase space
        for mode in 0..logical_state.physical_modes.num_modes {
            // Measure x-stabilizer: exp(2πi x/Δ)
            let x_measurement = self
                .measure_periodic_stabilizer(mode, QuadratureType::Position, spacing)
                .await?;
            measurements.push(x_measurement);

            // Measure p-stabilizer: exp(2πi p/Δ)
            let p_measurement = self
                .measure_periodic_stabilizer(mode, QuadratureType::Momentum, spacing)
                .await?;
            measurements.push(p_measurement);
        }

        Ok(measurements)
    }

    /// Measure periodic stabilizer
    async fn measure_periodic_stabilizer(
        &self,
        mode: usize,
        quadrature_type: QuadratureType,
        spacing: f64,
    ) -> DeviceResult<SyndromeMeasurement> {
        let logical_state = self.logical_state.as_ref().unwrap();
        let config = CVDeviceConfig::default();

        let phase = match quadrature_type {
            QuadratureType::Position => 0.0,
            QuadratureType::Momentum => PI / 2.0,
        };

        // Perform homodyne measurement
        let mut temp_state = logical_state.physical_modes.clone();
        let outcome = temp_state.homodyne_measurement(mode, phase, &config)?;

        // Calculate syndrome value (mod spacing)
        let syndrome_value = (outcome % spacing) / spacing;
        let expected_value = 0.0; // For ideal codeword
        let uncertainty = self.config.error_model.displacement_std;

        Ok(SyndromeMeasurement {
            stabilizer_id: mode * 2
                + if quadrature_type == QuadratureType::Position {
                    0
                } else {
                    1
                },
            outcome: syndrome_value,
            expected_value,
            uncertainty,
        })
    }

    /// Measure coherent state stabilizers
    async fn measure_coherent_state_stabilizers(
        &self,
        _amplitudes: &[Complex],
    ) -> DeviceResult<Vec<SyndromeMeasurement>> {
        // Simplified implementation for coherent state codes
        let logical_state = self.logical_state.as_ref().unwrap();
        let mut measurements = Vec::new();

        for mode in 0..logical_state.physical_modes.num_modes {
            let config = CVDeviceConfig::default();
            let mut temp_state = logical_state.physical_modes.clone();

            let outcome = temp_state.heterodyne_measurement(mode, &config)?;

            measurements.push(SyndromeMeasurement {
                stabilizer_id: mode,
                outcome: outcome.magnitude(),
                expected_value: 1.0, // Expected amplitude
                uncertainty: self.config.error_model.displacement_std,
            });
        }

        Ok(measurements)
    }

    /// Apply error correction based on syndrome
    pub async fn apply_correction(
        &mut self,
        syndrome: &CVSyndrome,
    ) -> DeviceResult<CorrectionResult> {
        if self.logical_state.is_none() {
            return Err(DeviceError::InvalidInput(
                "No logical state to correct".to_string(),
            ));
        }

        println!(
            "Applying error correction for syndrome {}",
            syndrome.syndrome_id
        );

        // Decode syndrome to determine correction
        let correction_operations = self.decode_syndrome(syndrome).await?;

        // Apply corrections to logical state
        let mut correction_success = true;
        let mut applied_operations = 0;

        for operation in &correction_operations {
            match self.apply_correction_operation(operation).await {
                Ok(_) => applied_operations += 1,
                Err(_) => {
                    correction_success = false;
                    break;
                }
            }
        }

        // Calculate correction fidelity
        let fidelity = if correction_success {
            0.95 - syndrome
                .measurements
                .iter()
                .map(|m| (m.outcome - m.expected_value).abs())
                .sum::<f64>()
                * 0.1
        } else {
            0.5
        };

        // Update statistics
        if correction_success {
            self.correction_stats.successful_corrections += 1;
        } else {
            self.correction_stats.failed_corrections += 1;
        }

        let total_corrections =
            self.correction_stats.successful_corrections + self.correction_stats.failed_corrections;
        self.correction_stats.average_fidelity =
            (self.correction_stats.average_fidelity * (total_corrections - 1) as f64 + fidelity)
                / total_corrections as f64;

        Ok(CorrectionResult {
            syndrome_id: syndrome.syndrome_id,
            correction_operations,
            success: correction_success,
            fidelity,
            applied_operations,
        })
    }

    /// Decode syndrome to determine correction operations
    async fn decode_syndrome(
        &self,
        syndrome: &CVSyndrome,
    ) -> DeviceResult<Vec<CorrectionOperation>> {
        match self.config.decoder_config.decoder_type {
            CVDecoderType::MaximumLikelihood => self.ml_decode(syndrome).await,
            CVDecoderType::MinimumDistance => self.minimum_distance_decode(syndrome).await,
            _ => Err(DeviceError::UnsupportedOperation(
                "Decoder type not implemented".to_string(),
            )),
        }
    }

    /// Maximum likelihood decoder
    async fn ml_decode(&self, syndrome: &CVSyndrome) -> DeviceResult<Vec<CorrectionOperation>> {
        let mut corrections = Vec::new();

        for measurement in &syndrome.measurements {
            let deviation = (measurement.outcome - measurement.expected_value).abs();

            if deviation > self.config.syndrome_threshold {
                // Determine correction based on measurement type and deviation
                let mode = measurement.stabilizer_id / 2;
                let is_position = measurement.stabilizer_id % 2 == 0;

                let correction_amplitude = if is_position {
                    Complex::new(-measurement.outcome, 0.0)
                } else {
                    Complex::new(0.0, -measurement.outcome)
                };

                corrections.push(CorrectionOperation {
                    operation_type: CorrectionOperationType::Displacement {
                        mode,
                        amplitude: correction_amplitude,
                    },
                    confidence: measurement.uncertainty,
                });
            }
        }

        Ok(corrections)
    }

    /// Minimum distance decoder
    async fn minimum_distance_decode(
        &self,
        syndrome: &CVSyndrome,
    ) -> DeviceResult<Vec<CorrectionOperation>> {
        // Simplified minimum distance decoder
        let mut corrections = Vec::new();

        // Find the syndrome with minimum Euclidean distance
        let mut min_distance = f64::INFINITY;
        let mut best_correction = None;

        for measurement in &syndrome.measurements {
            let distance = (measurement.outcome - measurement.expected_value).abs();

            if distance < min_distance && distance > self.config.syndrome_threshold {
                min_distance = distance;

                let mode = measurement.stabilizer_id / 2;
                let is_position = measurement.stabilizer_id % 2 == 0;

                let correction_amplitude = if is_position {
                    Complex::new(-measurement.outcome * 0.5, 0.0)
                } else {
                    Complex::new(0.0, -measurement.outcome * 0.5)
                };

                best_correction = Some(CorrectionOperation {
                    operation_type: CorrectionOperationType::Displacement {
                        mode,
                        amplitude: correction_amplitude,
                    },
                    confidence: 1.0 / (1.0 + distance),
                });
            }
        }

        if let Some(correction) = best_correction {
            corrections.push(correction);
        }

        Ok(corrections)
    }

    /// Apply a single correction operation
    async fn apply_correction_operation(
        &mut self,
        operation: &CorrectionOperation,
    ) -> DeviceResult<()> {
        if let Some(logical_state) = &mut self.logical_state {
            match &operation.operation_type {
                CorrectionOperationType::Displacement { mode, amplitude } => {
                    logical_state
                        .physical_modes
                        .apply_displacement(*mode, *amplitude)?;
                }
                CorrectionOperationType::Squeezing {
                    mode,
                    parameter,
                    phase,
                } => {
                    logical_state
                        .physical_modes
                        .apply_squeezing(*mode, *parameter, *phase)?;
                }
                CorrectionOperationType::PhaseRotation { mode, phase } => {
                    logical_state
                        .physical_modes
                        .apply_phase_rotation(*mode, *phase)?;
                }
            }
        }
        Ok(())
    }

    /// Get correction statistics
    pub fn get_correction_statistics(&self) -> &CorrectionStatistics {
        &self.correction_stats
    }

    /// Get current logical state
    pub fn get_logical_state(&self) -> Option<&CVLogicalState> {
        self.logical_state.as_ref()
    }

    /// Get syndrome history
    pub fn get_syndrome_history(&self) -> &[CVSyndrome] {
        &self.syndrome_history
    }
}

/// Error correction operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionOperation {
    /// Type of operation
    pub operation_type: CorrectionOperationType,
    /// Confidence in this correction
    pub confidence: f64,
}

/// Types of correction operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionOperationType {
    /// Displacement correction
    Displacement { mode: usize, amplitude: Complex },
    /// Squeezing correction
    Squeezing {
        mode: usize,
        parameter: f64,
        phase: f64,
    },
    /// Phase rotation correction
    PhaseRotation { mode: usize, phase: f64 },
}

/// Result of error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionResult {
    /// Syndrome ID that was corrected
    pub syndrome_id: usize,
    /// Operations applied
    pub correction_operations: Vec<CorrectionOperation>,
    /// Whether correction was successful
    pub success: bool,
    /// Correction fidelity
    pub fidelity: f64,
    /// Number of operations actually applied
    pub applied_operations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cv_error_corrector_creation() {
        let config = CVErrorCorrectionConfig::default();
        let corrector = CVErrorCorrector::new(config);
        assert!(corrector.logical_state.is_none());
        assert_eq!(corrector.syndrome_history.len(), 0);
    }

    #[tokio::test]
    async fn test_gkp_state_initialization() {
        let config = CVErrorCorrectionConfig::default();
        let mut corrector = CVErrorCorrector::new(config);

        let initial_state = GaussianState::vacuum_state(2);
        let logical_state = corrector
            .initialize_logical_state(initial_state)
            .await
            .unwrap();

        assert_eq!(logical_state.physical_modes.num_modes, 2);
        assert_eq!(logical_state.logical_info.len(), 1);
    }

    #[tokio::test]
    async fn test_syndrome_measurement() {
        let config = CVErrorCorrectionConfig::default();
        let mut corrector = CVErrorCorrector::new(config);

        let initial_state = GaussianState::vacuum_state(1);
        corrector
            .initialize_logical_state(initial_state)
            .await
            .unwrap();

        let syndrome = corrector.measure_syndrome().await.unwrap();
        assert_eq!(syndrome.syndrome_id, 0);
        assert!(!syndrome.measurements.is_empty());
        assert_eq!(corrector.syndrome_history.len(), 1);
    }

    #[tokio::test]
    async fn test_error_correction() {
        let config = CVErrorCorrectionConfig::default();
        let mut corrector = CVErrorCorrector::new(config);

        let initial_state = GaussianState::vacuum_state(1);
        corrector
            .initialize_logical_state(initial_state)
            .await
            .unwrap();

        let syndrome = corrector.measure_syndrome().await.unwrap();
        let result = corrector.apply_correction(&syndrome).await.unwrap();

        assert_eq!(result.syndrome_id, syndrome.syndrome_id);
        assert!(result.fidelity >= 0.0 && result.fidelity <= 1.0);
    }

    #[test]
    fn test_gkp_logical_operators() {
        let config = CVErrorCorrectionConfig::default();
        let corrector = CVErrorCorrector::new(config);

        let operators = corrector.build_gkp_logical_operators(0, PI.sqrt());

        // Check that logical operators have correct structure
        assert_eq!(operators.logical_x.displacements.len(), 1);
        assert_eq!(operators.logical_z.displacements.len(), 1);
        assert_eq!(operators.logical_y.displacements.len(), 1);
    }

    #[test]
    fn test_error_model_defaults() {
        let error_model = CVErrorModel::default();
        assert!(error_model.displacement_std > 0.0);
        assert!(error_model.phase_std > 0.0);
        assert!(error_model.loss_probability >= 0.0 && error_model.loss_probability <= 1.0);
        assert!(error_model.detector_efficiency >= 0.0 && error_model.detector_efficiency <= 1.0);
    }

    #[test]
    fn test_correction_statistics() {
        let corrector = CVErrorCorrector::new(CVErrorCorrectionConfig::default());
        let stats = corrector.get_correction_statistics();

        assert_eq!(stats.total_syndromes, 0);
        assert_eq!(stats.successful_corrections, 0);
        assert_eq!(stats.failed_corrections, 0);
    }
}
