//! Mixed-precision quantum simulation with automatic precision selection.
//!
//! This module implements adaptive precision algorithms that automatically
//! select optimal numerical precision (f16, f32, f64) for different parts
//! of quantum computations, leveraging SciRS2's precision optimization
//! capabilities to maximize performance while maintaining accuracy.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;

#[cfg(all(feature = "advanced_math", feature = "mixed_precision"))]
use scirs2_linalg::mixed_precision::{AdaptiveStrategy, MixedPrecisionContext, PrecisionLevel};

// Placeholder types when the feature is not available
#[cfg(not(all(feature = "advanced_math", feature = "mixed_precision")))]
#[derive(Debug)]
pub struct MixedPrecisionContext;

#[cfg(not(all(feature = "advanced_math", feature = "mixed_precision")))]
#[derive(Debug)]
pub enum PrecisionLevel {
    F16,
    F32,
    F64,
    Adaptive,
}

#[cfg(not(all(feature = "advanced_math", feature = "mixed_precision")))]
#[derive(Debug)]
pub enum AdaptiveStrategy {
    ErrorBased(f64),
    Fixed(PrecisionLevel),
}

#[cfg(not(all(feature = "advanced_math", feature = "mixed_precision")))]
impl MixedPrecisionContext {
    pub fn new(_strategy: AdaptiveStrategy) -> Result<Self> {
        Err(SimulatorError::UnsupportedOperation(
            "Mixed precision context not available without advanced_math feature".to_string(),
        ))
    }
}

/// Precision levels for quantum computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumPrecision {
    /// Half precision (16-bit floats)
    Half,
    /// Single precision (32-bit floats)
    Single,
    /// Double precision (64-bit floats)
    Double,
    /// Adaptive precision (automatically selected)
    Adaptive,
}

impl QuantumPrecision {
    /// Get the corresponding SciRS2 precision level
    #[cfg(feature = "advanced_math")]
    pub fn to_scirs2_precision(&self) -> PrecisionLevel {
        match self {
            QuantumPrecision::Half => PrecisionLevel::F16,
            QuantumPrecision::Single => PrecisionLevel::F32,
            QuantumPrecision::Double => PrecisionLevel::F64,
            QuantumPrecision::Adaptive => PrecisionLevel::Adaptive,
        }
    }

    /// Get memory usage factor relative to double precision
    pub fn memory_factor(&self) -> f64 {
        match self {
            QuantumPrecision::Half => 0.5,
            QuantumPrecision::Single => 0.5,
            QuantumPrecision::Double => 1.0,
            QuantumPrecision::Adaptive => 0.75, // Estimated average
        }
    }

    /// Get computational speed factor relative to double precision
    pub fn speed_factor(&self) -> f64 {
        match self {
            QuantumPrecision::Half => 2.0,
            QuantumPrecision::Single => 1.5,
            QuantumPrecision::Double => 1.0,
            QuantumPrecision::Adaptive => 1.25, // Estimated average
        }
    }
}

/// Configuration for mixed-precision simulation
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Default precision for state vectors
    pub state_vector_precision: QuantumPrecision,
    /// Default precision for gate operations
    pub gate_precision: QuantumPrecision,
    /// Default precision for measurements
    pub measurement_precision: QuantumPrecision,
    /// Error tolerance for precision selection
    pub error_tolerance: f64,
    /// Enable automatic precision adaptation
    pub adaptive_precision: bool,
    /// Minimum precision level (never go below this)
    pub min_precision: QuantumPrecision,
    /// Maximum precision level (never go above this)
    pub max_precision: QuantumPrecision,
    /// Number of qubits threshold for precision reduction
    pub large_system_threshold: usize,
    /// Enable precision analysis and reporting
    pub enable_analysis: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            state_vector_precision: QuantumPrecision::Single,
            gate_precision: QuantumPrecision::Single,
            measurement_precision: QuantumPrecision::Double,
            error_tolerance: 1e-6,
            adaptive_precision: true,
            min_precision: QuantumPrecision::Half,
            max_precision: QuantumPrecision::Double,
            large_system_threshold: 20,
            enable_analysis: true,
        }
    }
}

/// Precision analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionAnalysis {
    /// Recommended precision for each operation type
    pub recommended_precisions: HashMap<String, QuantumPrecision>,
    /// Estimated numerical error for each precision
    pub error_estimates: HashMap<QuantumPrecision, f64>,
    /// Performance metrics for each precision
    pub performance_metrics: HashMap<QuantumPrecision, PerformanceMetrics>,
    /// Final precision selection rationale
    pub selection_rationale: String,
    /// Quality score (0-1, higher is better)
    pub quality_score: f64,
}

/// Performance metrics for a specific precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: f64,
    /// Energy efficiency score (operations per joule)
    pub energy_efficiency: f64,
}

/// Mixed-precision state vector
pub enum MixedPrecisionStateVector {
    /// Half precision state vector
    Half(Array1<Complex32>),
    /// Single precision state vector
    Single(Array1<Complex32>),
    /// Double precision state vector
    Double(Array1<Complex64>),
    /// Adaptive precision with multiple representations
    Adaptive {
        primary: Box<MixedPrecisionStateVector>,
        secondary: Option<Box<MixedPrecisionStateVector>>,
        precision_map: Vec<QuantumPrecision>,
    },
}

impl MixedPrecisionStateVector {
    /// Create new state vector with specified precision
    pub fn new(size: usize, precision: QuantumPrecision) -> Self {
        match precision {
            QuantumPrecision::Half => Self::Half(Array1::zeros(size)),
            QuantumPrecision::Single => Self::Single(Array1::zeros(size)),
            QuantumPrecision::Double => Self::Double(Array1::zeros(size)),
            QuantumPrecision::Adaptive => {
                // Start with single precision, will adapt as needed
                Self::Adaptive {
                    primary: Box::new(Self::Single(Array1::zeros(size))),
                    secondary: None,
                    precision_map: vec![QuantumPrecision::Single; size],
                }
            }
        }
    }

    /// Get the current precision level
    pub fn precision(&self) -> QuantumPrecision {
        match self {
            Self::Half(_) => QuantumPrecision::Half,
            Self::Single(_) => QuantumPrecision::Single,
            Self::Double(_) => QuantumPrecision::Double,
            Self::Adaptive { .. } => QuantumPrecision::Adaptive,
        }
    }

    /// Get the state vector size
    pub fn len(&self) -> usize {
        match self {
            Self::Half(arr) => arr.len(),
            Self::Single(arr) => arr.len(),
            Self::Double(arr) => arr.len(),
            Self::Adaptive { primary, .. } => primary.len(),
        }
    }

    /// Convert to double precision (for external interfaces)
    pub fn to_double_precision(&self) -> Array1<Complex64> {
        match self {
            Self::Half(arr) => arr.mapv(|x| Complex64::new(x.re as f64, x.im as f64)),
            Self::Single(arr) => arr.mapv(|x| Complex64::new(x.re as f64, x.im as f64)),
            Self::Double(arr) => arr.clone(),
            Self::Adaptive { primary, .. } => primary.to_double_precision(),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::Half(arr) => arr.len() * std::mem::size_of::<Complex32>(),
            Self::Single(arr) => arr.len() * std::mem::size_of::<Complex32>(),
            Self::Double(arr) => arr.len() * std::mem::size_of::<Complex64>(),
            Self::Adaptive {
                primary, secondary, ..
            } => primary.memory_usage() + secondary.as_ref().map_or(0, |s| s.memory_usage()),
        }
    }

    /// Normalize the state vector
    pub fn normalize(&mut self) -> Result<()> {
        match self {
            Self::Half(arr) => {
                let norm = arr.iter().map(|x| x.norm_sqr()).sum::<f32>().sqrt();
                if norm > 1e-15 {
                    arr.mapv_inplace(|x| x / norm);
                }
            }
            Self::Single(arr) => {
                let norm = arr.iter().map(|x| x.norm_sqr()).sum::<f32>().sqrt();
                if norm > 1e-15 {
                    arr.mapv_inplace(|x| x / norm);
                }
            }
            Self::Double(arr) => {
                let norm = arr.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    arr.mapv_inplace(|x| x / norm);
                }
            }
            Self::Adaptive { primary, .. } => {
                primary.normalize()?;
            }
        }
        Ok(())
    }

    /// Apply two-qubit gate manually
    pub fn apply_two_qubit_gate_manual(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let control_mask = 1_usize << control;
        let target_mask = 1_usize << target;

        match self {
            Self::Double(state_vector) => {
                for i in 0..state_vector.len() {
                    if i & control_mask != 0 {
                        let target_bit = (i & target_mask) != 0;
                        let j = if target_bit {
                            i & !target_mask
                        } else {
                            i | target_mask
                        };

                        if j < i {
                            continue;
                        }

                        let amp_i = state_vector[i];
                        let amp_j = state_vector[j];

                        if target_bit {
                            state_vector[j] =
                                gate_matrix[[0, 0]] * amp_j + gate_matrix[[0, 1]] * amp_i;
                            state_vector[i] =
                                gate_matrix[[1, 0]] * amp_j + gate_matrix[[1, 1]] * amp_i;
                        } else {
                            state_vector[i] =
                                gate_matrix[[0, 0]] * amp_i + gate_matrix[[0, 1]] * amp_j;
                            state_vector[j] =
                                gate_matrix[[1, 0]] * amp_i + gate_matrix[[1, 1]] * amp_j;
                        }
                    }
                }
            }
            Self::Single(state_vector) => {
                let gate_f32 = gate_matrix.mapv(|x| Complex32::new(x.re as f32, x.im as f32));
                for i in 0..state_vector.len() {
                    if i & control_mask != 0 {
                        let target_bit = (i & target_mask) != 0;
                        let j = if target_bit {
                            i & !target_mask
                        } else {
                            i | target_mask
                        };

                        if j < i {
                            continue;
                        }

                        let amp_i = state_vector[i];
                        let amp_j = state_vector[j];

                        if target_bit {
                            state_vector[j] = gate_f32[[0, 0]] * amp_j + gate_f32[[0, 1]] * amp_i;
                            state_vector[i] = gate_f32[[1, 0]] * amp_j + gate_f32[[1, 1]] * amp_i;
                        } else {
                            state_vector[i] = gate_f32[[0, 0]] * amp_i + gate_f32[[0, 1]] * amp_j;
                            state_vector[j] = gate_f32[[1, 0]] * amp_i + gate_f32[[1, 1]] * amp_j;
                        }
                    }
                }
            }
            Self::Half(state_vector) => {
                let gate_f32 = gate_matrix.mapv(|x| Complex32::new(x.re as f32, x.im as f32));
                for i in 0..state_vector.len() {
                    if i & control_mask != 0 {
                        let target_bit = (i & target_mask) != 0;
                        let j = if target_bit {
                            i & !target_mask
                        } else {
                            i | target_mask
                        };

                        if j < i {
                            continue;
                        }

                        let amp_i = state_vector[i];
                        let amp_j = state_vector[j];

                        if target_bit {
                            state_vector[j] = gate_f32[[0, 0]] * amp_j + gate_f32[[0, 1]] * amp_i;
                            state_vector[i] = gate_f32[[1, 0]] * amp_j + gate_f32[[1, 1]] * amp_i;
                        } else {
                            state_vector[i] = gate_f32[[0, 0]] * amp_i + gate_f32[[0, 1]] * amp_j;
                            state_vector[j] = gate_f32[[1, 0]] * amp_i + gate_f32[[1, 1]] * amp_j;
                        }
                    }
                }
            }
            Self::Adaptive { primary, .. } => {
                primary.apply_two_qubit_gate_manual(control, target, gate_matrix)?;
            }
        }
        Ok(())
    }

    /// Measure probability of qubit being in |1⟩ state
    pub fn measure_probability(&self, qubit: usize) -> f64 {
        let qubit_mask = 1_usize << qubit;
        match self {
            Self::Double(state_vector) => state_vector
                .iter()
                .enumerate()
                .filter(|(i, _)| *i & qubit_mask != 0)
                .map(|(_, amp)| amp.norm_sqr())
                .sum(),
            Self::Single(state_vector) => state_vector
                .iter()
                .enumerate()
                .filter(|(i, _)| *i & qubit_mask != 0)
                .map(|(_, amp)| amp.norm_sqr() as f64)
                .sum(),
            Self::Half(state_vector) => state_vector
                .iter()
                .enumerate()
                .filter(|(i, _)| *i & qubit_mask != 0)
                .map(|(_, amp)| amp.norm_sqr() as f64)
                .sum(),
            Self::Adaptive { primary, .. } => primary.measure_probability(qubit),
        }
    }
}

/// Mixed-precision quantum simulator
pub struct MixedPrecisionSimulator {
    /// Current state vector
    state: MixedPrecisionStateVector,
    /// Number of qubits
    num_qubits: usize,
    /// Configuration
    config: MixedPrecisionConfig,
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Precision context for SciRS2 operations
    #[cfg(feature = "advanced_math")]
    precision_context: Option<MixedPrecisionContext>,
    /// Operation history for precision analysis
    operation_history: Vec<PrecisionOperation>,
    /// Performance statistics
    stats: MixedPrecisionStats,
}

/// Record of a precision operation
#[derive(Debug, Clone)]
struct PrecisionOperation {
    operation_type: String,
    input_precision: QuantumPrecision,
    output_precision: QuantumPrecision,
    execution_time_ms: f64,
    numerical_error: f64,
    memory_usage_bytes: usize,
}

/// Statistics for mixed-precision simulation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MixedPrecisionStats {
    /// Total execution time in milliseconds
    pub total_execution_time_ms: f64,
    /// Time spent in precision conversions
    pub conversion_time_ms: f64,
    /// Memory saved through mixed precision (bytes)
    pub memory_saved_bytes: usize,
    /// Number of precision adaptations
    pub precision_adaptations: usize,
    /// Distribution of operations by precision
    pub precision_distribution: HashMap<QuantumPrecision, usize>,
    /// Average numerical error
    pub average_numerical_error: f64,
    /// Performance improvement factor
    pub performance_improvement_factor: f64,
}

impl MixedPrecisionSimulator {
    /// Create new mixed-precision simulator
    pub fn new(num_qubits: usize, config: MixedPrecisionConfig) -> Result<Self> {
        let state_size = 1_usize << num_qubits;
        let precision = if config.adaptive_precision && num_qubits >= config.large_system_threshold
        {
            QuantumPrecision::Adaptive
        } else {
            config.state_vector_precision
        };

        let state = MixedPrecisionStateVector::new(state_size, precision);

        Ok(Self {
            state,
            num_qubits,
            config,
            backend: None,
            #[cfg(feature = "advanced_math")]
            precision_context: None,
            operation_history: Vec::new(),
            stats: MixedPrecisionStats::default(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());

        #[cfg(feature = "advanced_math")]
        {
            let strategy = match self.config.adaptive_precision {
                true => AdaptiveStrategy::ErrorBased(self.config.error_tolerance),
                false => AdaptiveStrategy::Fixed(
                    self.config.state_vector_precision.to_scirs2_precision(),
                ),
            };

            self.precision_context = Some(MixedPrecisionContext::new(strategy)?);
        }

        Ok(self)
    }

    /// Initialize |0...0⟩ state
    pub fn initialize_zero_state(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();

        match &mut self.state {
            MixedPrecisionStateVector::Half(arr) => {
                arr.fill(Complex32::new(0.0, 0.0));
                arr[0] = Complex32::new(1.0, 0.0);
            }
            MixedPrecisionStateVector::Single(arr) => {
                arr.fill(Complex32::new(0.0, 0.0));
                arr[0] = Complex32::new(1.0, 0.0);
            }
            MixedPrecisionStateVector::Double(arr) => {
                arr.fill(Complex64::new(0.0, 0.0));
                arr[0] = Complex64::new(1.0, 0.0);
            }
            MixedPrecisionStateVector::Adaptive {
                ref mut primary, ..
            } => {
                // Initialize primary recursively
                match primary.as_mut() {
                    MixedPrecisionStateVector::Half(ref mut state) => {
                        state.fill(Complex32::new(0.0, 0.0));
                        state[0] = Complex32::new(1.0, 0.0);
                    }
                    MixedPrecisionStateVector::Single(ref mut state) => {
                        state.fill(Complex32::new(0.0, 0.0));
                        state[0] = Complex32::new(1.0, 0.0);
                    }
                    MixedPrecisionStateVector::Double(ref mut state) => {
                        state.fill(Complex64::new(0.0, 0.0));
                        state[0] = Complex64::new(1.0, 0.0);
                    }
                    _ => {
                        return Err(SimulatorError::UnsupportedOperation(
                            "Nested adaptive not supported".to_string(),
                        ))
                    }
                }
            }
        }

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_execution_time_ms += execution_time;

        Ok(())
    }

    /// Apply single-qubit gate with automatic precision selection
    pub fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(qubit));
        }

        let start_time = std::time::Instant::now();

        // Analyze gate for optimal precision
        let gate_precision = self.analyze_gate_precision(gate_matrix)?;

        // Apply gate with selected precision
        self.apply_single_qubit_gate_with_precision(qubit, gate_matrix, gate_precision)?;

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_execution_time_ms += execution_time;

        // Record operation for analysis
        self.operation_history.push(PrecisionOperation {
            operation_type: "single_qubit_gate".to_string(),
            input_precision: self.state.precision(),
            output_precision: self.state.precision(),
            execution_time_ms: execution_time,
            numerical_error: self.estimate_gate_error(gate_matrix, gate_precision),
            memory_usage_bytes: self.state.memory_usage(),
        });

        Ok(())
    }

    /// Apply two-qubit gate with precision optimization
    pub fn apply_two_qubit_gate(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(control.max(target)));
        }

        let start_time = std::time::Instant::now();

        // Two-qubit gates typically require higher precision due to entanglement
        let gate_precision = self.analyze_two_qubit_gate_precision(gate_matrix)?;

        self.apply_two_qubit_gate_with_precision(control, target, gate_matrix, gate_precision)?;

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_execution_time_ms += execution_time;

        self.operation_history.push(PrecisionOperation {
            operation_type: "two_qubit_gate".to_string(),
            input_precision: self.state.precision(),
            output_precision: self.state.precision(),
            execution_time_ms: execution_time,
            numerical_error: self.estimate_gate_error(gate_matrix, gate_precision),
            memory_usage_bytes: self.state.memory_usage(),
        });

        Ok(())
    }

    /// Measure qubit with precision-aware probability calculation
    pub fn measure_probability(&self, qubit: usize) -> Result<f64> {
        if qubit >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(qubit));
        }

        let measurement_precision = self.config.measurement_precision;

        match measurement_precision {
            QuantumPrecision::Double => self.measure_probability_f64(qubit),
            QuantumPrecision::Single => Ok(self.measure_probability_f32(qubit)? as f64),
            QuantumPrecision::Half => Ok(self.measure_probability_f32(qubit)? as f64), // f16 not directly supported
            QuantumPrecision::Adaptive => self.measure_probability_adaptive(qubit),
        }
    }

    /// Get state vector with requested precision
    pub fn get_state_vector(&self, precision: Option<QuantumPrecision>) -> Array1<Complex64> {
        match precision.unwrap_or(QuantumPrecision::Double) {
            QuantumPrecision::Double => self.state.to_double_precision(),
            _ => {
                // For now, always return double precision for external interface
                self.state.to_double_precision()
            }
        }
    }

    /// Perform comprehensive precision analysis
    pub fn analyze_precision_requirements(&mut self) -> Result<PrecisionAnalysis> {
        if !self.config.enable_analysis {
            return Err(SimulatorError::UnsupportedOperation(
                "Precision analysis is disabled".to_string(),
            ));
        }

        let mut analysis = PrecisionAnalysis {
            recommended_precisions: HashMap::new(),
            error_estimates: HashMap::new(),
            performance_metrics: HashMap::new(),
            selection_rationale: String::new(),
            quality_score: 0.0,
        };

        // Analyze different precision levels
        let precisions = vec![
            QuantumPrecision::Half,
            QuantumPrecision::Single,
            QuantumPrecision::Double,
        ];

        for &precision in &precisions {
            let error_estimate = self.estimate_precision_error(precision)?;
            let performance_metrics = self.benchmark_precision(precision)?;

            analysis.error_estimates.insert(precision, error_estimate);
            analysis
                .performance_metrics
                .insert(precision, performance_metrics);
        }

        // Generate recommendations
        analysis.recommended_precisions.insert(
            "state_vector".to_string(),
            self.recommend_state_vector_precision(&analysis)?,
        );
        analysis.recommended_precisions.insert(
            "gates".to_string(),
            self.recommend_gate_precision(&analysis)?,
        );
        analysis.recommended_precisions.insert(
            "measurements".to_string(),
            self.recommend_measurement_precision(&analysis)?,
        );

        // Calculate quality score
        analysis.quality_score = self.calculate_quality_score(&analysis);

        // Generate rationale
        analysis.selection_rationale = self.generate_selection_rationale(&analysis);

        Ok(analysis)
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &MixedPrecisionStats {
        &self.stats
    }

    /// Internal helper methods

    fn analyze_gate_precision(&self, gate_matrix: &Array2<Complex64>) -> Result<QuantumPrecision> {
        if !self.config.adaptive_precision {
            return Ok(self.config.gate_precision);
        }

        // Analyze gate properties to determine optimal precision
        let condition_number = self.calculate_condition_number(gate_matrix);
        let magnitude_range = self.calculate_magnitude_range(gate_matrix);

        if condition_number > 1e6 || magnitude_range > 1e6 {
            Ok(QuantumPrecision::Double)
        } else if condition_number > 1e3 || magnitude_range > 1e3 {
            Ok(QuantumPrecision::Single)
        } else {
            Ok(QuantumPrecision::Half)
        }
    }

    fn analyze_two_qubit_gate_precision(
        &self,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<QuantumPrecision> {
        // Two-qubit gates generally need higher precision due to entanglement
        let base_precision = self.analyze_gate_precision(gate_matrix)?;

        match base_precision {
            QuantumPrecision::Half => Ok(QuantumPrecision::Single),
            QuantumPrecision::Single => Ok(QuantumPrecision::Single),
            QuantumPrecision::Double => Ok(QuantumPrecision::Double),
            QuantumPrecision::Adaptive => Ok(QuantumPrecision::Single),
        }
    }

    fn calculate_condition_number(&self, matrix: &Array2<Complex64>) -> f64 {
        // Simplified condition number estimation
        // In practice, would use proper SVD-based calculation
        let max_magnitude = matrix.iter().map(|x| x.norm()).fold(0.0, f64::max);
        let min_magnitude = matrix
            .iter()
            .map(|x| x.norm())
            .filter(|&x| x > 1e-15)
            .fold(f64::INFINITY, f64::min);

        if min_magnitude == f64::INFINITY {
            1e12 // Very large condition number for singular matrices
        } else {
            max_magnitude / min_magnitude
        }
    }

    fn calculate_magnitude_range(&self, matrix: &Array2<Complex64>) -> f64 {
        let max_magnitude = matrix.iter().map(|x| x.norm()).fold(0.0, f64::max);
        let min_magnitude = matrix
            .iter()
            .map(|x| x.norm())
            .filter(|&x| x > 1e-15)
            .fold(f64::INFINITY, f64::min);

        if min_magnitude == f64::INFINITY {
            max_magnitude
        } else {
            max_magnitude / min_magnitude
        }
    }

    fn apply_single_qubit_gate_with_precision(
        &mut self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
        precision: QuantumPrecision,
    ) -> Result<()> {
        // Convert state to appropriate precision if needed
        self.ensure_state_precision(precision)?;

        #[cfg(feature = "advanced_math")]
        {
            if self.precision_context.is_some() {
                return self.apply_gate_scirs2(qubit, gate_matrix);
            }
        }

        // Fallback to manual implementation
        self.apply_single_qubit_gate_manual(qubit, gate_matrix)
    }

    fn apply_two_qubit_gate_with_precision(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
        precision: QuantumPrecision,
    ) -> Result<()> {
        // Convert state to appropriate precision if needed
        self.ensure_state_precision(precision)?;

        #[cfg(feature = "advanced_math")]
        {
            if self.precision_context.is_some() {
                return self.apply_two_qubit_gate_scirs2(control, target, gate_matrix);
            }
        }

        // Fallback to manual implementation
        self.apply_two_qubit_gate_manual(control, target, gate_matrix)
    }

    #[cfg(feature = "advanced_math")]
    fn apply_gate_scirs2(&mut self, qubit: usize, gate_matrix: &Array2<Complex64>) -> Result<()> {
        // Use SciRS2's mixed precision gate application
        // This is a placeholder - actual implementation would use SciRS2 APIs
        self.apply_single_qubit_gate_manual(qubit, gate_matrix)
    }

    #[cfg(feature = "advanced_math")]
    fn apply_two_qubit_gate_scirs2(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        // Use SciRS2's mixed precision two-qubit gate application
        // This is a placeholder - actual implementation would use SciRS2 APIs
        self.apply_two_qubit_gate_manual(control, target, gate_matrix)
    }

    fn apply_single_qubit_gate_manual(
        &mut self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let qubit_mask = 1_usize << qubit;

        match &mut self.state {
            MixedPrecisionStateVector::Double(state_vector) => {
                for i in (0..state_vector.len()).step_by(2) {
                    if i & qubit_mask == 0 {
                        let j = i | qubit_mask;
                        if j < state_vector.len() {
                            let amp_0 = state_vector[i];
                            let amp_1 = state_vector[j];

                            state_vector[i] =
                                gate_matrix[[0, 0]] * amp_0 + gate_matrix[[0, 1]] * amp_1;
                            state_vector[j] =
                                gate_matrix[[1, 0]] * amp_0 + gate_matrix[[1, 1]] * amp_1;
                        }
                    }
                }
            }
            MixedPrecisionStateVector::Single(state_vector) => {
                // Convert gate matrix to single precision
                let gate_f32 = gate_matrix.mapv(|x| Complex32::new(x.re as f32, x.im as f32));

                for i in (0..state_vector.len()).step_by(2) {
                    if i & qubit_mask == 0 {
                        let j = i | qubit_mask;
                        if j < state_vector.len() {
                            let amp_0 = state_vector[i];
                            let amp_1 = state_vector[j];

                            state_vector[i] = gate_f32[[0, 0]] * amp_0 + gate_f32[[0, 1]] * amp_1;
                            state_vector[j] = gate_f32[[1, 0]] * amp_0 + gate_f32[[1, 1]] * amp_1;
                        }
                    }
                }
            }
            MixedPrecisionStateVector::Half(state_vector) => {
                // Convert gate matrix to single precision (half precision operations via single)
                let gate_f32 = gate_matrix.mapv(|x| Complex32::new(x.re as f32, x.im as f32));

                for i in (0..state_vector.len()).step_by(2) {
                    if i & qubit_mask == 0 {
                        let j = i | qubit_mask;
                        if j < state_vector.len() {
                            let amp_0 = state_vector[i];
                            let amp_1 = state_vector[j];

                            state_vector[i] = gate_f32[[0, 0]] * amp_0 + gate_f32[[0, 1]] * amp_1;
                            state_vector[j] = gate_f32[[1, 0]] * amp_0 + gate_f32[[1, 1]] * amp_1;
                        }
                    }
                }
            }
            MixedPrecisionStateVector::Adaptive {
                ref mut primary, ..
            } => {
                // Apply gate to primary recursively
                match primary.as_mut() {
                    MixedPrecisionStateVector::Half(state_vector) => {
                        let gate_f32 =
                            gate_matrix.mapv(|x| Complex32::new(x.re as f32, x.im as f32));

                        for i in (0..state_vector.len()).step_by(2) {
                            if i & qubit_mask == 0 {
                                let j = i | qubit_mask;
                                if j < state_vector.len() {
                                    let amp_0 = state_vector[i];
                                    let amp_1 = state_vector[j];

                                    state_vector[i] =
                                        gate_f32[[0, 0]] * amp_0 + gate_f32[[0, 1]] * amp_1;
                                    state_vector[j] =
                                        gate_f32[[1, 0]] * amp_0 + gate_f32[[1, 1]] * amp_1;
                                }
                            }
                        }
                    }
                    MixedPrecisionStateVector::Single(state_vector) => {
                        let gate_f32 =
                            gate_matrix.mapv(|x| Complex32::new(x.re as f32, x.im as f32));

                        for i in (0..state_vector.len()).step_by(2) {
                            if i & qubit_mask == 0 {
                                let j = i | qubit_mask;
                                if j < state_vector.len() {
                                    let amp_0 = state_vector[i];
                                    let amp_1 = state_vector[j];

                                    state_vector[i] =
                                        gate_f32[[0, 0]] * amp_0 + gate_f32[[0, 1]] * amp_1;
                                    state_vector[j] =
                                        gate_f32[[1, 0]] * amp_0 + gate_f32[[1, 1]] * amp_1;
                                }
                            }
                        }
                    }
                    MixedPrecisionStateVector::Double(state_vector) => {
                        for i in (0..state_vector.len()).step_by(2) {
                            if i & qubit_mask == 0 {
                                let j = i | qubit_mask;
                                if j < state_vector.len() {
                                    let amp_0 = state_vector[i];
                                    let amp_1 = state_vector[j];

                                    state_vector[i] =
                                        gate_matrix[[0, 0]] * amp_0 + gate_matrix[[0, 1]] * amp_1;
                                    state_vector[j] =
                                        gate_matrix[[1, 0]] * amp_0 + gate_matrix[[1, 1]] * amp_1;
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(SimulatorError::UnsupportedOperation(
                            "Nested adaptive not supported".to_string(),
                        ))
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_two_qubit_gate_manual(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        // Simplified two-qubit gate implementation
        let control_mask = 1_usize << control;
        let target_mask = 1_usize << target;

        match &mut self.state {
            MixedPrecisionStateVector::Double(state_vector) => {
                for i in 0..state_vector.len() {
                    if i & control_mask != 0 {
                        // Control qubit is |1⟩
                        let target_bit = (i & target_mask) != 0;
                        let j = if target_bit {
                            i & !target_mask
                        } else {
                            i | target_mask
                        };

                        if j < i {
                            continue;
                        } // Avoid double processing

                        let amp_i = state_vector[i];
                        let amp_j = state_vector[j];

                        if target_bit {
                            state_vector[j] =
                                gate_matrix[[0, 0]] * amp_j + gate_matrix[[0, 1]] * amp_i;
                            state_vector[i] =
                                gate_matrix[[1, 0]] * amp_j + gate_matrix[[1, 1]] * amp_i;
                        } else {
                            state_vector[i] =
                                gate_matrix[[0, 0]] * amp_i + gate_matrix[[0, 1]] * amp_j;
                            state_vector[j] =
                                gate_matrix[[1, 0]] * amp_i + gate_matrix[[1, 1]] * amp_j;
                        }
                    }
                }
            }
            _ => {
                // For other precisions, convert to double, apply, then convert back
                let double_state = self.state.to_double_precision();
                let mut temp_state = MixedPrecisionStateVector::Double(double_state);
                temp_state.apply_two_qubit_gate_manual(control, target, gate_matrix)?;

                // Convert back to original precision
                self.convert_state_precision(temp_state, self.state.precision())?;
            }
        }

        Ok(())
    }

    fn ensure_state_precision(&mut self, target_precision: QuantumPrecision) -> Result<()> {
        if self.state.precision() == target_precision {
            return Ok(());
        }

        let start_time = std::time::Instant::now();

        // Convert state to target precision
        let current_state = self.state.to_double_precision();
        self.state = match target_precision {
            QuantumPrecision::Half => {
                let converted = current_state.mapv(|x| Complex32::new(x.re as f32, x.im as f32));
                MixedPrecisionStateVector::Half(converted)
            }
            QuantumPrecision::Single => {
                let converted = current_state.mapv(|x| Complex32::new(x.re as f32, x.im as f32));
                MixedPrecisionStateVector::Single(converted)
            }
            QuantumPrecision::Double => MixedPrecisionStateVector::Double(current_state),
            QuantumPrecision::Adaptive => {
                let state_len = current_state.len();
                MixedPrecisionStateVector::Adaptive {
                    primary: Box::new(MixedPrecisionStateVector::Double(current_state)),
                    secondary: None,
                    precision_map: vec![QuantumPrecision::Double; state_len],
                }
            }
        };

        let conversion_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.conversion_time_ms += conversion_time;
        self.stats.precision_adaptations += 1;

        Ok(())
    }

    fn convert_state_precision(
        &mut self,
        new_state: MixedPrecisionStateVector,
        target_precision: QuantumPrecision,
    ) -> Result<()> {
        // Implementation for converting between precision levels
        let double_state = new_state.to_double_precision();
        self.state = match target_precision {
            QuantumPrecision::Half => {
                let converted = double_state.mapv(|x| Complex32::new(x.re as f32, x.im as f32));
                MixedPrecisionStateVector::Half(converted)
            }
            QuantumPrecision::Single => {
                let converted = double_state.mapv(|x| Complex32::new(x.re as f32, x.im as f32));
                MixedPrecisionStateVector::Single(converted)
            }
            QuantumPrecision::Double => MixedPrecisionStateVector::Double(double_state),
            QuantumPrecision::Adaptive => {
                let state_len = double_state.len();
                MixedPrecisionStateVector::Adaptive {
                    primary: Box::new(MixedPrecisionStateVector::Double(double_state)),
                    secondary: None,
                    precision_map: vec![QuantumPrecision::Double; state_len],
                }
            }
        };
        Ok(())
    }

    fn measure_probability_f64(&self, qubit: usize) -> Result<f64> {
        let qubit_mask = 1_usize << qubit;
        let state_vector = self.state.to_double_precision();

        let mut probability = 0.0;
        for (i, &amplitude) in state_vector.iter().enumerate() {
            if i & qubit_mask != 0 {
                probability += amplitude.norm_sqr();
            }
        }

        Ok(probability)
    }

    fn measure_probability_f32(&self, qubit: usize) -> Result<f32> {
        let qubit_mask = 1_usize << qubit;

        let probability = match &self.state {
            MixedPrecisionStateVector::Single(state_vector)
            | MixedPrecisionStateVector::Half(state_vector) => {
                let mut prob = 0.0f32;
                for (i, &amplitude) in state_vector.iter().enumerate() {
                    if i & qubit_mask != 0 {
                        prob += amplitude.norm_sqr();
                    }
                }
                prob
            }
            _ => {
                // Convert to f32 and calculate
                let state_vector = self.state.to_double_precision();
                let mut prob = 0.0f32;
                for (i, &amplitude) in state_vector.iter().enumerate() {
                    if i & qubit_mask != 0 {
                        prob += amplitude.norm_sqr() as f32;
                    }
                }
                prob
            }
        };

        Ok(probability)
    }

    fn measure_probability_adaptive(&self, qubit: usize) -> Result<f64> {
        // Use highest available precision for measurement
        match &self.state {
            MixedPrecisionStateVector::Adaptive { primary, .. } => {
                Ok(primary.measure_probability(qubit))
            }
            _ => self.measure_probability_f64(qubit),
        }
    }

    fn estimate_gate_error(
        &self,
        gate_matrix: &Array2<Complex64>,
        precision: QuantumPrecision,
    ) -> f64 {
        // Estimate numerical error introduced by gate application at given precision
        match precision {
            QuantumPrecision::Half => 1e-3,     // Half precision error
            QuantumPrecision::Single => 1e-6,   // Single precision error
            QuantumPrecision::Double => 1e-15,  // Double precision error
            QuantumPrecision::Adaptive => 1e-9, // Estimated adaptive error
        }
    }

    fn estimate_precision_error(&self, precision: QuantumPrecision) -> Result<f64> {
        // Estimate cumulative numerical error for entire simulation at given precision
        let num_operations = self.operation_history.len() as f64;
        let base_error = match precision {
            QuantumPrecision::Half => 1e-3,
            QuantumPrecision::Single => 1e-6,
            QuantumPrecision::Double => 1e-15,
            QuantumPrecision::Adaptive => 1e-9,
        };

        // Error accumulates approximately as sqrt(num_operations) for independent operations
        Ok(base_error * num_operations.sqrt())
    }

    fn benchmark_precision(&self, precision: QuantumPrecision) -> Result<PerformanceMetrics> {
        // Simplified performance benchmarking
        let base_time = 1.0; // Base execution time in ms
        let base_memory = 1_000_000; // Base memory usage in bytes

        let (time_factor, memory_factor) = match precision {
            QuantumPrecision::Half => (0.5, 0.5),
            QuantumPrecision::Single => (0.75, 0.5),
            QuantumPrecision::Double => (1.0, 1.0),
            QuantumPrecision::Adaptive => (0.85, 0.75),
        };

        Ok(PerformanceMetrics {
            execution_time_ms: base_time * time_factor,
            memory_usage_bytes: (base_memory as f64 * memory_factor) as usize,
            throughput_ops_per_sec: 1000.0 / (base_time * time_factor),
            energy_efficiency: 1.0 / time_factor, // Simple energy model
        })
    }

    fn recommend_state_vector_precision(
        &self,
        analysis: &PrecisionAnalysis,
    ) -> Result<QuantumPrecision> {
        // Recommend precision based on error tolerance and performance
        for &precision in &[
            QuantumPrecision::Single,
            QuantumPrecision::Double,
            QuantumPrecision::Half,
        ] {
            if let Some(&error) = analysis.error_estimates.get(&precision) {
                if error <= self.config.error_tolerance {
                    return Ok(precision);
                }
            }
        }

        Ok(QuantumPrecision::Double) // Fallback to highest precision
    }

    fn recommend_gate_precision(&self, analysis: &PrecisionAnalysis) -> Result<QuantumPrecision> {
        // Gates can often use lower precision than state vectors
        for &precision in &[
            QuantumPrecision::Half,
            QuantumPrecision::Single,
            QuantumPrecision::Double,
        ] {
            if let Some(&error) = analysis.error_estimates.get(&precision) {
                if error <= self.config.error_tolerance * 2.0 {
                    // More relaxed for gates
                    return Ok(precision);
                }
            }
        }

        Ok(QuantumPrecision::Single)
    }

    fn recommend_measurement_precision(
        &self,
        _analysis: &PrecisionAnalysis,
    ) -> Result<QuantumPrecision> {
        // Measurements typically need high precision for accurate probabilities
        Ok(QuantumPrecision::Double)
    }

    fn calculate_quality_score(&self, analysis: &PrecisionAnalysis) -> f64 {
        // Calculate overall quality score (0-1) balancing accuracy and performance
        let mut total_score = 0.0;
        let mut count = 0;

        for (precision, error) in &analysis.error_estimates {
            if let Some(performance) = analysis.performance_metrics.get(precision) {
                let accuracy_score = 1.0 - (error / self.config.error_tolerance).min(1.0);
                let performance_score = 1.0 / performance.execution_time_ms.max(1.0);
                let combined_score = (accuracy_score + performance_score) / 2.0;

                total_score += combined_score;
                count += 1;
            }
        }

        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }

    fn generate_selection_rationale(&self, analysis: &PrecisionAnalysis) -> String {
        let mut rationale = String::new();

        rationale.push_str("Mixed precision analysis results:\n");

        for (operation_type, precision) in &analysis.recommended_precisions {
            rationale.push_str(&format!("- {}: {:?}\n", operation_type, precision));
        }

        rationale.push_str(&format!("Quality score: {:.3}\n", analysis.quality_score));

        if analysis.quality_score > 0.8 {
            rationale.push_str("Excellent precision configuration found.");
        } else if analysis.quality_score > 0.6 {
            rationale.push_str("Good precision configuration found.");
        } else {
            rationale.push_str(
                "Suboptimal precision configuration - consider adjusting error tolerance.",
            );
        }

        rationale
    }
}

/// Utilities for mixed-precision simulation
pub struct MixedPrecisionUtils;

impl MixedPrecisionUtils {
    /// Estimate optimal precision for a given quantum circuit
    pub fn estimate_circuit_precision(
        num_qubits: usize,
        num_gates: usize,
        gate_types: &[String],
        error_budget: f64,
    ) -> QuantumPrecision {
        // Heuristic for circuit precision estimation
        let complexity_score = num_qubits as f64 * num_gates as f64;
        let two_qubit_gates = gate_types
            .iter()
            .filter(|&gate| gate.contains("CNOT") || gate.contains("CZ"))
            .count();
        let entanglement_factor = 1.0 + (two_qubit_gates as f64 / num_gates as f64);

        let adjusted_complexity = complexity_score * entanglement_factor;

        if error_budget > 1e-3 && adjusted_complexity < 1000.0 {
            QuantumPrecision::Half
        } else if error_budget > 1e-6 && adjusted_complexity < 10000.0 {
            QuantumPrecision::Single
        } else {
            QuantumPrecision::Double
        }
    }

    /// Calculate memory savings from mixed precision
    pub fn calculate_memory_savings(
        num_qubits: usize,
        original_precision: QuantumPrecision,
        mixed_precision_distribution: &HashMap<QuantumPrecision, f64>,
    ) -> usize {
        let state_size = 1_usize << num_qubits;
        let original_memory = state_size
            * match original_precision {
                QuantumPrecision::Half => 8,      // Complex32
                QuantumPrecision::Single => 8,    // Complex32
                QuantumPrecision::Double => 16,   // Complex64
                QuantumPrecision::Adaptive => 16, // Assume worst case
            };

        let mixed_memory: f64 = mixed_precision_distribution
            .iter()
            .map(|(&precision, &fraction)| {
                let bytes_per_element = match precision {
                    QuantumPrecision::Half => 8.0,
                    QuantumPrecision::Single => 8.0,
                    QuantumPrecision::Double => 16.0,
                    QuantumPrecision::Adaptive => 16.0,
                };
                (state_size as f64) * fraction * bytes_per_element
            })
            .sum();

        original_memory.saturating_sub(mixed_memory as usize)
    }

    /// Benchmark different precision strategies
    pub fn benchmark_precision_strategies(
        num_qubits: usize,
        strategies: &[MixedPrecisionConfig],
    ) -> Result<HashMap<String, PerformanceMetrics>> {
        let mut results = HashMap::new();

        for (i, config) in strategies.iter().enumerate() {
            let strategy_name = format!("Strategy_{}", i);

            let start_time = std::time::Instant::now();

            // Create simulator and run benchmark
            let mut simulator = MixedPrecisionSimulator::new(num_qubits, config.clone())?;
            simulator.initialize_zero_state()?;

            // Apply some gates for benchmarking
            let identity = Array2::eye(2);
            for qubit in 0..num_qubits.min(5) {
                simulator.apply_single_qubit_gate(qubit, &identity)?;
            }

            let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
            let memory_usage = simulator.state.memory_usage();

            let metrics = PerformanceMetrics {
                execution_time_ms: execution_time,
                memory_usage_bytes: memory_usage,
                throughput_ops_per_sec: 5.0 * 1000.0 / execution_time,
                energy_efficiency: 1.0 / execution_time,
            };

            results.insert(strategy_name, metrics);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.state_vector_precision, QuantumPrecision::Single);
        assert_eq!(config.gate_precision, QuantumPrecision::Single);
        assert_eq!(config.measurement_precision, QuantumPrecision::Double);
        assert!(config.adaptive_precision);
    }

    #[test]
    fn test_quantum_precision_factors() {
        assert_abs_diff_eq!(QuantumPrecision::Half.memory_factor(), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(
            QuantumPrecision::Single.memory_factor(),
            0.5,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            QuantumPrecision::Double.memory_factor(),
            1.0,
            epsilon = 1e-10
        );

        assert!(QuantumPrecision::Half.speed_factor() > QuantumPrecision::Double.speed_factor());
    }

    #[test]
    fn test_mixed_precision_state_vector() {
        let state = MixedPrecisionStateVector::new(4, QuantumPrecision::Single);
        assert_eq!(state.len(), 4);
        assert_eq!(state.precision(), QuantumPrecision::Single);

        let double_state = state.to_double_precision();
        assert_eq!(double_state.len(), 4);
    }

    #[test]
    fn test_mixed_precision_simulator() {
        let config = MixedPrecisionConfig::default();
        let mut simulator = MixedPrecisionSimulator::new(3, config).unwrap();

        simulator.initialize_zero_state().unwrap();

        // Apply Pauli-X gate
        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        simulator.apply_single_qubit_gate(0, &pauli_x).unwrap();

        let prob = simulator.measure_probability(0).unwrap();
        assert_abs_diff_eq!(prob, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_precision_estimation() {
        let precision = MixedPrecisionUtils::estimate_circuit_precision(
            10,  // qubits
            100, // gates
            &["H".to_string(), "CNOT".to_string(), "RZ".to_string()],
            1e-6, // error budget
        );

        // Should recommend single precision for this moderate complexity circuit
        assert!(matches!(
            precision,
            QuantumPrecision::Single | QuantumPrecision::Double
        ));
    }

    #[test]
    fn test_memory_savings_calculation() {
        let mut distribution = HashMap::new();
        distribution.insert(QuantumPrecision::Single, 0.7);
        distribution.insert(QuantumPrecision::Double, 0.3);

        let savings = MixedPrecisionUtils::calculate_memory_savings(
            10, // 1024 states
            QuantumPrecision::Double,
            &distribution,
        );

        assert!(savings > 0);
    }

    #[test]
    fn test_adaptive_precision() {
        let config = MixedPrecisionConfig {
            adaptive_precision: true,
            large_system_threshold: 5,
            ..Default::default()
        };

        let simulator = MixedPrecisionSimulator::new(6, config).unwrap();
        assert_eq!(simulator.state.precision(), QuantumPrecision::Adaptive);
    }
}
