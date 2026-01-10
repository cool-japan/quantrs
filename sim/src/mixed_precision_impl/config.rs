//! Configuration structures for mixed-precision quantum simulation.
//!
//! This module provides configuration types for precision levels,
//! adaptive strategies, and performance optimization settings.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SimulatorError};

// Note: scirs2_linalg mixed_precision module temporarily unavailable
// #[cfg(feature = "advanced_math")]
// use scirs2_linalg::mixed_precision::{AdaptiveStrategy, MixedPrecisionContext, PrecisionLevel};

// Placeholder types when the feature is not available
#[derive(Debug)]
pub struct MixedPrecisionContext;

#[derive(Debug)]
pub enum PrecisionLevel {
    F16,
    F32,
    F64,
    Adaptive,
}

#[derive(Debug)]
pub enum AdaptiveStrategy {
    ErrorBased(f64),
    Fixed(PrecisionLevel),
}

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
    /// Half precision (16-bit floats, FP16)
    Half,
    /// BFloat16 precision (16-bit with larger range)
    BFloat16,
    /// TensorFloat-32 (NVIDIA TF32 for Tensor Cores)
    TF32,
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
    pub const fn to_scirs2_precision(&self) -> PrecisionLevel {
        match self {
            Self::Half | Self::BFloat16 => PrecisionLevel::F16,
            Self::TF32 | Self::Single => PrecisionLevel::F32,
            Self::Double => PrecisionLevel::F64,
            Self::Adaptive => PrecisionLevel::Adaptive,
        }
    }

    /// Get memory usage factor relative to double precision
    #[must_use]
    pub const fn memory_factor(&self) -> f64 {
        match self {
            Self::Half => 0.25,
            Self::BFloat16 => 0.25,
            Self::TF32 => 0.5, // Same storage as FP32, but faster compute
            Self::Single => 0.5,
            Self::Double => 1.0,
            Self::Adaptive => 0.75, // Average case
        }
    }

    /// Get computational cost factor relative to double precision
    /// Lower is better - represents performance relative to FP64
    #[must_use]
    pub const fn computation_factor(&self) -> f64 {
        match self {
            Self::Half => 0.25,     // ~4x faster on Tensor Cores
            Self::BFloat16 => 0.25, // ~4x faster on Tensor Cores
            Self::TF32 => 0.35,     // ~2.8x faster on Tensor Cores
            Self::Single => 0.7,
            Self::Double => 1.0,
            Self::Adaptive => 0.6, // Average case
        }
    }

    /// Get typical numerical error for this precision
    #[must_use]
    pub const fn typical_error(&self) -> f64 {
        match self {
            Self::Half => 1e-3,     // 10-bit mantissa
            Self::BFloat16 => 1e-2, // 7-bit mantissa, but same range as FP32
            Self::TF32 => 1e-4,     // 10-bit mantissa with FP32 range
            Self::Single => 1e-6,   // 23-bit mantissa
            Self::Double => 1e-15,  // 52-bit mantissa
            Self::Adaptive => 1e-6, // Conservative estimate
        }
    }

    /// Check if this precision requires Tensor Cores
    #[must_use]
    pub const fn requires_tensor_cores(&self) -> bool {
        matches!(self, Self::TF32 | Self::BFloat16)
    }

    /// Check if this precision is a reduced-precision format
    #[must_use]
    pub const fn is_reduced_precision(&self) -> bool {
        matches!(self, Self::Half | Self::BFloat16 | Self::TF32)
    }

    /// Get the bit width of this precision format
    #[must_use]
    pub const fn bit_width(&self) -> usize {
        match self {
            Self::Half => 16,
            Self::BFloat16 => 16,
            Self::TF32 => 19, // Stored as 32-bit, but 19 effective bits
            Self::Single => 32,
            Self::Double => 64,
            Self::Adaptive => 32, // Default to single
        }
    }

    /// Get mantissa bits for this precision
    #[must_use]
    pub const fn mantissa_bits(&self) -> usize {
        match self {
            Self::Half => 10,
            Self::BFloat16 => 7,
            Self::TF32 => 10,
            Self::Single => 23,
            Self::Double => 52,
            Self::Adaptive => 23,
        }
    }

    /// Get exponent bits for this precision
    #[must_use]
    pub const fn exponent_bits(&self) -> usize {
        match self {
            Self::Half => 5,
            Self::BFloat16 => 8,
            Self::TF32 => 8,
            Self::Single => 8,
            Self::Double => 11,
            Self::Adaptive => 8,
        }
    }

    /// Check if this precision is sufficient for the given error tolerance
    #[must_use]
    pub fn is_sufficient_for_tolerance(&self, tolerance: f64) -> bool {
        self.typical_error() <= tolerance * 10.0 // Safety factor of 10
    }

    /// Get the next higher precision level
    #[must_use]
    pub const fn higher_precision(&self) -> Option<Self> {
        match self {
            Self::Half => Some(Self::BFloat16),
            Self::BFloat16 => Some(Self::TF32),
            Self::TF32 => Some(Self::Single),
            Self::Single => Some(Self::Double),
            Self::Double => None,
            Self::Adaptive => Some(Self::Double),
        }
    }

    /// Get the next lower precision level
    #[must_use]
    pub const fn lower_precision(&self) -> Option<Self> {
        match self {
            Self::Half => None,
            Self::BFloat16 => Some(Self::Half),
            Self::TF32 => Some(Self::BFloat16),
            Self::Single => Some(Self::TF32),
            Self::Double => Some(Self::Single),
            Self::Adaptive => Some(Self::Single),
        }
    }

    /// Select best precision for given accuracy and Tensor Core availability
    #[must_use]
    pub fn select_for_accuracy_and_tensor_cores(tolerance: f64, has_tensor_cores: bool) -> Self {
        if tolerance >= 1e-2 {
            if has_tensor_cores {
                Self::BFloat16
            } else {
                Self::Half
            }
        } else if tolerance >= 1e-4 {
            if has_tensor_cores {
                Self::TF32
            } else {
                Self::Single
            }
        } else if tolerance >= 1e-6 {
            Self::Single
        } else {
            Self::Double
        }
    }
}

/// Mixed precision configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl MixedPrecisionConfig {
    /// Create configuration optimized for accuracy
    #[must_use]
    pub const fn for_accuracy() -> Self {
        Self {
            state_vector_precision: QuantumPrecision::Double,
            gate_precision: QuantumPrecision::Double,
            measurement_precision: QuantumPrecision::Double,
            error_tolerance: 1e-12,
            adaptive_precision: false,
            min_precision: QuantumPrecision::Double,
            max_precision: QuantumPrecision::Double,
            large_system_threshold: 50,
            enable_analysis: true,
        }
    }

    /// Create configuration optimized for performance
    #[must_use]
    pub const fn for_performance() -> Self {
        Self {
            state_vector_precision: QuantumPrecision::Half,
            gate_precision: QuantumPrecision::Single,
            measurement_precision: QuantumPrecision::Single,
            error_tolerance: 1e-3,
            adaptive_precision: true,
            min_precision: QuantumPrecision::Half,
            max_precision: QuantumPrecision::Single,
            large_system_threshold: 10,
            enable_analysis: false,
        }
    }

    /// Create configuration balanced between accuracy and performance
    #[must_use]
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.error_tolerance <= 0.0 {
            return Err(SimulatorError::InvalidInput(
                "Error tolerance must be positive".to_string(),
            ));
        }

        if self.large_system_threshold == 0 {
            return Err(SimulatorError::InvalidInput(
                "Large system threshold must be positive".to_string(),
            ));
        }

        // Check precision consistency
        if self.min_precision as u8 > self.max_precision as u8 {
            return Err(SimulatorError::InvalidInput(
                "Minimum precision cannot be higher than maximum precision".to_string(),
            ));
        }

        Ok(())
    }

    /// Adjust configuration for a specific number of qubits
    pub const fn adjust_for_qubits(&mut self, num_qubits: usize) {
        if num_qubits >= self.large_system_threshold {
            // For large systems, reduce precision to save memory
            if self.adaptive_precision {
                match self.state_vector_precision {
                    QuantumPrecision::Double => {
                        self.state_vector_precision = QuantumPrecision::Single;
                    }
                    QuantumPrecision::Single => {
                        self.state_vector_precision = QuantumPrecision::Half;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Estimate memory usage for a given number of qubits
    #[must_use]
    pub fn estimate_memory_usage(&self, num_qubits: usize) -> usize {
        let state_vector_size = 1 << num_qubits;
        let base_memory = state_vector_size * 16; // Complex64 size

        let factor = self.state_vector_precision.memory_factor();
        (f64::from(base_memory) * factor) as usize
    }

    /// Check if the configuration is suitable for the available memory
    #[must_use]
    pub fn fits_in_memory(&self, num_qubits: usize, available_memory: usize) -> bool {
        self.estimate_memory_usage(num_qubits) <= available_memory
    }
}
