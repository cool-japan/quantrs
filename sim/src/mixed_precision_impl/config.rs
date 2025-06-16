//! Configuration structures for mixed-precision quantum simulation.
//!
//! This module provides configuration types for precision levels,
//! adaptive strategies, and performance optimization settings.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SimulatorError};

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
            QuantumPrecision::Half => 0.25,
            QuantumPrecision::Single => 0.5,
            QuantumPrecision::Double => 1.0,
            QuantumPrecision::Adaptive => 0.75, // Average case
        }
    }

    /// Get computational cost factor relative to double precision
    pub fn computation_factor(&self) -> f64 {
        match self {
            QuantumPrecision::Half => 0.5,
            QuantumPrecision::Single => 0.7,
            QuantumPrecision::Double => 1.0,
            QuantumPrecision::Adaptive => 0.8, // Average case
        }
    }

    /// Get typical numerical error for this precision
    pub fn typical_error(&self) -> f64 {
        match self {
            QuantumPrecision::Half => 1e-3,
            QuantumPrecision::Single => 1e-6,
            QuantumPrecision::Double => 1e-15,
            QuantumPrecision::Adaptive => 1e-6, // Conservative estimate
        }
    }

    /// Check if this precision is sufficient for the given error tolerance
    pub fn is_sufficient_for_tolerance(&self, tolerance: f64) -> bool {
        self.typical_error() <= tolerance * 10.0 // Safety factor of 10
    }

    /// Get the next higher precision level
    pub fn higher_precision(&self) -> Option<QuantumPrecision> {
        match self {
            QuantumPrecision::Half => Some(QuantumPrecision::Single),
            QuantumPrecision::Single => Some(QuantumPrecision::Double),
            QuantumPrecision::Double => None,
            QuantumPrecision::Adaptive => Some(QuantumPrecision::Double),
        }
    }

    /// Get the next lower precision level
    pub fn lower_precision(&self) -> Option<QuantumPrecision> {
        match self {
            QuantumPrecision::Half => None,
            QuantumPrecision::Single => Some(QuantumPrecision::Half),
            QuantumPrecision::Double => Some(QuantumPrecision::Single),
            QuantumPrecision::Adaptive => Some(QuantumPrecision::Single),
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
    pub fn for_accuracy() -> Self {
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
    pub fn for_performance() -> Self {
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
    pub fn adjust_for_qubits(&mut self, num_qubits: usize) {
        if num_qubits >= self.large_system_threshold {
            // For large systems, reduce precision to save memory
            if self.adaptive_precision {
                match self.state_vector_precision {
                    QuantumPrecision::Double => {
                        self.state_vector_precision = QuantumPrecision::Single
                    }
                    QuantumPrecision::Single => {
                        self.state_vector_precision = QuantumPrecision::Half
                    }
                    _ => {}
                }
            }
        }
    }

    /// Estimate memory usage for a given number of qubits
    pub fn estimate_memory_usage(&self, num_qubits: usize) -> usize {
        let state_vector_size = 1 << num_qubits;
        let base_memory = state_vector_size * 16; // Complex64 size

        let factor = self.state_vector_precision.memory_factor();
        (base_memory as f64 * factor) as usize
    }

    /// Check if the configuration is suitable for the available memory
    pub fn fits_in_memory(&self, num_qubits: usize, available_memory: usize) -> bool {
        self.estimate_memory_usage(num_qubits) <= available_memory
    }
}
