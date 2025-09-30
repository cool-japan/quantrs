//! Adaptive Precision Management for Quantum Simulations
//!
//! Dynamically adjusts numerical precision based on circuit characteristics,
//! error tolerance requirements, and computational resources to optimize
//! performance while maintaining accuracy.

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::Complex64;
use std::collections::HashMap;

/// Available precision levels for quantum computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrecisionLevel {
    /// Single precision (f32) - fastest but least accurate
    Single,
    /// Double precision (f64) - standard accuracy
    Double,
    /// Extended precision - highest accuracy but slowest
    Extended,
    /// Mixed precision - adaptive based on operation
    Mixed,
}

impl PrecisionLevel {
    /// Get the number of significant decimal digits for this precision level
    pub fn decimal_digits(self) -> usize {
        match self {
            PrecisionLevel::Single => 7,
            PrecisionLevel::Double => 15,
            PrecisionLevel::Extended => 34,
            PrecisionLevel::Mixed => 15, // Default to double
        }
    }

    /// Get the epsilon (machine precision) for this level
    pub fn epsilon(self) -> f64 {
        match self {
            PrecisionLevel::Single => f32::EPSILON as f64,
            PrecisionLevel::Double => f64::EPSILON,
            PrecisionLevel::Extended => 1e-34,
            PrecisionLevel::Mixed => f64::EPSILON,
        }
    }

    /// Get relative performance factor (1.0 = baseline double precision)
    pub fn performance_factor(self) -> f64 {
        match self {
            PrecisionLevel::Single => 1.8,   // ~80% faster
            PrecisionLevel::Double => 1.0,   // Baseline
            PrecisionLevel::Extended => 0.3, // ~70% slower
            PrecisionLevel::Mixed => 1.2,    // ~20% faster on average
        }
    }
}

/// Precision requirements for different quantum operations
#[derive(Debug, Clone)]
pub struct PrecisionRequirements {
    pub min_precision: PrecisionLevel,
    pub error_tolerance: f64,
    pub relative_importance: f64, // 0.0 to 1.0
}

impl Default for PrecisionRequirements {
    fn default() -> Self {
        Self {
            min_precision: PrecisionLevel::Double,
            error_tolerance: 1e-12,
            relative_importance: 0.5,
        }
    }
}

/// Adaptive precision manager
pub struct AdaptivePrecisionManager {
    /// Current global precision level
    current_precision: PrecisionLevel,
    /// Operation-specific precision requirements
    operation_requirements: HashMap<String, PrecisionRequirements>,
    /// Accumulated error estimates
    error_estimates: HashMap<String, f64>,
    /// Performance statistics
    statistics: PrecisionStatistics,
    /// Target error budget
    global_error_budget: f64,
}

/// Precision management statistics
#[derive(Debug, Clone, Default)]
pub struct PrecisionStatistics {
    pub total_operations: u64,
    pub precision_downgrades: u64,
    pub precision_upgrades: u64,
    pub mixed_precision_ops: u64,
    pub error_budget_used: f64,
    pub average_speedup: f64,
    pub memory_savings: f64,
}

impl AdaptivePrecisionManager {
    /// Create a new adaptive precision manager
    pub fn new(global_error_budget: f64) -> Self {
        let mut manager = Self {
            current_precision: PrecisionLevel::Double,
            operation_requirements: HashMap::new(),
            error_estimates: HashMap::new(),
            statistics: PrecisionStatistics::default(),
            global_error_budget,
        };

        // Initialize with common quantum operation requirements
        manager.initialize_default_requirements();
        manager
    }

    /// Initialize default precision requirements for quantum operations
    fn initialize_default_requirements(&mut self) {
        let operations = vec![
            // High-precision operations
            (
                "eigenvalue_decomposition",
                PrecisionRequirements {
                    min_precision: PrecisionLevel::Double,
                    error_tolerance: 1e-14,
                    relative_importance: 0.9,
                },
            ),
            (
                "quantum_fourier_transform",
                PrecisionRequirements {
                    min_precision: PrecisionLevel::Double,
                    error_tolerance: 1e-13,
                    relative_importance: 0.8,
                },
            ),
            (
                "phase_estimation",
                PrecisionRequirements {
                    min_precision: PrecisionLevel::Double,
                    error_tolerance: 1e-12,
                    relative_importance: 0.9,
                },
            ),
            // Medium-precision operations
            (
                "gate_application",
                PrecisionRequirements {
                    min_precision: PrecisionLevel::Single,
                    error_tolerance: 1e-10,
                    relative_importance: 0.6,
                },
            ),
            (
                "state_vector_norm",
                PrecisionRequirements {
                    min_precision: PrecisionLevel::Single,
                    error_tolerance: 1e-8,
                    relative_importance: 0.4,
                },
            ),
            (
                "expectation_value",
                PrecisionRequirements {
                    min_precision: PrecisionLevel::Double,
                    error_tolerance: 1e-11,
                    relative_importance: 0.7,
                },
            ),
            // Lower-precision operations
            (
                "probability_calculation",
                PrecisionRequirements {
                    min_precision: PrecisionLevel::Single,
                    error_tolerance: 1e-6,
                    relative_importance: 0.3,
                },
            ),
            (
                "measurement_sampling",
                PrecisionRequirements {
                    min_precision: PrecisionLevel::Single,
                    error_tolerance: 1e-5,
                    relative_importance: 0.2,
                },
            ),
        ];

        for (op_name, requirements) in operations {
            self.operation_requirements
                .insert(op_name.to_string(), requirements);
        }
    }

    /// Determine optimal precision for a specific operation
    pub fn determine_optimal_precision(
        &mut self,
        operation_name: &str,
        input_size: usize,
        target_accuracy: Option<f64>,
    ) -> PrecisionLevel {
        self.statistics.total_operations += 1;

        // Get operation requirements
        let requirements = self
            .operation_requirements
            .get(operation_name)
            .cloned()
            .unwrap_or_default();

        // Factor in target accuracy if provided
        let effective_tolerance = if let Some(target) = target_accuracy {
            target.min(requirements.error_tolerance)
        } else {
            requirements.error_tolerance
        };

        // Consider current error budget
        let remaining_budget = self.global_error_budget - self.statistics.error_budget_used;

        // Size-based precision adjustment
        let size_factor = self.calculate_size_factor(input_size);

        // Determine precision level
        let optimal_precision = self.select_precision_level(
            &requirements,
            effective_tolerance,
            remaining_budget,
            size_factor,
        );

        // Update statistics
        self.update_precision_statistics(optimal_precision, &requirements);

        // Track error contribution
        let estimated_error =
            self.estimate_operation_error(operation_name, optimal_precision, input_size);
        self.error_estimates
            .insert(operation_name.to_string(), estimated_error);
        self.statistics.error_budget_used += estimated_error;

        optimal_precision
    }

    /// Calculate size factor for precision adjustment
    fn calculate_size_factor(&self, input_size: usize) -> f64 {
        // Larger problems may accumulate more numerical error
        match input_size {
            0..=100 => 1.0,
            101..=1000 => 1.2,
            1001..=10000 => 1.5,
            _ => 2.0,
        }
    }

    /// Select the appropriate precision level
    fn select_precision_level(
        &self,
        requirements: &PrecisionRequirements,
        effective_tolerance: f64,
        remaining_budget: f64,
        size_factor: f64,
    ) -> PrecisionLevel {
        let adjusted_tolerance = effective_tolerance / size_factor;

        // If we're running out of error budget, upgrade precision
        if remaining_budget < self.global_error_budget * 0.1 {
            return match requirements.min_precision {
                PrecisionLevel::Single => PrecisionLevel::Double,
                PrecisionLevel::Double => PrecisionLevel::Extended,
                other => other,
            };
        }

        // Select based on tolerance requirements
        if adjusted_tolerance < 1e-14 {
            PrecisionLevel::Extended
        } else if adjusted_tolerance < 1e-10 {
            PrecisionLevel::Double
        } else if adjusted_tolerance < 1e-6 {
            PrecisionLevel::Single
        } else {
            PrecisionLevel::Single
        }
    }

    /// Update precision statistics
    fn update_precision_statistics(
        &mut self,
        chosen_precision: PrecisionLevel,
        requirements: &PrecisionRequirements,
    ) {
        match chosen_precision.cmp(&requirements.min_precision) {
            std::cmp::Ordering::Less => {
                // This shouldn't happen in a well-designed system
            }
            std::cmp::Ordering::Greater => {
                self.statistics.precision_upgrades += 1;
            }
            std::cmp::Ordering::Equal => {
                // No change
            }
        }

        if chosen_precision == PrecisionLevel::Mixed {
            self.statistics.mixed_precision_ops += 1;
        }

        // Update performance metrics
        let speedup = chosen_precision.performance_factor();
        self.statistics.average_speedup = (self.statistics.average_speedup
            * (self.statistics.total_operations - 1) as f64
            + speedup)
            / self.statistics.total_operations as f64;

        // Memory savings (rough estimate)
        let memory_factor = match chosen_precision {
            PrecisionLevel::Single => 0.5,   // Half the memory of double
            PrecisionLevel::Double => 1.0,   // Baseline
            PrecisionLevel::Extended => 2.0, // Double the memory
            PrecisionLevel::Mixed => 0.75,   // Average savings
        };

        self.statistics.memory_savings = (self.statistics.memory_savings
            * (self.statistics.total_operations - 1) as f64
            + memory_factor)
            / self.statistics.total_operations as f64;
    }

    /// Estimate numerical error for an operation at given precision
    fn estimate_operation_error(
        &self,
        operation_name: &str,
        precision: PrecisionLevel,
        input_size: usize,
    ) -> f64 {
        let base_error = precision.epsilon();
        let size_scaling = (input_size as f64).log2() * 0.1; // Error grows logarithmically with size

        // Operation-specific error factors
        let operation_factor = match operation_name {
            "eigenvalue_decomposition" => 10.0, // Iterative algorithms accumulate error
            "quantum_fourier_transform" => 5.0, // Many arithmetic operations
            "matrix_multiplication" => 3.0,     // O(n³) operations
            "gate_application" => 1.0,          // Simple operations
            "probability_calculation" => 0.5,   // Usually normalized
            _ => 2.0,                           // Default factor
        };

        base_error * operation_factor * (1.0 + size_scaling)
    }

    /// Get current precision statistics
    pub fn get_statistics(&self) -> &PrecisionStatistics {
        &self.statistics
    }

    /// Reset the error budget and statistics
    pub fn reset_error_budget(&mut self) {
        self.statistics.error_budget_used = 0.0;
        self.error_estimates.clear();
    }

    /// Check if we're within error budget
    pub fn within_error_budget(&self) -> bool {
        self.statistics.error_budget_used < self.global_error_budget
    }

    /// Get current error budget utilization (0.0 to 1.0)
    pub fn error_budget_utilization(&self) -> f64 {
        self.statistics.error_budget_used / self.global_error_budget
    }

    /// Add custom precision requirements for an operation
    pub fn set_operation_requirements(
        &mut self,
        operation_name: String,
        requirements: PrecisionRequirements,
    ) {
        self.operation_requirements
            .insert(operation_name, requirements);
    }

    /// Suggest optimal precision level for a quantum circuit
    pub fn suggest_circuit_precision(
        &self,
        circuit_operations: &[(String, usize)],
        target_fidelity: f64,
    ) -> PrecisionLevel {
        let error_tolerance = 1.0 - target_fidelity;
        let total_operations = circuit_operations.len();

        // Allocate error budget across operations
        let per_operation_budget = error_tolerance / total_operations as f64;

        // Find the most stringent precision requirement
        let mut max_precision = PrecisionLevel::Single;

        for (op_name, size) in circuit_operations {
            let requirements = self
                .operation_requirements
                .get(op_name)
                .cloned()
                .unwrap_or_default();

            if per_operation_budget < requirements.error_tolerance {
                max_precision = match max_precision {
                    PrecisionLevel::Single => {
                        if requirements.min_precision != PrecisionLevel::Single {
                            requirements.min_precision
                        } else {
                            PrecisionLevel::Double
                        }
                    }
                    PrecisionLevel::Double => {
                        if per_operation_budget < 1e-14 {
                            PrecisionLevel::Extended
                        } else {
                            PrecisionLevel::Double
                        }
                    }
                    PrecisionLevel::Extended => PrecisionLevel::Extended,
                    PrecisionLevel::Mixed => PrecisionLevel::Mixed,
                };
            }
        }

        max_precision
    }
}

/// Adapt precision level for a complete quantum circuit
pub fn adapt_precision_for_circuit(
    circuit_operations: &[(String, usize)],
    target_fidelity: f64,
    error_budget: f64,
) -> QuantRS2Result<PrecisionLevel> {
    let manager = AdaptivePrecisionManager::new(error_budget);
    Ok(manager.suggest_circuit_precision(circuit_operations, target_fidelity))
}

/// Precision-aware complex number operations
pub struct PrecisionAwareOps;

impl PrecisionAwareOps {
    /// Multiply two complex numbers with specified precision
    pub fn multiply_complex(a: Complex64, b: Complex64, precision: PrecisionLevel) -> Complex64 {
        match precision {
            PrecisionLevel::Single => {
                let a32 = Complex::<f32>::new(a.re as f32, a.im as f32);
                let b32 = Complex::<f32>::new(b.re as f32, b.im as f32);
                let result32 = a32 * b32;
                Complex64::new(result32.re as f64, result32.im as f64)
            }
            PrecisionLevel::Double | PrecisionLevel::Mixed => a * b,
            PrecisionLevel::Extended => {
                // For extended precision, we'd use a higher-precision library
                // For now, fall back to double precision
                a * b
            }
        }
    }

    /// Add two complex numbers with specified precision
    pub fn add_complex(a: Complex64, b: Complex64, precision: PrecisionLevel) -> Complex64 {
        match precision {
            PrecisionLevel::Single => {
                let a32 = Complex::<f32>::new(a.re as f32, a.im as f32);
                let b32 = Complex::<f32>::new(b.re as f32, b.im as f32);
                let result32 = a32 + b32;
                Complex64::new(result32.re as f64, result32.im as f64)
            }
            PrecisionLevel::Double | PrecisionLevel::Mixed => a + b,
            PrecisionLevel::Extended => a + b,
        }
    }

    /// Compute norm with specified precision
    pub fn norm_complex(a: Complex64, precision: PrecisionLevel) -> f64 {
        match precision {
            PrecisionLevel::Single => {
                let a32 = Complex::<f32>::new(a.re as f32, a.im as f32);
                a32.norm() as f64
            }
            PrecisionLevel::Double | PrecisionLevel::Mixed | PrecisionLevel::Extended => a.norm(),
        }
    }
}

// Import Complex for single precision operations
use scirs2_core::Complex;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_level_properties() {
        assert_eq!(PrecisionLevel::Single.decimal_digits(), 7);
        assert_eq!(PrecisionLevel::Double.decimal_digits(), 15);
        assert!(
            PrecisionLevel::Single.performance_factor()
                > PrecisionLevel::Double.performance_factor()
        );
    }

    #[test]
    fn test_adaptive_precision_manager() {
        let mut manager = AdaptivePrecisionManager::new(1e-10);

        // Test operation precision determination
        let precision = manager.determine_optimal_precision("gate_application", 100, Some(1e-8));
        assert!(precision == PrecisionLevel::Single || precision == PrecisionLevel::Double);

        // Test high-precision requirement
        let high_precision =
            manager.determine_optimal_precision("eigenvalue_decomposition", 1000, Some(1e-14));
        assert!(
            high_precision == PrecisionLevel::Double || high_precision == PrecisionLevel::Extended
        );
    }

    #[test]
    fn test_error_budget_management() {
        let mut manager = AdaptivePrecisionManager::new(1e-6);

        // Consume some error budget
        manager.determine_optimal_precision("gate_application", 100, None);
        manager.determine_optimal_precision("measurement_sampling", 50, None);

        assert!(manager.error_budget_utilization() > 0.0);
        assert!(manager.error_budget_utilization() < 1.0);
    }

    #[test]
    fn test_circuit_precision_suggestion() {
        let manager = AdaptivePrecisionManager::new(1e-10);

        let operations = vec![
            ("gate_application".to_string(), 100),
            ("measurement_sampling".to_string(), 50),
            ("expectation_value".to_string(), 200),
        ];

        let suggested = manager.suggest_circuit_precision(&operations, 0.999);
        assert!(matches!(
            suggested,
            PrecisionLevel::Single | PrecisionLevel::Double
        ));

        // High fidelity should suggest higher precision
        let high_fidelity = manager.suggest_circuit_precision(&operations, 0.9999999);
        assert!(matches!(
            high_fidelity,
            PrecisionLevel::Double | PrecisionLevel::Extended
        ));
    }

    #[test]
    fn test_precision_aware_operations() {
        let a = Complex64::new(1.1234567890123456, 2.9876543210987654);
        let b = Complex64::new(std::f64::consts::PI, std::f64::consts::SQRT_2);

        let single_result = PrecisionAwareOps::multiply_complex(a, b, PrecisionLevel::Single);
        let double_result = PrecisionAwareOps::multiply_complex(a, b, PrecisionLevel::Double);

        // Single precision should be less accurate
        assert!((single_result - double_result).norm() > 0.0);

        // But should be reasonably close for this simple case
        assert!((single_result - double_result).norm() < 1e-6);
    }

    #[test]
    fn test_operation_requirements() {
        let mut manager = AdaptivePrecisionManager::new(1e-10);

        let custom_req = PrecisionRequirements {
            min_precision: PrecisionLevel::Extended,
            error_tolerance: 1e-16,
            relative_importance: 1.0,
        };

        manager.set_operation_requirements("custom_operation".to_string(), custom_req);

        let precision = manager.determine_optimal_precision("custom_operation", 1000, None);
        assert_eq!(precision, PrecisionLevel::Extended);
    }

    #[test]
    fn test_size_factor_scaling() {
        let manager = AdaptivePrecisionManager::new(1e-10);

        let small_factor = manager.calculate_size_factor(50);
        let large_factor = manager.calculate_size_factor(50000);

        assert!(large_factor > small_factor);
        assert_eq!(small_factor, 1.0); // Small sizes don't need adjustment
    }
}
