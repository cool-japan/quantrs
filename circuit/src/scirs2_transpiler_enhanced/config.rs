//! Configuration types for enhanced transpiler

use super::hardware::HardwareSpec;
use super::passes::{ExportFormat, PerformanceConstraints, TranspilationPass};
use serde::{Deserialize, Serialize};

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Light optimization (fast)
    Light,
    /// Medium optimization (balanced)
    Medium,
    /// Aggressive optimization (slow but optimal)
    Aggressive,
    /// Custom optimization with specific passes
    Custom,
}

/// Enhanced transpiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTranspilerConfig {
    /// Target hardware specification
    pub hardware_spec: HardwareSpec,

    /// Enable ML-based routing optimization
    pub enable_ml_routing: bool,

    /// Enable hardware-aware gate decomposition
    pub enable_hw_decomposition: bool,

    /// Enable real-time performance prediction
    pub enable_performance_prediction: bool,

    /// Enable advanced error mitigation
    pub enable_error_mitigation: bool,

    /// Enable cross-platform optimization
    pub enable_cross_platform: bool,

    /// Enable visual circuit representation
    pub enable_visual_output: bool,

    /// Optimization level (0-3)
    pub optimization_level: OptimizationLevel,

    /// Custom optimization passes
    pub custom_passes: Vec<TranspilationPass>,

    /// Performance constraints
    pub performance_constraints: PerformanceConstraints,

    /// Export formats
    pub export_formats: Vec<ExportFormat>,
}

impl Default for EnhancedTranspilerConfig {
    fn default() -> Self {
        Self {
            hardware_spec: HardwareSpec::default(),
            enable_ml_routing: true,
            enable_hw_decomposition: true,
            enable_performance_prediction: true,
            enable_error_mitigation: true,
            enable_cross_platform: true,
            enable_visual_output: true,
            optimization_level: OptimizationLevel::Aggressive,
            custom_passes: Vec::new(),
            performance_constraints: PerformanceConstraints::default(),
            export_formats: vec![
                ExportFormat::QASM3,
                ExportFormat::OpenQASM,
                ExportFormat::Cirq,
            ],
        }
    }
}
