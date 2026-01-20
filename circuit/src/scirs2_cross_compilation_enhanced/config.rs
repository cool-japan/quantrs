//! Configuration types and enums for cross-compilation
//!
//! This module contains all configuration structures and enumeration types
//! used by the enhanced cross-compiler.

use crate::optimization::pass_manager::OptimizationLevel;
use serde::{Deserialize, Serialize};

/// Enhanced cross-compilation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedCrossCompilationConfig {
    /// Base compilation configuration
    pub base_config: CrossCompilationConfig,

    /// Enable ML-based optimization
    pub enable_ml_optimization: bool,

    /// Enable multi-stage compilation
    pub enable_multistage_compilation: bool,

    /// Enable target-specific optimization
    pub enable_target_optimization: bool,

    /// Enable real-time monitoring
    pub enable_realtime_monitoring: bool,

    /// Enable comprehensive validation
    pub enable_comprehensive_validation: bool,

    /// Enable visual compilation flow
    pub enable_visual_flow: bool,

    /// Source frameworks
    pub source_frameworks: Vec<QuantumFramework>,

    /// Target platforms
    pub target_platforms: Vec<TargetPlatform>,

    /// Compilation strategies
    pub compilation_strategies: Vec<CompilationStrategy>,

    /// Optimization passes
    pub optimization_passes: Vec<OptimizationPass>,
}

impl Default for EnhancedCrossCompilationConfig {
    fn default() -> Self {
        Self {
            base_config: CrossCompilationConfig::default(),
            enable_ml_optimization: true,
            enable_multistage_compilation: true,
            enable_target_optimization: true,
            enable_realtime_monitoring: true,
            enable_comprehensive_validation: true,
            enable_visual_flow: true,
            source_frameworks: vec![
                QuantumFramework::QuantRS2,
                QuantumFramework::Qiskit,
                QuantumFramework::Cirq,
                QuantumFramework::PennyLane,
            ],
            target_platforms: vec![
                TargetPlatform::IBMQuantum,
                TargetPlatform::GoogleSycamore,
                TargetPlatform::IonQ,
                TargetPlatform::Rigetti,
            ],
            compilation_strategies: vec![
                CompilationStrategy::OptimizeDepth,
                CompilationStrategy::OptimizeGateCount,
                CompilationStrategy::OptimizeFidelity,
            ],
            optimization_passes: vec![
                OptimizationPass::GateFusion,
                OptimizationPass::RotationMerging,
                OptimizationPass::CommutationAnalysis,
            ],
        }
    }
}

/// Base cross-compilation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCompilationConfig {
    /// Optimization level
    pub optimization_level: OptimizationLevel,

    /// Preserve semantics
    pub preserve_semantics: bool,

    /// Enable error correction
    pub enable_error_correction: bool,

    /// Validation threshold
    pub validation_threshold: f64,
}

impl Default for CrossCompilationConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Medium,
            preserve_semantics: true,
            enable_error_correction: true,
            validation_threshold: 0.999,
        }
    }
}

/// Quantum frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumFramework {
    QuantRS2,
    Qiskit,
    Cirq,
    PennyLane,
    PyQuil,
    QSharp,
    Braket,
    OpenQASM,
}

/// Target platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetPlatform {
    IBMQuantum,
    GoogleSycamore,
    IonQ,
    Rigetti,
    Honeywell,
    AWSBraket,
    AzureQuantum,
    Simulator,
}

/// Compilation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompilationStrategy {
    OptimizeDepth,
    OptimizeGateCount,
    OptimizeFidelity,
    OptimizeExecutionTime,
    BalancedOptimization,
    CustomStrategy,
}

/// Optimization passes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationPass {
    GateFusion,
    RotationMerging,
    CommutationAnalysis,
    TemplateMatching,
    PeepholeOptimization,
    GlobalPhaseOptimization,
    NativeGateDecomposition,
    LayoutOptimization,
}
