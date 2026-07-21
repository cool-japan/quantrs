//! Cross-Platform Circuit Migration Tools
//!
//! This module provides comprehensive tools for migrating quantum circuits
//! between different quantum computing platforms with automatic optimization,
//! gate translation, topology mapping, and performance analysis.
//!
//! The public configuration/result types live here; the migration pipeline
//! itself ([`CircuitMigrationEngine`]) lives in [`engine`], and the internal
//! circuit-analysis types it uses live in [`analysis`].

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use quantrs2_circuit::prelude::*;
use serde::{Deserialize, Serialize};

use crate::translation::HardwareBackend;

mod analysis;
mod engine;
#[cfg(test)]
mod tests;

pub use engine::CircuitMigrationEngine;

/// Cross-platform circuit migration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Source platform
    pub source_platform: HardwareBackend,
    /// Target platform
    pub target_platform: HardwareBackend,
    /// Migration strategy
    pub strategy: MigrationStrategy,
    /// Optimization settings
    pub optimization: MigrationOptimizationConfig,
    /// Mapping configuration
    pub mapping_config: MigrationMappingConfig,
    /// Translation settings
    pub translation_config: MigrationTranslationConfig,
    /// Performance requirements
    pub performance_requirements: MigrationPerformanceRequirements,
    /// Validation settings
    pub validation_config: MigrationValidationConfig,
}

/// Migration strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// Direct translation with minimal changes
    Direct,
    /// Optimize for target platform
    Optimized,
    /// Preserve fidelity at all costs
    FidelityPreserving,
    /// Minimize execution time
    TimeOptimized,
    /// Minimize resource usage
    ResourceOptimized,
    /// Custom strategy with weights
    Custom {
        fidelity_weight: f64,
        time_weight: f64,
        resource_weight: f64,
    },
}

/// Migration optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationOptimizationConfig {
    /// Enable circuit optimization
    pub enable_optimization: bool,
    /// Optimization passes to apply
    pub optimization_passes: Vec<OptimizationPass>,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Enable SciRS2-powered optimization
    pub enable_scirs2_optimization: bool,
    /// Multi-objective optimization weights
    pub multi_objective_weights: HashMap<String, f64>,
}

/// Optimization passes for migration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationPass {
    /// Gate set reduction
    GateSetReduction,
    /// Circuit depth minimization
    DepthMinimization,
    /// Qubit layout optimization
    LayoutOptimization,
    /// Gate scheduling optimization
    SchedulingOptimization,
    /// Error mitigation insertion
    ErrorMitigation,
    /// Parallelization optimization
    Parallelization,
    /// Resource usage optimization
    ResourceOptimization,
}

/// Migration mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationMappingConfig {
    /// Mapping strategy
    pub strategy: MappingStrategy,
    /// Consider hardware connectivity
    pub consider_connectivity: bool,
    /// Optimize for target topology
    pub optimize_for_topology: bool,
    /// Maximum SWAP overhead allowed
    pub max_swap_overhead: f64,
    /// Enable adaptive mapping
    pub enable_adaptive_mapping: bool,
    /// Beta.3: Simple mapping fallback enabled
    /// Future: Full SciRS2 mapping configuration (post-beta.3)
    pub scirs2_config_placeholder: bool,
}

/// Qubit mapping strategies for migration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappingStrategy {
    /// Preserve original qubit indices if possible
    PreserveIndices,
    /// Map to highest fidelity qubits
    HighestFidelity,
    /// Minimize connectivity overhead
    MinimizeSwaps,
    /// Optimize for circuit structure
    CircuitAware,
    /// Use graph-based algorithms
    GraphBased,
    /// SciRS2-powered intelligent mapping
    SciRS2Optimized,
}

/// Migration translation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationTranslationConfig {
    /// Gate translation strategy
    pub gate_strategy: GateTranslationStrategy,
    /// Allow gate decomposition
    pub allow_decomposition: bool,
    /// Maximum decomposition depth
    pub max_decomposition_depth: usize,
    /// Preserve gate semantics
    pub preserve_semantics: bool,
    /// Target gate set
    pub target_gate_set: Option<HashSet<String>>,
    /// Custom gate mappings
    pub custom_mappings: HashMap<String, Vec<String>>,
}

/// Gate translation strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateTranslationStrategy {
    /// Use native gates when possible
    PreferNative,
    /// Minimize gate count
    MinimizeGates,
    /// Preserve fidelity
    PreserveFidelity,
    /// Minimize circuit depth
    MinimizeDepth,
    /// Custom priority order
    CustomPriority(Vec<String>),
}

/// Migration performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPerformanceRequirements {
    /// Minimum acceptable fidelity
    pub min_fidelity: Option<f64>,
    /// Maximum acceptable execution time
    pub max_execution_time: Option<Duration>,
    /// Maximum circuit depth increase
    pub max_depth_increase: Option<f64>,
    /// Maximum gate count increase
    pub max_gate_increase: Option<f64>,
    /// Required accuracy level
    pub accuracy_level: AccuracyLevel,
}

/// Accuracy levels for migration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccuracyLevel {
    /// Best effort migration
    BestEffort,
    /// Maintain statistical accuracy
    Statistical,
    /// Preserve quantum advantage
    QuantumAdvantage,
    /// Exact equivalence required
    Exact,
}

/// Migration validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationValidationConfig {
    /// Enable validation
    pub enable_validation: bool,
    /// Validation methods
    pub validation_methods: Vec<ValidationMethod>,
    /// Statistical test confidence level
    pub confidence_level: f64,
    /// Number of validation runs
    pub validation_runs: usize,
    /// Enable cross-validation
    pub enable_cross_validation: bool,
}

/// Validation methods for migration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationMethod {
    /// Functional equivalence testing
    FunctionalEquivalence,
    /// Statistical outcome comparison
    StatisticalComparison,
    /// Fidelity measurement
    FidelityMeasurement,
    /// Process tomography comparison
    ProcessTomography,
    /// Benchmark circuit testing
    BenchmarkTesting,
}

/// Circuit migration result
#[derive(Debug, Clone)]
pub struct MigrationResult<const N: usize> {
    /// Migrated circuit
    pub migrated_circuit: Circuit<N>,
    /// Migration metrics
    pub metrics: MigrationMetrics,
    /// Applied transformations
    pub transformations: Vec<AppliedTransformation>,
    /// Validation results
    pub validation: Option<ValidationResult>,
    /// Migration warnings
    pub warnings: Vec<MigrationWarning>,
    /// Success status
    pub success: bool,
}

/// Migration metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationMetrics {
    /// Original circuit metrics
    pub original: CircuitMetrics,
    /// Migrated circuit metrics
    pub migrated: CircuitMetrics,
    /// Migration statistics
    pub migration_stats: MigrationStatistics,
    /// Performance comparison
    pub performance_comparison: PerformanceComparison,
}

/// Circuit metrics for migration analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetrics {
    /// Number of qubits
    pub qubit_count: usize,
    /// Circuit depth
    pub depth: usize,
    /// Gate count
    pub gate_count: usize,
    /// Gate count by type
    pub gate_counts: HashMap<String, usize>,
    /// Estimated fidelity
    pub estimated_fidelity: f64,
    /// Estimated execution time
    pub estimated_execution_time: Duration,
    /// Resource requirements
    pub resource_requirements: ResourceMetrics,
}

/// Resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Memory requirements (MB)
    pub memory_mb: f64,
    /// CPU time requirements
    pub cpu_time: Duration,
    /// QPU time requirements
    pub qpu_time: Duration,
    /// Network bandwidth (if applicable)
    pub network_bandwidth: Option<f64>,
}

/// Migration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatistics {
    /// Migration time
    pub migration_time: Duration,
    /// Number of transformations applied
    pub transformations_applied: usize,
    /// Optimization iterations performed
    pub optimization_iterations: usize,
    /// Mapping overhead
    pub mapping_overhead: f64,
    /// Translation efficiency
    pub translation_efficiency: f64,
}

/// Performance comparison between original and migrated circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// Fidelity change
    pub fidelity_change: f64,
    /// Execution time change
    pub execution_time_change: f64,
    /// Circuit depth change
    pub depth_change: f64,
    /// Gate count change
    pub gate_count_change: f64,
    /// Resource usage change
    pub resource_change: f64,
    /// Overall quality score
    pub quality_score: f64,
}

/// Applied transformation during migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedTransformation {
    /// Transformation type
    pub transformation_type: TransformationType,
    /// Description
    pub description: String,
    /// Impact on metrics
    pub impact: TransformationImpact,
    /// Applied at stage
    pub stage: MigrationStage,
}

/// Types of transformations during migration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformationType {
    GateTranslation,
    QubitMapping,
    CircuitOptimization,
    ErrorMitigation,
    Decomposition,
    Parallelization,
    Scheduling,
}

/// Impact of a transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationImpact {
    /// Fidelity impact
    pub fidelity_impact: f64,
    /// Time impact
    pub time_impact: f64,
    /// Resource impact
    pub resource_impact: f64,
    /// Confidence in impact estimate
    pub confidence: f64,
}

/// Migration stages
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStage {
    Analysis,
    Translation,
    Mapping,
    Optimization,
    Validation,
    Finalization,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation success
    pub overall_success: bool,
    /// Individual validation results
    pub method_results: HashMap<ValidationMethod, ValidationMethodResult>,
    /// Statistical comparison results
    pub statistical_results: StatisticalValidationResult,
    /// Confidence score
    pub confidence_score: f64,
}

/// Result of a specific validation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMethodResult {
    /// Method success
    pub success: bool,
    /// Score (0.0 to 1.0)
    pub score: f64,
    /// Details
    pub details: String,
    /// Statistical significance
    pub p_value: Option<f64>,
}

/// Statistical validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalValidationResult {
    /// Distribution comparison results
    pub distribution_comparison: DistributionComparison,
    /// Fidelity comparison
    pub fidelity_comparison: FidelityComparison,
    /// Error analysis
    pub error_analysis: ErrorAnalysis,
}

/// Distribution comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionComparison {
    /// Kolmogorov-Smirnov test result
    pub ks_test_p_value: f64,
    /// Chi-square test result
    pub chi_square_p_value: f64,
    /// Distribution distance
    pub distance: f64,
    /// Similarity score
    pub similarity_score: f64,
}

/// Fidelity comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityComparison {
    /// Average fidelity original
    pub original_fidelity: f64,
    /// Average fidelity migrated
    pub migrated_fidelity: f64,
    /// Fidelity loss
    pub fidelity_loss: f64,
    /// Statistical significance
    pub significance: f64,
}

/// Error analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    /// Error rate comparison
    pub error_rate_comparison: f64,
    /// Error correlation
    pub error_correlation: f64,
    /// Systematic errors detected
    pub systematic_errors: Vec<String>,
    /// Random error estimate
    pub random_error_estimate: f64,
}

/// Migration warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationWarning {
    /// Warning type
    pub warning_type: WarningType,
    /// Warning message
    pub message: String,
    /// Severity level
    pub severity: WarningSeverity,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Types of migration warnings
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningType {
    FidelityLoss,
    PerformanceDegradation,
    UnsupportedGates,
    TopologyMismatch,
    ResourceLimitations,
    ValidationFailure,
    ApproximationUsed,
}

/// Warning severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WarningSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            source_platform: HardwareBackend::IBMQuantum,
            target_platform: HardwareBackend::AmazonBraket,
            strategy: MigrationStrategy::Optimized,
            optimization: MigrationOptimizationConfig {
                enable_optimization: true,
                optimization_passes: vec![
                    OptimizationPass::GateSetReduction,
                    OptimizationPass::LayoutOptimization,
                    OptimizationPass::DepthMinimization,
                ],
                max_iterations: 100,
                convergence_threshold: 1e-6,
                enable_scirs2_optimization: true,
                multi_objective_weights: [
                    ("fidelity".to_string(), 0.4),
                    ("time".to_string(), 0.3),
                    ("resources".to_string(), 0.3),
                ]
                .iter()
                .cloned()
                .collect(),
            },
            mapping_config: MigrationMappingConfig {
                strategy: MappingStrategy::SciRS2Optimized,
                consider_connectivity: true,
                optimize_for_topology: true,
                max_swap_overhead: 2.0,
                enable_adaptive_mapping: true,
                scirs2_config_placeholder: true,
            },
            translation_config: MigrationTranslationConfig {
                gate_strategy: GateTranslationStrategy::PreferNative,
                allow_decomposition: true,
                max_decomposition_depth: 3,
                preserve_semantics: true,
                target_gate_set: None,
                custom_mappings: HashMap::new(),
            },
            performance_requirements: MigrationPerformanceRequirements {
                min_fidelity: Some(0.95),
                max_execution_time: None,
                max_depth_increase: Some(2.0),
                max_gate_increase: Some(1.5),
                accuracy_level: AccuracyLevel::Statistical,
            },
            validation_config: MigrationValidationConfig {
                enable_validation: true,
                validation_methods: vec![
                    ValidationMethod::FunctionalEquivalence,
                    ValidationMethod::StatisticalComparison,
                    ValidationMethod::FidelityMeasurement,
                ],
                confidence_level: 0.95,
                validation_runs: 100,
                enable_cross_validation: true,
            },
        }
    }
}
