//! Configuration types for hardware compiler passes

use std::collections::HashSet;
use std::time::Duration;

use crate::adaptive_compilation::AdaptiveCompilationConfig;

/// Analysis depth levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisDepth {
    Basic,
    Intermediate,
    Advanced,
    Comprehensive,
}

impl Default for AnalysisDepth {
    fn default() -> Self {
        Self::Basic
    }
}

/// Optimization objectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationObjective {
    /// Minimize circuit depth
    MinimizeDepth,
    /// Minimize gate count
    MinimizeGateCount,
    /// Minimize error probability
    MinimizeError,
    /// Maximize fidelity
    MaximizeFidelity,
    /// Minimize execution time
    MinimizeExecutionTime,
    /// Minimize resource usage
    MinimizeResources,
    /// Minimize crosstalk effects
    MinimizeCrosstalk,
}

/// Advanced compiler pass configuration with SciRS2 integration
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Enable hardware-aware gate synthesis
    pub enable_gate_synthesis: bool,
    /// Enable error-aware optimization
    pub enable_error_optimization: bool,
    /// Enable timing-aware scheduling
    pub enable_timing_optimization: bool,
    /// Enable crosstalk mitigation
    pub enable_crosstalk_mitigation: bool,
    /// Enable resource optimization
    pub enable_resource_optimization: bool,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Optimization tolerance
    pub tolerance: f64,
    /// Target compilation platform
    pub target: super::types::CompilationTarget,
    /// SciRS2 optimization integration
    pub scirs2_config: SciRS2Config,
    /// Parallel compilation settings
    pub parallel_config: ParallelConfig,
    /// Adaptive compilation settings
    pub adaptive_config: Option<AdaptiveCompilationConfig>,
    /// Performance monitoring
    pub performance_monitoring: bool,
    /// Circuit analysis depth
    pub analysis_depth: AnalysisDepth,
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Hardware constraints
    pub constraints: HardwareConstraints,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            enable_gate_synthesis: true,
            enable_error_optimization: true,
            enable_timing_optimization: false,
            enable_crosstalk_mitigation: false,
            enable_resource_optimization: true,
            max_iterations: 50,
            tolerance: 1e-6,
            target: super::types::CompilationTarget::Custom {
                name: "default".to_string(),
                capabilities: crate::backend_traits::BackendCapabilities::default(),
                constraints: HardwareConstraints::default(),
            },
            scirs2_config: SciRS2Config::default(),
            parallel_config: ParallelConfig::default(),
            adaptive_config: None,
            performance_monitoring: false,
            analysis_depth: AnalysisDepth::Basic,
            objectives: vec![OptimizationObjective::MinimizeError],
            constraints: HardwareConstraints::default(),
        }
    }
}

/// SciRS2 optimization configuration
#[derive(Debug, Clone)]
pub struct SciRS2Config {
    /// Enable graph optimization algorithms
    pub enable_graph_optimization: bool,
    /// Enable statistical analysis
    pub enable_statistical_analysis: bool,
    /// Enable advanced optimization methods
    pub enable_advanced_optimization: bool,
    /// Enable linear algebra optimization
    pub enable_linalg_optimization: bool,
    /// Optimization method selection
    pub optimization_method: SciRS2OptimizationMethod,
    /// Statistical significance threshold
    pub significance_threshold: f64,
}

impl Default for SciRS2Config {
    fn default() -> Self {
        Self {
            enable_graph_optimization: true,
            enable_statistical_analysis: true,
            enable_advanced_optimization: false,
            enable_linalg_optimization: true,
            optimization_method: SciRS2OptimizationMethod::NelderMead,
            significance_threshold: 0.05,
        }
    }
}

/// SciRS2 optimization methods
#[derive(Debug, Clone, PartialEq)]
pub enum SciRS2OptimizationMethod {
    NelderMead,
    BFGS,
    ConjugateGradient,
    SimulatedAnnealing,
    GeneticAlgorithm,
    ParticleSwarm,
    DifferentialEvolution,
    Custom(String),
}

/// Parallel compilation configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Enable parallel pass execution
    pub enable_parallel_passes: bool,
    /// Number of worker threads
    pub num_threads: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Enable SIMD optimization
    pub enable_simd: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enable_parallel_passes: true,
            num_threads: num_cpus::get(),
            chunk_size: 50,
            enable_simd: true,
        }
    }
}

/// Hardware constraints for compilation
#[derive(Debug, Clone)]
pub struct HardwareConstraints {
    /// Maximum circuit depth
    pub max_depth: Option<usize>,
    /// Maximum number of gates
    pub max_gates: Option<usize>,
    /// Maximum execution time (microseconds)
    pub max_execution_time: Option<f64>,
    /// Minimum fidelity threshold
    pub min_fidelity_threshold: f64,
    /// Maximum allowed error rate
    pub max_error_rate: f64,
    /// Forbidden qubit pairs
    pub forbidden_pairs: HashSet<(usize, usize)>,
    /// Minimum idle time between operations (nanoseconds)
    pub min_idle_time: f64,
}

impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            max_depth: None,
            max_gates: None,
            max_execution_time: None,
            min_fidelity_threshold: 0.9,
            max_error_rate: 0.1,
            forbidden_pairs: HashSet::new(),
            min_idle_time: 100.0,
        }
    }
}

/// Pass execution priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Pass execution configuration
#[derive(Debug, Clone)]
pub struct PassConfig {
    /// Pass name
    pub name: String,
    /// Execution priority
    pub priority: PassPriority,
    /// Maximum execution time
    pub timeout: Duration,
    /// Enable detailed metrics collection
    pub collect_metrics: bool,
    /// Pass-specific parameters
    pub parameters: std::collections::HashMap<String, String>,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            priority: PassPriority::Medium,
            timeout: Duration::from_secs(30),
            collect_metrics: true,
            parameters: std::collections::HashMap::new(),
        }
    }
}