//! Configuration types for the universal annealing compiler.
//!
//! This module contains configuration structures and enums for controlling
//! the compiler's behavior and optimization strategies.

use std::time::Duration;

/// Universal compiler configuration
#[derive(Debug, Clone)]
pub struct UniversalCompilerConfig {
    /// Enable automatic platform discovery
    pub auto_platform_discovery: bool,
    /// Compilation optimization level
    pub optimization_level: OptimizationLevel,
    /// Resource allocation strategy
    pub allocation_strategy: ResourceAllocationStrategy,
    /// Cost budget constraints
    pub cost_constraints: CostConstraints,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Error correction requirements
    pub error_correction: ErrorCorrectionRequirements,
    /// Scheduling preferences
    pub scheduling_preferences: SchedulingPreferences,
}

impl Default for UniversalCompilerConfig {
    fn default() -> Self {
        Self {
            auto_platform_discovery: true,
            optimization_level: OptimizationLevel::Aggressive,
            allocation_strategy: ResourceAllocationStrategy::CostEffective,
            cost_constraints: CostConstraints::default(),
            performance_requirements: PerformanceRequirements::default(),
            error_correction: ErrorCorrectionRequirements::default(),
            scheduling_preferences: SchedulingPreferences::default(),
        }
    }
}

/// Optimization levels for compilation
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Standard optimization
    Standard,
    /// Aggressive optimization
    Aggressive,
    /// Maximum optimization
    Maximum,
}

/// Resource allocation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceAllocationStrategy {
    /// Minimize cost
    CostOptimal,
    /// Maximize performance
    PerformanceOptimal,
    /// Balance cost and performance
    CostEffective,
    /// Minimize time to solution
    TimeOptimal,
    /// Maximize reliability
    ReliabilityOptimal,
    /// Custom strategy
    Custom(String),
}

/// Cost constraints for compilation
#[derive(Debug, Clone)]
pub struct CostConstraints {
    /// Maximum total cost
    pub max_total_cost: Option<f64>,
    /// Maximum cost per job
    pub max_cost_per_job: Option<f64>,
    /// Cost optimization target
    pub cost_target: CostTarget,
    /// Budget allocation
    pub budget_allocation: BudgetAllocation,
}

impl Default for CostConstraints {
    fn default() -> Self {
        Self {
            max_total_cost: Some(1000.0),
            max_cost_per_job: Some(100.0),
            cost_target: CostTarget::Minimize,
            budget_allocation: BudgetAllocation::Balanced,
        }
    }
}

/// Cost optimization targets
#[derive(Debug, Clone, PartialEq)]
pub enum CostTarget {
    /// Minimize total cost
    Minimize,
    /// Stay within budget
    BudgetConstrained,
    /// Maximize cost efficiency
    EfficiencyOptimal,
    /// Performance per dollar
    PerformancePerDollar,
}

/// Budget allocation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetAllocation {
    /// Equal allocation
    Equal,
    /// Performance-weighted allocation
    PerformanceWeighted,
    /// Priority-based allocation
    PriorityBased,
    /// Balanced allocation
    Balanced,
}

/// Performance requirements specification
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Minimum solution quality
    pub min_solution_quality: f64,
    /// Required success probability
    pub required_success_probability: f64,
    /// Performance guarantees
    pub performance_guarantees: Vec<PerformanceGuarantee>,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_execution_time: Some(Duration::from_secs(3600)),
            min_solution_quality: 0.8,
            required_success_probability: 0.9,
            performance_guarantees: vec![],
        }
    }
}

/// Performance guarantee types
#[derive(Debug, Clone)]
pub enum PerformanceGuarantee {
    /// Time-bound guarantee
    TimeBound { max_time: Duration, confidence: f64 },
    /// Quality guarantee
    QualityGuarantee { min_quality: f64, confidence: f64 },
    /// Availability guarantee
    AvailabilityGuarantee {
        uptime: f64,
        measurement_window: Duration,
    },
    /// Scalability guarantee
    ScalabilityGuarantee {
        max_problem_size: usize,
        performance_degradation: f64,
    },
}

/// Error correction requirements
#[derive(Debug, Clone)]
pub struct ErrorCorrectionRequirements {
    /// Enable error correction
    pub enable_error_correction: bool,
    /// Error correction strategy
    pub strategy: ErrorCorrectionStrategy,
    /// Error threshold
    pub error_threshold: f64,
    /// Redundancy level
    pub redundancy_level: RedundancyLevel,
}

impl Default for ErrorCorrectionRequirements {
    fn default() -> Self {
        Self {
            enable_error_correction: true,
            strategy: ErrorCorrectionStrategy::Automatic,
            error_threshold: 0.01,
            redundancy_level: RedundancyLevel::Medium,
        }
    }
}

/// Error correction strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCorrectionStrategy {
    /// No error correction
    None,
    /// Basic error correction
    Basic,
    /// Advanced error correction
    Advanced,
    /// Automatic selection
    Automatic,
    /// Hardware-optimized
    HardwareOptimized,
}

/// Redundancy levels for error correction
#[derive(Debug, Clone, PartialEq)]
pub enum RedundancyLevel {
    /// Minimal redundancy
    Minimal,
    /// Low redundancy
    Low,
    /// Medium redundancy
    Medium,
    /// High redundancy
    High,
    /// Maximum redundancy
    Maximum,
}

/// Scheduling preferences
#[derive(Debug, Clone)]
pub struct SchedulingPreferences {
    /// Scheduling priority
    pub priority: SchedulingPriority,
    /// Resource preferences
    pub resource_preferences: ResourcePreferences,
    /// Geographic preferences
    pub geographic_preferences: GeographicPreferences,
}

impl Default for SchedulingPreferences {
    fn default() -> Self {
        Self {
            priority: SchedulingPriority::Normal,
            resource_preferences: ResourcePreferences::default(),
            geographic_preferences: GeographicPreferences::default(),
        }
    }
}

/// Scheduling priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Resource preferences
#[derive(Debug, Clone)]
pub struct ResourcePreferences {
    /// Preferred platforms
    pub preferred_platforms: Vec<super::platform::QuantumPlatform>,
    /// Minimum qubits required
    pub min_qubits: usize,
    /// Maximum qubits preferred
    pub max_qubits: Option<usize>,
}

impl Default for ResourcePreferences {
    fn default() -> Self {
        Self {
            preferred_platforms: vec![],
            min_qubits: 1,
            max_qubits: None,
        }
    }
}

/// Geographic preferences
#[derive(Debug, Clone)]
pub struct GeographicPreferences {
    /// Preferred regions
    pub preferred_regions: Vec<String>,
    /// Data residency requirements
    pub data_residency: Option<String>,
}

impl Default for GeographicPreferences {
    fn default() -> Self {
        Self {
            preferred_regions: vec![],
            data_residency: None,
        }
    }
}
