//! Type definitions for advanced testing framework

use super::Duration;

/// Problem specification for test generation
#[derive(Debug, Clone)]
pub struct ProblemSpecification {
    /// Type of problem to generate
    pub problem_type: ProblemType,
    /// Size range for the problem
    pub size_range: (usize, usize),
    /// Density specifications
    pub density: DensitySpec,
    /// Constraint specifications
    pub constraints: ConstraintSpec,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// Types of problems for testing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemType {
    /// Random Ising model
    RandomIsing,
    /// Max-Cut problem
    MaxCut,
    /// Vertex cover problem
    VertexCover,
    /// Traveling salesman problem
    TSP,
    /// Portfolio optimization
    Portfolio,
    /// Custom problem type
    Custom(String),
}

/// Density specifications for problem generation
#[derive(Debug, Clone)]
pub struct DensitySpec {
    /// Edge density range
    pub edge_density: (f64, f64),
    /// Constraint density range
    pub constraint_density: Option<(f64, f64)>,
    /// Bias sparsity (fraction of zero biases)
    pub bias_sparsity: Option<f64>,
}

/// Constraint specifications
#[derive(Debug, Clone)]
pub struct ConstraintSpec {
    /// Number of constraints
    pub num_constraints: Option<usize>,
    /// Types of constraints
    pub constraint_types: Vec<ConstraintType>,
    /// Constraint strength range
    pub strength_range: (f64, f64),
}

/// Types of constraints
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintType {
    /// Equality constraint
    Equality,
    /// Inequality constraint
    Inequality,
    /// Cardinality constraint
    Cardinality,
    /// Custom constraint
    Custom(String),
}

/// Expected metrics for test validation
#[derive(Debug, Clone)]
pub struct ExpectedMetrics {
    /// Expected solution quality range
    pub solution_quality: (f64, f64),
    /// Expected runtime range
    pub runtime: (Duration, Duration),
    /// Expected success rate
    pub success_rate: f64,
    /// Convergence expectations
    pub convergence: ConvergenceExpectation,
}

/// Convergence expectations
#[derive(Debug, Clone)]
pub struct ConvergenceExpectation {
    /// Expected convergence time
    pub convergence_time: Duration,
    /// Expected final energy
    pub final_energy: Option<f64>,
    /// Expected energy gap
    pub energy_gap: Option<f64>,
}

/// Validation criterion for test results
#[derive(Debug, Clone)]
pub struct ValidationCriterion {
    /// Type of criterion
    pub criterion_type: CriterionType,
    /// Expected value or range
    pub expected_value: CriterionValue,
    /// Tolerance for comparison
    pub tolerance: f64,
    /// Whether this criterion is mandatory
    pub mandatory: bool,
}

/// Types of validation criteria
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CriterionType {
    /// Performance criterion
    Performance,
    /// Quality criterion
    Quality,
    /// Runtime criterion
    Runtime,
    /// Memory criterion
    Memory,
    /// Convergence criterion
    Convergence,
}

/// Values for criteria validation
#[derive(Debug, Clone)]
pub enum CriterionValue {
    /// Single target value
    Target(f64),
    /// Range of acceptable values
    Range(f64, f64),
    /// Minimum threshold
    Minimum(f64),
    /// Maximum threshold
    Maximum(f64),
}

/// Test execution result
#[derive(Debug, Clone)]
pub struct TestExecutionResult {
    /// Solution quality achieved
    pub solution_quality: f64,
    /// Execution time
    pub execution_time: Duration,
    /// Final energy found
    pub final_energy: f64,
    /// Best solution found
    pub best_solution: Vec<i8>,
    /// Whether convergence was achieved
    pub convergence_achieved: bool,
    /// Memory usage during execution
    pub memory_used: usize,
}

/// Validation result for a criterion
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// The criterion being validated
    pub criterion: ValidationCriterion,
    /// Whether the criterion passed
    pub passed: bool,
    /// Actual value observed
    pub actual_value: f64,
    /// Deviation from expected
    pub deviation: f64,
    /// Additional notes
    pub notes: Option<String>,
}

/// Load patterns for stress testing
#[derive(Debug, Clone)]
pub enum LoadPattern {
    /// Constant load
    Constant(f64),
    /// Linear ramp
    LinearRamp {
        start: f64,
        end: f64,
        duration: Duration,
    },
    /// Exponential ramp
    ExponentialRamp {
        start: f64,
        end: f64,
        duration: Duration,
    },
    /// Spike pattern
    Spike {
        base_load: f64,
        spike_load: f64,
        spike_duration: Duration,
    },
    /// Cyclic pattern
    Cyclic {
        min_load: f64,
        max_load: f64,
        period: Duration,
    },
}

/// Size progression for stress testing
#[derive(Debug, Clone)]
pub enum SizeProgression {
    /// Linear progression
    Linear {
        start: usize,
        end: usize,
        step: usize,
    },
    /// Exponential progression
    Exponential {
        start: usize,
        end: usize,
        factor: f64,
    },
    /// Custom sequence
    Custom(Vec<usize>),
}

/// Platform types for testing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlatformType {
    /// Classical simulation
    Classical,
    /// D-Wave quantum annealer
    DWave,
    /// AWS Braket
    AWSBraket,
    /// Fujitsu Digital Annealer
    FujitsuDA,
    /// Custom platform
    Custom(String),
}

/// Platform availability status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlatformAvailability {
    /// Available for testing
    Available,
    /// Temporarily unavailable
    Unavailable,
    /// Under maintenance
    Maintenance,
    /// Requires credentials
    RequiresAuth,
}

/// Types of resources to monitor
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// CPU usage
    CPU,
    /// Memory usage
    Memory,
    /// Disk I/O
    DiskIO,
    /// Network I/O
    NetworkIO,
    /// GPU usage
    GPU,
    /// Custom resource
    Custom(String),
}

/// Types of properties for testing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyType {
    /// Correctness property
    Correctness,
    /// Performance property
    Performance,
    /// Safety property
    Safety,
    /// Liveness property
    Liveness,
    /// Consistency property
    Consistency,
}

/// Values for property evaluation
#[derive(Debug, Clone)]
pub enum PropertyValue {
    /// Boolean value
    Boolean(bool),
    /// Numeric value
    Numeric(f64),
    /// String value
    String(String),
    /// Vector value
    Vector(Vec<f64>),
}

/// Direction of performance trends
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Improving performance
    Improving,
    /// Degrading performance
    Degrading,
    /// Stable performance
    Stable,
    /// Volatile performance
    Volatile,
}

/// Types of test errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestErrorType {
    /// Assertion failure
    AssertionFailure,
    /// Runtime error
    RuntimeError,
    /// Timeout error
    TimeoutError,
    /// Resource error
    ResourceError,
    /// Configuration error
    ConfigurationError,
}

/// Types of failure patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailurePatternType {
    /// Temporal pattern
    Temporal,
    /// Conditional pattern
    Conditional,
    /// Sequential pattern
    Sequential,
    /// Correlation pattern
    Correlation,
}

/// Types of analytics engines
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalyticsEngineType {
    /// Statistical analysis
    Statistical,
    /// Machine learning
    MachineLearning,
    /// Pattern recognition
    PatternRecognition,
    /// Time series analysis
    TimeSeries,
}

/// Regression detection algorithm types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegressionAlgorithmType {
    /// Statistical process control
    StatisticalProcessControl,
    /// Change point detection
    ChangePointDetection,
    /// Time series analysis
    TimeSeriesAnalysis,
    /// Machine learning based
    MachineLearning,
    /// Anomaly detection
    AnomalyDetection,
}

/// Statistical model types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatisticalModelType {
    /// Linear regression
    LinearRegression,
    /// Moving average
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA model
    ARIMA,
    /// Gaussian process
    GaussianProcess,
}

/// Scalability analysis algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalabilityAlgorithm {
    /// Linear regression analysis
    LinearRegression,
    /// Power law fitting
    PowerLaw,
    /// Polynomial fitting
    Polynomial,
    /// Machine learning based
    MachineLearning,
}

/// Strategies for test case generation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationStrategy {
    /// Random generation
    Random,
    /// Exhaustive generation
    Exhaustive,
    /// Boundary value analysis
    BoundaryValue,
    /// Equivalence class partitioning
    EquivalenceClass,
    /// Pairwise testing
    Pairwise,
}

/// Scope of invariant application
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvariantScope {
    /// Global invariant
    Global,
    /// Local to function
    Local,
    /// Temporal invariant
    Temporal,
    /// Conditional invariant
    Conditional,
}

/// Types of pattern conditions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionType {
    /// Environment condition
    Environment,
    /// Configuration condition
    Configuration,
    /// Performance condition
    Performance,
    /// Temporal condition
    Temporal,
}

/// Operators for condition evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionOperator {
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Contains
    Contains,
    /// Matches pattern
    Matches,
}

/// Stress test criterion types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StressCriterionType {
    /// Throughput maintenance
    ThroughputMaintenance,
    /// Response time
    ResponseTime,
    /// Resource utilization
    ResourceUtilization,
    /// Error rate
    ErrorRate,
    /// Recovery time
    RecoveryTime,
}

/// Analytics output formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalyticsOutputFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// HTML report
    HTML,
    /// PDF report
    PDF,
    /// Custom format
    Custom(String),
}

/// Report formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportFormat {
    /// JSON format
    JSON,
    /// HTML format
    HTML,
    /// PDF format
    PDF,
    /// Markdown format
    Markdown,
}

/// Chart types for visualization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChartType {
    /// Line chart
    Line,
    /// Bar chart
    Bar,
    /// Scatter plot
    Scatter,
    /// Histogram
    Histogram,
    /// Heatmap
    Heatmap,
    /// Box plot
    BoxPlot,
}

/// Rendering engine types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderingEngineType {
    /// SVG rendering
    SVG,
    /// Canvas rendering
    Canvas,
    /// WebGL rendering
    WebGL,
    /// Server-side rendering
    ServerSide,
    /// Custom rendering
    Custom(String),
}
