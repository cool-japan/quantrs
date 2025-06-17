//! Performance analysis and bottleneck detection

use super::*;

/// Bottleneck analysis
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Resource utilization analysis
    pub resource_utilization: ResourceUtilizationAnalysis,
    /// Dependency analysis
    pub dependency_analysis: DependencyAnalysis,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Identified bottleneck
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Location
    pub location: String,
    /// Impact severity
    pub severity: f64,
    /// Resource affected
    pub resource: String,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Types of bottlenecks
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Algorithm,
    Synchronization,
    Custom { description: String },
}

/// Resource utilization analysis
#[derive(Debug, Clone)]
pub struct ResourceUtilizationAnalysis {
    /// CPU utilization breakdown
    pub cpu_breakdown: CpuUtilizationBreakdown,
    /// Memory utilization breakdown
    pub memory_breakdown: MemoryUtilizationBreakdown,
    /// IO utilization breakdown
    pub io_breakdown: IoUtilizationBreakdown,
    /// Network utilization breakdown
    pub network_breakdown: NetworkUtilizationBreakdown,
}

/// CPU utilization breakdown
#[derive(Debug, Clone)]
pub struct CpuUtilizationBreakdown {
    /// User time percentage
    pub user_time: f64,
    /// System time percentage
    pub system_time: f64,
    /// Idle time percentage
    pub idle_time: f64,
    /// Wait time percentage
    pub wait_time: f64,
    /// Per-core utilization
    pub per_core_utilization: Vec<f64>,
    /// Context switches per second
    pub context_switches: f64,
}

impl Default for CpuUtilizationBreakdown {
    fn default() -> Self {
        Self {
            user_time: 45.0,
            system_time: 15.0,
            idle_time: 35.0,
            wait_time: 5.0,
            per_core_utilization: vec![45.0, 48.0, 42.0, 50.0],
            context_switches: 1500.0,
        }
    }
}

/// Memory utilization breakdown
#[derive(Debug, Clone)]
pub struct MemoryUtilizationBreakdown {
    /// Used memory percentage
    pub used_memory: f64,
    /// Cached memory percentage
    pub cached_memory: f64,
    /// Buffer memory percentage
    pub buffer_memory: f64,
    /// Available memory percentage
    pub available_memory: f64,
    /// Memory allocation rate
    pub allocation_rate: f64,
    /// Garbage collection overhead
    pub gc_overhead: f64,
}

impl Default for MemoryUtilizationBreakdown {
    fn default() -> Self {
        Self {
            used_memory: 65.0,
            cached_memory: 20.0,
            buffer_memory: 5.0,
            available_memory: 10.0,
            allocation_rate: 1024.0,
            gc_overhead: 2.0,
        }
    }
}

/// IO utilization breakdown
#[derive(Debug, Clone)]
pub struct IoUtilizationBreakdown {
    /// Read operations per second
    pub read_ops: f64,
    /// Write operations per second
    pub write_ops: f64,
    /// Read throughput (MB/s)
    pub read_throughput: f64,
    /// Write throughput (MB/s)
    pub write_throughput: f64,
    /// IO wait time
    pub io_wait_time: f64,
    /// Queue depth
    pub queue_depth: f64,
}

impl Default for IoUtilizationBreakdown {
    fn default() -> Self {
        Self {
            read_ops: 500.0,
            write_ops: 200.0,
            read_throughput: 100.0,
            write_throughput: 50.0,
            io_wait_time: 2.5,
            queue_depth: 4.0,
        }
    }
}

/// Network utilization breakdown
#[derive(Debug, Clone)]
pub struct NetworkUtilizationBreakdown {
    /// Incoming bandwidth (Mbps)
    pub incoming_bandwidth: f64,
    /// Outgoing bandwidth (Mbps)
    pub outgoing_bandwidth: f64,
    /// Packet rate (packets/s)
    pub packet_rate: f64,
    /// Network latency (ms)
    pub latency: f64,
    /// Packet loss rate
    pub packet_loss: f64,
    /// Connection count
    pub connection_count: usize,
}

impl Default for NetworkUtilizationBreakdown {
    fn default() -> Self {
        Self {
            incoming_bandwidth: 150.0,
            outgoing_bandwidth: 75.0,
            packet_rate: 5000.0,
            latency: 1.2,
            packet_loss: 0.01,
            connection_count: 25,
        }
    }
}

/// Dependency analysis
#[derive(Debug, Clone)]
pub struct DependencyAnalysis {
    /// Critical path analysis
    pub critical_path: Vec<String>,
    /// Dependency graph
    pub dependency_graph: DependencyGraph,
    /// Parallelization opportunities
    pub parallelization_opportunities: Vec<ParallelizationOpportunity>,
    /// Serialization bottlenecks
    pub serialization_bottlenecks: Vec<String>,
}

/// Dependency graph
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Nodes (operations)
    pub nodes: Vec<DependencyNode>,
    /// Edges (dependencies)
    pub edges: Vec<DependencyEdge>,
    /// Graph properties
    pub properties: GraphProperties,
}

/// Dependency node
#[derive(Debug, Clone)]
pub struct DependencyNode {
    /// Node identifier
    pub id: String,
    /// Operation name
    pub operation: String,
    /// Execution time
    pub execution_time: Duration,
    /// Resource requirements
    pub resource_requirements: HashMap<String, f64>,
}

/// Dependency edge
#[derive(Debug, Clone)]
pub struct DependencyEdge {
    /// Source node
    pub source: String,
    /// Target node
    pub target: String,
    /// Dependency type
    pub dependency_type: DependencyType,
    /// Data transfer size
    pub data_size: usize,
}

/// Types of dependencies
#[derive(Debug, Clone, PartialEq)]
pub enum DependencyType {
    DataDependency,
    ControlDependency,
    ResourceDependency,
    SynchronizationDependency,
}

/// Graph properties
#[derive(Debug, Clone)]
pub struct GraphProperties {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Graph density
    pub density: f64,
    /// Average path length
    pub avg_path_length: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

impl Default for GraphProperties {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            density: 0.0,
            avg_path_length: 0.0,
            clustering_coefficient: 0.0,
        }
    }
}

/// Parallelization opportunity
#[derive(Debug, Clone)]
pub struct ParallelizationOpportunity {
    /// Operations that can be parallelized
    pub operations: Vec<String>,
    /// Potential speedup
    pub potential_speedup: f64,
    /// Parallelization strategy
    pub strategy: ParallelizationStrategy,
    /// Implementation complexity
    pub complexity: ComplexityLevel,
}

/// Parallelization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelizationStrategy {
    TaskParallelism,
    DataParallelism,
    PipelineParallelism,
    HybridParallelism,
}

/// Complexity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    /// Optimization type
    pub optimization_type: OptimizationType,
    /// Description
    pub description: String,
    /// Potential improvement
    pub potential_improvement: f64,
    /// Implementation effort
    pub implementation_effort: EffortLevel,
    /// Risk level
    pub risk_level: RiskLevel,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    AlgorithmOptimization,
    DataStructureOptimization,
    MemoryOptimization,
    CacheOptimization,
    ParallelizationOptimization,
    CompilerOptimization,
    HardwareOptimization,
}

/// Effort levels
#[derive(Debug, Clone, PartialEq)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Comparative analysis
#[derive(Debug, Clone)]
pub struct ComparativeAnalysis {
    /// Baseline comparison
    pub baseline_comparison: BaselineComparison,
    /// Algorithm comparisons
    pub algorithm_comparisons: Vec<AlgorithmComparison>,
    /// Performance regression analysis
    pub regression_analysis: RegressionAnalysis,
    /// A/B test results
    pub ab_test_results: Vec<ABTestResult>,
}

/// Baseline comparison
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Current performance
    pub current_performance: HashMap<String, f64>,
    /// Baseline performance
    pub baseline_performance: HashMap<String, f64>,
    /// Performance changes
    pub performance_changes: HashMap<String, f64>,
    /// Statistical significance
    pub statistical_significance: HashMap<String, bool>,
}

/// Algorithm comparison
#[derive(Debug, Clone)]
pub struct AlgorithmComparison {
    /// Algorithm names
    pub algorithms: Vec<String>,
    /// Performance metrics comparison
    pub performance_comparison: HashMap<String, Vec<f64>>,
    /// Statistical tests
    pub statistical_tests: Vec<HypothesisTestResult>,
    /// Recommendation
    pub recommendation: String,
}

/// Regression analysis
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    /// Performance regression detected
    pub regression_detected: bool,
    /// Regression severity
    pub regression_severity: f64,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Potential causes
    pub potential_causes: Vec<String>,
    /// Timeline analysis
    pub timeline_analysis: TimelineAnalysis,
}

/// Timeline analysis
#[derive(Debug, Clone)]
pub struct TimelineAnalysis {
    /// Key events
    pub key_events: Vec<TimelineEvent>,
    /// Performance correlations
    pub correlations: Vec<PerformanceCorrelation>,
    /// Change point detection
    pub change_points: Vec<ChangePoint>,
}

/// Timeline event
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Event description
    pub description: String,
    /// Event type
    pub event_type: EventType,
    /// Impact assessment
    pub impact: f64,
}

/// Event types
#[derive(Debug, Clone, PartialEq)]
pub enum EventType {
    CodeChange,
    ConfigurationChange,
    HardwareChange,
    EnvironmentChange,
    DataChange,
    External,
}

/// Performance correlation
#[derive(Debug, Clone)]
pub struct PerformanceCorrelation {
    /// Metric 1
    pub metric1: String,
    /// Metric 2
    pub metric2: String,
    /// Correlation coefficient
    pub correlation: f64,
    /// P-value
    pub p_value: f64,
    /// Correlation type
    pub correlation_type: CorrelationType,
}

/// Correlation types
#[derive(Debug, Clone, PartialEq)]
pub enum CorrelationType {
    Positive,
    Negative,
    NonLinear,
    Spurious,
}

/// Change point
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Change point timestamp
    pub timestamp: Instant,
    /// Affected metric
    pub metric: String,
    /// Change magnitude
    pub magnitude: f64,
    /// Confidence level
    pub confidence: f64,
    /// Change type
    pub change_type: ChangeType,
}

/// Types of changes
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeType {
    LevelShift,
    TrendChange,
    VarianceChange,
    DistributionChange,
}

/// A/B test result
#[derive(Debug, Clone)]
pub struct ABTestResult {
    /// Test name
    pub test_name: String,
    /// Variant A results
    pub variant_a: TestVariantResult,
    /// Variant B results
    pub variant_b: TestVariantResult,
    /// Statistical significance
    pub statistical_significance: bool,
    /// Effect size
    pub effect_size: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Recommendation
    pub recommendation: String,
}

/// Test variant result
#[derive(Debug, Clone)]
pub struct TestVariantResult {
    /// Sample size
    pub sample_size: usize,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Standard deviations
    pub std_devs: HashMap<String, f64>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}
