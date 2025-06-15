//! Type definitions for performance analysis

use super::*;

/// Analysis errors
#[derive(Debug, Clone)]
pub enum AnalysisError {
    /// Configuration error
    ConfigurationError(String),
    /// Data collection error
    DataCollectionError(String),
    /// Analysis computation error
    ComputationError(String),
    /// Insufficient data
    InsufficientData(String),
    /// Model training error
    ModelTrainingError(String),
    /// Report generation error
    ReportGenerationError(String),
    /// System error
    SystemError(String),
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            AnalysisError::DataCollectionError(msg) => write!(f, "Data collection error: {}", msg),
            AnalysisError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            AnalysisError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            AnalysisError::ModelTrainingError(msg) => write!(f, "Model training error: {}", msg),
            AnalysisError::ReportGenerationError(msg) => {
                write!(f, "Report generation error: {}", msg)
            }
            AnalysisError::SystemError(msg) => write!(f, "System error: {}", msg),
        }
    }
}

impl std::error::Error for AnalysisError {}

/// Performance metrics database
#[derive(Debug)]
pub struct MetricsDatabase {
    /// Time series data
    pub time_series: HashMap<String, TimeSeries>,
    /// Aggregated metrics
    pub aggregated_metrics: HashMap<String, AggregatedMetric>,
    /// Historical data
    pub historical_data: HistoricalData,
    /// Metadata
    pub metadata: MetricsMetadata,
}

/// Time series data structure
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Timestamps
    pub timestamps: Vec<Instant>,
    /// Values
    pub values: Vec<f64>,
    /// Metric name
    pub metric_name: String,
    /// Units
    pub units: String,
    /// Sampling rate
    pub sampling_rate: f64,
}

/// Aggregated metric
#[derive(Debug, Clone)]
pub struct AggregatedMetric {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// Percentiles
    pub percentiles: HashMap<u8, f64>,
    /// Number of samples
    pub sample_count: usize,
    /// Total duration
    pub duration: Duration,
}

/// Historical performance data
#[derive(Debug, Clone)]
pub struct HistoricalData {
    /// Daily summaries
    pub daily_summaries: Vec<DailySummary>,
    /// Trend analysis
    pub trends: TrendAnalysis,
    /// Performance baselines
    pub baselines: HashMap<String, Baseline>,
    /// Regression models
    pub regression_models: Vec<RegressionModel>,
}

/// Daily performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailySummary {
    /// Date
    pub date: String,
    /// Key performance indicators
    pub kpis: HashMap<String, f64>,
    /// Problem statistics
    pub problem_stats: ProblemStatistics,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Problem statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemStatistics {
    /// Number of problems solved
    pub problems_solved: usize,
    /// Average problem size
    pub avg_problem_size: f64,
    /// Problem size distribution
    pub size_distribution: HashMap<String, usize>,
    /// Problem types
    pub problem_types: HashMap<String, usize>,
    /// Success rate
    pub success_rate: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization (%)
    pub cpu_utilization: f64,
    /// Memory utilization (%)
    pub memory_utilization: f64,
    /// GPU utilization (%)
    pub gpu_utilization: Option<f64>,
    /// Network utilization (%)
    pub network_utilization: f64,
    /// Storage I/O (MB/s)
    pub storage_io: f64,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Solution quality score
    pub solution_quality: f64,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Stability index
    pub stability_index: f64,
    /// Reproducibility score
    pub reproducibility: f64,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
}

/// Trend analysis results
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Performance trends
    pub performance_trends: HashMap<String, TrendDirection>,
    /// Seasonal patterns
    pub seasonal_patterns: Vec<SeasonalPattern>,
    /// Anomaly detection results
    pub anomalies: Vec<Anomaly>,
    /// Forecasts
    pub forecasts: HashMap<String, Forecast>,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Oscillating,
    Unknown,
}

/// Seasonal pattern
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Period (in measurements)
    pub period: usize,
    /// Amplitude
    pub amplitude: f64,
    /// Phase shift
    pub phase_shift: f64,
    /// Confidence
    pub confidence: f64,
}

/// Types of seasonal patterns
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Daily,
    Weekly,
    Monthly,
    Custom { period_name: String },
}

/// Detected anomaly
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Timestamp
    pub timestamp: Instant,
    /// Metric affected
    pub metric: String,
    /// Anomaly value
    pub value: f64,
    /// Expected value
    pub expected_value: f64,
    /// Severity score
    pub severity: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Description
    pub description: String,
}

/// Types of anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    Spike,
    Drop,
    Drift,
    Oscillation,
    Discontinuity,
}

/// Performance forecast
#[derive(Debug, Clone)]
pub struct Forecast {
    /// Forecasted values
    pub values: Vec<f64>,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Forecast horizon
    pub horizon: Duration,
    /// Model used
    pub model_type: String,
    /// Accuracy metrics
    pub accuracy: ForecastAccuracy,
}

/// Forecast accuracy metrics
#[derive(Debug, Clone)]
pub struct ForecastAccuracy {
    /// Mean absolute error
    pub mae: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Mean absolute percentage error
    pub mape: f64,
    /// R-squared
    pub r_squared: f64,
}

/// Performance baseline
#[derive(Debug, Clone)]
pub struct Baseline {
    /// Baseline value
    pub value: f64,
    /// Tolerance range
    pub tolerance: f64,
    /// Measurement timestamp
    pub timestamp: Instant,
    /// Conditions when measured
    pub conditions: HashMap<String, String>,
    /// Confidence level
    pub confidence: f64,
}

/// Regression model for trend analysis
#[derive(Debug, Clone)]
pub struct RegressionModel {
    /// Model type
    pub model_type: RegressionType,
    /// Coefficients
    pub coefficients: Vec<f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Standard error
    pub standard_error: f64,
    /// Feature names
    pub features: Vec<String>,
}

/// Types of regression models
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionType {
    Linear,
    Polynomial { degree: usize },
    Exponential,
    Logarithmic,
    PowerLaw,
}

/// Metrics metadata
#[derive(Debug, Clone)]
pub struct MetricsMetadata {
    /// Collection start time
    pub collection_start: Instant,
    /// System information
    pub system_info: SystemInfo,
    /// Software versions
    pub software_versions: HashMap<String, String>,
    /// Configuration hash
    pub config_hash: String,
}

/// System information
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// CPU information
    pub cpu: CpuInfo,
    /// Memory information
    pub memory: MemoryInfo,
    /// GPU information
    pub gpu: Option<GpuInfo>,
    /// Network information
    pub network: NetworkInfo,
}

/// CPU information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Model name
    pub model: String,
    /// Number of cores
    pub cores: usize,
    /// Base frequency (GHz)
    pub base_frequency: f64,
    /// Cache sizes (KB)
    pub cache_sizes: Vec<usize>,
    /// Architecture
    pub architecture: String,
}

/// Memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total memory (GB)
    pub total_memory: f64,
    /// Memory type (DDR4, DDR5, etc.)
    pub memory_type: String,
    /// Memory speed (MHz)
    pub memory_speed: f64,
    /// Number of channels
    pub channels: usize,
}

/// GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU model
    pub model: String,
    /// VRAM size (GB)
    pub vram: f64,
    /// CUDA cores / Stream processors
    pub cores: usize,
    /// Base clock (MHz)
    pub base_clock: f64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
}

/// Network information
#[derive(Debug, Clone)]
pub struct NetworkInfo {
    /// Network interfaces
    pub interfaces: Vec<NetworkInterface>,
    /// Latency measurements
    pub latency_measurements: HashMap<String, f64>,
    /// Bandwidth measurements
    pub bandwidth_measurements: HashMap<String, f64>,
}

/// Network interface
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    /// Interface name
    pub name: String,
    /// Interface type
    pub interface_type: String,
    /// Maximum speed (Gbps)
    pub max_speed: f64,
    /// Current utilization (%)
    pub utilization: f64,
}

/// System info collection
impl SystemInfo {
    pub fn collect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            cpu: CpuInfo {
                model: "Mock CPU".to_string(),
                cores: 8,
                base_frequency: 3.2,
                cache_sizes: vec![32, 256, 8192],
                architecture: std::env::consts::ARCH.to_string(),
            },
            memory: MemoryInfo {
                total_memory: 16.0,
                memory_type: "DDR4".to_string(),
                memory_speed: 3200.0,
                channels: 2,
            },
            gpu: Some(GpuInfo {
                model: "Mock GPU".to_string(),
                vram: 8.0,
                cores: 2048,
                base_clock: 1500.0,
                memory_bandwidth: 448.0,
            }),
            network: NetworkInfo {
                interfaces: vec![NetworkInterface {
                    name: "eth0".to_string(),
                    interface_type: "Ethernet".to_string(),
                    max_speed: 1.0,
                    utilization: 15.5,
                }],
                latency_measurements: HashMap::new(),
                bandwidth_measurements: HashMap::new(),
            },
        }
    }
}
