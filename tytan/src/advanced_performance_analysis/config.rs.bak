//! Configuration for performance analysis

use super::*;

/// Configuration for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Monitoring frequency (Hz)
    pub monitoring_frequency: f64,
    /// Metrics collection level
    pub collection_level: MetricsLevel,
    /// Analysis depth
    pub analysis_depth: AnalysisDepth,
    /// Enable comparative analysis
    pub comparative_analysis: bool,
    /// Enable performance prediction
    pub performance_prediction: bool,
    /// Statistical analysis settings
    pub statistical_analysis: StatisticalAnalysisConfig,
    /// Visualization settings
    pub visualization: VisualizationConfig,
}

/// Levels of metrics collection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricsLevel {
    /// Basic metrics only
    Basic,
    /// Detailed metrics
    Detailed,
    /// Comprehensive metrics with overhead
    Comprehensive,
    /// Custom metric selection
    Custom { metrics: Vec<String> },
}

/// Analysis depth levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Surface-level analysis
    Surface,
    /// Deep analysis with statistical tests
    Deep,
    /// Exhaustive analysis with ML models
    Exhaustive,
    /// Real-time adaptive analysis
    Adaptive,
}

/// Statistical analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisConfig {
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Number of bootstrap samples
    pub bootstrap_samples: usize,
    /// Enable hypothesis testing
    pub hypothesis_testing: bool,
    /// Significance level
    pub significance_level: f64,
    /// Enable outlier detection
    pub outlier_detection: bool,
    /// Outlier detection method
    pub outlier_method: OutlierDetectionMethod,
}

/// Outlier detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutlierDetectionMethod {
    /// Z-score based
    ZScore { threshold: f64 },
    /// Interquartile range
    IQR { multiplier: f64 },
    /// Isolation forest
    IsolationForest,
    /// Local outlier factor
    LocalOutlierFactor,
    /// Statistical tests
    StatisticalTests,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Enable real-time plots
    pub real_time_plots: bool,
    /// Plot update frequency
    pub plot_update_frequency: f64,
    /// Export formats
    pub export_formats: Vec<ExportFormat>,
    /// Dashboard settings
    pub dashboard: DashboardConfig,
}

/// Export formats for results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExportFormat {
    CSV,
    JSON,
    PNG,
    SVG,
    PDF,
    HTML,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Enable web dashboard
    pub enable_web_dashboard: bool,
    /// Dashboard port
    pub port: u16,
    /// Update interval (seconds)
    pub update_interval: f64,
    /// Enable alerts
    pub enable_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

/// Create default analysis configuration
pub fn create_default_analysis_config() -> AnalysisConfig {
    AnalysisConfig {
        real_time_monitoring: true,
        monitoring_frequency: 1.0, // 1 Hz
        collection_level: MetricsLevel::Detailed,
        analysis_depth: AnalysisDepth::Deep,
        comparative_analysis: true,
        performance_prediction: true,
        statistical_analysis: StatisticalAnalysisConfig {
            confidence_level: 0.95,
            bootstrap_samples: 1000,
            hypothesis_testing: true,
            significance_level: 0.05,
            outlier_detection: true,
            outlier_method: OutlierDetectionMethod::IQR { multiplier: 1.5 },
        },
        visualization: VisualizationConfig {
            real_time_plots: true,
            plot_update_frequency: 0.5, // 0.5 Hz
            export_formats: vec![ExportFormat::PNG, ExportFormat::CSV, ExportFormat::HTML],
            dashboard: DashboardConfig {
                enable_web_dashboard: true,
                port: 8080,
                update_interval: 2.0, // 2 seconds
                enable_alerts: true,
                alert_thresholds: {
                    let mut thresholds = HashMap::new();
                    thresholds.insert("cpu_utilization".to_string(), 80.0);
                    thresholds.insert("memory_utilization".to_string(), 85.0);
                    thresholds.insert("io_utilization".to_string(), 90.0);
                    thresholds
                },
            },
        },
    }
}

/// Create lightweight configuration for basic monitoring
pub fn create_lightweight_config() -> AnalysisConfig {
    let mut config = create_default_analysis_config();
    config.collection_level = MetricsLevel::Basic;
    config.analysis_depth = AnalysisDepth::Surface;
    config.comparative_analysis = false;
    config.performance_prediction = false;
    config
}
