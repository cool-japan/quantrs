//! Feature Extraction Configuration Types

/// Feature extraction configuration
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    /// Enable automatic feature extraction
    pub enable_auto_extraction: bool,
    /// Circuit structure features
    pub circuit_features: CircuitFeatureConfig,
    /// Hardware-specific features
    pub hardware_features: HardwareFeatureConfig,
    /// Temporal features
    pub temporal_features: TemporalFeatureConfig,
    /// Statistical features
    pub statistical_features: StatisticalFeatureConfig,
    /// Graph-based features
    pub graph_features: GraphFeatureConfig,
    /// Feature selection methods
    pub feature_selection: FeatureSelectionConfig,
    /// Dimensionality reduction
    pub dimensionality_reduction: DimensionalityReductionConfig,
}

/// Circuit feature configuration
#[derive(Debug, Clone)]
pub struct CircuitFeatureConfig {
    /// Basic circuit properties
    pub basic_properties: bool,
    /// Gate type distributions
    pub gate_distributions: bool,
    /// Circuit depth analysis
    pub depth_analysis: bool,
    /// Connectivity patterns
    pub connectivity_patterns: bool,
    /// Entanglement measures
    pub entanglement_measures: bool,
    /// Symmetry analysis
    pub symmetry_analysis: bool,
    /// Critical path analysis
    pub critical_path_analysis: bool,
}

/// Hardware feature configuration
#[derive(Debug, Clone)]
pub struct HardwareFeatureConfig {
    /// Topology features
    pub topology_features: bool,
    /// Calibration features
    pub calibration_features: bool,
    /// Error rate features
    pub error_rate_features: bool,
    /// Timing features
    pub timing_features: bool,
    /// Resource utilization features
    pub resource_features: bool,
    /// Temperature and drift features
    pub environmental_features: bool,
}

/// Temporal feature configuration
#[derive(Debug, Clone)]
pub struct TemporalFeatureConfig {
    /// Time series analysis
    pub time_series_analysis: bool,
    /// Trend detection
    pub trend_detection: bool,
    /// Seasonality analysis
    pub seasonality_analysis: bool,
    /// Autocorrelation features
    pub autocorrelation_features: bool,
    /// Fourier features
    pub fourier_features: bool,
}

/// Statistical feature configuration
#[derive(Debug, Clone)]
pub struct StatisticalFeatureConfig {
    /// Moment features
    pub moment_features: bool,
    /// Distribution fitting
    pub distribution_fitting: bool,
    /// Correlation features
    pub correlation_features: bool,
    /// Outlier detection features
    pub outlier_features: bool,
    /// Normality tests
    pub normality_tests: bool,
}

/// Graph feature configuration
#[derive(Debug, Clone)]
pub struct GraphFeatureConfig {
    /// Centrality measures
    pub centrality_measures: bool,
    /// Community detection features
    pub community_features: bool,
    /// Spectral features
    pub spectral_features: bool,
    /// Path-based features
    pub path_features: bool,
    /// Clustering features
    pub clustering_features: bool,
}

/// Feature selection configuration
#[derive(Debug, Clone)]
pub struct FeatureSelectionConfig {
    /// Enable feature selection
    pub enable_selection: bool,
    /// Selection methods
    pub selection_methods: Vec<FeatureSelectionMethod>,
    /// Number of features to select
    pub num_features: Option<usize>,
    /// Selection threshold
    pub selection_threshold: f64,
}

/// Feature selection methods
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureSelectionMethod {
    VarianceThreshold,
    UnivariateSelection,
    RecursiveFeatureElimination,
    FeatureImportance,
    MutualInformation,
    CorrelationFilter,
    LassoSelection,
}

/// Dimensionality reduction configuration
#[derive(Debug, Clone)]
pub struct DimensionalityReductionConfig {
    /// Enable dimensionality reduction
    pub enable_reduction: bool,
    /// Reduction methods
    pub reduction_methods: Vec<DimensionalityReductionMethod>,
    /// Target dimensionality
    pub target_dimensions: Option<usize>,
    /// Variance threshold
    pub variance_threshold: f64,
}

/// Dimensionality reduction methods
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionalityReductionMethod {
    PCA,
    ICA,
    LDA,
    TSNE,
    UMAP,
    AutoEncoder,
    VariationalAutoEncoder,
}
