//! Configuration structures for clustering analysis

use super::algorithms::{ClusteringAlgorithm, DistanceMetric};

/// Configuration for clustering analysis
#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    /// Clustering algorithm to use
    pub algorithm: ClusteringAlgorithm,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Feature extraction method
    pub feature_extraction: FeatureExtractionMethod,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Cache distance matrices
    pub cache_distances: bool,
    /// Analysis depth level
    pub analysis_depth: AnalysisDepth,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Visualization settings
    pub visualization: VisualizationConfig,
}

/// Feature extraction methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureExtractionMethod {
    /// Use raw solution vectors
    Raw,
    /// Use energy and basic statistics
    EnergyBased,
    /// Use structural features
    Structural,
    /// Use locality-sensitive hashing
    LSH { num_hashes: usize, num_bits: usize },
    /// Principal component analysis
    PCA { num_components: usize },
    /// Auto-encoder features
    AutoEncoder { hidden_layers: Vec<usize> },
    /// Custom feature extraction
    Custom { name: String },
}

/// Analysis depth levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisDepth {
    /// Basic clustering only
    Basic,
    /// Standard analysis with key metrics
    Standard,
    /// Comprehensive analysis with all features
    Comprehensive,
    /// Deep analysis with advanced techniques
    Deep,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Enable visualization output
    pub enabled: bool,
    /// Dimensionality reduction for visualization
    pub dimensionality_reduction: DimensionalityReduction,
    /// Plot types to generate
    pub plot_types: Vec<PlotType>,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Output format
    pub output_format: OutputFormat,
}

/// Dimensionality reduction methods for visualization
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionalityReduction {
    /// Principal Component Analysis
    PCA,
    /// t-Distributed Stochastic Neighbor Embedding
    TSNE { perplexity: f64 },
    /// Uniform Manifold Approximation and Projection
    UMAP { n_neighbors: usize, min_dist: f64 },
    /// Multi-dimensional Scaling
    MDS,
    /// Linear Discriminant Analysis
    LDA,
}

/// Plot types for visualization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlotType {
    /// Scatter plot of solutions
    ScatterPlot,
    /// Energy histogram
    EnergyHistogram,
    /// Cluster silhouette plot
    SilhouettePlot,
    /// Dendrogram for hierarchical clustering
    Dendrogram,
    /// Landscape heat map
    LandscapeHeatMap,
    /// Convergence trajectories
    ConvergenceTrajectories,
    /// Correlation matrix
    CorrelationMatrix,
}

/// Color schemes for visualization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorScheme {
    /// Default color scheme
    Default,
    /// Viridis color scheme
    Viridis,
    /// Plasma color scheme
    Plasma,
    /// Spectral color scheme
    Spectral,
    /// Custom color scheme
    Custom(Vec<String>),
}

/// Output formats for visualization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    /// PNG image
    PNG,
    /// SVG vector graphics
    SVG,
    /// PDF document
    PDF,
    /// HTML interactive plot
    HTML,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::KMeans {
                k: 5,
                max_iterations: 100,
            },
            distance_metric: DistanceMetric::Euclidean,
            feature_extraction: FeatureExtractionMethod::Raw,
            parallel_processing: true,
            cache_distances: true,
            analysis_depth: AnalysisDepth::Standard,
            seed: None,
            visualization: VisualizationConfig {
                enabled: true,
                dimensionality_reduction: DimensionalityReduction::PCA,
                plot_types: vec![PlotType::ScatterPlot, PlotType::EnergyHistogram],
                color_scheme: ColorScheme::Default,
                output_format: OutputFormat::PNG,
            },
        }
    }
}

/// Create a basic clustering configuration
#[must_use]
pub fn create_basic_clustering_config() -> ClusteringConfig {
    ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans {
            k: 5,
            max_iterations: 100,
        },
        distance_metric: DistanceMetric::Euclidean,
        feature_extraction: FeatureExtractionMethod::Raw,
        analysis_depth: AnalysisDepth::Basic,
        ..Default::default()
    }
}

/// Create a comprehensive clustering configuration
#[must_use]
pub fn create_comprehensive_clustering_config() -> ClusteringConfig {
    ClusteringConfig {
        algorithm: ClusteringAlgorithm::DBSCAN {
            eps: 0.5,
            min_samples: 5,
        },
        distance_metric: DistanceMetric::Euclidean,
        feature_extraction: FeatureExtractionMethod::Structural,
        analysis_depth: AnalysisDepth::Comprehensive,
        parallel_processing: true,
        cache_distances: true,
        visualization: VisualizationConfig {
            enabled: true,
            dimensionality_reduction: DimensionalityReduction::TSNE { perplexity: 30.0 },
            plot_types: vec![
                PlotType::ScatterPlot,
                PlotType::EnergyHistogram,
                PlotType::SilhouettePlot,
                PlotType::LandscapeHeatMap,
            ],
            color_scheme: ColorScheme::Viridis,
            output_format: OutputFormat::SVG,
        },
        ..Default::default()
    }
}
