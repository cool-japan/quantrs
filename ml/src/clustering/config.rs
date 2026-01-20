//! Configuration types for quantum clustering

use crate::dimensionality_reduction::{QuantumDistanceMetric, QuantumEnhancementLevel};

/// Quantum clustering algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusteringAlgorithm {
    /// Quantum K-Means
    QuantumKMeans,
    /// Quantum Hierarchical Clustering
    QuantumHierarchical,
    /// Quantum DBSCAN
    QuantumDBSCAN,
    /// Quantum Spectral Clustering
    QuantumSpectral,
    /// Quantum Fuzzy C-Means
    QuantumFuzzyCMeans,
    /// Quantum Gaussian Mixture Models
    QuantumGMM,
    /// Quantum Mean-Shift
    QuantumMeanShift,
    /// Quantum Affinity Propagation
    QuantumAffinityPropagation,
}

/// Quantum clustering configuration
#[derive(Debug, Clone)]
pub struct QuantumClusteringConfig {
    /// Algorithm to use
    pub algorithm: ClusteringAlgorithm,
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of qubits for quantum operations
    pub num_qubits: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for QuantumClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::QuantumKMeans,
            n_clusters: 3,
            max_iterations: 100,
            tolerance: 1e-4,
            num_qubits: 4,
            random_state: None,
        }
    }
}

/// Affinity types for spectral clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AffinityType {
    /// RBF (Gaussian) kernel
    RBF,
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial,
    /// Quantum kernel
    QuantumKernel,
    /// Nearest neighbors
    NearestNeighbors,
}

/// Covariance types for Gaussian Mixture Models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovarianceType {
    /// Full covariance matrices
    Full,
    /// Diagonal covariance matrices
    Diagonal,
    /// Tied covariance matrix
    Tied,
    /// Spherical covariance
    Spherical,
    /// Quantum-enhanced covariance
    QuantumEnhanced,
}

/// Ensemble combination methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnsembleCombinationMethod {
    /// Majority voting
    MajorityVoting,
    /// Weighted voting
    WeightedVoting,
    /// Consensus clustering
    ConsensusClustering,
    /// Quantum consensus
    QuantumConsensus,
}

/// Measurement strategies for quantum operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementStrategy {
    /// Standard quantum measurements
    Standard,
    /// Adaptive measurements
    AdaptiveMeasurements,
    /// Optimal measurements
    OptimalMeasurements,
}

/// State preparation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatePreparationMethod {
    /// Angle encoding
    AngleEncoding,
    /// Amplitude encoding
    AmplitudeEncoding,
    /// Variational state preparation
    VariationalStatePreparation,
}

/// Entanglement structures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntanglementStructure {
    /// Linear entanglement
    Linear,
    /// Circular entanglement
    Circular,
    /// Hardware efficient entanglement
    HardwareEfficient,
    /// All-to-all entanglement
    AllToAll,
}

/// Graph methods for graph clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphMethod {
    /// Standard graph construction
    Standard,
    /// Quantum graph construction
    QuantumGraph,
    /// k-nearest neighbors graph
    KNearestNeighbors,
    /// Epsilon neighborhood graph
    EpsilonNeighborhood,
}

/// Community detection algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommunityAlgorithm {
    /// Modularity-based detection
    Modularity,
    /// Quantum community detection
    QuantumCommunityDetection,
    /// Louvain algorithm
    Louvain,
    /// Label propagation
    LabelPropagation,
}

/// Time series distance metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeSeriesDistanceMetric {
    /// Dynamic time warping
    DTW,
    /// Euclidean distance
    Euclidean,
    /// Quantum temporal distance
    QuantumTemporal,
}

/// Dimensionality reduction methods for high-dimensional clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DimensionalityReduction {
    /// Principal Component Analysis
    PCA,
    /// Quantum PCA
    QuantumPCA,
    /// t-SNE
    TSNE,
    /// UMAP
    UMAP,
}

/// Configuration for Quantum K-Means clustering
#[derive(Debug, Clone)]
pub struct QuantumKMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Distance metric to use
    pub distance_metric: QuantumDistanceMetric,
    /// Number of quantum repetitions
    pub quantum_reps: usize,
    /// Quantum enhancement level
    pub enhancement_level: QuantumEnhancementLevel,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for Quantum DBSCAN clustering
#[derive(Debug, Clone)]
pub struct QuantumDBSCANConfig {
    /// Epsilon neighborhood radius
    pub eps: f64,
    /// Minimum samples in neighborhood
    pub min_samples: usize,
    /// Distance metric to use
    pub distance_metric: QuantumDistanceMetric,
    /// Quantum enhancement level
    pub enhancement_level: QuantumEnhancementLevel,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for Quantum Spectral clustering
#[derive(Debug, Clone)]
pub struct QuantumSpectralConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Affinity type
    pub affinity: AffinityType,
    /// Gamma parameter for RBF kernel
    pub gamma: f64,
    /// Quantum enhancement level
    pub enhancement_level: QuantumEnhancementLevel,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for Quantum Fuzzy C-Means clustering
#[derive(Debug, Clone)]
pub struct QuantumFuzzyCMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Fuzziness parameter
    pub fuzziness: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Distance metric to use
    pub distance_metric: QuantumDistanceMetric,
    /// Quantum enhancement level
    pub enhancement_level: QuantumEnhancementLevel,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for Quantum Gaussian Mixture Models
#[derive(Debug, Clone)]
pub struct QuantumGMMConfig {
    /// Number of components
    pub n_components: usize,
    /// Covariance type
    pub covariance_type: CovarianceType,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Quantum enhancement level
    pub enhancement_level: QuantumEnhancementLevel,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for clustering ensembles
#[derive(Debug, Clone)]
pub struct ClusteringEnsembleConfig {
    /// Base clustering algorithms
    pub base_algorithms: Vec<ClusteringAlgorithm>,
    /// Number of ensemble members
    pub n_members: usize,
    /// Combination method
    pub combination_method: EnsembleCombinationMethod,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for graph clustering
#[derive(Debug, Clone)]
pub struct GraphClusteringConfig {
    /// Graph construction method
    pub graph_method: GraphMethod,
    /// Community detection algorithm
    pub community_algorithm: CommunityAlgorithm,
    /// Number of neighbors for graph construction
    pub n_neighbors: usize,
    /// Quantum enhancement level
    pub enhancement_level: QuantumEnhancementLevel,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for time series clustering
#[derive(Debug, Clone)]
pub struct TimeSeriesClusteringConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Time series distance metric
    pub ts_distance_metric: TimeSeriesDistanceMetric,
    /// Window size for analysis
    pub window_size: usize,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for high-dimensional clustering
#[derive(Debug, Clone)]
pub struct HighDimClusteringConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Dimensionality reduction method
    pub dim_reduction: DimensionalityReduction,
    /// Target dimensionality after reduction
    pub target_dim: usize,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for streaming clustering
#[derive(Debug, Clone)]
pub struct StreamingClusteringConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Memory size for streaming
    pub memory_size: usize,
    /// Forgetting factor
    pub forgetting_factor: f64,
    /// Random seed
    pub seed: Option<u64>,
}

/// Configuration for quantum-native clustering
#[derive(Debug, Clone)]
pub struct QuantumNativeConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// State preparation method
    pub state_preparation: StatePreparationMethod,
    /// Measurement strategy
    pub measurement_strategy: MeasurementStrategy,
    /// Entanglement structure
    pub entanglement_structure: EntanglementStructure,
    /// Random seed
    pub seed: Option<u64>,
}

// Default implementations
impl Default for QuantumKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            max_iterations: 100,
            tolerance: 1e-4,
            distance_metric: QuantumDistanceMetric::QuantumEuclidean,
            quantum_reps: 2,
            enhancement_level: QuantumEnhancementLevel::Moderate,
            seed: None,
        }
    }
}

impl Default for QuantumDBSCANConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
            distance_metric: QuantumDistanceMetric::QuantumEuclidean,
            enhancement_level: QuantumEnhancementLevel::Moderate,
            seed: None,
        }
    }
}

impl Default for QuantumSpectralConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            affinity: AffinityType::RBF,
            gamma: 1.0,
            enhancement_level: QuantumEnhancementLevel::Moderate,
            seed: None,
        }
    }
}

impl Default for QuantumFuzzyCMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            fuzziness: 2.0,
            max_iterations: 100,
            tolerance: 1e-4,
            distance_metric: QuantumDistanceMetric::QuantumEuclidean,
            enhancement_level: QuantumEnhancementLevel::Moderate,
            seed: None,
        }
    }
}

impl Default for QuantumGMMConfig {
    fn default() -> Self {
        Self {
            n_components: 3,
            covariance_type: CovarianceType::Full,
            max_iterations: 100,
            tolerance: 1e-4,
            enhancement_level: QuantumEnhancementLevel::Moderate,
            seed: None,
        }
    }
}
