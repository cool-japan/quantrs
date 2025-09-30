//! Configuration types for quantum dimensionality reduction

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Quantum dimensionality reduction algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DimensionalityReductionAlgorithm {
    /// Quantum Principal Component Analysis
    QPCA,
    /// Quantum Independent Component Analysis
    QICA,
    /// Quantum t-distributed Stochastic Neighbor Embedding
    QtSNE,
    /// Quantum Uniform Manifold Approximation and Projection
    QUMAP,
    /// Quantum Linear Discriminant Analysis
    QLDA,
    /// Quantum Factor Analysis
    QFactorAnalysis,
    /// Quantum Canonical Correlation Analysis
    QCCA,
    /// Quantum Non-negative Matrix Factorization
    QNMF,
    /// Quantum Variational Autoencoder
    QVAE,
    /// Quantum Denoising Autoencoder
    QDenoisingAE,
    /// Quantum Sparse Autoencoder
    QSparseAE,
    /// Quantum Manifold Learning
    QManifoldLearning,
    /// Quantum Kernel PCA
    QKernelPCA,
    /// Quantum Multidimensional Scaling
    QMDS,
    /// Quantum Isomap
    QIsomap,
    /// Quantum Mutual Information Selection
    QMutualInfoSelection,
    /// Quantum Recursive Feature Elimination
    QRFE,
    /// Quantum LASSO
    QLASSO,
    /// Quantum Ridge Regression
    QRidge,
    /// Quantum Variance Thresholding
    QVarianceThresholding,
    /// Quantum Time Series Dimensionality Reduction
    QTimeSeriesDR,
    /// Quantum Image/Tensor Dimensionality Reduction
    QImageTensorDR,
    /// Quantum Graph Dimensionality Reduction
    QGraphDR,
    /// Quantum Streaming Dimensionality Reduction
    QStreamingDR,
}

/// Quantum distance metrics for dimensionality reduction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumDistanceMetric {
    /// Quantum Euclidean distance
    QuantumEuclidean,
    /// Quantum Manhattan distance
    QuantumManhattan,
    /// Quantum cosine similarity
    QuantumCosine,
    /// Quantum fidelity-based distance
    QuantumFidelity,
    /// Quantum trace distance
    QuantumTrace,
    /// Quantum Wasserstein distance
    QuantumWasserstein,
    /// Quantum kernel-based distance
    QuantumKernel,
    /// Quantum entanglement-based distance
    QuantumEntanglement,
}

/// Quantum enhancement levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumEnhancementLevel {
    /// No quantum enhancement (classical)
    Classical,
    /// Light quantum enhancement
    Light,
    /// Moderate quantum enhancement
    Moderate,
    /// Full quantum enhancement
    Full,
    /// Experimental quantum features
    Experimental,
}

/// Eigensolvers for quantum algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumEigensolver {
    /// Variational Quantum Eigensolver
    VQE,
    /// Quantum Approximate Optimization Algorithm
    QAOA,
    /// Quantum Phase Estimation
    QPE,
    /// Quantum Lanczos Algorithm
    QuantumLanczos,
    /// Quantum Power Method
    QuantumPowerMethod,
}

/// Feature map types for quantum kernel methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumFeatureMap {
    /// Z-feature map
    ZFeatureMap,
    /// ZZ-feature map
    ZZFeatureMap,
    /// Pauli feature map
    PauliFeatureMap,
    /// Custom parameterized feature map
    CustomFeatureMap,
}

/// Autoencoder architectures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoencoderArchitecture {
    /// Standard variational autoencoder
    Standard,
    /// Beta-VAE with controlled disentanglement
    BetaVAE,
    /// WAE (Wasserstein autoencoder)
    WAE,
    /// InfoVAE
    InfoVAE,
    /// Adversarial autoencoder
    AdversarialAE,
}

/// Configuration for Quantum Principal Component Analysis
#[derive(Debug, Clone)]
pub struct QPCAConfig {
    /// Number of components to keep
    pub n_components: usize,
    /// Quantum eigensolver to use
    pub eigensolver: QuantumEigensolver,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits for quantum computation
    pub num_qubits: usize,
    /// Whether to whiten the components
    pub whiten: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Convergence tolerance for eigensolvers
    pub tolerance: f64,
    /// Maximum iterations for iterative eigensolvers
    pub max_iterations: usize,
}

/// Configuration for Quantum Independent Component Analysis
#[derive(Debug, Clone)]
pub struct QICAConfig {
    /// Number of components to extract
    pub n_components: usize,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits for quantum computation
    pub num_qubits: usize,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Non-linearity function type
    pub nonlinearity: String,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Configuration for Quantum t-SNE
#[derive(Debug, Clone)]
pub struct QtSNEConfig {
    /// Number of components in the embedded space
    pub n_components: usize,
    /// Perplexity parameter
    pub perplexity: f64,
    /// Early exaggeration factor
    pub early_exaggeration: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits for quantum computation
    pub num_qubits: usize,
    /// Distance metric to use
    pub distance_metric: QuantumDistanceMetric,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Configuration for Quantum UMAP
#[derive(Debug, Clone)]
pub struct QUMAPConfig {
    /// Number of components in the embedded space
    pub n_components: usize,
    /// Number of neighbors to consider
    pub n_neighbors: usize,
    /// Minimum distance in embedded space
    pub min_dist: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs for optimization
    pub n_epochs: usize,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits for quantum computation
    pub num_qubits: usize,
    /// Distance metric to use
    pub distance_metric: QuantumDistanceMetric,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Configuration for Quantum Linear Discriminant Analysis
#[derive(Debug, Clone)]
pub struct QLDAConfig {
    /// Number of components to keep
    pub n_components: Option<usize>,
    /// Shrinkage parameter for regularization
    pub shrinkage: Option<f64>,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits for quantum computation
    pub num_qubits: usize,
    /// Solver for eigenvalue problems
    pub solver: QuantumEigensolver,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Configuration for Quantum Autoencoder
#[derive(Debug, Clone)]
pub struct QAutoencoderConfig {
    /// Encoder layer dimensions
    pub encoder_layers: Vec<usize>,
    /// Decoder layer dimensions
    pub decoder_layers: Vec<usize>,
    /// Latent dimension
    pub latent_dim: usize,
    /// Autoencoder architecture
    pub architecture: AutoencoderArchitecture,
    /// Learning rate
    pub learning_rate: f64,
    /// Training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits for quantum computation
    pub num_qubits: usize,
    /// Beta parameter for Beta-VAE
    pub beta: f64,
    /// Noise level for denoising autoencoder
    pub noise_level: f64,
    /// Sparsity parameter for sparse autoencoder
    pub sparsity_parameter: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Configuration for Quantum Factor Analysis
#[derive(Debug, Clone)]
pub struct QFactorAnalysisConfig {
    /// Number of factors
    pub n_factors: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits
    pub num_qubits: usize,
    /// Random state
    pub random_state: Option<u64>,
}

/// Configuration for Quantum Canonical Correlation Analysis
#[derive(Debug, Clone)]
pub struct QCCAConfig {
    /// Number of components
    pub n_components: usize,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits
    pub num_qubits: usize,
    /// Random state
    pub random_state: Option<u64>,
}

/// Configuration for Quantum Non-negative Matrix Factorization
#[derive(Debug, Clone)]
pub struct QNMFConfig {
    /// Number of components
    pub n_components: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits
    pub num_qubits: usize,
    /// Random state
    pub random_state: Option<u64>,
}

/// Configuration for Quantum Manifold Learning
#[derive(Debug, Clone)]
pub struct QManifoldConfig {
    /// Number of components
    pub n_components: usize,
    /// Number of neighbors
    pub n_neighbors: usize,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits
    pub num_qubits: usize,
    /// Distance metric
    pub distance_metric: QuantumDistanceMetric,
    /// Random state
    pub random_state: Option<u64>,
}

/// Configuration for Quantum Kernel PCA
#[derive(Debug, Clone)]
pub struct QKernelPCAConfig {
    /// Number of components
    pub n_components: usize,
    /// Quantum feature map
    pub feature_map: QuantumFeatureMap,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits
    pub num_qubits: usize,
    /// Kernel parameters
    pub kernel_params: HashMap<String, f64>,
    /// Random state
    pub random_state: Option<u64>,
}

/// Configuration for Quantum Feature Selection
#[derive(Debug, Clone)]
pub struct QFeatureSelectionConfig {
    /// Number of features to select
    pub n_features: usize,
    /// Selection method
    pub method: String,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits
    pub num_qubits: usize,
    /// Random state
    pub random_state: Option<u64>,
}

/// Configuration for specialized quantum methods
#[derive(Debug, Clone)]
pub struct QSpecializedConfig {
    /// Method-specific parameters
    pub params: HashMap<String, f64>,
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    /// Number of qubits
    pub num_qubits: usize,
    /// Random state
    pub random_state: Option<u64>,
}

/// Trained state for dimensionality reduction
#[derive(Debug, Clone)]
pub struct DRTrainedState {
    /// Principal components or transformation matrix
    pub components: Array2<f64>,
    /// Explained variance ratios
    pub explained_variance_ratio: Array1<f64>,
    /// Mean of training data
    pub mean: Array1<f64>,
    /// Scaling factors
    pub scale: Option<Array1<f64>>,
    /// Quantum circuit parameters
    pub quantum_parameters: HashMap<String, f64>,
    /// Model-specific parameters
    pub model_parameters: HashMap<String, String>,
    /// Training data statistics
    pub training_statistics: HashMap<String, f64>,
}

// Default implementations
impl Default for QPCAConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            eigensolver: QuantumEigensolver::VQE,
            quantum_enhancement: QuantumEnhancementLevel::Moderate,
            num_qubits: 4,
            whiten: false,
            random_state: None,
            tolerance: 1e-6,
            max_iterations: 1000,
        }
    }
}

impl Default for QICAConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iterations: 200,
            tolerance: 1e-4,
            quantum_enhancement: QuantumEnhancementLevel::Moderate,
            num_qubits: 4,
            learning_rate: 1.0,
            nonlinearity: "logcosh".to_string(),
            random_state: None,
        }
    }
}

impl Default for QtSNEConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            early_exaggeration: 12.0,
            learning_rate: 200.0,
            max_iterations: 1000,
            quantum_enhancement: QuantumEnhancementLevel::Moderate,
            num_qubits: 4,
            distance_metric: QuantumDistanceMetric::QuantumEuclidean,
            random_state: None,
        }
    }
}

impl Default for QAutoencoderConfig {
    fn default() -> Self {
        Self {
            encoder_layers: vec![128, 64, 32],
            decoder_layers: vec![32, 64, 128],
            latent_dim: 16,
            architecture: AutoencoderArchitecture::Standard,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            quantum_enhancement: QuantumEnhancementLevel::Moderate,
            num_qubits: 4,
            beta: 1.0,
            noise_level: 0.1,
            sparsity_parameter: 0.01,
            random_state: None,
        }
    }
}

impl Default for QUMAPConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 15,
            min_dist: 0.1,
            learning_rate: 1.0,
            n_epochs: 200,
            quantum_enhancement: QuantumEnhancementLevel::Moderate,
            num_qubits: 4,
            distance_metric: QuantumDistanceMetric::QuantumEuclidean,
            random_state: None,
        }
    }
}

impl Default for QLDAConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            shrinkage: None,
            quantum_enhancement: QuantumEnhancementLevel::Moderate,
            num_qubits: 4,
            solver: QuantumEigensolver::VQE,
            random_state: None,
        }
    }
}

impl Default for QManifoldConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 10,
            quantum_enhancement: QuantumEnhancementLevel::Moderate,
            num_qubits: 4,
            distance_metric: QuantumDistanceMetric::QuantumEuclidean,
            random_state: None,
        }
    }
}

impl Default for QKernelPCAConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            feature_map: QuantumFeatureMap::ZFeatureMap,
            quantum_enhancement: QuantumEnhancementLevel::Moderate,
            num_qubits: 4,
            kernel_params: HashMap::new(),
            random_state: None,
        }
    }
}
