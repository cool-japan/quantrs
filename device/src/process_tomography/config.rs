//! Configuration types for process tomography

use std::collections::HashMap;

/// Configuration for SciRS2-enhanced process tomography
#[derive(Debug, Clone)]
pub struct SciRS2ProcessTomographyConfig {
    /// Number of input states for process characterization
    pub num_input_states: usize,
    /// Number of measurement shots per state
    pub shots_per_state: usize,
    /// Reconstruction method
    pub reconstruction_method: ReconstructionMethod,
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Enable compressed sensing reconstruction
    pub enable_compressed_sensing: bool,
    /// Enable maximum likelihood estimation
    pub enable_mle: bool,
    /// Enable Bayesian inference
    pub enable_bayesian: bool,
    /// Enable process structure analysis
    pub enable_structure_analysis: bool,
    /// Enable multi-process tomography
    pub enable_multi_process: bool,
    /// Optimization settings
    pub optimization_config: OptimizationConfig,
    /// Validation settings
    pub validation_config: ProcessValidationConfig,
}

/// Process reconstruction methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructionMethod {
    /// Linear inversion (fast but can produce unphysical results)
    LinearInversion,
    /// Maximum likelihood estimation (physical but slower)
    MaximumLikelihood,
    /// Compressed sensing (sparse process assumption)
    CompressedSensing,
    /// Bayesian inference with priors
    BayesianInference,
    /// Ensemble methods combining multiple approaches
    EnsembleMethods,
    /// Machine learning based reconstruction
    MachineLearning,
}

/// Optimization configuration for process reconstruction
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Enable parallel optimization
    pub enable_parallel: bool,
    /// Enable adaptive step sizing
    pub adaptive_step_size: bool,
    /// Regularization parameters
    pub regularization: RegularizationConfig,
}

/// Optimization algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationAlgorithm {
    LBFGS,
    ConjugateGradient,
    TrustRegion,
    DifferentialEvolution,
    SimulatedAnnealing,
    GeneticAlgorithm,
    ParticleSwarm,
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization strength (sparsity)
    pub l1_strength: f64,
    /// L2 regularization strength (smoothness)
    pub l2_strength: f64,
    /// Trace preservation constraint strength
    pub trace_strength: f64,
    /// Positivity constraint strength
    pub positivity_strength: f64,
}

/// Process validation configuration
#[derive(Debug, Clone)]
pub struct ProcessValidationConfig {
    /// Enable cross-validation
    pub enable_cross_validation: bool,
    /// Number of CV folds
    pub cv_folds: usize,
    /// Enable bootstrap validation
    pub enable_bootstrap: bool,
    /// Number of bootstrap samples
    pub bootstrap_samples: usize,
    /// Enable process benchmarking
    pub enable_benchmarking: bool,
    /// Benchmark processes to compare against
    pub benchmark_processes: Vec<String>,
}

impl Default for SciRS2ProcessTomographyConfig {
    fn default() -> Self {
        Self {
            num_input_states: 36, // 6^n for n qubits (standard set)
            shots_per_state: 10000,
            reconstruction_method: ReconstructionMethod::MaximumLikelihood,
            confidence_level: 0.95,
            enable_compressed_sensing: true,
            enable_mle: true,
            enable_bayesian: false,
            enable_structure_analysis: true,
            enable_multi_process: false,
            optimization_config: OptimizationConfig::default(),
            validation_config: ProcessValidationConfig::default(),
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            algorithm: OptimizationAlgorithm::LBFGS,
            enable_parallel: true,
            adaptive_step_size: true,
            regularization: RegularizationConfig::default(),
        }
    }
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_strength: 0.001,
            l2_strength: 0.01,
            trace_strength: 1000.0,
            positivity_strength: 100.0,
        }
    }
}

impl Default for ProcessValidationConfig {
    fn default() -> Self {
        Self {
            enable_cross_validation: true,
            cv_folds: 5,
            enable_bootstrap: true,
            bootstrap_samples: 100,
            enable_benchmarking: true,
            benchmark_processes: vec![
                "identity".to_string(),
                "pauli_x".to_string(),
                "pauli_y".to_string(),
                "pauli_z".to_string(),
                "hadamard".to_string(),
            ],
        }
    }
}
