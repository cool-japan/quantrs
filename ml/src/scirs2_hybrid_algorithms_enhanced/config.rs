//! Configuration types for enhanced hybrid quantum-classical algorithms

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Enhanced hybrid algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedHybridConfig {
    /// Base hybrid configuration
    pub base_config: HybridAlgorithmConfig,

    /// Enable ML-driven optimization
    pub enable_ml_optimization: bool,

    /// Enable adaptive parameter learning
    pub enable_adaptive_learning: bool,

    /// Enable real-time performance tuning
    pub enable_realtime_tuning: bool,

    /// Enable comprehensive benchmarking
    pub enable_benchmarking: bool,

    /// Enable distributed computation
    pub enable_distributed: bool,

    /// Enable visual analytics
    pub enable_visual_analytics: bool,

    /// Algorithm variants
    pub algorithm_variants: Vec<HybridAlgorithm>,

    /// Optimization strategies
    pub optimization_strategies: Vec<OptimizationStrategy>,

    /// Performance targets
    pub performance_targets: PerformanceTargets,

    /// Analysis options
    pub analysis_options: HybridAnalysisOptions,
}

impl Default for EnhancedHybridConfig {
    fn default() -> Self {
        Self {
            base_config: HybridAlgorithmConfig::default(),
            enable_ml_optimization: true,
            enable_adaptive_learning: true,
            enable_realtime_tuning: true,
            enable_benchmarking: true,
            enable_distributed: true,
            enable_visual_analytics: true,
            algorithm_variants: vec![
                HybridAlgorithm::VQE,
                HybridAlgorithm::QAOA,
                HybridAlgorithm::VQC,
            ],
            optimization_strategies: vec![
                OptimizationStrategy::AdaptiveGradient,
                OptimizationStrategy::NaturalGradient,
                OptimizationStrategy::QuantumNaturalGradient,
            ],
            performance_targets: PerformanceTargets::default(),
            analysis_options: HybridAnalysisOptions::default(),
        }
    }
}

/// Base hybrid algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridAlgorithmConfig {
    /// Maximum iterations
    pub max_iterations: usize,

    /// Convergence threshold
    pub convergence_threshold: f64,

    /// Learning rate
    pub learning_rate: f64,

    /// Number of measurement shots
    pub num_shots: usize,

    /// Batch size for parallel execution
    pub batch_size: usize,

    /// Gradient method
    pub gradient_method: GradientMethod,

    /// Optimizer type
    pub optimizer_type: OptimizerType,

    /// Hardware backend
    pub hardware_backend: HardwareBackend,
}

impl Default for HybridAlgorithmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            learning_rate: 0.1,
            num_shots: 10000,
            batch_size: 10,
            gradient_method: GradientMethod::ParameterShift,
            optimizer_type: OptimizerType::Adam,
            hardware_backend: HardwareBackend::Simulator,
        }
    }
}

/// Hybrid algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HybridAlgorithm {
    VQE,       // Variational Quantum Eigensolver
    QAOA,      // Quantum Approximate Optimization Algorithm
    VQC,       // Variational Quantum Classifier
    QNN,       // Quantum Neural Network
    QGAN,      // Quantum Generative Adversarial Network
    VQA,       // Variational Quantum Algorithm (generic)
    ADAPT,     // Adaptive Derivative-Assembled Pseudo-Trotter
    QuantumRL, // Quantum Reinforcement Learning
}

/// Optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    StandardGradient,
    AdaptiveGradient,
    NaturalGradient,
    QuantumNaturalGradient,
    SPSA,   // Simultaneous Perturbation Stochastic Approximation
    COBYLA, // Constrained Optimization BY Linear Approximation
    NelderMead,
    Bayesian,
    EvolutionaryStrategies,
    ReinforcementLearning,
}

/// Gradient calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientMethod {
    FiniteDifference,
    ParameterShift,
    HadamardTest,
    DirectMeasurement,
    MLEstimation,
}

/// Optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    GradientDescent,
    Adam,
    RMSprop,
    AdaGrad,
    LBFGS,
    Newton,
    TrustRegion,
    Custom,
}

/// Hardware backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareBackend {
    Simulator,
    IBMQ,
    IonQ,
    Rigetti,
    AzureQuantum,
    AmazonBraket,
    Custom,
}

/// Performance targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub target_accuracy: f64,
    pub max_runtime: Duration,
    pub max_circuit_evaluations: usize,
    pub min_convergence_rate: f64,
    pub resource_budget: ResourceBudget,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_accuracy: 0.999,
            max_runtime: Duration::from_secs(3600),
            max_circuit_evaluations: 100000,
            min_convergence_rate: 0.001,
            resource_budget: ResourceBudget::default(),
        }
    }
}

/// Resource budget constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub max_qubits: usize,
    pub max_gates: usize,
    pub max_depth: usize,
    pub max_cost: f64,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self {
            max_qubits: 100,
            max_gates: 10000,
            max_depth: 1000,
            max_cost: 1000.0,
        }
    }
}

/// Analysis options for hybrid algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridAnalysisOptions {
    pub track_convergence: bool,
    pub analyze_landscape: bool,
    pub detect_barren_plateaus: bool,
    pub monitor_entanglement: bool,
    pub profile_performance: bool,
    pub validate_gradients: bool,
}

impl Default for HybridAnalysisOptions {
    fn default() -> Self {
        Self {
            track_convergence: true,
            analyze_landscape: true,
            detect_barren_plateaus: true,
            monitor_entanglement: true,
            profile_performance: true,
            validate_gradients: true,
        }
    }
}
