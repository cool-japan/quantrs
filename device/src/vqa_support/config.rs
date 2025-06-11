//! Configuration structures and enums for VQA algorithms

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Configuration for VQA execution with SciRS2 optimization
#[derive(Debug, Clone)]
pub struct VQAConfig {
    /// VQA algorithm type
    pub algorithm_type: VQAAlgorithmType,
    /// Optimization configuration
    pub optimization_config: VQAOptimizationConfig,
    /// Statistical analysis configuration
    pub statistical_config: VQAStatisticalConfig,
    /// Hardware-aware optimization settings
    pub hardware_config: VQAHardwareConfig,
    /// Noise mitigation settings
    pub noise_mitigation: VQANoiseMitigation,
    /// Validation and monitoring settings
    pub validation_config: VQAValidationConfig,
}

/// Types of VQA algorithms supported
#[derive(Debug, Clone, PartialEq)]
pub enum VQAAlgorithmType {
    /// Variational Quantum Eigensolver
    VQE,
    /// Quantum Approximate Optimization Algorithm
    QAOA,
    /// Variational Quantum Classifier
    VQC,
    /// Quantum Neural Network
    QNN,
    /// Variational Quantum Factoring
    VQF,
    /// Custom VQA with user-defined ansatz
    Custom(String),
}

/// VQA optimization configuration using SciRS2
#[derive(Debug, Clone)]
pub struct VQAOptimizationConfig {
    /// Primary optimizer
    pub primary_optimizer: VQAOptimizer,
    /// Fallback optimizers for robustness
    pub fallback_optimizers: Vec<VQAOptimizer>,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Parameter bounds
    pub parameter_bounds: Option<Vec<(f64, f64)>>,
    /// Gradient estimation method
    pub gradient_method: GradientMethod,
    /// Enable adaptive optimization
    pub enable_adaptive: bool,
    /// Multi-start optimization
    pub multi_start_config: MultiStartConfig,
    /// Warm restart settings
    pub warm_restart: WarmRestartConfig,
}

/// VQA optimizers leveraging SciRS2
#[derive(Debug, Clone, PartialEq)]
pub enum VQAOptimizer {
    /// L-BFGS-B (for bounded optimization)
    LBFGSB,
    /// COBYLA (for constrained optimization)
    COBYLA,
    /// Nelder-Mead simplex
    NelderMead,
    /// Differential Evolution
    DifferentialEvolution,
    /// Simulated Annealing
    SimulatedAnnealing,
    /// Basin Hopping (global optimization)
    BasinHopping,
    /// Dual Annealing
    DualAnnealing,
    /// Powell's method
    Powell,
    /// Particle Swarm Optimization
    PSO,
    /// Quantum Natural Gradient
    QNG,
    /// SPSA (Simultaneous Perturbation Stochastic Approximation)
    SPSA,
    /// Custom optimizer
    Custom(String),
}

/// Gradient estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum GradientMethod {
    /// Parameter shift rule
    ParameterShift,
    /// Finite differences
    FiniteDifference,
    /// Forward differences
    ForwardDifference,
    /// Central differences
    CentralDifference,
    /// Natural gradient
    NaturalGradient,
    /// Automatic differentiation (if available)
    AutomaticDifferentiation,
}

/// Multi-start optimization configuration
#[derive(Debug, Clone)]
pub struct MultiStartConfig {
    pub enable_multi_start: bool,
    pub num_starts: usize,
    pub initial_point_strategy: InitialPointStrategy,
    pub convergence_criterion: ConvergenceCriterion,
}

/// Strategies for generating initial points
#[derive(Debug, Clone, PartialEq)]
pub enum InitialPointStrategy {
    Random,
    LatinHypercube,
    Sobol,
    Grid,
    PreviousBest,
    AdaptiveSampling,
}

/// Convergence criteria for multi-start optimization
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceCriterion {
    BestValue,
    ValueStability,
    ParameterStability,
    StatisticalTest,
}

/// Warm restart configuration
#[derive(Debug, Clone)]
pub struct WarmRestartConfig {
    pub enable_warm_restart: bool,
    pub restart_threshold: f64,
    pub max_restarts: usize,
    pub restart_strategy: RestartStrategy,
}

/// Restart strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RestartStrategy {
    RandomPerturbation,
    BestKnownSolution,
    AdaptiveStep,
    MetaLearning,
}

/// Statistical analysis configuration for VQA
#[derive(Debug, Clone)]
pub struct VQAStatisticalConfig {
    /// Enable statistical monitoring
    pub enable_monitoring: bool,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Number of samples for statistical analysis
    pub num_samples: usize,
    /// Enable distribution fitting
    pub enable_distribution_fitting: bool,
    /// Enable correlation analysis
    pub enable_correlation_analysis: bool,
    /// Enable outlier detection
    pub enable_outlier_detection: bool,
}

/// Hardware-aware optimization configuration
#[derive(Debug, Clone)]
pub struct VQAHardwareConfig {
    /// Enable hardware-aware optimization
    pub enable_hardware_aware: bool,
    /// Account for gate fidelities
    pub use_gate_fidelities: bool,
    /// Account for connectivity constraints
    pub use_connectivity_constraints: bool,
    /// Enable adaptive shot allocation
    pub adaptive_shots: AdaptiveShotConfig,
    /// Hardware-specific optimization
    pub hardware_optimization: HardwareOptimizationConfig,
}

/// Adaptive shot allocation configuration
#[derive(Debug, Clone)]
pub struct AdaptiveShotConfig {
    pub enable_adaptive: bool,
    pub initial_shots: usize,
    pub max_shots: usize,
    pub shot_budget: usize,
    pub allocation_strategy: ShotAllocationStrategy,
    pub precision_target: f64,
}

/// Shot allocation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ShotAllocationStrategy {
    Uniform,
    ProportionalToVariance,
    ExpectedImprovement,
    UncertaintyBased,
    AdaptiveBayesian,
}

/// Hardware-specific optimization configuration
#[derive(Debug, Clone)]
pub struct HardwareOptimizationConfig {
    pub optimize_for_hardware: bool,
    pub qubit_layout_optimization: bool,
    pub gate_scheduling_optimization: bool,
    pub error_mitigation_integration: bool,
}

/// Noise mitigation configuration for VQA
#[derive(Debug, Clone)]
pub struct VQANoiseMitigation {
    /// Enable noise mitigation
    pub enable_mitigation: bool,
    /// Zero-noise extrapolation
    pub zero_noise_extrapolation: ZNEConfig,
    /// Readout error mitigation
    pub readout_error_mitigation: bool,
    /// Symmetry verification
    pub symmetry_verification: bool,
    /// Error mitigation overhead budget
    pub mitigation_budget: f64,
}

/// Zero-noise extrapolation configuration
#[derive(Debug, Clone)]
pub struct ZNEConfig {
    pub enable_zne: bool,
    pub noise_factors: Vec<f64>,
    pub extrapolation_method: String,
    pub mitigation_overhead: f64,
}

/// VQA validation and monitoring configuration
#[derive(Debug, Clone)]
pub struct VQAValidationConfig {
    /// Enable cross-validation
    pub enable_cross_validation: bool,
    /// Number of CV folds
    pub cv_folds: usize,
    /// Enable convergence monitoring
    pub enable_convergence_monitoring: bool,
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    /// Validation metrics to compute
    pub validation_metrics: Vec<ValidationMetric>,
}

/// Validation metrics for VQA
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationMetric {
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
    /// R-squared
    R2,
    /// Variance explained
    VarianceExplained,
    /// Fidelity to target state
    StateFidelity,
    /// Energy variance
    EnergyVariance,
    /// Optimization efficiency
    OptimizationEfficiency,
}

/// Comprehensive VQA execution result with SciRS2 analysis
#[derive(Debug, Clone)]
pub struct VQAResult {
    /// Device identifier
    pub device_id: String,
    /// Algorithm type used
    pub algorithm_type: VQAAlgorithmType,
    /// Configuration used
    pub config: VQAConfig,
    /// Optimal parameters found
    pub optimal_parameters: Array1<f64>,
    /// Optimal objective value
    pub optimal_value: f64,
    /// Optimization trajectory
    pub optimization_trajectory: OptimizationTrajectory,
    /// Statistical analysis of the optimization
    pub statistical_analysis: VQAStatisticalAnalysis,
    /// Hardware performance analysis
    pub hardware_analysis: VQAHardwareAnalysis,
    /// Validation results
    pub validation_results: VQAValidationResults,
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Execution metadata
    pub execution_metadata: VQAExecutionMetadata,
}

/// Optimization trajectory tracking
#[derive(Debug, Clone)]
pub struct OptimizationTrajectory {
    /// Parameter values at each iteration
    pub parameter_history: Vec<Array1<f64>>,
    /// Objective values at each iteration
    pub objective_history: Array1<f64>,
    /// Gradient norms (if available)
    pub gradient_norms: Option<Array1<f64>>,
    /// Step sizes
    pub step_sizes: Array1<f64>,
    /// Iteration timestamps
    pub timestamps: Vec<Instant>,
    /// Convergence indicators
    pub convergence_indicators: ConvergenceIndicators,
}

/// Convergence indicators
#[derive(Debug, Clone)]
pub struct ConvergenceIndicators {
    pub parameter_convergence: bool,
    pub objective_convergence: bool,
    pub gradient_convergence: bool,
    pub stagnation_detected: bool,
}

/// Forward declarations for analysis structures (to be implemented in other modules)
#[derive(Debug, Clone)]
pub struct VQAStatisticalAnalysis;

#[derive(Debug, Clone)]
pub struct VQAHardwareAnalysis;

#[derive(Debug, Clone)]
pub struct VQAValidationResults;

#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis;

#[derive(Debug, Clone)]
pub struct ResourceUtilization;

#[derive(Debug, Clone)]
pub struct VQAExecutionMetadata;

impl Default for VQAConfig {
    fn default() -> Self {
        Self {
            algorithm_type: VQAAlgorithmType::VQE,
            optimization_config: VQAOptimizationConfig {
                primary_optimizer: VQAOptimizer::LBFGSB,
                fallback_optimizers: vec![VQAOptimizer::COBYLA, VQAOptimizer::NelderMead],
                max_iterations: 1000,
                convergence_tolerance: 1e-6,
                parameter_bounds: None,
                gradient_method: GradientMethod::ParameterShift,
                enable_adaptive: true,
                multi_start_config: MultiStartConfig {
                    enable_multi_start: true,
                    num_starts: 5,
                    initial_point_strategy: InitialPointStrategy::LatinHypercube,
                    convergence_criterion: ConvergenceCriterion::BestValue,
                },
                warm_restart: WarmRestartConfig {
                    enable_warm_restart: true,
                    restart_threshold: 1e-4,
                    max_restarts: 3,
                    restart_strategy: RestartStrategy::AdaptiveStep,
                },
            },
            statistical_config: VQAStatisticalConfig {
                enable_monitoring: true,
                confidence_level: 0.95,
                num_samples: 100,
                enable_distribution_fitting: true,
                enable_correlation_analysis: true,
                enable_outlier_detection: true,
            },
            hardware_config: VQAHardwareConfig {
                enable_hardware_aware: true,
                use_gate_fidelities: true,
                use_connectivity_constraints: true,
                adaptive_shots: AdaptiveShotConfig {
                    enable_adaptive: true,
                    initial_shots: 1000,
                    max_shots: 10000,
                    shot_budget: 100000,
                    allocation_strategy: ShotAllocationStrategy::UncertaintyBased,
                    precision_target: 1e-3,
                },
                hardware_optimization: HardwareOptimizationConfig {
                    optimize_for_hardware: true,
                    qubit_layout_optimization: true,
                    gate_scheduling_optimization: true,
                    error_mitigation_integration: true,
                },
            },
            noise_mitigation: VQANoiseMitigation {
                enable_mitigation: true,
                zero_noise_extrapolation: ZNEConfig {
                    enable_zne: true,
                    noise_factors: vec![1.0, 3.0, 5.0],
                    extrapolation_method: "linear".to_string(),
                    mitigation_overhead: 3.0,
                },
                readout_error_mitigation: true,
                symmetry_verification: true,
                mitigation_budget: 5.0,
            },
            validation_config: VQAValidationConfig {
                enable_cross_validation: true,
                cv_folds: 5,
                enable_convergence_monitoring: true,
                enable_performance_tracking: true,
                validation_metrics: vec![
                    ValidationMetric::MSE,
                    ValidationMetric::R2,
                    ValidationMetric::StateFidelity,
                    ValidationMetric::EnergyVariance,
                ],
            },
        }
    }
}

impl VQAConfig {
    /// Create new VQA configuration with algorithm type
    pub fn new(algorithm_type: VQAAlgorithmType) -> Self {
        let mut config = Self::default();
        config.algorithm_type = algorithm_type;
        config
    }

    /// Create VQE-specific configuration
    pub fn vqe() -> Self {
        Self::new(VQAAlgorithmType::VQE)
    }

    /// Create QAOA-specific configuration
    pub fn qaoa() -> Self {
        let mut config = Self::new(VQAAlgorithmType::QAOA);
        // QAOA-specific optimizations
        config.optimization_config.primary_optimizer = VQAOptimizer::COBYLA;
        config.optimization_config.gradient_method = GradientMethod::ParameterShift;
        config
    }

    /// Create VQC-specific configuration
    pub fn vqc() -> Self {
        let mut config = Self::new(VQAAlgorithmType::VQC);
        // VQC-specific optimizations
        config.optimization_config.primary_optimizer = VQAOptimizer::QNG;
        config.validation_config.validation_metrics =
            vec![ValidationMetric::MSE, ValidationMetric::R2];
        config
    }

    /// Enable hardware-aware optimization
    pub fn with_hardware_aware(mut self, enable: bool) -> Self {
        self.hardware_config.enable_hardware_aware = enable;
        self
    }

    /// Set optimization method
    pub fn with_optimizer(mut self, optimizer: VQAOptimizer) -> Self {
        self.optimization_config.primary_optimizer = optimizer;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.optimization_config.max_iterations = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.optimization_config.convergence_tolerance = tolerance;
        self
    }
}

impl OptimizationTrajectory {
    /// Create new optimization trajectory
    pub fn new() -> Self {
        Self {
            parameter_history: Vec::new(),
            objective_history: Array1::zeros(0),
            gradient_norms: None,
            step_sizes: Array1::zeros(0),
            timestamps: Vec::new(),
            convergence_indicators: ConvergenceIndicators {
                parameter_convergence: false,
                objective_convergence: false,
                gradient_convergence: false,
                stagnation_detected: false,
            },
        }
    }
}

impl std::fmt::Display for VQAOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            VQAOptimizer::LBFGSB => "L-BFGS-B",
            VQAOptimizer::COBYLA => "COBYLA",
            VQAOptimizer::NelderMead => "Nelder-Mead",
            VQAOptimizer::DifferentialEvolution => "Differential Evolution",
            VQAOptimizer::SimulatedAnnealing => "Simulated Annealing",
            VQAOptimizer::BasinHopping => "Basin Hopping",
            VQAOptimizer::DualAnnealing => "Dual Annealing",
            VQAOptimizer::Powell => "Powell",
            VQAOptimizer::PSO => "Particle Swarm Optimization",
            VQAOptimizer::QNG => "Quantum Natural Gradient",
            VQAOptimizer::SPSA => "SPSA",
            VQAOptimizer::Custom(name) => name,
        };
        write!(f, "{}", name)
    }
}
