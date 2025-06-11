//! Configuration structures for dynamical decoupling

use std::collections::HashMap;

/// Configuration for dynamical decoupling with SciRS2 optimization
#[derive(Debug, Clone)]
pub struct DynamicalDecouplingConfig {
    /// DD sequence type
    pub sequence_type: DDSequenceType,
    /// Sequence optimization configuration
    pub optimization_config: DDOptimizationConfig,
    /// Hardware adaptation settings
    pub hardware_adaptation: DDHardwareConfig,
    /// Noise characterization settings
    pub noise_characterization: DDNoiseConfig,
    /// Performance analysis settings
    pub performance_config: DDPerformanceConfig,
    /// Validation and testing settings
    pub validation_config: DDValidationConfig,
}

impl Default for DynamicalDecouplingConfig {
    fn default() -> Self {
        Self {
            sequence_type: DDSequenceType::CPMG,
            optimization_config: DDOptimizationConfig::default(),
            hardware_adaptation: DDHardwareConfig::default(),
            noise_characterization: DDNoiseConfig::default(),
            performance_config: DDPerformanceConfig::default(),
            validation_config: DDValidationConfig::default(),
        }
    }
}

/// Types of dynamical decoupling sequences
#[derive(Debug, Clone, PartialEq)]
pub enum DDSequenceType {
    /// Carr-Purcell (CP) sequence
    CarrPurcell,
    /// Carr-Purcell-Meiboom-Gill (CPMG) sequence
    CPMG,
    /// XY-4 sequence
    XY4,
    /// XY-8 sequence
    XY8,
    /// XY-16 sequence
    XY16,
    /// Knill dynamical decoupling (KDD)
    KDD,
    /// Uhrig dynamical decoupling (UDD)
    UDD,
    /// Quadratic dynamical decoupling (QDD)
    QDD,
    /// Concatenated dynamical decoupling (CDD)
    CDD,
    /// Robust dynamical decoupling (RDD)
    RDD,
    /// Optimized sequences using SciRS2
    SciRS2Optimized,
    /// Custom user-defined sequence
    Custom(String),
}

/// DD sequence optimization configuration using SciRS2
#[derive(Debug, Clone)]
pub struct DDOptimizationConfig {
    /// Enable sequence optimization
    pub enable_optimization: bool,
    /// Optimization objective
    pub optimization_objective: DDOptimizationObjective,
    /// Optimization algorithm
    pub optimization_algorithm: DDOptimizationAlgorithm,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Parameter bounds for optimization
    pub parameter_bounds: Option<Vec<(f64, f64)>>,
    /// Multi-objective optimization weights
    pub multi_objective_weights: HashMap<String, f64>,
    /// Enable adaptive optimization
    pub enable_adaptive: bool,
}

impl Default for DDOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            optimization_objective: DDOptimizationObjective::MaximizeCoherenceTime,
            optimization_algorithm: DDOptimizationAlgorithm::GradientFree,
            max_iterations: 1000,
            convergence_tolerance: 1e-6,
            parameter_bounds: None,
            multi_objective_weights: HashMap::new(),
            enable_adaptive: true,
        }
    }
}

/// DD optimization objectives
#[derive(Debug, Clone, PartialEq)]
pub enum DDOptimizationObjective {
    /// Maximize coherence time
    MaximizeCoherenceTime,
    /// Minimize decoherence rate
    MinimizeDecoherenceRate,
    /// Maximize process fidelity
    MaximizeProcessFidelity,
    /// Minimize gate overhead
    MinimizeGateOverhead,
    /// Maximize robustness to noise
    MaximizeRobustness,
    /// Multi-objective optimization
    MultiObjective,
    /// Custom objective function
    Custom(String),
}

/// DD optimization algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum DDOptimizationAlgorithm {
    /// Gradient-free optimization
    GradientFree,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Differential evolution
    DifferentialEvolution,
    /// Bayesian optimization
    BayesianOptimization,
    /// Reinforcement learning
    ReinforcementLearning,
}

/// Hardware adaptation configuration for DD
#[derive(Debug, Clone)]
pub struct DDHardwareConfig {
    /// Enable hardware-aware optimization
    pub enable_hardware_aware: bool,
    /// Account for gate set constraints
    pub gate_set_constraints: bool,
    /// Account for connectivity constraints
    pub connectivity_constraints: bool,
    /// Account for timing constraints
    pub timing_constraints: bool,
    /// Hardware-specific pulse optimization
    pub pulse_optimization: DDPulseConfig,
    /// Error characterization integration
    pub error_characterization: bool,
}

impl Default for DDHardwareConfig {
    fn default() -> Self {
        Self {
            enable_hardware_aware: true,
            gate_set_constraints: true,
            connectivity_constraints: true,
            timing_constraints: true,
            pulse_optimization: DDPulseConfig::default(),
            error_characterization: true,
        }
    }
}

/// DD pulse optimization configuration
#[derive(Debug, Clone)]
pub struct DDPulseConfig {
    /// Enable pulse-level optimization
    pub enable_pulse_optimization: bool,
    /// Pulse shape optimization
    pub pulse_shape_optimization: bool,
    /// Composite pulse sequences
    pub composite_pulses: bool,
    /// Adiabatic pulses
    pub adiabatic_pulses: bool,
    /// Optimal control pulses
    pub optimal_control: bool,
}

impl Default for DDPulseConfig {
    fn default() -> Self {
        Self {
            enable_pulse_optimization: false,
            pulse_shape_optimization: false,
            composite_pulses: true,
            adiabatic_pulses: false,
            optimal_control: false,
        }
    }
}

/// Noise characterization configuration for DD
#[derive(Debug, Clone)]
pub struct DDNoiseConfig {
    /// Enable noise characterization
    pub enable_characterization: bool,
    /// Noise types to consider
    pub noise_types: Vec<NoiseType>,
    /// Spectral noise analysis
    pub spectral_analysis: bool,
    /// Temporal correlation analysis
    pub temporal_correlation: bool,
    /// Spatial correlation analysis
    pub spatial_correlation: bool,
    /// Non-Markovian noise modeling
    pub non_markovian_modeling: bool,
}

impl Default for DDNoiseConfig {
    fn default() -> Self {
        Self {
            enable_characterization: true,
            noise_types: vec![
                NoiseType::AmplitudeDamping,
                NoiseType::PhaseDamping,
                NoiseType::Depolarizing,
            ],
            spectral_analysis: true,
            temporal_correlation: true,
            spatial_correlation: false,
            non_markovian_modeling: false,
        }
    }
}

/// Types of noise affecting qubits
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Amplitude damping (T1 decay)
    AmplitudeDamping,
    /// Phase damping (T2 dephasing)
    PhaseDamping,
    /// Depolarizing noise
    Depolarizing,
    /// Pauli noise
    Pauli,
    /// Coherent errors
    CoherentErrors,
    /// 1/f noise
    OneOverFNoise,
    /// Random telegraph noise
    RandomTelegraphNoise,
    /// Charge noise
    ChargeNoise,
    /// Flux noise
    FluxNoise,
    /// Cross-talk
    CrossTalk,
}

/// Performance analysis configuration for DD
#[derive(Debug, Clone)]
pub struct DDPerformanceConfig {
    /// Enable performance analysis
    pub enable_analysis: bool,
    /// Performance metrics to calculate
    pub metrics: Vec<DDPerformanceMetric>,
    /// Statistical analysis depth
    pub statistical_depth: StatisticalDepth,
    /// Enable benchmarking
    pub enable_benchmarking: bool,
    /// Benchmarking configuration
    pub benchmarking_config: DDBenchmarkingConfig,
}

impl Default for DDPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_analysis: true,
            metrics: vec![
                DDPerformanceMetric::CoherenceTime,
                DDPerformanceMetric::ProcessFidelity,
                DDPerformanceMetric::GateOverhead,
            ],
            statistical_depth: StatisticalDepth::Comprehensive,
            enable_benchmarking: true,
            benchmarking_config: DDBenchmarkingConfig::default(),
        }
    }
}

/// DD performance metrics
#[derive(Debug, Clone, PartialEq)]
pub enum DDPerformanceMetric {
    /// Effective coherence time
    CoherenceTime,
    /// Process fidelity
    ProcessFidelity,
    /// Gate count overhead
    GateOverhead,
    /// Execution time overhead
    TimeOverhead,
    /// Robustness to parameter variations
    RobustnessScore,
    /// Noise suppression factor
    NoiseSuppressionFactor,
    /// Resource efficiency
    ResourceEfficiency,
}

/// Statistical analysis depth
#[derive(Debug, Clone, PartialEq)]
pub enum StatisticalDepth {
    /// Basic statistical analysis
    Basic,
    /// Comprehensive statistical analysis
    Comprehensive,
    /// Advanced statistical analysis with machine learning
    Advanced,
}

/// DD benchmarking configuration
#[derive(Debug, Clone)]
pub struct DDBenchmarkingConfig {
    /// Enable comparative benchmarking
    pub enable_comparative: bool,
    /// Benchmark protocols
    pub protocols: Vec<BenchmarkProtocol>,
    /// Number of benchmark runs
    pub benchmark_runs: usize,
    /// Statistical confidence level
    pub confidence_level: f64,
}

impl Default for DDBenchmarkingConfig {
    fn default() -> Self {
        Self {
            enable_comparative: true,
            protocols: vec![
                BenchmarkProtocol::RandomizedBenchmarking,
                BenchmarkProtocol::ProcessTomography,
            ],
            benchmark_runs: 100,
            confidence_level: 0.95,
        }
    }
}

/// Benchmark protocols for DD
#[derive(Debug, Clone, PartialEq)]
pub enum BenchmarkProtocol {
    /// Randomized benchmarking
    RandomizedBenchmarking,
    /// Process tomography
    ProcessTomography,
    /// Gate set tomography
    GateSetTomography,
    /// Cross-entropy benchmarking
    CrossEntropyBenchmarking,
    /// Cycle benchmarking
    CycleBenchmarking,
}

/// DD validation configuration
#[derive(Debug, Clone)]
pub struct DDValidationConfig {
    /// Enable validation
    pub enable_validation: bool,
    /// Cross-validation folds
    pub cross_validation_folds: usize,
    /// Out-of-sample validation fraction
    pub out_of_sample_fraction: f64,
    /// Enable robustness testing
    pub enable_robustness_testing: bool,
    /// Robustness test parameters
    pub robustness_test_config: RobustnessTestConfig,
    /// Enable generalization analysis
    pub enable_generalization: bool,
}

impl Default for DDValidationConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            cross_validation_folds: 5,
            out_of_sample_fraction: 0.2,
            enable_robustness_testing: true,
            robustness_test_config: RobustnessTestConfig::default(),
            enable_generalization: true,
        }
    }
}

/// Robustness test configuration
#[derive(Debug, Clone)]
pub struct RobustnessTestConfig {
    /// Parameter variation ranges
    pub parameter_variations: HashMap<String, (f64, f64)>,
    /// Noise level variations
    pub noise_variations: Vec<f64>,
    /// Hardware variation tests
    pub hardware_variations: bool,
    /// Systematic error tests
    pub systematic_errors: bool,
}

impl Default for RobustnessTestConfig {
    fn default() -> Self {
        let mut parameter_variations = HashMap::new();
        parameter_variations.insert("pulse_amplitude".to_string(), (0.8, 1.2));
        parameter_variations.insert("pulse_duration".to_string(), (0.9, 1.1));

        Self {
            parameter_variations,
            noise_variations: vec![0.5, 1.0, 1.5, 2.0],
            hardware_variations: true,
            systematic_errors: true,
        }
    }
}
