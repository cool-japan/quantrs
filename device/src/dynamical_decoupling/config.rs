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

/// Adaptive DD configuration
#[derive(Debug, Clone)]
pub struct AdaptiveDDConfig {
    /// Enable adaptive sequence selection
    pub enable_adaptive_selection: bool,
    /// Adaptation criteria
    pub adaptation_criteria: AdaptationCriteria,
    /// Real-time monitoring configuration
    pub monitoring_config: MonitoringConfig,
    /// Feedback control parameters
    pub feedback_control: FeedbackControlConfig,
    /// Learning parameters
    pub learning_config: LearningConfig,
}

impl Default for AdaptiveDDConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_selection: true,
            adaptation_criteria: AdaptationCriteria::default(),
            monitoring_config: MonitoringConfig::default(),
            feedback_control: FeedbackControlConfig::default(),
            learning_config: LearningConfig::default(),
        }
    }
}

/// Criteria for adaptive DD selection
#[derive(Debug, Clone)]
pub struct AdaptationCriteria {
    /// Coherence time threshold
    pub coherence_threshold: f64,
    /// Fidelity threshold
    pub fidelity_threshold: f64,
    /// Noise level threshold
    pub noise_threshold: f64,
    /// Performance degradation tolerance
    pub performance_tolerance: f64,
    /// Adaptation frequency
    pub adaptation_frequency: AdaptationFrequency,
}

impl Default for AdaptationCriteria {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.1, // 10% degradation
            fidelity_threshold: 0.95,
            noise_threshold: 0.05,
            performance_tolerance: 0.1,
            adaptation_frequency: AdaptationFrequency::Dynamic,
        }
    }
}

/// Adaptation frequency options
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationFrequency {
    /// Fixed time intervals
    Fixed(std::time::Duration),
    /// Event-driven adaptation
    EventDriven,
    /// Dynamic based on performance
    Dynamic,
    /// Continuous adaptation
    Continuous,
}

/// Monitoring configuration for adaptive DD
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable real-time monitoring
    pub enable_realtime: bool,
    /// Monitoring metrics
    pub metrics: Vec<MonitoringMetric>,
    /// Sampling rate
    pub sampling_rate: f64,
    /// Buffer size for historical data
    pub buffer_size: usize,
    /// Alert thresholds
    pub alert_thresholds: std::collections::HashMap<String, f64>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            metrics: vec![
                MonitoringMetric::CoherenceTime,
                MonitoringMetric::Fidelity,
                MonitoringMetric::NoiseLevel,
            ],
            sampling_rate: 1000.0, // Hz
            buffer_size: 1000,
            alert_thresholds: std::collections::HashMap::new(),
        }
    }
}

/// Monitoring metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MonitoringMetric {
    /// Coherence time
    CoherenceTime,
    /// Process fidelity
    Fidelity,
    /// Noise level
    NoiseLevel,
    /// Gate error rate
    GateErrorRate,
    /// Readout error rate
    ReadoutErrorRate,
    /// Cross-talk strength
    CrosstalkStrength,
    /// Temperature
    Temperature,
    /// Power consumption
    PowerConsumption,
}

/// Feedback control configuration
#[derive(Debug, Clone)]
pub struct FeedbackControlConfig {
    /// Control algorithm
    pub control_algorithm: ControlAlgorithm,
    /// PID parameters
    pub pid_parameters: PIDParameters,
    /// Control bandwidth
    pub control_bandwidth: f64,
    /// Actuator limits
    pub actuator_limits: ActuatorLimits,
    /// Safety margins
    pub safety_margins: SafetyMargins,
}

impl Default for FeedbackControlConfig {
    fn default() -> Self {
        Self {
            control_algorithm: ControlAlgorithm::PID,
            pid_parameters: PIDParameters::default(),
            control_bandwidth: 100.0, // Hz
            actuator_limits: ActuatorLimits::default(),
            safety_margins: SafetyMargins::default(),
        }
    }
}

/// Control algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum ControlAlgorithm {
    /// PID control
    PID,
    /// Model predictive control
    MPC,
    /// Adaptive control
    Adaptive,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Neural network control
    NeuralNetwork,
}

/// PID controller parameters
#[derive(Debug, Clone)]
pub struct PIDParameters {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Integral windup limit
    pub windup_limit: f64,
    /// Output limits
    pub output_limits: (f64, f64),
}

impl Default for PIDParameters {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.01,
            windup_limit: 100.0,
            output_limits: (-100.0, 100.0),
        }
    }
}

/// Actuator limits
#[derive(Debug, Clone)]
pub struct ActuatorLimits {
    /// Maximum pulse amplitude
    pub max_amplitude: f64,
    /// Maximum pulse duration
    pub max_duration: f64,
    /// Maximum frequency
    pub max_frequency: f64,
    /// Rate limits
    pub rate_limits: std::collections::HashMap<String, f64>,
}

impl Default for ActuatorLimits {
    fn default() -> Self {
        Self {
            max_amplitude: 1.0,
            max_duration: 1e-6, // 1 μs
            max_frequency: 1e9, // 1 GHz
            rate_limits: std::collections::HashMap::new(),
        }
    }
}

/// Safety margins
#[derive(Debug, Clone)]
pub struct SafetyMargins {
    /// Temperature margin
    pub temperature_margin: f64,
    /// Power margin
    pub power_margin: f64,
    /// Coherence margin
    pub coherence_margin: f64,
    /// Fidelity margin
    pub fidelity_margin: f64,
}

impl Default for SafetyMargins {
    fn default() -> Self {
        Self {
            temperature_margin: 0.1, // 10%
            power_margin: 0.2,       // 20%
            coherence_margin: 0.15,  // 15%
            fidelity_margin: 0.05,   // 5%
        }
    }
}

/// Learning configuration for adaptive DD
#[derive(Debug, Clone)]
pub struct LearningConfig {
    /// Learning algorithm
    pub learning_algorithm: LearningAlgorithm,
    /// Learning rate
    pub learning_rate: f64,
    /// Experience replay buffer size
    pub replay_buffer_size: usize,
    /// Exploration strategy
    pub exploration_strategy: ExplorationStrategy,
    /// Update frequency
    pub update_frequency: usize,
    /// Target update frequency
    pub target_update_frequency: usize,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            learning_algorithm: LearningAlgorithm::QLearning,
            learning_rate: 0.01,
            replay_buffer_size: 10000,
            exploration_strategy: ExplorationStrategy::EpsilonGreedy(0.1),
            update_frequency: 10,
            target_update_frequency: 100,
        }
    }
}

/// Learning algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum LearningAlgorithm {
    /// Q-learning
    QLearning,
    /// Deep Q-Network
    DQN,
    /// Policy gradient
    PolicyGradient,
    /// Actor-Critic
    ActorCritic,
    /// Proximal Policy Optimization
    PPO,
}

/// Exploration strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ExplorationStrategy {
    /// Epsilon-greedy exploration
    EpsilonGreedy(f64),
    /// Boltzmann exploration
    Boltzmann(f64),
    /// Upper confidence bound
    UCB(f64),
    /// Thompson sampling
    ThompsonSampling,
}

/// Types of dynamical decoupling sequences
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DDSequenceType {
    /// Hahn Echo sequence
    HahnEcho,
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
    /// Composite DD sequence
    Composite,
    /// Multi-qubit coordinated sequence
    MultiQubitCoordinated,
    /// Optimized sequences using SciRS2
    SciRS2Optimized,
    /// Adaptive DD sequence
    Adaptive,
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
    /// Measurement noise
    MeasurementNoise,
    /// Control noise
    ControlNoise,
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

/// Real-time DD configuration
#[derive(Debug, Clone)]
pub struct RealTimeDDConfig {
    /// Enable real-time adaptation
    pub enable_realtime: bool,
    /// Response time requirements
    pub response_time: std::time::Duration,
    /// Real-time scheduling priority
    pub scheduling_priority: SchedulingPriority,
    /// Resource allocation
    pub resource_allocation: ResourceAllocation,
    /// Latency requirements
    pub latency_requirements: LatencyRequirements,
}

impl Default for RealTimeDDConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            response_time: std::time::Duration::from_micros(10), // 10 μs
            scheduling_priority: SchedulingPriority::High,
            resource_allocation: ResourceAllocation::default(),
            latency_requirements: LatencyRequirements::default(),
        }
    }
}

/// Scheduling priorities
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingPriority {
    Low,
    Normal,
    High,
    Critical,
    RealTime,
}

/// Resource allocation configuration
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// CPU cores for DD processing
    pub cpu_cores: usize,
    /// Memory allocation (bytes)
    pub memory_allocation: usize,
    /// GPU acceleration
    pub gpu_acceleration: bool,
    /// Network bandwidth allocation
    pub network_bandwidth: f64,
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            cpu_cores: 2,
            memory_allocation: 1024 * 1024 * 100, // 100 MB
            gpu_acceleration: false,
            network_bandwidth: 1e6, // 1 Mbps
        }
    }
}

/// Latency requirements
#[derive(Debug, Clone)]
pub struct LatencyRequirements {
    /// Maximum control latency
    pub max_control_latency: std::time::Duration,
    /// Maximum measurement latency
    pub max_measurement_latency: std::time::Duration,
    /// Maximum adaptation latency
    pub max_adaptation_latency: std::time::Duration,
    /// Jitter tolerance
    pub jitter_tolerance: std::time::Duration,
}

impl Default for LatencyRequirements {
    fn default() -> Self {
        Self {
            max_control_latency: std::time::Duration::from_nanos(100),
            max_measurement_latency: std::time::Duration::from_micros(1),
            max_adaptation_latency: std::time::Duration::from_micros(10),
            jitter_tolerance: std::time::Duration::from_nanos(10),
        }
    }
}
