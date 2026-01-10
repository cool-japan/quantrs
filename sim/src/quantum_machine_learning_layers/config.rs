//! Quantum Machine Learning Configuration Types
//!
//! This module contains all configuration types for the QML framework.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum machine learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLConfig {
    /// Number of qubits in the quantum layer
    pub num_qubits: usize,
    /// QML architecture type
    pub architecture_type: QMLArchitectureType,
    /// Layer configuration for each QML layer
    pub layer_configs: Vec<QMLLayerConfig>,
    /// Training algorithm configuration
    pub training_config: QMLTrainingConfig,
    /// Hardware-aware optimization settings
    pub hardware_optimization: HardwareOptimizationConfig,
    /// Classical preprocessing configuration
    pub classical_preprocessing: ClassicalPreprocessingConfig,
    /// Hybrid training configuration
    pub hybrid_training: HybridTrainingConfig,
    /// Enable quantum advantage analysis
    pub quantum_advantage_analysis: bool,
    /// Noise-aware training settings
    pub noise_aware_training: NoiseAwareTrainingConfig,
    /// Performance optimization settings
    pub performance_optimization: PerformanceOptimizationConfig,
}

impl Default for QMLConfig {
    fn default() -> Self {
        Self {
            num_qubits: 8,
            architecture_type: QMLArchitectureType::VariationalQuantumCircuit,
            layer_configs: vec![QMLLayerConfig {
                layer_type: QMLLayerType::ParameterizedQuantumCircuit,
                num_parameters: 16,
                ansatz_type: AnsatzType::Hardware,
                entanglement_pattern: EntanglementPattern::Linear,
                rotation_gates: vec![RotationGate::RY, RotationGate::RZ],
                depth: 4,
                enable_gradient_computation: true,
            }],
            training_config: QMLTrainingConfig::default(),
            hardware_optimization: HardwareOptimizationConfig::default(),
            classical_preprocessing: ClassicalPreprocessingConfig::default(),
            hybrid_training: HybridTrainingConfig::default(),
            quantum_advantage_analysis: true,
            noise_aware_training: NoiseAwareTrainingConfig::default(),
            performance_optimization: PerformanceOptimizationConfig::default(),
        }
    }
}

/// QML architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QMLArchitectureType {
    VariationalQuantumCircuit,
    QuantumConvolutionalNN,
    QuantumRecurrentNN,
    QuantumGraphNN,
    QuantumAttentionNetwork,
    QuantumTransformer,
    HybridClassicalQuantum,
    QuantumBoltzmannMachine,
    QuantumGAN,
    QuantumAutoencoder,
}

/// QML layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLLayerConfig {
    pub layer_type: QMLLayerType,
    pub num_parameters: usize,
    pub ansatz_type: AnsatzType,
    pub entanglement_pattern: EntanglementPattern,
    pub rotation_gates: Vec<RotationGate>,
    pub depth: usize,
    pub enable_gradient_computation: bool,
}

/// Types of QML layers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QMLLayerType {
    ParameterizedQuantumCircuit,
    QuantumConvolutional,
    QuantumPooling,
    QuantumDense,
    QuantumLSTM,
    QuantumGRU,
    QuantumAttention,
    QuantumDropout,
    QuantumBatchNorm,
    DataReUpload,
}

/// Ansatz types for parameterized quantum circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnsatzType {
    Hardware,
    ProblemSpecific,
    AllToAll,
    Layered,
    Alternating,
    BrickWall,
    Tree,
    Custom,
}

/// Entanglement patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntanglementPattern {
    Linear,
    Circular,
    AllToAll,
    Star,
    Grid,
    Random,
    Block,
    Custom,
}

/// Rotation gates for parameterized circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotationGate {
    RX,
    RY,
    RZ,
    U3,
    Phase,
}

/// QML training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLTrainingConfig {
    pub algorithm: QMLTrainingAlgorithm,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub gradient_method: GradientMethod,
    pub optimizer: OptimizerType,
    pub regularization: RegularizationConfig,
    pub early_stopping: EarlyStoppingConfig,
    pub lr_schedule: LearningRateSchedule,
}

impl Default for QMLTrainingConfig {
    fn default() -> Self {
        Self {
            algorithm: QMLTrainingAlgorithm::ParameterShift,
            learning_rate: 0.01,
            epochs: 100,
            batch_size: 32,
            gradient_method: GradientMethod::ParameterShift,
            optimizer: OptimizerType::Adam,
            regularization: RegularizationConfig::default(),
            early_stopping: EarlyStoppingConfig::default(),
            lr_schedule: LearningRateSchedule::Constant,
        }
    }
}

/// QML training algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QMLTrainingAlgorithm {
    ParameterShift,
    FiniteDifference,
    QuantumNaturalGradient,
    SPSA,
    QAOA,
    VQE,
    Rotosolve,
    HybridTraining,
}

/// Gradient computation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientMethod {
    ParameterShift,
    FiniteDifference,
    Adjoint,
    Backpropagation,
    QuantumFisherInformation,
}

/// Optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdaGrad,
    RMSprop,
    Momentum,
    LBFGS,
    QuantumNaturalGradient,
    SPSA,
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    pub l1_strength: f64,
    pub l2_strength: f64,
    pub dropout_prob: f64,
    pub parameter_bounds: Option<(f64, f64)>,
    pub enable_clipping: bool,
    pub gradient_clip_threshold: f64,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_strength: 0.0,
            l2_strength: 0.001,
            dropout_prob: 0.1,
            parameter_bounds: Some((-PI, PI)),
            enable_clipping: true,
            gradient_clip_threshold: 1.0,
        }
    }
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub patience: usize,
    pub min_delta: f64,
    pub monitor_metric: String,
    pub mode_max: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 10,
            min_delta: 1e-6,
            monitor_metric: "val_loss".to_string(),
            mode_max: false,
        }
    }
}

/// Learning rate schedules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant,
    ExponentialDecay,
    StepDecay,
    CosineAnnealing,
    WarmRestart,
    ReduceOnPlateau,
}

/// Hardware optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimizationConfig {
    pub target_hardware: QuantumHardwareTarget,
    pub minimize_gate_count: bool,
    pub minimize_depth: bool,
    pub noise_aware: bool,
    pub connectivity_constraints: ConnectivityConstraints,
    pub gate_fidelities: HashMap<String, f64>,
    pub enable_parallelization: bool,
    pub optimization_level: HardwareOptimizationLevel,
}

impl Default for HardwareOptimizationConfig {
    fn default() -> Self {
        let mut gate_fidelities = HashMap::new();
        gate_fidelities.insert("single_qubit".to_string(), 0.999);
        gate_fidelities.insert("two_qubit".to_string(), 0.99);

        Self {
            target_hardware: QuantumHardwareTarget::Simulator,
            minimize_gate_count: true,
            minimize_depth: true,
            noise_aware: false,
            connectivity_constraints: ConnectivityConstraints::AllToAll,
            gate_fidelities,
            enable_parallelization: true,
            optimization_level: HardwareOptimizationLevel::Medium,
        }
    }
}

/// Quantum hardware targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumHardwareTarget {
    Simulator,
    IBM,
    Google,
    IonQ,
    Rigetti,
    Quantinuum,
    Xanadu,
    Custom,
}

/// Connectivity constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityConstraints {
    AllToAll,
    Linear,
    Grid(usize, usize),
    Custom(Vec<(usize, usize)>),
    HeavyHex,
    Square,
}

/// Hardware optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareOptimizationLevel {
    Basic,
    Medium,
    Aggressive,
    Maximum,
}

/// Classical preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalPreprocessingConfig {
    pub feature_scaling: bool,
    pub scaling_method: ScalingMethod,
    pub enable_pca: bool,
    pub pca_components: Option<usize>,
    pub encoding_method: DataEncodingMethod,
    pub feature_selection: FeatureSelectionConfig,
}

impl Default for ClassicalPreprocessingConfig {
    fn default() -> Self {
        Self {
            feature_scaling: true,
            scaling_method: ScalingMethod::StandardScaler,
            enable_pca: false,
            pca_components: None,
            encoding_method: DataEncodingMethod::Amplitude,
            feature_selection: FeatureSelectionConfig::default(),
        }
    }
}

/// Scaling methods for classical preprocessing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingMethod {
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileUniform,
    PowerTransformer,
}

/// Data encoding methods for quantum circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataEncodingMethod {
    Amplitude,
    Angle,
    Basis,
    QuantumFeatureMap,
    IQP,
    PauliFeatureMap,
    DataReUpload,
}

/// Feature selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    pub enabled: bool,
    pub method: FeatureSelectionMethod,
    pub num_features: Option<usize>,
    pub threshold: f64,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: FeatureSelectionMethod::VarianceThreshold,
            num_features: None,
            threshold: 0.0,
        }
    }
}

/// Feature selection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    VarianceThreshold,
    UnivariateSelection,
    RecursiveFeatureElimination,
    L1Based,
    TreeBased,
    QuantumFeatureImportance,
}

/// Hybrid training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridTrainingConfig {
    pub enabled: bool,
    pub classical_architecture: ClassicalArchitecture,
    pub interface_config: QuantumClassicalInterface,
    pub alternating_schedule: AlternatingSchedule,
    pub gradient_flow: GradientFlowConfig,
}

impl Default for HybridTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            classical_architecture: ClassicalArchitecture::MLP,
            interface_config: QuantumClassicalInterface::Expectation,
            alternating_schedule: AlternatingSchedule::Simultaneous,
            gradient_flow: GradientFlowConfig::default(),
        }
    }
}

/// Classical neural network architectures for hybrid training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassicalArchitecture {
    MLP,
    CNN,
    RNN,
    LSTM,
    Transformer,
    ResNet,
    Custom,
}

/// Quantum-classical interfaces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumClassicalInterface {
    Expectation,
    Sampling,
    StateTomography,
    ProcessTomography,
    ShadowTomography,
}

/// Alternating training schedules for hybrid systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlternatingSchedule {
    Simultaneous,
    Alternating,
    ClassicalFirst,
    QuantumFirst,
    Custom,
}

/// Gradient flow configuration for hybrid training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowConfig {
    pub classical_to_quantum: bool,
    pub quantum_to_classical: bool,
    pub gradient_scaling: f64,
    pub enable_clipping: bool,
    pub accumulation_steps: usize,
}

impl Default for GradientFlowConfig {
    fn default() -> Self {
        Self {
            classical_to_quantum: true,
            quantum_to_classical: true,
            gradient_scaling: 1.0,
            enable_clipping: true,
            accumulation_steps: 1,
        }
    }
}

/// Noise-aware training configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoiseAwareTrainingConfig {
    pub enabled: bool,
    pub noise_parameters: NoiseParameters,
    pub error_mitigation: ErrorMitigationConfig,
    pub noise_characterization: NoiseCharacterizationConfig,
    pub robust_training: RobustTrainingConfig,
}

/// Noise parameters for quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseParameters {
    pub single_qubit_error: f64,
    pub two_qubit_error: f64,
    pub measurement_error: f64,
    pub coherence_times: (f64, f64),
    pub gate_times: HashMap<String, f64>,
}

impl Default for NoiseParameters {
    fn default() -> Self {
        let mut gate_times = HashMap::new();
        gate_times.insert("single_qubit".to_string(), 50e-9);
        gate_times.insert("two_qubit".to_string(), 200e-9);

        Self {
            single_qubit_error: 0.001,
            two_qubit_error: 0.01,
            measurement_error: 0.01,
            coherence_times: (50e-6, 100e-6),
            gate_times,
        }
    }
}

/// Error mitigation configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorMitigationConfig {
    pub zero_noise_extrapolation: bool,
    pub readout_error_mitigation: bool,
    pub symmetry_verification: bool,
    pub virtual_distillation: VirtualDistillationConfig,
    pub quantum_error_correction: bool,
}

/// Virtual distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualDistillationConfig {
    pub enabled: bool,
    pub num_copies: usize,
    pub protocol: DistillationProtocol,
}

impl Default for VirtualDistillationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_copies: 2,
            protocol: DistillationProtocol::Standard,
        }
    }
}

/// Distillation protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistillationProtocol {
    Standard,
    Improved,
    QuantumAdvantage,
}

/// Noise characterization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseCharacterizationConfig {
    pub enabled: bool,
    pub method: NoiseCharacterizationMethod,
    pub benchmarking: BenchmarkingProtocols,
    pub calibration_frequency: CalibrationFrequency,
}

impl Default for NoiseCharacterizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: NoiseCharacterizationMethod::ProcessTomography,
            benchmarking: BenchmarkingProtocols::default(),
            calibration_frequency: CalibrationFrequency::Daily,
        }
    }
}

/// Noise characterization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseCharacterizationMethod {
    ProcessTomography,
    RandomizedBenchmarking,
    GateSetTomography,
    QuantumDetectorTomography,
    CrossEntropyBenchmarking,
}

/// Benchmarking protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingProtocols {
    pub randomized_benchmarking: bool,
    pub quantum_volume: bool,
    pub cross_entropy_benchmarking: bool,
    pub mirror_benchmarking: bool,
}

impl Default for BenchmarkingProtocols {
    fn default() -> Self {
        Self {
            randomized_benchmarking: true,
            quantum_volume: false,
            cross_entropy_benchmarking: false,
            mirror_benchmarking: false,
        }
    }
}

/// Calibration frequency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationFrequency {
    RealTime,
    Hourly,
    Daily,
    Weekly,
    Manual,
}

/// Robust training configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RobustTrainingConfig {
    pub enabled: bool,
    pub noise_injection: NoiseInjectionConfig,
    pub adversarial_training: AdversarialTrainingConfig,
    pub ensemble_methods: EnsembleMethodsConfig,
}

/// Noise injection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseInjectionConfig {
    pub enabled: bool,
    pub injection_probability: f64,
    pub noise_strength: f64,
    pub noise_type: NoiseType,
}

impl Default for NoiseInjectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            injection_probability: 0.1,
            noise_strength: 0.01,
            noise_type: NoiseType::Depolarizing,
        }
    }
}

/// Noise types for training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    Depolarizing,
    AmplitudeDamping,
    PhaseDamping,
    BitFlip,
    PhaseFlip,
    Pauli,
}

/// Adversarial training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialTrainingConfig {
    pub enabled: bool,
    pub attack_strength: f64,
    pub attack_method: AdversarialAttackMethod,
    pub defense_method: AdversarialDefenseMethod,
}

impl Default for AdversarialTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            attack_strength: 0.01,
            attack_method: AdversarialAttackMethod::FGSM,
            defense_method: AdversarialDefenseMethod::AdversarialTraining,
        }
    }
}

/// Adversarial attack methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdversarialAttackMethod {
    FGSM,
    PGD,
    CarliniWagner,
    QuantumAdversarial,
}

/// Adversarial defense methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdversarialDefenseMethod {
    AdversarialTraining,
    DefensiveDistillation,
    CertifiedDefenses,
    QuantumErrorCorrection,
}

/// Ensemble methods configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMethodsConfig {
    pub enabled: bool,
    pub num_ensemble: usize,
    pub ensemble_method: EnsembleMethod,
    pub voting_strategy: VotingStrategy,
}

impl Default for EnsembleMethodsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_ensemble: 5,
            ensemble_method: EnsembleMethod::Bagging,
            voting_strategy: VotingStrategy::MajorityVoting,
        }
    }
}

/// Ensemble methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnsembleMethod {
    Bagging,
    Boosting,
    RandomForest,
    QuantumEnsemble,
}

/// Voting strategies for ensembles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VotingStrategy {
    MajorityVoting,
    WeightedVoting,
    SoftVoting,
    QuantumVoting,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimizationConfig {
    pub enabled: bool,
    pub memory_optimization: MemoryOptimizationConfig,
    pub computation_optimization: ComputationOptimizationConfig,
    pub parallelization: ParallelizationConfig,
    pub caching: CachingConfig,
}

impl Default for PerformanceOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_optimization: MemoryOptimizationConfig::default(),
            computation_optimization: ComputationOptimizationConfig::default(),
            parallelization: ParallelizationConfig::default(),
            caching: CachingConfig::default(),
        }
    }
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    pub enabled: bool,
    pub memory_mapping: bool,
    pub gradient_checkpointing: bool,
    pub memory_pool_size: Option<usize>,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_mapping: false,
            gradient_checkpointing: false,
            memory_pool_size: None,
        }
    }
}

/// Computation optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationOptimizationConfig {
    pub enabled: bool,
    pub mixed_precision: bool,
    pub simd_optimization: bool,
    pub jit_compilation: bool,
}

impl Default for ComputationOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mixed_precision: false,
            simd_optimization: true,
            jit_compilation: false,
        }
    }
}

/// Parallelization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationConfig {
    pub enabled: bool,
    pub num_threads: Option<usize>,
    pub data_parallelism: bool,
    pub model_parallelism: bool,
    pub pipeline_parallelism: bool,
}

impl Default for ParallelizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_threads: None,
            data_parallelism: true,
            model_parallelism: false,
            pipeline_parallelism: false,
        }
    }
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    pub enabled: bool,
    pub cache_size: usize,
    pub cache_gradients: bool,
    pub cache_intermediate: bool,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_size: 1000,
            cache_gradients: true,
            cache_intermediate: false,
        }
    }
}
