//! Core types and configurations for Quantum Machine Learning Layers
//!
//! This module contains all the basic types, enums, and configuration structures
//! used throughout the QML layers framework.

use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
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
            layer_configs: vec![
                QMLLayerConfig {
                    layer_type: QMLLayerType::ParameterizedQuantumCircuit,
                    num_parameters: 16,
                    ansatz_type: AnsatzType::Hardware,
                    entanglement_pattern: EntanglementPattern::Linear,
                    rotation_gates: vec![RotationGate::RY, RotationGate::RZ],
                    depth: 4,
                    enable_gradient_computation: true,
                },
            ],
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
    /// Variational Quantum Circuit (VQC)
    VariationalQuantumCircuit,
    /// Quantum Convolutional Neural Network
    QuantumConvolutionalNN,
    /// Quantum Recurrent Neural Network
    QuantumRecurrentNN,
    /// Quantum Graph Neural Network
    QuantumGraphNN,
    /// Quantum Attention Network
    QuantumAttentionNetwork,
    /// Quantum Transformer
    QuantumTransformer,
    /// Hybrid Classical-Quantum Network
    HybridClassicalQuantum,
    /// Quantum Boltzmann Machine
    QuantumBoltzmannMachine,
    /// Quantum Generative Adversarial Network
    QuantumGAN,
    /// Quantum Autoencoder
    QuantumAutoencoder,
}

/// QML layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLLayerConfig {
    /// Type of QML layer
    pub layer_type: QMLLayerType,
    /// Number of trainable parameters
    pub num_parameters: usize,
    /// Ansatz type for parameterized circuits
    pub ansatz_type: AnsatzType,
    /// Entanglement pattern
    pub entanglement_pattern: EntanglementPattern,
    /// Rotation gates to use
    pub rotation_gates: Vec<RotationGate>,
    /// Circuit depth
    pub depth: usize,
    /// Enable gradient computation
    pub enable_gradient_computation: bool,
}

/// Types of QML layers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QMLLayerType {
    /// Parameterized Quantum Circuit layer
    ParameterizedQuantumCircuit,
    /// Quantum Convolutional layer
    QuantumConvolutional,
    /// Quantum Pooling layer
    QuantumPooling,
    /// Quantum Dense layer (fully connected)
    QuantumDense,
    /// Quantum LSTM layer
    QuantumLSTM,
    /// Quantum GRU layer
    QuantumGRU,
    /// Quantum Attention layer
    QuantumAttention,
    /// Quantum Dropout layer
    QuantumDropout,
    /// Quantum Batch Normalization layer
    QuantumBatchNorm,
    /// Data Re-uploading layer
    DataReUpload,
}

/// Ansatz types for parameterized quantum circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnsatzType {
    /// Hardware-efficient ansatz
    Hardware,
    /// Problem-specific ansatz
    ProblemSpecific,
    /// All-to-all connectivity ansatz
    AllToAll,
    /// Layered ansatz
    Layered,
    /// Alternating ansatz
    Alternating,
    /// Brick-wall ansatz
    BrickWall,
    /// Tree ansatz
    Tree,
    /// Custom ansatz
    Custom,
}

/// Entanglement patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntanglementPattern {
    /// Linear entanglement chain
    Linear,
    /// Circular entanglement
    Circular,
    /// All-to-all entanglement
    AllToAll,
    /// Star topology entanglement
    Star,
    /// Grid topology entanglement
    Grid,
    /// Random entanglement
    Random,
    /// Block entanglement
    Block,
    /// Custom pattern
    Custom,
}

/// Rotation gates for parameterized circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotationGate {
    /// Rotation around X-axis
    RX,
    /// Rotation around Y-axis
    RY,
    /// Rotation around Z-axis
    RZ,
    /// Arbitrary single-qubit rotation
    U3,
    /// Phase gate
    Phase,
}

/// QML training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLTrainingConfig {
    /// Training algorithm type
    pub algorithm: QMLTrainingAlgorithm,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Gradient computation method
    pub gradient_method: GradientMethod,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Regularization parameters
    pub regularization: RegularizationConfig,
    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,
    /// Learning rate scheduling
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
    /// Parameter-shift rule gradient descent
    ParameterShift,
    /// Finite difference gradient descent
    FiniteDifference,
    /// Quantum Natural Gradient
    QuantumNaturalGradient,
    /// SPSA (Simultaneous Perturbation Stochastic Approximation)
    SPSA,
    /// Quantum Approximate Optimization Algorithm
    QAOA,
    /// Variational Quantum Eigensolver
    VQE,
    /// Quantum Machine Learning with Rotosolve
    Rotosolve,
    /// Hybrid Classical-Quantum training
    HybridTraining,
}

/// Gradient computation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientMethod {
    /// Parameter-shift rule
    ParameterShift,
    /// Finite difference
    FiniteDifference,
    /// Adjoint differentiation
    Adjoint,
    /// Backpropagation through quantum circuit
    Backpropagation,
    /// Quantum Fisher Information
    QuantumFisherInformation,
}

/// Optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// Adam optimizer
    Adam,
    /// AdaGrad optimizer
    AdaGrad,
    /// RMSprop optimizer
    RMSprop,
    /// Momentum optimizer
    Momentum,
    /// L-BFGS optimizer
    LBFGS,
    /// Quantum Natural Gradient
    QuantumNaturalGradient,
    /// SPSA optimizer
    SPSA,
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_strength: f64,
    /// L2 regularization strength
    pub l2_strength: f64,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Parameter constraint bounds
    pub parameter_bounds: Option<(f64, f64)>,
    /// Enable parameter clipping
    pub enable_clipping: bool,
    /// Gradient clipping threshold
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
    /// Enable early stopping
    pub enabled: bool,
    /// Patience (number of epochs without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: f64,
    /// Metric to monitor
    pub monitor_metric: String,
    /// Whether higher values are better
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
    /// Constant learning rate
    Constant,
    /// Exponential decay
    ExponentialDecay,
    /// Step decay
    StepDecay,
    /// Cosine annealing
    CosineAnnealing,
    /// Warm restart
    WarmRestart,
    /// Reduce on plateau
    ReduceOnPlateau,
}

// Hardware optimization and other configuration types would continue here...
// This is a partial extraction focusing on the core types.

/// Hardware optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimizationConfig {
    /// Target quantum hardware
    pub target_hardware: QuantumHardwareTarget,
    /// Enable gate count minimization
    pub minimize_gate_count: bool,
    /// Enable circuit depth minimization
    pub minimize_depth: bool,
    /// Enable noise-aware optimization
    pub noise_aware: bool,
    /// Connectivity constraints
    pub connectivity_constraints: ConnectivityConstraints,
    /// Gate fidelity constraints
    pub gate_fidelities: HashMap<String, f64>,
    /// Enable parallelization
    pub enable_parallelization: bool,
    /// Compilation optimization level
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
    /// Generic simulator
    Simulator,
    /// IBM Quantum devices
    IBM,
    /// Google Quantum AI devices
    Google,
    /// IonQ devices
    IonQ,
    /// Rigetti devices
    Rigetti,
    /// Honeywell/Quantinuum devices
    Quantinuum,
    /// Xanadu devices
    Xanadu,
    /// Custom hardware specification
    Custom,
}

/// Connectivity constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityConstraints {
    /// All-to-all connectivity
    AllToAll,
    /// Linear chain connectivity
    Linear,
    /// Grid connectivity
    Grid(usize, usize), // rows, cols
    /// Custom connectivity graph
    Custom(Vec<(usize, usize)>), // edge list
    /// Heavy-hex connectivity (IBM)
    HeavyHex,
    /// Square lattice connectivity
    Square,
}

/// Hardware optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareOptimizationLevel {
    /// Basic optimization
    Basic,
    /// Medium optimization
    Medium,
    /// Aggressive optimization
    Aggressive,
    /// Maximum optimization
    Maximum,
}

// Placeholder structs for other configuration types to maintain API compatibility
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClassicalPreprocessingConfig {
    pub feature_scaling: bool,
    pub encoding_method: DataEncodingMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataEncodingMethod {
    Amplitude,
    Angle,
    Basis,
}

impl Default for DataEncodingMethod {
    fn default() -> Self {
        Self::Amplitude
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HybridTrainingConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoiseAwareTrainingConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceOptimizationConfig {
    pub enable_caching: bool,
    pub memory_optimization: bool,
}