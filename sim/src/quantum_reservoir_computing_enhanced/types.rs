//! Type definitions for Quantum Reservoir Computing
//!
//! This module provides all enums and type definitions for the QRC framework.

use serde::{Deserialize, Serialize};

/// Advanced quantum reservoir architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumReservoirArchitecture {
    /// Random quantum circuit with tunable connectivity
    RandomCircuit,
    /// Spin chain with configurable interactions
    SpinChain,
    /// Transverse field Ising model with variable field strength
    TransverseFieldIsing,
    /// Small-world network with rewiring probability
    SmallWorld,
    /// Fully connected all-to-all interactions
    FullyConnected,
    /// Scale-free network following power-law degree distribution
    ScaleFree,
    /// Hierarchical modular architecture with multiple levels
    HierarchicalModular,
    /// Adaptive topology that evolves during computation
    AdaptiveTopology,
    /// Quantum cellular automaton structure
    QuantumCellularAutomaton,
    /// Ring topology with long-range connections
    Ring,
    /// Grid/lattice topology with configurable dimensions
    Grid,
    /// Tree topology with branching factor
    Tree,
    /// Hypergraph topology with higher-order interactions
    Hypergraph,
    /// Tensor network inspired architecture
    TensorNetwork,
    /// Custom user-defined architecture
    Custom,
}

/// Advanced reservoir dynamics types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReservoirDynamics {
    /// Unitary evolution with perfect coherence
    Unitary,
    /// Open system dynamics with Lindblad operators
    Open,
    /// Noisy intermediate-scale quantum (NISQ) dynamics
    NISQ,
    /// Adiabatic quantum evolution
    Adiabatic,
    /// Floquet dynamics with periodic driving
    Floquet,
    /// Quantum walk dynamics
    QuantumWalk,
    /// Continuous-time quantum dynamics
    ContinuousTime,
    /// Digital quantum simulation with Trotter decomposition
    DigitalQuantum,
    /// Variational quantum dynamics
    Variational,
    /// Hamiltonian learning dynamics
    HamiltonianLearning,
    /// Many-body localized dynamics
    ManyBodyLocalized,
    /// Quantum chaotic dynamics
    QuantumChaotic,
}

/// Advanced input encoding methods for temporal data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputEncoding {
    /// Amplitude encoding with normalization
    Amplitude,
    /// Phase encoding with full 2π range
    Phase,
    /// Basis state encoding with binary representation
    BasisState,
    /// Coherent state encoding with displacement
    Coherent,
    /// Squeezed state encoding with squeezing parameter
    Squeezed,
    /// Angle encoding with rotation gates
    Angle,
    /// IQP encoding with diagonal unitaries
    IQP,
    /// Data re-uploading with multiple layers
    DataReUploading,
    /// Quantum feature map encoding
    QuantumFeatureMap,
    /// Variational encoding with trainable parameters
    VariationalEncoding,
    /// Temporal encoding with time-dependent parameters
    TemporalEncoding,
    /// Fourier encoding for frequency domain
    FourierEncoding,
    /// Wavelet encoding for multi-resolution
    WaveletEncoding,
    /// Haar random encoding
    HaarRandom,
    /// Graph encoding for structured data
    GraphEncoding,
}

/// Advanced output measurement strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputMeasurement {
    /// Pauli expectation values (X, Y, Z)
    PauliExpectation,
    /// Computational basis probability measurements
    Probability,
    /// Two-qubit correlation functions
    Correlations,
    /// Entanglement entropy and concurrence
    Entanglement,
    /// State fidelity with reference states
    Fidelity,
    /// Quantum Fisher information
    QuantumFisherInformation,
    /// Variance of observables
    Variance,
    /// Higher-order moments and cumulants
    HigherOrderMoments,
    /// Spectral properties and eigenvalues
    SpectralProperties,
    /// Quantum coherence measures
    QuantumCoherence,
    /// Purity and mixedness measures
    Purity,
    /// Quantum mutual information
    QuantumMutualInformation,
    /// Process tomography observables
    ProcessTomography,
    /// Temporal correlations
    TemporalCorrelations,
    /// Non-linear readout functions
    NonLinearReadout,
}

/// Advanced learning algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    /// Ridge regression with L2 regularization
    Ridge,
    /// LASSO regression with L1 regularization
    LASSO,
    /// Elastic Net combining L1 and L2 regularization
    ElasticNet,
    /// Recursive Least Squares with forgetting factor
    RecursiveLeastSquares,
    /// Kalman filter for adaptive learning
    KalmanFilter,
    /// Extended Kalman filter for nonlinear systems
    ExtendedKalmanFilter,
    /// Neural network readout layer
    NeuralNetwork,
    /// Support Vector Regression
    SupportVectorRegression,
    /// Gaussian Process regression
    GaussianProcess,
    /// Random Forest regression
    RandomForest,
    /// Gradient boosting regression
    GradientBoosting,
    /// Online gradient descent
    OnlineGradientDescent,
    /// Adam optimizer
    Adam,
    /// Meta-learning approach
    MetaLearning,
}

/// Neural network activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// Rectified Linear Unit
    ReLU,
    /// Leaky `ReLU`
    LeakyReLU,
    /// Exponential Linear Unit
    ELU,
    /// Sigmoid activation
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Swish activation
    Swish,
    /// GELU activation
    GELU,
    /// Linear activation
    Linear,
}

/// Memory kernel types for time series modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryKernel {
    /// Exponential decay kernel
    Exponential,
    /// Power law kernel
    PowerLaw,
    /// Gaussian kernel
    Gaussian,
    /// Polynomial kernel
    Polynomial,
    /// Rational kernel
    Rational,
    /// Sinusoidal kernel
    Sinusoidal,
    /// Custom kernel
    Custom,
}

/// Memory capacity test tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTask {
    /// Delay line memory
    DelayLine,
    /// Temporal XOR task
    TemporalXOR,
    /// Parity check task
    Parity,
    /// Sequence prediction
    SequencePrediction,
    /// Pattern completion
    PatternCompletion,
    /// Temporal integration
    TemporalIntegration,
}

/// Information processing capacity functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IPCFunction {
    /// Linear function
    Linear,
    /// Quadratic function
    Quadratic,
    /// Cubic function
    Cubic,
    /// Sine function
    Sine,
    /// Product function
    Product,
    /// XOR function
    XOR,
}
