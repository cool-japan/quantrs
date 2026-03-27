//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::hardware_aware_qml::AdaptationState;
use crate::quantum_reservoir_computing_enhanced::MemoryMetrics;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

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
/// Connectivity constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectivityConstraints {
    /// All-to-all connectivity
    AllToAll,
    /// Linear chain
    Linear,
    /// 2D grid
    Grid2D,
    /// Heavy-hex lattice
    HeavyHex,
    /// Custom topology
    Custom,
}
/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Training error (RMSE)
    pub training_error: f64,
    /// Test error (RMSE)
    pub test_error: f64,
    /// Training time in milliseconds
    pub training_time_ms: f64,
    /// Number of training examples
    pub num_examples: usize,
    /// Echo state property measure
    pub echo_state_property: f64,
}
/// Time series modeling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    /// Enable ARIMA-like modeling
    pub enable_arima: bool,
    /// AR order (autoregressive)
    pub ar_order: usize,
    /// MA order (moving average)
    pub ma_order: usize,
    /// Differencing order
    pub diff_order: usize,
    /// Enable nonlinear autoregressive model
    pub enable_nar: bool,
    /// NAR model order
    pub nar_order: usize,
    /// Memory kernel type
    pub memory_kernel: MemoryKernel,
    /// Kernel parameters
    pub kernel_params: Vec<f64>,
    /// Enable seasonal decomposition
    pub enable_seasonal: bool,
    /// Seasonal period
    pub seasonal_period: usize,
    /// Trend detection method
    pub trend_method: TrendDetectionMethod,
    /// Enable change point detection
    pub enable_changepoint: bool,
    /// Anomaly detection threshold
    pub anomaly_threshold: f64,
}
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
/// Adaptive learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningConfig {
    /// Enable online learning
    pub enable_online: bool,
    /// Learning rate decay schedule
    pub lr_schedule: LearningRateSchedule,
    /// Initial learning rate
    pub initial_lr: f64,
    /// Minimum learning rate
    pub min_lr: f64,
    /// Learning rate decay factor
    pub lr_decay: f64,
    /// Adaptation window size
    pub adaptation_window: usize,
    /// Plasticity mechanisms
    pub plasticity_type: PlasticityType,
    /// Homeostatic regulation
    pub enable_homeostasis: bool,
    /// Target activity level
    pub target_activity: f64,
    /// Activity regulation rate
    pub regulation_rate: f64,
    /// Enable meta-learning
    pub enable_meta_learning: bool,
    /// Meta-learning update frequency
    pub meta_update_frequency: usize,
}
/// Memory analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysisConfig {
    /// Enable memory capacity estimation
    pub enable_capacity_estimation: bool,
    /// Memory capacity test tasks
    pub capacity_tasks: Vec<MemoryTask>,
    /// Enable nonlinear memory analysis
    pub enable_nonlinear: bool,
    /// Nonlinearity test orders
    pub nonlinearity_orders: Vec<usize>,
    /// Enable temporal correlation analysis
    pub enable_temporal_correlation: bool,
    /// Correlation lag range
    pub correlation_lags: Vec<usize>,
    /// Information processing capacity
    pub enable_ipc: bool,
    /// IPC test functions
    pub ipc_functions: Vec<IPCFunction>,
    /// Enable entropy analysis
    pub enable_entropy: bool,
    /// Entropy measures
    pub entropy_measures: Vec<EntropyMeasure>,
}
/// Error mitigation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorMitigationMethod {
    /// Zero noise extrapolation
    ZNE,
    /// Probabilistic error cancellation
    PEC,
    /// Readout error mitigation
    ReadoutCorrection,
    /// Symmetry verification
    SymmetryVerification,
    /// Virtual distillation
    VirtualDistillation,
    /// Measurement error mitigation
    MeasurementCorrection,
}
/// Native quantum gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeGate {
    /// Rotation around Z axis
    RZ,
    /// Square root of X gate
    SX,
    /// CNOT gate
    CNOT,
    /// CZ gate
    CZ,
    /// iSWAP gate
    ISwap,
    /// Molmer-Sorensen gate
    MS,
    /// Arbitrary rotation
    U3,
}
/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingConfig {
    /// Enable comprehensive benchmarking
    pub enable_comprehensive: bool,
    /// Benchmark datasets
    pub datasets: Vec<BenchmarkDataset>,
    /// Performance metrics
    pub metrics: Vec<PerformanceMetric>,
    /// Statistical tests
    pub statistical_tests: Vec<StatisticalTest>,
    /// Comparison methods
    pub comparison_methods: Vec<ComparisonMethod>,
    /// Number of benchmark runs
    pub num_runs: usize,
    /// Confidence level for statistics
    pub confidence_level: f64,
    /// Enable cross-validation
    pub enable_cross_validation: bool,
    /// Cross-validation strategy
    pub cv_strategy: CrossValidationStrategy,
}
/// Trend detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDetectionMethod {
    /// Linear regression trend
    LinearRegression,
    /// Polynomial trend fitting
    Polynomial,
    /// Moving average trend
    MovingAverage,
    /// Hodrick-Prescott filter
    HodrickPrescott,
    /// Kalman filter trend
    KalmanFilter,
    /// Spectral analysis
    Spectral,
}
/// Benchmark datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkDataset {
    /// Mackey-Glass time series
    MackeyGlass,
    /// Lorenz attractor
    Lorenz,
    /// Sine wave with noise
    Sine,
    /// Chaotic time series
    Chaotic,
    /// Stock market data
    Financial,
    /// Weather data
    Weather,
    /// EEG signal
    EEG,
    /// Speech recognition
    Speech,
    /// Synthetic nonlinear
    SyntheticNonlinear,
}
/// Cross-validation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation
    KFold,
    /// Time series split
    TimeSeriesSplit,
    /// Leave-one-out
    LeaveOneOut,
    /// Stratified K-fold
    StratifiedKFold,
    /// Group K-fold
    GroupKFold,
}
/// Learning rate schedules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant,
    /// Exponential decay
    Exponential,
    /// Step decay
    Step,
    /// Polynomial decay
    Polynomial,
    /// Cosine annealing
    CosineAnnealing,
    /// Warm restart
    WarmRestart,
    /// Adaptive based on performance
    Adaptive,
}
/// Hardware optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimizationConfig {
    /// Target quantum platform
    pub platform: QuantumPlatform,
    /// Enable noise-aware training
    pub enable_noise_aware: bool,
    /// Error mitigation methods
    pub error_mitigation: Vec<ErrorMitigationMethod>,
    /// Enable circuit optimization
    pub enable_circuit_optimization: bool,
    /// Gate set optimization
    pub native_gate_set: Vec<NativeGate>,
    /// Connectivity constraints
    pub connectivity_constraints: ConnectivityConstraints,
    /// Enable device calibration
    pub enable_calibration: bool,
    /// Calibration frequency
    pub calibration_frequency: usize,
    /// Performance monitoring
    pub enable_monitoring: bool,
    /// Real-time adaptation to hardware
    pub enable_hardware_adaptation: bool,
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
/// Plasticity mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlasticityType {
    /// Hebbian learning
    Hebbian,
    /// Anti-Hebbian learning
    AntiHebbian,
    /// Spike-timing dependent plasticity
    STDP,
    /// Homeostatic scaling
    Homeostatic,
    /// Metaplasticity
    Metaplasticity,
    /// Quantum plasticity
    Quantum,
}
/// Entropy measures for memory analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntropyMeasure {
    /// Shannon entropy
    Shannon,
    /// Renyi entropy
    Renyi,
    /// Von Neumann entropy
    VonNeumann,
    /// Tsallis entropy
    Tsallis,
    /// Mutual information
    MutualInformation,
    /// Transfer entropy
    TransferEntropy,
}
/// Statistical tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticalTest {
    /// Student's t-test
    TTest,
    /// Wilcoxon rank-sum test
    WilcoxonRankSum,
    /// Kruskal-Wallis test
    KruskalWallis,
    /// ANOVA
    ANOVA,
    /// Friedman test
    Friedman,
    /// Bootstrap test
    Bootstrap,
}
/// Advanced learning algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedLearningConfig {
    /// Primary learning algorithm
    pub algorithm: LearningAlgorithm,
    /// Regularization parameter (lambda)
    pub regularization: f64,
    /// L1 ratio for Elastic Net (0.0 = Ridge, 1.0 = LASSO)
    pub l1_ratio: f64,
    /// Forgetting factor for RLS
    pub forgetting_factor: f64,
    /// Process noise for Kalman filter
    pub process_noise: f64,
    /// Measurement noise for Kalman filter
    pub measurement_noise: f64,
    /// Neural network architecture
    pub nn_architecture: Vec<usize>,
    /// Neural network activation function
    pub nn_activation: ActivationFunction,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Enable ensemble methods
    pub enable_ensemble: bool,
    /// Number of ensemble members
    pub ensemble_size: usize,
}
/// Enhanced quantum reservoir state
#[derive(Debug, Clone)]
pub struct QuantumReservoirState {
    /// Current quantum state vector
    pub state_vector: Array1<Complex64>,
    /// Evolution history buffer
    pub state_history: VecDeque<Array1<Complex64>>,
    /// Observable measurements cache
    pub observables: HashMap<String, f64>,
    /// Two-qubit correlation matrix
    pub correlations: Array2<f64>,
    /// Higher-order correlations
    pub higher_order_correlations: HashMap<String, f64>,
    /// Entanglement measures
    pub entanglement_measures: HashMap<String, f64>,
    /// Memory capacity metrics
    pub memory_metrics: MemoryMetrics,
    /// Time index counter
    pub time_index: usize,
    /// Last update timestamp
    pub last_update: f64,
    /// Reservoir activity level
    pub activity_level: f64,
    /// Adaptation state
    pub adaptation_state: AdaptationState,
    /// Performance tracking
    pub performance_history: VecDeque<f64>,
}
impl QuantumReservoirState {
    /// Create new reservoir state
    #[must_use]
    pub fn new(num_qubits: usize, memory_capacity: usize) -> Self {
        let state_size = 1 << num_qubits;
        let mut state_vector = Array1::zeros(state_size);
        state_vector[0] = Complex64::new(1.0, 0.0);
        Self {
            state_vector,
            state_history: VecDeque::with_capacity(memory_capacity),
            observables: HashMap::new(),
            correlations: Array2::zeros((num_qubits, num_qubits)),
            higher_order_correlations: HashMap::new(),
            entanglement_measures: HashMap::new(),
            memory_metrics: MemoryMetrics::default(),
            time_index: 0,
            last_update: 0.0,
            activity_level: 0.0,
            adaptation_state: AdaptationState::default(),
            performance_history: VecDeque::with_capacity(memory_capacity),
        }
    }
    /// Update state and maintain history
    pub fn update_state(&mut self, new_state: Array1<Complex64>) {
        self.state_history.push_back(self.state_vector.clone());
        if self.state_history.len() > self.state_history.capacity() {
            self.state_history.pop_front();
        }
        self.state_vector = new_state;
        self.time_index += 1;
    }
}
/// Training example for reservoir learning
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input data
    pub input: Array1<f64>,
    /// Reservoir state after processing
    pub reservoir_state: Array1<f64>,
    /// Target output
    pub target: Array1<f64>,
    /// Prediction error
    pub error: f64,
}
/// Performance metrics for reservoir computing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReservoirMetrics {
    /// Total training examples processed
    pub training_examples: usize,
    /// Current prediction accuracy
    pub prediction_accuracy: f64,
    /// Memory capacity estimate
    pub memory_capacity: f64,
    /// Information processing capacity
    pub processing_capacity: f64,
    /// Generalization error
    pub generalization_error: f64,
    /// Echo state property indicator
    pub echo_state_property: f64,
    /// Average processing time per input
    pub avg_processing_time_ms: f64,
    /// Quantum resource utilization
    pub quantum_resource_usage: f64,
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
/// Comparison methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonMethod {
    /// Echo State Network
    ESN,
    /// Long Short-Term Memory
    LSTM,
    /// Gated Recurrent Unit
    GRU,
    /// Transformer
    Transformer,
    /// Support Vector Machine
    SVM,
    /// Random Forest
    RandomForest,
    /// Linear regression
    LinearRegression,
}
/// Quantum computing platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumPlatform {
    /// Classical simulator
    Simulator,
    /// IBM Quantum
    IBM,
    /// Google Quantum AI
    Google,
    /// `IonQ` trapped ion
    IonQ,
    /// Rigetti superconducting
    Rigetti,
    /// Quantinuum trapped ion
    Quantinuum,
    /// Xanadu photonic
    Xanadu,
    /// Atom Computing neutral atom
    AtomComputing,
    /// Generic NISQ device
    GenericNISQ,
}
/// Training data for reservoir computing
#[derive(Debug, Clone)]
pub struct ReservoirTrainingData {
    /// Input time series
    pub inputs: Vec<Array1<f64>>,
    /// Target outputs
    pub targets: Vec<Array1<f64>>,
    /// Time stamps
    pub timestamps: Vec<f64>,
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
/// Topology and connectivity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Connectivity density (0.0 to 1.0)
    pub connectivity_density: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Small-world rewiring probability
    pub rewiring_probability: f64,
    /// Scale-free power law exponent
    pub power_law_exponent: f64,
    /// Number of hierarchical levels
    pub hierarchical_levels: usize,
    /// Modular structure parameters
    pub modularity_strength: f64,
    /// Number of modules
    pub num_modules: usize,
    /// Enable adaptive topology
    pub enable_adaptive: bool,
    /// Topology adaptation rate
    pub adaptation_rate: f64,
    /// Minimum connection strength
    pub min_connection_strength: f64,
    /// Maximum connection strength
    pub max_connection_strength: f64,
    /// Connection pruning threshold
    pub pruning_threshold: f64,
}
/// Advanced quantum reservoir computing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumReservoirConfig {
    /// Number of qubits in the reservoir
    pub num_qubits: usize,
    /// Reservoir architecture type
    pub architecture: QuantumReservoirArchitecture,
    /// Dynamics evolution type
    pub dynamics: ReservoirDynamics,
    /// Input encoding method
    pub input_encoding: InputEncoding,
    /// Output measurement strategy
    pub output_measurement: OutputMeasurement,
    /// Advanced learning algorithm configuration
    pub learning_config: AdvancedLearningConfig,
    /// Time series modeling configuration
    pub time_series_config: TimeSeriesConfig,
    /// Topology and connectivity configuration
    pub topology_config: TopologyConfig,
    /// Adaptive learning configuration
    pub adaptive_config: AdaptiveLearningConfig,
    /// Memory analysis configuration
    pub memory_config: MemoryAnalysisConfig,
    /// Hardware optimization configuration
    pub hardware_config: HardwareOptimizationConfig,
    /// Benchmarking configuration
    pub benchmark_config: BenchmarkingConfig,
    /// Time step for evolution
    pub time_step: f64,
    /// Number of evolution steps per input
    pub evolution_steps: usize,
    /// Reservoir coupling strength
    pub coupling_strength: f64,
    /// Noise level (for NISQ dynamics)
    pub noise_level: f64,
    /// Memory capacity (time steps to remember)
    pub memory_capacity: usize,
    /// Enable real-time adaptation
    pub adaptive_learning: bool,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Washout period (initial time steps to ignore)
    pub washout_period: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Enable quantum error correction
    pub enable_qec: bool,
    /// Precision for calculations
    pub precision: f64,
}
/// Performance metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Root Mean Squared Error
    RMSE,
    /// R-squared
    R2,
    /// Memory capacity
    MemoryCapacity,
    /// Processing speed
    ProcessingSpeed,
    /// Training time
    TrainingTime,
    /// Generalization error
    GeneralizationError,
    /// Information processing capacity
    IPC,
    /// Quantum advantage metric
    QuantumAdvantage,
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
