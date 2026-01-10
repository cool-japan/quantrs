//! Configuration types for Quantum Reservoir Computing
//!
//! This module provides all configuration structs for the QRC framework.

use serde::{Deserialize, Serialize};

use super::types::{
    ActivationFunction, IPCFunction, InputEncoding, LearningAlgorithm, MemoryKernel, MemoryTask,
    OutputMeasurement, QuantumReservoirArchitecture, ReservoirDynamics,
};

/// Enhanced quantum reservoir computing configuration
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
    /// Memory analysis configuration
    pub memory_config: MemoryAnalysisConfig,
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

impl Default for QuantumReservoirConfig {
    fn default() -> Self {
        Self {
            num_qubits: 8,
            architecture: QuantumReservoirArchitecture::RandomCircuit,
            dynamics: ReservoirDynamics::Unitary,
            input_encoding: InputEncoding::Amplitude,
            output_measurement: OutputMeasurement::PauliExpectation,
            learning_config: AdvancedLearningConfig::default(),
            time_series_config: TimeSeriesConfig::default(),
            memory_config: MemoryAnalysisConfig::default(),
            time_step: 0.1,
            evolution_steps: 10,
            coupling_strength: 1.0,
            noise_level: 0.01,
            memory_capacity: 100,
            adaptive_learning: true,
            learning_rate: 0.01,
            washout_period: 50,
            random_seed: None,
            enable_qec: false,
            precision: 1e-8,
        }
    }
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

impl Default for AdvancedLearningConfig {
    fn default() -> Self {
        Self {
            algorithm: LearningAlgorithm::Ridge,
            regularization: 1e-6,
            l1_ratio: 0.5,
            forgetting_factor: 0.99,
            process_noise: 1e-4,
            measurement_noise: 1e-3,
            nn_architecture: vec![64, 32, 16],
            nn_activation: ActivationFunction::ReLU,
            epochs: 100,
            batch_size: 32,
            early_stopping_patience: 10,
            cv_folds: 5,
            enable_ensemble: false,
            ensemble_size: 5,
        }
    }
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
    /// Enable change point detection
    pub enable_changepoint: bool,
    /// Anomaly detection threshold
    pub anomaly_threshold: f64,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            enable_arima: true,
            ar_order: 2,
            ma_order: 1,
            diff_order: 1,
            enable_nar: true,
            nar_order: 3,
            memory_kernel: MemoryKernel::Exponential,
            kernel_params: vec![0.9, 0.1],
            enable_seasonal: false,
            seasonal_period: 12,
            enable_changepoint: false,
            anomaly_threshold: 2.0,
        }
    }
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
}

impl Default for MemoryAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_capacity_estimation: true,
            capacity_tasks: vec![
                MemoryTask::DelayLine,
                MemoryTask::TemporalXOR,
                MemoryTask::Parity,
            ],
            enable_nonlinear: true,
            nonlinearity_orders: vec![2, 3, 4],
            enable_temporal_correlation: true,
            correlation_lags: (1..=20).collect(),
            enable_ipc: true,
            ipc_functions: vec![
                IPCFunction::Linear,
                IPCFunction::Quadratic,
                IPCFunction::Cubic,
            ],
            enable_entropy: true,
        }
    }
}
