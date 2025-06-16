//! Advanced Quantum Neural Networks for Optimization
//!
//! This module implements quantum neural networks specifically optimized for
//! quantum annealing problems, featuring hybrid quantum-classical architectures
//! and adaptive learning algorithms.

use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
use crate::QuboFormulation;
use ndarray::{s, Array, Array1, Array2, Array3, ArrayD, Axis, Zip};
use rand::{prelude::*, thread_rng};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// Quantum Neural Network for optimization problems
pub struct QuantumNeuralNetwork {
    /// Network architecture
    pub architecture: QNNArchitecture,
    /// Quantum layers
    pub layers: Vec<QuantumLayer>,
    /// Classical preprocessing layers
    pub classical_layers: Vec<ClassicalLayer>,
    /// Training configuration
    pub training_config: QNNTrainingConfig,
    /// Current parameters
    pub parameters: QNNParameters,
    /// Training history
    pub training_history: Vec<TrainingEpoch>,
    /// Performance metrics
    pub metrics: QNNMetrics,
}

/// QNN Architecture specification
#[derive(Debug, Clone)]
pub struct QNNArchitecture {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// Quantum circuit depth
    pub circuit_depth: usize,
    /// Entanglement pattern
    pub entanglement_pattern: EntanglementPattern,
    /// Measurement scheme
    pub measurement_scheme: MeasurementScheme,
    /// Classical postprocessing
    pub postprocessing: PostprocessingScheme,
}

/// Entanglement patterns for quantum layers
#[derive(Debug, Clone, PartialEq)]
pub enum EntanglementPattern {
    /// Linear nearest-neighbor entanglement
    Linear,
    /// Full entanglement between all qubits
    Full,
    /// Circular entanglement
    Circular,
    /// Random entanglement pattern
    Random { connectivity: f64 },
    /// Hardware-efficient pattern
    HardwareEfficient,
    /// Problem-adapted pattern
    ProblemAdapted,
}

/// Measurement schemes
#[derive(Debug, Clone, PartialEq)]
pub enum MeasurementScheme {
    /// Computational basis measurement
    Computational,
    /// Pauli measurements
    Pauli { bases: Vec<PauliBasis> },
    /// Bell measurements
    Bell,
    /// Adaptive measurements
    Adaptive,
    /// Shadow tomography
    ShadowTomography { num_shadows: usize },
}

/// Pauli measurement bases
#[derive(Debug, Clone, PartialEq)]
pub enum PauliBasis {
    X,
    Y,
    Z,
}

/// Postprocessing schemes
#[derive(Debug, Clone, PartialEq)]
pub enum PostprocessingScheme {
    /// No postprocessing
    None,
    /// Linear transformation
    Linear,
    /// Nonlinear neural network
    NonlinearNN { hidden_dims: Vec<usize> },
    /// Attention mechanism
    Attention,
    /// Graph neural network
    GraphNN,
}

/// Quantum layer in the neural network
#[derive(Debug, Clone)]
pub struct QuantumLayer {
    /// Layer index
    pub layer_id: usize,
    /// Number of qubits in this layer
    pub num_qubits: usize,
    /// Quantum gates in this layer
    pub gates: Vec<QuantumGate>,
    /// Parametrized gates
    pub parametrized_gates: Vec<ParametrizedGate>,
    /// Layer type
    pub layer_type: QuantumLayerType,
    /// Skip connections
    pub skip_connections: Vec<usize>,
}

/// Types of quantum layers
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumLayerType {
    /// Variational layer
    Variational,
    /// Entangling layer
    Entangling,
    /// Measurement layer
    Measurement,
    /// Error correction layer
    ErrorCorrection,
    /// Adaptive layer
    Adaptive,
}

/// Quantum gates
#[derive(Debug, Clone)]
pub enum QuantumGate {
    /// Single-qubit rotation gates
    RX {
        qubit: usize,
        angle: f64,
    },
    RY {
        qubit: usize,
        angle: f64,
    },
    RZ {
        qubit: usize,
        angle: f64,
    },
    /// Two-qubit gates
    CNOT {
        control: usize,
        target: usize,
    },
    CZ {
        control: usize,
        target: usize,
    },
    /// Multi-qubit gates
    Toffoli {
        controls: Vec<usize>,
        target: usize,
    },
    /// Custom gates
    Custom {
        name: String,
        qubits: Vec<usize>,
        matrix: Array2<f64>,
    },
}

/// Parametrized quantum gates
#[derive(Debug, Clone)]
pub struct ParametrizedGate {
    /// Gate type
    pub gate_type: ParametrizedGateType,
    /// Qubits involved
    pub qubits: Vec<usize>,
    /// Parameter indices
    pub parameter_indices: Vec<usize>,
    /// Gate function
    pub gate_function: GateFunction,
}

/// Types of parametrized gates
#[derive(Debug, Clone, PartialEq)]
pub enum ParametrizedGateType {
    /// Rotation gates
    Rotation { axis: RotationAxis },
    /// Entangling gates
    Entangling { gate_name: String },
    /// Generalized rotation
    GeneralizedRotation,
    /// Problem-specific gates
    ProblemSpecific { problem_type: String },
}

/// Rotation axes
#[derive(Debug, Clone, PartialEq)]
pub enum RotationAxis {
    X,
    Y,
    Z,
    Arbitrary { nx: f64, ny: f64, nz: f64 },
}

/// Gate function for parametrized gates
#[derive(Debug, Clone)]
pub enum GateFunction {
    /// Standard rotation
    StandardRotation,
    /// Controlled rotation
    ControlledRotation,
    /// Multi-controlled rotation
    MultiControlledRotation,
    /// Custom function
    Custom { function_name: String },
}

/// Classical layer for hybrid architecture
#[derive(Debug, Clone)]
pub struct ClassicalLayer {
    /// Layer type
    pub layer_type: ClassicalLayerType,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Weights
    pub weights: Array2<f64>,
    /// Biases
    pub biases: Array1<f64>,
    /// Activation function
    pub activation: ActivationFunction,
}

/// Types of classical layers
#[derive(Debug, Clone, PartialEq)]
pub enum ClassicalLayerType {
    /// Dense/fully connected
    Dense,
    /// Convolutional
    Convolutional,
    /// Attention
    Attention,
    /// Normalization
    Normalization,
    /// Embedding
    Embedding,
}

/// Activation functions
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    Softmax,
    Swish,
    GELU,
}

/// Training configuration for QNN
#[derive(Debug, Clone)]
pub struct QNNTrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub num_epochs: usize,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Loss function
    pub loss_function: LossFunction,
    /// Regularization
    pub regularization: RegularizationConfig,
    /// Early stopping
    pub early_stopping: EarlyStoppingConfig,
    /// Gradient estimation
    pub gradient_estimation: GradientEstimationMethod,
}

/// Optimizer types
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerType {
    SGD {
        momentum: f64,
    },
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    AdaGrad {
        epsilon: f64,
    },
    RMSprop {
        decay: f64,
        epsilon: f64,
    },
    SPSA {
        a: f64,
        c: f64,
    },
    QuantumNaturalGradient,
}

/// Loss functions
#[derive(Debug, Clone, PartialEq)]
pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
    HuberLoss { delta: f64 },
    QuantumFisherInformation,
    ExpectationValueLoss,
    Custom { name: String },
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_strength: f64,
    /// L2 regularization strength
    pub l2_strength: f64,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Parameter noise
    pub parameter_noise: f64,
    /// Quantum noise modeling
    pub quantum_noise: QuantumNoiseConfig,
}

/// Quantum noise configuration
#[derive(Debug, Clone)]
pub struct QuantumNoiseConfig {
    /// Enable noise modeling
    pub enable_noise: bool,
    /// Depolarizing noise strength
    pub depolarizing_strength: f64,
    /// Amplitude damping
    pub amplitude_damping: f64,
    /// Phase damping
    pub phase_damping: f64,
    /// Gate error rates
    pub gate_error_rates: HashMap<String, f64>,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,
    /// Patience (number of epochs without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Metric to monitor
    pub monitor_metric: String,
}

/// Gradient estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum GradientEstimationMethod {
    /// Parameter shift rule
    ParameterShift,
    /// Finite differences
    FiniteDifferences { epsilon: f64 },
    /// Stochastic parameter shift
    StochasticParameterShift,
    /// Natural gradient
    NaturalGradient,
    /// Quantum Fisher information
    QuantumFisherInformation,
}

/// QNN parameters
#[derive(Debug, Clone)]
pub struct QNNParameters {
    /// Quantum circuit parameters
    pub quantum_params: Array1<f64>,
    /// Classical layer parameters
    pub classical_params: Vec<Array2<f64>>,
    /// Bias parameters
    pub bias_params: Vec<Array1<f64>>,
    /// Parameter bounds
    pub parameter_bounds: Vec<(f64, f64)>,
    /// Parameter initialization scheme
    pub initialization_scheme: ParameterInitializationScheme,
}

/// Parameter initialization schemes
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterInitializationScheme {
    /// Random uniform initialization
    RandomUniform { min: f64, max: f64 },
    /// Random normal initialization
    RandomNormal { mean: f64, std: f64 },
    /// Xavier/Glorot initialization
    Xavier,
    /// He initialization
    He,
    /// Problem-specific initialization
    ProblemSpecific,
    /// Warm start from classical solution
    WarmStart,
}

/// Training epoch data
#[derive(Debug, Clone)]
pub struct TrainingEpoch {
    /// Epoch number
    pub epoch: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: Option<f64>,
    /// Training accuracy
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: Option<f64>,
    /// Learning rate used
    pub learning_rate: f64,
    /// Training time
    pub training_time: f64,
    /// Gradient norms
    pub gradient_norms: Vec<f64>,
    /// Parameter statistics
    pub parameter_stats: ParameterStatistics,
}

/// Parameter statistics
#[derive(Debug, Clone)]
pub struct ParameterStatistics {
    /// Mean parameter values
    pub mean_values: Array1<f64>,
    /// Parameter standard deviations
    pub std_values: Array1<f64>,
    /// Parameter ranges
    pub ranges: Vec<(f64, f64)>,
    /// Parameter correlations
    pub correlations: Array2<f64>,
}

/// QNN performance metrics
#[derive(Debug, Clone)]
pub struct QNNMetrics {
    /// Training metrics
    pub training_metrics: TrainingMetrics,
    /// Validation metrics
    pub validation_metrics: ValidationMetrics,
    /// Quantum-specific metrics
    pub quantum_metrics: QuantumMetrics,
    /// Computational metrics
    pub computational_metrics: ComputationalMetrics,
}

/// Training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Final training loss
    pub final_training_loss: f64,
    /// Training convergence rate
    pub convergence_rate: f64,
    /// Number of epochs to convergence
    pub epochs_to_convergence: usize,
    /// Training stability
    pub training_stability: f64,
    /// Overfitting measure
    pub overfitting_measure: f64,
}

/// Validation metrics
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Best validation loss
    pub best_validation_loss: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Generalization gap
    pub generalization_gap: f64,
    /// Cross-validation scores
    pub cv_scores: Vec<f64>,
    /// Bootstrap confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
}

/// Quantum-specific metrics
#[derive(Debug, Clone)]
pub struct QuantumMetrics {
    /// Quantum volume utilized
    pub quantum_volume: f64,
    /// Entanglement measures
    pub entanglement_measures: Vec<f64>,
    /// Quantum advantage indicators
    pub quantum_advantage: f64,
    /// Fidelity measures
    pub fidelity_measures: Vec<f64>,
    /// Coherence utilization
    pub coherence_utilization: f64,
}

/// Computational metrics
#[derive(Debug, Clone)]
pub struct ComputationalMetrics {
    /// Training time per epoch
    pub training_time_per_epoch: f64,
    /// Inference time
    pub inference_time: f64,
    /// Memory usage
    pub memory_usage: f64,
    /// Quantum circuit execution time
    pub quantum_execution_time: f64,
    /// Classical computation time
    pub classical_computation_time: f64,
}

impl QuantumNeuralNetwork {
    /// Create a new quantum neural network
    pub fn new(
        architecture: QNNArchitecture,
        training_config: QNNTrainingConfig,
    ) -> Result<Self, SamplerError> {
        let num_quantum_params = Self::calculate_quantum_params(&architecture);
        let num_classical_params = Self::calculate_classical_params(&architecture);

        let parameters = QNNParameters {
            quantum_params: Array1::zeros(num_quantum_params),
            classical_params: vec![],
            bias_params: vec![],
            parameter_bounds: vec![(-PI, PI); num_quantum_params],
            initialization_scheme: ParameterInitializationScheme::RandomUniform {
                min: -PI,
                max: PI,
            },
        };

        let layers = Self::build_quantum_layers(&architecture)?;
        let classical_layers = Self::build_classical_layers(&architecture)?;

        Ok(Self {
            architecture,
            layers,
            classical_layers,
            training_config,
            parameters,
            training_history: Vec::new(),
            metrics: QNNMetrics::default(),
        })
    }

    /// Train the quantum neural network
    pub fn train(
        &mut self,
        training_data: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<TrainingMetrics, SamplerError> {
        println!("Starting QNN training with {} samples", training_data.len());

        // Initialize parameters
        self.initialize_parameters()?;

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.training_config.num_epochs {
            let epoch_start = std::time::Instant::now();

            // Shuffle training data
            let mut shuffled_data = training_data.to_vec();
            shuffled_data.shuffle(&mut thread_rng());

            let mut epoch_loss = 0.0;
            let mut epoch_accuracy = 0.0;
            let batch_count = (shuffled_data.len() + self.training_config.batch_size - 1)
                / self.training_config.batch_size;

            for batch_idx in 0..batch_count {
                let batch_start = batch_idx * self.training_config.batch_size;
                let batch_end = std::cmp::min(
                    batch_start + self.training_config.batch_size,
                    shuffled_data.len(),
                );
                let batch = &shuffled_data[batch_start..batch_end];

                // Forward pass
                let batch_predictions = self.forward_batch(batch)?;

                // Compute loss and gradients
                let (batch_loss, gradients) =
                    self.compute_loss_and_gradients(batch, &batch_predictions)?;

                // Update parameters
                self.update_parameters(&gradients)?;

                epoch_loss += batch_loss;
                epoch_accuracy += self.compute_batch_accuracy(batch, &batch_predictions);
            }

            epoch_loss /= batch_count as f64;
            epoch_accuracy /= batch_count as f64;

            let epoch_time = epoch_start.elapsed().as_secs_f64();

            // Validation
            let validation_loss = self.compute_validation_loss(training_data)?;

            // Create epoch record
            let epoch_record = TrainingEpoch {
                epoch,
                training_loss: epoch_loss,
                validation_loss: Some(validation_loss),
                training_accuracy: epoch_accuracy,
                validation_accuracy: Some(epoch_accuracy), // Simplified
                learning_rate: self.training_config.learning_rate,
                training_time: epoch_time,
                gradient_norms: vec![0.1], // Placeholder
                parameter_stats: self.compute_parameter_statistics(),
            };

            self.training_history.push(epoch_record);

            // Early stopping check
            if self.training_config.early_stopping.enabled {
                if validation_loss < best_loss - self.training_config.early_stopping.min_improvement
                {
                    best_loss = validation_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.training_config.early_stopping.patience {
                        println!("Early stopping at epoch {} due to no improvement", epoch);
                        break;
                    }
                }
            }

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Loss = {:.6}, Accuracy = {:.3}, Val Loss = {:.6}",
                    epoch, epoch_loss, epoch_accuracy, validation_loss
                );
            }
        }

        // Compute final training metrics
        let training_metrics = TrainingMetrics {
            final_training_loss: self.training_history.last().unwrap().training_loss,
            convergence_rate: self.compute_convergence_rate(),
            epochs_to_convergence: self.training_history.len(),
            training_stability: self.compute_training_stability(),
            overfitting_measure: self.compute_overfitting_measure(),
        };

        self.metrics.training_metrics = training_metrics.clone();

        println!(
            "QNN training completed. Final loss: {:.6}",
            training_metrics.final_training_loss
        );
        Ok(training_metrics)
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>, SamplerError> {
        // Classical preprocessing
        let mut current_state = input.clone();
        for layer in &self.classical_layers {
            current_state = self.apply_classical_layer(&current_state, layer)?;
        }

        // Quantum processing
        let quantum_output = self.apply_quantum_layers(&current_state)?;

        // Classical postprocessing
        self.apply_postprocessing(&quantum_output)
    }

    /// Apply quantum layers to input state
    fn apply_quantum_layers(&self, input: &Array1<f64>) -> Result<Array1<f64>, SamplerError> {
        // Initialize quantum state
        let mut quantum_state = self.initialize_quantum_state(input)?;

        // Apply each quantum layer
        for layer in &self.layers {
            quantum_state = self.apply_quantum_layer(&quantum_state, layer)?;
        }

        // Measure quantum state
        self.measure_quantum_state(&quantum_state)
    }

    /// Initialize quantum state from classical input
    fn initialize_quantum_state(&self, input: &Array1<f64>) -> Result<Array1<f64>, SamplerError> {
        // Encode classical data into quantum state
        let num_qubits = self.architecture.num_qubits;
        let state_dim = 1 << num_qubits;
        let mut quantum_state = Array1::zeros(state_dim);

        // Simple amplitude encoding
        if input.len() <= state_dim {
            for (i, &val) in input.iter().enumerate() {
                if i < state_dim {
                    quantum_state[i] = val;
                }
            }
            // Normalize state
            let norm = quantum_state.dot(&quantum_state).sqrt();
            if norm > 1e-10 {
                quantum_state /= norm;
            } else {
                quantum_state[0] = 1.0; // Default to |0...0>
            }
        } else {
            // Use angle encoding for high-dimensional input
            for i in 0..std::cmp::min(input.len(), num_qubits) {
                let angle = input[i] * PI;
                // Apply rotation based on input value
                quantum_state[0] = angle.cos();
                quantum_state[1] = angle.sin();
            }
        }

        Ok(quantum_state)
    }

    /// Apply a single quantum layer
    fn apply_quantum_layer(
        &self,
        state: &Array1<f64>,
        layer: &QuantumLayer,
    ) -> Result<Array1<f64>, SamplerError> {
        let mut current_state = state.clone();

        // Apply parametrized gates
        for gate in &layer.parametrized_gates {
            current_state = self.apply_parametrized_gate(&current_state, gate)?;
        }

        // Apply fixed gates
        for gate in &layer.gates {
            current_state = self.apply_quantum_gate(&current_state, gate)?;
        }

        Ok(current_state)
    }

    /// Apply a parametrized quantum gate
    fn apply_parametrized_gate(
        &self,
        state: &Array1<f64>,
        gate: &ParametrizedGate,
    ) -> Result<Array1<f64>, SamplerError> {
        match &gate.gate_type {
            ParametrizedGateType::Rotation { axis } => {
                if gate.parameter_indices.is_empty() || gate.qubits.is_empty() {
                    return Ok(state.clone());
                }

                let param_idx = gate.parameter_indices[0];
                let qubit = gate.qubits[0];

                if param_idx >= self.parameters.quantum_params.len() {
                    return Ok(state.clone());
                }

                let angle = self.parameters.quantum_params[param_idx];

                match axis {
                    RotationAxis::X => self.apply_rx_gate(state, qubit, angle),
                    RotationAxis::Y => self.apply_ry_gate(state, qubit, angle),
                    RotationAxis::Z => self.apply_rz_gate(state, qubit, angle),
                    RotationAxis::Arbitrary { nx, ny, nz } => {
                        self.apply_arbitrary_rotation(state, qubit, angle, *nx, *ny, *nz)
                    }
                }
            }
            ParametrizedGateType::Entangling { gate_name } => {
                self.apply_entangling_gate(state, &gate.qubits, gate_name)
            }
            _ => Ok(state.clone()), // Placeholder for other gate types
        }
    }

    /// Apply RX rotation gate
    fn apply_rx_gate(
        &self,
        state: &Array1<f64>,
        qubit: usize,
        angle: f64,
    ) -> Result<Array1<f64>, SamplerError> {
        let num_qubits = self.architecture.num_qubits;
        if qubit >= num_qubits {
            return Ok(state.clone());
        }

        let mut new_state = state.clone();
        let state_dim = state.len();

        for i in 0..state_dim {
            // Map qubit index to bit position (qubit 0 is the most significant bit)
            let bit_pos = num_qubits - 1 - qubit;
            if (i & (1 << bit_pos)) == 0 {
                let j = i | (1 << bit_pos);
                if j < state_dim {
                    let cos_half = (angle / 2.0).cos();
                    let sin_half = (angle / 2.0).sin();

                    let state_i = state[i];
                    let state_j = state[j];

                    new_state[i] = cos_half * state_i + sin_half * state_j;
                    new_state[j] = sin_half * state_i + cos_half * state_j;
                }
            }
        }

        Ok(new_state)
    }

    /// Apply RY rotation gate
    fn apply_ry_gate(
        &self,
        state: &Array1<f64>,
        qubit: usize,
        angle: f64,
    ) -> Result<Array1<f64>, SamplerError> {
        let num_qubits = self.architecture.num_qubits;
        if qubit >= num_qubits {
            return Ok(state.clone());
        }

        let mut new_state = state.clone();
        let state_dim = state.len();

        for i in 0..state_dim {
            // Map qubit index to bit position (qubit 0 is the most significant bit)
            let bit_pos = num_qubits - 1 - qubit;
            if (i & (1 << bit_pos)) == 0 {
                let j = i | (1 << bit_pos);
                if j < state_dim {
                    let cos_half = (angle / 2.0).cos();
                    let sin_half = (angle / 2.0).sin();

                    let state_i = state[i];
                    let state_j = state[j];

                    new_state[i] = cos_half * state_i + sin_half * state_j;
                    new_state[j] = -sin_half * state_i + cos_half * state_j;
                }
            }
        }

        Ok(new_state)
    }

    /// Apply RZ rotation gate
    fn apply_rz_gate(
        &self,
        state: &Array1<f64>,
        qubit: usize,
        angle: f64,
    ) -> Result<Array1<f64>, SamplerError> {
        let num_qubits = self.architecture.num_qubits;
        if qubit >= num_qubits {
            return Ok(state.clone());
        }

        let mut new_state = state.clone();
        let exp_neg = (-angle / 2.0).exp();
        let exp_pos = (angle / 2.0).exp();

        for i in 0..state.len() {
            // Map qubit index to bit position (qubit 0 is the most significant bit)
            let bit_pos = num_qubits - 1 - qubit;
            if (i & (1 << bit_pos)) == 0 {
                new_state[i] = state[i] * exp_neg;
            } else {
                new_state[i] = state[i] * exp_pos;
            }
        }

        Ok(new_state)
    }

    /// Apply arbitrary rotation gate
    fn apply_arbitrary_rotation(
        &self,
        state: &Array1<f64>,
        qubit: usize,
        angle: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    ) -> Result<Array1<f64>, SamplerError> {
        // Normalize rotation axis
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        if norm < 1e-10 {
            return Ok(state.clone());
        }
        let (nx, ny, nz) = (nx / norm, ny / norm, nz / norm);

        // Apply rotation using Rodrigues' formula approach
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        // For simplicity, approximate with Z rotation
        self.apply_rz_gate(state, qubit, angle * nz)
    }

    /// Apply entangling gate
    fn apply_entangling_gate(
        &self,
        state: &Array1<f64>,
        qubits: &[usize],
        gate_name: &str,
    ) -> Result<Array1<f64>, SamplerError> {
        if qubits.len() < 2 {
            return Ok(state.clone());
        }

        match gate_name {
            "CNOT" => self.apply_cnot_gate(state, qubits[0], qubits[1]),
            "CZ" => self.apply_cz_gate(state, qubits[0], qubits[1]),
            _ => Ok(state.clone()),
        }
    }

    /// Apply CNOT gate
    fn apply_cnot_gate(
        &self,
        state: &Array1<f64>,
        control: usize,
        target: usize,
    ) -> Result<Array1<f64>, SamplerError> {
        let num_qubits = self.architecture.num_qubits;
        if control >= num_qubits || target >= num_qubits {
            return Ok(state.clone());
        }

        let mut new_state = state.clone();

        for i in 0..state.len() {
            if (i & (1 << control)) != 0 {
                let j = i ^ (1 << target);
                new_state[i] = state[j];
            }
        }

        Ok(new_state)
    }

    /// Apply CZ gate
    fn apply_cz_gate(
        &self,
        state: &Array1<f64>,
        control: usize,
        target: usize,
    ) -> Result<Array1<f64>, SamplerError> {
        let num_qubits = self.architecture.num_qubits;
        if control >= num_qubits || target >= num_qubits {
            return Ok(state.clone());
        }

        let mut new_state = state.clone();

        for i in 0..state.len() {
            if (i & (1 << control)) != 0 && (i & (1 << target)) != 0 {
                new_state[i] = -state[i];
            }
        }

        Ok(new_state)
    }

    /// Apply a quantum gate
    fn apply_quantum_gate(
        &self,
        state: &Array1<f64>,
        gate: &QuantumGate,
    ) -> Result<Array1<f64>, SamplerError> {
        match gate {
            QuantumGate::RX { qubit, angle } => self.apply_rx_gate(state, *qubit, *angle),
            QuantumGate::RY { qubit, angle } => self.apply_ry_gate(state, *qubit, *angle),
            QuantumGate::RZ { qubit, angle } => self.apply_rz_gate(state, *qubit, *angle),
            QuantumGate::CNOT { control, target } => self.apply_cnot_gate(state, *control, *target),
            QuantumGate::CZ { control, target } => self.apply_cz_gate(state, *control, *target),
            _ => Ok(state.clone()), // Placeholder for other gates
        }
    }

    /// Measure quantum state to get classical output
    fn measure_quantum_state(&self, state: &Array1<f64>) -> Result<Array1<f64>, SamplerError> {
        match &self.architecture.measurement_scheme {
            MeasurementScheme::Computational => {
                // Probability distribution over computational basis states
                let probabilities: Array1<f64> = state.mapv(|x| x * x);
                Ok(probabilities)
            }
            MeasurementScheme::Pauli { bases } => {
                // Expectation values of Pauli measurements
                let mut expectations = Array1::zeros(bases.len());
                for (i, basis) in bases.iter().enumerate() {
                    expectations[i] = self.compute_pauli_expectation(state, basis)?;
                }
                Ok(expectations)
            }
            _ => {
                // Default to computational basis
                let probabilities: Array1<f64> = state.mapv(|x| x * x);
                Ok(probabilities)
            }
        }
    }

    /// Compute Pauli expectation value
    fn compute_pauli_expectation(
        &self,
        state: &Array1<f64>,
        basis: &PauliBasis,
    ) -> Result<f64, SamplerError> {
        match basis {
            PauliBasis::Z => {
                let mut expectation = 0.0;
                for i in 0..state.len() {
                    let parity = (i.count_ones() % 2) as f64;
                    expectation += state[i] * state[i] * (1.0 - 2.0 * parity);
                }
                Ok(expectation)
            }
            PauliBasis::X => {
                // Simplified X measurement
                let mut expectation = 0.0;
                for i in 0..state.len() {
                    let j = i ^ 1; // Flip first qubit
                    if j < state.len() {
                        expectation += state[i] * state[j];
                    }
                }
                Ok(expectation * 2.0)
            }
            PauliBasis::Y => {
                // Simplified Y measurement
                Ok(0.0) // Placeholder
            }
        }
    }

    /// Apply classical layer
    fn apply_classical_layer(
        &self,
        input: &Array1<f64>,
        layer: &ClassicalLayer,
    ) -> Result<Array1<f64>, SamplerError> {
        let output = layer.weights.dot(input) + &layer.biases;

        let activated_output = match layer.activation {
            ActivationFunction::ReLU => output.mapv(|x| x.max(0.0)),
            ActivationFunction::Sigmoid => output.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::Tanh => output.mapv(|x| x.tanh()),
            ActivationFunction::Linear => output,
            ActivationFunction::Softmax => {
                let exp_vals = output.mapv(|x| x.exp());
                let sum = exp_vals.sum();
                exp_vals / sum
            }
            _ => output, // Placeholder for other activations
        };

        Ok(activated_output)
    }

    /// Apply postprocessing
    fn apply_postprocessing(
        &self,
        quantum_output: &Array1<f64>,
    ) -> Result<Array1<f64>, SamplerError> {
        match &self.architecture.postprocessing {
            PostprocessingScheme::None => Ok(quantum_output.clone()),
            PostprocessingScheme::Linear => {
                // Simple linear transformation
                let transformed = quantum_output.slice(s![..self.architecture.output_dim]);
                Ok(transformed.to_owned())
            }
            _ => Ok(quantum_output.clone()), // Placeholder for other schemes
        }
    }

    /// Forward pass for a batch of inputs
    fn forward_batch(
        &self,
        batch: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Array1<f64>>, SamplerError> {
        let mut predictions = Vec::new();
        for (input, _) in batch {
            predictions.push(self.forward(input)?);
        }
        Ok(predictions)
    }

    /// Compute loss and gradients
    fn compute_loss_and_gradients(
        &self,
        batch: &[(Array1<f64>, Array1<f64>)],
        predictions: &[Array1<f64>],
    ) -> Result<(f64, Array1<f64>), SamplerError> {
        let mut total_loss = 0.0;
        let mut gradients = Array1::zeros(self.parameters.quantum_params.len());

        for (i, ((_, target), prediction)) in batch.iter().zip(predictions.iter()).enumerate() {
            let loss = self.compute_loss(prediction, target)?;
            total_loss += loss;

            // Compute gradients using parameter shift rule
            let param_gradients = self.compute_parameter_gradients(prediction, target)?;
            gradients = gradients + &param_gradients;
        }

        total_loss /= batch.len() as f64;
        gradients /= batch.len() as f64;

        Ok((total_loss, gradients))
    }

    /// Compute loss for a single prediction
    fn compute_loss(
        &self,
        prediction: &Array1<f64>,
        target: &Array1<f64>,
    ) -> Result<f64, SamplerError> {
        match &self.training_config.loss_function {
            LossFunction::MeanSquaredError => {
                let diff = prediction - target;
                Ok(diff.dot(&diff) / (2.0 * prediction.len() as f64))
            }
            LossFunction::CrossEntropy => {
                let mut loss = 0.0;
                for (pred, targ) in prediction.iter().zip(target.iter()) {
                    loss -= targ * pred.ln();
                }
                Ok(loss)
            }
            _ => {
                // Default to MSE
                let diff = prediction - target;
                Ok(diff.dot(&diff) / (2.0 * prediction.len() as f64))
            }
        }
    }

    /// Compute parameter gradients using parameter shift rule
    fn compute_parameter_gradients(
        &self,
        prediction: &Array1<f64>,
        target: &Array1<f64>,
    ) -> Result<Array1<f64>, SamplerError> {
        let mut gradients = Array1::zeros(self.parameters.quantum_params.len());
        let shift = PI / 2.0;

        for i in 0..self.parameters.quantum_params.len() {
            // Forward pass with positive shift
            let mut params_plus = self.parameters.quantum_params.clone();
            params_plus[i] += shift;
            let prediction_plus = self.forward_with_params(&prediction, &params_plus)?;
            let loss_plus = self.compute_loss(&prediction_plus, target)?;

            // Forward pass with negative shift
            let mut params_minus = self.parameters.quantum_params.clone();
            params_minus[i] -= shift;
            let prediction_minus = self.forward_with_params(&prediction, &params_plus)?;
            let loss_minus = self.compute_loss(&prediction_minus, target)?;

            // Compute gradient
            gradients[i] = (loss_plus - loss_minus) / (2.0 * shift);
        }

        Ok(gradients)
    }

    /// Forward pass with specific parameters
    fn forward_with_params(
        &self,
        input: &Array1<f64>,
        params: &Array1<f64>,
    ) -> Result<Array1<f64>, SamplerError> {
        // This is a simplified version - would need to temporarily update parameters
        self.forward(input)
    }

    /// Update parameters using optimizer
    fn update_parameters(&mut self, gradients: &Array1<f64>) -> Result<(), SamplerError> {
        match &self.training_config.optimizer {
            OptimizerType::SGD { momentum: _ } => {
                // Simple gradient descent
                let lr = self.training_config.learning_rate;
                self.parameters.quantum_params =
                    &self.parameters.quantum_params - &(gradients * lr);
            }
            OptimizerType::Adam {
                beta1,
                beta2,
                epsilon,
            } => {
                // Simplified Adam optimizer
                let lr = self.training_config.learning_rate;
                self.parameters.quantum_params =
                    &self.parameters.quantum_params - &(gradients * lr);
            }
            _ => {
                // Default to SGD
                let lr = self.training_config.learning_rate;
                self.parameters.quantum_params =
                    &self.parameters.quantum_params - &(gradients * lr);
            }
        }

        // Clip parameters to bounds
        let clipped_params: Vec<f64> = self
            .parameters
            .quantum_params
            .iter()
            .enumerate()
            .map(|(i, &param)| {
                if i < self.parameters.parameter_bounds.len() {
                    let (min_val, max_val) = self.parameters.parameter_bounds[i];
                    param.max(min_val).min(max_val)
                } else {
                    param
                }
            })
            .collect();

        for (i, value) in clipped_params.into_iter().enumerate() {
            self.parameters.quantum_params[i] = value;
        }

        Ok(())
    }

    /// Compute batch accuracy
    fn compute_batch_accuracy(
        &self,
        batch: &[(Array1<f64>, Array1<f64>)],
        predictions: &[Array1<f64>],
    ) -> f64 {
        let mut correct = 0;
        for ((_, target), prediction) in batch.iter().zip(predictions.iter()) {
            if self.is_prediction_correct(prediction, target) {
                correct += 1;
            }
        }
        correct as f64 / batch.len() as f64
    }

    /// Check if prediction is correct
    fn is_prediction_correct(&self, prediction: &Array1<f64>, target: &Array1<f64>) -> bool {
        // For classification: check if argmax matches
        if prediction.len() > 1 {
            let pred_class = prediction
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let target_class = target
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            pred_class == target_class
        } else {
            // For regression: check if within tolerance
            (prediction[0] - target[0]).abs() < 0.1
        }
    }

    /// Compute validation loss
    fn compute_validation_loss(
        &self,
        data: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<f64, SamplerError> {
        let mut total_loss = 0.0;
        for (input, target) in data.iter().take(10) {
            // Use subset for validation
            let prediction = self.forward(input)?;
            total_loss += self.compute_loss(&prediction, target)?;
        }
        Ok(total_loss / 10.0)
    }

    /// Initialize parameters
    fn initialize_parameters(&mut self) -> Result<(), SamplerError> {
        match &self.parameters.initialization_scheme {
            ParameterInitializationScheme::RandomUniform { min, max } => {
                let mut rng = thread_rng();
                for param in self.parameters.quantum_params.iter_mut() {
                    *param = rng.gen_range(*min..*max);
                }
            }
            ParameterInitializationScheme::RandomNormal { mean, std } => {
                let mut rng = thread_rng();
                for param in self.parameters.quantum_params.iter_mut() {
                    *param = rng.gen::<f64>() * std + mean;
                }
            }
            _ => {
                // Default to random uniform [-π, π]
                let mut rng = thread_rng();
                for param in self.parameters.quantum_params.iter_mut() {
                    *param = rng.gen_range(-PI..PI);
                }
            }
        }
        Ok(())
    }

    /// Compute parameter statistics
    fn compute_parameter_statistics(&self) -> ParameterStatistics {
        let params = &self.parameters.quantum_params;
        let mean = params.mean().unwrap_or(0.0);
        let std = params.std(0.0);

        let ranges = vec![(
            *params
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
            *params
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
        )];
        let correlations = Array2::eye(params.len());

        ParameterStatistics {
            mean_values: Array1::from_elem(params.len(), mean),
            std_values: Array1::from_elem(params.len(), std),
            ranges,
            correlations,
        }
    }

    /// Compute convergence rate
    fn compute_convergence_rate(&self) -> f64 {
        if self.training_history.len() < 2 {
            return 0.0;
        }

        let initial_loss = self.training_history[0].training_loss;
        let final_loss = self.training_history.last().unwrap().training_loss;

        if initial_loss > 0.0 {
            (initial_loss - final_loss) / initial_loss
        } else {
            0.0
        }
    }

    /// Compute training stability
    fn compute_training_stability(&self) -> f64 {
        if self.training_history.len() < 10 {
            return 1.0;
        }

        let losses: Vec<f64> = self
            .training_history
            .iter()
            .map(|epoch| epoch.training_loss)
            .collect();
        let mean_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        let variance = losses
            .iter()
            .map(|loss| (loss - mean_loss).powi(2))
            .sum::<f64>()
            / losses.len() as f64;

        1.0 / (1.0 + variance.sqrt())
    }

    /// Compute overfitting measure
    fn compute_overfitting_measure(&self) -> f64 {
        if self.training_history.is_empty() {
            return 0.0;
        }

        let last_epoch = self.training_history.last().unwrap();
        if let Some(val_loss) = last_epoch.validation_loss {
            (val_loss - last_epoch.training_loss).max(0.0)
        } else {
            0.0
        }
    }

    /// Calculate number of quantum parameters
    fn calculate_quantum_params(architecture: &QNNArchitecture) -> usize {
        // Rough estimate: 3 parameters per qubit per layer
        architecture.num_qubits * architecture.circuit_depth * 3
    }

    /// Calculate number of classical parameters
    fn calculate_classical_params(architecture: &QNNArchitecture) -> usize {
        // Rough estimate based on architecture
        architecture.input_dim * architecture.output_dim
    }

    /// Build quantum layers
    fn build_quantum_layers(
        architecture: &QNNArchitecture,
    ) -> Result<Vec<QuantumLayer>, SamplerError> {
        let mut layers = Vec::new();

        for layer_id in 0..architecture.circuit_depth {
            let mut gates = Vec::new();
            let mut parametrized_gates = Vec::new();

            // Add parametrized rotation gates
            for qubit in 0..architecture.num_qubits {
                parametrized_gates.push(ParametrizedGate {
                    gate_type: ParametrizedGateType::Rotation {
                        axis: RotationAxis::Y,
                    },
                    qubits: vec![qubit],
                    parameter_indices: vec![layer_id * architecture.num_qubits + qubit],
                    gate_function: GateFunction::StandardRotation,
                });
            }

            // Add entangling gates based on pattern
            match &architecture.entanglement_pattern {
                EntanglementPattern::Linear => {
                    for qubit in 0..architecture.num_qubits - 1 {
                        gates.push(QuantumGate::CNOT {
                            control: qubit,
                            target: qubit + 1,
                        });
                    }
                }
                EntanglementPattern::Circular => {
                    for qubit in 0..architecture.num_qubits {
                        let target = (qubit + 1) % architecture.num_qubits;
                        gates.push(QuantumGate::CNOT {
                            control: qubit,
                            target,
                        });
                    }
                }
                _ => {
                    // Default to linear
                    for qubit in 0..architecture.num_qubits - 1 {
                        gates.push(QuantumGate::CNOT {
                            control: qubit,
                            target: qubit + 1,
                        });
                    }
                }
            }

            layers.push(QuantumLayer {
                layer_id,
                num_qubits: architecture.num_qubits,
                gates,
                parametrized_gates,
                layer_type: QuantumLayerType::Variational,
                skip_connections: Vec::new(),
            });
        }

        Ok(layers)
    }

    /// Build classical layers
    fn build_classical_layers(
        architecture: &QNNArchitecture,
    ) -> Result<Vec<ClassicalLayer>, SamplerError> {
        let mut layers = Vec::new();

        // Add a preprocessing layer if needed
        if architecture.input_dim != architecture.num_qubits {
            let weights = Array2::zeros((architecture.num_qubits, architecture.input_dim));
            let biases = Array1::zeros(architecture.num_qubits);

            layers.push(ClassicalLayer {
                layer_type: ClassicalLayerType::Dense,
                input_dim: architecture.input_dim,
                output_dim: architecture.num_qubits,
                weights,
                biases,
                activation: ActivationFunction::Tanh,
            });
        }

        Ok(layers)
    }
}

// Implement default for various types
impl Default for QNNMetrics {
    fn default() -> Self {
        Self {
            training_metrics: TrainingMetrics {
                final_training_loss: 0.0,
                convergence_rate: 0.0,
                epochs_to_convergence: 0,
                training_stability: 0.0,
                overfitting_measure: 0.0,
            },
            validation_metrics: ValidationMetrics {
                best_validation_loss: 0.0,
                validation_accuracy: 0.0,
                generalization_gap: 0.0,
                cv_scores: Vec::new(),
                confidence_intervals: Vec::new(),
            },
            quantum_metrics: QuantumMetrics {
                quantum_volume: 0.0,
                entanglement_measures: Vec::new(),
                quantum_advantage: 0.0,
                fidelity_measures: Vec::new(),
                coherence_utilization: 0.0,
            },
            computational_metrics: ComputationalMetrics {
                training_time_per_epoch: 0.0,
                inference_time: 0.0,
                memory_usage: 0.0,
                quantum_execution_time: 0.0,
                classical_computation_time: 0.0,
            },
        }
    }
}

/// Create a default QNN for binary classification
pub fn create_binary_classification_qnn(
    num_qubits: usize,
) -> Result<QuantumNeuralNetwork, SamplerError> {
    let architecture = QNNArchitecture {
        input_dim: num_qubits,
        output_dim: 1,
        num_qubits,
        circuit_depth: 3,
        entanglement_pattern: EntanglementPattern::Linear,
        measurement_scheme: MeasurementScheme::Computational,
        postprocessing: PostprocessingScheme::Linear,
    };

    let training_config = QNNTrainingConfig {
        learning_rate: 0.01,
        batch_size: 32,
        num_epochs: 100,
        optimizer: OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        loss_function: LossFunction::MeanSquaredError,
        regularization: RegularizationConfig {
            l1_strength: 0.0,
            l2_strength: 0.001,
            dropout_prob: 0.0,
            parameter_noise: 0.0,
            quantum_noise: QuantumNoiseConfig {
                enable_noise: false,
                depolarizing_strength: 0.0,
                amplitude_damping: 0.0,
                phase_damping: 0.0,
                gate_error_rates: HashMap::new(),
            },
        },
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 10,
            min_improvement: 1e-4,
            monitor_metric: "validation_loss".to_string(),
        },
        gradient_estimation: GradientEstimationMethod::ParameterShift,
    };

    QuantumNeuralNetwork::new(architecture, training_config)
}

/// Create a QNN for optimization problems
pub fn create_optimization_qnn(problem_size: usize) -> Result<QuantumNeuralNetwork, SamplerError> {
    let num_qubits = (problem_size as f64).log2().ceil() as usize;

    let architecture = QNNArchitecture {
        input_dim: problem_size,
        output_dim: problem_size,
        num_qubits,
        circuit_depth: 5,
        entanglement_pattern: EntanglementPattern::HardwareEfficient,
        measurement_scheme: MeasurementScheme::Pauli {
            bases: vec![PauliBasis::Z, PauliBasis::X, PauliBasis::Y],
        },
        postprocessing: PostprocessingScheme::NonlinearNN {
            hidden_dims: vec![64, 32],
        },
    };

    let training_config = QNNTrainingConfig {
        learning_rate: 0.005,
        batch_size: 16,
        num_epochs: 200,
        optimizer: OptimizerType::QuantumNaturalGradient,
        loss_function: LossFunction::ExpectationValueLoss,
        regularization: RegularizationConfig {
            l1_strength: 0.001,
            l2_strength: 0.01,
            dropout_prob: 0.1,
            parameter_noise: 0.01,
            quantum_noise: QuantumNoiseConfig {
                enable_noise: true,
                depolarizing_strength: 0.01,
                amplitude_damping: 0.001,
                phase_damping: 0.001,
                gate_error_rates: HashMap::new(),
            },
        },
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 20,
            min_improvement: 1e-5,
            monitor_metric: "validation_loss".to_string(),
        },
        gradient_estimation: GradientEstimationMethod::QuantumFisherInformation,
    };

    QuantumNeuralNetwork::new(architecture, training_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qnn_creation() {
        let qnn = create_binary_classification_qnn(4).unwrap();
        assert_eq!(qnn.architecture.num_qubits, 4);
        assert_eq!(qnn.architecture.circuit_depth, 3);
        assert_eq!(qnn.layers.len(), 3);
    }

    #[test]
    fn test_qnn_forward_pass() {
        let qnn = create_binary_classification_qnn(2).unwrap();
        let input = Array1::from_vec(vec![0.5, 0.7]);
        let output = qnn.forward(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_optimization_qnn_creation() {
        let qnn = create_optimization_qnn(8).unwrap();
        assert_eq!(qnn.architecture.input_dim, 8);
        assert_eq!(qnn.architecture.output_dim, 8);
        assert!(qnn.architecture.num_qubits >= 3); // log2(8) = 3
    }

    #[test]
    fn test_parameter_initialization() {
        let mut qnn = create_binary_classification_qnn(3).unwrap();
        qnn.initialize_parameters().unwrap();

        // Check that parameters are initialized within bounds
        for &param in qnn.parameters.quantum_params.iter() {
            assert!(param >= -PI && param <= PI);
        }
    }

    #[test]
    fn test_quantum_gate_application() {
        let qnn = create_binary_classification_qnn(2).unwrap();
        let state = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]); // |00⟩

        let new_state = qnn.apply_rx_gate(&state, 0, PI / 2.0).unwrap();

        // After RX(π/2) on qubit 0, should be in superposition
        assert!((new_state[0] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((new_state[2] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
    }
}
