//! Quantum Machine Learning (QML) integration for seamless ML workflows.
//!
//! This module provides comprehensive integration between quantum simulation
//! backends and machine learning frameworks, enabling hybrid classical-quantum
//! algorithms, variational quantum eigensolvers (VQE), quantum neural networks
//! (QNN), and other QML applications with automatic differentiation and
//! hardware-aware optimization.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::autodiff_vqe::{AutoDiffContext, ParametricCircuit, VQEWithAutodiff};
use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use crate::statevector::StateVectorSimulator;

/// QML framework types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QMLFramework {
    /// PyTorch integration
    PyTorch,
    /// TensorFlow/Keras integration
    TensorFlow,
    /// JAX integration
    JAX,
    /// SciRS2 native ML
    SciRS2,
    /// Custom framework
    Custom,
}

/// QML integration configuration
#[derive(Debug, Clone)]
pub struct QMLIntegrationConfig {
    /// Target ML framework
    pub framework: QMLFramework,
    /// Enable automatic differentiation
    pub enable_autodiff: bool,
    /// Enable gradient optimization
    pub enable_gradient_optimization: bool,
    /// Batch size for circuit evaluation
    pub batch_size: usize,
    /// Enable parameter sharing across circuits
    pub enable_parameter_sharing: bool,
    /// Enable hardware-aware optimization
    pub hardware_aware_optimization: bool,
    /// Memory limit for gradient computation
    pub gradient_memory_limit: usize,
    /// Enable distributed training
    pub enable_distributed_training: bool,
    /// Enable mixed precision training
    pub enable_mixed_precision: bool,
}

impl Default for QMLIntegrationConfig {
    fn default() -> Self {
        Self {
            framework: QMLFramework::SciRS2,
            enable_autodiff: true,
            enable_gradient_optimization: true,
            batch_size: 32,
            enable_parameter_sharing: true,
            hardware_aware_optimization: true,
            gradient_memory_limit: 8_000_000_000, // 8GB
            enable_distributed_training: false,
            enable_mixed_precision: false,
        }
    }
}

/// Quantum machine learning layer types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QMLLayerType {
    /// Variational quantum circuit layer
    VariationalCircuit,
    /// Quantum convolutional layer
    QuantumConvolutional,
    /// Quantum recurrent layer
    QuantumRecurrent,
    /// Quantum attention layer
    QuantumAttention,
    /// Data encoding layer
    DataEncoding,
    /// Measurement layer
    Measurement,
    /// Classical processing layer
    Classical,
}

/// Quantum ML layer definition
#[derive(Debug, Clone)]
pub struct QMLLayer {
    /// Layer type
    pub layer_type: QMLLayerType,
    /// Layer name
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Trainable parameters
    pub parameters: Vec<f64>,
    /// Parameter names for tracking
    pub parameter_names: Vec<String>,
    /// Circuit template
    pub circuit_template: Option<InterfaceCircuit>,
    /// Classical processing function
    pub classical_function: Option<String>,
    /// Layer configuration
    pub config: LayerConfig,
}

/// Layer configuration
#[derive(Debug, Clone, Default)]
pub struct LayerConfig {
    /// Number of repetitions (for ansatz layers)
    pub repetitions: usize,
    /// Entangling pattern
    pub entangling_pattern: Vec<(usize, usize)>,
    /// Activation function
    pub activation: Option<String>,
    /// Regularization parameters
    pub regularization: Option<RegularizationConfig>,
    /// Hardware mapping
    pub hardware_mapping: Option<Vec<usize>>,
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
}

/// Quantum neural network model
#[derive(Debug, Clone)]
pub struct QuantumNeuralNetwork {
    /// Network layers
    pub layers: Vec<QMLLayer>,
    /// Global parameters
    pub global_parameters: HashMap<String, f64>,
    /// Network metadata
    pub metadata: QNNMetadata,
    /// Training configuration
    pub training_config: TrainingConfig,
}

/// QNN metadata
#[derive(Debug, Clone, Default)]
pub struct QNNMetadata {
    /// Model name
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: Option<std::time::SystemTime>,
    /// Total number of parameters
    pub total_parameters: usize,
    /// Number of trainable parameters
    pub trainable_parameters: usize,
    /// Model complexity score
    pub complexity_score: f64,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Loss function
    pub loss_function: LossFunction,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Validation split
    pub validation_split: f64,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
    /// Learning rate scheduler
    pub lr_scheduler: Option<LRScheduler>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            optimizer: OptimizerType::Adam,
            loss_function: LossFunction::MeanSquaredError,
            epochs: 100,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping_patience: Some(10),
            lr_scheduler: None,
        }
    }
}

/// Optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    LBFGS,
    NaturalGradient,
    QuantumNaturalGradient,
}

/// Loss functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossFunction {
    MeanSquaredError,
    MeanAbsoluteError,
    CrossEntropy,
    BinaryCrossEntropy,
    Hinge,
    CustomQuantum,
}

/// Learning rate schedulers
#[derive(Debug, Clone)]
pub enum LRScheduler {
    StepLR { step_size: usize, gamma: f64 },
    ExponentialLR { gamma: f64 },
    CosineAnnealingLR { t_max: usize },
    ReduceLROnPlateau { patience: usize, factor: f64 },
}

/// QML integration engine
pub struct QMLIntegration {
    /// Configuration
    config: QMLIntegrationConfig,
    /// Circuit interface
    circuit_interface: CircuitInterface,
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Autodiff context
    autodiff_context: Option<AutoDiffContext>,
    /// Parameter cache
    parameter_cache: Arc<Mutex<HashMap<String, Vec<f64>>>>,
    /// Gradient cache
    gradient_cache: Arc<Mutex<HashMap<String, Vec<f64>>>>,
    /// Training statistics
    stats: QMLTrainingStats,
}

/// QML training statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QMLTrainingStats {
    /// Total training time
    pub total_training_time_ms: f64,
    /// Number of parameter updates
    pub parameter_updates: usize,
    /// Number of gradient computations
    pub gradient_computations: usize,
    /// Average gradient computation time
    pub avg_gradient_time_ms: f64,
    /// Number of circuit evaluations
    pub circuit_evaluations: usize,
    /// Average circuit evaluation time
    pub avg_circuit_time_ms: f64,
    /// Training loss history
    pub loss_history: Vec<f64>,
    /// Validation loss history
    pub validation_loss_history: Vec<f64>,
    /// Parameter norm history
    pub parameter_norm_history: Vec<f64>,
    /// Gradient norm history
    pub gradient_norm_history: Vec<f64>,
}

impl QMLIntegration {
    /// Create new QML integration
    pub fn new(config: QMLIntegrationConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;

        Ok(Self {
            config,
            circuit_interface,
            backend: None,
            autodiff_context: None,
            parameter_cache: Arc::new(Mutex::new(HashMap::new())),
            gradient_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: QMLTrainingStats::default(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        self.circuit_interface = self.circuit_interface.with_backend()?;

        if self.config.enable_autodiff {
            self.autodiff_context = Some(AutoDiffContext::new(
                Vec::new(),
                crate::autodiff_vqe::GradientMethod::ParameterShift,
            ));
        }

        Ok(self)
    }

    /// Train quantum neural network
    pub fn train_qnn(
        &mut self,
        mut qnn: QuantumNeuralNetwork,
        training_data: &[TrainingExample],
        validation_data: Option<&[TrainingExample]>,
    ) -> Result<TrainingResult> {
        let start_time = std::time::Instant::now();

        // Initialize optimizer
        let mut optimizer = self.create_optimizer(&qnn.training_config)?;

        // Initialize learning rate scheduler
        let mut lr_scheduler = qnn.training_config.lr_scheduler.clone();

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..qnn.training_config.epochs {
            let epoch_start = std::time::Instant::now();

            // Training phase
            let train_loss = self.train_epoch(&mut qnn, training_data, &mut optimizer)?;
            self.stats.loss_history.push(train_loss);

            // Validation phase
            let val_loss = if let Some(val_data) = validation_data {
                self.validate_epoch(&qnn, val_data)?
            } else {
                train_loss
            };
            self.stats.validation_loss_history.push(val_loss);

            // Update learning rate scheduler
            if let Some(ref mut scheduler) = lr_scheduler {
                self.update_lr_scheduler(scheduler, val_loss, &mut optimizer)?;
            }

            // Early stopping check
            if let Some(patience) = qnn.training_config.early_stopping_patience {
                if val_loss < best_loss {
                    best_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        println!("Early stopping at epoch {} due to no improvement", epoch);
                        break;
                    }
                }
            }

            // Compute parameter and gradient norms
            let param_norm = self.compute_parameter_norm(&qnn)?;
            let grad_norm = self.compute_last_gradient_norm()?;
            self.stats.parameter_norm_history.push(param_norm);
            self.stats.gradient_norm_history.push(grad_norm);

            println!(
                "Epoch {}: train_loss={:.6}, val_loss={:.6}, time={:.2}ms",
                epoch,
                train_loss,
                val_loss,
                epoch_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_training_time_ms += total_time;

        Ok(TrainingResult {
            trained_qnn: qnn.clone(),
            final_loss: *self.stats.loss_history.last().unwrap_or(&0.0),
            final_validation_loss: *self.stats.validation_loss_history.last().unwrap_or(&0.0),
            epochs_completed: self.stats.loss_history.len(),
            total_time_ms: total_time,
            converged: patience_counter
                < qnn
                    .training_config
                    .early_stopping_patience
                    .unwrap_or(usize::MAX),
        })
    }

    /// Train single epoch
    fn train_epoch(
        &mut self,
        qnn: &mut QuantumNeuralNetwork,
        training_data: &[TrainingExample],
        optimizer: &mut Box<dyn QMLOptimizer>,
    ) -> Result<f64> {
        let mut total_loss = 0.0;
        let batch_size = qnn.training_config.batch_size;
        let num_batches = (training_data.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(training_data.len());
            let batch = &training_data[start_idx..end_idx];

            // Forward pass
            let (predictions, loss) = self.forward_pass(qnn, batch)?;
            total_loss += loss;

            // Backward pass (compute gradients)
            let gradients = self.backward_pass(qnn, batch, &predictions)?;

            // Update parameters
            optimizer.update_parameters(qnn, &gradients)?;

            self.stats.parameter_updates += 1;
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Validate single epoch
    fn validate_epoch(
        &mut self,
        qnn: &QuantumNeuralNetwork,
        validation_data: &[TrainingExample],
    ) -> Result<f64> {
        let mut total_loss = 0.0;
        let batch_size = qnn.training_config.batch_size;
        let num_batches = (validation_data.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(validation_data.len());
            let batch = &validation_data[start_idx..end_idx];

            let (_, loss) = self.forward_pass(qnn, batch)?;
            total_loss += loss;
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Forward pass through the quantum neural network
    fn forward_pass(
        &mut self,
        qnn: &QuantumNeuralNetwork,
        batch: &[TrainingExample],
    ) -> Result<(Vec<Array1<f64>>, f64)> {
        let start_time = std::time::Instant::now();

        let mut predictions = Vec::new();
        let mut total_loss = 0.0;

        for example in batch {
            // Evaluate quantum circuit with current parameters
            let prediction = self.evaluate_qnn(qnn, &example.input)?;

            // Compute loss
            let loss = self.compute_loss(
                &prediction,
                &example.target,
                &qnn.training_config.loss_function,
            )?;

            predictions.push(prediction);
            total_loss += loss;
        }

        let eval_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.avg_circuit_time_ms =
            (self.stats.avg_circuit_time_ms * self.stats.circuit_evaluations as f64 + eval_time)
                / (self.stats.circuit_evaluations + batch.len()) as f64;
        self.stats.circuit_evaluations += batch.len();

        Ok((predictions, total_loss / batch.len() as f64))
    }

    /// Backward pass to compute gradients
    fn backward_pass(
        &mut self,
        qnn: &QuantumNeuralNetwork,
        batch: &[TrainingExample],
        predictions: &[Array1<f64>],
    ) -> Result<HashMap<String, Vec<f64>>> {
        let start_time = std::time::Instant::now();

        let mut gradients = HashMap::new();

        if self.config.enable_autodiff {
            // Use automatic differentiation
            gradients = self.compute_gradients_autodiff(qnn, batch, predictions)?;
        } else {
            // Use parameter shift rule or finite differences
            gradients = self.compute_gradients_parameter_shift(qnn, batch)?;
        }

        let grad_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.avg_gradient_time_ms =
            (self.stats.avg_gradient_time_ms * self.stats.gradient_computations as f64 + grad_time)
                / (self.stats.gradient_computations + 1) as f64;
        self.stats.gradient_computations += 1;

        // Cache gradients
        {
            let mut cache = self.gradient_cache.lock().unwrap();
            for (param_name, grad) in &gradients {
                cache.insert(param_name.clone(), grad.clone());
            }
        }

        Ok(gradients)
    }

    /// Evaluate quantum neural network
    fn evaluate_qnn(
        &mut self,
        qnn: &QuantumNeuralNetwork,
        input: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Start with initial state
        let total_qubits = qnn.layers.iter().map(|l| l.num_qubits).max().unwrap_or(1);
        let mut state = Array1::zeros(1 << total_qubits);
        state[0] = Complex64::new(1.0, 0.0); // |0...0⟩

        let mut current_output = input.clone();

        // Process each layer
        for layer in &qnn.layers {
            current_output = self.evaluate_layer(layer, &current_output, &mut state)?;
        }

        Ok(current_output)
    }

    /// Evaluate single layer
    fn evaluate_layer(
        &mut self,
        layer: &QMLLayer,
        input: &Array1<f64>,
        state: &mut Array1<Complex64>,
    ) -> Result<Array1<f64>> {
        match layer.layer_type {
            QMLLayerType::DataEncoding => {
                self.apply_data_encoding(layer, input, state)?;
                Ok(input.clone()) // Pass through for now
            }
            QMLLayerType::VariationalCircuit => {
                self.apply_variational_circuit(layer, state)?;
                self.measure_qubits(layer, state)
            }
            QMLLayerType::Measurement => self.measure_qubits(layer, state),
            QMLLayerType::Classical => self.apply_classical_processing(layer, input),
            _ => {
                // Placeholder for other layer types
                Ok(input.clone())
            }
        }
    }

    /// Apply data encoding layer
    fn apply_data_encoding(
        &mut self,
        layer: &QMLLayer,
        input: &Array1<f64>,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        // Amplitude encoding: encode classical data into quantum amplitudes
        for (i, &value) in input.iter().enumerate() {
            if i < layer.num_qubits {
                // Apply rotation proportional to input value
                let angle = value * std::f64::consts::PI;
                self.apply_ry_rotation(i, angle, state)?;
            }
        }
        Ok(())
    }

    /// Apply variational circuit layer
    fn apply_variational_circuit(
        &mut self,
        layer: &QMLLayer,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        if let Some(circuit_template) = &layer.circuit_template {
            // Create parameterized circuit
            let mut circuit = circuit_template.clone();
            self.parameterize_circuit(&mut circuit, &layer.parameters)?;

            // Compile and execute circuit
            let compiled = self.circuit_interface.compile_circuit(
                &circuit,
                crate::circuit_interfaces::SimulationBackend::StateVector,
            )?;
            let result = self
                .circuit_interface
                .execute_circuit(&compiled, Some(state.clone()))?;

            if let Some(final_state) = result.final_state {
                *state = final_state;
            }
        }
        Ok(())
    }

    /// Measure qubits
    fn measure_qubits(&self, layer: &QMLLayer, state: &Array1<Complex64>) -> Result<Array1<f64>> {
        let mut measurements = Array1::zeros(layer.num_qubits);

        for qubit in 0..layer.num_qubits {
            let prob = self.compute_measurement_probability(qubit, state)?;
            measurements[qubit] = prob;
        }

        Ok(measurements)
    }

    /// Apply classical processing
    fn apply_classical_processing(
        &self,
        layer: &QMLLayer,
        input: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Simple linear transformation for now
        Ok(input.clone())
    }

    /// Apply RY rotation gate
    fn apply_ry_rotation(
        &self,
        qubit: usize,
        angle: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let qubit_mask = 1 << qubit;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..state.len() {
            if i & qubit_mask == 0 {
                let j = i | qubit_mask;
                if j < state.len() {
                    let amp_0 = state[i];
                    let amp_1 = state[j];

                    state[i] = cos_half * amp_0 - sin_half * amp_1;
                    state[j] = sin_half * amp_0 + cos_half * amp_1;
                }
            }
        }

        Ok(())
    }

    /// Parameterize circuit with current parameter values
    fn parameterize_circuit(
        &self,
        circuit: &mut InterfaceCircuit,
        parameters: &[f64],
    ) -> Result<()> {
        let mut param_idx = 0;

        for gate in &mut circuit.gates {
            match &mut gate.gate_type {
                InterfaceGateType::RX(ref mut angle)
                | InterfaceGateType::RY(ref mut angle)
                | InterfaceGateType::RZ(ref mut angle) => {
                    if param_idx < parameters.len() {
                        *angle = parameters[param_idx];
                        param_idx += 1;
                    }
                }
                InterfaceGateType::Phase(ref mut angle) => {
                    if param_idx < parameters.len() {
                        *angle = parameters[param_idx];
                        param_idx += 1;
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Compute measurement probability for a qubit
    fn compute_measurement_probability(
        &self,
        qubit: usize,
        state: &Array1<Complex64>,
    ) -> Result<f64> {
        let qubit_mask = 1 << qubit;
        let mut prob_one = 0.0;

        for (i, &amplitude) in state.iter().enumerate() {
            if i & qubit_mask != 0 {
                prob_one += amplitude.norm_sqr();
            }
        }

        Ok(prob_one)
    }

    /// Compute loss
    fn compute_loss(
        &self,
        prediction: &Array1<f64>,
        target: &Array1<f64>,
        loss_fn: &LossFunction,
    ) -> Result<f64> {
        match loss_fn {
            LossFunction::MeanSquaredError => {
                let diff = prediction - target;
                Ok(diff.mapv(|x| x * x).mean().unwrap_or(0.0))
            }
            LossFunction::MeanAbsoluteError => {
                let diff = prediction - target;
                Ok(diff.mapv(|x| x.abs()).mean().unwrap_or(0.0))
            }
            LossFunction::CrossEntropy => {
                // Simplified cross-entropy
                let mut loss = 0.0;
                for (i, (&pred, &targ)) in prediction.iter().zip(target.iter()).enumerate() {
                    if targ > 0.0 {
                        loss -= targ * pred.ln();
                    }
                }
                Ok(loss)
            }
            _ => Ok(0.0), // Placeholder for other loss functions
        }
    }

    /// Compute gradients using automatic differentiation
    fn compute_gradients_autodiff(
        &mut self,
        qnn: &QuantumNeuralNetwork,
        batch: &[TrainingExample],
        predictions: &[Array1<f64>],
    ) -> Result<HashMap<String, Vec<f64>>> {
        // Placeholder for autodiff implementation
        self.compute_gradients_parameter_shift(qnn, batch)
    }

    /// Compute gradients using parameter shift rule
    fn compute_gradients_parameter_shift(
        &mut self,
        qnn: &QuantumNeuralNetwork,
        batch: &[TrainingExample],
    ) -> Result<HashMap<String, Vec<f64>>> {
        let mut gradients = HashMap::new();
        let shift = std::f64::consts::PI / 2.0;

        // Collect all parameters
        let mut all_params = Vec::new();
        let mut param_names = Vec::new();

        for layer in &qnn.layers {
            for (i, &param) in layer.parameters.iter().enumerate() {
                all_params.push(param);
                param_names.push(format!("{}_{}", layer.name, i));
            }
        }

        for (param_idx, param_name) in param_names.iter().enumerate() {
            let mut param_grad = 0.0;

            for example in batch {
                // Evaluate with positive shift
                let mut qnn_plus = qnn.clone();
                self.shift_parameter(&mut qnn_plus, param_idx, shift)?;
                let pred_plus = self.evaluate_qnn(&qnn_plus, &example.input)?;
                let loss_plus = self.compute_loss(
                    &pred_plus,
                    &example.target,
                    &qnn.training_config.loss_function,
                )?;

                // Evaluate with negative shift
                let mut qnn_minus = qnn.clone();
                self.shift_parameter(&mut qnn_minus, param_idx, -shift)?;
                let pred_minus = self.evaluate_qnn(&qnn_minus, &example.input)?;
                let loss_minus = self.compute_loss(
                    &pred_minus,
                    &example.target,
                    &qnn.training_config.loss_function,
                )?;

                // Compute gradient using parameter shift rule
                param_grad += (loss_plus - loss_minus) / 2.0;
            }

            param_grad /= batch.len() as f64;
            gradients.insert(param_name.clone(), vec![param_grad]);
        }

        Ok(gradients)
    }

    /// Shift a parameter in the QNN
    fn shift_parameter(
        &self,
        qnn: &mut QuantumNeuralNetwork,
        param_idx: usize,
        shift: f64,
    ) -> Result<()> {
        let mut current_idx = 0;

        for layer in &mut qnn.layers {
            if current_idx + layer.parameters.len() > param_idx {
                let local_idx = param_idx - current_idx;
                layer.parameters[local_idx] += shift;
                return Ok(());
            }
            current_idx += layer.parameters.len();
        }

        Err(SimulatorError::InvalidInput(format!(
            "Parameter index {} out of bounds",
            param_idx
        )))
    }

    /// Create optimizer
    fn create_optimizer(&self, config: &TrainingConfig) -> Result<Box<dyn QMLOptimizer>> {
        match config.optimizer {
            OptimizerType::Adam => Ok(Box::new(AdamOptimizer::new(config.learning_rate))),
            OptimizerType::SGD => Ok(Box::new(SGDOptimizer::new(config.learning_rate))),
            _ => Ok(Box::new(AdamOptimizer::new(config.learning_rate))), // Default to Adam
        }
    }

    /// Update learning rate scheduler
    fn update_lr_scheduler(
        &self,
        scheduler: &mut LRScheduler,
        current_loss: f64,
        optimizer: &mut Box<dyn QMLOptimizer>,
    ) -> Result<()> {
        match scheduler {
            LRScheduler::StepLR {
                step_size: _,
                gamma,
            } => {
                optimizer.update_learning_rate(*gamma);
            }
            LRScheduler::ExponentialLR { gamma } => {
                optimizer.update_learning_rate(*gamma);
            }
            LRScheduler::ReduceLROnPlateau {
                patience: _,
                factor,
            } => {
                // Simple implementation - reduce LR if loss plateaus
                optimizer.update_learning_rate(*factor);
            }
            _ => {}
        }
        Ok(())
    }

    /// Compute parameter norm
    fn compute_parameter_norm(&self, qnn: &QuantumNeuralNetwork) -> Result<f64> {
        let mut norm_squared = 0.0;

        for layer in &qnn.layers {
            for &param in &layer.parameters {
                norm_squared += param * param;
            }
        }

        Ok(norm_squared.sqrt())
    }

    /// Compute last gradient norm
    fn compute_last_gradient_norm(&self) -> Result<f64> {
        let cache = self.gradient_cache.lock().unwrap();
        let mut norm_squared = 0.0;

        for (_, grads) in cache.iter() {
            for &grad in grads {
                norm_squared += grad * grad;
            }
        }

        Ok(norm_squared.sqrt())
    }

    /// Get training statistics
    pub fn get_stats(&self) -> &QMLTrainingStats {
        &self.stats
    }

    /// Reset training statistics
    pub fn reset_stats(&mut self) {
        self.stats = QMLTrainingStats::default();
    }
}

/// Training example
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input data
    pub input: Array1<f64>,
    /// Target output
    pub target: Array1<f64>,
}

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Trained QNN
    pub trained_qnn: QuantumNeuralNetwork,
    /// Final training loss
    pub final_loss: f64,
    /// Final validation loss
    pub final_validation_loss: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Total training time
    pub total_time_ms: f64,
    /// Whether training converged
    pub converged: bool,
}

/// QML optimizer trait
pub trait QMLOptimizer {
    /// Update parameters using computed gradients
    fn update_parameters(
        &mut self,
        qnn: &mut QuantumNeuralNetwork,
        gradients: &HashMap<String, Vec<f64>>,
    ) -> Result<()>;

    /// Update learning rate
    fn update_learning_rate(&mut self, factor: f64);

    /// Get current learning rate
    fn get_learning_rate(&self) -> f64;
}

/// Adam optimizer implementation
pub struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    step: usize,
    m: HashMap<String, Vec<f64>>, // First moment estimates
    v: HashMap<String, Vec<f64>>, // Second moment estimates
}

impl AdamOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl QMLOptimizer for AdamOptimizer {
    fn update_parameters(
        &mut self,
        qnn: &mut QuantumNeuralNetwork,
        gradients: &HashMap<String, Vec<f64>>,
    ) -> Result<()> {
        self.step += 1;

        for (param_name, grads) in gradients {
            // Initialize moments if needed
            if !self.m.contains_key(param_name) {
                self.m.insert(param_name.clone(), vec![0.0; grads.len()]);
                self.v.insert(param_name.clone(), vec![0.0; grads.len()]);
            }

            let mut updates = Vec::new();

            {
                let m = self.m.get_mut(param_name).unwrap();
                let v = self.v.get_mut(param_name).unwrap();

                for (i, &grad) in grads.iter().enumerate() {
                    // Update biased first moment estimate
                    m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad;

                    // Update biased second moment estimate
                    v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad * grad;

                    // Compute bias-corrected first moment estimate
                    let m_hat = m[i] / (1.0 - self.beta1.powi(self.step as i32));

                    // Compute bias-corrected second moment estimate
                    let v_hat = v[i] / (1.0 - self.beta2.powi(self.step as i32));

                    // Update parameter
                    let update = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                    updates.push((i, -update));
                }
            }

            // Apply updates
            for (i, update) in updates {
                self.update_qnn_parameter(qnn, param_name, i, update)?;
            }
        }

        Ok(())
    }

    fn update_learning_rate(&mut self, factor: f64) {
        self.learning_rate *= factor;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

impl AdamOptimizer {
    fn update_qnn_parameter(
        &self,
        qnn: &mut QuantumNeuralNetwork,
        param_name: &str,
        param_idx: usize,
        update: f64,
    ) -> Result<()> {
        // Parse parameter name to find the layer and parameter index
        let parts: Vec<&str> = param_name.split('_').collect();
        if parts.len() >= 2 {
            let layer_name = parts[0];

            for layer in &mut qnn.layers {
                if layer.name == layer_name && param_idx < layer.parameters.len() {
                    layer.parameters[param_idx] += update;
                    return Ok(());
                }
            }
        }

        Err(SimulatorError::InvalidInput(format!(
            "Parameter {} not found",
            param_name
        )))
    }
}

/// SGD optimizer implementation
pub struct SGDOptimizer {
    learning_rate: f64,
    momentum: f64,
    velocity: HashMap<String, Vec<f64>>,
}

impl SGDOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.9,
            velocity: HashMap::new(),
        }
    }
}

impl QMLOptimizer for SGDOptimizer {
    fn update_parameters(
        &mut self,
        qnn: &mut QuantumNeuralNetwork,
        gradients: &HashMap<String, Vec<f64>>,
    ) -> Result<()> {
        for (param_name, grads) in gradients {
            // Initialize velocity if needed
            if !self.velocity.contains_key(param_name) {
                self.velocity
                    .insert(param_name.clone(), vec![0.0; grads.len()]);
            }

            let mut updates = Vec::new();

            {
                let velocity = self.velocity.get_mut(param_name).unwrap();

                for (i, &grad) in grads.iter().enumerate() {
                    // Update velocity with momentum
                    velocity[i] = self.momentum * velocity[i] - self.learning_rate * grad;
                    updates.push((i, velocity[i]));
                }
            }

            // Apply updates
            for (i, update) in updates {
                self.update_qnn_parameter(qnn, param_name, i, update)?;
            }
        }

        Ok(())
    }

    fn update_learning_rate(&mut self, factor: f64) {
        self.learning_rate *= factor;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

impl SGDOptimizer {
    fn update_qnn_parameter(
        &self,
        qnn: &mut QuantumNeuralNetwork,
        param_name: &str,
        param_idx: usize,
        update: f64,
    ) -> Result<()> {
        // Parse parameter name to find the layer and parameter index
        let parts: Vec<&str> = param_name.split('_').collect();
        if parts.len() >= 2 {
            let layer_name = parts[0];

            for layer in &mut qnn.layers {
                if layer.name == layer_name && param_idx < layer.parameters.len() {
                    layer.parameters[param_idx] += update;
                    return Ok(());
                }
            }
        }

        Err(SimulatorError::InvalidInput(format!(
            "Parameter {} not found",
            param_name
        )))
    }
}

/// QML utilities
pub struct QMLUtils;

impl QMLUtils {
    /// Create a simple variational quantum classifier
    pub fn create_vqc(num_qubits: usize, num_layers: usize) -> QuantumNeuralNetwork {
        let mut layers = Vec::new();

        // Data encoding layer
        layers.push(QMLLayer {
            layer_type: QMLLayerType::DataEncoding,
            name: "encoding".to_string(),
            num_qubits,
            parameters: Vec::new(),
            parameter_names: Vec::new(),
            circuit_template: None,
            classical_function: None,
            config: LayerConfig::default(),
        });

        // Variational layers
        for layer_idx in 0..num_layers {
            let num_params = num_qubits * 3; // 3 parameters per qubit (RX, RY, RZ)
            let parameters = (0..num_params)
                .map(|_| fastrand::f64() * 2.0 * std::f64::consts::PI)
                .collect();
            let parameter_names = (0..num_params).map(|i| format!("param_{}", i)).collect();

            layers.push(QMLLayer {
                layer_type: QMLLayerType::VariationalCircuit,
                name: format!("var_layer_{}", layer_idx),
                num_qubits,
                parameters,
                parameter_names,
                circuit_template: Some(Self::create_variational_circuit_template(num_qubits)),
                classical_function: None,
                config: LayerConfig {
                    repetitions: 1,
                    entangling_pattern: (0..num_qubits - 1).map(|i| (i, i + 1)).collect(),
                    ..Default::default()
                },
            });
        }

        // Measurement layer
        layers.push(QMLLayer {
            layer_type: QMLLayerType::Measurement,
            name: "measurement".to_string(),
            num_qubits,
            parameters: Vec::new(),
            parameter_names: Vec::new(),
            circuit_template: None,
            classical_function: None,
            config: LayerConfig::default(),
        });

        QuantumNeuralNetwork {
            layers,
            global_parameters: HashMap::new(),
            metadata: QNNMetadata {
                name: Some("VQC".to_string()),
                total_parameters: num_layers * num_qubits * 3,
                trainable_parameters: num_layers * num_qubits * 3,
                ..Default::default()
            },
            training_config: TrainingConfig::default(),
        }
    }

    /// Create variational circuit template
    fn create_variational_circuit_template(num_qubits: usize) -> InterfaceCircuit {
        let mut circuit = InterfaceCircuit::new(num_qubits, 0);

        // Add parameterized rotation gates
        for qubit in 0..num_qubits {
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::RX(0.0), vec![qubit]));
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.0), vec![qubit]));
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::RZ(0.0), vec![qubit]));
        }

        // Add entangling gates
        for qubit in 0..num_qubits - 1 {
            circuit.add_gate(InterfaceGate::new(
                InterfaceGateType::CNOT,
                vec![qubit, qubit + 1],
            ));
        }

        circuit
    }

    /// Create training data for XOR problem
    pub fn create_xor_training_data() -> Vec<TrainingExample> {
        vec![
            TrainingExample {
                input: Array1::from(vec![0.0, 0.0]),
                target: Array1::from(vec![0.0]),
            },
            TrainingExample {
                input: Array1::from(vec![0.0, 1.0]),
                target: Array1::from(vec![1.0]),
            },
            TrainingExample {
                input: Array1::from(vec![1.0, 0.0]),
                target: Array1::from(vec![1.0]),
            },
            TrainingExample {
                input: Array1::from(vec![1.0, 1.0]),
                target: Array1::from(vec![0.0]),
            },
        ]
    }

    /// Benchmark QML integration
    pub fn benchmark_qml_integration() -> Result<QMLBenchmarkResults> {
        let mut results = QMLBenchmarkResults::default();

        let configs = vec![
            QMLIntegrationConfig {
                framework: QMLFramework::SciRS2,
                enable_autodiff: false,
                batch_size: 4,
                ..Default::default()
            },
            QMLIntegrationConfig {
                framework: QMLFramework::SciRS2,
                enable_autodiff: true,
                batch_size: 4,
                ..Default::default()
            },
        ];

        for (i, config) in configs.into_iter().enumerate() {
            let mut integration = QMLIntegration::new(config)?;
            let mut qnn = Self::create_vqc(2, 2);
            qnn.training_config.epochs = 10;

            let training_data = Self::create_xor_training_data();

            let start = std::time::Instant::now();
            let _result = integration.train_qnn(qnn, &training_data, None)?;
            let time = start.elapsed().as_secs_f64() * 1000.0;

            results.training_times.push((format!("config_{}", i), time));
        }

        Ok(results)
    }
}

/// QML benchmark results
#[derive(Debug, Clone, Default)]
pub struct QMLBenchmarkResults {
    /// Training times by configuration
    pub training_times: Vec<(String, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_qml_integration_creation() {
        let config = QMLIntegrationConfig::default();
        let integration = QMLIntegration::new(config);
        assert!(integration.is_ok());
    }

    #[test]
    fn test_quantum_neural_network_creation() {
        let qnn = QMLUtils::create_vqc(2, 2);
        assert_eq!(qnn.layers.len(), 4); // encoding + 2 variational + measurement
        assert_eq!(qnn.metadata.total_parameters, 12); // 2 layers * 2 qubits * 3 params
    }

    #[test]
    fn test_training_data_creation() {
        let data = QMLUtils::create_xor_training_data();
        assert_eq!(data.len(), 4);
        assert_eq!(data[0].input, Array1::from(vec![0.0, 0.0]));
        assert_eq!(data[0].target, Array1::from(vec![0.0]));
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = AdamOptimizer::new(0.01);
        assert_eq!(optimizer.get_learning_rate(), 0.01);

        optimizer.update_learning_rate(0.5);
        assert_abs_diff_eq!(optimizer.get_learning_rate(), 0.005, epsilon = 1e-10);
    }

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = SGDOptimizer::new(0.1);
        assert_eq!(optimizer.get_learning_rate(), 0.1);

        optimizer.update_learning_rate(0.9);
        assert_abs_diff_eq!(optimizer.get_learning_rate(), 0.09, epsilon = 1e-10);
    }

    #[test]
    fn test_qml_layer_types() {
        let layer_types = vec![
            QMLLayerType::VariationalCircuit,
            QMLLayerType::DataEncoding,
            QMLLayerType::Measurement,
            QMLLayerType::Classical,
        ];
        assert_eq!(layer_types.len(), 4);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.optimizer, OptimizerType::Adam);
        assert_eq!(config.loss_function, LossFunction::MeanSquaredError);
    }

    #[test]
    fn test_measurement_probability_computation() {
        let config = QMLIntegrationConfig::default();
        let integration = QMLIntegration::new(config).unwrap();

        // Create a simple state |01⟩
        let mut state = Array1::zeros(4);
        state[1] = Complex64::new(1.0, 0.0); // |01⟩

        let prob0 = integration
            .compute_measurement_probability(0, &state)
            .unwrap();
        let prob1 = integration
            .compute_measurement_probability(1, &state)
            .unwrap();

        assert_abs_diff_eq!(prob0, 1.0, epsilon = 1e-10); // Qubit 0 is in |1⟩
        assert_abs_diff_eq!(prob1, 0.0, epsilon = 1e-10); // Qubit 1 is in |0⟩
    }

    #[test]
    fn test_loss_computation() {
        let config = QMLIntegrationConfig::default();
        let integration = QMLIntegration::new(config).unwrap();

        let prediction = Array1::from(vec![0.8, 0.2]);
        let target = Array1::from(vec![1.0, 0.0]);

        let mse = integration
            .compute_loss(&prediction, &target, &LossFunction::MeanSquaredError)
            .unwrap();
        let mae = integration
            .compute_loss(&prediction, &target, &LossFunction::MeanAbsoluteError)
            .unwrap();

        assert_abs_diff_eq!(mse, 0.04, epsilon = 1e-10); // ((0.8-1.0)^2 + (0.2-0.0)^2) / 2 = (0.04 + 0.04) / 2
        assert_abs_diff_eq!(mae, 0.2, epsilon = 1e-10); // (0.2 + 0.2) / 2
    }

    #[test]
    fn test_circuit_template_creation() {
        let circuit = QMLUtils::create_variational_circuit_template(3);
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.gates.len(), 11); // 3*3 rotation gates + 2 CNOT gates
    }
}
