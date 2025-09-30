//! TensorFlow Quantum compatibility layer for QuantRS2-ML
//!
//! This module provides a compatibility layer that mimics TensorFlow Quantum APIs,
//! allowing easy migration of TFQ models to QuantRS2 while maintaining familiar
//! interfaces and conventions.

use crate::circuit_integration::{QuantumLayer, QuantumMLExecutor};
use crate::error::{MLError, Result};
use crate::simulator_backends::{DynamicCircuit, Observable, SimulationResult, SimulatorBackend};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayD, Axis};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// TensorFlow Quantum-style quantum layer
pub struct QuantumCircuitLayer {
    /// Quantum circuit
    circuit: Circuit<8>, // Fixed const size for now
    /// Parameter symbols
    symbols: Vec<String>,
    /// Observable for measurement
    observable: Observable,
    /// Backend for execution
    backend: Arc<dyn SimulatorBackend>,
    /// Differentiable flag
    differentiable: bool,
    /// Repetitions for sampling
    repetitions: Option<usize>,
}

impl QuantumCircuitLayer {
    /// Create new quantum circuit layer
    pub fn new(
        circuit: Circuit<8>, // Fixed const size for now
        symbols: Vec<String>,
        observable: Observable,
        backend: Arc<dyn SimulatorBackend>,
    ) -> Self {
        Self {
            circuit,
            symbols,
            observable,
            backend,
            differentiable: true,
            repetitions: None,
        }
    }

    /// Set differentiable flag
    pub fn set_differentiable(mut self, differentiable: bool) -> Self {
        self.differentiable = differentiable;
        self
    }

    /// Set repetitions for sampling
    pub fn set_repetitions(mut self, repetitions: usize) -> Self {
        self.repetitions = Some(repetitions);
        self
    }

    /// Forward pass through quantum layer
    pub fn forward(&self, inputs: &Array2<f64>, parameters: &Array2<f64>) -> Result<Array1<f64>> {
        let batch_size = inputs.nrows();
        let mut outputs = Array1::zeros(batch_size);

        for batch_idx in 0..batch_size {
            // Combine input data with trainable parameters
            let input_data = inputs.row(batch_idx);
            let param_data = parameters.row(batch_idx % parameters.nrows());
            let combined_params: Vec<f64> = input_data
                .iter()
                .chain(param_data.iter())
                .copied()
                .collect();

            // Execute quantum circuit
            let dynamic_circuit =
                crate::simulator_backends::DynamicCircuit::from_circuit(self.circuit.clone())?;
            let expectation = self.backend.expectation_value(
                &dynamic_circuit,
                &combined_params,
                &self.observable,
            )?;

            outputs[batch_idx] = expectation;
        }

        Ok(outputs)
    }

    /// Compute gradients using parameter shift rule
    pub fn compute_gradients(
        &self,
        inputs: &Array2<f64>,
        parameters: &Array2<f64>,
        upstream_gradients: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        if !self.differentiable {
            return Err(MLError::InvalidConfiguration(
                "Layer is not differentiable".to_string(),
            ));
        }

        let batch_size = inputs.nrows();
        let num_input_params = inputs.ncols();
        let num_trainable_params = parameters.ncols();

        let mut input_gradients = Array2::zeros((batch_size, num_input_params));
        let mut param_gradients = Array2::zeros((batch_size, num_trainable_params));

        for batch_idx in 0..batch_size {
            let input_data = inputs.row(batch_idx);
            let param_data = parameters.row(batch_idx % parameters.nrows());
            let combined_params: Vec<f64> = input_data
                .iter()
                .chain(param_data.iter())
                .copied()
                .collect();

            // Compute gradients using parameter shift rule
            let dynamic_circuit =
                crate::simulator_backends::DynamicCircuit::from_circuit(self.circuit.clone())?;
            let gradients = self.backend.compute_gradients(
                &dynamic_circuit,
                &combined_params,
                &self.observable,
                crate::simulator_backends::GradientMethod::ParameterShift,
            )?;

            // Split gradients between inputs and parameters
            let upstream_grad = upstream_gradients[batch_idx];
            for (i, grad) in gradients.iter().enumerate() {
                if i < num_input_params {
                    input_gradients[[batch_idx, i]] = grad * upstream_grad;
                } else {
                    param_gradients[[batch_idx, i - num_input_params]] = grad * upstream_grad;
                }
            }
        }

        Ok((input_gradients, param_gradients))
    }
}

/// TensorFlow Quantum-style PQC (Parameterized Quantum Circuit) layer
pub struct PQCLayer {
    /// Base quantum circuit layer
    layer: QuantumCircuitLayer,
    /// Input scaling factor
    input_scaling: f64,
    /// Parameter initialization strategy
    init_strategy: ParameterInitStrategy,
    /// Regularization
    regularization: Option<RegularizationType>,
}

/// Parameter initialization strategies
#[derive(Debug, Clone)]
pub enum ParameterInitStrategy {
    /// Random normal initialization
    RandomNormal { mean: f64, std: f64 },
    /// Random uniform initialization
    RandomUniform { low: f64, high: f64 },
    /// Zero initialization
    Zeros,
    /// Ones initialization
    Ones,
    /// Custom initialization
    Custom(Vec<f64>),
}

/// Regularization types
#[derive(Debug, Clone)]
pub enum RegularizationType {
    /// L1 regularization
    L1(f64),
    /// L2 regularization
    L2(f64),
    /// Dropout (for quantum circuits)
    Dropout(f64),
}

impl PQCLayer {
    /// Create new PQC layer
    pub fn new(
        circuit: Circuit<8>, // Fixed const size for now
        symbols: Vec<String>,
        observable: Observable,
        backend: Arc<dyn SimulatorBackend>,
    ) -> Self {
        let layer = QuantumCircuitLayer::new(circuit, symbols, observable, backend);

        Self {
            layer,
            input_scaling: 1.0,
            init_strategy: ParameterInitStrategy::RandomNormal {
                mean: 0.0,
                std: 0.1,
            },
            regularization: None,
        }
    }

    /// Set input scaling
    pub fn with_input_scaling(mut self, scaling: f64) -> Self {
        self.input_scaling = scaling;
        self
    }

    /// Set parameter initialization strategy
    pub fn with_initialization(mut self, strategy: ParameterInitStrategy) -> Self {
        self.init_strategy = strategy;
        self
    }

    /// Set regularization
    pub fn with_regularization(mut self, regularization: RegularizationType) -> Self {
        self.regularization = Some(regularization);
        self
    }

    /// Initialize parameters
    pub fn initialize_parameters(&self, batch_size: usize, num_params: usize) -> Array2<f64> {
        match &self.init_strategy {
            ParameterInitStrategy::RandomNormal { mean, std } => {
                // Use Box-Muller transform for normal distribution
                Array2::from_shape_fn((batch_size, num_params), |_| {
                    let u1 = fastrand::f64();
                    let u2 = fastrand::f64();
                    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    mean + std * z0
                })
            }
            ParameterInitStrategy::RandomUniform { low, high } => {
                Array2::from_shape_fn((batch_size, num_params), |_| {
                    fastrand::f64() * (high - low) + low
                })
            }
            ParameterInitStrategy::Zeros => Array2::zeros((batch_size, num_params)),
            ParameterInitStrategy::Ones => Array2::ones((batch_size, num_params)),
            ParameterInitStrategy::Custom(values) => {
                let mut params = Array2::zeros((batch_size, num_params));
                for i in 0..batch_size {
                    for j in 0..num_params.min(values.len()) {
                        params[[i, j]] = values[j];
                    }
                }
                params
            }
        }
    }

    /// Forward pass with input scaling
    pub fn forward(&self, inputs: &Array2<f64>, parameters: &Array2<f64>) -> Result<Array1<f64>> {
        // Scale inputs
        let scaled_inputs = inputs * self.input_scaling;

        // Forward through quantum layer
        let outputs = self.layer.forward(&scaled_inputs, parameters)?;

        // Apply regularization if needed
        // (In practice, regularization would be applied during loss computation)

        Ok(outputs)
    }

    /// Compute gradients with regularization
    pub fn compute_gradients(
        &self,
        inputs: &Array2<f64>,
        parameters: &Array2<f64>,
        upstream_gradients: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let scaled_inputs = inputs * self.input_scaling;
        let (mut input_grads, mut param_grads) =
            self.layer
                .compute_gradients(&scaled_inputs, parameters, upstream_gradients)?;

        // Scale input gradients
        input_grads *= self.input_scaling;

        // Add regularization gradients
        if let Some(ref reg) = self.regularization {
            match reg {
                RegularizationType::L1(lambda) => {
                    param_grads += &(parameters.mapv(|x| lambda * x.signum()));
                }
                RegularizationType::L2(lambda) => {
                    param_grads += &(parameters * (2.0 * lambda));
                }
                RegularizationType::Dropout(_) => {
                    // Dropout would be applied during forward pass
                }
            }
        }

        Ok((input_grads, param_grads))
    }
}

/// Quantum convolutional layer (TFQ-style)
pub struct QuantumConvolutionalLayer {
    /// Base PQC layer
    pqc: PQCLayer,
    /// Convolution parameters
    filter_size: (usize, usize),
    /// Stride
    stride: (usize, usize),
    /// Padding
    padding: PaddingType,
}

/// Padding types for quantum convolution
#[derive(Debug, Clone)]
pub enum PaddingType {
    /// Valid padding (no padding)
    Valid,
    /// Same padding (maintain input size)
    Same,
    /// Custom padding
    Custom(usize),
}

impl QuantumConvolutionalLayer {
    /// Create new quantum convolutional layer
    pub fn new(
        circuit: Circuit<8>, // Fixed const size for now
        symbols: Vec<String>,
        observable: Observable,
        backend: Arc<dyn SimulatorBackend>,
        filter_size: (usize, usize),
    ) -> Self {
        let pqc = PQCLayer::new(circuit, symbols, observable, backend);

        Self {
            pqc,
            filter_size,
            stride: (1, 1),
            padding: PaddingType::Valid,
        }
    }

    /// Set stride
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn with_padding(mut self, padding: PaddingType) -> Self {
        self.padding = padding;
        self
    }

    /// Apply quantum convolution to input tensor
    pub fn forward(&self, inputs: &Array4<f64>, parameters: &Array2<f64>) -> Result<Array4<f64>> {
        let (batch_size, height, width, channels) = inputs.dim();
        let (filter_h, filter_w) = self.filter_size;
        let (stride_h, stride_w) = self.stride;

        // Calculate output dimensions
        let output_h = (height - filter_h) / stride_h + 1;
        let output_w = (width - filter_w) / stride_w + 1;

        let mut outputs = Array4::zeros((batch_size, output_h, output_w, 1));

        for batch in 0..batch_size {
            for out_y in 0..output_h {
                for out_x in 0..output_w {
                    // Extract patch
                    let start_y = out_y * stride_h;
                    let start_x = out_x * stride_w;

                    let mut patch_data = Array2::zeros((1, filter_h * filter_w * channels));
                    let mut patch_idx = 0;

                    for dy in 0..filter_h {
                        for dx in 0..filter_w {
                            for c in 0..channels {
                                if start_y + dy < height && start_x + dx < width {
                                    patch_data[[0, patch_idx]] =
                                        inputs[[batch, start_y + dy, start_x + dx, c]];
                                }
                                patch_idx += 1;
                            }
                        }
                    }

                    // Apply quantum circuit to patch
                    let result = self.pqc.forward(&patch_data, parameters)?;
                    outputs[[batch, out_y, out_x, 0]] = result[0];
                }
            }
        }

        Ok(outputs)
    }
}

/// TensorFlow Quantum-style model builder
pub struct TFQModel {
    /// Layers in the model
    layers: Vec<Box<dyn TFQLayer>>,
    /// Input shape
    input_shape: Vec<usize>,
    /// Loss function
    loss_function: TFQLossFunction,
    /// Optimizer
    optimizer: TFQOptimizer,
}

/// TensorFlow Quantum layer trait
pub trait TFQLayer: Send + Sync {
    /// Forward pass
    fn forward(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>>;

    /// Backward pass
    fn backward(&self, upstream_gradients: &ArrayD<f64>) -> Result<ArrayD<f64>>;

    /// Get trainable parameters
    fn get_parameters(&self) -> Vec<Array1<f64>>;

    /// Set trainable parameters
    fn set_parameters(&mut self, params: Vec<Array1<f64>>) -> Result<()>;

    /// Layer name
    fn name(&self) -> &str;
}

/// TensorFlow Quantum loss functions
#[derive(Debug, Clone)]
pub enum TFQLossFunction {
    /// Mean squared error
    MeanSquaredError,
    /// Binary crossentropy
    BinaryCrossentropy,
    /// Categorical crossentropy
    CategoricalCrossentropy,
    /// Hinge loss
    Hinge,
    /// Custom loss function
    Custom(String),
}

/// TensorFlow Quantum optimizers
#[derive(Debug, Clone)]
pub enum TFQOptimizer {
    /// Adam optimizer
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// SGD optimizer
    SGD { learning_rate: f64, momentum: f64 },
    /// RMSprop optimizer
    RMSprop {
        learning_rate: f64,
        rho: f64,
        epsilon: f64,
    },
}

impl TFQModel {
    /// Create new TFQ model
    pub fn new(input_shape: Vec<usize>) -> Self {
        Self {
            layers: Vec::new(),
            input_shape,
            loss_function: TFQLossFunction::MeanSquaredError,
            optimizer: TFQOptimizer::Adam {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        }
    }

    /// Add layer to model
    pub fn add_layer(&mut self, layer: Box<dyn TFQLayer>) {
        self.layers.push(layer);
    }

    /// Set loss function
    pub fn set_loss(mut self, loss: TFQLossFunction) -> Self {
        self.loss_function = loss;
        self
    }

    /// Set optimizer
    pub fn set_optimizer(mut self, optimizer: TFQOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Compile model
    pub fn compile(&mut self) -> Result<()> {
        // Validate layer connections and prepare for training
        if self.layers.is_empty() {
            return Err(MLError::InvalidConfiguration(
                "Model must have at least one layer".to_string(),
            ));
        }

        Ok(())
    }

    /// Forward pass through model
    pub fn predict(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        let mut current = inputs.clone();

        for layer in &self.layers {
            current = layer.forward(&current)?;
        }

        Ok(current)
    }

    /// Train model for one epoch
    pub fn train_step(&mut self, inputs: &ArrayD<f64>, targets: &ArrayD<f64>) -> Result<f64> {
        // Forward pass
        let predictions = self.predict(inputs)?;

        // Compute loss
        let loss = self.compute_loss(&predictions, targets)?;

        // Backward pass
        let mut gradients = self.compute_loss_gradients(&predictions, targets)?;

        for layer in self.layers.iter().rev() {
            gradients = layer.backward(&gradients)?;
        }

        // Update parameters
        self.update_parameters()?;

        Ok(loss)
    }

    /// Compute loss
    fn compute_loss(&self, predictions: &ArrayD<f64>, targets: &ArrayD<f64>) -> Result<f64> {
        match &self.loss_function {
            TFQLossFunction::MeanSquaredError => {
                let diff = predictions - targets;
                Ok(diff.mapv(|x| x * x).mean().unwrap())
            }
            TFQLossFunction::BinaryCrossentropy => {
                let epsilon = 1e-15;
                let clipped_preds = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                let loss = targets * clipped_preds.mapv(|x| x.ln())
                    + (1.0 - targets) * clipped_preds.mapv(|x| (1.0 - x).ln());
                Ok(-loss.mean().unwrap())
            }
            _ => Err(MLError::InvalidConfiguration(
                "Loss function not implemented".to_string(),
            )),
        }
    }

    /// Compute loss gradients
    fn compute_loss_gradients(
        &self,
        predictions: &ArrayD<f64>,
        targets: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>> {
        match &self.loss_function {
            TFQLossFunction::MeanSquaredError => {
                Ok(2.0 * (predictions - targets) / predictions.len() as f64)
            }
            TFQLossFunction::BinaryCrossentropy => {
                let epsilon = 1e-15;
                let clipped_preds = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                Ok((clipped_preds.clone() - targets)
                    / (clipped_preds.clone() * (1.0 - &clipped_preds)))
            }
            _ => Err(MLError::InvalidConfiguration(
                "Loss gradient not implemented".to_string(),
            )),
        }
    }

    /// Update model parameters
    fn update_parameters(&mut self) -> Result<()> {
        // Placeholder - would implement parameter updates based on optimizer
        Ok(())
    }
}

/// TensorFlow Quantum-style quantum dataset utilities
pub struct QuantumDataset {
    /// Circuit data
    circuits: Vec<DynamicCircuit>,
    /// Parameter data
    parameters: Array2<f64>,
    /// Labels
    labels: Array1<f64>,
    /// Batch size
    batch_size: usize,
}

impl QuantumDataset {
    /// Create new quantum dataset
    pub fn new(
        circuits: Vec<Circuit<8>>, // Fixed const size for now
        parameters: Array2<f64>,
        labels: Array1<f64>,
        batch_size: usize,
    ) -> Result<Self> {
        let dynamic_circuits: std::result::Result<Vec<DynamicCircuit>, crate::error::MLError> =
            circuits
                .into_iter()
                .map(|c| DynamicCircuit::from_circuit(c))
                .collect();

        Ok(Self {
            circuits: dynamic_circuits?,
            parameters,
            labels,
            batch_size,
        })
    }

    /// Get batch iterator
    pub fn batches(&self) -> QuantumDatasetIterator {
        QuantumDatasetIterator::new(self)
    }

    /// Shuffle dataset
    pub fn shuffle(&mut self) {
        let n = self.circuits.len();
        let mut indices: Vec<usize> = (0..n).collect();

        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = fastrand::usize(0..=i);
            indices.swap(i, j);
        }

        // Reorder data based on shuffled indices
        let mut new_circuits = Vec::with_capacity(n);
        let mut new_parameters = Array2::zeros(self.parameters.dim());
        let mut new_labels = Array1::zeros(self.labels.dim());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            new_circuits.push(self.circuits[old_idx].clone());
            new_parameters
                .row_mut(new_idx)
                .assign(&self.parameters.row(old_idx));
            new_labels[new_idx] = self.labels[old_idx];
        }

        self.circuits = new_circuits;
        self.parameters = new_parameters;
        self.labels = new_labels;
    }
}

/// Iterator for quantum dataset batches
pub struct QuantumDatasetIterator<'a> {
    dataset: &'a QuantumDataset,
    current_batch: usize,
    total_batches: usize,
}

impl<'a> QuantumDatasetIterator<'a> {
    fn new(dataset: &'a QuantumDataset) -> Self {
        let total_batches = (dataset.circuits.len() + dataset.batch_size - 1) / dataset.batch_size;
        Self {
            dataset,
            current_batch: 0,
            total_batches,
        }
    }
}

impl<'a> Iterator for QuantumDatasetIterator<'a> {
    type Item = (Vec<DynamicCircuit>, Array2<f64>, Array1<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.total_batches {
            return None;
        }

        let start_idx = self.current_batch * self.dataset.batch_size;
        let end_idx =
            ((self.current_batch + 1) * self.dataset.batch_size).min(self.dataset.circuits.len());

        let batch_circuits = self.dataset.circuits[start_idx..end_idx].to_vec();
        let batch_parameters = self
            .dataset
            .parameters
            .slice(s![start_idx..end_idx, ..])
            .to_owned();
        let batch_labels = self.dataset.labels.slice(s![start_idx..end_idx]).to_owned();

        self.current_batch += 1;
        Some((batch_circuits, batch_parameters, batch_labels))
    }
}

/// TensorFlow Quantum-style utilities
pub mod tfq_utils {
    use super::*;

    /// Convert QuantRS2 circuit to TFQ-compatible format
    pub fn circuit_to_tfq_format(circuit: &DynamicCircuit) -> Result<TFQCircuitFormat> {
        // TODO: Implement proper gate extraction from DynamicCircuit
        // This requires implementing gates() method on DynamicCircuit
        // For now, return empty gates list
        let tfq_gates: Vec<TFQGate> = Vec::new();

        Ok(TFQCircuitFormat {
            gates: tfq_gates,
            num_qubits: circuit.num_qubits(),
        })
    }

    /// Create quantum data encoding circuit
    pub fn create_data_encoding_circuit(
        num_qubits: usize,
        encoding_type: DataEncodingType,
    ) -> Result<DynamicCircuit> {
        let mut builder: Circuit<8> = CircuitBuilder::new(); // Fixed size for compatibility

        match encoding_type {
            DataEncodingType::Amplitude => {
                // Create amplitude encoding circuit
                for qubit in 0..num_qubits {
                    builder.ry(qubit, 0.0)?; // Parameterized rotation
                }
            }
            DataEncodingType::Angle => {
                // Create angle encoding circuit
                for qubit in 0..num_qubits {
                    builder.rz(qubit, 0.0)?; // Parameterized rotation
                }
            }
            DataEncodingType::Basis => {
                // Basis encoding (computational basis)
                for qubit in 0..num_qubits {
                    builder.x(qubit)?; // Conditional X gate based on data
                }
            }
        }

        let circuit = builder.build();
        DynamicCircuit::from_circuit(circuit)
    }

    /// Create hardware-efficient ansatz
    pub fn create_hardware_efficient_ansatz(
        num_qubits: usize,
        layers: usize,
    ) -> Result<DynamicCircuit> {
        let mut builder: Circuit<8> = CircuitBuilder::new(); // Fixed size for compatibility

        for layer in 0..layers {
            // Rotation gates
            for qubit in 0..num_qubits {
                builder.ry(qubit, 0.0)?;
                builder.rz(qubit, 0.0)?;
            }

            // Entangling gates
            for qubit in 0..num_qubits - 1 {
                builder.cnot(qubit, qubit + 1)?;
            }

            // Add final entangling gate if needed
            if layer < layers - 1 && num_qubits > 2 {
                builder.cnot(num_qubits - 1, 0)?;
            }
        }

        let circuit = builder.build();
        DynamicCircuit::from_circuit(circuit)
    }

    /// Batch quantum circuit execution
    pub fn batch_execute_circuits(
        circuits: &[DynamicCircuit],
        parameters: &Array2<f64>,
        observables: &[Observable],
        backend: &dyn SimulatorBackend,
    ) -> Result<Array2<f64>> {
        let batch_size = circuits.len();
        let num_observables = observables.len();
        let mut results = Array2::zeros((batch_size, num_observables));

        for (circuit_idx, circuit) in circuits.iter().enumerate() {
            let params = parameters.row(circuit_idx % parameters.nrows());

            for (obs_idx, observable) in observables.iter().enumerate() {
                let expectation =
                    backend.expectation_value(circuit, params.as_slice().unwrap(), observable)?;
                results[[circuit_idx, obs_idx]] = expectation;
            }
        }

        Ok(results)
    }
}

/// TensorFlow Quantum circuit format
#[derive(Debug, Clone)]
pub struct TFQCircuitFormat {
    /// Gate sequence
    gates: Vec<TFQGate>,
    /// Number of qubits
    num_qubits: usize,
}

/// TensorFlow Quantum gate representation
#[derive(Debug, Clone)]
pub struct TFQGate {
    /// Gate type
    gate_type: String,
    /// Target qubits
    qubits: Vec<usize>,
    /// Parameters
    parameters: Vec<f64>,
}

/// Data encoding types for TFQ compatibility
#[derive(Debug, Clone)]
pub enum DataEncodingType {
    /// Amplitude encoding
    Amplitude,
    /// Angle encoding
    Angle,
    /// Basis encoding
    Basis,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator_backends::{BackendCapabilities, StatevectorBackend};

    #[test]
    #[ignore]
    fn test_quantum_circuit_layer() {
        let mut builder = CircuitBuilder::new();
        builder.ry(0, 0.0).unwrap();
        builder.ry(1, 0.0).unwrap();
        builder.cnot(0, 1).unwrap();
        let circuit = builder.build();

        let symbols = vec!["theta1".to_string(), "theta2".to_string()];
        let observable = Observable::PauliZ(vec![0, 1]);
        let backend = Arc::new(StatevectorBackend::new(8));

        let layer = QuantumCircuitLayer::new(circuit, symbols, observable, backend);

        let inputs = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let parameters = Array2::from_shape_vec((2, 2), vec![0.5, 0.6, 0.7, 0.8]).unwrap();

        let result = layer.forward(&inputs, &parameters);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pqc_layer_initialization() -> Result<()> {
        let mut builder = CircuitBuilder::new();
        builder.h(0)?;
        let circuit = builder.build();

        let symbols = vec!["param1".to_string()];
        let observable = Observable::PauliZ(vec![0]);
        let backend = Arc::new(StatevectorBackend::new(8));

        let pqc = PQCLayer::new(circuit, symbols, observable, backend).with_initialization(
            ParameterInitStrategy::RandomNormal {
                mean: 0.0,
                std: 0.1,
            },
        );

        let params = pqc.initialize_parameters(5, 3);
        assert_eq!(params.shape(), &[5, 3]);
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_tfq_utils() {
        let circuit = tfq_utils::create_data_encoding_circuit(3, DataEncodingType::Angle).unwrap();
        assert_eq!(circuit.num_qubits(), 3);

        let ansatz = tfq_utils::create_hardware_efficient_ansatz(4, 2).unwrap();
        assert_eq!(ansatz.num_qubits(), 4);
    }

    #[test]
    fn test_quantum_dataset() -> Result<()> {
        let circuits = vec![CircuitBuilder::new().build(), CircuitBuilder::new().build()];
        let parameters =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0]);

        let dataset = QuantumDataset::new(circuits, parameters, labels, 1);
        let dataset = dataset?;
        let batches: Vec<_> = dataset.batches().collect();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].0.len(), 1); // Batch size 1
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_tfq_model() {
        let mut model = TFQModel::new(vec![2, 2])
            .set_loss(TFQLossFunction::MeanSquaredError)
            .set_optimizer(TFQOptimizer::Adam {
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            });

        assert!(model.compile().is_ok());
    }
}
