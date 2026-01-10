//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::simulator_backends::{DynamicCircuit, Observable, SimulationResult, SimulatorBackend};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayD, Axis};
use std::sync::Arc;

use super::functions::{Differentiator, TFQLayer};

/// TensorFlow Quantum circuit format
#[derive(Debug, Clone)]
pub struct TFQCircuitFormat {
    /// Gate sequence
    pub(crate) gates: Vec<TFQGate>,
    /// Number of qubits
    pub(crate) num_qubits: usize,
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
        let predictions = self.predict(inputs)?;
        let loss = self.compute_loss(&predictions, targets)?;
        let mut gradients = self.compute_loss_gradients(&predictions, targets)?;
        for layer in self.layers.iter().rev() {
            gradients = layer.backward(&gradients)?;
        }
        self.update_parameters()?;
        Ok(loss)
    }
    /// Compute loss
    fn compute_loss(&self, predictions: &ArrayD<f64>, targets: &ArrayD<f64>) -> Result<f64> {
        match &self.loss_function {
            TFQLossFunction::MeanSquaredError => {
                let diff = predictions - targets;
                diff.mapv(|x| x * x).mean().ok_or_else(|| {
                    MLError::InvalidConfiguration("Cannot compute mean of empty array".to_string())
                })
            }
            TFQLossFunction::BinaryCrossentropy => {
                let epsilon = 1e-15;
                let clipped_preds = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                let loss = targets * clipped_preds.mapv(|x| x.ln())
                    + (1.0 - targets) * clipped_preds.mapv(|x| (1.0 - x).ln());
                let mean_loss = loss.mean().ok_or_else(|| {
                    MLError::InvalidConfiguration("Cannot compute mean of empty array".to_string())
                })?;
                Ok(-mean_loss)
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
        Ok(())
    }
}
/// Expectation layer with analytical computation
pub struct ExpectationLayer {
    /// Base circuit
    circuit: Circuit<8>,
    /// Observables to measure
    observables: Vec<Observable>,
    /// Backend for execution
    backend: Arc<dyn SimulatorBackend>,
    /// Differentiator
    differentiator: Box<dyn Differentiator>,
}
impl ExpectationLayer {
    /// Create new expectation layer
    pub fn new(
        circuit: Circuit<8>,
        observables: Vec<Observable>,
        backend: Arc<dyn SimulatorBackend>,
    ) -> Self {
        Self {
            circuit,
            observables,
            backend,
            differentiator: Box::new(ParameterShiftDifferentiator::new()),
        }
    }
    /// Set differentiator
    pub fn with_differentiator(mut self, differentiator: Box<dyn Differentiator>) -> Self {
        self.differentiator = differentiator;
        self
    }
    /// Forward pass computing expectation values
    pub fn forward(&self, parameters: &Array2<f64>) -> Result<Array2<f64>> {
        let batch_size = parameters.nrows();
        let num_observables = self.observables.len();
        let mut outputs = Array2::zeros((batch_size, num_observables));
        let dynamic_circuit =
            crate::simulator_backends::DynamicCircuit::from_circuit(self.circuit.clone())?;
        for batch_idx in 0..batch_size {
            let params = parameters.row(batch_idx);
            let params_slice = params.as_slice().ok_or_else(|| {
                MLError::InvalidConfiguration("Parameters must be contiguous".to_string())
            })?;
            for (obs_idx, observable) in self.observables.iter().enumerate() {
                let expectation =
                    self.backend
                        .expectation_value(&dynamic_circuit, params_slice, observable)?;
                outputs[[batch_idx, obs_idx]] = expectation;
            }
        }
        Ok(outputs)
    }
    /// Compute gradients
    pub fn compute_gradients(&self, parameters: &Array2<f64>) -> Result<Array3<f64>> {
        let batch_size = parameters.nrows();
        let num_params = parameters.ncols();
        let num_observables = self.observables.len();
        let mut gradients = Array3::zeros((batch_size, num_observables, num_params));
        let dynamic_circuit =
            crate::simulator_backends::DynamicCircuit::from_circuit(self.circuit.clone())?;
        for batch_idx in 0..batch_size {
            let params = parameters.row(batch_idx);
            let params_slice = params.as_slice().ok_or_else(|| {
                MLError::InvalidConfiguration("Parameters must be contiguous".to_string())
            })?;
            for (obs_idx, observable) in self.observables.iter().enumerate() {
                let obs_grads = self.differentiator.differentiate(
                    &dynamic_circuit,
                    params_slice,
                    observable,
                    self.backend.as_ref(),
                )?;
                for (param_idx, &grad) in obs_grads.iter().enumerate() {
                    gradients[[batch_idx, obs_idx, param_idx]] = grad;
                }
            }
        }
        Ok(gradients)
    }
}
/// Sample layer - returns measurement samples from quantum circuit
/// Similar to tfq.layers.Sample
pub struct SampleLayer {
    /// Circuit to sample
    circuit: Circuit<8>,
    /// Backend for execution
    backend: Arc<dyn SimulatorBackend>,
    /// Number of shots
    num_shots: usize,
}
impl SampleLayer {
    /// Create new sample layer
    pub fn new(circuit: Circuit<8>, backend: Arc<dyn SimulatorBackend>, num_shots: usize) -> Self {
        Self {
            circuit,
            backend,
            num_shots,
        }
    }
    /// Forward pass - sample from circuit
    pub fn forward(&self, parameters: &Array2<f64>) -> Result<Vec<Vec<usize>>> {
        let batch_size = parameters.nrows();
        let mut all_samples = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let params = parameters.row(batch_idx);
            let params_slice = params.as_slice().ok_or_else(|| {
                MLError::InvalidConfiguration("Parameters must be contiguous".to_string())
            })?;
            let dynamic_circuit = DynamicCircuit::from_circuit(self.circuit.clone())?;
            let result = self.backend.execute_circuit(
                &dynamic_circuit,
                params_slice,
                Some(self.num_shots),
            )?;
            let samples = if let Some(ref measurements) = result.measurements {
                measurements.to_vec()
            } else {
                let probs = result.probabilities.as_ref().ok_or_else(|| {
                    MLError::InvalidConfiguration(
                        "No measurements or probabilities available".to_string(),
                    )
                })?;
                let mut samples = Vec::with_capacity(self.num_shots);
                for _ in 0..self.num_shots {
                    let r = fastrand::f64();
                    let mut cumsum = 0.0;
                    for (i, &p) in probs.iter().enumerate() {
                        cumsum += p;
                        if r < cumsum {
                            samples.push(i);
                            break;
                        }
                    }
                }
                samples
            };
            all_samples.push(samples);
        }
        Ok(all_samples)
    }
}
/// Unitary layer - applies a unitary matrix to qubits
pub struct UnitaryLayer {
    /// Target qubits
    qubits: Vec<usize>,
    /// Unitary matrix (as flattened complex array)
    unitary: Array2<scirs2_core::Complex64>,
    static_mode: bool,
}
impl UnitaryLayer {
    /// Create new unitary layer
    pub fn new(qubits: Vec<usize>, unitary: Array2<scirs2_core::Complex64>) -> Result<Self> {
        let size = 1 << qubits.len();
        if unitary.shape() != [size, size] {
            return Err(MLError::InvalidConfiguration(format!(
                "Unitary matrix must be {}x{}",
                size, size
            )));
        }
        Ok(Self {
            qubits,
            unitary,
            static_mode: false,
        })
    }
    /// Check if matrix is unitary
    pub fn is_unitary(&self) -> bool {
        let u_dag = self.unitary.t().mapv(|x| x.conj());
        let product = u_dag.dot(&self.unitary);
        let size = self.unitary.shape()[0];
        let identity = Array2::eye(size).mapv(|x| scirs2_core::Complex64::new(x, 0.0));
        let diff = &product - &identity;
        let norm: f64 = diff.iter().map(|x| x.norm()).sum();
        norm < 1e-10
    }
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
/// Parameter shift rule differentiator
#[derive(Debug, Clone)]
pub struct ParameterShiftDifferentiator {
    /// Shift amount (default: π/2)
    pub(super) shift: f64,
}
impl ParameterShiftDifferentiator {
    /// Create new parameter shift differentiator
    pub fn new() -> Self {
        Self {
            shift: std::f64::consts::PI / 2.0,
        }
    }
    /// Create with custom shift
    pub fn with_shift(shift: f64) -> Self {
        Self { shift }
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
    /// Differentiation method
    pub(crate) differentiation_method: DifferentiationMethod,
}
impl PQCLayer {
    /// Create new PQC layer
    pub fn new(
        circuit: Circuit<8>,
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
            differentiation_method: DifferentiationMethod::ParameterShift,
        }
    }
    /// Set input scaling
    pub fn with_input_scaling(mut self, scaling: f64) -> Self {
        self.input_scaling = scaling;
        self
    }
    /// Set differentiation method
    pub fn with_differentiation(mut self, method: DifferentiationMethod) -> Self {
        self.differentiation_method = method;
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
            ParameterInitStrategy::GlorotUniform => {
                let limit = (6.0 / (2.0 * num_params as f64)).sqrt();
                Array2::from_shape_fn((batch_size, num_params), |_| {
                    fastrand::f64() * (2.0 * limit) - limit
                })
            }
            ParameterInitStrategy::GlorotNormal => {
                let std = (2.0 / (2.0 * num_params as f64)).sqrt();
                Array2::from_shape_fn((batch_size, num_params), |_| {
                    let u1 = fastrand::f64();
                    let u2 = fastrand::f64();
                    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    std * z0
                })
            }
        }
    }
    /// Forward pass with input scaling
    pub fn forward(&self, inputs: &Array2<f64>, parameters: &Array2<f64>) -> Result<Array1<f64>> {
        let scaled_inputs = inputs * self.input_scaling;
        let outputs = self.layer.forward(&scaled_inputs, parameters)?;
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
        input_grads *= self.input_scaling;
        if let Some(ref reg) = self.regularization {
            match reg {
                RegularizationType::L1(lambda) => {
                    param_grads += &(parameters.mapv(|x| lambda * x.signum()));
                }
                RegularizationType::L2(lambda) => {
                    param_grads += &(parameters * (2.0 * lambda));
                }
                RegularizationType::ElasticNet { l1_ratio, alpha } => {
                    let l1_part = parameters.mapv(|x| alpha * l1_ratio * x.signum());
                    let l2_part = parameters * (2.0 * alpha * (1.0 - l1_ratio));
                    param_grads += &(l1_part + l2_part);
                }
                RegularizationType::Dropout(_) => {}
            }
        }
        Ok((input_grads, param_grads))
    }
}
/// TensorFlow Quantum-style quantum layer
pub struct QuantumCircuitLayer {
    /// Quantum circuit
    circuit: Circuit<8>,
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
        circuit: Circuit<8>,
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
            let input_data = inputs.row(batch_idx);
            let param_data = parameters.row(batch_idx % parameters.nrows());
            let combined_params: Vec<f64> = input_data
                .iter()
                .chain(param_data.iter())
                .copied()
                .collect();
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
            let dynamic_circuit =
                crate::simulator_backends::DynamicCircuit::from_circuit(self.circuit.clone())?;
            let gradients = self.backend.compute_gradients(
                &dynamic_circuit,
                &combined_params,
                &self.observable,
                crate::simulator_backends::GradientMethod::ParameterShift,
            )?;
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
/// Regularization types
#[derive(Debug, Clone)]
pub enum RegularizationType {
    /// L1 regularization (Lasso)
    L1(f64),
    /// L2 regularization (Ridge)
    L2(f64),
    /// ElasticNet regularization (combined L1 + L2)
    /// ElasticNet(l1_ratio, alpha) where:
    /// - l1_ratio: mixing parameter (0 = L2 only, 1 = L1 only)
    /// - alpha: overall regularization strength
    /// Loss = alpha * (l1_ratio * |θ| + (1 - l1_ratio) * θ²)
    ElasticNet { l1_ratio: f64, alpha: f64 },
    /// Dropout (for quantum circuits)
    Dropout(f64),
}
/// Finite difference differentiator
#[derive(Debug, Clone)]
pub struct FiniteDifferenceDifferentiator {
    /// Step size for finite differences
    pub(super) epsilon: f64,
}
impl FiniteDifferenceDifferentiator {
    /// Create new finite difference differentiator
    pub fn new() -> Self {
        Self { epsilon: 1e-4 }
    }
    /// Create with custom epsilon
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self { epsilon }
    }
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
/// Sampled expectation layer that uses measurement sampling
pub struct SampledExpectationLayer {
    /// Base circuit
    circuit: Circuit<8>,
    /// Observable to measure
    observable: Observable,
    /// Backend for execution
    backend: Arc<dyn SimulatorBackend>,
    /// Number of shots per measurement
    num_shots: usize,
}
impl SampledExpectationLayer {
    /// Create new sampled expectation layer
    pub fn new(
        circuit: Circuit<8>,
        observable: Observable,
        backend: Arc<dyn SimulatorBackend>,
        num_shots: usize,
    ) -> Self {
        Self {
            circuit,
            observable,
            backend,
            num_shots,
        }
    }
    /// Set number of shots
    pub fn with_shots(mut self, num_shots: usize) -> Self {
        self.num_shots = num_shots;
        self
    }
    /// Forward pass with sampled measurements
    pub fn forward(&self, parameters: &Array2<f64>) -> Result<Array1<f64>> {
        let batch_size = parameters.nrows();
        let mut outputs = Array1::zeros(batch_size);
        for batch_idx in 0..batch_size {
            let params = parameters.row(batch_idx);
            let params_slice = params.as_slice().ok_or_else(|| {
                MLError::InvalidConfiguration("Parameters must be contiguous".to_string())
            })?;
            let dynamic_circuit =
                crate::simulator_backends::DynamicCircuit::from_circuit(self.circuit.clone())?;
            let result = self.backend.execute_circuit(
                &dynamic_circuit,
                params_slice,
                Some(self.num_shots),
            )?;
            let expectation = self.compute_expectation_from_samples(&result)?;
            outputs[batch_idx] = expectation;
        }
        Ok(outputs)
    }
    /// Compute expectation value from measurement samples
    fn compute_expectation_from_samples(&self, result: &SimulationResult) -> Result<f64> {
        match &self.observable {
            Observable::PauliZ(qubits) => {
                if let Some(ref measurements) = result.measurements {
                    let mut expectation = 0.0;
                    let total_shots = measurements.len() as f64;
                    for &measurement in measurements.iter() {
                        let mut parity = 0usize;
                        for &qubit in qubits {
                            parity ^= (measurement >> qubit) & 1;
                        }
                        let eigenvalue = if parity == 0 { 1.0 } else { -1.0 };
                        expectation += eigenvalue / total_shots;
                    }
                    Ok(expectation)
                } else if let Some(ref probabilities) = result.probabilities {
                    let mut expectation = 0.0;
                    for (i, &prob) in probabilities.iter().enumerate() {
                        let mut parity = 0usize;
                        for &qubit in qubits {
                            parity ^= (i >> qubit) & 1;
                        }
                        let eigenvalue = if parity == 0 { 1.0 } else { -1.0 };
                        expectation += eigenvalue * prob;
                    }
                    Ok(expectation)
                } else {
                    let dynamic_circuit = crate::simulator_backends::DynamicCircuit::from_circuit(
                        self.circuit.clone(),
                    )?;
                    self.backend
                        .expectation_value(&dynamic_circuit, &[], &self.observable)
                }
            }
            _ => {
                let dynamic_circuit =
                    crate::simulator_backends::DynamicCircuit::from_circuit(self.circuit.clone())?;
                self.backend
                    .expectation_value(&dynamic_circuit, &[], &self.observable)
            }
        }
    }
}
/// Noise model types for quantum circuits
#[derive(Debug, Clone)]
pub enum NoiseModel {
    /// Depolarizing noise
    Depolarizing {
        /// Single-qubit error rate
        single_qubit_rate: f64,
        /// Two-qubit error rate
        two_qubit_rate: f64,
    },
    /// Amplitude damping (T1 decay)
    AmplitudeDamping {
        /// Damping rate
        gamma: f64,
    },
    /// Phase damping (T2 dephasing)
    PhaseDamping {
        /// Damping rate
        gamma: f64,
    },
    /// Combined thermal relaxation
    ThermalRelaxation {
        /// T1 time (amplitude damping)
        t1: f64,
        /// T2 time (phase damping)
        t2: f64,
        /// Gate time
        gate_time: f64,
    },
    /// Readout error
    ReadoutError {
        /// Probability of 0 -> 1 error
        p0_to_1: f64,
        /// Probability of 1 -> 0 error
        p1_to_0: f64,
    },
}
/// Differentiation method for PQC layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiationMethod {
    /// Parameter shift rule (2N+1 circuit evaluations for N parameters)
    ParameterShift,
    /// Adjoint method (2 circuit evaluations, more memory efficient)
    Adjoint,
    /// Finite differences (for debugging)
    FiniteDifference,
}
/// Noisy PQC layer with noise simulation
pub struct NoisyPQCLayer {
    /// Base PQC layer
    pqc: PQCLayer,
    /// Noise model
    noise_model: NoiseModel,
    /// Number of trajectory samples for noise averaging
    num_trajectories: usize,
}
impl NoisyPQCLayer {
    /// Create new noisy PQC layer
    pub fn new(
        circuit: Circuit<8>,
        symbols: Vec<String>,
        observable: Observable,
        backend: Arc<dyn SimulatorBackend>,
        noise_model: NoiseModel,
    ) -> Self {
        let pqc = PQCLayer::new(circuit, symbols, observable, backend);
        Self {
            pqc,
            noise_model,
            num_trajectories: 100,
        }
    }
    /// Set number of trajectory samples
    pub fn with_trajectories(mut self, num_trajectories: usize) -> Self {
        self.num_trajectories = num_trajectories;
        self
    }
    /// Apply noise to expectation value
    fn apply_noise(&self, clean_expectation: f64) -> f64 {
        match &self.noise_model {
            NoiseModel::Depolarizing {
                single_qubit_rate, ..
            } => {
                let noise_factor = 1.0 - 4.0 * single_qubit_rate / 3.0;
                clean_expectation * noise_factor.max(0.0)
            }
            NoiseModel::AmplitudeDamping { gamma } => {
                let damping = (-gamma).exp();
                clean_expectation * damping
            }
            NoiseModel::PhaseDamping { gamma } => {
                let coherence = (-gamma / 2.0).exp();
                clean_expectation * coherence
            }
            NoiseModel::ThermalRelaxation { t1, t2, gate_time } => {
                let t1_factor = (-gate_time / t1).exp();
                let t2_factor = (-gate_time / t2).exp();
                clean_expectation * t1_factor.min(t2_factor)
            }
            NoiseModel::ReadoutError { p0_to_1, p1_to_0 } => {
                let fidelity = 1.0 - p0_to_1 - p1_to_0;
                clean_expectation * fidelity
            }
        }
    }
    /// Forward pass with noise
    pub fn forward(&self, inputs: &Array2<f64>, parameters: &Array2<f64>) -> Result<Array1<f64>> {
        let clean_outputs = self.pqc.forward(inputs, parameters)?;
        let noisy_outputs = clean_outputs.mapv(|x| self.apply_noise(x));
        Ok(noisy_outputs)
    }
}
/// Quantum state representation for TFQ compatibility
#[derive(Clone)]
pub struct QuantumState {
    /// State vector (complex amplitudes)
    amplitudes: Vec<scirs2_core::Complex64>,
    /// Number of qubits
    num_qubits: usize,
}
impl QuantumState {
    /// Create new quantum state from amplitudes
    pub fn new(amplitudes: Vec<scirs2_core::Complex64>) -> Result<Self> {
        let n = amplitudes.len();
        if !n.is_power_of_two() {
            return Err(MLError::InvalidConfiguration(
                "State vector length must be a power of 2".to_string(),
            ));
        }
        let num_qubits = n.trailing_zeros() as usize;
        Ok(Self {
            amplitudes,
            num_qubits,
        })
    }
    /// Create zero state |0...0>
    pub fn zero_state(num_qubits: usize) -> Self {
        let n = 1 << num_qubits;
        let mut amplitudes = vec![scirs2_core::Complex64::new(0.0, 0.0); n];
        amplitudes[0] = scirs2_core::Complex64::new(1.0, 0.0);
        Self {
            amplitudes,
            num_qubits,
        }
    }
    /// Create plus state |+...+>
    pub fn plus_state(num_qubits: usize) -> Self {
        let n = 1 << num_qubits;
        let amp = 1.0 / (n as f64).sqrt();
        let amplitudes = vec![scirs2_core::Complex64::new(amp, 0.0); n];
        Self {
            amplitudes,
            num_qubits,
        }
    }
    /// Get probability of measuring a specific outcome
    pub fn probability(&self, outcome: usize) -> f64 {
        if outcome >= self.amplitudes.len() {
            return 0.0;
        }
        (self.amplitudes[outcome].re.powi(2) + self.amplitudes[outcome].im.powi(2))
    }
    /// Get all probabilities
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .map(|a| a.re.powi(2) + a.im.powi(2))
            .collect()
    }
    /// Sample from the state
    pub fn sample(&self, num_shots: usize) -> Vec<usize> {
        let probs = self.probabilities();
        let mut samples = Vec::with_capacity(num_shots);
        for _ in 0..num_shots {
            let r = fastrand::f64();
            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    samples.push(i);
                    break;
                }
            }
        }
        samples
    }
    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}
/// Quantum metric layer for computing quantum Fisher information
pub struct QuantumMetricLayer {
    /// Circuit to analyze
    circuit: Circuit<8>,
    /// Backend for execution
    backend: Arc<dyn SimulatorBackend>,
}
impl QuantumMetricLayer {
    /// Create new quantum metric layer
    pub fn new(circuit: Circuit<8>, backend: Arc<dyn SimulatorBackend>) -> Self {
        Self { circuit, backend }
    }
    /// Compute quantum Fisher information matrix (diagonal approximation)
    pub fn compute_fisher_diagonal(&self, parameters: &[f64]) -> Result<Vec<f64>> {
        let dynamic_circuit = DynamicCircuit::from_circuit(self.circuit.clone())?;
        let n_params = parameters.len();
        let mut fisher_diag = vec![0.0; n_params];
        let epsilon = 1e-4;
        for i in 0..n_params {
            let mut params_plus = parameters.to_vec();
            params_plus[i] += epsilon;
            let result_plus = self
                .backend
                .execute_circuit(&dynamic_circuit, &params_plus, None)?;
            let mut params_minus = parameters.to_vec();
            params_minus[i] -= epsilon;
            let result_minus =
                self.backend
                    .execute_circuit(&dynamic_circuit, &params_minus, None)?;
            if let (Some(ref p_plus), Some(ref p_minus)) =
                (result_plus.probabilities, result_minus.probabilities)
            {
                let mut fii = 0.0;
                for (pp, pm) in p_plus.iter().zip(p_minus.iter()) {
                    let dp = (pp - pm) / (2.0 * epsilon);
                    let p_avg = (pp + pm) / 2.0;
                    if p_avg > 1e-10 {
                        fii += dp * dp / p_avg;
                    }
                }
                fisher_diag[i] = fii;
            }
        }
        Ok(fisher_diag)
    }
}
/// TensorFlow Quantum-style quantum dataset utilities
pub struct QuantumDataset {
    /// Circuit data
    pub(crate) circuits: Vec<DynamicCircuit>,
    /// Parameter data
    pub(crate) parameters: Array2<f64>,
    /// Labels
    pub(crate) labels: Array1<f64>,
    /// Batch size
    pub(crate) batch_size: usize,
}
impl QuantumDataset {
    /// Create new quantum dataset
    pub fn new(
        circuits: Vec<Circuit<8>>,
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
        for i in (1..n).rev() {
            let j = fastrand::usize(0..=i);
            indices.swap(i, j);
        }
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
/// Controlled PQC layer that uses input data to control the circuit
///
/// This layer allows the quantum circuit to be parameterized by both
/// trainable weights AND the input data, similar to TFQ's ControlledPQC.
pub struct ControlledPQCLayer {
    /// Base quantum circuit layer
    layer: QuantumCircuitLayer,
    /// Number of input-controlled parameters
    num_control_params: usize,
    /// Input scaling factor
    input_scaling: f64,
    /// Parameter initialization strategy
    init_strategy: ParameterInitStrategy,
}
impl ControlledPQCLayer {
    /// Create new controlled PQC layer
    pub fn new(
        circuit: Circuit<8>,
        symbols: Vec<String>,
        observable: Observable,
        backend: Arc<dyn SimulatorBackend>,
        num_control_params: usize,
    ) -> Self {
        let layer = QuantumCircuitLayer::new(circuit, symbols, observable, backend);
        Self {
            layer,
            num_control_params,
            input_scaling: 1.0,
            init_strategy: ParameterInitStrategy::RandomNormal {
                mean: 0.0,
                std: 0.1,
            },
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
    /// Forward pass with input control
    pub fn forward(
        &self,
        control_inputs: &Array2<f64>,
        trainable_params: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let batch_size = control_inputs.nrows();
        let mut outputs = Array1::zeros(batch_size);
        for batch_idx in 0..batch_size {
            let control = control_inputs
                .row(batch_idx)
                .mapv(|x| x * self.input_scaling);
            let trainable = trainable_params.row(batch_idx % trainable_params.nrows());
            let combined_params: Vec<f64> =
                control.iter().chain(trainable.iter()).copied().collect();
            let dynamic_circuit = crate::simulator_backends::DynamicCircuit::from_circuit(
                self.layer.circuit.clone(),
            )?;
            let expectation = self.layer.backend.expectation_value(
                &dynamic_circuit,
                &combined_params,
                &self.layer.observable,
            )?;
            outputs[batch_idx] = expectation;
        }
        Ok(outputs)
    }
}
/// Iterator for quantum dataset batches
pub struct QuantumDatasetIterator<'a> {
    pub(super) dataset: &'a QuantumDataset,
    pub(super) current_batch: usize,
    pub(super) total_batches: usize,
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
/// SPSA (Simultaneous Perturbation Stochastic Approximation) differentiator
#[derive(Debug, Clone)]
pub struct SPSADifferentiator {
    /// Perturbation size
    pub(super) epsilon: f64,
    /// Number of samples for averaging
    pub(super) num_samples: usize,
}
impl SPSADifferentiator {
    /// Create new SPSA differentiator
    pub fn new() -> Self {
        Self {
            epsilon: 0.1,
            num_samples: 10,
        }
    }
    /// Create with custom parameters
    pub fn with_params(epsilon: f64, num_samples: usize) -> Self {
        Self {
            epsilon,
            num_samples,
        }
    }
}
/// Quantum natural gradient optimizer
pub struct QuantumNaturalGradient {
    /// Learning rate
    learning_rate: f64,
    /// Regularization for inverting Fisher matrix
    regularization: f64,
}
impl QuantumNaturalGradient {
    /// Create new quantum natural gradient optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            regularization: 1e-4,
        }
    }
    /// Set regularization
    pub fn with_regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }
    /// Apply natural gradient update
    pub fn step(&self, parameters: &mut [f64], gradients: &[f64], fisher_diag: &[f64]) {
        for i in 0..parameters.len() {
            let fii = fisher_diag[i] + self.regularization;
            parameters[i] -= self.learning_rate * gradients[i] / fii;
        }
    }
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
    /// Glorot/Xavier uniform initialization
    /// Draws samples from uniform distribution U(-limit, limit)
    /// where limit = sqrt(6 / (fan_in + fan_out))
    GlorotUniform,
    /// Glorot/Xavier normal initialization
    /// Draws samples from normal distribution N(0, std)
    /// where std = sqrt(2 / (fan_in + fan_out))
    GlorotNormal,
}
/// AddCircuit layer - appends one circuit to another
/// Similar to tfq.layers.AddCircuit
pub struct AddCircuitLayer {
    /// Circuit to append
    append_circuit: Circuit<8>,
}
impl AddCircuitLayer {
    /// Create new AddCircuit layer
    pub fn new(append_circuit: Circuit<8>) -> Self {
        Self { append_circuit }
    }
    /// Get the circuit to append
    pub fn append_circuit(&self) -> &Circuit<8> {
        &self.append_circuit
    }
    /// Forward pass - returns circuits that should be executed after input circuits
    ///
    /// Note: In TFQ, this actually concatenates circuits. Here we return the append circuit
    /// as a DynamicCircuit since circuit concatenation happens at execution time.
    pub fn forward(
        &self,
        input_circuits: &[DynamicCircuit],
    ) -> Result<Vec<(DynamicCircuit, DynamicCircuit)>> {
        let append_dynamic = DynamicCircuit::from_circuit(self.append_circuit.clone())?;
        input_circuits
            .iter()
            .map(|input| Ok((input.clone(), append_dynamic.clone())))
            .collect()
    }
    /// Forward pass - appends gates from append_circuit to each input circuit
    ///
    /// This creates new circuits by combining the gates from both circuits.
    pub fn forward_combined(&self, input_circuits: &[Circuit<8>]) -> Result<Vec<Circuit<8>>> {
        input_circuits
            .iter()
            .map(|input| {
                let mut builder = CircuitBuilder::new();
                for gate in input.gates() {
                    builder.add_gate_arc(gate.clone()).map_err(|e| {
                        MLError::InvalidConfiguration(format!("Failed to add gate: {}", e))
                    })?;
                }
                for gate in self.append_circuit.gates() {
                    builder.add_gate_arc(gate.clone()).map_err(|e| {
                        MLError::InvalidConfiguration(format!("Failed to add gate: {}", e))
                    })?;
                }
                Ok(builder.build())
            })
            .collect()
    }
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
impl QuantumConvolutionalLayer {
    /// Create new quantum convolutional layer
    pub fn new(
        circuit: Circuit<8>,
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
        let output_h = (height - filter_h) / stride_h + 1;
        let output_w = (width - filter_w) / stride_w + 1;
        let mut outputs = Array4::zeros((batch_size, output_h, output_w, 1));
        for batch in 0..batch_size {
            for out_y in 0..output_h {
                for out_x in 0..output_w {
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
                    let result = self.pqc.forward(&patch_data, parameters)?;
                    outputs[[batch, out_y, out_x, 0]] = result[0];
                }
            }
        }
        Ok(outputs)
    }
}
