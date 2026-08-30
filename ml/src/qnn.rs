//! Quantum Neural Networks (QNNs) with parameterised quantum circuits.
//!
//! [`QuantumNeuralNetwork`] wraps a parameterised quantum circuit as a
//! differentiable layer, supporting forward passes, parameter-shift gradient
//! computation, and stochastic gradient descent-based training.

use crate::error::{MLError, Result};
use crate::optimization::Optimizer;
use quantrs2_circuit::builder::Simulator;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::FRAC_PI_2;
use std::fmt;

/// Maximum number of qubits the built-in state-vector forward pass will
/// simulate.  Above this size a dense state vector becomes impractical and the
/// forward pass returns an honest [`MLError::NotSupported`] instead of
/// fabricating an answer.
const MAX_FORWARD_QUBITS: usize = 16;

/// When the number of trainable parameters does not exceed this bound, training
/// uses exact parameter-shift-rule gradients; larger circuits fall back to the
/// SPSA stochastic gradient estimator to keep the number of circuit evaluations
/// tractable.
const PARAMETER_SHIFT_MAX_PARAMS: usize = 64;

/// Compute the expectation value `<ψ|P_q|ψ>` of a single-qubit Pauli operator
/// `P ∈ {X, Y, Z}` acting on qubit `qubit` for a state vector `amplitudes`.
///
/// The state vector has `2^n` entries indexed so that bit `qubit` (LSB = qubit
/// 0) selects the computational basis state of that qubit.  The result is a real
/// number in `[-1, 1]`.
fn single_qubit_pauli_expectation(
    amplitudes: &[Complex64],
    pauli: char,
    qubit: usize,
) -> Result<f64> {
    let dim = amplitudes.len();
    if dim == 0 || dim & (dim - 1) != 0 {
        return Err(MLError::ComputationError(format!(
            "state-vector dimension {dim} is not a positive power of two"
        )));
    }
    let n = dim.trailing_zeros() as usize;
    if qubit >= n {
        return Err(MLError::ComputationError(format!(
            "qubit index {qubit} out of range for {n}-qubit state"
        )));
    }

    let bit = 1usize << qubit;
    let value = match pauli {
        'Z' => {
            let mut expectation = 0.0_f64;
            for (j, amp) in amplitudes.iter().enumerate() {
                let prob = amp.norm_sqr();
                if j & bit == 0 {
                    expectation += prob;
                } else {
                    expectation -= prob;
                }
            }
            expectation
        }
        'X' => {
            // <X_q> = 2 Re[ Σ_{j: bit q = 0} conj(ψ_j) · ψ_{j ⊕ bit} ]
            let mut sum = Complex64::new(0.0, 0.0);
            for (j, amp) in amplitudes.iter().enumerate() {
                if j & bit == 0 {
                    sum += amp.conj() * amplitudes[j ^ bit];
                }
            }
            2.0 * sum.re
        }
        'Y' => {
            // <Y_q> = 2 Im[ Σ_{j: bit q = 0} conj(ψ_j) · ψ_{j ⊕ bit} ]
            let mut sum = Complex64::new(0.0, 0.0);
            for (j, amp) in amplitudes.iter().enumerate() {
                if j & bit == 0 {
                    sum += amp.conj() * amplitudes[j ^ bit];
                }
            }
            2.0 * sum.im
        }
        other => {
            return Err(MLError::ComputationError(format!(
                "unsupported Pauli operator '{other}'"
            )))
        }
    };
    Ok(value)
}

/// Activation function types for quantum layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    /// Linear activation (identity)
    Linear,
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
}

/// Represents a layer type in a quantum neural network
#[derive(Debug, Clone)]
pub enum QNNLayerType {
    /// Encoding layer for converting classical data to quantum states
    EncodingLayer {
        /// Number of classical features to encode
        num_features: usize,
    },

    /// Variational layer with trainable parameters
    VariationalLayer {
        /// Number of trainable parameters
        num_params: usize,
    },

    /// Entanglement layer to create entanglement between qubits
    EntanglementLayer {
        /// Connectivity pattern, e.g., "full", "linear", "circular"
        connectivity: String,
    },

    /// Measurement layer to extract classical information
    MeasurementLayer {
        /// Measurement basis, e.g., "computational", "Pauli-X", "Pauli-Y", "Pauli-Z"
        measurement_basis: String,
    },
}

/// Results from training a quantum neural network
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final loss value after training
    pub final_loss: f64,

    /// Training accuracy (for classification tasks)
    pub accuracy: f64,

    /// Loss history during training
    pub loss_history: Vec<f64>,

    /// Optimal parameters found during training
    pub optimal_parameters: Array1<f64>,
}

/// Represents a quantum neural network.
///
/// A QNN consists of an ordered sequence of [`QNNLayerType`] layers that map
/// classical input vectors to output predictions via a parameterised quantum
/// circuit evaluated on a state-vector simulator.
///
/// # Examples
///
/// ```rust
/// use quantrs2_ml::qnn::{QuantumNeuralNetwork, QNNLayerType};
///
/// let layers = vec![
///     QNNLayerType::EncodingLayer { num_features: 2 },
///     QNNLayerType::VariationalLayer { num_params: 4 },
/// ];
/// let qnn = QuantumNeuralNetwork::new(layers, 2, 2, 1)
///     .expect("failed to create QNN");
/// assert_eq!(qnn.num_qubits, 2);
/// ```
#[derive(Debug, Clone)]
pub struct QuantumNeuralNetwork {
    /// The layers that make up the network
    pub layers: Vec<QNNLayerType>,

    /// The number of qubits used in the network
    pub num_qubits: usize,

    /// The dimension of the input data
    pub input_dim: usize,

    /// The dimension of the output data
    pub output_dim: usize,

    /// Network parameters (weights)
    pub parameters: Array1<f64>,
}

impl QuantumNeuralNetwork {
    /// Creates a new quantum neural network
    pub fn new(
        layers: Vec<QNNLayerType>,
        num_qubits: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        // Validate the layers and structure
        if layers.is_empty() {
            return Err(MLError::ModelCreationError(
                "QNN must have at least one layer".to_string(),
            ));
        }

        // Determine parameter count from variational layers
        let num_params = layers
            .iter()
            .filter_map(|layer| match layer {
                QNNLayerType::VariationalLayer { num_params } => Some(num_params),
                _ => None,
            })
            .sum::<usize>();

        // Create random initial parameters
        let parameters = Array1::from_vec(
            (0..num_params)
                .map(|_| thread_rng().random::<f64>() * 2.0 * std::f64::consts::PI)
                .collect(),
        );

        Ok(QuantumNeuralNetwork {
            layers,
            num_qubits,
            input_dim,
            output_dim,
            parameters,
        })
    }

    /// Append the parameterised gates described by [`Self::layers`] onto a
    /// state-vector circuit of register size `N`.
    ///
    /// * `EncodingLayer`   — angle-encodes the classical `input` features onto
    ///   `RY` rotations (data re-uploading friendly).
    /// * `VariationalLayer`— applies the trainable `parameters` as alternating
    ///   `RY`/`RZ` sweeps across the active qubits (every qubit receives an `RY`
    ///   before any qubit receives an `RZ`).
    /// * `EntanglementLayer`— applies a `CNOT` pattern (`linear`, `circular`,
    ///   or `full`) across the active qubits.
    /// * `MeasurementLayer`— readout is handled separately in [`Self::readout`].
    fn append_layers<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        input: &Array1<f64>,
        parameters: &Array1<f64>,
    ) -> Result<()> {
        let num_qubits = self.num_qubits.min(N);
        if num_qubits == 0 {
            return Err(MLError::ModelCreationError(
                "QNN requires at least one qubit".to_string(),
            ));
        }

        let mut param_idx = 0usize;
        for layer in &self.layers {
            match layer {
                QNNLayerType::EncodingLayer { num_features } => {
                    let count = (*num_features).min(input.len());
                    for feature in 0..count {
                        let qubit = feature % num_qubits;
                        circuit.ry(qubit, input[feature])?;
                    }
                }
                QNNLayerType::VariationalLayer { num_params } => {
                    // Hardware-efficient ansatz: sweep every qubit with an `RY`
                    // rotation, then every qubit with an `RZ` rotation, and keep
                    // alternating for as many parameters as the layer declares.
                    //
                    // The axis must be derived from the sweep index rather than
                    // from `local` itself: with an even qubit count `local % 2` is
                    // fully determined by the qubit parity, so every odd qubit
                    // would receive `RZ` rotations only.  Those commute with the
                    // Pauli-`Z` readout, leaving the odd qubits' expectation values
                    // without a single trainable parameter of their own.
                    for local in 0..*num_params {
                        if param_idx >= parameters.len() {
                            break;
                        }
                        let qubit = local % num_qubits;
                        let sweep = local / num_qubits;
                        if sweep % 2 == 0 {
                            circuit.ry(qubit, parameters[param_idx])?;
                        } else {
                            circuit.rz(qubit, parameters[param_idx])?;
                        }
                        param_idx += 1;
                    }
                }
                QNNLayerType::EntanglementLayer { connectivity } => {
                    if num_qubits > 1 {
                        match connectivity.as_str() {
                            "linear" => {
                                for q in 0..num_qubits - 1 {
                                    circuit.cnot(q, q + 1)?;
                                }
                            }
                            "circular" => {
                                for q in 0..num_qubits {
                                    circuit.cnot(q, (q + 1) % num_qubits)?;
                                }
                            }
                            // "full" (default): all-to-all nearest entanglement
                            _ => {
                                for a in 0..num_qubits {
                                    for b in (a + 1)..num_qubits {
                                        circuit.cnot(a, b)?;
                                    }
                                }
                            }
                        }
                    }
                }
                QNNLayerType::MeasurementLayer { .. } => {
                    // Measurement is performed as an expectation-value readout in
                    // `readout`; no gates are appended here.
                }
            }
        }
        Ok(())
    }

    /// Convert a simulated state vector into an `output_dim`-length vector of
    /// single-qubit Pauli expectation values.
    ///
    /// Output `k` reads qubit `k mod num_qubits` in Pauli basis `Z`, `X`, then
    /// `Y` (cycling as `k` grows), giving up to `3 · num_qubits` distinct
    /// observables.  Each value lies in `[-1, 1]`.
    fn readout(&self, amplitudes: &[Complex64]) -> Result<Array1<f64>> {
        let num_qubits = self.num_qubits;
        if num_qubits == 0 {
            return Err(MLError::ModelCreationError(
                "QNN requires at least one qubit".to_string(),
            ));
        }
        let mut output = Array1::zeros(self.output_dim);
        for k in 0..self.output_dim {
            let qubit = k % num_qubits;
            let pauli = match (k / num_qubits) % 3 {
                0 => 'Z',
                1 => 'X',
                _ => 'Y',
            };
            output[k] = single_qubit_pauli_expectation(amplitudes, pauli, qubit)?;
        }
        Ok(output)
    }

    /// Build the parameterised circuit on an `N`-qubit register, simulate it on
    /// the real state-vector backend, and read out the expectation-value output.
    fn run_sized<const N: usize>(
        &self,
        input: &Array1<f64>,
        parameters: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let mut circuit = Circuit::<N>::new();
        self.append_layers::<N>(&mut circuit, input, parameters)?;
        let simulator = StateVectorSimulator::new();
        let register = simulator.run(&circuit)?;
        self.readout(register.amplitudes())
    }

    /// Simulate the network for `input`/`parameters`, dispatching to the
    /// smallest supported register that can hold `num_qubits`, and return the
    /// expectation-value output.
    fn measure_outputs(
        &self,
        input: &Array1<f64>,
        parameters: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        match self.num_qubits {
            0 => Err(MLError::ModelCreationError(
                "QNN requires at least one qubit".to_string(),
            )),
            1..=2 => self.run_sized::<2>(input, parameters),
            3..=4 => self.run_sized::<4>(input, parameters),
            5..=8 => self.run_sized::<8>(input, parameters),
            9..=16 => self.run_sized::<16>(input, parameters),
            n => Err(MLError::NotSupported(format!(
                "QNN forward pass supports at most {MAX_FORWARD_QUBITS} qubits on the \
                 state-vector backend, got {n}"
            ))),
        }
    }

    /// Runs the network on a given input, returning the measured expectation
    /// values (one per output dimension, each in `[-1, 1]`).
    pub fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        self.measure_outputs(input, &self.parameters)
    }

    /// Parameter-shift gradient of a single output component with respect to
    /// every trainable parameter, evaluated at the current parameters.
    ///
    /// For the Pauli-rotation gates used by [`Self::append_layers`] the exact
    /// gradient of an expectation value is
    /// `(⟨O⟩(θ+π/2) − ⟨O⟩(θ−π/2)) / 2`.
    pub fn output_component_gradient(
        &self,
        input: &Array1<f64>,
        output_index: usize,
    ) -> Result<Array1<f64>> {
        if output_index >= self.output_dim {
            return Err(MLError::InvalidParameter(format!(
                "output index {output_index} out of range for output dimension {}",
                self.output_dim
            )));
        }
        let num_params = self.parameters.len();
        let mut gradient = Array1::zeros(num_params);
        let mut params = self.parameters.clone();
        for j in 0..num_params {
            let original = params[j];
            params[j] = original + FRAC_PI_2;
            let plus = self.measure_outputs(input, &params)?[output_index];
            params[j] = original - FRAC_PI_2;
            let minus = self.measure_outputs(input, &params)?[output_index];
            params[j] = original;
            gradient[j] = (plus - minus) / 2.0;
        }
        Ok(gradient)
    }

    /// Mean-squared-error loss of the network over `(x, y)` using an explicit
    /// parameter vector (used by the training routines below).
    fn loss_with_parameters(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        parameters: &Array1<f64>,
    ) -> Result<f64> {
        let n = x.nrows();
        if n == 0 {
            return Err(MLError::DataError("dataset is empty".to_string()));
        }
        let mut total = 0.0;
        for i in 0..n {
            let out = self.measure_outputs(&x.row(i).to_owned(), parameters)?;
            let cols = out.len().min(y.ncols());
            for k in 0..cols {
                let diff = out[k] - y[[i, k]];
                total += diff * diff;
            }
        }
        Ok(total / n as f64)
    }

    /// Exact batch parameter-shift gradient of the MSE loss with respect to
    /// every trainable parameter.
    fn parameter_shift_gradient(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let num_params = self.parameters.len();
        let ncols = y.ncols();

        // Base outputs (unshifted) used to form the MSE residuals.
        let mut base_outputs = Vec::with_capacity(n);
        for i in 0..n {
            base_outputs.push(self.forward(&x.row(i).to_owned())?);
        }

        let mut gradient = Array1::zeros(num_params);
        let mut params = self.parameters.clone();
        for j in 0..num_params {
            let original = params[j];
            let mut accum = 0.0;
            for i in 0..n {
                let xi = x.row(i).to_owned();
                params[j] = original + FRAC_PI_2;
                let out_plus = self.measure_outputs(&xi, &params)?;
                params[j] = original - FRAC_PI_2;
                let out_minus = self.measure_outputs(&xi, &params)?;
                let cols = base_outputs[i].len().min(ncols);
                for k in 0..cols {
                    let residual = base_outputs[i][k] - y[[i, k]];
                    let d_output = (out_plus[k] - out_minus[k]) / 2.0;
                    accum += 2.0 * residual * d_output;
                }
            }
            params[j] = original;
            gradient[j] = accum / n as f64;
        }
        Ok(gradient)
    }

    /// SPSA (simultaneous perturbation) stochastic estimate of the MSE-loss
    /// gradient — used when the parameter count makes exact parameter-shift
    /// gradients too expensive.
    fn spsa_gradient(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<Array1<f64>> {
        let num_params = self.parameters.len();
        let perturbation = 0.1_f64;

        let mut delta = vec![0.0_f64; num_params];
        for value in delta.iter_mut() {
            *value = if thread_rng().random::<f64>() < 0.5 {
                -1.0
            } else {
                1.0
            };
        }

        let mut params_plus = self.parameters.clone();
        let mut params_minus = self.parameters.clone();
        for j in 0..num_params {
            params_plus[j] += perturbation * delta[j];
            params_minus[j] -= perturbation * delta[j];
        }

        let loss_plus = self.loss_with_parameters(x, y, &params_plus)?;
        let loss_minus = self.loss_with_parameters(x, y, &params_minus)?;

        let mut gradient = Array1::zeros(num_params);
        for j in 0..num_params {
            gradient[j] = (loss_plus - loss_minus) / (2.0 * perturbation * delta[j]);
        }
        Ok(gradient)
    }

    /// Fraction of training samples the network classifies correctly.
    ///
    /// For single-output networks a sample counts as correct when the predicted
    /// value is within `0.5` of the target; for multi-output networks the
    /// `argmax` of the prediction must match the `argmax` of the target row.
    fn training_accuracy(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<f64> {
        let n = x.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        let ncols = y.ncols();
        let mut correct = 0usize;
        for i in 0..n {
            let out = self.forward(&x.row(i).to_owned())?;
            if out.len() == 1 || ncols == 1 {
                if (out[0] - y[[i, 0]]).abs() < 0.5 {
                    correct += 1;
                }
            } else {
                let cols = out.len().min(ncols);
                let mut pred_idx = 0usize;
                let mut true_idx = 0usize;
                for k in 1..cols {
                    if out[k] > out[pred_idx] {
                        pred_idx = k;
                    }
                    if y[[i, k]] > y[[i, true_idx]] {
                        true_idx = k;
                    }
                }
                if pred_idx == true_idx {
                    correct += 1;
                }
            }
        }
        Ok(correct as f64 / n as f64)
    }

    /// Trains the network on a dataset by gradient descent on the MSE loss.
    ///
    /// Gradients are computed with the exact parameter-shift rule for small
    /// circuits and with the SPSA estimator otherwise.  Parameters are updated
    /// in place and the real per-epoch loss trajectory is returned.
    pub fn train(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<TrainingResult> {
        let n = x_train.nrows();
        if n == 0 {
            return Err(MLError::DataError("training set is empty".to_string()));
        }
        if y_train.nrows() != n {
            return Err(MLError::DimensionMismatch(format!(
                "x_train has {n} rows but y_train has {}",
                y_train.nrows()
            )));
        }

        let use_parameter_shift = self.parameters.len() <= PARAMETER_SHIFT_MAX_PARAMS
            && self.num_qubits <= MAX_FORWARD_QUBITS;

        let mut loss_history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let gradient = if use_parameter_shift {
                self.parameter_shift_gradient(x_train, y_train)?
            } else {
                self.spsa_gradient(x_train, y_train)?
            };
            for j in 0..self.parameters.len() {
                self.parameters[j] -= learning_rate * gradient[j];
            }
            loss_history.push(self.loss_with_parameters(x_train, y_train, &self.parameters)?);
        }

        let final_loss = match loss_history.last() {
            Some(&loss) => loss,
            None => self.loss_with_parameters(x_train, y_train, &self.parameters)?,
        };
        let accuracy = self.training_accuracy(x_train, y_train)?;

        Ok(TrainingResult {
            final_loss,
            accuracy,
            loss_history,
            optimal_parameters: self.parameters.clone(),
        })
    }

    /// Trains the network on a dataset with 1D labels (compatibility method)
    pub fn train_1d(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<TrainingResult> {
        // Convert 1D labels to 2D
        let y_2d = y_train.clone().into_shape((y_train.len(), 1))?;
        self.train(x_train, &y_2d, epochs, learning_rate)
    }

    /// Predicts the output for a given input
    pub fn predict(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        self.forward(input)
    }

    /// Predicts the output for a batch of inputs
    pub fn predict_batch(&self, inputs: &Array2<f64>) -> Result<Array2<f64>> {
        let batch_size = inputs.nrows();
        let mut outputs = Array2::zeros((batch_size, self.output_dim));

        for (i, row) in inputs.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            let input = row.to_owned();
            let output = self.predict(&input)?;
            outputs.row_mut(i).assign(&output);
        }

        Ok(outputs)
    }
}

/// Builder for quantum neural networks
///
/// Provides a fluent API to construct a [`QuantumNeuralNetwork`] by adding
/// encoding, variational, entanglement, and measurement layers.
///
/// # Examples
///
/// ```rust
/// use quantrs2_ml::qnn::QNNBuilder;
///
/// let qnn = QNNBuilder::new()
///     .with_qubits(2)
///     .with_input_dim(2)
///     .with_output_dim(1)
///     .add_encoding_layer(2)
///     .add_variational_layer(4)
///     .build()
///     .expect("valid QNN configuration");
/// assert_eq!(qnn.num_qubits, 2);
/// ```
#[derive(Debug, Clone)]
pub struct QNNBuilder {
    layers: Vec<QNNLayerType>,
    num_qubits: usize,
    input_dim: usize,
    output_dim: usize,
}

impl QNNBuilder {
    /// Creates a new QNN builder
    pub fn new() -> Self {
        QNNBuilder {
            layers: Vec::new(),
            num_qubits: 0,
            input_dim: 0,
            output_dim: 0,
        }
    }

    /// Sets the number of qubits
    pub fn with_qubits(mut self, num_qubits: usize) -> Self {
        self.num_qubits = num_qubits;
        self
    }

    /// Sets the input dimension
    pub fn with_input_dim(mut self, input_dim: usize) -> Self {
        self.input_dim = input_dim;
        self
    }

    /// Sets the output dimension
    pub fn with_output_dim(mut self, output_dim: usize) -> Self {
        self.output_dim = output_dim;
        self
    }

    /// Adds an encoding layer
    pub fn add_encoding_layer(mut self, num_features: usize) -> Self {
        self.layers
            .push(QNNLayerType::EncodingLayer { num_features });
        self
    }

    /// Adds a layer (alias for add_encoding_layer for compatibility)
    pub fn add_layer(self, size: usize) -> Self {
        self.add_encoding_layer(size)
    }

    /// Adds a variational layer
    pub fn add_variational_layer(mut self, num_params: usize) -> Self {
        self.layers
            .push(QNNLayerType::VariationalLayer { num_params });
        self
    }

    /// Adds an entanglement layer
    pub fn add_entanglement_layer(mut self, connectivity: &str) -> Self {
        self.layers.push(QNNLayerType::EntanglementLayer {
            connectivity: connectivity.to_string(),
        });
        self
    }

    /// Adds a measurement layer
    pub fn add_measurement_layer(mut self, measurement_basis: &str) -> Self {
        self.layers.push(QNNLayerType::MeasurementLayer {
            measurement_basis: measurement_basis.to_string(),
        });
        self
    }

    /// Builds the quantum neural network
    pub fn build(self) -> Result<QuantumNeuralNetwork> {
        if self.num_qubits == 0 {
            return Err(MLError::ModelCreationError(
                "Number of qubits must be greater than 0".to_string(),
            ));
        }

        if self.input_dim == 0 {
            return Err(MLError::ModelCreationError(
                "Input dimension must be greater than 0".to_string(),
            ));
        }

        if self.output_dim == 0 {
            return Err(MLError::ModelCreationError(
                "Output dimension must be greater than 0".to_string(),
            ));
        }

        if self.layers.is_empty() {
            return Err(MLError::ModelCreationError(
                "QNN must have at least one layer".to_string(),
            ));
        }

        QuantumNeuralNetwork::new(
            self.layers,
            self.num_qubits,
            self.input_dim,
            self.output_dim,
        )
    }
}

impl fmt::Display for QNNLayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QNNLayerType::EncodingLayer { num_features } => {
                write!(f, "Encoding Layer (features: {})", num_features)
            }
            QNNLayerType::VariationalLayer { num_params } => {
                write!(f, "Variational Layer (parameters: {})", num_params)
            }
            QNNLayerType::EntanglementLayer { connectivity } => {
                write!(f, "Entanglement Layer (connectivity: {})", connectivity)
            }
            QNNLayerType::MeasurementLayer { measurement_basis } => {
                write!(f, "Measurement Layer (basis: {})", measurement_basis)
            }
        }
    }
}

/// Quantum neural network layer for use in other modules
///
/// A single dense-like layer in a hybrid quantum-classical network, mapping
/// `input_dim` features to `output_dim` features through a chosen activation.
///
/// # Examples
///
/// ```rust
/// use quantrs2_ml::qnn::{QNNLayer, ActivationType};
///
/// let layer = QNNLayer::new(4, 2, ActivationType::ReLU);
/// assert_eq!(layer.input_dim, 4);
/// assert_eq!(layer.output_dim, 2);
/// ```
#[derive(Debug, Clone)]
pub struct QNNLayer {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationType,
}

impl QNNLayer {
    /// Create a new QNN layer
    pub fn new(input_dim: usize, output_dim: usize, activation: ActivationType) -> Self {
        Self {
            input_dim,
            output_dim,
            activation,
        }
    }
}
