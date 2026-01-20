//! Quantum Machine Learning Framework
//!
//! Main framework implementation for QML training and inference.

use super::config::*;
use super::layers::*;
use super::types::*;
use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Main quantum machine learning layers framework
#[derive(Debug)]
pub struct QuantumMLFramework {
    /// Configuration
    config: QMLConfig,
    /// QML layers
    pub layers: Vec<Box<dyn QMLLayer>>,
    /// Current training state
    training_state: QMLTrainingState,
    /// `SciRS2` backend for numerical operations
    backend: Option<SciRS2Backend>,
    /// Performance statistics
    stats: QMLStats,
    /// Training history
    training_history: Vec<QMLTrainingResult>,
}

impl QuantumMLFramework {
    /// Create new quantum ML framework
    pub fn new(config: QMLConfig) -> Result<Self> {
        let mut framework = Self {
            config,
            layers: Vec::new(),
            training_state: QMLTrainingState::new(),
            backend: None,
            stats: QMLStats::new(),
            training_history: Vec::new(),
        };

        framework.initialize_layers()?;

        let backend = SciRS2Backend::new();
        if backend.is_available() {
            framework.backend = Some(backend);
        }

        Ok(framework)
    }

    fn initialize_layers(&mut self) -> Result<()> {
        for layer_config in &self.config.layer_configs {
            let layer = self.create_layer(layer_config)?;
            self.layers.push(layer);
        }
        Ok(())
    }

    fn create_layer(&self, config: &QMLLayerConfig) -> Result<Box<dyn QMLLayer>> {
        match config.layer_type {
            QMLLayerType::ParameterizedQuantumCircuit => Ok(Box::new(
                ParameterizedQuantumCircuitLayer::new(self.config.num_qubits, config.clone())?,
            )),
            QMLLayerType::QuantumConvolutional => Ok(Box::new(QuantumConvolutionalLayer::new(
                self.config.num_qubits,
                config.clone(),
            )?)),
            QMLLayerType::QuantumDense => Ok(Box::new(QuantumDenseLayer::new(
                self.config.num_qubits,
                config.clone(),
            )?)),
            QMLLayerType::QuantumLSTM => Ok(Box::new(QuantumLSTMLayer::new(
                self.config.num_qubits,
                config.clone(),
            )?)),
            QMLLayerType::QuantumAttention => Ok(Box::new(QuantumAttentionLayer::new(
                self.config.num_qubits,
                config.clone(),
            )?)),
            _ => Err(SimulatorError::InvalidConfiguration(format!(
                "Layer type {:?} not yet implemented",
                config.layer_type
            ))),
        }
    }

    /// Forward pass through the quantum ML model
    pub fn forward(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let mut current_state = self.encode_input(input)?;

        for layer in &mut self.layers {
            current_state = layer.forward(&current_state)?;
        }

        let output = self.decode_output(&current_state)?;
        self.stats.forward_passes += 1;

        Ok(output)
    }

    /// Backward pass for gradient computation
    pub fn backward(&mut self, loss_gradient: &Array1<f64>) -> Result<Array1<f64>> {
        let mut grad = loss_gradient.clone();

        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }

        self.stats.backward_passes += 1;
        Ok(grad)
    }

    /// Train the quantum ML model
    pub fn train(
        &mut self,
        training_data: &[(Array1<f64>, Array1<f64>)],
        validation_data: Option<&[(Array1<f64>, Array1<f64>)]>,
    ) -> Result<QMLTrainingResult> {
        let mut best_validation_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let mut training_metrics = Vec::new();

        let training_start = std::time::Instant::now();

        for epoch in 0..self.config.training_config.epochs {
            let epoch_start = std::time::Instant::now();
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            for batch in training_data.chunks(self.config.training_config.batch_size) {
                let batch_loss = self.train_batch(batch)?;
                epoch_loss += batch_loss;
                num_batches += 1;
            }

            epoch_loss /= f64::from(num_batches);

            let validation_loss = if let Some(val_data) = validation_data {
                self.evaluate(val_data)?
            } else {
                epoch_loss
            };

            let epoch_time = epoch_start.elapsed();

            let metrics = QMLEpochMetrics {
                epoch,
                training_loss: epoch_loss,
                validation_loss,
                epoch_time,
                learning_rate: self.get_current_learning_rate(epoch),
            };

            training_metrics.push(metrics.clone());

            if self.config.training_config.early_stopping.enabled {
                if validation_loss
                    < best_validation_loss - self.config.training_config.early_stopping.min_delta
                {
                    best_validation_loss = validation_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.training_config.early_stopping.patience {
                        break;
                    }
                }
            }

            self.update_learning_rate(epoch, validation_loss);
        }

        let total_training_time = training_start.elapsed();

        let result = QMLTrainingResult {
            final_training_loss: training_metrics.last().map_or(0.0, |m| m.training_loss),
            final_validation_loss: training_metrics.last().map_or(0.0, |m| m.validation_loss),
            best_validation_loss,
            epochs_trained: training_metrics.len(),
            total_training_time,
            training_metrics,
            quantum_advantage_metrics: self.compute_quantum_advantage_metrics()?,
        };

        self.training_history.push(result.clone());
        Ok(result)
    }

    fn train_batch(&mut self, batch: &[(Array1<f64>, Array1<f64>)]) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut total_gradients: Vec<Array1<f64>> =
            (0..self.layers.len()).map(|_| Array1::zeros(0)).collect();

        for (input, target) in batch {
            let prediction = self.forward(input)?;
            let loss = Self::compute_loss(&prediction, target)?;
            total_loss += loss;

            let loss_gradient = Self::compute_loss_gradient(&prediction, target)?;
            let gradients = self.compute_gradients(&loss_gradient)?;

            for (i, grad) in gradients.iter().enumerate() {
                if total_gradients[i].is_empty() {
                    total_gradients[i] = grad.clone();
                } else {
                    total_gradients[i] += grad;
                }
            }
        }

        let batch_size = batch.len() as f64;
        for grad in &mut total_gradients {
            *grad /= batch_size;
        }

        self.apply_gradients(&total_gradients)?;
        Ok(total_loss / batch_size)
    }

    /// Evaluate the model on validation data
    pub fn evaluate(&mut self, data: &[(Array1<f64>, Array1<f64>)]) -> Result<f64> {
        let mut total_loss = 0.0;

        for (input, target) in data {
            let prediction = self.forward(input)?;
            let loss = Self::compute_loss(&prediction, target)?;
            total_loss += loss;
        }

        Ok(total_loss / data.len() as f64)
    }

    fn encode_input(&self, input: &Array1<f64>) -> Result<Array1<Complex64>> {
        match self.config.classical_preprocessing.encoding_method {
            DataEncodingMethod::Amplitude => self.encode_amplitude(input),
            DataEncodingMethod::Angle => self.encode_angle(input),
            DataEncodingMethod::Basis => self.encode_basis(input),
            DataEncodingMethod::QuantumFeatureMap => self.encode_quantum_feature_map(input),
            _ => Err(SimulatorError::InvalidConfiguration(
                "Encoding method not implemented".to_string(),
            )),
        }
    }

    fn encode_amplitude(&self, input: &Array1<f64>) -> Result<Array1<Complex64>> {
        let n_qubits = self.config.num_qubits;
        let state_size = 1 << n_qubits;
        let mut state = Array1::zeros(state_size);

        let norm = input.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(SimulatorError::InvalidState("Zero input norm".to_string()));
        }

        for (i, &val) in input.iter().enumerate() {
            if i < state_size {
                state[i] = Complex64::new(val / norm, 0.0);
            }
        }

        Ok(state)
    }

    fn encode_angle(&self, input: &Array1<f64>) -> Result<Array1<Complex64>> {
        let n_qubits = self.config.num_qubits;
        let state_size = 1 << n_qubits;
        let mut state = Array1::zeros(state_size);

        state[0] = Complex64::new(1.0, 0.0);

        for (i, &angle) in input.iter().enumerate() {
            if i < n_qubits {
                state = self.apply_ry_rotation(&state, i, angle)?;
            }
        }

        Ok(state)
    }

    fn encode_basis(&self, input: &Array1<f64>) -> Result<Array1<Complex64>> {
        let n_qubits = self.config.num_qubits;
        let state_size = 1 << n_qubits;
        let mut state = Array1::zeros(state_size);

        let mut binary_index = 0;
        for (i, &val) in input.iter().enumerate() {
            if i < n_qubits && val > 0.5 {
                binary_index |= 1 << i;
            }
        }

        state[binary_index] = Complex64::new(1.0, 0.0);
        Ok(state)
    }

    fn encode_quantum_feature_map(&self, input: &Array1<f64>) -> Result<Array1<Complex64>> {
        let n_qubits = self.config.num_qubits;
        let state_size = 1 << n_qubits;
        let mut state = Array1::zeros(state_size);

        let hadamard_coeff = 1.0 / (n_qubits as f64 / 2.0).exp2();
        for i in 0..state_size {
            state[i] = Complex64::new(hadamard_coeff, 0.0);
        }

        for (i, &feature) in input.iter().enumerate() {
            if i < n_qubits {
                state = self.apply_rz_rotation(&state, i, feature * PI)?;
            }
        }

        for i in 0..(n_qubits - 1) {
            if i + 1 < input.len() {
                let interaction = input[i] * input[i + 1];
                state = self.apply_cnot_interaction(&state, i, i + 1, interaction * PI)?;
            }
        }

        Ok(state)
    }

    fn apply_ry_rotation(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> Result<Array1<Complex64>> {
        let n_qubits = self.config.num_qubits;
        let state_size = 1 << n_qubits;
        let mut new_state = state.clone();

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..state_size {
            if i & (1 << qubit) == 0 {
                let j = i | (1 << qubit);
                if j < state_size {
                    let state_0 = state[i];
                    let state_1 = state[j];

                    new_state[i] = Complex64::new(cos_half, 0.0) * state_0
                        - Complex64::new(sin_half, 0.0) * state_1;
                    new_state[j] = Complex64::new(sin_half, 0.0) * state_0
                        + Complex64::new(cos_half, 0.0) * state_1;
                }
            }
        }

        Ok(new_state)
    }

    fn apply_rz_rotation(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> Result<Array1<Complex64>> {
        let n_qubits = self.config.num_qubits;
        let state_size = 1 << n_qubits;
        let mut new_state = state.clone();

        let phase_0 = Complex64::from_polar(1.0, -angle / 2.0);
        let phase_1 = Complex64::from_polar(1.0, angle / 2.0);

        for i in 0..state_size {
            if i & (1 << qubit) == 0 {
                new_state[i] *= phase_0;
            } else {
                new_state[i] *= phase_1;
            }
        }

        Ok(new_state)
    }

    fn apply_cnot_interaction(
        &self,
        state: &Array1<Complex64>,
        control: usize,
        target: usize,
        interaction: f64,
    ) -> Result<Array1<Complex64>> {
        let n_qubits = self.config.num_qubits;
        let state_size = 1 << n_qubits;
        let mut new_state = state.clone();

        let phase = Complex64::from_polar(1.0, interaction);

        for i in 0..state_size {
            if (i & (1 << control)) != 0 && (i & (1 << target)) != 0 {
                new_state[i] *= phase;
            }
        }

        Ok(new_state)
    }

    fn decode_output(&self, state: &Array1<Complex64>) -> Result<Array1<f64>> {
        let n_qubits = self.config.num_qubits;
        let mut output = Array1::zeros(n_qubits);

        for qubit in 0..n_qubits {
            let expectation = Self::measure_pauli_z_expectation(state, qubit)?;
            output[qubit] = expectation;
        }

        Ok(output)
    }

    fn measure_pauli_z_expectation(state: &Array1<Complex64>, qubit: usize) -> Result<f64> {
        let state_size = state.len();
        let mut expectation = 0.0;

        for i in 0..state_size {
            let probability = state[i].norm_sqr();
            if i & (1 << qubit) == 0 {
                expectation += probability;
            } else {
                expectation -= probability;
            }
        }

        Ok(expectation)
    }

    fn compute_loss(prediction: &Array1<f64>, target: &Array1<f64>) -> Result<f64> {
        if prediction.shape() != target.shape() {
            return Err(SimulatorError::InvalidInput(format!(
                "Shape mismatch: prediction {:?} != target {:?}",
                prediction.shape(),
                target.shape()
            )));
        }

        let diff = prediction - target;
        let mse = diff.iter().map(|x| x * x).sum::<f64>() / diff.len() as f64;
        Ok(mse)
    }

    fn compute_loss_gradient(
        prediction: &Array1<f64>,
        target: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let diff = prediction - target;
        let grad = 2.0 * &diff / diff.len() as f64;
        Ok(grad)
    }

    fn compute_gradients(&mut self, loss_gradient: &Array1<f64>) -> Result<Vec<Array1<f64>>> {
        let mut gradients = Vec::new();

        for layer_idx in 0..self.layers.len() {
            let layer_gradient = match self.config.training_config.gradient_method {
                GradientMethod::ParameterShift => {
                    self.compute_parameter_shift_gradient(layer_idx, loss_gradient)?
                }
                GradientMethod::FiniteDifference => {
                    self.compute_finite_difference_gradient(layer_idx, loss_gradient)?
                }
                _ => {
                    return Err(SimulatorError::InvalidConfiguration(
                        "Gradient method not implemented".to_string(),
                    ))
                }
            };
            gradients.push(layer_gradient);
        }

        Ok(gradients)
    }

    fn compute_parameter_shift_gradient(
        &mut self,
        layer_idx: usize,
        loss_gradient: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let layer = &self.layers[layer_idx];
        let parameters = layer.get_parameters();
        let mut gradient = Array1::zeros(parameters.len());

        let shift = PI / 2.0;

        for (param_idx, &param_val) in parameters.iter().enumerate() {
            let mut params_plus = parameters.clone();
            params_plus[param_idx] = param_val + shift;
            self.layers[layer_idx].set_parameters(&params_plus);
            let output_plus = self.forward_layer(layer_idx, loss_gradient)?;

            let mut params_minus = parameters.clone();
            params_minus[param_idx] = param_val - shift;
            self.layers[layer_idx].set_parameters(&params_minus);
            let output_minus = self.forward_layer(layer_idx, loss_gradient)?;

            gradient[param_idx] = (output_plus.sum() - output_minus.sum()) / 2.0;

            self.layers[layer_idx].set_parameters(&parameters);
        }

        Ok(gradient)
    }

    fn compute_finite_difference_gradient(
        &mut self,
        layer_idx: usize,
        loss_gradient: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let layer = &self.layers[layer_idx];
        let parameters = layer.get_parameters();
        let mut gradient = Array1::zeros(parameters.len());

        let eps = 1e-6;

        for (param_idx, &param_val) in parameters.iter().enumerate() {
            let mut params_plus = parameters.clone();
            params_plus[param_idx] = param_val + eps;
            self.layers[layer_idx].set_parameters(&params_plus);
            let output_plus = self.forward_layer(layer_idx, loss_gradient)?;

            let mut params_minus = parameters.clone();
            params_minus[param_idx] = param_val - eps;
            self.layers[layer_idx].set_parameters(&params_minus);
            let output_minus = self.forward_layer(layer_idx, loss_gradient)?;

            gradient[param_idx] = (output_plus.sum() - output_minus.sum()) / (2.0 * eps);

            self.layers[layer_idx].set_parameters(&parameters);
        }

        Ok(gradient)
    }

    fn forward_layer(&mut self, _layer_idx: usize, input: &Array1<f64>) -> Result<Array1<f64>> {
        self.forward(input)
    }

    fn apply_gradients(&mut self, gradients: &[Array1<f64>]) -> Result<()> {
        for (layer_idx, gradient) in gradients.iter().enumerate() {
            let layer = &mut self.layers[layer_idx];
            let mut parameters = layer.get_parameters();

            match self.config.training_config.optimizer {
                OptimizerType::SGD => {
                    for (param, grad) in parameters.iter_mut().zip(gradient.iter()) {
                        *param -= self.config.training_config.learning_rate * grad;
                    }
                }
                OptimizerType::Adam => {
                    for (param, grad) in parameters.iter_mut().zip(gradient.iter()) {
                        *param -= self.config.training_config.learning_rate * grad;
                    }
                }
                _ => {
                    for (param, grad) in parameters.iter_mut().zip(gradient.iter()) {
                        *param -= self.config.training_config.learning_rate * grad;
                    }
                }
            }

            if let Some((min_val, max_val)) =
                self.config.training_config.regularization.parameter_bounds
            {
                for param in &mut parameters {
                    *param = param.clamp(min_val, max_val);
                }
            }

            layer.set_parameters(&parameters);
        }

        Ok(())
    }

    fn get_current_learning_rate(&self, epoch: usize) -> f64 {
        let base_lr = self.config.training_config.learning_rate;

        match self.config.training_config.lr_schedule {
            LearningRateSchedule::Constant => base_lr,
            LearningRateSchedule::ExponentialDecay => base_lr * 0.95_f64.powi(epoch as i32),
            LearningRateSchedule::StepDecay => {
                if epoch % 50 == 0 && epoch > 0 {
                    base_lr * 0.5_f64.powi((epoch / 50) as i32)
                } else {
                    base_lr
                }
            }
            LearningRateSchedule::CosineAnnealing => {
                let progress = epoch as f64 / self.config.training_config.epochs as f64;
                base_lr * 0.5 * (1.0 + (PI * progress).cos())
            }
            _ => base_lr,
        }
    }

    fn update_learning_rate(&mut self, epoch: usize, _validation_loss: f64) {
        let current_lr = self.get_current_learning_rate(epoch);
        self.training_state.current_learning_rate = current_lr;
    }

    fn compute_quantum_advantage_metrics(&self) -> Result<QuantumAdvantageMetrics> {
        Ok(QuantumAdvantageMetrics {
            quantum_volume: 0.0,
            classical_simulation_cost: 0.0,
            quantum_speedup_factor: 1.0,
            circuit_depth: self.layers.iter().map(|l| l.get_depth()).sum(),
            gate_count: self.layers.iter().map(|l| l.get_gate_count()).sum(),
            entanglement_measure: 0.0,
        })
    }

    #[must_use]
    pub const fn get_stats(&self) -> &QMLStats {
        &self.stats
    }

    #[must_use]
    pub fn get_training_history(&self) -> &[QMLTrainingResult] {
        &self.training_history
    }

    #[must_use]
    pub fn get_layers(&self) -> &[Box<dyn QMLLayer>] {
        &self.layers
    }

    #[must_use]
    pub const fn get_config(&self) -> &QMLConfig {
        &self.config
    }

    pub fn encode_amplitude_public(&self, input: &Array1<f64>) -> Result<Array1<Complex64>> {
        self.encode_amplitude(input)
    }

    pub fn encode_angle_public(&self, input: &Array1<f64>) -> Result<Array1<Complex64>> {
        self.encode_angle(input)
    }

    pub fn encode_basis_public(&self, input: &Array1<f64>) -> Result<Array1<Complex64>> {
        self.encode_basis(input)
    }

    pub fn encode_quantum_feature_map_public(
        &self,
        input: &Array1<f64>,
    ) -> Result<Array1<Complex64>> {
        self.encode_quantum_feature_map(input)
    }

    pub fn measure_pauli_z_expectation_public(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
    ) -> Result<f64> {
        Self::measure_pauli_z_expectation(state, qubit)
    }

    #[must_use]
    pub fn get_current_learning_rate_public(&self, epoch: usize) -> f64 {
        self.get_current_learning_rate(epoch)
    }

    pub fn compute_loss_public(
        &self,
        prediction: &Array1<f64>,
        target: &Array1<f64>,
    ) -> Result<f64> {
        Self::compute_loss(prediction, target)
    }

    pub fn compute_loss_gradient_public(
        &self,
        prediction: &Array1<f64>,
        target: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        Self::compute_loss_gradient(prediction, target)
    }
}
