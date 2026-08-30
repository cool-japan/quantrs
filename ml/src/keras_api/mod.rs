//! Keras-style model building API for QuantRS2-ML
//!
//! This module provides a Keras-like interface for building quantum machine learning
//! models, with both Sequential and Functional API patterns familiar to Keras users.

mod attention;
mod callbacks;
mod conv;
mod layers;
mod quantum_layers;
mod rnn;
mod schedules;

pub use attention::*;
pub use callbacks::*;
pub use conv::*;
pub use layers::*;
pub use quantum_layers::*;
pub use rnn::*;
pub use schedules::*;

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{s, ArrayD, Axis, IxDyn};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

/// Keras-style layer trait
pub trait KerasLayer: Send + Sync {
    /// Build the layer (called during model compilation)
    fn build(&mut self, input_shape: &[usize]) -> Result<()>;

    /// Forward pass through the layer
    fn call(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>>;

    /// Compute output shape given input shape
    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize>;

    /// Get layer name
    fn name(&self) -> &str;

    /// Get trainable parameters
    fn get_weights(&self) -> Vec<ArrayD<f64>>;

    /// Set trainable parameters
    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()>;

    /// Get number of parameters
    fn count_params(&self) -> usize {
        self.get_weights().iter().map(|w| w.len()).sum()
    }

    /// Check if layer is built
    fn built(&self) -> bool;

    /// Get a stable type tag identifying this layer's concrete kind (e.g.
    /// `"Dense"`, `"QuantumDense"`, `"Activation"`).
    ///
    /// Exporters such as [`crate::onnx_export::ONNXExporter`] only see layers
    /// behind `dyn KerasLayer` and must still dispatch on the concrete layer
    /// kind. This default implementation derives the tag from the concrete
    /// Rust type name (via `std::any::type_name`), so every existing
    /// `KerasLayer` implementor gets a correct, distinct tag for free without
    /// needing to override this method; a layer may still override it if it
    /// wants a different public-facing name than its Rust type name.
    fn layer_type(&self) -> &'static str {
        let full_path = std::any::type_name::<Self>();
        full_path.rsplit("::").next().unwrap_or(full_path)
    }
}

/// Activation function types
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    /// Linear activation (identity)
    Linear,
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax activation
    Softmax,
    /// Leaky ReLU with alpha
    LeakyReLU(f64),
    /// ELU with alpha
    ELU(f64),
}

/// Weight initializer types
#[derive(Debug, Clone)]
pub enum InitializerType {
    /// All zeros
    Zeros,
    /// All ones
    Ones,
    /// Glorot uniform (Xavier uniform)
    GlorotUniform,
    /// Glorot normal (Xavier normal)
    GlorotNormal,
    /// He uniform
    HeUniform,
}

/// Sequential model
pub struct Sequential {
    /// Layers in the model
    layers: Vec<Box<dyn KerasLayer>>,
    /// Model name
    name: String,
    /// Built flag
    built: bool,
    /// Compiled flag
    compiled: bool,
    /// Input shape
    input_shape: Option<Vec<usize>>,
    /// Loss function
    loss: Option<LossFunction>,
    /// Optimizer
    optimizer: Option<OptimizerType>,
    /// Metrics
    metrics: Vec<MetricType>,
}

impl Sequential {
    /// Create new sequential model
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            name: format!("sequential_{}", fastrand::u32(..)),
            built: false,
            compiled: false,
            input_shape: None,
            loss: None,
            optimizer: None,
            metrics: Vec::new(),
        }
    }

    /// Set model name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add layer to model
    pub fn add(&mut self, layer: Box<dyn KerasLayer>) {
        self.layers.push(layer);
        self.built = false;
    }

    /// Get the model's layers in order.
    ///
    /// Exposed so callers outside this module (e.g. the ONNX exporter in
    /// [`crate::onnx_export`]) can walk the real layer list rather than being
    /// forced to reach for a hardcoded stand-in, since `layers` itself is a
    /// private field of `Sequential`.
    pub fn layers(&self) -> &[Box<dyn KerasLayer>] {
        &self.layers
    }

    /// Compute the model's output shape for a given input shape by chaining
    /// each layer's own [`KerasLayer::compute_output_shape`] in sequence,
    /// exactly like [`Self::build`] does.
    pub fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let mut current_shape = input_shape.to_vec();
        for layer in &self.layers {
            current_shape = layer.compute_output_shape(&current_shape);
        }
        current_shape
    }

    /// Build the model with given input shape
    pub fn build(&mut self, input_shape: Vec<usize>) -> Result<()> {
        self.input_shape = Some(input_shape.clone());
        let mut current_shape = input_shape;

        for layer in &mut self.layers {
            layer.build(&current_shape)?;
            current_shape = layer.compute_output_shape(&current_shape);
        }

        self.built = true;
        Ok(())
    }

    /// Compile the model
    pub fn compile(
        mut self,
        loss: LossFunction,
        optimizer: OptimizerType,
        metrics: Vec<MetricType>,
    ) -> Self {
        self.loss = Some(loss);
        self.optimizer = Some(optimizer);
        self.metrics = metrics;
        self.compiled = true;
        self
    }

    /// Get model summary
    pub fn summary(&self) -> ModelSummary {
        let mut layers_info = Vec::new();
        let mut total_params = 0;
        let mut trainable_params = 0;

        let mut current_shape = self.input_shape.clone().unwrap_or_default();

        for layer in &self.layers {
            let output_shape = layer.compute_output_shape(&current_shape);
            let params = layer.count_params();

            layers_info.push(LayerInfo {
                name: layer.name().to_string(),
                layer_type: "Layer".to_string(),
                output_shape: output_shape.clone(),
                param_count: params,
            });

            total_params += params;
            trainable_params += params;
            current_shape = output_shape;
        }

        ModelSummary {
            layers: layers_info,
            total_params,
            trainable_params,
            non_trainable_params: 0,
        }
    }

    /// Forward pass (predict)
    pub fn predict(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::InvalidConfiguration(
                "Model must be built before prediction".to_string(),
            ));
        }

        let mut current = inputs.clone();

        for layer in &self.layers {
            current = layer.call(&current)?;
        }

        Ok(current)
    }

    /// Train the model
    #[allow(non_snake_case)]
    pub fn fit(
        &mut self,
        X: &ArrayD<f64>,
        y: &ArrayD<f64>,
        epochs: usize,
        batch_size: Option<usize>,
        validation_data: Option<(&ArrayD<f64>, &ArrayD<f64>)>,
        callbacks: Vec<Box<dyn Callback>>,
    ) -> Result<TrainingHistory> {
        if !self.compiled {
            return Err(MLError::InvalidConfiguration(
                "Model must be compiled before training".to_string(),
            ));
        }

        let batch_size = batch_size.unwrap_or(32);
        let n_samples = X.shape()[0];
        let n_batches = (n_samples + batch_size - 1) / batch_size;

        let mut history = TrainingHistory::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut epoch_metrics: HashMap<String, f64> = HashMap::new();

            for metric in &self.metrics {
                epoch_metrics.insert(metric.name(), 0.0);
            }

            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = ((batch_idx + 1) * batch_size).min(n_samples);

                let X_batch = X.slice(s![start_idx..end_idx, ..]);
                let y_batch = y.slice(s![start_idx..end_idx, ..]);

                let predictions = self.predict(&X_batch.to_owned().into_dyn())?;

                let loss = self.compute_loss(&predictions, &y_batch.to_owned().into_dyn())?;
                epoch_loss += loss;

                self.backward_pass(
                    &X_batch.to_owned().into_dyn(),
                    &y_batch.to_owned().into_dyn(),
                )?;

                for metric in &self.metrics {
                    let metric_value =
                        metric.compute(&predictions, &y_batch.to_owned().into_dyn())?;
                    *epoch_metrics.entry(metric.name()).or_insert(0.0) += metric_value;
                }
            }

            epoch_loss /= n_batches as f64;
            for value in epoch_metrics.values_mut() {
                *value /= n_batches as f64;
            }

            let (val_loss, val_metrics) = if let Some((X_val, y_val)) = validation_data {
                let val_predictions = self.predict(X_val)?;
                let val_loss = self.compute_loss(&val_predictions, y_val)?;

                let mut val_metrics = HashMap::new();
                for metric in &self.metrics {
                    let metric_value = metric.compute(&val_predictions, y_val)?;
                    val_metrics.insert(format!("val_{}", metric.name()), metric_value);
                }

                (Some(val_loss), val_metrics)
            } else {
                (None, HashMap::new())
            };

            history.add_epoch(epoch_loss, epoch_metrics, val_loss, val_metrics);

            for callback in &callbacks {
                callback.on_epoch_end(epoch, &history)?;
            }

            println!("Epoch {}/{} - loss: {:.4}", epoch + 1, epochs, epoch_loss);
        }

        Ok(history)
    }

    /// Evaluate the model
    #[allow(non_snake_case)]
    pub fn evaluate(
        &self,
        X: &ArrayD<f64>,
        y: &ArrayD<f64>,
        _batch_size: Option<usize>,
    ) -> Result<HashMap<String, f64>> {
        let predictions = self.predict(X)?;
        let loss = self.compute_loss(&predictions, y)?;

        let mut results = HashMap::new();
        results.insert("loss".to_string(), loss);

        for metric in &self.metrics {
            let metric_value = metric.compute(&predictions, y)?;
            results.insert(metric.name(), metric_value);
        }

        Ok(results)
    }

    /// Compute loss
    fn compute_loss(&self, predictions: &ArrayD<f64>, targets: &ArrayD<f64>) -> Result<f64> {
        if let Some(ref loss_fn) = self.loss {
            loss_fn.compute(predictions, targets)
        } else {
            Err(MLError::InvalidConfiguration(
                "Loss function not specified".to_string(),
            ))
        }
    }

    /// Backward pass: for every layer with trainable weights, estimate a
    /// real SPSA (simultaneous perturbation stochastic approximation)
    /// gradient of the compiled loss function with respect to that layer's
    /// flattened weights (by actually re-running `predict` on `x_batch` and
    /// `compute_loss` at perturbed weight values -- not fabricated), then
    /// apply a real gradient-descent step at the compiled optimizer's
    /// learning rate. Previously this was a complete no-op, so `fit`
    /// reported an honest per-batch loss but never updated any layer's
    /// weights.
    fn backward_pass(&mut self, x_batch: &ArrayD<f64>, targets: &ArrayD<f64>) -> Result<()> {
        const PERTURBATION_SCALE: f64 = 1e-3;
        let learning_rate = self
            .optimizer
            .as_ref()
            .map(optimizer_learning_rate)
            .unwrap_or(0.01);

        for layer_idx in 0..self.layers.len() {
            let weights = self.layers[layer_idx].get_weights();
            if weights.is_empty() {
                continue;
            }
            let shapes: Vec<Vec<usize>> = weights.iter().map(|w| w.shape().to_vec()).collect();
            let flat: Vec<f64> = weights.iter().flat_map(|w| w.iter().cloned()).collect();

            let mut rng = thread_rng();
            let direction: Vec<f64> = (0..flat.len())
                .map(|_| if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 })
                .collect();

            let plus_flat: Vec<f64> = flat
                .iter()
                .zip(direction.iter())
                .map(|(w, d)| w + PERTURBATION_SCALE * d)
                .collect();
            self.set_layer_weights_from_flat(layer_idx, &shapes, &plus_flat)?;
            let predictions_plus = self.predict(x_batch)?;
            let loss_plus = self.compute_loss(&predictions_plus, targets)?;

            let minus_flat: Vec<f64> = flat
                .iter()
                .zip(direction.iter())
                .map(|(w, d)| w - PERTURBATION_SCALE * d)
                .collect();
            self.set_layer_weights_from_flat(layer_idx, &shapes, &minus_flat)?;
            let predictions_minus = self.predict(x_batch)?;
            let loss_minus = self.compute_loss(&predictions_minus, targets)?;

            let loss_delta = loss_plus - loss_minus;
            let gradient: Vec<f64> = direction
                .iter()
                .map(|d| (loss_delta / (2.0 * PERTURBATION_SCALE)) * d)
                .collect();
            let updated_flat: Vec<f64> = flat
                .iter()
                .zip(gradient.iter())
                .map(|(w, g)| w - learning_rate * g)
                .collect();
            self.set_layer_weights_from_flat(layer_idx, &shapes, &updated_flat)?;
        }

        Ok(())
    }

    /// Reshape a flat weight vector back into the per-tensor shapes it came
    /// from and apply it to layer `layer_idx` via [`KerasLayer::set_weights`].
    fn set_layer_weights_from_flat(
        &mut self,
        layer_idx: usize,
        shapes: &[Vec<usize>],
        flat: &[f64],
    ) -> Result<()> {
        let mut offset = 0;
        let mut weights = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let len: usize = shape.iter().product();
            let slice = flat[offset..offset + len].to_vec();
            let array = ArrayD::from_shape_vec(IxDyn(shape), slice).map_err(|e| {
                MLError::ComputationError(format!("Failed to reshape layer weights: {e}"))
            })?;
            weights.push(array);
            offset += len;
        }
        self.layers[layer_idx].set_weights(weights)
    }
}

/// Extract the learning rate carried by an [`OptimizerType`] variant.
fn optimizer_learning_rate(optimizer: &OptimizerType) -> f64 {
    match optimizer {
        OptimizerType::SGD { learning_rate, .. } => *learning_rate,
        OptimizerType::Adam { learning_rate, .. } => *learning_rate,
        OptimizerType::RMSprop { learning_rate, .. } => *learning_rate,
        OptimizerType::AdaGrad { learning_rate, .. } => *learning_rate,
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

/// Loss functions
#[derive(Debug, Clone)]
pub enum LossFunction {
    /// Mean squared error
    MeanSquaredError,
    /// Binary crossentropy
    BinaryCrossentropy,
    /// Categorical crossentropy
    CategoricalCrossentropy,
    /// Sparse categorical crossentropy
    SparseCategoricalCrossentropy,
    /// Mean absolute error
    MeanAbsoluteError,
    /// Huber loss
    Huber(f64),
}

impl LossFunction {
    /// Compute loss
    pub fn compute(&self, predictions: &ArrayD<f64>, targets: &ArrayD<f64>) -> Result<f64> {
        match self {
            LossFunction::MeanSquaredError => {
                let diff = predictions - targets;
                diff.mapv(|x| x * x).mean().ok_or_else(|| {
                    MLError::ComputationError("Failed to compute mean of empty array".to_string())
                })
            }
            LossFunction::BinaryCrossentropy => {
                let epsilon = 1e-15;
                let clipped_preds = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                let loss = targets * clipped_preds.mapv(|x| x.ln())
                    + (1.0 - targets) * clipped_preds.mapv(|x| (1.0 - x).ln());
                loss.mean().map(|m| -m).ok_or_else(|| {
                    MLError::ComputationError("Failed to compute mean of empty array".to_string())
                })
            }
            LossFunction::MeanAbsoluteError => {
                let diff = predictions - targets;
                diff.mapv(|x| x.abs()).mean().ok_or_else(|| {
                    MLError::ComputationError("Failed to compute mean of empty array".to_string())
                })
            }
            _ => Err(MLError::InvalidConfiguration(
                "Loss function not implemented".to_string(),
            )),
        }
    }
}

/// Optimizer types
#[derive(Debug, Clone)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD { learning_rate: f64, momentum: f64 },
    /// Adam optimizer
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// RMSprop optimizer
    RMSprop {
        learning_rate: f64,
        rho: f64,
        epsilon: f64,
    },
    /// AdaGrad optimizer
    AdaGrad { learning_rate: f64, epsilon: f64 },
}

/// Metric types
#[derive(Debug, Clone)]
pub enum MetricType {
    /// Accuracy
    Accuracy,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// F1 Score
    F1Score,
    /// Mean Absolute Error
    MeanAbsoluteError,
    /// Mean Squared Error
    MeanSquaredError,
}

impl MetricType {
    /// Get metric name
    pub fn name(&self) -> String {
        match self {
            MetricType::Accuracy => "accuracy".to_string(),
            MetricType::Precision => "precision".to_string(),
            MetricType::Recall => "recall".to_string(),
            MetricType::F1Score => "f1_score".to_string(),
            MetricType::MeanAbsoluteError => "mean_absolute_error".to_string(),
            MetricType::MeanSquaredError => "mean_squared_error".to_string(),
        }
    }

    /// Compute metric
    pub fn compute(&self, predictions: &ArrayD<f64>, targets: &ArrayD<f64>) -> Result<f64> {
        match self {
            MetricType::Accuracy => {
                let pred_classes = predictions.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 });
                let correct = pred_classes
                    .iter()
                    .zip(targets.iter())
                    .filter(|(&pred, &target)| (pred - target).abs() < 1e-6)
                    .count();
                Ok(correct as f64 / targets.len() as f64)
            }
            MetricType::MeanAbsoluteError => {
                let diff = predictions - targets;
                diff.mapv(|x| x.abs()).mean().ok_or_else(|| {
                    MLError::ComputationError("Failed to compute mean of empty array".to_string())
                })
            }
            MetricType::MeanSquaredError => {
                let diff = predictions - targets;
                diff.mapv(|x| x * x).mean().ok_or_else(|| {
                    MLError::ComputationError("Failed to compute mean of empty array".to_string())
                })
            }
            MetricType::Precision => {
                let true_positives = predictions
                    .iter()
                    .zip(targets.iter())
                    .filter(|(&pred, &target)| pred > 0.5 && target > 0.5)
                    .count() as f64;
                let predicted_positives =
                    predictions.iter().filter(|&&pred| pred > 0.5).count() as f64;
                if predicted_positives > 0.0 {
                    Ok(true_positives / predicted_positives)
                } else {
                    Ok(0.0)
                }
            }
            MetricType::Recall => {
                let true_positives = predictions
                    .iter()
                    .zip(targets.iter())
                    .filter(|(&pred, &target)| pred > 0.5 && target > 0.5)
                    .count() as f64;
                let actual_positives =
                    targets.iter().filter(|&&target| target > 0.5).count() as f64;
                if actual_positives > 0.0 {
                    Ok(true_positives / actual_positives)
                } else {
                    Ok(0.0)
                }
            }
            MetricType::F1Score => {
                let precision = MetricType::Precision.compute(predictions, targets)?;
                let recall = MetricType::Recall.compute(predictions, targets)?;
                if precision + recall > 0.0 {
                    Ok(2.0 * precision * recall / (precision + recall))
                } else {
                    Ok(0.0)
                }
            }
        }
    }
}

/// Training history
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Training loss for each epoch
    pub loss: Vec<f64>,
    /// Training metrics for each epoch
    pub metrics: Vec<HashMap<String, f64>>,
    /// Validation loss for each epoch
    pub val_loss: Vec<f64>,
    /// Validation metrics for each epoch
    pub val_metrics: Vec<HashMap<String, f64>>,
}

impl TrainingHistory {
    /// Create new training history
    pub fn new() -> Self {
        Self {
            loss: Vec::new(),
            metrics: Vec::new(),
            val_loss: Vec::new(),
            val_metrics: Vec::new(),
        }
    }

    /// Add epoch results
    pub fn add_epoch(
        &mut self,
        loss: f64,
        metrics: HashMap<String, f64>,
        val_loss: Option<f64>,
        val_metrics: HashMap<String, f64>,
    ) {
        self.loss.push(loss);
        self.metrics.push(metrics);

        if let Some(val_loss) = val_loss {
            self.val_loss.push(val_loss);
        }
        self.val_metrics.push(val_metrics);
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Model summary information
#[derive(Debug)]
pub struct ModelSummary {
    /// Layer information
    pub layers: Vec<LayerInfo>,
    /// Total number of parameters
    pub total_params: usize,
    /// Number of trainable parameters
    pub trainable_params: usize,
    /// Number of non-trainable parameters
    pub non_trainable_params: usize,
}

/// Layer information for summary
#[derive(Debug)]
pub struct LayerInfo {
    /// Layer name
    pub name: String,
    /// Layer type
    pub layer_type: String,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Parameter count
    pub param_count: usize,
}

/// Model input specification
pub struct Input {
    /// Input shape (excluding batch dimension)
    pub shape: Vec<usize>,
    /// Input name
    pub name: Option<String>,
    /// Data type
    pub dtype: DataType,
}

impl Input {
    /// Create new input specification
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            name: None,
            dtype: DataType::Float64,
        }
    }

    /// Set input name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set data type
    pub fn dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }
}

/// Data types
#[derive(Debug, Clone)]
pub enum DataType {
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
}

/// Utility functions for building models
pub mod utils {
    use super::*;

    /// Create a simple sequential model for classification
    pub fn create_classification_model(
        _input_dim: usize,
        num_classes: usize,
        hidden_layers: Vec<usize>,
    ) -> Sequential {
        let mut model = Sequential::new();

        for (i, &units) in hidden_layers.iter().enumerate() {
            model.add(Box::new(
                Dense::new(units)
                    .activation(ActivationFunction::ReLU)
                    .name(format!("dense_{}", i)),
            ));
        }

        let output_activation = if num_classes == 2 {
            ActivationFunction::Sigmoid
        } else {
            ActivationFunction::Softmax
        };

        model.add(Box::new(
            Dense::new(num_classes)
                .activation(output_activation)
                .name("output"),
        ));

        model
    }

    /// Create a quantum neural network model
    pub fn create_quantum_model(
        num_qubits: usize,
        num_classes: usize,
        num_layers: usize,
    ) -> Sequential {
        let mut model = Sequential::new();

        model.add(Box::new(
            QuantumDense::new(num_qubits, num_classes)
                .num_layers(num_layers)
                .ansatz_type(QuantumAnsatzType::HardwareEfficient)
                .name("quantum_layer"),
        ));

        if num_classes > 1 {
            model.add(Box::new(
                Activation::new(ActivationFunction::Softmax).name("softmax"),
            ));
        }

        model
    }

    /// Create a hybrid quantum-classical model
    pub fn create_hybrid_model(
        _input_dim: usize,
        num_qubits: usize,
        num_classes: usize,
        classical_hidden: Vec<usize>,
    ) -> Sequential {
        let mut model = Sequential::new();

        for (i, &units) in classical_hidden.iter().enumerate() {
            model.add(Box::new(
                Dense::new(units)
                    .activation(ActivationFunction::ReLU)
                    .name(format!("classical_{}", i)),
            ));
        }

        model.add(Box::new(
            QuantumDense::new(num_qubits, 64)
                .num_layers(2)
                .ansatz_type(QuantumAnsatzType::HardwareEfficient)
                .name("quantum_layer"),
        ));

        model.add(Box::new(
            Dense::new(num_classes)
                .activation(if num_classes == 2 {
                    ActivationFunction::Sigmoid
                } else {
                    ActivationFunction::Softmax
                })
                .name("output"),
        ));

        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_dense_layer() {
        let mut dense = Dense::new(10)
            .activation(ActivationFunction::ReLU)
            .name("test_dense");

        assert!(!dense.built());

        dense.build(&[5]).expect("Should build successfully");

        assert!(dense.built());
        assert_eq!(dense.compute_output_shape(&[32, 5]), vec![32, 10]);
    }

    #[test]
    fn test_sequential_model() {
        let mut model = Sequential::new();
        model.add(Box::new(Dense::new(10)));
        model.add(Box::new(Activation::new(ActivationFunction::ReLU)));
        model.add(Box::new(Dense::new(5)));

        model
            .build(vec![32, 20])
            .expect("Should build successfully");

        let summary = model.summary();
        assert_eq!(summary.layers.len(), 3);
    }

    #[test]
    fn test_activation_functions() {
        let relu = ActivationFunction::ReLU;
        let sigmoid = ActivationFunction::Sigmoid;
        let _tanh = ActivationFunction::Tanh;

        let mut act_relu = Activation::new(relu);
        act_relu.build(&[10]).expect("Should build");

        let mut act_sigmoid = Activation::new(sigmoid);
        act_sigmoid.build(&[10]).expect("Should build");
    }

    /// Regression test for the "backward_pass is a no-op" bug: `fit` must
    /// actually change the Dense layer's weights and reduce the training
    /// loss on an easily learnable linear-regression toy problem.
    #[test]
    fn test_fit_updates_weights_and_reduces_loss() {
        let mut model = Sequential::new();
        model.add(Box::new(Dense::new(2).name("dense1")));
        model.build(vec![4, 3]).expect("build should succeed");
        let mut model = model.compile(
            LossFunction::MeanSquaredError,
            OptimizerType::SGD {
                learning_rate: 0.5,
                momentum: 0.0,
            },
            vec![],
        );

        let x = Array::from_shape_vec(
            IxDyn(&[4, 3]),
            vec![0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        )
        .expect("valid shape");
        let y = Array::from_shape_vec(IxDyn(&[4, 2]), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
            .expect("valid shape");

        let weights_before = model.layers[0].get_weights();
        let initial_predictions = model.predict(&x).expect("predict should succeed");
        let initial_loss = model
            .compute_loss(&initial_predictions, &y)
            .expect("loss should compute");

        let history = model
            .fit(&x, &y, 150, Some(4), None, vec![])
            .expect("fit should succeed");

        let weights_after = model.layers[0].get_weights();
        let weights_changed = weights_before[0]
            .iter()
            .zip(weights_after[0].iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(
            weights_changed,
            "expected Dense layer weights to change after fit"
        );

        let final_loss = *history.loss.last().expect("history should have losses");
        assert!(
            final_loss < initial_loss,
            "expected training loss to decrease: initial={initial_loss}, final={final_loss}"
        );
    }

    /// Regression test for the "Precision/Recall/F1Score always error" gap:
    /// these metrics must now be computed from real confusion-matrix counts.
    #[test]
    fn test_precision_recall_f1_are_computed() {
        let predictions =
            Array::from_shape_vec(IxDyn(&[4]), vec![0.9, 0.1, 0.8, 0.3]).expect("valid shape");
        let targets =
            Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.0, 0.0, 1.0]).expect("valid shape");

        let precision = MetricType::Precision
            .compute(&predictions, &targets)
            .expect("precision should compute");
        let recall = MetricType::Recall
            .compute(&predictions, &targets)
            .expect("recall should compute");
        let f1 = MetricType::F1Score
            .compute(&predictions, &targets)
            .expect("f1 should compute");

        // Predicted positives (pred > 0.5): indices 0, 2. True positives
        // among those (target > 0.5 too): index 0 only -> precision = 1/2.
        assert!((precision - 0.5).abs() < 1e-9, "precision was {precision}");
        // Actual positives (target > 0.5): indices 0, 3. True positives
        // found: index 0 only -> recall = 1/2.
        assert!((recall - 0.5).abs() < 1e-9, "recall was {recall}");
        // Equal precision and recall -> F1 equals the same value.
        assert!((f1 - 0.5).abs() < 1e-9, "f1 was {f1}");
    }
}
