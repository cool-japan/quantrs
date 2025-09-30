//! PyTorch-like API for quantum machine learning models
//!
//! This module provides a familiar PyTorch-style interface for building,
//! training, and deploying quantum ML models, making it easier for classical
//! ML practitioners to adopt quantum algorithms.

use crate::circuit_integration::QuantumMLExecutor;
use crate::error::{MLError, Result};
use crate::scirs2_integration::{SciRS2Array, SciRS2Optimizer};
use crate::simulator_backends::{Observable, SimulatorBackend};
use scirs2_core::ndarray::{Array1, Array2, ArrayD, Axis, Dimension, IxDyn};
use quantrs2_circuit::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Base trait for all quantum ML modules
pub trait QuantumModule: Send + Sync {
    /// Forward pass
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array>;

    /// Get all parameters
    fn parameters(&self) -> Vec<Parameter>;

    /// Set training mode
    fn train(&mut self, mode: bool);

    /// Check if module is in training mode
    fn training(&self) -> bool;

    /// Zero gradients of all parameters
    fn zero_grad(&mut self);

    /// Module name for debugging
    fn name(&self) -> &str;
}

/// Quantum parameter wrapper
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter data
    pub data: SciRS2Array,
    /// Parameter name
    pub name: String,
    /// Whether parameter requires gradient
    pub requires_grad: bool,
}

impl Parameter {
    /// Create new parameter
    pub fn new(data: SciRS2Array, name: impl Into<String>) -> Self {
        Self {
            data,
            name: name.into(),
            requires_grad: true,
        }
    }

    /// Create parameter without gradients
    pub fn no_grad(data: SciRS2Array, name: impl Into<String>) -> Self {
        Self {
            data,
            name: name.into(),
            requires_grad: false,
        }
    }

    /// Get parameter shape
    pub fn shape(&self) -> &[usize] {
        self.data.data.shape()
    }

    /// Get parameter size
    pub fn numel(&self) -> usize {
        self.data.data.len()
    }
}

/// Quantum linear layer
pub struct QuantumLinear {
    /// Weight parameters
    weights: Parameter,
    /// Bias parameters (optional)
    bias: Option<Parameter>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Training mode
    training: bool,
    /// Circuit executor
    executor: QuantumMLExecutor<8>, // Fixed const size for now
}

impl QuantumLinear {
    /// Create new quantum linear layer
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        let weight_data = ArrayD::zeros(IxDyn(&[out_features, in_features]));
        let weights = Parameter::new(SciRS2Array::with_grad(weight_data), "weight");

        Ok(Self {
            weights,
            bias: None,
            in_features,
            out_features,
            training: true,
            executor: QuantumMLExecutor::new(),
        })
    }

    /// Create with bias
    pub fn with_bias(mut self) -> Result<Self> {
        let bias_data = ArrayD::zeros(IxDyn(&[self.out_features]));
        self.bias = Some(Parameter::new(SciRS2Array::with_grad(bias_data), "bias"));
        Ok(self)
    }

    /// Initialize weights using Xavier/Glorot uniform
    pub fn init_xavier_uniform(&mut self) -> Result<()> {
        let fan_in = self.in_features as f64;
        let fan_out = self.out_features as f64;
        let bound = (6.0 / (fan_in + fan_out)).sqrt();

        for elem in self.weights.data.data.iter_mut() {
            *elem = (fastrand::f64() * 2.0 - 1.0) * bound;
        }

        Ok(())
    }
}

impl QuantumModule for QuantumLinear {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        // Quantum linear transformation: output = input @ weights.T + bias
        let output = input.matmul(&self.weights.data)?;

        if let Some(ref bias) = self.bias {
            output.add(&bias.data)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weights.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.weights.data.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.data.zero_grad();
        }
    }

    fn name(&self) -> &str {
        "QuantumLinear"
    }
}

/// Quantum convolutional layer
pub struct QuantumConv2d {
    /// Convolution parameters
    weights: Parameter,
    /// Bias parameters
    bias: Option<Parameter>,
    /// Input channels
    in_channels: usize,
    /// Output channels
    out_channels: usize,
    /// Kernel size
    kernel_size: (usize, usize),
    /// Stride
    stride: (usize, usize),
    /// Padding
    padding: (usize, usize),
    /// Training mode
    training: bool,
}

impl QuantumConv2d {
    /// Create new quantum conv2d layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Result<Self> {
        let weight_shape = [out_channels, in_channels, kernel_size.0, kernel_size.1];
        let weight_data = ArrayD::zeros(IxDyn(&weight_shape));
        let weights = Parameter::new(SciRS2Array::with_grad(weight_data), "weight");

        Ok(Self {
            weights,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            training: true,
        })
    }

    /// Set stride
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Add bias
    pub fn with_bias(mut self) -> Result<Self> {
        let bias_data = ArrayD::zeros(IxDyn(&[self.out_channels]));
        self.bias = Some(Parameter::new(SciRS2Array::with_grad(bias_data), "bias"));
        Ok(self)
    }
}

impl QuantumModule for QuantumConv2d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        // Quantum convolution implementation (simplified)
        // In practice, this would implement quantum convolution operations
        let output_data = input.data.clone(); // Placeholder
        let mut output = SciRS2Array::new(output_data, input.requires_grad);

        if let Some(ref bias) = self.bias {
            output = output.add(&bias.data)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weights.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.weights.data.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.data.zero_grad();
        }
    }

    fn name(&self) -> &str {
        "QuantumConv2d"
    }
}

/// Quantum activation functions
pub struct QuantumActivation {
    /// Activation function type
    activation_type: ActivationType,
    /// Training mode
    training: bool,
}

/// Activation function types
#[derive(Debug, Clone)]
pub enum ActivationType {
    /// Quantum ReLU (using rotation gates)
    QReLU,
    /// Quantum Sigmoid
    QSigmoid,
    /// Quantum Tanh
    QTanh,
    /// Quantum Softmax
    QSoftmax,
    /// Identity (no activation)
    Identity,
}

impl QuantumActivation {
    /// Create new activation layer
    pub fn new(activation_type: ActivationType) -> Self {
        Self {
            activation_type,
            training: true,
        }
    }

    /// Create ReLU activation
    pub fn relu() -> Self {
        Self::new(ActivationType::QReLU)
    }

    /// Create Sigmoid activation
    pub fn sigmoid() -> Self {
        Self::new(ActivationType::QSigmoid)
    }

    /// Create Tanh activation
    pub fn tanh() -> Self {
        Self::new(ActivationType::QTanh)
    }

    /// Create Softmax activation
    pub fn softmax() -> Self {
        Self::new(ActivationType::QSoftmax)
    }
}

impl QuantumModule for QuantumActivation {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        match self.activation_type {
            ActivationType::QReLU => {
                // Quantum ReLU: max(0, x) approximation using quantum gates
                let output_data = input.data.mapv(|x| x.max(0.0));
                Ok(SciRS2Array::new(output_data, input.requires_grad))
            }
            ActivationType::QSigmoid => {
                // Quantum sigmoid approximation
                let output_data = input.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                Ok(SciRS2Array::new(output_data, input.requires_grad))
            }
            ActivationType::QTanh => {
                // Quantum tanh
                let output_data = input.data.mapv(|x| x.tanh());
                Ok(SciRS2Array::new(output_data, input.requires_grad))
            }
            ActivationType::QSoftmax => {
                // Quantum softmax
                let max_val = input.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_data = input.data.mapv(|x| (x - max_val).exp());
                let sum_exp = exp_data.sum();
                let output_data = exp_data.mapv(|x| x / sum_exp);
                Ok(SciRS2Array::new(output_data, input.requires_grad))
            }
            ActivationType::Identity => {
                Ok(SciRS2Array::new(input.data.clone(), input.requires_grad))
            }
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new() // Activation functions typically don't have parameters
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        // No parameters to zero
    }

    fn name(&self) -> &str {
        "QuantumActivation"
    }
}

/// Sequential container for quantum modules
pub struct QuantumSequential {
    /// Ordered modules
    modules: Vec<Box<dyn QuantumModule>>,
    /// Training mode
    training: bool,
}

impl QuantumSequential {
    /// Create new sequential container
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Add module to sequence
    pub fn add(mut self, module: Box<dyn QuantumModule>) -> Self {
        self.modules.push(module);
        self
    }

    /// Get number of modules
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl QuantumModule for QuantumSequential {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let mut output = input.clone();

        for module in &mut self.modules {
            output = module.forward(&output)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut all_params = Vec::new();

        for module in &self.modules {
            all_params.extend(module.parameters());
        }

        all_params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
        for module in &mut self.modules {
            module.train(mode);
        }
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
    }

    fn name(&self) -> &str {
        "QuantumSequential"
    }
}

/// Loss functions for quantum ML
pub trait QuantumLoss {
    /// Compute loss
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array>;

    /// Loss function name
    fn name(&self) -> &str;
}

/// Mean Squared Error loss
pub struct QuantumMSELoss;

impl QuantumLoss for QuantumMSELoss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let diff = predictions.data.clone() - &targets.data;
        let squared_diff = &diff * &diff;
        let mse = squared_diff.mean().unwrap();

        let loss_data = ArrayD::from_elem(IxDyn(&[]), mse);
        Ok(SciRS2Array::new(loss_data, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "MSELoss"
    }
}

/// Cross Entropy loss
pub struct QuantumCrossEntropyLoss;

impl QuantumLoss for QuantumCrossEntropyLoss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        // Compute softmax of predictions
        let max_val = predictions
            .data
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_preds = predictions.data.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_preds.sum();
        let softmax = exp_preds.mapv(|x| x / sum_exp);

        // Compute cross entropy
        let log_softmax = softmax.mapv(|x| x.ln());
        let cross_entropy = -(&targets.data * &log_softmax).sum();

        let loss_data = ArrayD::from_elem(IxDyn(&[]), cross_entropy);
        Ok(SciRS2Array::new(loss_data, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }
}

/// Training utilities
pub struct QuantumTrainer {
    /// Model to train
    model: Box<dyn QuantumModule>,
    /// Optimizer
    optimizer: SciRS2Optimizer,
    /// Loss function
    loss_fn: Box<dyn QuantumLoss>,
    /// Training history
    history: TrainingHistory,
}

/// Training history
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Loss values per epoch
    pub losses: Vec<f64>,
    /// Accuracy values per epoch (if applicable)
    pub accuracies: Vec<f64>,
    /// Validation losses
    pub val_losses: Vec<f64>,
    /// Validation accuracies
    pub val_accuracies: Vec<f64>,
}

impl TrainingHistory {
    /// Create new training history
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            accuracies: Vec::new(),
            val_losses: Vec::new(),
            val_accuracies: Vec::new(),
        }
    }

    /// Add training metrics
    pub fn add_training(&mut self, loss: f64, accuracy: Option<f64>) {
        self.losses.push(loss);
        if let Some(acc) = accuracy {
            self.accuracies.push(acc);
        }
    }

    /// Add validation metrics
    pub fn add_validation(&mut self, loss: f64, accuracy: Option<f64>) {
        self.val_losses.push(loss);
        if let Some(acc) = accuracy {
            self.val_accuracies.push(acc);
        }
    }
}

impl QuantumTrainer {
    /// Create new trainer
    pub fn new(
        model: Box<dyn QuantumModule>,
        optimizer: SciRS2Optimizer,
        loss_fn: Box<dyn QuantumLoss>,
    ) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            history: TrainingHistory::new(),
        }
    }

    /// Train for one epoch
    pub fn train_epoch(&mut self, dataloader: &mut dyn DataLoader) -> Result<f64> {
        self.model.train(true);
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some((inputs, targets)) = dataloader.next_batch()? {
            // Zero gradients
            self.model.zero_grad();

            // Forward pass
            let predictions = self.model.forward(&inputs)?;

            // Compute loss
            let loss = self.loss_fn.forward(&predictions, &targets)?;
            total_loss += loss.data[[0]];

            // Backward pass
            // loss.backward()?; // Would implement backpropagation

            // Optimizer step
            let mut params = HashMap::new();
            for (i, param) in self.model.parameters().iter().enumerate() {
                params.insert(format!("param_{}", i), param.data.clone());
            }
            self.optimizer.step(&mut params)?;

            num_batches += 1;
        }

        let avg_loss = total_loss / num_batches as f64;
        self.history.add_training(avg_loss, None);
        Ok(avg_loss)
    }

    /// Evaluate on validation set
    pub fn evaluate(&mut self, dataloader: &mut dyn DataLoader) -> Result<f64> {
        self.model.train(false);
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some((inputs, targets)) = dataloader.next_batch()? {
            // Forward pass (no gradients)
            let predictions = self.model.forward(&inputs)?;

            // Compute loss
            let loss = self.loss_fn.forward(&predictions, &targets)?;
            total_loss += loss.data[[0]];

            num_batches += 1;
        }

        let avg_loss = total_loss / num_batches as f64;
        self.history.add_validation(avg_loss, None);
        Ok(avg_loss)
    }

    /// Get training history
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }
}

/// Data loader trait
pub trait DataLoader {
    /// Get next batch
    fn next_batch(&mut self) -> Result<Option<(SciRS2Array, SciRS2Array)>>;

    /// Reset to beginning
    fn reset(&mut self);

    /// Get batch size
    fn batch_size(&self) -> usize;
}

/// Simple in-memory data loader
pub struct MemoryDataLoader {
    /// Input data
    inputs: SciRS2Array,
    /// Target data
    targets: SciRS2Array,
    /// Batch size
    batch_size: usize,
    /// Current position
    current_pos: usize,
    /// Shuffle data
    shuffle: bool,
    /// Indices for shuffling
    indices: Vec<usize>,
}

impl MemoryDataLoader {
    /// Create new memory data loader
    pub fn new(
        inputs: SciRS2Array,
        targets: SciRS2Array,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self> {
        let num_samples = inputs.data.shape()[0];
        if targets.data.shape()[0] != num_samples {
            return Err(MLError::InvalidConfiguration(
                "Input and target batch sizes don't match".to_string(),
            ));
        }

        let indices: Vec<usize> = (0..num_samples).collect();

        Ok(Self {
            inputs,
            targets,
            batch_size,
            current_pos: 0,
            shuffle,
            indices,
        })
    }

    /// Shuffle indices
    fn shuffle_indices(&mut self) {
        if self.shuffle {
            // Simple shuffle using Fisher-Yates
            for i in (1..self.indices.len()).rev() {
                let j = fastrand::usize(0..=i);
                self.indices.swap(i, j);
            }
        }
    }
}

impl DataLoader for MemoryDataLoader {
    fn next_batch(&mut self) -> Result<Option<(SciRS2Array, SciRS2Array)>> {
        if self.current_pos >= self.indices.len() {
            return Ok(None);
        }

        let end_pos = (self.current_pos + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_pos..end_pos];

        // Extract batch data (simplified - would use proper indexing)
        let batch_inputs = self.inputs.clone(); // Placeholder
        let batch_targets = self.targets.clone(); // Placeholder

        self.current_pos = end_pos;

        Ok(Some((batch_inputs, batch_targets)))
    }

    fn reset(&mut self) {
        self.current_pos = 0;
        self.shuffle_indices();
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Utility functions for building quantum models
pub mod quantum_nn {
    use super::*;

    /// Create a simple quantum feedforward network
    pub fn create_feedforward(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        activation: ActivationType,
    ) -> Result<QuantumSequential> {
        let mut model = QuantumSequential::new();

        let mut prev_size = input_size;

        // Hidden layers
        for &hidden_size in hidden_sizes {
            model = model.add(Box::new(
                QuantumLinear::new(prev_size, hidden_size)?.with_bias()?,
            ));
            model = model.add(Box::new(QuantumActivation::new(activation.clone())));
            prev_size = hidden_size;
        }

        // Output layer
        model = model.add(Box::new(
            QuantumLinear::new(prev_size, output_size)?.with_bias()?,
        ));

        Ok(model)
    }

    /// Create quantum CNN
    pub fn create_cnn(input_channels: usize, num_classes: usize) -> Result<QuantumSequential> {
        let model = QuantumSequential::new()
            .add(Box::new(
                QuantumConv2d::new(input_channels, 32, (3, 3))?.with_bias()?,
            ))
            .add(Box::new(QuantumActivation::relu()))
            .add(Box::new(QuantumConv2d::new(32, 64, (3, 3))?.with_bias()?))
            .add(Box::new(QuantumActivation::relu()))
            .add(Box::new(QuantumLinear::new(64, num_classes)?.with_bias()?));

        Ok(model)
    }

    /// Initialize model parameters
    pub fn init_parameters(model: &mut dyn QuantumModule, init_type: InitType) -> Result<()> {
        for mut param in model.parameters() {
            match init_type {
                InitType::Xavier => {
                    // Xavier/Glorot initialization
                    let fan_in = param.shape().iter().rev().skip(1).product::<usize>() as f64;
                    let fan_out = param.shape()[0] as f64;
                    let bound = (6.0 / (fan_in + fan_out)).sqrt();

                    for elem in param.data.data.iter_mut() {
                        *elem = (fastrand::f64() * 2.0 - 1.0) * bound;
                    }
                }
                InitType::He => {
                    // He initialization
                    let fan_in = param.shape().iter().rev().skip(1).product::<usize>() as f64;
                    let std = (2.0 / fan_in).sqrt();

                    for elem in param.data.data.iter_mut() {
                        *elem = fastrand::f64() * std;
                    }
                }
                InitType::Normal(mean, std) => {
                    // Normal initialization
                    for elem in param.data.data.iter_mut() {
                        *elem = mean + std * fastrand::f64();
                    }
                }
                InitType::Uniform(low, high) => {
                    // Uniform initialization
                    for elem in param.data.data.iter_mut() {
                        *elem = low + (high - low) * fastrand::f64();
                    }
                }
            }
        }
        Ok(())
    }
}

/// Parameter initialization types
#[derive(Debug, Clone, Copy)]
pub enum InitType {
    /// Xavier/Glorot initialization
    Xavier,
    /// He initialization
    He,
    /// Normal distribution
    Normal(f64, f64), // mean, std
    /// Uniform distribution
    Uniform(f64, f64), // low, high
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_linear() {
        let mut linear = QuantumLinear::new(4, 2).unwrap();
        assert_eq!(linear.in_features, 4);
        assert_eq!(linear.out_features, 2);
        assert_eq!(linear.parameters().len(), 1); // weights only

        let linear_with_bias = linear.with_bias().unwrap();
        // Would have 2 parameters: weights and bias
    }

    #[test]
    fn test_quantum_sequential() {
        let model = QuantumSequential::new()
            .add(Box::new(QuantumLinear::new(4, 8).unwrap()))
            .add(Box::new(QuantumActivation::relu()))
            .add(Box::new(QuantumLinear::new(8, 2).unwrap()));

        assert_eq!(model.len(), 3);
        assert!(!model.is_empty());
    }

    #[test]
    fn test_quantum_activation() {
        let mut relu = QuantumActivation::relu();
        let input_data = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 1.0]).unwrap();
        let input = SciRS2Array::new(input_data, false);

        let output = relu.forward(&input).unwrap();
        assert_eq!(output.data[[0]], 0.0); // ReLU(-1) = 0
        assert_eq!(output.data[[1]], 1.0); // ReLU(1) = 1
    }

    #[test]
    #[ignore]
    fn test_quantum_loss() {
        let mse_loss = QuantumMSELoss;

        let pred_data = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let target_data = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.5, 1.8]).unwrap();

        let predictions = SciRS2Array::new(pred_data, false);
        let targets = SciRS2Array::new(target_data, false);

        let loss = mse_loss.forward(&predictions, &targets).unwrap();
        assert!(loss.data[[0]] > 0.0); // Should have positive loss
    }

    #[test]
    fn test_parameter() {
        let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
        let param = Parameter::new(SciRS2Array::new(data, true), "test_param");

        assert_eq!(param.name, "test_param");
        assert!(param.requires_grad);
        assert_eq!(param.shape(), &[2, 3]);
        assert_eq!(param.numel(), 6);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();
        history.add_training(0.5, Some(0.8));
        history.add_validation(0.6, Some(0.7));

        assert_eq!(history.losses.len(), 1);
        assert_eq!(history.accuracies.len(), 1);
        assert_eq!(history.val_losses.len(), 1);
        assert_eq!(history.val_accuracies.len(), 1);
    }
}
