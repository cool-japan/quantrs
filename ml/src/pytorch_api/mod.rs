//! PyTorch-like API for quantum machine learning models
//!
//! This module provides a familiar PyTorch-style interface for building,
//! training, and deploying quantum ML models, making it easier for classical
//! ML practitioners to adopt quantum algorithms.

mod conv;
mod data;
mod layers;
mod loss;
mod rnn;
mod schedulers;
mod transformer;

pub use conv::*;
pub use data::*;
pub use layers::*;
pub use loss::*;
pub use rnn::*;
pub use schedulers::*;
pub use transformer::*;

use crate::error::{MLError, Result};
use crate::scirs2_integration::{SciRS2Array, SciRS2Optimizer};
use scirs2_core::ndarray::{ArrayD, IxDyn};

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

impl Default for QuantumSequential {
    fn default() -> Self {
        Self::new()
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

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
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
    pub fn train_epoch<D: DataLoader>(&mut self, dataloader: &mut D) -> Result<f64> {
        self.model.train(true);

        let mut epoch_loss = 0.0;
        let mut batches = 0;

        while let Some((inputs, targets)) = dataloader.next_batch()? {
            // Zero gradients
            self.model.zero_grad();

            // Forward pass
            let predictions = self.model.forward(&inputs)?;

            // Compute loss
            let loss = self.loss_fn.forward(&predictions, &targets)?;
            let loss_val = loss.data.iter().next().copied().unwrap_or(0.0);

            epoch_loss += loss_val;
            batches += 1;
        }

        let avg_loss = if batches > 0 {
            epoch_loss / batches as f64
        } else {
            0.0
        };
        self.history.add_training(avg_loss, None);

        Ok(avg_loss)
    }

    /// Evaluate model
    pub fn evaluate<D: DataLoader>(&mut self, dataloader: &mut D) -> Result<f64> {
        self.model.train(false);

        let mut total_loss = 0.0;
        let mut batches = 0;

        while let Some((inputs, targets)) = dataloader.next_batch()? {
            let predictions = self.model.forward(&inputs)?;
            let loss = self.loss_fn.forward(&predictions, &targets)?;
            let loss_val = loss.data.iter().next().copied().unwrap_or(0.0);

            total_loss += loss_val;
            batches += 1;
        }

        let avg_loss = if batches > 0 {
            total_loss / batches as f64
        } else {
            0.0
        };

        Ok(avg_loss)
    }

    /// Get training history
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_linear() {
        let linear = QuantumLinear::new(4, 2).expect("QuantumLinear creation should succeed");
        assert_eq!(linear.in_features, 4);
        assert_eq!(linear.out_features, 2);
        assert_eq!(linear.parameters().len(), 1); // weights only

        let _linear_with_bias = linear.with_bias().expect("Adding bias should succeed");
        // Would have 2 parameters: weights and bias
    }

    #[test]
    fn test_quantum_sequential() {
        let model = QuantumSequential::new()
            .add(Box::new(
                QuantumLinear::new(4, 8).expect("QuantumLinear creation should succeed"),
            ))
            .add(Box::new(QuantumActivation::relu()))
            .add(Box::new(
                QuantumLinear::new(8, 2).expect("QuantumLinear creation should succeed"),
            ));

        assert_eq!(model.len(), 3);
        assert!(!model.is_empty());
    }

    #[test]
    fn test_quantum_activation() {
        let mut relu = QuantumActivation::relu();
        let input_data = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 1.0])
            .expect("Valid shape for input data");
        let input = SciRS2Array::new(input_data, false);

        let output = relu.forward(&input).expect("Forward pass should succeed");
        assert_eq!(output.data[[0]], 0.0); // ReLU(-1) = 0
        assert_eq!(output.data[[1]], 1.0); // ReLU(1) = 1
    }

    #[test]
    #[ignore]
    fn test_quantum_loss() {
        let mse_loss = QuantumMSELoss;

        let pred_data = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0])
            .expect("Valid shape for predictions");
        let target_data =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.5, 1.8]).expect("Valid shape for targets");

        let predictions = SciRS2Array::new(pred_data, false);
        let targets = SciRS2Array::new(target_data, false);

        let loss = mse_loss
            .forward(&predictions, &targets)
            .expect("Loss computation should succeed");
        assert!(loss.data[[0]] > 0.0); // Should have positive loss
    }

    #[test]
    fn test_parameter() {
        let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6])
            .expect("Valid shape for parameter data");
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
