//! Sklearn-compatible regressors

use super::{SklearnEstimator, SklearnRegressor};
use crate::error::{MLError, Result};
use crate::qnn::{QNNBuilder, QuantumNeuralNetwork};
use crate::simulator_backends::{SimulatorBackend, StatevectorBackend};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;

/// Quantum MLP Regressor (sklearn-compatible)
pub struct QuantumMLPRegressor {
    /// Internal QNN
    qnn: Option<QuantumNeuralNetwork>,
    /// Network configuration
    hidden_layer_sizes: Vec<usize>,
    /// Activation function
    activation: String,
    /// Solver
    solver: String,
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Random state
    random_state: Option<u64>,
    /// Backend
    backend: Arc<dyn SimulatorBackend>,
    /// Fitted flag
    fitted: bool,
}

impl QuantumMLPRegressor {
    /// Create new Quantum MLP Regressor
    pub fn new() -> Self {
        Self {
            qnn: None,
            hidden_layer_sizes: vec![10],
            activation: "relu".to_string(),
            solver: "adam".to_string(),
            learning_rate: 0.001,
            max_iter: 200,
            random_state: None,
            backend: Arc::new(StatevectorBackend::new(10)),
            fitted: false,
        }
    }

    /// Set hidden layer sizes
    pub fn set_hidden_layer_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_layer_sizes = sizes;
        self
    }

    /// Set learning rate
    pub fn set_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum iterations
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl Default for QuantumMLPRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl SklearnEstimator for QuantumMLPRegressor {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()> {
        let y = y.ok_or_else(|| {
            MLError::InvalidConfiguration("Target values required for regression".to_string())
        })?;

        // Build QNN for regression
        let _input_size = X.ncols();
        let output_size = 1; // Single output for regression

        let mut builder = QNNBuilder::new();

        // Add hidden layers
        for &size in &self.hidden_layer_sizes {
            builder = builder.add_layer(size);
        }

        // Add output layer
        builder = builder.add_layer(output_size);

        let mut qnn = builder.build()?;

        // Reshape target for training
        let y_reshaped = y.clone().into_shape((y.len(), 1)).map_err(|e| {
            MLError::InvalidConfiguration(format!("Failed to reshape target: {}", e))
        })?;

        // Train QNN
        qnn.train(X, &y_reshaped, self.max_iter, self.learning_rate)?;

        self.qnn = Some(qnn);
        self.fitted = true;

        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "hidden_layer_sizes".to_string(),
            format!("{:?}", self.hidden_layer_sizes),
        );
        params.insert("activation".to_string(), self.activation.clone());
        params.insert("solver".to_string(), self.solver.clone());
        params.insert("learning_rate".to_string(), self.learning_rate.to_string());
        params.insert("max_iter".to_string(), self.max_iter.to_string());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "learning_rate" => {
                    self.learning_rate = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid learning_rate: {}", value))
                    })?;
                }
                "max_iter" => {
                    self.max_iter = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid max_iter: {}", value))
                    })?;
                }
                "activation" => {
                    self.activation = value;
                }
                "solver" => {
                    self.solver = value;
                }
                _ => {
                    // Skip unknown parameters
                }
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnRegressor for QuantumMLPRegressor {
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }

        let qnn = self
            .qnn
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QNN model not initialized".to_string()))?;
        let predictions = qnn.predict_batch(X)?;

        // Extract single column for regression
        Ok(predictions.column(0).to_owned())
    }
}
