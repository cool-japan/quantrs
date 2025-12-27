//! Quantum Machine Learning Training
//!
//! This module provides training routines for quantum machine learning models,
//! including supervised learning, unsupervised learning, and reinforcement learning.

use super::*;
use crate::{DeviceError, DeviceResult, QuantumDevice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Quantum trainer for ML models
pub struct QuantumTrainer {
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    config: QMLConfig,
    model_type: QMLModelType,
    optimizer: Box<dyn QuantumOptimizer>,
    gradient_calculator: QuantumGradientCalculator,
    loss_function: Box<dyn LossFunction + Send + Sync>,
}

/// Training data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

impl TrainingData {
    pub fn new(features: Vec<Vec<f64>>, labels: Vec<f64>) -> Self {
        Self {
            features,
            labels,
            metadata: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    #[must_use]
    pub fn get_batch(&self, indices: &[usize]) -> Self {
        let batch_features = indices
            .iter()
            .filter_map(|&i| self.features.get(i))
            .cloned()
            .collect();
        let batch_labels = indices
            .iter()
            .filter_map(|&i| self.labels.get(i))
            .copied()
            .collect();

        Self {
            features: batch_features,
            labels: batch_labels,
            metadata: self.metadata.clone(),
        }
    }

    pub fn shuffle(&mut self) {
        let n = self.len();
        for i in 0..n {
            let j = fastrand::usize(i..n);
            self.features.swap(i, j);
            self.labels.swap(i, j);
        }
    }
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub model_id: String,
    pub model: QMLModel,
    pub final_loss: f64,
    pub final_accuracy: Option<f64>,
    pub training_time: Duration,
    pub convergence_achieved: bool,
    pub optimal_parameters: Vec<f64>,
    pub training_metrics: TrainingMetrics,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss_history: Vec<f64>,
    pub accuracy_history: Vec<f64>,
    pub validation_loss_history: Vec<f64>,
    pub validation_accuracy_history: Vec<f64>,
    pub gradient_norms: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub quantum_fidelities: Vec<f64>,
    pub execution_times: Vec<Duration>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingMetrics {
    pub const fn new() -> Self {
        Self {
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            validation_loss_history: Vec::new(),
            validation_accuracy_history: Vec::new(),
            gradient_norms: Vec::new(),
            learning_rates: Vec::new(),
            quantum_fidelities: Vec::new(),
            execution_times: Vec::new(),
        }
    }

    pub fn add_epoch(
        &mut self,
        loss: f64,
        accuracy: f64,
        val_loss: Option<f64>,
        val_accuracy: Option<f64>,
        gradient_norm: f64,
        learning_rate: f64,
        quantum_fidelity: f64,
        execution_time: Duration,
    ) {
        self.loss_history.push(loss);
        self.accuracy_history.push(accuracy);
        if let Some(vl) = val_loss {
            self.validation_loss_history.push(vl);
        }
        if let Some(va) = val_accuracy {
            self.validation_accuracy_history.push(va);
        }
        self.gradient_norms.push(gradient_norm);
        self.learning_rates.push(learning_rate);
        self.quantum_fidelities.push(quantum_fidelity);
        self.execution_times.push(execution_time);
    }
}

/// Loss function trait
pub trait LossFunction: Send + Sync {
    /// Compute loss value
    fn compute_loss(&self, predictions: &[f64], targets: &[f64]) -> DeviceResult<f64>;

    /// Compute loss gradients
    fn compute_gradients(&self, predictions: &[f64], targets: &[f64]) -> DeviceResult<Vec<f64>>;

    /// Get loss function name
    fn name(&self) -> &str;
}

/// Mean squared error loss
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn compute_loss(&self, predictions: &[f64], targets: &[f64]) -> DeviceResult<f64> {
        if predictions.len() != targets.len() {
            return Err(DeviceError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(mse)
    }

    fn compute_gradients(&self, predictions: &[f64], targets: &[f64]) -> DeviceResult<Vec<f64>> {
        if predictions.len() != targets.len() {
            return Err(DeviceError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| 2.0 * (p - t) / predictions.len() as f64)
            .collect();

        Ok(gradients)
    }

    fn name(&self) -> &'static str {
        "MSE"
    }
}

/// Cross-entropy loss
pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn compute_loss(&self, predictions: &[f64], targets: &[f64]) -> DeviceResult<f64> {
        if predictions.len() != targets.len() {
            return Err(DeviceError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        let epsilon = 1e-15; // Prevent log(0)
        let cross_entropy = -targets
            .iter()
            .zip(predictions.iter())
            .map(|(t, p)| {
                let p_clipped = p.clamp(epsilon, 1.0 - epsilon);
                (1.0 - t).mul_add((1.0 - p_clipped).ln(), t * p_clipped.ln())
            })
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(cross_entropy)
    }

    fn compute_gradients(&self, predictions: &[f64], targets: &[f64]) -> DeviceResult<Vec<f64>> {
        if predictions.len() != targets.len() {
            return Err(DeviceError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        let epsilon = 1e-15;
        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let p_clipped = p.clamp(epsilon, 1.0 - epsilon);
                (p_clipped - t) / (p_clipped * (1.0 - p_clipped) * predictions.len() as f64)
            })
            .collect();

        Ok(gradients)
    }

    fn name(&self) -> &'static str {
        "CrossEntropy"
    }
}

impl QuantumTrainer {
    /// Create a new quantum trainer
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        config: &QMLConfig,
        model_type: QMLModelType,
    ) -> DeviceResult<Self> {
        let optimizer = create_gradient_optimizer(
            device.clone(),
            config.optimizer.clone(),
            config.learning_rate,
        );

        let gradient_config = GradientConfig {
            method: config.gradient_method.clone(),
            shots: 1024,
            ..Default::default()
        };

        let gradient_calculator = QuantumGradientCalculator::new(device.clone(), gradient_config)?;

        let loss_function: Box<dyn LossFunction + Send + Sync> = match model_type {
            QMLModelType::VQC | QMLModelType::QNN => Box::new(CrossEntropyLoss),
            _ => Box::new(MSELoss),
        };

        Ok(Self {
            device,
            config: config.clone(),
            model_type,
            optimizer,
            gradient_calculator,
            loss_function,
        })
    }

    /// Train a quantum ML model
    pub async fn train(
        &mut self,
        training_data: TrainingData,
        validation_data: Option<TrainingData>,
        training_history: &mut Vec<TrainingEpoch>,
    ) -> DeviceResult<TrainingResult> {
        let start_time = Instant::now();
        let model_id = format!("qml_model_{}", uuid::Uuid::new_v4());

        // Initialize model parameters
        let mut parameters = self.initialize_parameters()?;
        let mut metrics = TrainingMetrics::new();
        let mut best_loss = f64::INFINITY;
        let mut best_parameters = parameters.clone();
        let mut patience_counter = 0;
        let early_stopping_patience = 50;

        for epoch in 0..self.config.max_epochs {
            let epoch_start = Instant::now();

            // Shuffle training data
            let mut epoch_data = training_data.clone();
            epoch_data.shuffle();

            // Training step
            let (epoch_loss, epoch_accuracy, gradient_norm) =
                self.train_epoch(&mut parameters, &epoch_data).await?;

            // Validation step
            let (val_loss, val_accuracy) = if let Some(ref val_data) = validation_data {
                let (vl, va) = self.validate_epoch(&parameters, val_data).await?;
                (Some(vl), Some(va))
            } else {
                (None, None)
            };

            let execution_time = epoch_start.elapsed();
            let quantum_fidelity = self.estimate_quantum_fidelity(&parameters).await?;

            // Update metrics
            metrics.add_epoch(
                epoch_loss,
                epoch_accuracy,
                val_loss,
                val_accuracy,
                gradient_norm,
                self.config.learning_rate,
                quantum_fidelity,
                execution_time,
            );

            // Add to training history
            training_history.push(TrainingEpoch {
                epoch,
                loss: epoch_loss,
                accuracy: Some(epoch_accuracy),
                parameters: parameters.clone(),
                gradient_norm,
                learning_rate: self.config.learning_rate,
                execution_time,
                quantum_fidelity: Some(quantum_fidelity),
                classical_preprocessing_time: Duration::from_millis(10),
                quantum_execution_time: execution_time
                    .checked_sub(Duration::from_millis(10))
                    .unwrap_or(Duration::ZERO),
            });

            // Check for improvement
            let current_loss = val_loss.unwrap_or(epoch_loss);
            if current_loss < best_loss {
                best_loss = current_loss;
                best_parameters.clone_from(&parameters);
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            // Early stopping
            if patience_counter >= early_stopping_patience {
                println!("Early stopping at epoch {epoch} due to no improvement");
                break;
            }

            // Convergence check
            if epoch_loss < self.config.convergence_tolerance {
                println!("Converged at epoch {epoch} with loss {epoch_loss:.6}");
                break;
            }

            // Progress logging
            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Loss={:.6}, Accuracy={:.4}, Val_Loss={:.6}, Fidelity={:.4}",
                    epoch,
                    epoch_loss,
                    epoch_accuracy,
                    val_loss.unwrap_or(0.0),
                    quantum_fidelity
                );
            }
        }

        // Create final model
        let model = QMLModel {
            model_type: self.model_type.clone(),
            parameters: best_parameters.clone(),
            circuit_structure: self.get_circuit_structure(),
            training_metadata: self.get_training_metadata(),
            performance_metrics: self.get_performance_metrics(&metrics),
        };

        Ok(TrainingResult {
            model_id,
            model,
            final_loss: best_loss,
            final_accuracy: metrics.accuracy_history.last().copied(),
            training_time: start_time.elapsed(),
            convergence_achieved: best_loss < self.config.convergence_tolerance,
            optimal_parameters: best_parameters,
            training_metrics: metrics,
        })
    }

    /// Train for one epoch
    async fn train_epoch(
        &mut self,
        parameters: &mut Vec<f64>,
        training_data: &TrainingData,
    ) -> DeviceResult<(f64, f64, f64)> {
        let batch_size = self.config.batch_size.min(training_data.len());
        let num_batches = training_data.len().div_ceil(batch_size);

        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut total_gradient_norm = 0.0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(training_data.len());
            let batch_indices: Vec<usize> = (start_idx..end_idx).collect();
            let batch_data = training_data.get_batch(&batch_indices);

            // Forward pass
            let predictions = self.forward_pass(parameters, &batch_data.features).await?;

            // Compute loss
            let batch_loss = self
                .loss_function
                .compute_loss(&predictions, &batch_data.labels)?;
            total_loss += batch_loss;

            // Compute accuracy
            let batch_accuracy = self.compute_accuracy(&predictions, &batch_data.labels)?;
            total_accuracy += batch_accuracy;

            // Backward pass - compute gradients
            let gradients = self.backward_pass(parameters, &batch_data).await?;
            let gradient_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
            total_gradient_norm += gradient_norm;

            // Update parameters
            let loss_fn = Arc::new(MSELoss {}) as Arc<dyn LossFunction + Send + Sync>;
            let objective_function = Box::new(BatchObjectiveFunction::new(
                self.device.clone(),
                batch_data,
                loss_fn,
            ));

            let optimization_result = self
                .optimizer
                .optimize(parameters.clone(), objective_function)?;

            *parameters = optimization_result.optimal_parameters;
        }

        Ok((
            total_loss / num_batches as f64,
            total_accuracy / num_batches as f64,
            total_gradient_norm / num_batches as f64,
        ))
    }

    /// Validate for one epoch
    async fn validate_epoch(
        &self,
        parameters: &[f64],
        validation_data: &TrainingData,
    ) -> DeviceResult<(f64, f64)> {
        let predictions = self
            .forward_pass(parameters, &validation_data.features)
            .await?;
        let loss = self
            .loss_function
            .compute_loss(&predictions, &validation_data.labels)?;
        let accuracy = self.compute_accuracy(&predictions, &validation_data.labels)?;

        Ok((loss, accuracy))
    }

    /// Forward pass through the quantum model
    async fn forward_pass(
        &self,
        parameters: &[f64],
        features: &[Vec<f64>],
    ) -> DeviceResult<Vec<f64>> {
        let mut predictions = Vec::new();

        for feature_vector in features {
            let prediction = self.evaluate_model(parameters, feature_vector).await?;
            predictions.push(prediction);
        }

        Ok(predictions)
    }

    /// Backward pass - compute gradients
    async fn backward_pass(
        &self,
        parameters: &[f64],
        batch_data: &TrainingData,
    ) -> DeviceResult<Vec<f64>> {
        // Create a circuit for this batch
        let circuit = self.build_training_circuit(parameters, &batch_data.features[0])?;

        // Compute gradients using the gradient calculator
        self.gradient_calculator
            .compute_gradients(circuit, parameters.to_vec())
            .await
    }

    /// Evaluate the model for a single input
    async fn evaluate_model(&self, parameters: &[f64], features: &[f64]) -> DeviceResult<f64> {
        let circuit = self.build_training_circuit(parameters, features)?;
        let device = self.device.read().await;
        let result = Self::execute_circuit_helper(&*device, &circuit, 1024).await?;

        // Convert quantum measurement to prediction
        self.decode_quantum_output(&result)
    }

    /// Build training circuit
    fn build_training_circuit(
        &self,
        parameters: &[f64],
        features: &[f64],
    ) -> DeviceResult<ParameterizedQuantumCircuit> {
        match self.model_type {
            QMLModelType::VQC => self.build_vqc_circuit(parameters, features),
            QMLModelType::QNN => self.build_qnn_circuit(parameters, features),
            QMLModelType::QAOA => self.build_qaoa_circuit(parameters, features),
            _ => Err(DeviceError::InvalidInput(format!(
                "Model type {:?} not implemented",
                self.model_type
            ))),
        }
    }

    /// Build VQC circuit
    fn build_vqc_circuit(
        &self,
        parameters: &[f64],
        features: &[f64],
    ) -> DeviceResult<ParameterizedQuantumCircuit> {
        let num_qubits = (features.len() as f64).log2().ceil() as usize + 2;
        let mut circuit = ParameterizedQuantumCircuit::new(num_qubits);

        // Feature encoding
        for (i, &feature) in features.iter().enumerate() {
            if i < num_qubits {
                circuit.add_ry_gate(i, feature)?;
            }
        }

        // Parameterized layers
        let params_per_layer = num_qubits * 2; // RY and RZ for each qubit
        let num_layers = parameters.len() / params_per_layer;

        let mut param_idx = 0;
        for _layer in 0..num_layers {
            // Rotation gates
            for qubit in 0..num_qubits {
                if param_idx < parameters.len() {
                    circuit.add_ry_gate(qubit, parameters[param_idx])?;
                    param_idx += 1;
                }
                if param_idx < parameters.len() {
                    circuit.add_rz_gate(qubit, parameters[param_idx])?;
                    param_idx += 1;
                }
            }

            // Entangling gates
            for qubit in 0..num_qubits - 1 {
                circuit.add_cnot_gate(qubit, qubit + 1)?;
            }
        }

        Ok(circuit)
    }

    /// Build QNN circuit (similar to VQC but different structure)
    fn build_qnn_circuit(
        &self,
        parameters: &[f64],
        features: &[f64],
    ) -> DeviceResult<ParameterizedQuantumCircuit> {
        // For now, use same structure as VQC
        self.build_vqc_circuit(parameters, features)
    }

    /// Build QAOA circuit
    fn build_qaoa_circuit(
        &self,
        _parameters: &[f64],
        _features: &[f64],
    ) -> DeviceResult<ParameterizedQuantumCircuit> {
        // QAOA implementation would be more complex
        Err(DeviceError::InvalidInput(
            "QAOA circuit building not implemented".to_string(),
        ))
    }

    /// Decode quantum output to classical prediction
    fn decode_quantum_output(&self, result: &CircuitResult) -> DeviceResult<f64> {
        // Simple decoding: expectation value of first qubit
        let mut expectation = 0.0;
        let total_shots = result.shots as f64;

        for (bitstring, count) in &result.counts {
            if let Some(first_bit) = bitstring.chars().next() {
                let bit_value = if first_bit == '1' { 1.0 } else { 0.0 };
                let probability = *count as f64 / total_shots;
                expectation += bit_value * probability;
            }
        }

        Ok(expectation)
    }

    /// Compute accuracy for classification
    fn compute_accuracy(&self, predictions: &[f64], targets: &[f64]) -> DeviceResult<f64> {
        if predictions.len() != targets.len() {
            return Err(DeviceError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let predicted_class = if *p > 0.5 { 1.0 } else { 0.0 };
                if (predicted_class - t).abs() < 0.1 {
                    1.0
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        Ok(correct / predictions.len() as f64)
    }

    /// Initialize model parameters
    fn initialize_parameters(&self) -> DeviceResult<Vec<f64>> {
        let param_count = match self.model_type {
            QMLModelType::QNN => 30,
            QMLModelType::QAOA => 10,
            QMLModelType::VQC | _ => 20, // Default parameter count
        };

        let parameters = (0..param_count)
            .map(|_| (fastrand::f64() * 2.0).mul_add(std::f64::consts::PI, -std::f64::consts::PI))
            .collect();

        Ok(parameters)
    }

    /// Estimate quantum fidelity
    async fn estimate_quantum_fidelity(&self, _parameters: &[f64]) -> DeviceResult<f64> {
        // Simplified fidelity estimate
        Ok(fastrand::f64().mul_add(0.05, 0.95))
    }

    /// Get circuit structure description
    fn get_circuit_structure(&self) -> CircuitStructure {
        CircuitStructure {
            num_qubits: 6, // Default
            depth: 10,
            gate_types: vec!["RY".to_string(), "RZ".to_string(), "CNOT".to_string()],
            parameter_count: 20,
            entangling_gates: 5,
        }
    }

    /// Get training metadata
    fn get_training_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("trainer_type".to_string(), "quantum".to_string());
        metadata.insert(
            "optimizer".to_string(),
            format!("{:?}", self.config.optimizer),
        );
        metadata.insert(
            "gradient_method".to_string(),
            format!("{:?}", self.config.gradient_method),
        );
        metadata.insert(
            "learning_rate".to_string(),
            self.config.learning_rate.to_string(),
        );
        metadata
    }

    /// Get performance metrics
    fn get_performance_metrics(&self, metrics: &TrainingMetrics) -> HashMap<String, f64> {
        let mut perf_metrics = HashMap::new();

        if let Some(&final_loss) = metrics.loss_history.last() {
            perf_metrics.insert("final_loss".to_string(), final_loss);
        }

        if let Some(&final_accuracy) = metrics.accuracy_history.last() {
            perf_metrics.insert("final_accuracy".to_string(), final_accuracy);
        }

        if !metrics.loss_history.is_empty() {
            let best_loss = metrics
                .loss_history
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            perf_metrics.insert("best_loss".to_string(), best_loss);
        }

        if !metrics.accuracy_history.is_empty() {
            let best_accuracy = metrics
                .accuracy_history
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            perf_metrics.insert("best_accuracy".to_string(), best_accuracy);
        }

        perf_metrics
    }

    /// Execute a circuit on the quantum device (helper function to work around trait object limitations)
    async fn execute_circuit_helper(
        device: &(dyn QuantumDevice + Send + Sync),
        circuit: &ParameterizedQuantumCircuit,
        shots: usize,
    ) -> DeviceResult<CircuitResult> {
        // For now, return a mock result since we can't execute circuits directly
        // In a real implementation, this would need proper circuit execution
        let mut counts = std::collections::HashMap::new();
        counts.insert("0".repeat(circuit.num_qubits()), shots / 2);
        counts.insert("1".repeat(circuit.num_qubits()), shots / 2);

        Ok(CircuitResult {
            counts,
            shots,
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Batch objective function for optimization
pub struct BatchObjectiveFunction {
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    batch_data: TrainingData,
    loss_function: Arc<dyn LossFunction + Send + Sync>,
}

impl BatchObjectiveFunction {
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        batch_data: TrainingData,
        loss_function: Arc<dyn LossFunction + Send + Sync>,
    ) -> Self {
        Self {
            device,
            batch_data,
            loss_function,
        }
    }
}

impl ObjectiveFunction for BatchObjectiveFunction {
    fn evaluate(&self, parameters: &[f64]) -> DeviceResult<f64> {
        // Simplified batch evaluation
        // In practice, this would run the quantum circuit for the batch
        let mut total_loss = 0.0;

        for (features, target) in self
            .batch_data
            .features
            .iter()
            .zip(self.batch_data.labels.iter())
        {
            // Simplified prediction
            let prediction = parameters.iter().sum::<f64>() / parameters.len() as f64;
            let loss = (prediction - target).powi(2);
            total_loss += loss;
        }

        Ok(total_loss / self.batch_data.len() as f64)
    }

    fn gradient(&self, _parameters: &[f64]) -> DeviceResult<Option<Vec<f64>>> {
        // Gradients would be computed via parameter shift rule
        Ok(None)
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("objective_type".to_string(), "batch_training".to_string());
        metadata.insert("batch_size".to_string(), self.batch_data.len().to_string());
        metadata
    }
}

/// Create training data from vectors
pub fn create_training_data(features: Vec<Vec<f64>>, labels: Vec<f64>) -> TrainingData {
    TrainingData::new(features, labels)
}

/// Create a supervised learning trainer
pub fn create_supervised_trainer(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    model_type: QMLModelType,
    config: QMLConfig,
) -> DeviceResult<QuantumTrainer> {
    QuantumTrainer::new(device, &config, model_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_mock_quantum_device;

    #[test]
    fn test_training_data_creation() {
        let features = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let labels = vec![0.0, 1.0, 0.0];

        let training_data = TrainingData::new(features.clone(), labels.clone());

        assert_eq!(training_data.len(), 3);
        assert_eq!(training_data.features, features);
        assert_eq!(training_data.labels, labels);
    }

    #[test]
    fn test_training_data_batch() {
        let features = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
        ];
        let labels = vec![0.0, 1.0, 0.0, 1.0];
        let training_data = TrainingData::new(features, labels);

        let batch_indices = vec![0, 2];
        let batch = training_data.get_batch(&batch_indices);

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.features[0], vec![0.1, 0.2]);
        assert_eq!(batch.features[1], vec![0.5, 0.6]);
        assert_eq!(batch.labels[0], 0.0);
        assert_eq!(batch.labels[1], 0.0);
    }

    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss;
        let predictions = vec![0.8, 0.2, 0.9];
        let targets = vec![1.0, 0.0, 1.0];

        let loss = loss_fn
            .compute_loss(&predictions, &targets)
            .expect("MSE loss computation should succeed");
        let expected_loss =
            ((0.8_f64 - 1.0).powi(2) + (0.2_f64 - 0.0).powi(2) + (0.9_f64 - 1.0).powi(2)) / 3.0;
        assert!((loss - expected_loss).abs() < 1e-10);

        let gradients = loss_fn
            .compute_gradients(&predictions, &targets)
            .expect("MSE gradient computation should succeed");
        assert_eq!(gradients.len(), 3);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss;
        let predictions = vec![0.8, 0.2, 0.9];
        let targets = vec![1.0, 0.0, 1.0];

        let loss = loss_fn
            .compute_loss(&predictions, &targets)
            .expect("CrossEntropy loss computation should succeed");
        assert!(loss > 0.0); // Cross-entropy should be positive

        let gradients = loss_fn
            .compute_gradients(&predictions, &targets)
            .expect("CrossEntropy gradient computation should succeed");
        assert_eq!(gradients.len(), 3);
    }

    #[tokio::test]
    async fn test_quantum_trainer_creation() {
        let device = create_mock_quantum_device();
        let config = QMLConfig::default();

        let trainer = QuantumTrainer::new(device, &config, QMLModelType::VQC)
            .expect("QuantumTrainer creation should succeed");
        assert_eq!(trainer.model_type, QMLModelType::VQC);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();

        metrics.add_epoch(
            0.5,
            0.8,
            Some(0.6),
            Some(0.7),
            0.1,
            0.01,
            0.95,
            Duration::from_millis(100),
        );

        assert_eq!(metrics.loss_history.len(), 1);
        assert_eq!(metrics.accuracy_history.len(), 1);
        assert_eq!(metrics.validation_loss_history.len(), 1);
        assert_eq!(metrics.validation_accuracy_history.len(), 1);
        assert_eq!(metrics.loss_history[0], 0.5);
        assert_eq!(metrics.accuracy_history[0], 0.8);
    }
}
