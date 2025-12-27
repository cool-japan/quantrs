//! Quantum Machine Learning Accelerators
//!
//! This module provides quantum machine learning acceleration capabilities,
//! integrating variational quantum algorithms, quantum neural networks,
//! and hybrid quantum-classical optimization routines.

use crate::{CircuitExecutor, CircuitResult, DeviceError, DeviceResult, QuantumDevice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

pub mod classical_integration;
pub mod gradients;
pub mod hardware_acceleration;
pub mod inference;
pub mod optimization;
pub mod quantum_neural_networks;
pub mod training;
pub mod variational_algorithms;

pub use classical_integration::*;
pub use gradients::*;
pub use hardware_acceleration::*;
pub use inference::*;
pub use optimization::*;
pub use quantum_neural_networks::*;
pub use training::*;
pub use variational_algorithms::*;

/// Quantum Machine Learning Accelerator
pub struct QMLAccelerator {
    /// Quantum device backend
    pub device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    /// QML configuration
    pub config: QMLConfig,
    /// Training history
    pub training_history: Vec<TrainingEpoch>,
    /// Model registry
    pub model_registry: ModelRegistry,
    /// Hardware acceleration manager
    pub hardware_manager: HardwareAccelerationManager,
    /// Connection status
    pub is_connected: bool,
}

/// Configuration for QML accelerator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLConfig {
    /// Maximum number of qubits
    pub max_qubits: usize,
    /// Optimization algorithm
    pub optimizer: OptimizerType,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Batch size for hybrid training
    pub batch_size: usize,
    /// Enable hardware acceleration
    pub enable_hardware_acceleration: bool,
    /// Gradient computation method
    pub gradient_method: GradientMethod,
    /// Noise resilience level
    pub noise_resilience: NoiseResilienceLevel,
    /// Circuit depth limit
    pub max_circuit_depth: usize,
    /// Parameter update frequency
    pub parameter_update_frequency: usize,
}

impl Default for QMLConfig {
    fn default() -> Self {
        Self {
            max_qubits: 20,
            optimizer: OptimizerType::Adam,
            learning_rate: 0.01,
            max_epochs: 1000,
            convergence_tolerance: 1e-6,
            batch_size: 32,
            enable_hardware_acceleration: true,
            gradient_method: GradientMethod::ParameterShift,
            noise_resilience: NoiseResilienceLevel::Medium,
            max_circuit_depth: 100,
            parameter_update_frequency: 10,
        }
    }
}

/// Types of optimizers for QML
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Gradient descent
    GradientDescent,
    /// Adam optimizer
    Adam,
    /// AdaGrad optimizer
    AdaGrad,
    /// RMSprop optimizer
    RMSprop,
    /// Simultaneous Perturbation Stochastic Approximation
    SPSA,
    /// Quantum Natural Gradient
    QuantumNaturalGradient,
    /// Nelder-Mead
    NelderMead,
    /// COBYLA (Constrained Optimization BY Linear Approximation)
    COBYLA,
}

/// Gradient computation methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientMethod {
    /// Parameter shift rule
    ParameterShift,
    /// Finite differences
    FiniteDifference,
    /// Linear combination of unitaries
    LinearCombination,
    /// Quantum natural gradient
    QuantumNaturalGradient,
    /// Adjoint method
    Adjoint,
}

/// Noise resilience levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseResilienceLevel {
    Low,
    Medium,
    High,
    Adaptive,
}

/// Training epoch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEpoch {
    pub epoch: usize,
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub parameters: Vec<f64>,
    pub gradient_norm: f64,
    pub learning_rate: f64,
    pub execution_time: Duration,
    pub quantum_fidelity: Option<f64>,
    pub classical_preprocessing_time: Duration,
    pub quantum_execution_time: Duration,
}

/// QML model types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QMLModelType {
    /// Variational Quantum Classifier
    VQC,
    /// Quantum Neural Network
    QNN,
    /// Quantum Approximate Optimization Algorithm
    QAOA,
    /// Variational Quantum Eigensolver
    VQE,
    /// Quantum Generative Adversarial Network
    QGAN,
    /// Quantum Convolutional Neural Network
    QCNN,
    /// Hybrid Classical-Quantum Network
    HybridNetwork,
}

impl QMLAccelerator {
    /// Create a new QML accelerator
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        config: QMLConfig,
    ) -> DeviceResult<Self> {
        let model_registry = ModelRegistry::new();
        let hardware_manager = HardwareAccelerationManager::new(&config)?;

        Ok(Self {
            device,
            config,
            training_history: Vec::new(),
            model_registry,
            hardware_manager,
            is_connected: false,
        })
    }

    /// Connect to quantum hardware
    pub async fn connect(&mut self) -> DeviceResult<()> {
        let device = self.device.read().await;
        if !device.is_available().await? {
            return Err(DeviceError::DeviceNotInitialized(
                "Quantum device not available".to_string(),
            ));
        }

        self.hardware_manager.initialize().await?;
        self.is_connected = true;
        Ok(())
    }

    /// Disconnect from hardware
    pub async fn disconnect(&mut self) -> DeviceResult<()> {
        self.hardware_manager.shutdown().await?;
        self.is_connected = false;
        Ok(())
    }

    /// Train a quantum machine learning model
    pub async fn train_model(
        &mut self,
        model_type: QMLModelType,
        training_data: training::TrainingData,
        validation_data: Option<training::TrainingData>,
    ) -> DeviceResult<training::TrainingResult> {
        if !self.is_connected {
            return Err(DeviceError::DeviceNotInitialized(
                "QML accelerator not connected".to_string(),
            ));
        }

        let mut trainer = QuantumTrainer::new(self.device.clone(), &self.config, model_type)?;

        let result = trainer
            .train(training_data, validation_data, &mut self.training_history)
            .await?;

        // Register the trained model
        self.model_registry
            .register_model(result.model_id.clone(), result.model.clone())?;

        Ok(result)
    }

    /// Perform inference with a trained model
    pub async fn inference(
        &self,
        model_id: &str,
        input_data: InferenceData,
    ) -> DeviceResult<InferenceResult> {
        if !self.is_connected {
            return Err(DeviceError::DeviceNotInitialized(
                "QML accelerator not connected".to_string(),
            ));
        }

        let model = self.model_registry.get_model(model_id)?;
        let inference_engine = QuantumInferenceEngine::new(self.device.clone(), &self.config)?;

        inference_engine.inference(model, input_data).await
    }

    /// Optimize quantum circuit parameters
    pub async fn optimize_parameters(
        &mut self,
        initial_parameters: Vec<f64>,
        objective_function: Box<dyn ObjectiveFunction + Send + Sync>,
    ) -> DeviceResult<OptimizationResult> {
        let mut optimizer =
            create_gradient_optimizer(self.device.clone(), OptimizerType::Adam, 0.01);

        optimizer.optimize(initial_parameters, objective_function)
    }

    /// Compute gradients using quantum methods
    pub async fn compute_gradients(
        &self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Vec<f64>,
    ) -> DeviceResult<Vec<f64>> {
        let gradient_calculator =
            QuantumGradientCalculator::new(self.device.clone(), GradientConfig::default())?;

        gradient_calculator
            .compute_gradients(circuit, parameters)
            .await
    }

    /// Get training statistics
    pub fn get_training_statistics(&self) -> TrainingStatistics {
        TrainingStatistics::from_history(&self.training_history)
    }

    /// Export trained model
    pub async fn export_model(
        &self,
        model_id: &str,
        format: ModelExportFormat,
    ) -> DeviceResult<Vec<u8>> {
        let model = self.model_registry.get_model(model_id)?;
        model.export(format).await
    }

    /// Import trained model
    pub async fn import_model(
        &mut self,
        model_data: Vec<u8>,
        format: ModelExportFormat,
    ) -> DeviceResult<String> {
        let model = QMLModel::import(model_data, format).await?;
        let model_id = format!("imported_model_{}", uuid::Uuid::new_v4());

        self.model_registry
            .register_model(model_id.clone(), model)?;
        Ok(model_id)
    }

    /// Get hardware acceleration metrics
    pub async fn get_acceleration_metrics(&self) -> HardwareAccelerationMetrics {
        self.hardware_manager.get_metrics().await
    }

    /// Benchmark quantum vs classical performance
    pub async fn benchmark_performance(
        &self,
        model_type: QMLModelType,
        problem_size: usize,
    ) -> DeviceResult<PerformanceBenchmark> {
        let benchmark_engine = PerformanceBenchmarkEngine::new(self.device.clone(), &self.config)?;

        benchmark_engine.benchmark(model_type, problem_size).await
    }

    /// Get QML accelerator diagnostics
    pub async fn get_diagnostics(&self) -> QMLDiagnostics {
        let device = self.device.read().await;
        let device_props = device.properties().await.unwrap_or_default();

        QMLDiagnostics {
            is_connected: self.is_connected,
            total_models: self.model_registry.model_count(),
            training_epochs_completed: self.training_history.len(),
            hardware_acceleration_enabled: self.config.enable_hardware_acceleration,
            active_model_count: self.model_registry.active_model_count(),
            average_training_time: self.calculate_average_training_time(),
            quantum_advantage_ratio: self.hardware_manager.get_quantum_advantage_ratio().await,
            device_properties: device_props,
        }
    }

    fn calculate_average_training_time(&self) -> Duration {
        if self.training_history.is_empty() {
            return Duration::from_secs(0);
        }

        let total_time: Duration = self
            .training_history
            .iter()
            .map(|epoch| epoch.execution_time)
            .sum();

        total_time / self.training_history.len() as u32
    }
}

/// Inference data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceData {
    pub features: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

/// Inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub prediction: f64,
    pub confidence: Option<f64>,
    pub quantum_fidelity: Option<f64>,
    pub execution_time: Duration,
    pub metadata: HashMap<String, String>,
}

/// QML model representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLModel {
    pub model_type: QMLModelType,
    pub parameters: Vec<f64>,
    pub circuit_structure: CircuitStructure,
    pub training_metadata: HashMap<String, String>,
    pub performance_metrics: HashMap<String, f64>,
}

impl QMLModel {
    pub async fn export(&self, format: ModelExportFormat) -> DeviceResult<Vec<u8>> {
        match format {
            ModelExportFormat::JSON => serde_json::to_vec(self)
                .map_err(|e| DeviceError::InvalidInput(format!("JSON export error: {e}"))),
            ModelExportFormat::Binary => {
                bincode::serde::encode_to_vec(self, bincode::config::standard())
                    .map_err(|e| DeviceError::InvalidInput(format!("Binary export error: {e:?}")))
            }
            ModelExportFormat::ONNX => {
                // Placeholder for ONNX export
                Err(DeviceError::InvalidInput(
                    "ONNX export not yet implemented".to_string(),
                ))
            }
        }
    }

    pub async fn import(data: Vec<u8>, format: ModelExportFormat) -> DeviceResult<Self> {
        match format {
            ModelExportFormat::JSON => serde_json::from_slice(&data)
                .map_err(|e| DeviceError::InvalidInput(format!("JSON import error: {e}"))),
            ModelExportFormat::Binary => {
                bincode::serde::decode_from_slice(&data, bincode::config::standard())
                    .map(|(v, _consumed)| v)
                    .map_err(|e| DeviceError::InvalidInput(format!("Binary import error: {e:?}")))
            }
            ModelExportFormat::ONNX => Err(DeviceError::InvalidInput(
                "ONNX import not yet implemented".to_string(),
            )),
        }
    }
}

/// Model export formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelExportFormat {
    JSON,
    Binary,
    ONNX,
}

/// Circuit structure representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitStructure {
    pub num_qubits: usize,
    pub depth: usize,
    pub gate_types: Vec<String>,
    pub parameter_count: usize,
    pub entangling_gates: usize,
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatistics {
    pub total_epochs: usize,
    pub final_loss: f64,
    pub best_loss: f64,
    pub average_loss: f64,
    pub convergence_epoch: Option<usize>,
    pub total_training_time: Duration,
    pub average_epoch_time: Duration,
}

impl TrainingStatistics {
    pub fn from_history(history: &[TrainingEpoch]) -> Self {
        if history.is_empty() {
            return Self {
                total_epochs: 0,
                final_loss: 0.0,
                best_loss: f64::INFINITY,
                average_loss: 0.0,
                convergence_epoch: None,
                total_training_time: Duration::from_secs(0),
                average_epoch_time: Duration::from_secs(0),
            };
        }

        let total_epochs = history.len();
        // Safe to use expect here since we already verified history is not empty above
        let final_loss = history
            .last()
            .expect("history should not be empty after early return check")
            .loss;
        let best_loss = history.iter().map(|e| e.loss).fold(f64::INFINITY, f64::min);
        let average_loss = history.iter().map(|e| e.loss).sum::<f64>() / total_epochs as f64;
        let total_training_time = history.iter().map(|e| e.execution_time).sum();
        let average_epoch_time = total_training_time / total_epochs as u32;

        Self {
            total_epochs,
            final_loss,
            best_loss,
            average_loss,
            convergence_epoch: None, // Could implement convergence detection
            total_training_time,
            average_epoch_time,
        }
    }
}

/// QML diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLDiagnostics {
    pub is_connected: bool,
    pub total_models: usize,
    pub training_epochs_completed: usize,
    pub hardware_acceleration_enabled: bool,
    pub active_model_count: usize,
    pub average_training_time: Duration,
    pub quantum_advantage_ratio: f64,
    pub device_properties: HashMap<String, String>,
}

/// Model registry for managing trained models
pub struct ModelRegistry {
    models: HashMap<String, QMLModel>,
    active_models: HashMap<String, bool>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            active_models: HashMap::new(),
        }
    }

    pub fn register_model(&mut self, id: String, model: QMLModel) -> DeviceResult<()> {
        self.models.insert(id.clone(), model);
        self.active_models.insert(id, true);
        Ok(())
    }

    pub fn get_model(&self, id: &str) -> DeviceResult<&QMLModel> {
        self.models
            .get(id)
            .ok_or_else(|| DeviceError::InvalidInput(format!("Model {id} not found")))
    }

    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    pub fn active_model_count(&self) -> usize {
        self.active_models
            .values()
            .filter(|&&active| active)
            .count()
    }

    pub fn deactivate_model(&mut self, id: &str) -> DeviceResult<()> {
        if self.active_models.contains_key(id) {
            self.active_models.insert(id.to_string(), false);
            Ok(())
        } else {
            Err(DeviceError::InvalidInput(format!("Model {id} not found")))
        }
    }
}

/// Create a VQC (Variational Quantum Classifier) accelerator
pub fn create_vqc_accelerator(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    num_qubits: usize,
) -> DeviceResult<QMLAccelerator> {
    let config = QMLConfig {
        max_qubits: num_qubits,
        optimizer: OptimizerType::Adam,
        gradient_method: GradientMethod::ParameterShift,
        ..Default::default()
    };

    QMLAccelerator::new(device, config)
}

/// Create a QAOA accelerator
pub fn create_qaoa_accelerator(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    problem_size: usize,
) -> DeviceResult<QMLAccelerator> {
    let config = QMLConfig {
        max_qubits: problem_size,
        optimizer: OptimizerType::COBYLA,
        gradient_method: GradientMethod::FiniteDifference,
        max_circuit_depth: 50,
        ..Default::default()
    };

    QMLAccelerator::new(device, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[tokio::test]
    async fn test_qml_accelerator_creation() {
        let device = create_mock_quantum_device();
        let accelerator = QMLAccelerator::new(device, QMLConfig::default())
            .expect("QML accelerator creation should succeed with mock device");

        assert_eq!(accelerator.config.max_qubits, 20);
        assert!(!accelerator.is_connected);
    }

    #[tokio::test]
    async fn test_model_registry() {
        let mut registry = ModelRegistry::new();
        assert_eq!(registry.model_count(), 0);

        let model = QMLModel {
            model_type: QMLModelType::VQC,
            parameters: vec![0.1, 0.2, 0.3],
            circuit_structure: CircuitStructure {
                num_qubits: 4,
                depth: 10,
                gate_types: vec!["RY".to_string(), "CNOT".to_string()],
                parameter_count: 8,
                entangling_gates: 4,
            },
            training_metadata: HashMap::new(),
            performance_metrics: HashMap::new(),
        };

        registry
            .register_model("test_model".to_string(), model)
            .expect("registering model should succeed");
        assert_eq!(registry.model_count(), 1);
        assert_eq!(registry.active_model_count(), 1);

        let retrieved = registry
            .get_model("test_model")
            .expect("retrieving registered model should succeed");
        assert_eq!(retrieved.model_type, QMLModelType::VQC);
    }

    #[test]
    fn test_training_statistics() {
        let history = vec![
            TrainingEpoch {
                epoch: 0,
                loss: 1.0,
                accuracy: Some(0.5),
                parameters: vec![0.1],
                gradient_norm: 0.5,
                learning_rate: 0.01,
                execution_time: Duration::from_millis(100),
                quantum_fidelity: Some(0.95),
                classical_preprocessing_time: Duration::from_millis(10),
                quantum_execution_time: Duration::from_millis(90),
            },
            TrainingEpoch {
                epoch: 1,
                loss: 0.5,
                accuracy: Some(0.7),
                parameters: vec![0.2],
                gradient_norm: 0.3,
                learning_rate: 0.01,
                execution_time: Duration::from_millis(120),
                quantum_fidelity: Some(0.96),
                classical_preprocessing_time: Duration::from_millis(15),
                quantum_execution_time: Duration::from_millis(105),
            },
        ];

        let stats = TrainingStatistics::from_history(&history);
        assert_eq!(stats.total_epochs, 2);
        assert_eq!(stats.final_loss, 0.5);
        assert_eq!(stats.best_loss, 0.5);
        assert_eq!(stats.average_loss, 0.75);
    }
}
