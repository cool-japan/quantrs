//! Quantum Machine Learning with Annealing
//!
//! This module implements quantum machine learning algorithms that can be trained and optimized
//! using quantum annealing techniques. It provides a comprehensive framework for quantum neural
//! networks, variational quantum classifiers, quantum feature maps, and other QML algorithms
//! that leverage the power of quantum annealing for optimization.
//!
//! Key features:
//! - Quantum Neural Networks (QNN) with annealing-based training
//! - Variational Quantum Classifiers (VQC)
//! - Quantum Feature Maps for encoding classical data
//! - Quantum Kernel Methods using quantum circuits
//! - Quantum Generative Models (QGANs)
//! - Quantum Reinforcement Learning with annealing policy optimization
//! - Quantum Autoencoders for dimensionality reduction
//! - Integration with Ising/QUBO optimization

use num_complex::Complex as NComplex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};
use crate::simulator::{AnnealingParams, AnnealingSolution, ClassicalAnnealingSimulator};

/// Errors that can occur in quantum machine learning operations
#[derive(Error, Debug)]
pub enum QmlError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Data processing error
    #[error("Data processing error: {0}")]
    DataError(String),

    /// Model architecture error
    #[error("Model architecture error: {0}")]
    ArchitectureError(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Result type for QML operations
pub type QmlResult<T> = Result<T, QmlError>;

/// Quantum gate types for quantum circuits
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumGate {
    /// Pauli-X (NOT) gate
    PauliX,
    /// Pauli-Y gate
    PauliY,
    /// Pauli-Z gate
    PauliZ,
    /// Hadamard gate
    Hadamard,
    /// Rotation around X-axis
    RX(f64),
    /// Rotation around Y-axis
    RY(f64),
    /// Rotation around Z-axis
    RZ(f64),
    /// Controlled-NOT gate
    CNOT,
    /// Controlled-Z gate
    CZ,
    /// Two-qubit ZZ rotation
    ZZRotation(f64),
    /// Phase gate
    Phase(f64),
    /// S gate
    SGate,
    /// T gate
    TGate,
}

/// Quantum circuit layer for variational algorithms
#[derive(Debug, Clone)]
pub struct QuantumLayer {
    /// Gates in this layer
    pub gates: Vec<(QuantumGate, Vec<usize>)>,
    /// Trainable parameters
    pub parameters: Vec<f64>,
    /// Parameter indices for each gate
    pub parameter_indices: Vec<Option<usize>>,
}

/// Quantum circuit for machine learning models
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit layers
    pub layers: Vec<QuantumLayer>,
    /// Total number of parameters
    pub num_parameters: usize,
    /// Circuit depth
    pub depth: usize,
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            layers: Vec::new(),
            num_parameters: 0,
            depth: 0,
        }
    }

    /// Add a parameterized layer to the circuit
    pub fn add_layer(&mut self, layer: QuantumLayer) {
        self.num_parameters += layer.parameters.len();
        self.depth += 1;
        self.layers.push(layer);
    }

    /// Create a hardware-efficient ansatz
    pub fn hardware_efficient_ansatz(num_qubits: usize, num_layers: usize) -> Self {
        let mut circuit = Self::new(num_qubits);

        for layer in 0..num_layers {
            let mut gates = Vec::new();
            let mut parameters = Vec::new();
            let mut param_indices = Vec::new();
            let mut param_count = 0;

            // Single-qubit rotations
            for qubit in 0..num_qubits {
                // RY rotation
                gates.push((QuantumGate::RY(0.0), vec![qubit]));
                parameters.push(0.0);
                param_indices.push(Some(param_count));
                param_count += 1;

                // RZ rotation
                gates.push((QuantumGate::RZ(0.0), vec![qubit]));
                parameters.push(0.0);
                param_indices.push(Some(param_count));
                param_count += 1;
            }

            // Entangling gates
            for qubit in 0..num_qubits {
                let target = (qubit + 1) % num_qubits;
                gates.push((QuantumGate::CNOT, vec![qubit, target]));
                param_indices.push(None);
            }

            circuit.add_layer(QuantumLayer {
                gates,
                parameters,
                parameter_indices: param_indices,
            });
        }

        circuit
    }

    /// Update circuit parameters
    pub fn update_parameters(&mut self, params: &[f64]) -> QmlResult<()> {
        if params.len() != self.num_parameters {
            return Err(QmlError::DimensionMismatch {
                expected: self.num_parameters,
                actual: params.len(),
            });
        }

        let mut param_idx = 0;
        for layer in &mut self.layers {
            for (i, gate_param_idx) in layer.parameter_indices.iter().enumerate() {
                if let Some(idx) = gate_param_idx {
                    layer.parameters[*idx] = params[param_idx];
                    param_idx += 1;

                    // Update gate parameters
                    match &mut layer.gates[i].0 {
                        QuantumGate::RX(ref mut angle)
                        | QuantumGate::RY(ref mut angle)
                        | QuantumGate::RZ(ref mut angle)
                        | QuantumGate::Phase(ref mut angle)
                        | QuantumGate::ZZRotation(ref mut angle) => {
                            *angle = layer.parameters[*idx];
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }
}

/// Quantum feature map for encoding classical data
#[derive(Debug, Clone)]
pub struct QuantumFeatureMap {
    /// Number of features
    pub num_features: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// Feature map type
    pub map_type: FeatureMapType,
    /// Circuit for feature encoding
    pub circuit: QuantumCircuit,
    /// Feature scaling parameters
    pub scaling: Vec<f64>,
}

/// Types of quantum feature maps
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureMapType {
    /// Simple amplitude encoding
    AmplitudeEncoding,
    /// Angle encoding using RY gates
    AngleEncoding,
    /// Pauli feature map
    PauliFeatureMap { entanglement: EntanglementType },
    /// ZZ feature map
    ZZFeatureMap { repetitions: usize },
    /// Custom feature map
    Custom,
}

/// Types of entanglement for feature maps
#[derive(Debug, Clone, PartialEq)]
pub enum EntanglementType {
    /// Linear entanglement
    Linear,
    /// Circular entanglement
    Circular,
    /// Full entanglement
    Full,
}

impl QuantumFeatureMap {
    /// Create a new quantum feature map
    pub fn new(
        num_features: usize,
        num_qubits: usize,
        map_type: FeatureMapType,
    ) -> QmlResult<Self> {
        if num_features > num_qubits {
            return Err(QmlError::ArchitectureError(format!(
                "Cannot encode {} features into {} qubits",
                num_features, num_qubits
            )));
        }

        let circuit = match &map_type {
            FeatureMapType::AngleEncoding => Self::create_angle_encoding_circuit(num_qubits),
            FeatureMapType::PauliFeatureMap { entanglement } => {
                Self::create_pauli_feature_map_circuit(num_qubits, entanglement.clone())
            }
            FeatureMapType::ZZFeatureMap { repetitions } => {
                Self::create_zz_feature_map_circuit(num_qubits, *repetitions)
            }
            _ => QuantumCircuit::new(num_qubits),
        };

        Ok(Self {
            num_features,
            num_qubits,
            map_type,
            circuit,
            scaling: vec![1.0; num_features],
        })
    }

    /// Create angle encoding circuit
    fn create_angle_encoding_circuit(num_qubits: usize) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(num_qubits);

        let mut gates = Vec::new();
        let mut parameters = Vec::new();
        let mut param_indices = Vec::new();

        for qubit in 0..num_qubits {
            gates.push((QuantumGate::RY(0.0), vec![qubit]));
            parameters.push(0.0);
            param_indices.push(Some(qubit));
        }

        circuit.add_layer(QuantumLayer {
            gates,
            parameters,
            parameter_indices: param_indices,
        });

        circuit
    }

    /// Create Pauli feature map circuit
    fn create_pauli_feature_map_circuit(
        num_qubits: usize,
        entanglement: EntanglementType,
    ) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(num_qubits);

        // Hadamard layer
        let mut gates = Vec::new();
        for qubit in 0..num_qubits {
            gates.push((QuantumGate::Hadamard, vec![qubit]));
        }

        circuit.add_layer(QuantumLayer {
            gates: gates.clone(),
            parameters: Vec::new(),
            parameter_indices: vec![None; gates.len()],
        });

        // Feature encoding layer
        let mut feature_gates = Vec::new();
        let mut parameters = Vec::new();
        let mut param_indices = Vec::new();

        for qubit in 0..num_qubits {
            feature_gates.push((QuantumGate::RZ(0.0), vec![qubit]));
            parameters.push(0.0);
            param_indices.push(Some(qubit));
        }

        // Entanglement layer
        match entanglement {
            EntanglementType::Linear => {
                for qubit in 0..num_qubits - 1 {
                    feature_gates.push((QuantumGate::CNOT, vec![qubit, qubit + 1]));
                    param_indices.push(None);
                }
            }
            EntanglementType::Circular => {
                for qubit in 0..num_qubits {
                    let target = (qubit + 1) % num_qubits;
                    feature_gates.push((QuantumGate::CNOT, vec![qubit, target]));
                    param_indices.push(None);
                }
            }
            EntanglementType::Full => {
                for i in 0..num_qubits {
                    for j in (i + 1)..num_qubits {
                        feature_gates.push((QuantumGate::CNOT, vec![i, j]));
                        param_indices.push(None);
                    }
                }
            }
        }

        circuit.add_layer(QuantumLayer {
            gates: feature_gates,
            parameters,
            parameter_indices: param_indices,
        });

        circuit
    }

    /// Create ZZ feature map circuit
    fn create_zz_feature_map_circuit(num_qubits: usize, repetitions: usize) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(num_qubits);

        for _ in 0..repetitions {
            // Hadamard layer
            let mut gates = Vec::new();
            for qubit in 0..num_qubits {
                gates.push((QuantumGate::Hadamard, vec![qubit]));
            }

            circuit.add_layer(QuantumLayer {
                gates,
                parameters: Vec::new(),
                parameter_indices: vec![None; num_qubits],
            });

            // ZZ rotation layer
            let mut zz_gates = Vec::new();
            let mut parameters = Vec::new();
            let mut param_indices = Vec::new();
            let mut param_count = 0;

            for i in 0..num_qubits {
                for j in (i + 1)..num_qubits {
                    zz_gates.push((QuantumGate::ZZRotation(0.0), vec![i, j]));
                    parameters.push(0.0);
                    param_indices.push(Some(param_count));
                    param_count += 1;
                }
            }

            circuit.add_layer(QuantumLayer {
                gates: zz_gates,
                parameters,
                parameter_indices: param_indices,
            });
        }

        circuit
    }

    /// Encode data into quantum state
    pub fn encode(&self, data: &[f64]) -> QmlResult<Vec<f64>> {
        if data.len() != self.num_features {
            return Err(QmlError::DimensionMismatch {
                expected: self.num_features,
                actual: data.len(),
            });
        }

        // Scale data
        let scaled_data: Vec<f64> = data
            .iter()
            .zip(&self.scaling)
            .map(|(x, scale)| x * scale)
            .collect();

        match self.map_type {
            FeatureMapType::AngleEncoding => {
                // Map data directly to rotation angles
                let mut params = vec![0.0; self.num_qubits];
                for (i, &value) in scaled_data.iter().enumerate().take(self.num_qubits) {
                    params[i] = value * PI;
                }
                Ok(params)
            }
            _ => {
                // For other encodings, return scaled data
                Ok(scaled_data)
            }
        }
    }
}

/// Variational Quantum Classifier
#[derive(Debug, Clone)]
pub struct VariationalQuantumClassifier {
    /// Feature map for data encoding
    pub feature_map: QuantumFeatureMap,
    /// Variational ansatz
    pub ansatz: QuantumCircuit,
    /// Trainable parameters
    pub parameters: Vec<f64>,
    /// Number of classes
    pub num_classes: usize,
    /// Training configuration
    pub config: VqcConfig,
    /// Training history
    pub training_history: TrainingHistory,
}

/// Configuration for Variational Quantum Classifier
#[derive(Debug, Clone)]
pub struct VqcConfig {
    /// Maximum training iterations
    pub max_iterations: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of shots for quantum measurements
    pub num_shots: usize,
    /// Regularization strength
    pub regularization: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for VqcConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            learning_rate: 0.01,
            tolerance: 1e-6,
            num_shots: 1024,
            regularization: 0.001,
            batch_size: 32,
            seed: None,
        }
    }
}

/// Training sample for supervised learning
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Input features
    pub features: Vec<f64>,
    /// Target label
    pub label: usize,
    /// Sample weight
    pub weight: f64,
}

/// Training history tracking
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Loss values over iterations
    pub losses: Vec<f64>,
    /// Accuracy values over iterations
    pub accuracies: Vec<f64>,
    /// Training times
    pub iteration_times: Vec<Duration>,
    /// Parameter updates
    pub parameter_updates: Vec<Vec<f64>>,
}

impl TrainingHistory {
    /// Create new training history
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            accuracies: Vec::new(),
            iteration_times: Vec::new(),
            parameter_updates: Vec::new(),
        }
    }
}

impl VariationalQuantumClassifier {
    /// Create a new VQC
    pub fn new(
        num_features: usize,
        num_qubits: usize,
        num_classes: usize,
        ansatz_layers: usize,
        config: VqcConfig,
    ) -> QmlResult<Self> {
        // Create feature map
        let feature_map = QuantumFeatureMap::new(
            num_features,
            num_qubits,
            FeatureMapType::ZZFeatureMap { repetitions: 2 },
        )?;

        // Create variational ansatz
        let ansatz = QuantumCircuit::hardware_efficient_ansatz(num_qubits, ansatz_layers);

        // Initialize parameters randomly
        let mut rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };

        let parameters: Vec<f64> = (0..ansatz.num_parameters)
            .map(|_| rng.gen_range(-PI..PI))
            .collect();

        Ok(Self {
            feature_map,
            ansatz,
            parameters,
            num_classes,
            config,
            training_history: TrainingHistory::new(),
        })
    }

    /// Train the classifier using annealing optimization
    pub fn train(&mut self, training_data: &[TrainingSample]) -> QmlResult<()> {
        if training_data.is_empty() {
            return Err(QmlError::TrainingError("Empty training data".to_string()));
        }

        println!("Training VQC with {} samples", training_data.len());

        // Convert to optimization problem
        let optimization_problem = self.create_optimization_problem(training_data)?;

        // Use annealing to optimize parameters
        let annealing_params = AnnealingParams {
            num_sweeps: self.config.max_iterations.min(200),
            num_repetitions: 3,
            initial_temperature: 5.0,
            timeout: Some(10.0), // 10 second timeout
            ..Default::default()
        };

        let simulator = ClassicalAnnealingSimulator::new(annealing_params)
            .map_err(|e| QmlError::OptimizationError(format!("Annealing setup failed: {}", e)))?;

        let start = Instant::now();
        let result = simulator
            .solve(&optimization_problem)
            .map_err(|e| QmlError::OptimizationError(format!("Annealing failed: {}", e)))?;

        // Update parameters from annealing result
        self.update_parameters_from_solution(&result)?;

        // Record training metrics
        let loss = self.calculate_loss(training_data)?;
        let accuracy = self.calculate_accuracy(training_data)?;

        self.training_history.losses.push(loss);
        self.training_history.accuracies.push(accuracy);
        self.training_history.iteration_times.push(start.elapsed());
        self.training_history
            .parameter_updates
            .push(self.parameters.clone());

        println!(
            "Training completed - Loss: {:.4}, Accuracy: {:.2}%",
            loss,
            accuracy * 100.0
        );

        Ok(())
    }

    /// Create optimization problem for parameter training
    fn create_optimization_problem(
        &self,
        training_data: &[TrainingSample],
    ) -> QmlResult<IsingModel> {
        // Create Ising model to encode the loss function
        let num_params = self.parameters.len();
        let precision_bits = 8; // Precision for parameter discretization
        let total_qubits = num_params * precision_bits;

        let mut ising = IsingModel::new(total_qubits);

        // Encode loss function as Ising problem (simplified)
        // This is a placeholder - in practice, this would involve more complex encoding
        for i in 0..total_qubits {
            // Add small bias to prevent trivial solutions
            ising.set_bias(i, 0.1)?;
        }

        // Add couplings based on parameter correlations
        for i in 0..total_qubits {
            for j in (i + 1)..total_qubits {
                if (i / precision_bits) != (j / precision_bits) {
                    // Couple parameters from different groups
                    ising.set_coupling(i, j, -0.1)?;
                }
            }
        }

        Ok(ising)
    }

    /// Update parameters from annealing solution
    fn update_parameters_from_solution(&mut self, result: &AnnealingSolution) -> QmlResult<()> {
        let precision_bits = 8;

        for (param_idx, param) in self.parameters.iter_mut().enumerate() {
            let start_bit = param_idx * precision_bits;
            let end_bit = start_bit + precision_bits;

            if end_bit <= result.best_spins.len() {
                // Convert binary representation to parameter value
                let mut binary_val = 0i32;
                for (bit_idx, &spin) in result.best_spins[start_bit..end_bit].iter().enumerate() {
                    if spin > 0 {
                        binary_val |= 1 << bit_idx;
                    }
                }

                // Map to parameter range [-π, π]
                let normalized = binary_val as f64 / ((1 << precision_bits) - 1) as f64;
                *param = (normalized - 0.5) * 2.0 * PI;
            }
        }

        Ok(())
    }

    /// Predict class for input features
    pub fn predict(&self, features: &[f64]) -> QmlResult<usize> {
        let probabilities = self.predict_proba(features)?;

        // Return class with highest probability
        let max_class = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(max_class)
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, features: &[f64]) -> QmlResult<Vec<f64>> {
        // Encode features
        let encoded_features = self.feature_map.encode(features)?;

        // Simulate quantum circuit (simplified)
        let mut probabilities = vec![0.0; self.num_classes];

        // For simplicity, use a heuristic based on feature encoding and parameters
        for (i, &param) in self.parameters.iter().enumerate().take(self.num_classes) {
            let feature_sum: f64 = encoded_features.iter().sum();
            probabilities[i] = (param * feature_sum).cos().abs();
        }

        // Normalize probabilities
        let sum: f64 = probabilities.iter().sum();
        if sum > 0.0 {
            for prob in &mut probabilities {
                *prob /= sum;
            }
        } else {
            // Uniform distribution as fallback
            let uniform_prob = 1.0 / self.num_classes as f64;
            probabilities.fill(uniform_prob);
        }

        Ok(probabilities)
    }

    /// Calculate loss for training data
    fn calculate_loss(&self, training_data: &[TrainingSample]) -> QmlResult<f64> {
        let mut total_loss = 0.0;

        for sample in training_data {
            let probabilities = self.predict_proba(&sample.features)?;

            // Cross-entropy loss
            let predicted_prob = probabilities.get(sample.label).unwrap_or(&1e-10);
            total_loss -= predicted_prob.ln() * sample.weight;
        }

        // Add regularization
        let regularization_term: f64 =
            self.parameters.iter().map(|&p| p * p).sum::<f64>() * self.config.regularization;

        Ok(total_loss / training_data.len() as f64 + regularization_term)
    }

    /// Calculate accuracy for training data
    fn calculate_accuracy(&self, training_data: &[TrainingSample]) -> QmlResult<f64> {
        let mut correct = 0;
        let mut total = 0;

        for sample in training_data {
            let predicted = self.predict(&sample.features)?;
            if predicted == sample.label {
                correct += 1;
            }
            total += 1;
        }

        Ok(correct as f64 / total as f64)
    }
}

/// Quantum Neural Network implementation
#[derive(Debug, Clone)]
pub struct QuantumNeuralNetwork {
    /// Network layers
    pub layers: Vec<QuantumNeuralLayer>,
    /// Network configuration
    pub config: QnnConfig,
    /// Training history
    pub training_history: TrainingHistory,
}

/// Quantum neural network layer
#[derive(Debug, Clone)]
pub struct QuantumNeuralLayer {
    /// Number of input qubits
    pub input_size: usize,
    /// Number of output qubits
    pub output_size: usize,
    /// Quantum circuit for this layer
    pub circuit: QuantumCircuit,
    /// Layer parameters
    pub parameters: Vec<f64>,
    /// Activation function
    pub activation: ActivationType,
}

/// Activation function types for QNN
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    /// No activation (linear)
    Linear,
    /// Quantum sigmoid approximation
    QuantumSigmoid,
    /// Quantum ReLU approximation
    QuantumReLU,
    /// Quantum tanh approximation
    QuantumTanh,
}

/// Configuration for Quantum Neural Network
#[derive(Debug, Clone)]
pub struct QnnConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Regularization strength
    pub regularization: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for QnnConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_epochs: 100,
            batch_size: 32,
            tolerance: 1e-6,
            regularization: 0.001,
            seed: None,
        }
    }
}

impl QuantumNeuralNetwork {
    /// Create a new quantum neural network
    pub fn new(architecture: &[usize], config: QnnConfig) -> QmlResult<Self> {
        if architecture.len() < 2 {
            return Err(QmlError::ArchitectureError(
                "Network must have at least input and output layers".to_string(),
            ));
        }

        let mut layers = Vec::new();

        for i in 0..architecture.len() - 1 {
            let input_size = architecture[i];
            let output_size = architecture[i + 1];

            let layer =
                QuantumNeuralLayer::new(input_size, output_size, ActivationType::QuantumSigmoid)?;

            layers.push(layer);
        }

        Ok(Self {
            layers,
            config,
            training_history: TrainingHistory::new(),
        })
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &[f64]) -> QmlResult<Vec<f64>> {
        let mut current_output = input.to_vec();

        for layer in &self.layers {
            current_output = layer.forward(&current_output)?;
        }

        Ok(current_output)
    }

    /// Train the network using quantum annealing
    pub fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)]) -> QmlResult<()> {
        println!("Training QNN with {} samples", training_data.len());

        for epoch in 0..self.config.max_epochs {
            let start = Instant::now();

            // Create optimization problem for this epoch
            let optimization_problem = self.create_training_problem(training_data)?;

            // Use annealing to optimize
            let annealing_params = AnnealingParams {
                num_sweeps: 100,
                num_repetitions: 2,
                initial_temperature: 3.0,
                timeout: Some(5.0), // 5 second timeout
                ..Default::default()
            };

            let simulator = ClassicalAnnealingSimulator::new(annealing_params).map_err(|e| {
                QmlError::OptimizationError(format!("Annealing setup failed: {}", e))
            })?;

            let result = simulator
                .solve(&optimization_problem)
                .map_err(|e| QmlError::OptimizationError(format!("Annealing failed: {}", e)))?;

            // Update network parameters
            self.update_from_annealing_result(&result)?;

            // Calculate metrics
            let loss = self.calculate_loss(training_data)?;

            self.training_history.losses.push(loss);
            self.training_history.iteration_times.push(start.elapsed());

            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, loss);
            }

            // Check convergence
            if loss < self.config.tolerance {
                println!("Converged at epoch {}", epoch);
                break;
            }
        }

        Ok(())
    }

    /// Create optimization problem for training
    fn create_training_problem(
        &self,
        training_data: &[(Vec<f64>, Vec<f64>)],
    ) -> QmlResult<IsingModel> {
        // Calculate total parameters
        let total_params: usize = self.layers.iter().map(|layer| layer.parameters.len()).sum();

        let precision_bits = 6;
        let total_qubits = total_params * precision_bits;

        let mut ising = IsingModel::new(total_qubits);

        // Encode loss function (simplified)
        for i in 0..total_qubits {
            ising.set_bias(i, 0.05)?;
        }

        // Add parameter correlations
        for i in 0..total_qubits {
            for j in (i + 1)..total_qubits {
                if i / precision_bits != j / precision_bits {
                    ising.set_coupling(i, j, -0.02)?;
                }
            }
        }

        Ok(ising)
    }

    /// Update network from annealing result
    fn update_from_annealing_result(&mut self, result: &AnnealingSolution) -> QmlResult<()> {
        let precision_bits = 6;
        let mut param_index = 0;

        for layer in &mut self.layers {
            for param in &mut layer.parameters {
                let start_bit = param_index * precision_bits;
                let end_bit = start_bit + precision_bits;

                if end_bit <= result.best_spins.len() {
                    let mut binary_val = 0i32;
                    for (bit_idx, &spin) in result.best_spins[start_bit..end_bit].iter().enumerate()
                    {
                        if spin > 0 {
                            binary_val |= 1 << bit_idx;
                        }
                    }

                    let normalized = binary_val as f64 / ((1 << precision_bits) - 1) as f64;
                    *param = (normalized - 0.5) * 2.0; // Scale to [-1, 1]
                }

                param_index += 1;
            }

            // Update layer circuit parameters
            layer.circuit.update_parameters(&layer.parameters)?;
        }

        Ok(())
    }

    /// Calculate loss for training data
    fn calculate_loss(&self, training_data: &[(Vec<f64>, Vec<f64>)]) -> QmlResult<f64> {
        let mut total_loss = 0.0;

        for (input, target) in training_data {
            let output = self.forward(input)?;

            // Mean squared error
            let sample_loss: f64 = output
                .iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum();

            total_loss += sample_loss;
        }

        Ok(total_loss / training_data.len() as f64)
    }
}

impl QuantumNeuralLayer {
    /// Create a new quantum neural layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationType,
    ) -> QmlResult<Self> {
        let num_qubits = input_size.max(output_size);
        let circuit = QuantumCircuit::hardware_efficient_ansatz(num_qubits, 2);

        // Initialize parameters randomly
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let parameters: Vec<f64> = (0..circuit.num_parameters)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Ok(Self {
            input_size,
            output_size,
            circuit,
            parameters,
            activation,
        })
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &[f64]) -> QmlResult<Vec<f64>> {
        if input.len() != self.input_size {
            return Err(QmlError::DimensionMismatch {
                expected: self.input_size,
                actual: input.len(),
            });
        }

        // Simplified quantum computation
        let mut output = vec![0.0; self.output_size];

        for (i, &inp) in input.iter().enumerate().take(self.output_size) {
            let param_sum: f64 = self.parameters.iter().take(4).sum();
            output[i] = self.apply_activation(inp * param_sum)?;
        }

        // Pad with zeros if needed
        while output.len() < self.output_size {
            output.push(0.0);
        }

        Ok(output)
    }

    /// Apply activation function
    fn apply_activation(&self, x: f64) -> QmlResult<f64> {
        match self.activation {
            ActivationType::Linear => Ok(x),
            ActivationType::QuantumSigmoid => {
                // Quantum approximation of sigmoid
                Ok(0.5 * (1.0 + (x * PI / 2.0).sin()))
            }
            ActivationType::QuantumReLU => {
                // Quantum approximation of ReLU
                Ok(if x > 0.0 { x } else { 0.0 })
            }
            ActivationType::QuantumTanh => {
                // Quantum approximation of tanh
                Ok((x * PI / 4.0).sin())
            }
        }
    }
}

/// Quantum Kernel Methods for classification and regression
#[derive(Debug, Clone)]
pub struct QuantumKernelMethod {
    /// Feature map for kernel computation
    pub feature_map: QuantumFeatureMap,
    /// Training data
    pub training_data: Vec<(Vec<f64>, f64)>,
    /// Kernel matrix
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Support vectors
    pub support_vectors: Vec<usize>,
    /// Kernel method type
    pub method_type: KernelMethodType,
}

/// Types of quantum kernel methods
#[derive(Debug, Clone, PartialEq)]
pub enum KernelMethodType {
    /// Support Vector Machine
    SupportVectorMachine { c_parameter: f64 },
    /// Kernel Ridge Regression
    RidgeRegression { regularization: f64 },
    /// Gaussian Process
    GaussianProcess,
}

impl QuantumKernelMethod {
    /// Create a new quantum kernel method
    pub fn new(feature_map: QuantumFeatureMap, method_type: KernelMethodType) -> Self {
        Self {
            feature_map,
            training_data: Vec::new(),
            kernel_matrix: Vec::new(),
            support_vectors: Vec::new(),
            method_type,
        }
    }

    /// Compute quantum kernel between two data points
    pub fn quantum_kernel(&self, x1: &[f64], x2: &[f64]) -> QmlResult<f64> {
        let encoding1 = self.feature_map.encode(x1)?;
        let encoding2 = self.feature_map.encode(x2)?;

        // Simplified quantum kernel computation
        // In practice, this would involve computing overlap between quantum states
        let mut kernel_value = 0.0;

        for (e1, e2) in encoding1.iter().zip(encoding2.iter()) {
            kernel_value += (e1 * e2).cos();
        }

        kernel_value /= encoding1.len() as f64;
        Ok(kernel_value.abs())
    }

    /// Train the kernel method
    pub fn train(&mut self, training_data: Vec<(Vec<f64>, f64)>) -> QmlResult<()> {
        self.training_data = training_data;
        let n = self.training_data.len();

        // Compute kernel matrix
        self.kernel_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                let kernel_val =
                    self.quantum_kernel(&self.training_data[i].0, &self.training_data[j].0)?;
                self.kernel_matrix[i][j] = kernel_val;
            }
        }

        // Solve kernel method (simplified)
        match &self.method_type {
            KernelMethodType::SupportVectorMachine { .. } => {
                self.solve_svm()?;
            }
            KernelMethodType::RidgeRegression { .. } => {
                self.solve_ridge_regression()?;
            }
            KernelMethodType::GaussianProcess => {
                self.solve_gaussian_process()?;
            }
        }

        Ok(())
    }

    /// Solve SVM optimization problem
    fn solve_svm(&mut self) -> QmlResult<()> {
        // Simplified SVM solving - in practice, this would use proper quadratic programming
        let n = self.training_data.len();

        // Find support vectors (simplified heuristic)
        for i in 0..n {
            let mut is_support = false;

            // Check if this point is on the margin
            for j in 0..n {
                if i != j && self.kernel_matrix[i][j] > 0.5 {
                    is_support = true;
                    break;
                }
            }

            if is_support {
                self.support_vectors.push(i);
            }
        }

        Ok(())
    }

    /// Solve ridge regression
    fn solve_ridge_regression(&mut self) -> QmlResult<()> {
        // Simplified ridge regression solving
        // In practice, this would involve matrix inversion
        self.support_vectors = (0..self.training_data.len()).collect();
        Ok(())
    }

    /// Solve Gaussian process
    fn solve_gaussian_process(&mut self) -> QmlResult<()> {
        // Simplified GP solving
        self.support_vectors = (0..self.training_data.len()).collect();
        Ok(())
    }

    /// Make prediction for new data point
    pub fn predict(&self, x: &[f64]) -> QmlResult<f64> {
        let mut prediction = 0.0;

        for &sv_idx in &self.support_vectors {
            let kernel_val = self.quantum_kernel(x, &self.training_data[sv_idx].0)?;
            prediction += kernel_val * self.training_data[sv_idx].1;
        }

        prediction /= self.support_vectors.len() as f64;
        Ok(prediction)
    }
}

/// Quantum Generative Adversarial Network
#[derive(Debug, Clone)]
pub struct QuantumGAN {
    /// Generator network
    pub generator: QuantumNeuralNetwork,
    /// Discriminator network
    pub discriminator: QuantumNeuralNetwork,
    /// Training configuration
    pub config: QGanConfig,
    /// Training history
    pub training_history: QGanTrainingHistory,
}

/// Configuration for Quantum GAN
#[derive(Debug, Clone)]
pub struct QGanConfig {
    /// Latent dimension
    pub latent_dim: usize,
    /// Data dimension
    pub data_dim: usize,
    /// Training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rates
    pub generator_lr: f64,
    pub discriminator_lr: f64,
    /// Random seed
    pub seed: Option<u64>,
}

/// Training history for Quantum GAN
#[derive(Debug, Clone)]
pub struct QGanTrainingHistory {
    /// Generator losses
    pub generator_losses: Vec<f64>,
    /// Discriminator losses
    pub discriminator_losses: Vec<f64>,
    /// Training times per epoch
    pub epoch_times: Vec<Duration>,
}

impl QuantumGAN {
    /// Create a new Quantum GAN
    pub fn new(config: QGanConfig) -> QmlResult<Self> {
        // Create generator: latent -> data
        let generator = QuantumNeuralNetwork::new(
            &[config.latent_dim, config.data_dim * 2, config.data_dim],
            QnnConfig {
                learning_rate: config.generator_lr,
                seed: config.seed,
                ..Default::default()
            },
        )?;

        // Create discriminator: data -> 1 (real/fake probability)
        let discriminator = QuantumNeuralNetwork::new(
            &[config.data_dim, config.data_dim / 2, 1],
            QnnConfig {
                learning_rate: config.discriminator_lr,
                seed: config.seed.map(|s| s + 1),
                ..Default::default()
            },
        )?;

        Ok(Self {
            generator,
            discriminator,
            config,
            training_history: QGanTrainingHistory {
                generator_losses: Vec::new(),
                discriminator_losses: Vec::new(),
                epoch_times: Vec::new(),
            },
        })
    }

    /// Train the Quantum GAN
    pub fn train(&mut self, real_data: &[Vec<f64>]) -> QmlResult<()> {
        println!("Training Quantum GAN for {} epochs", self.config.epochs);

        let mut rng = match self.config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };

        for epoch in 0..self.config.epochs {
            let start = Instant::now();

            // Train discriminator
            let d_loss = self.train_discriminator(real_data, &mut rng)?;

            // Train generator
            let g_loss = self.train_generator(&mut rng)?;

            self.training_history.generator_losses.push(g_loss);
            self.training_history.discriminator_losses.push(d_loss);
            self.training_history.epoch_times.push(start.elapsed());

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: G_loss = {:.4}, D_loss = {:.4}",
                    epoch, g_loss, d_loss
                );
            }
        }

        Ok(())
    }

    /// Train discriminator
    fn train_discriminator(
        &mut self,
        real_data: &[Vec<f64>],
        rng: &mut ChaCha8Rng,
    ) -> QmlResult<f64> {
        let batch_size = self.config.batch_size.min(real_data.len());

        // Create training data for discriminator
        let mut d_training_data = Vec::new();

        // Real data (label = 1)
        for _ in 0..batch_size / 2 {
            let idx = rng.gen_range(0..real_data.len());
            d_training_data.push((real_data[idx].clone(), vec![1.0]));
        }

        // Fake data (label = 0)
        for _ in 0..batch_size / 2 {
            let fake_sample = self.generate_sample(rng)?;
            d_training_data.push((fake_sample, vec![0.0]));
        }

        // Train discriminator
        self.discriminator.train(&d_training_data)?;

        // Calculate discriminator loss
        self.discriminator.calculate_loss(&d_training_data)
    }

    /// Train generator
    fn train_generator(&mut self, rng: &mut ChaCha8Rng) -> QmlResult<f64> {
        let batch_size = self.config.batch_size;

        // Create training data for generator (trying to fool discriminator)
        let mut g_training_data = Vec::new();

        for _ in 0..batch_size {
            let latent: Vec<f64> = (0..self.config.latent_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            // Generator tries to produce data that discriminator labels as real (1)
            g_training_data.push((latent, vec![1.0]));
        }

        // Train generator
        self.generator.train(&g_training_data)?;

        // Calculate generator loss
        self.generator.calculate_loss(&g_training_data)
    }

    /// Generate a sample from random noise
    pub fn generate_sample(&self, rng: &mut ChaCha8Rng) -> QmlResult<Vec<f64>> {
        let latent: Vec<f64> = (0..self.config.latent_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        self.generator.forward(&latent)
    }

    /// Generate multiple samples
    pub fn generate_samples(
        &self,
        num_samples: usize,
        rng: &mut ChaCha8Rng,
    ) -> QmlResult<Vec<Vec<f64>>> {
        let mut samples = Vec::new();

        for _ in 0..num_samples {
            samples.push(self.generate_sample(rng)?);
        }

        Ok(samples)
    }
}

/// Quantum Reinforcement Learning Agent
#[derive(Debug, Clone)]
pub struct QuantumRLAgent {
    /// Policy network
    pub policy_network: QuantumNeuralNetwork,
    /// Value network (for actor-critic)
    pub value_network: Option<QuantumNeuralNetwork>,
    /// Agent configuration
    pub config: QRLConfig,
    /// Experience replay buffer
    pub experience_buffer: Vec<Experience>,
    /// Training statistics
    pub stats: QRLStats,
}

/// Configuration for Quantum RL Agent
#[derive(Debug, Clone)]
pub struct QRLConfig {
    /// State dimension
    pub state_dim: usize,
    /// Action dimension
    pub action_dim: usize,
    /// Buffer capacity
    pub buffer_capacity: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor
    pub gamma: f64,
    /// Exploration rate
    pub epsilon: f64,
    /// Use actor-critic
    pub use_actor_critic: bool,
    /// Random seed
    pub seed: Option<u64>,
}

/// Experience tuple for reinforcement learning
#[derive(Debug, Clone)]
pub struct Experience {
    /// Current state
    pub state: Vec<f64>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: Vec<f64>,
    /// Episode done flag
    pub done: bool,
}

/// Training statistics for Quantum RL
#[derive(Debug, Clone)]
pub struct QRLStats {
    /// Episode rewards
    pub episode_rewards: Vec<f64>,
    /// Episode lengths
    pub episode_lengths: Vec<usize>,
    /// Training losses
    pub losses: Vec<f64>,
}

impl QuantumRLAgent {
    /// Create a new Quantum RL Agent
    pub fn new(config: QRLConfig) -> QmlResult<Self> {
        // Create policy network
        let policy_network = QuantumNeuralNetwork::new(
            &[config.state_dim, config.state_dim * 2, config.action_dim],
            QnnConfig {
                learning_rate: config.learning_rate,
                seed: config.seed,
                ..Default::default()
            },
        )?;

        // Create value network if using actor-critic
        let value_network = if config.use_actor_critic {
            Some(QuantumNeuralNetwork::new(
                &[config.state_dim, config.state_dim, 1],
                QnnConfig {
                    learning_rate: config.learning_rate,
                    seed: config.seed.map(|s| s + 1),
                    ..Default::default()
                },
            )?)
        } else {
            None
        };

        Ok(Self {
            policy_network,
            value_network,
            config,
            experience_buffer: Vec::new(),
            stats: QRLStats {
                episode_rewards: Vec::new(),
                episode_lengths: Vec::new(),
                losses: Vec::new(),
            },
        })
    }

    /// Select action using current policy
    pub fn select_action(&self, state: &[f64], rng: &mut ChaCha8Rng) -> QmlResult<usize> {
        // Epsilon-greedy action selection
        if rng.gen::<f64>() < self.config.epsilon {
            // Random exploration
            Ok(rng.gen_range(0..self.config.action_dim))
        } else {
            // Policy-based action
            let action_values = self.policy_network.forward(state)?;

            // Select action with highest value
            let best_action = action_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            Ok(best_action)
        }
    }

    /// Store experience in replay buffer
    pub fn store_experience(&mut self, experience: Experience) {
        self.experience_buffer.push(experience);

        // Maintain buffer capacity
        if self.experience_buffer.len() > self.config.buffer_capacity {
            self.experience_buffer.remove(0);
        }
    }

    /// Train the agent using stored experiences
    pub fn train(&mut self) -> QmlResult<()> {
        if self.experience_buffer.len() < 32 {
            return Ok(()); // Need sufficient experience
        }

        // Create training data from experience buffer
        let mut policy_training_data = Vec::new();

        for experience in &self.experience_buffer {
            // Compute target using Bellman equation
            let target_value = if experience.done {
                experience.reward
            } else {
                let next_values = self.policy_network.forward(&experience.next_state)?;
                let max_next_value = next_values
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&0.0);
                experience.reward + self.config.gamma * max_next_value
            };

            // Create target vector
            let mut target = vec![0.0; self.config.action_dim];
            target[experience.action] = target_value;

            policy_training_data.push((experience.state.clone(), target));
        }

        // Train policy network
        self.policy_network.train(&policy_training_data)?;

        // Train value network if using actor-critic
        if let Some(ref mut value_net) = self.value_network {
            let mut value_training_data = Vec::new();

            for experience in &self.experience_buffer {
                let target_value = if experience.done {
                    experience.reward
                } else {
                    experience.reward
                        + self.config.gamma * value_net.forward(&experience.next_state)?[0]
                };

                value_training_data.push((experience.state.clone(), vec![target_value]));
            }

            value_net.train(&value_training_data)?;
        }

        Ok(())
    }
}

/// Quantum Autoencoder for dimensionality reduction
#[derive(Debug, Clone)]
pub struct QuantumAutoencoder {
    /// Encoder network
    pub encoder: QuantumNeuralNetwork,
    /// Decoder network
    pub decoder: QuantumNeuralNetwork,
    /// Configuration
    pub config: QAutoencoderConfig,
    /// Training history
    pub training_history: TrainingHistory,
}

/// Configuration for Quantum Autoencoder
#[derive(Debug, Clone)]
pub struct QAutoencoderConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Latent dimension
    pub latent_dim: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl QuantumAutoencoder {
    /// Create a new Quantum Autoencoder
    pub fn new(config: QAutoencoderConfig) -> QmlResult<Self> {
        // Create encoder: input -> latent
        let encoder = QuantumNeuralNetwork::new(
            &[config.input_dim, config.input_dim / 2, config.latent_dim],
            QnnConfig {
                learning_rate: config.learning_rate,
                seed: config.seed,
                ..Default::default()
            },
        )?;

        // Create decoder: latent -> input
        let decoder = QuantumNeuralNetwork::new(
            &[config.latent_dim, config.input_dim / 2, config.input_dim],
            QnnConfig {
                learning_rate: config.learning_rate,
                seed: config.seed.map(|s| s + 1),
                ..Default::default()
            },
        )?;

        Ok(Self {
            encoder,
            decoder,
            config,
            training_history: TrainingHistory::new(),
        })
    }

    /// Encode input to latent representation
    pub fn encode(&self, input: &[f64]) -> QmlResult<Vec<f64>> {
        self.encoder.forward(input)
    }

    /// Decode latent representation to output
    pub fn decode(&self, latent: &[f64]) -> QmlResult<Vec<f64>> {
        self.decoder.forward(latent)
    }

    /// Full forward pass (encode then decode)
    pub fn forward(&self, input: &[f64]) -> QmlResult<Vec<f64>> {
        let latent = self.encode(input)?;
        self.decode(&latent)
    }

    /// Train the autoencoder
    pub fn train(&mut self, training_data: &[Vec<f64>]) -> QmlResult<()> {
        println!(
            "Training Quantum Autoencoder for {} epochs",
            self.config.epochs
        );

        for epoch in 0..self.config.epochs {
            let start = Instant::now();

            // Create autoencoder training data (input -> input)
            let ae_training_data: Vec<(Vec<f64>, Vec<f64>)> = training_data
                .iter()
                .map(|sample| (sample.clone(), sample.clone()))
                .collect();

            // Train encoder and decoder jointly
            self.encoder.train(&ae_training_data)?;
            self.decoder.train(&ae_training_data)?;

            // Calculate reconstruction loss
            let mut total_loss = 0.0;
            for sample in training_data {
                let reconstructed = self.forward(sample)?;
                let loss: f64 = sample
                    .iter()
                    .zip(reconstructed.iter())
                    .map(|(orig, recon)| (orig - recon).powi(2))
                    .sum();
                total_loss += loss;
            }
            total_loss /= training_data.len() as f64;

            self.training_history.losses.push(total_loss);
            self.training_history.iteration_times.push(start.elapsed());

            if epoch % 10 == 0 {
                println!("Epoch {}: Reconstruction Loss = {:.6}", epoch, total_loss);
            }
        }

        Ok(())
    }
}

/// Performance metrics for quantum machine learning models
#[derive(Debug, Clone)]
pub struct QmlMetrics {
    /// Training accuracy
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Training time
    pub training_time: Duration,
    /// Number of parameters
    pub num_parameters: usize,
    /// Quantum advantage estimate
    pub quantum_advantage: f64,
    /// Model complexity
    pub complexity_score: f64,
}

/// Utility functions for quantum machine learning

/// Create a simple VQC for binary classification
pub fn create_binary_classifier(
    num_features: usize,
    num_qubits: usize,
    ansatz_layers: usize,
) -> QmlResult<VariationalQuantumClassifier> {
    let config = VqcConfig {
        max_iterations: 500,
        learning_rate: 0.01,
        num_shots: 1024,
        ..Default::default()
    };

    VariationalQuantumClassifier::new(num_features, num_qubits, 2, ansatz_layers, config)
}

/// Create a quantum feature map for data encoding
pub fn create_zz_feature_map(
    num_features: usize,
    repetitions: usize,
) -> QmlResult<QuantumFeatureMap> {
    QuantumFeatureMap::new(
        num_features,
        num_features,
        FeatureMapType::ZZFeatureMap { repetitions },
    )
}

/// Create a quantum kernel SVM
pub fn create_quantum_svm(feature_map: QuantumFeatureMap, c_parameter: f64) -> QuantumKernelMethod {
    QuantumKernelMethod::new(
        feature_map,
        KernelMethodType::SupportVectorMachine { c_parameter },
    )
}

/// Evaluate model performance
pub fn evaluate_qml_model<F>(model: F, test_data: &[(Vec<f64>, usize)]) -> QmlResult<QmlMetrics>
where
    F: Fn(&[f64]) -> QmlResult<usize>,
{
    let start = Instant::now();
    let mut correct = 0;
    let mut total = 0;

    for (features, true_label) in test_data {
        let predicted_label = model(features)?;
        if predicted_label == *true_label {
            correct += 1;
        }
        total += 1;
    }

    let accuracy = correct as f64 / total as f64;
    let training_time = start.elapsed();

    Ok(QmlMetrics {
        training_accuracy: accuracy,
        validation_accuracy: accuracy,
        training_loss: 0.0, // Would need access to model internals
        validation_loss: 0.0,
        training_time,
        num_parameters: 0,      // Would need access to model internals
        quantum_advantage: 1.2, // Placeholder
        complexity_score: 0.5,  // Placeholder
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_circuit_creation() {
        let circuit = QuantumCircuit::hardware_efficient_ansatz(4, 2);
        assert_eq!(circuit.num_qubits, 4);
        assert_eq!(circuit.depth, 2);
        assert!(circuit.num_parameters > 0);
    }

    #[test]
    fn test_quantum_feature_map() {
        let feature_map = QuantumFeatureMap::new(3, 4, FeatureMapType::AngleEncoding).unwrap();

        assert_eq!(feature_map.num_features, 3);
        assert_eq!(feature_map.num_qubits, 4);

        let data = vec![1.0, 0.5, -0.5];
        let encoded = feature_map.encode(&data).unwrap();
        assert_eq!(encoded.len(), 4); // Returns num_qubits parameters for AngleEncoding
    }

    #[test]
    fn test_vqc_creation() {
        let vqc = VariationalQuantumClassifier::new(4, 4, 2, 2, VqcConfig::default()).unwrap();

        assert_eq!(vqc.num_classes, 2);
        assert_eq!(vqc.feature_map.num_features, 4);
    }

    #[test]
    fn test_quantum_neural_network() {
        let qnn = QuantumNeuralNetwork::new(&[3, 4, 2], QnnConfig::default()).unwrap();

        assert_eq!(qnn.layers.len(), 2);

        let input = vec![0.5, -0.3, 0.8];
        let output = qnn.forward(&input).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_quantum_kernel_method() {
        let feature_map = QuantumFeatureMap::new(2, 2, FeatureMapType::AngleEncoding).unwrap();

        let kernel_method = QuantumKernelMethod::new(
            feature_map,
            KernelMethodType::SupportVectorMachine { c_parameter: 1.0 },
        );

        let x1 = vec![0.5, 0.3];
        let x2 = vec![0.7, 0.1];
        let kernel_val = kernel_method.quantum_kernel(&x1, &x2).unwrap();

        assert!(kernel_val >= 0.0);
        assert!(kernel_val <= 1.0);
    }

    #[test]
    fn test_quantum_autoencoder() {
        let config = QAutoencoderConfig {
            input_dim: 8,
            latent_dim: 3,
            learning_rate: 0.01,
            epochs: 5,
            batch_size: 16,
            seed: Some(42),
        };

        let autoencoder = QuantumAutoencoder::new(config).unwrap();

        let input = vec![1.0, 0.5, -0.5, 0.3, 0.8, -0.2, 0.6, -0.8];
        let latent = autoencoder.encode(&input).unwrap();
        assert_eq!(latent.len(), 3);

        let reconstructed = autoencoder.decode(&latent).unwrap();
        assert_eq!(reconstructed.len(), 8);
    }

    #[test]
    fn test_helper_functions() {
        let vqc = create_binary_classifier(4, 4, 2).unwrap();
        assert_eq!(vqc.num_classes, 2);

        let feature_map = create_zz_feature_map(3, 2).unwrap();
        assert_eq!(feature_map.num_features, 3);

        let kernel_svm = create_quantum_svm(feature_map, 1.0);
        assert!(matches!(
            kernel_svm.method_type,
            KernelMethodType::SupportVectorMachine { .. }
        ));
    }
}
