//! Quantum Neural Networks
//!
//! This module implements various quantum neural network architectures including
//! parameterized quantum circuits, quantum convolutional networks, and hybrid models.

use super::*;
use crate::{CircuitResult, DeviceError, DeviceResult, QuantumDevice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Quantum Neural Network trait
pub trait QuantumNeuralNetwork: Send + Sync {
    /// Forward pass through the network
    fn forward(&self, input: &[f64]) -> DeviceResult<Vec<f64>>;

    /// Get trainable parameters
    fn parameters(&self) -> &[f64];

    /// Set trainable parameters
    fn set_parameters(&mut self, params: Vec<f64>) -> DeviceResult<()>;

    /// Get parameter count
    fn parameter_count(&self) -> usize;

    /// Get network architecture description
    fn architecture(&self) -> QNNArchitecture;
}

/// QNN Architecture description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNNArchitecture {
    pub network_type: QNNType,
    pub num_qubits: usize,
    pub num_layers: usize,
    pub num_parameters: usize,
    pub input_encoding: InputEncoding,
    pub output_decoding: OutputDecoding,
    pub entangling_strategy: EntanglingStrategy,
}

/// Types of quantum neural networks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QNNType {
    /// Parameterized Quantum Circuit
    PQC,
    /// Quantum Convolutional Neural Network
    QCNN,
    /// Variational Quantum Classifier
    VQC,
    /// Quantum Generative Adversarial Network
    QGAN,
    /// Hybrid Classical-Quantum Network
    HybridCQN,
    /// Quantum Recurrent Neural Network
    QRNN,
}

/// Input encoding strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputEncoding {
    /// Amplitude encoding
    Amplitude,
    /// Angle encoding
    Angle,
    /// Basis encoding
    Basis,
    /// Coherent state encoding (for CV systems)
    CoherentState,
    /// Displacement encoding
    Displacement,
}

/// Output decoding strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputDecoding {
    /// Expectation value of Pauli operators
    PauliExpectation,
    /// Measurement probabilities
    Probabilities,
    /// Fidelity measurement
    Fidelity,
    /// Coherent state measurement
    CoherentMeasurement,
}

/// Entangling strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntanglingStrategy {
    Linear,
    Circular,
    AllToAll,
    Random,
    Hardware,
    Custom(Vec<(usize, usize)>),
}

/// Parameterized Quantum Circuit Network
pub struct PQCNetwork {
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    num_qubits: usize,
    num_layers: usize,
    parameters: Vec<f64>,
    input_encoding: InputEncoding,
    output_decoding: OutputDecoding,
    entangling_strategy: EntanglingStrategy,
    measurement_operators: Vec<PauliOperator>,
}

impl PQCNetwork {
    /// Create a new PQC network
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        num_qubits: usize,
        num_layers: usize,
        input_encoding: InputEncoding,
        output_decoding: OutputDecoding,
        entangling_strategy: EntanglingStrategy,
    ) -> Self {
        let parameter_count = Self::calculate_parameter_count(num_qubits, num_layers);
        let parameters = (0..parameter_count)
            .map(|_| fastrand::f64() * 2.0 * std::f64::consts::PI)
            .collect();

        let measurement_operators = (0..num_qubits).map(|_| PauliOperator::Z).collect();

        Self {
            device,
            num_qubits,
            num_layers,
            parameters,
            input_encoding,
            output_decoding,
            entangling_strategy,
            measurement_operators,
        }
    }

    const fn calculate_parameter_count(num_qubits: usize, num_layers: usize) -> usize {
        // Each layer has 3 rotation gates per qubit
        3 * num_qubits * num_layers
    }

    /// Build the quantum circuit for given input
    pub async fn build_circuit(&self, input: &[f64]) -> DeviceResult<ParameterizedQuantumCircuit> {
        let mut circuit = ParameterizedQuantumCircuit::new(self.num_qubits);

        // Input encoding
        self.encode_input(&mut circuit, input).await?;

        // Parameterized layers
        let mut param_idx = 0;
        for layer in 0..self.num_layers {
            // Rotation gates
            for qubit in 0..self.num_qubits {
                circuit.add_rx_gate(qubit, self.parameters[param_idx])?;
                param_idx += 1;
                circuit.add_ry_gate(qubit, self.parameters[param_idx])?;
                param_idx += 1;
                circuit.add_rz_gate(qubit, self.parameters[param_idx])?;
                param_idx += 1;
            }

            // Entangling gates
            self.add_entangling_gates(&mut circuit, layer).await?;
        }

        Ok(circuit)
    }

    async fn encode_input(
        &self,
        circuit: &mut ParameterizedQuantumCircuit,
        input: &[f64],
    ) -> DeviceResult<()> {
        match self.input_encoding {
            InputEncoding::Angle => {
                // Encode input as rotation angles
                let padded_input = self.pad_input(input, self.num_qubits);
                for (qubit, &value) in padded_input.iter().enumerate() {
                    circuit.add_ry_gate(qubit, value)?;
                }
            }
            InputEncoding::Amplitude => {
                // Amplitude encoding requires state preparation
                // This is a simplified implementation
                for qubit in 0..self.num_qubits {
                    circuit.add_h_gate(qubit)?;
                }
                // Would need more sophisticated amplitude encoding
            }
            InputEncoding::Basis => {
                // Basis encoding: encode classical bits as computational basis states
                let binary_input = self.convert_to_binary(input);
                for (qubit, &bit) in binary_input.iter().enumerate() {
                    if bit == 1 {
                        circuit.add_x_gate(qubit)?;
                    }
                }
            }
            _ => {
                return Err(DeviceError::InvalidInput(format!(
                    "Input encoding {:?} not implemented for PQC",
                    self.input_encoding
                )));
            }
        }
        Ok(())
    }

    async fn add_entangling_gates(
        &self,
        circuit: &mut ParameterizedQuantumCircuit,
        _layer: usize,
    ) -> DeviceResult<()> {
        match &self.entangling_strategy {
            EntanglingStrategy::Linear => {
                for qubit in 0..self.num_qubits - 1 {
                    circuit.add_cnot_gate(qubit, qubit + 1)?;
                }
            }
            EntanglingStrategy::Circular => {
                for qubit in 0..self.num_qubits - 1 {
                    circuit.add_cnot_gate(qubit, qubit + 1)?;
                }
                if self.num_qubits > 2 {
                    circuit.add_cnot_gate(self.num_qubits - 1, 0)?;
                }
            }
            EntanglingStrategy::AllToAll => {
                for i in 0..self.num_qubits {
                    for j in i + 1..self.num_qubits {
                        circuit.add_cnot_gate(i, j)?;
                    }
                }
            }
            EntanglingStrategy::Custom(connections) => {
                for &(control, target) in connections {
                    if control < self.num_qubits && target < self.num_qubits {
                        circuit.add_cnot_gate(control, target)?;
                    }
                }
            }
            _ => {
                // Default to linear
                for qubit in 0..self.num_qubits - 1 {
                    circuit.add_cnot_gate(qubit, qubit + 1)?;
                }
            }
        }
        Ok(())
    }

    fn pad_input(&self, input: &[f64], target_size: usize) -> Vec<f64> {
        let mut padded = input.to_vec();
        while padded.len() < target_size {
            padded.push(0.0);
        }
        padded.truncate(target_size);
        padded
    }

    fn convert_to_binary(&self, input: &[f64]) -> Vec<u8> {
        let mut binary = Vec::new();
        for &value in input {
            let int_value = (value * 255.0) as u8;
            for i in 0..8 {
                binary.push((int_value >> i) & 1);
                if binary.len() >= self.num_qubits {
                    break;
                }
            }
            if binary.len() >= self.num_qubits {
                break;
            }
        }
        while binary.len() < self.num_qubits {
            binary.push(0);
        }
        binary.truncate(self.num_qubits);
        binary
    }

    async fn decode_output(&self, circuit_result: &CircuitResult) -> DeviceResult<Vec<f64>> {
        match self.output_decoding {
            OutputDecoding::PauliExpectation => {
                // Compute expectation values of Pauli operators
                let mut expectations = Vec::new();
                for (qubit, pauli_op) in self.measurement_operators.iter().enumerate() {
                    let expectation =
                        self.compute_pauli_expectation(circuit_result, qubit, pauli_op)?;
                    expectations.push(expectation);
                }
                Ok(expectations)
            }
            OutputDecoding::Probabilities => {
                // Convert measurement counts to probabilities
                let total_shots = circuit_result.shots as f64;
                let mut probs = Vec::new();

                for i in 0..self.num_qubits {
                    let mut prob_one = 0.0;
                    for (bitstring, count) in &circuit_result.counts {
                        if let Some(bit_char) = bitstring.chars().nth(i) {
                            if bit_char == '1' {
                                prob_one += *count as f64 / total_shots;
                            }
                        }
                    }
                    probs.push(prob_one);
                }
                Ok(probs)
            }
            _ => Err(DeviceError::InvalidInput(format!(
                "Output decoding {:?} not implemented",
                self.output_decoding
            ))),
        }
    }

    fn compute_pauli_expectation(
        &self,
        circuit_result: &CircuitResult,
        qubit: usize,
        pauli_op: &PauliOperator,
    ) -> DeviceResult<f64> {
        let mut expectation = 0.0;
        let total_shots = circuit_result.shots as f64;

        for (bitstring, count) in &circuit_result.counts {
            let probability = *count as f64 / total_shots;

            let eigenvalue = if let Some(bit_char) = bitstring.chars().nth(qubit) {
                match pauli_op {
                    PauliOperator::Z => {
                        if bit_char == '0' {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    PauliOperator::X | PauliOperator::Y => {
                        // Would need different measurement basis
                        return Err(DeviceError::InvalidInput(
                            "X and Y Pauli measurements require basis rotation".to_string(),
                        ));
                    }
                    PauliOperator::I => 1.0,
                }
            } else {
                0.0
            };

            expectation += probability * eigenvalue;
        }

        Ok(expectation)
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

impl QuantumNeuralNetwork for PQCNetwork {
    fn forward(&self, input: &[f64]) -> DeviceResult<Vec<f64>> {
        // This would need to be async in practice
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            DeviceError::ExecutionFailed(format!("Failed to create tokio runtime: {e}"))
        })?;
        rt.block_on(async {
            let circuit = self.build_circuit(input).await?;
            let device = self.device.read().await;
            let result = Self::execute_circuit_helper(&*device, &circuit, 1024).await?;
            self.decode_output(&result).await
        })
    }

    fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    fn set_parameters(&mut self, params: Vec<f64>) -> DeviceResult<()> {
        if params.len() != self.parameters.len() {
            return Err(DeviceError::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.parameters.len(),
                params.len()
            )));
        }
        self.parameters = params;
        Ok(())
    }

    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    fn architecture(&self) -> QNNArchitecture {
        QNNArchitecture {
            network_type: QNNType::PQC,
            num_qubits: self.num_qubits,
            num_layers: self.num_layers,
            num_parameters: self.parameters.len(),
            input_encoding: self.input_encoding.clone(),
            output_decoding: self.output_decoding.clone(),
            entangling_strategy: self.entangling_strategy.clone(),
        }
    }
}

/// Quantum Convolutional Neural Network
pub struct QCNN {
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    num_qubits: usize,
    conv_layers: Vec<QConvLayer>,
    pooling_layers: Vec<QPoolingLayer>,
    parameters: Vec<f64>,
    input_encoding: InputEncoding,
}

/// Quantum Convolutional Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QConvLayer {
    pub kernel_size: usize,
    pub stride: usize,
    pub num_filters: usize,
    pub parameter_indices: Vec<usize>,
}

/// Quantum Pooling Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QPoolingLayer {
    pub pool_size: usize,
    pub pool_type: QPoolingType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QPoolingType {
    Max,
    Average,
    Measurement,
}

impl QCNN {
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        num_qubits: usize,
        conv_layers: Vec<QConvLayer>,
        pooling_layers: Vec<QPoolingLayer>,
        input_encoding: InputEncoding,
    ) -> Self {
        let total_params = conv_layers.iter()
            .map(|layer| layer.num_filters * layer.kernel_size * 3) // 3 rotation gates per kernel
            .sum();

        let parameters = (0..total_params)
            .map(|_| fastrand::f64() * 2.0 * std::f64::consts::PI)
            .collect();

        Self {
            device,
            num_qubits,
            conv_layers,
            pooling_layers,
            parameters,
            input_encoding,
        }
    }

    pub async fn build_circuit(&self, input: &[f64]) -> DeviceResult<ParameterizedQuantumCircuit> {
        let mut circuit = ParameterizedQuantumCircuit::new(self.num_qubits);

        // Input encoding
        self.encode_input(&mut circuit, input).await?;

        let mut current_qubits = self.num_qubits;

        // Apply convolutional and pooling layers alternately
        for (conv_layer, pool_layer) in self.conv_layers.iter().zip(self.pooling_layers.iter()) {
            // Apply convolutional layer
            self.apply_conv_layer(&mut circuit, conv_layer, current_qubits)
                .await?;

            // Apply pooling layer
            current_qubits = self
                .apply_pooling_layer(&mut circuit, pool_layer, current_qubits)
                .await?;
        }

        Ok(circuit)
    }

    async fn encode_input(
        &self,
        circuit: &mut ParameterizedQuantumCircuit,
        input: &[f64],
    ) -> DeviceResult<()> {
        match self.input_encoding {
            InputEncoding::Angle => {
                let padded_input = self.pad_input(input, self.num_qubits);
                for (qubit, &value) in padded_input.iter().enumerate() {
                    circuit.add_ry_gate(qubit, value)?;
                }
            }
            InputEncoding::Amplitude => {
                // Initialize in superposition
                for qubit in 0..self.num_qubits {
                    circuit.add_h_gate(qubit)?;
                }
            }
            _ => {
                return Err(DeviceError::InvalidInput(format!(
                    "Input encoding {:?} not implemented for QCNN",
                    self.input_encoding
                )));
            }
        }
        Ok(())
    }

    async fn apply_conv_layer(
        &self,
        circuit: &mut ParameterizedQuantumCircuit,
        layer: &QConvLayer,
        num_active_qubits: usize,
    ) -> DeviceResult<()> {
        let num_windows = (num_active_qubits - layer.kernel_size) / layer.stride + 1;

        for window in 0..num_windows {
            let start_qubit = window * layer.stride;

            for filter in 0..layer.num_filters {
                let param_offset = filter * layer.kernel_size * 3;

                // Apply parameterized gates to qubits in the window
                for i in 0..layer.kernel_size {
                    let qubit = start_qubit + i;
                    let param_base = param_offset + i * 3;

                    if param_base + 2 < self.parameters.len() {
                        circuit.add_rx_gate(qubit, self.parameters[param_base])?;
                        circuit.add_ry_gate(qubit, self.parameters[param_base + 1])?;
                        circuit.add_rz_gate(qubit, self.parameters[param_base + 2])?;
                    }
                }

                // Apply entangling gates within the window
                for i in 0..layer.kernel_size - 1 {
                    let control = start_qubit + i;
                    let target = start_qubit + i + 1;
                    circuit.add_cnot_gate(control, target)?;
                }
            }
        }

        Ok(())
    }

    async fn apply_pooling_layer(
        &self,
        circuit: &mut ParameterizedQuantumCircuit,
        layer: &QPoolingLayer,
        num_active_qubits: usize,
    ) -> DeviceResult<usize> {
        let num_pools = num_active_qubits / layer.pool_size;

        match layer.pool_type {
            QPoolingType::Measurement => {
                // Measure and discard some qubits (simplified)
                // In practice, this would involve partial measurements
                Ok(num_pools)
            }
            QPoolingType::Max | QPoolingType::Average => {
                // Apply pooling unitaries (simplified)
                for pool in 0..num_pools {
                    let start_qubit = pool * layer.pool_size;

                    // Apply pooling gates
                    for i in 0..layer.pool_size - 1 {
                        let qubit1 = start_qubit + i;
                        let qubit2 = start_qubit + i + 1;
                        circuit.add_cnot_gate(qubit1, qubit2)?;
                    }
                }
                Ok(num_pools)
            }
        }
    }

    fn pad_input(&self, input: &[f64], target_size: usize) -> Vec<f64> {
        let mut padded = input.to_vec();
        while padded.len() < target_size {
            padded.push(0.0);
        }
        padded.truncate(target_size);
        padded
    }
}

impl QuantumNeuralNetwork for QCNN {
    fn forward(&self, input: &[f64]) -> DeviceResult<Vec<f64>> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            DeviceError::ExecutionFailed(format!("Failed to create tokio runtime: {e}"))
        })?;
        rt.block_on(async {
            let circuit = self.build_circuit(input).await?;
            let device = self.device.read().await;
            let result = Self::execute_circuit_helper(&*device, &circuit, 1024).await?;

            // Simple output decoding for QCNN
            let mut output = Vec::new();
            let total_shots = result.shots as f64;

            for i in 0..self.num_qubits.min(8) {
                // Limit output size
                let mut prob_one = 0.0;
                for (bitstring, count) in &result.counts {
                    if let Some(bit_char) = bitstring.chars().nth(i) {
                        if bit_char == '1' {
                            prob_one += *count as f64 / total_shots;
                        }
                    }
                }
                output.push(prob_one);
            }

            Ok(output)
        })
    }

    fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    fn set_parameters(&mut self, params: Vec<f64>) -> DeviceResult<()> {
        if params.len() != self.parameters.len() {
            return Err(DeviceError::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.parameters.len(),
                params.len()
            )));
        }
        self.parameters = params;
        Ok(())
    }

    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    fn architecture(&self) -> QNNArchitecture {
        QNNArchitecture {
            network_type: QNNType::QCNN,
            num_qubits: self.num_qubits,
            num_layers: self.conv_layers.len(),
            num_parameters: self.parameters.len(),
            input_encoding: self.input_encoding.clone(),
            output_decoding: OutputDecoding::Probabilities,
            entangling_strategy: EntanglingStrategy::Linear,
        }
    }
}

impl QCNN {
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

/// Variational Quantum Classifier
pub struct VQC {
    pqc_network: PQCNetwork,
    class_mapping: HashMap<usize, String>,
}

impl VQC {
    pub fn new(
        device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
        num_qubits: usize,
        num_layers: usize,
        num_classes: usize,
    ) -> Self {
        let pqc_network = PQCNetwork::new(
            device,
            num_qubits,
            num_layers,
            InputEncoding::Angle,
            OutputDecoding::PauliExpectation,
            EntanglingStrategy::Linear,
        );

        let class_mapping = (0..num_classes)
            .map(|i| (i, format!("class_{i}")))
            .collect();

        Self {
            pqc_network,
            class_mapping,
        }
    }

    pub fn classify(&self, input: &[f64]) -> DeviceResult<ClassificationResult> {
        let raw_output = self.pqc_network.forward(input)?;

        // Convert quantum outputs to class probabilities
        let class_probs = self.softmax(&raw_output);

        // Find predicted class
        let (predicted_class, confidence) = class_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((0, 0.0), |(idx, &prob)| (idx, prob));

        let class_name = self
            .class_mapping
            .get(&predicted_class)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        Ok(ClassificationResult {
            predicted_class,
            class_name,
            confidence,
            class_probabilities: class_probs,
        })
    }

    fn softmax(&self, values: &[f64]) -> Vec<f64> {
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f64> = values.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum_exp).collect()
    }
}

impl QuantumNeuralNetwork for VQC {
    fn forward(&self, input: &[f64]) -> DeviceResult<Vec<f64>> {
        self.pqc_network.forward(input)
    }

    fn parameters(&self) -> &[f64] {
        self.pqc_network.parameters()
    }

    fn set_parameters(&mut self, params: Vec<f64>) -> DeviceResult<()> {
        self.pqc_network.set_parameters(params)
    }

    fn parameter_count(&self) -> usize {
        self.pqc_network.parameter_count()
    }

    fn architecture(&self) -> QNNArchitecture {
        let mut arch = self.pqc_network.architecture();
        arch.network_type = QNNType::VQC;
        arch
    }
}

/// Classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub predicted_class: usize,
    pub class_name: String,
    pub confidence: f64,
    pub class_probabilities: Vec<f64>,
}

/// Create a PQC network for classification
pub fn create_pqc_classifier(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    num_features: usize,
    num_classes: usize,
    num_layers: usize,
) -> DeviceResult<VQC> {
    let num_qubits =
        (num_features as f64).log2().ceil() as usize + (num_classes as f64).log2().ceil() as usize;
    Ok(VQC::new(device, num_qubits, num_layers, num_classes))
}

/// Create a QCNN for image classification
pub fn create_qcnn_classifier(
    device: Arc<RwLock<dyn QuantumDevice + Send + Sync>>,
    image_size: usize,
) -> DeviceResult<QCNN> {
    let num_qubits = (image_size as f64).log2().ceil() as usize;

    let conv_layers = vec![
        QConvLayer {
            kernel_size: 2,
            stride: 1,
            num_filters: 2,
            parameter_indices: (0..12).collect(), // 2 filters * 2 kernel_size * 3 params
        },
        QConvLayer {
            kernel_size: 2,
            stride: 1,
            num_filters: 1,
            parameter_indices: (12..18).collect(), // 1 filter * 2 kernel_size * 3 params
        },
    ];

    let pooling_layers = vec![
        QPoolingLayer {
            pool_size: 2,
            pool_type: QPoolingType::Measurement,
        },
        QPoolingLayer {
            pool_size: 2,
            pool_type: QPoolingType::Measurement,
        },
    ];

    Ok(QCNN::new(
        device,
        num_qubits,
        conv_layers,
        pooling_layers,
        InputEncoding::Angle,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_mock_quantum_device;

    #[test]
    fn test_pqc_network_creation() {
        let device = create_mock_quantum_device();
        let network = PQCNetwork::new(
            device,
            4,
            2,
            InputEncoding::Angle,
            OutputDecoding::PauliExpectation,
            EntanglingStrategy::Linear,
        );

        assert_eq!(network.num_qubits, 4);
        assert_eq!(network.num_layers, 2);
        assert_eq!(network.parameter_count(), 24); // 3 * 4 * 2
    }

    #[test]
    fn test_vqc_creation() {
        let device = create_mock_quantum_device();
        let classifier = VQC::new(device, 4, 2, 3);

        assert_eq!(classifier.class_mapping.len(), 3);
        assert_eq!(classifier.parameter_count(), 24);
    }

    #[test]
    fn test_qcnn_creation() {
        let device = create_mock_quantum_device();
        let conv_layers = vec![QConvLayer {
            kernel_size: 2,
            stride: 1,
            num_filters: 1,
            parameter_indices: (0..6).collect(),
        }];
        let pooling_layers = vec![QPoolingLayer {
            pool_size: 2,
            pool_type: QPoolingType::Max,
        }];

        let qcnn = QCNN::new(device, 4, conv_layers, pooling_layers, InputEncoding::Angle);

        assert_eq!(qcnn.num_qubits, 4);
        assert_eq!(qcnn.parameter_count(), 6);
    }

    #[test]
    fn test_softmax() {
        let classifier = {
            let device = create_mock_quantum_device();
            VQC::new(device, 4, 2, 3)
        };

        let input = vec![1.0, 2.0, 3.0];
        let output = classifier.softmax(&input);

        assert_eq!(output.len(), 3);
        assert!((output.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_parameter_operations() {
        let device = create_mock_quantum_device();
        let mut network = PQCNetwork::new(
            device,
            4,
            2,
            InputEncoding::Angle,
            OutputDecoding::PauliExpectation,
            EntanglingStrategy::Linear,
        );

        let original_params = network.parameters().to_vec();
        let new_params = vec![0.0; network.parameter_count()];

        network
            .set_parameters(new_params.clone())
            .expect("Setting parameters should succeed");
        assert_eq!(network.parameters(), &new_params);
        assert_ne!(network.parameters(), &original_params);

        // Test invalid parameter count
        let invalid_params = vec![0.0; 5];
        assert!(network.set_parameters(invalid_params).is_err());
    }
}
