//! QML Layers Implementation
//!
//! This module provides quantum machine learning layer implementations.

use super::config::{AnsatzType, EntanglementPattern, QMLLayerConfig, RotationGate};
use super::types::{
    AttentionHead, ConvolutionalFilter, DenseConnection, LSTMGate, LSTMGateType, PQCGate,
    PQCGateType, TwoQubitGate,
};
use crate::error::Result;
use crate::statevector::StateVectorSimulator;
use crate::SimulatorError;
use scirs2_core::ndarray::Array1;
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Trait for QML layers
pub trait QMLLayer: std::fmt::Debug + Send + Sync {
    /// Forward pass through the layer
    fn forward(&mut self, input: &Array1<Complex64>) -> Result<Array1<Complex64>>;

    /// Backward pass through the layer
    fn backward(&mut self, gradient: &Array1<f64>) -> Result<Array1<f64>>;

    /// Get layer parameters
    fn get_parameters(&self) -> Array1<f64>;

    /// Set layer parameters
    fn set_parameters(&mut self, parameters: &Array1<f64>);

    /// Get circuit depth
    fn get_depth(&self) -> usize;

    /// Get gate count
    fn get_gate_count(&self) -> usize;

    /// Get number of parameters
    fn get_num_parameters(&self) -> usize;
}

/// Parameterized Quantum Circuit Layer
#[derive(Debug)]
pub struct ParameterizedQuantumCircuitLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Layer configuration
    config: QMLLayerConfig,
    /// Parameters (rotation angles)
    parameters: Array1<f64>,
    /// Circuit structure
    circuit_structure: Vec<PQCGate>,
    /// Internal state vector simulator
    simulator: StateVectorSimulator,
}

impl ParameterizedQuantumCircuitLayer {
    /// Create new PQC layer
    pub fn new(num_qubits: usize, config: QMLLayerConfig) -> Result<Self> {
        let mut layer = Self {
            num_qubits,
            config: config.clone(),
            parameters: Array1::zeros(config.num_parameters),
            circuit_structure: Vec::new(),
            simulator: StateVectorSimulator::new(),
        };

        // Initialize parameters randomly
        layer.initialize_parameters();

        // Build circuit structure
        layer.build_circuit_structure()?;

        Ok(layer)
    }

    /// Initialize parameters randomly
    fn initialize_parameters(&mut self) {
        let mut rng = thread_rng();
        for param in &mut self.parameters {
            *param = rng.random_range(-PI..PI);
        }
    }

    /// Build circuit structure based on ansatz
    fn build_circuit_structure(&mut self) -> Result<()> {
        match self.config.ansatz_type {
            AnsatzType::Hardware => self.build_hardware_efficient_ansatz(),
            AnsatzType::Layered => self.build_layered_ansatz(),
            AnsatzType::BrickWall => self.build_brick_wall_ansatz(),
            _ => Err(SimulatorError::InvalidConfiguration(
                "Ansatz type not implemented".to_string(),
            )),
        }
    }

    /// Build hardware-efficient ansatz
    fn build_hardware_efficient_ansatz(&mut self) -> Result<()> {
        let mut param_idx = 0;

        for _layer in 0..self.config.depth {
            // Single-qubit rotations
            for qubit in 0..self.num_qubits {
                for &gate_type in &self.config.rotation_gates {
                    if param_idx < self.parameters.len() {
                        self.circuit_structure.push(PQCGate {
                            gate_type: PQCGateType::SingleQubit(gate_type),
                            qubits: vec![qubit],
                            parameter_index: Some(param_idx),
                        });
                        param_idx += 1;
                    }
                }
            }

            // Entangling gates
            self.add_entangling_gates(&param_idx);
        }

        Ok(())
    }

    /// Build layered ansatz
    fn build_layered_ansatz(&mut self) -> Result<()> {
        // Similar to hardware-efficient but with different structure
        self.build_hardware_efficient_ansatz()
    }

    /// Build brick-wall ansatz
    fn build_brick_wall_ansatz(&mut self) -> Result<()> {
        let mut param_idx = 0;

        for layer in 0..self.config.depth {
            // Alternating CNOT pattern (brick-wall)
            let offset = layer % 2;
            for i in (offset..self.num_qubits - 1).step_by(2) {
                self.circuit_structure.push(PQCGate {
                    gate_type: PQCGateType::TwoQubit(TwoQubitGate::CNOT),
                    qubits: vec![i, i + 1],
                    parameter_index: None,
                });
            }

            // Single-qubit rotations
            for qubit in 0..self.num_qubits {
                if param_idx < self.parameters.len() {
                    self.circuit_structure.push(PQCGate {
                        gate_type: PQCGateType::SingleQubit(RotationGate::RY),
                        qubits: vec![qubit],
                        parameter_index: Some(param_idx),
                    });
                    param_idx += 1;
                }
            }
        }

        Ok(())
    }

    /// Add entangling gates based on entanglement pattern
    fn add_entangling_gates(&mut self, _param_idx: &usize) {
        match self.config.entanglement_pattern {
            EntanglementPattern::Linear => {
                for i in 0..(self.num_qubits - 1) {
                    self.circuit_structure.push(PQCGate {
                        gate_type: PQCGateType::TwoQubit(TwoQubitGate::CNOT),
                        qubits: vec![i, i + 1],
                        parameter_index: None,
                    });
                }
            }
            EntanglementPattern::Circular => {
                for i in 0..self.num_qubits {
                    let next = (i + 1) % self.num_qubits;
                    self.circuit_structure.push(PQCGate {
                        gate_type: PQCGateType::TwoQubit(TwoQubitGate::CNOT),
                        qubits: vec![i, next],
                        parameter_index: None,
                    });
                }
            }
            EntanglementPattern::AllToAll => {
                for i in 0..self.num_qubits {
                    for j in (i + 1)..self.num_qubits {
                        self.circuit_structure.push(PQCGate {
                            gate_type: PQCGateType::TwoQubit(TwoQubitGate::CNOT),
                            qubits: vec![i, j],
                            parameter_index: None,
                        });
                    }
                }
            }
            _ => {
                // Default to linear
                for i in 0..(self.num_qubits - 1) {
                    self.circuit_structure.push(PQCGate {
                        gate_type: PQCGateType::TwoQubit(TwoQubitGate::CNOT),
                        qubits: vec![i, i + 1],
                        parameter_index: None,
                    });
                }
            }
        }
    }
}

impl QMLLayer for ParameterizedQuantumCircuitLayer {
    fn forward(&mut self, input: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        let mut state = input.clone();

        // Apply each gate in the circuit
        for gate in &self.circuit_structure {
            state = self.apply_gate(&state, gate)?;
        }

        Ok(state)
    }

    fn backward(&mut self, gradient: &Array1<f64>) -> Result<Array1<f64>> {
        // Simplified backward pass - in practice would use automatic differentiation
        Ok(gradient.clone())
    }

    fn get_parameters(&self) -> Array1<f64> {
        self.parameters.clone()
    }

    fn set_parameters(&mut self, parameters: &Array1<f64>) {
        self.parameters = parameters.clone();
    }

    fn get_depth(&self) -> usize {
        self.config.depth
    }

    fn get_gate_count(&self) -> usize {
        self.circuit_structure.len()
    }

    fn get_num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

impl ParameterizedQuantumCircuitLayer {
    /// Apply a single gate to the quantum state
    fn apply_gate(&self, state: &Array1<Complex64>, gate: &PQCGate) -> Result<Array1<Complex64>> {
        match &gate.gate_type {
            PQCGateType::SingleQubit(rotation_gate) => {
                let angle = if let Some(param_idx) = gate.parameter_index {
                    self.parameters[param_idx]
                } else {
                    0.0
                };
                Self::apply_single_qubit_gate(state, gate.qubits[0], *rotation_gate, angle)
            }
            PQCGateType::TwoQubit(two_qubit_gate) => {
                Self::apply_two_qubit_gate(state, gate.qubits[0], gate.qubits[1], *two_qubit_gate)
            }
        }
    }

    /// Apply single-qubit rotation gate
    fn apply_single_qubit_gate(
        state: &Array1<Complex64>,
        qubit: usize,
        gate_type: RotationGate,
        angle: f64,
    ) -> Result<Array1<Complex64>> {
        let state_size = state.len();
        let mut new_state = Array1::zeros(state_size);

        match gate_type {
            RotationGate::RX => {
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();

                for i in 0..state_size {
                    if i & (1 << qubit) == 0 {
                        let j = i | (1 << qubit);
                        if j < state_size {
                            new_state[i] = Complex64::new(cos_half, 0.0) * state[i]
                                + Complex64::new(0.0, -sin_half) * state[j];
                            new_state[j] = Complex64::new(0.0, -sin_half) * state[i]
                                + Complex64::new(cos_half, 0.0) * state[j];
                        }
                    }
                }
            }
            RotationGate::RY => {
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();

                for i in 0..state_size {
                    if i & (1 << qubit) == 0 {
                        let j = i | (1 << qubit);
                        if j < state_size {
                            new_state[i] = Complex64::new(cos_half, 0.0) * state[i]
                                - Complex64::new(sin_half, 0.0) * state[j];
                            new_state[j] = Complex64::new(sin_half, 0.0) * state[i]
                                + Complex64::new(cos_half, 0.0) * state[j];
                        }
                    }
                }
            }
            RotationGate::RZ => {
                let phase_0 = Complex64::from_polar(1.0, -angle / 2.0);
                let phase_1 = Complex64::from_polar(1.0, angle / 2.0);

                for i in 0..state_size {
                    if i & (1 << qubit) == 0 {
                        new_state[i] = phase_0 * state[i];
                    } else {
                        new_state[i] = phase_1 * state[i];
                    }
                }
            }
            _ => {
                return Err(SimulatorError::InvalidGate(
                    "Gate type not implemented".to_string(),
                ))
            }
        }

        Ok(new_state)
    }

    /// Apply two-qubit gate
    fn apply_two_qubit_gate(
        state: &Array1<Complex64>,
        control: usize,
        target: usize,
        gate_type: TwoQubitGate,
    ) -> Result<Array1<Complex64>> {
        let state_size = state.len();
        let mut new_state = state.clone();

        match gate_type {
            TwoQubitGate::CNOT => {
                for i in 0..state_size {
                    if (i & (1 << control)) != 0 {
                        // Control qubit is |1⟩, flip target
                        let j = i ^ (1 << target);
                        new_state[i] = state[j];
                    }
                }
            }
            TwoQubitGate::CZ => {
                for i in 0..state_size {
                    if (i & (1 << control)) != 0 && (i & (1 << target)) != 0 {
                        // Both qubits are |1⟩, apply phase
                        new_state[i] = -state[i];
                    }
                }
            }
            TwoQubitGate::SWAP => {
                for i in 0..state_size {
                    let ctrl_bit = (i & (1 << control)) != 0;
                    let targ_bit = (i & (1 << target)) != 0;
                    if ctrl_bit != targ_bit {
                        // Swap the qubits
                        let j = i ^ (1 << control) ^ (1 << target);
                        new_state[i] = state[j];
                    }
                }
            }
            TwoQubitGate::CPhase => {
                for i in 0..state_size {
                    if (i & (1 << control)) != 0 && (i & (1 << target)) != 0 {
                        // Both qubits are |1⟩, apply phase (similar to CZ)
                        new_state[i] = -state[i];
                    }
                }
            }
        }

        Ok(new_state)
    }
}

/// Quantum Convolutional Layer
#[derive(Debug)]
pub struct QuantumConvolutionalLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Layer configuration
    config: QMLLayerConfig,
    /// Parameters
    parameters: Array1<f64>,
    /// Convolutional structure
    conv_structure: Vec<ConvolutionalFilter>,
}

impl QuantumConvolutionalLayer {
    /// Create new quantum convolutional layer
    pub fn new(num_qubits: usize, config: QMLLayerConfig) -> Result<Self> {
        let mut layer = Self {
            num_qubits,
            config: config.clone(),
            parameters: Array1::zeros(config.num_parameters),
            conv_structure: Vec::new(),
        };

        layer.initialize_parameters();
        layer.build_convolutional_structure()?;

        Ok(layer)
    }

    /// Initialize parameters
    fn initialize_parameters(&mut self) {
        let mut rng = thread_rng();
        for param in &mut self.parameters {
            *param = rng.random_range(-PI..PI);
        }
    }

    /// Build convolutional structure
    fn build_convolutional_structure(&mut self) -> Result<()> {
        // Create sliding window filters
        let filter_size = 2; // 2-qubit filters
        let stride = 1;

        let mut param_idx = 0;
        for start in (0..=(self.num_qubits - filter_size)).step_by(stride) {
            if param_idx + 2 <= self.parameters.len() {
                self.conv_structure.push(ConvolutionalFilter {
                    qubits: vec![start, start + 1],
                    parameter_indices: vec![param_idx, param_idx + 1],
                });
                param_idx += 2;
            }
        }

        Ok(())
    }
}

impl QMLLayer for QuantumConvolutionalLayer {
    fn forward(&mut self, input: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        let mut state = input.clone();

        // Apply convolutional filters
        for filter in &self.conv_structure {
            state = self.apply_convolutional_filter(&state, filter)?;
        }

        Ok(state)
    }

    fn backward(&mut self, gradient: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(gradient.clone())
    }

    fn get_parameters(&self) -> Array1<f64> {
        self.parameters.clone()
    }

    fn set_parameters(&mut self, parameters: &Array1<f64>) {
        self.parameters = parameters.clone();
    }

    fn get_depth(&self) -> usize {
        self.conv_structure.len()
    }

    fn get_gate_count(&self) -> usize {
        self.conv_structure.len() * 4 // Approximate gates per filter
    }

    fn get_num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

impl QuantumConvolutionalLayer {
    /// Apply convolutional filter
    fn apply_convolutional_filter(
        &self,
        state: &Array1<Complex64>,
        filter: &ConvolutionalFilter,
    ) -> Result<Array1<Complex64>> {
        let mut new_state = state.clone();

        // Apply parameterized two-qubit unitaries
        let param1 = self.parameters[filter.parameter_indices[0]];
        let param2 = self.parameters[filter.parameter_indices[1]];

        // Apply RY rotations followed by CNOT
        new_state = Self::apply_ry_to_state(&new_state, filter.qubits[0], param1)?;
        new_state = Self::apply_ry_to_state(&new_state, filter.qubits[1], param2)?;
        new_state = Self::apply_cnot_to_state(&new_state, filter.qubits[0], filter.qubits[1])?;

        Ok(new_state)
    }

    /// Apply RY rotation to state
    fn apply_ry_to_state(
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> Result<Array1<Complex64>> {
        let state_size = state.len();
        let mut new_state = Array1::zeros(state_size);

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..state_size {
            if i & (1 << qubit) == 0 {
                let j = i | (1 << qubit);
                if j < state_size {
                    new_state[i] = Complex64::new(cos_half, 0.0) * state[i]
                        - Complex64::new(sin_half, 0.0) * state[j];
                    new_state[j] = Complex64::new(sin_half, 0.0) * state[i]
                        + Complex64::new(cos_half, 0.0) * state[j];
                }
            }
        }

        Ok(new_state)
    }

    /// Apply CNOT to state
    fn apply_cnot_to_state(
        state: &Array1<Complex64>,
        control: usize,
        target: usize,
    ) -> Result<Array1<Complex64>> {
        let state_size = state.len();
        let mut new_state = state.clone();

        for i in 0..state_size {
            if (i & (1 << control)) != 0 {
                let j = i ^ (1 << target);
                new_state[i] = state[j];
            }
        }

        Ok(new_state)
    }
}

/// Quantum Dense Layer (fully connected)
#[derive(Debug)]
pub struct QuantumDenseLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Layer configuration
    config: QMLLayerConfig,
    /// Parameters
    parameters: Array1<f64>,
    /// Dense layer structure
    dense_structure: Vec<DenseConnection>,
}

impl QuantumDenseLayer {
    /// Create new quantum dense layer
    pub fn new(num_qubits: usize, config: QMLLayerConfig) -> Result<Self> {
        let mut layer = Self {
            num_qubits,
            config: config.clone(),
            parameters: Array1::zeros(config.num_parameters),
            dense_structure: Vec::new(),
        };

        layer.initialize_parameters();
        layer.build_dense_structure()?;

        Ok(layer)
    }

    /// Initialize parameters
    fn initialize_parameters(&mut self) {
        let mut rng = thread_rng();
        for param in &mut self.parameters {
            *param = rng.random_range(-PI..PI);
        }
    }

    /// Build dense layer structure (all-to-all connectivity)
    fn build_dense_structure(&mut self) -> Result<()> {
        let mut param_idx = 0;

        // Create all-to-all connections
        for i in 0..self.num_qubits {
            for j in (i + 1)..self.num_qubits {
                if param_idx < self.parameters.len() {
                    self.dense_structure.push(DenseConnection {
                        qubit1: i,
                        qubit2: j,
                        parameter_index: param_idx,
                    });
                    param_idx += 1;
                }
            }
        }

        Ok(())
    }
}

impl QMLLayer for QuantumDenseLayer {
    fn forward(&mut self, input: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        let mut state = input.clone();

        // Apply dense connections
        for connection in &self.dense_structure {
            state = self.apply_dense_connection(&state, connection)?;
        }

        Ok(state)
    }

    fn backward(&mut self, gradient: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(gradient.clone())
    }

    fn get_parameters(&self) -> Array1<f64> {
        self.parameters.clone()
    }

    fn set_parameters(&mut self, parameters: &Array1<f64>) {
        self.parameters = parameters.clone();
    }

    fn get_depth(&self) -> usize {
        1 // Dense layer is typically single depth
    }

    fn get_gate_count(&self) -> usize {
        self.dense_structure.len() * 2 // Approximate gates per connection
    }

    fn get_num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

impl QuantumDenseLayer {
    /// Apply dense connection (parameterized two-qubit gate)
    fn apply_dense_connection(
        &self,
        state: &Array1<Complex64>,
        connection: &DenseConnection,
    ) -> Result<Array1<Complex64>> {
        let angle = self.parameters[connection.parameter_index];

        // Apply parameterized two-qubit rotation
        Self::apply_parameterized_two_qubit_gate(state, connection.qubit1, connection.qubit2, angle)
    }

    /// Apply parameterized two-qubit gate
    fn apply_parameterized_two_qubit_gate(
        state: &Array1<Complex64>,
        qubit1: usize,
        qubit2: usize,
        angle: f64,
    ) -> Result<Array1<Complex64>> {
        let state_size = state.len();
        let mut new_state = state.clone();

        // Apply controlled rotation
        let cos_val = angle.cos();
        let sin_val = angle.sin();

        for i in 0..state_size {
            if (i & (1 << qubit1)) != 0 && (i & (1 << qubit2)) != 0 {
                // Both qubits are |1⟩
                let phase = Complex64::new(cos_val, sin_val);
                new_state[i] *= phase;
            }
        }

        Ok(new_state)
    }
}

/// Quantum LSTM Layer
#[derive(Debug)]
pub struct QuantumLSTMLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Layer configuration
    config: QMLLayerConfig,
    /// Parameters
    parameters: Array1<f64>,
    /// LSTM gates
    lstm_gates: Vec<LSTMGate>,
    /// Hidden state
    hidden_state: Option<Array1<Complex64>>,
    /// Cell state
    cell_state: Option<Array1<Complex64>>,
}

impl QuantumLSTMLayer {
    /// Create new quantum LSTM layer
    pub fn new(num_qubits: usize, config: QMLLayerConfig) -> Result<Self> {
        let mut layer = Self {
            num_qubits,
            config: config.clone(),
            parameters: Array1::zeros(config.num_parameters),
            lstm_gates: Vec::new(),
            hidden_state: None,
            cell_state: None,
        };

        layer.initialize_parameters();
        layer.build_lstm_structure()?;

        Ok(layer)
    }

    /// Initialize parameters
    fn initialize_parameters(&mut self) {
        let mut rng = thread_rng();
        for param in &mut self.parameters {
            *param = rng.random_range(-PI..PI);
        }
    }

    /// Build LSTM structure
    fn build_lstm_structure(&mut self) -> Result<()> {
        let params_per_gate = self.parameters.len() / 4; // Forget, input, output, candidate gates

        self.lstm_gates = vec![
            LSTMGate {
                gate_type: LSTMGateType::Forget,
                parameter_start: 0,
                parameter_count: params_per_gate,
            },
            LSTMGate {
                gate_type: LSTMGateType::Input,
                parameter_start: params_per_gate,
                parameter_count: params_per_gate,
            },
            LSTMGate {
                gate_type: LSTMGateType::Output,
                parameter_start: 2 * params_per_gate,
                parameter_count: params_per_gate,
            },
            LSTMGate {
                gate_type: LSTMGateType::Candidate,
                parameter_start: 3 * params_per_gate,
                parameter_count: params_per_gate,
            },
        ];

        Ok(())
    }
}

impl QMLLayer for QuantumLSTMLayer {
    fn forward(&mut self, input: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        // Initialize states if first time
        if self.hidden_state.is_none() {
            let state_size = 1 << self.num_qubits;
            let mut hidden = Array1::zeros(state_size);
            let mut cell = Array1::zeros(state_size);
            // Initialize with |0...0⟩ state
            hidden[0] = Complex64::new(1.0, 0.0);
            cell[0] = Complex64::new(1.0, 0.0);
            self.hidden_state = Some(hidden);
            self.cell_state = Some(cell);
        }

        let mut current_state = input.clone();

        // Apply LSTM gates
        for gate in &self.lstm_gates {
            current_state = self.apply_lstm_gate(&current_state, gate)?;
        }

        // Update internal states
        self.hidden_state = Some(current_state.clone());

        Ok(current_state)
    }

    fn backward(&mut self, gradient: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(gradient.clone())
    }

    fn get_parameters(&self) -> Array1<f64> {
        self.parameters.clone()
    }

    fn set_parameters(&mut self, parameters: &Array1<f64>) {
        self.parameters = parameters.clone();
    }

    fn get_depth(&self) -> usize {
        self.lstm_gates.len()
    }

    fn get_gate_count(&self) -> usize {
        self.parameters.len() // Each parameter corresponds roughly to one gate
    }

    fn get_num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

impl QuantumLSTMLayer {
    /// Apply LSTM gate
    fn apply_lstm_gate(
        &self,
        state: &Array1<Complex64>,
        gate: &LSTMGate,
    ) -> Result<Array1<Complex64>> {
        let mut new_state = state.clone();

        // Apply parameterized unitaries based on gate parameters
        for i in 0..gate.parameter_count {
            let param_idx = gate.parameter_start + i;
            if param_idx < self.parameters.len() {
                let angle = self.parameters[param_idx];
                let qubit = i % self.num_qubits;

                // Apply rotation gate
                new_state = Self::apply_rotation(&new_state, qubit, angle)?;
            }
        }

        Ok(new_state)
    }

    /// Apply rotation gate
    fn apply_rotation(
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> Result<Array1<Complex64>> {
        let state_size = state.len();
        let mut new_state = Array1::zeros(state_size);

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..state_size {
            if i & (1 << qubit) == 0 {
                let j = i | (1 << qubit);
                if j < state_size {
                    new_state[i] = Complex64::new(cos_half, 0.0) * state[i]
                        - Complex64::new(sin_half, 0.0) * state[j];
                    new_state[j] = Complex64::new(sin_half, 0.0) * state[i]
                        + Complex64::new(cos_half, 0.0) * state[j];
                }
            }
        }

        Ok(new_state)
    }

    /// Get LSTM gates reference
    #[must_use]
    pub fn get_lstm_gates(&self) -> &[LSTMGate] {
        &self.lstm_gates
    }
}

/// Quantum Attention Layer
#[derive(Debug)]
pub struct QuantumAttentionLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Layer configuration
    config: QMLLayerConfig,
    /// Parameters
    parameters: Array1<f64>,
    /// Attention structure
    attention_structure: Vec<AttentionHead>,
}

impl QuantumAttentionLayer {
    /// Create new quantum attention layer
    pub fn new(num_qubits: usize, config: QMLLayerConfig) -> Result<Self> {
        let mut layer = Self {
            num_qubits,
            config: config.clone(),
            parameters: Array1::zeros(config.num_parameters),
            attention_structure: Vec::new(),
        };

        layer.initialize_parameters();
        layer.build_attention_structure()?;

        Ok(layer)
    }

    /// Initialize parameters
    fn initialize_parameters(&mut self) {
        let mut rng = thread_rng();
        for param in &mut self.parameters {
            *param = rng.random_range(-PI..PI);
        }
    }

    /// Build attention structure
    fn build_attention_structure(&mut self) -> Result<()> {
        let num_heads = 2; // Multi-head attention
        let params_per_head = self.parameters.len() / num_heads;

        for head in 0..num_heads {
            self.attention_structure.push(AttentionHead {
                head_id: head,
                parameter_start: head * params_per_head,
                parameter_count: params_per_head,
                query_qubits: (0..self.num_qubits / 2).collect(),
                key_qubits: (self.num_qubits / 2..self.num_qubits).collect(),
            });
        }

        Ok(())
    }
}

impl QMLLayer for QuantumAttentionLayer {
    fn forward(&mut self, input: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        let mut state = input.clone();

        // Apply attention heads
        for head in &self.attention_structure {
            state = self.apply_attention_head(&state, head)?;
        }

        Ok(state)
    }

    fn backward(&mut self, gradient: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(gradient.clone())
    }

    fn get_parameters(&self) -> Array1<f64> {
        self.parameters.clone()
    }

    fn set_parameters(&mut self, parameters: &Array1<f64>) {
        self.parameters = parameters.clone();
    }

    fn get_depth(&self) -> usize {
        self.attention_structure.len()
    }

    fn get_gate_count(&self) -> usize {
        self.parameters.len()
    }

    fn get_num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

impl QuantumAttentionLayer {
    /// Apply attention head
    fn apply_attention_head(
        &self,
        state: &Array1<Complex64>,
        head: &AttentionHead,
    ) -> Result<Array1<Complex64>> {
        let mut new_state = state.clone();

        // Simplified quantum attention mechanism
        for i in 0..head.parameter_count {
            let param_idx = head.parameter_start + i;
            if param_idx < self.parameters.len() {
                let angle = self.parameters[param_idx];

                // Apply cross-attention between query and key qubits
                if i < head.query_qubits.len() && i < head.key_qubits.len() {
                    let query_qubit = head.query_qubits[i];
                    let key_qubit = head.key_qubits[i];

                    new_state =
                        Self::apply_attention_gate(&new_state, query_qubit, key_qubit, angle)?;
                }
            }
        }

        Ok(new_state)
    }

    /// Apply attention gate (parameterized two-qubit interaction)
    fn apply_attention_gate(
        state: &Array1<Complex64>,
        query_qubit: usize,
        key_qubit: usize,
        angle: f64,
    ) -> Result<Array1<Complex64>> {
        let state_size = state.len();
        let mut new_state = state.clone();

        // Apply controlled rotation based on attention score
        let cos_val = angle.cos();
        let sin_val = angle.sin();

        for i in 0..state_size {
            if (i & (1 << query_qubit)) != 0 {
                // Query qubit is |1⟩, apply attention
                let key_state = (i & (1 << key_qubit)) != 0;
                let attention_phase = if key_state {
                    Complex64::new(cos_val, sin_val)
                } else {
                    Complex64::new(cos_val, -sin_val)
                };
                new_state[i] *= attention_phase;
            }
        }

        Ok(new_state)
    }

    /// Get attention structure reference
    #[must_use]
    pub fn get_attention_structure(&self) -> &[AttentionHead] {
        &self.attention_structure
    }
}
