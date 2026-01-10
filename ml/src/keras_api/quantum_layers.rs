//! Quantum layers for Keras-like API

use super::KerasLayer;
use crate::error::{MLError, Result};
use crate::simulator_backends::{DynamicCircuit, Observable, SimulatorBackend, StatevectorBackend};
use quantrs2_circuit::prelude::*;
use scirs2_core::ndarray::{s, ArrayD, IxDyn};
use std::sync::Arc;

/// Quantum Dense layer
pub struct QuantumDense {
    /// Number of qubits
    num_qubits: usize,
    /// Number of output features
    units: usize,
    /// Quantum circuit ansatz
    ansatz_type: QuantumAnsatzType,
    /// Number of layers in ansatz
    num_layers: usize,
    /// Observable for measurement
    observable: Observable,
    /// Backend
    backend: Arc<dyn SimulatorBackend>,
    /// Layer name
    name: String,
    /// Built flag
    built: bool,
    /// Input shape
    input_shape: Option<Vec<usize>>,
    /// Quantum parameters
    quantum_weights: Vec<ArrayD<f64>>,
}

/// Quantum ansatz types
#[derive(Debug, Clone)]
pub enum QuantumAnsatzType {
    /// Hardware efficient ansatz
    HardwareEfficient,
    /// Real amplitudes ansatz
    RealAmplitudes,
    /// Strongly entangling layers
    StronglyEntangling,
    /// Custom ansatz
    Custom(DynamicCircuit),
}

impl QuantumDense {
    /// Create new quantum dense layer
    pub fn new(num_qubits: usize, units: usize) -> Self {
        Self {
            num_qubits,
            units,
            ansatz_type: QuantumAnsatzType::HardwareEfficient,
            num_layers: 1,
            observable: Observable::PauliZ(vec![0]),
            backend: Arc::new(StatevectorBackend::new(10)),
            name: format!("quantum_dense_{}", fastrand::u32(..)),
            built: false,
            input_shape: None,
            quantum_weights: Vec::new(),
        }
    }

    /// Set ansatz type
    pub fn ansatz_type(mut self, ansatz_type: QuantumAnsatzType) -> Self {
        self.ansatz_type = ansatz_type;
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set observable
    pub fn observable(mut self, observable: Observable) -> Self {
        self.observable = observable;
        self
    }

    /// Set backend
    pub fn backend(mut self, backend: Arc<dyn SimulatorBackend>) -> Self {
        self.backend = backend;
        self
    }

    /// Set layer name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Build quantum circuit based on ansatz type
    fn build_quantum_circuit(&self) -> Result<DynamicCircuit> {
        let mut builder: Circuit<8> = Circuit::new();

        match &self.ansatz_type {
            QuantumAnsatzType::HardwareEfficient => {
                for layer in 0..self.num_layers {
                    if layer == 0 {
                        for qubit in 0..self.num_qubits {
                            builder.ry(qubit, 0.0)?;
                        }
                    }

                    for qubit in 0..self.num_qubits {
                        builder.ry(qubit, 0.0)?;
                        builder.rz(qubit, 0.0)?;
                    }

                    for qubit in 0..self.num_qubits - 1 {
                        builder.cnot(qubit, qubit + 1)?;
                    }
                    if self.num_qubits > 2 {
                        builder.cnot(self.num_qubits - 1, 0)?;
                    }
                }
            }
            QuantumAnsatzType::RealAmplitudes => {
                for layer in 0..self.num_layers {
                    if layer == 0 {
                        for qubit in 0..self.num_qubits {
                            builder.ry(qubit, 0.0)?;
                        }
                    }

                    for qubit in 0..self.num_qubits {
                        builder.ry(qubit, 0.0)?;
                    }

                    for qubit in 0..self.num_qubits - 1 {
                        builder.cnot(qubit, qubit + 1)?;
                    }
                }
            }
            QuantumAnsatzType::StronglyEntangling => {
                for layer in 0..self.num_layers {
                    if layer == 0 {
                        for qubit in 0..self.num_qubits {
                            builder.ry(qubit, 0.0)?;
                        }
                    }

                    for qubit in 0..self.num_qubits {
                        builder.rx(qubit, 0.0)?;
                        builder.ry(qubit, 0.0)?;
                        builder.rz(qubit, 0.0)?;
                    }

                    for qubit in 0..self.num_qubits - 1 {
                        builder.cnot(qubit, qubit + 1)?;
                    }
                    if self.num_qubits > 2 {
                        builder.cnot(self.num_qubits - 1, 0)?;
                    }
                }
            }
            QuantumAnsatzType::Custom(circuit) => {
                return Ok(circuit.clone());
            }
        }

        let circuit = builder.build();
        DynamicCircuit::from_circuit(circuit)
    }
}

impl KerasLayer for QuantumDense {
    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        self.input_shape = Some(input_shape.to_vec());

        let num_params = match &self.ansatz_type {
            QuantumAnsatzType::HardwareEfficient => self.num_qubits * 2 * self.num_layers,
            QuantumAnsatzType::RealAmplitudes => self.num_qubits * self.num_layers,
            QuantumAnsatzType::StronglyEntangling => self.num_qubits * 3 * self.num_layers,
            QuantumAnsatzType::Custom(_) => 10,
        };

        let params = ArrayD::from_shape_fn(IxDyn(&[self.units, num_params]), |_| {
            fastrand::f64() * 2.0 * std::f64::consts::PI
        });
        self.quantum_weights.push(params);

        self.built = true;
        Ok(())
    }

    fn call(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::InvalidConfiguration(
                "Layer must be built before calling".to_string(),
            ));
        }

        let batch_size = inputs.shape()[0];
        let mut outputs = ArrayD::zeros(IxDyn(&[batch_size, self.units]));

        for batch_idx in 0..batch_size {
            for unit_idx in 0..self.units {
                let circuit = self.build_quantum_circuit()?;

                let input_slice = inputs.slice(s![batch_idx, ..]);
                let param_slice = self.quantum_weights[0].slice(s![unit_idx, ..]);

                let combined_params: Vec<f64> = input_slice
                    .iter()
                    .chain(param_slice.iter())
                    .copied()
                    .collect();

                let expectation =
                    self.backend
                        .expectation_value(&circuit, &combined_params, &self.observable)?;

                outputs[[batch_idx, unit_idx]] = expectation;
            }
        }

        Ok(outputs)
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let mut output_shape = input_shape.to_vec();
        let last_idx = output_shape.len() - 1;
        output_shape[last_idx] = self.units;
        output_shape
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        self.quantum_weights.clone()
    }

    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()> {
        if weights.len() != self.quantum_weights.len() {
            return Err(MLError::InvalidConfiguration(
                "Number of weight arrays doesn't match layer structure".to_string(),
            ));
        }
        self.quantum_weights = weights;
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }
}
