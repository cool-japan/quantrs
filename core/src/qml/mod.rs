//! Quantum Machine Learning (QML) primitives and layers
//!
//! This module provides building blocks for quantum machine learning,
//! including parameterized quantum circuits, data encoding strategies,
//! and common QML layer patterns.

pub mod encoding;
pub mod generative_adversarial;
pub mod layers;
pub mod nlp;
pub mod reinforcement_learning;
pub mod training;

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

// Re-export Parameter from layers module
pub use layers::Parameter;

/// Trait for quantum machine learning layers
pub trait QMLLayer: Send + Sync {
    /// Get the number of qubits this layer acts on
    fn num_qubits(&self) -> usize;

    /// Get the parameters of this layer
    fn parameters(&self) -> &[Parameter];

    /// Get mutable access to parameters
    fn parameters_mut(&mut self) -> &mut [Parameter];

    /// Set parameter values
    fn set_parameters(&mut self, values: &[f64]) -> QuantRS2Result<()> {
        if values.len() != self.parameters().len() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.parameters().len(),
                values.len()
            )));
        }

        for (param, &value) in self.parameters_mut().iter_mut().zip(values.iter()) {
            param.value = value;
        }

        Ok(())
    }

    /// Get the gates that make up this layer
    fn gates(&self) -> Vec<Box<dyn GateOp>>;

    /// Compute gradients with respect to parameters
    fn compute_gradients(
        &self,
        state: &Array1<Complex64>,
        loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>>;

    /// Get layer name
    fn name(&self) -> &str;
}

/// Data encoding strategies for QML
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EncodingStrategy {
    /// Amplitude encoding: data encoded in state amplitudes
    Amplitude,
    /// Angle encoding: data encoded as rotation angles
    Angle,
    /// IQP encoding: data encoded in diagonal gates
    IQP,
    /// Basis encoding: data encoded in computational basis
    Basis,
}

/// Configuration for QML circuits
#[derive(Debug, Clone)]
pub struct QMLConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Data encoding strategy
    pub encoding: EncodingStrategy,
    /// Entanglement pattern
    pub entanglement: EntanglementPattern,
    /// Whether to reupload data in each layer
    pub data_reuploading: bool,
}

impl Default for QMLConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            num_layers: 2,
            encoding: EncodingStrategy::Angle,
            entanglement: EntanglementPattern::Full,
            data_reuploading: false,
        }
    }
}

/// Entanglement patterns for QML layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntanglementPattern {
    /// No entanglement
    None,
    /// Linear nearest-neighbor entanglement
    Linear,
    /// Circular nearest-neighbor entanglement
    Circular,
    /// All-to-all entanglement
    Full,
    /// Alternating pairs
    Alternating,
}

/// A parameterized quantum circuit for QML
pub struct QMLCircuit {
    /// Configuration
    config: QMLConfig,
    /// The layers in the circuit
    layers: Vec<Box<dyn QMLLayer>>,
    /// Parameter count
    num_parameters: usize,
}

impl QMLCircuit {
    /// Create a new QML circuit
    pub fn new(config: QMLConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
            num_parameters: 0,
        }
    }

    /// Add a layer to the circuit
    pub fn add_layer(&mut self, layer: Box<dyn QMLLayer>) -> QuantRS2Result<()> {
        if layer.num_qubits() != self.config.num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Layer has {} qubits, circuit has {}",
                layer.num_qubits(),
                self.config.num_qubits
            )));
        }

        self.num_parameters += layer.parameters().len();
        self.layers.push(layer);
        Ok(())
    }

    /// Get all parameters in the circuit
    pub fn parameters(&self) -> Vec<&Parameter> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters().iter())
            .collect()
    }

    /// Set all parameters in the circuit
    pub fn set_parameters(&mut self, values: &[f64]) -> QuantRS2Result<()> {
        if values.len() != self.num_parameters {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.num_parameters,
                values.len()
            )));
        }

        let mut offset = 0;
        for layer in &mut self.layers {
            let layer_params = layer.parameters().len();
            layer.set_parameters(&values[offset..offset + layer_params])?;
            offset += layer_params;
        }

        Ok(())
    }

    /// Get all gates in the circuit
    pub fn gates(&self) -> Vec<Box<dyn GateOp>> {
        self.layers.iter().flat_map(|layer| layer.gates()).collect()
    }

    /// Compute gradients for all parameters
    pub fn compute_gradients(
        &self,
        state: &Array1<Complex64>,
        loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        let mut all_gradients = Vec::new();

        for layer in &self.layers {
            let layer_grads = layer.compute_gradients(state, loss_gradient)?;
            all_gradients.extend(layer_grads);
        }

        Ok(all_gradients)
    }
}

/// Helper function to create entangling gates based on pattern
pub fn create_entangling_gates(
    num_qubits: usize,
    pattern: EntanglementPattern,
) -> Vec<(QubitId, QubitId)> {
    match pattern {
        EntanglementPattern::None => vec![],

        EntanglementPattern::Linear => (0..num_qubits - 1)
            .map(|i| (QubitId(i as u32), QubitId((i + 1) as u32)))
            .collect(),

        EntanglementPattern::Circular => {
            let mut gates = vec![];
            for i in 0..num_qubits {
                gates.push((QubitId(i as u32), QubitId(((i + 1) % num_qubits) as u32)));
            }
            gates
        }

        EntanglementPattern::Full => {
            let mut gates = vec![];
            for i in 0..num_qubits {
                for j in i + 1..num_qubits {
                    gates.push((QubitId(i as u32), QubitId(j as u32)));
                }
            }
            gates
        }

        EntanglementPattern::Alternating => {
            let mut gates = vec![];
            // Even pairs
            for i in (0..num_qubits - 1).step_by(2) {
                gates.push((QubitId(i as u32), QubitId((i + 1) as u32)));
            }
            // Odd pairs
            for i in (1..num_qubits - 1).step_by(2) {
                gates.push((QubitId(i as u32), QubitId((i + 1) as u32)));
            }
            gates
        }
    }
}

/// Compute the quantum Fisher information matrix
pub fn quantum_fisher_information(
    circuit: &QMLCircuit,
    _state: &Array1<Complex64>,
) -> QuantRS2Result<Array2<f64>> {
    let num_params = circuit.num_parameters;
    let fisher = Array2::zeros((num_params, num_params));

    // Compute Fisher information using parameter shift rule
    // F_ij = 4 * Re(<∂_i ψ | ∂_j ψ> - <∂_i ψ | ψ><ψ | ∂_j ψ>)

    // This is a placeholder - full implementation would compute
    // derivatives of the state with respect to all parameters

    Ok(fisher)
}

/// Natural gradient for quantum optimization
pub fn natural_gradient(
    gradients: &[f64],
    fisher: &Array2<f64>,
    regularization: f64,
) -> QuantRS2Result<Vec<f64>> {
    // Add regularization to diagonal
    let mut regularized_fisher = fisher.clone();
    for i in 0..fisher.nrows() {
        regularized_fisher[(i, i)] += regularization;
    }

    // Solve F * natural_grad = grad using LU decomposition
    // This is a placeholder - would use SciRS2's linear solver when available
    let _grad_array = Array1::from_vec(gradients.to_vec());

    // For now, return the regular gradient
    // In practice, would solve: regularized_fisher * natural_grad = grad
    Ok(gradients.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entanglement_patterns() {
        let linear = create_entangling_gates(4, EntanglementPattern::Linear);
        assert_eq!(linear.len(), 3);
        assert_eq!(linear[0], (QubitId(0), QubitId(1)));

        let circular = create_entangling_gates(4, EntanglementPattern::Circular);
        assert_eq!(circular.len(), 4);
        assert_eq!(circular[3], (QubitId(3), QubitId(0)));

        let full = create_entangling_gates(3, EntanglementPattern::Full);
        assert_eq!(full.len(), 3); // 3 choose 2

        let none = create_entangling_gates(4, EntanglementPattern::None);
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn test_qml_circuit() {
        let config = QMLConfig {
            num_qubits: 2,
            num_layers: 1,
            ..Default::default()
        };

        let circuit = QMLCircuit::new(config);
        assert_eq!(circuit.num_parameters, 0);
    }
}
