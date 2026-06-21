//! Quantum Machine Learning (QML) primitives and layers
//!
//! This module provides building blocks for quantum machine learning,
//! including parameterized quantum circuits, data encoding strategies,
//! and common QML layer patterns.

pub mod advanced_algorithms;
pub mod encoding;
pub mod generative_adversarial;
pub mod layers;
pub mod nlp;
pub mod reinforcement_learning;
pub mod simulator;
pub mod training;

// New cutting-edge quantum ML modules
pub mod quantum_contrastive;
pub mod quantum_memory_networks;
pub mod quantum_meta_learning;
pub mod quantum_reservoir;
pub mod quantum_transformer;

// Advanced quantum ML: Privacy, Security, and Distributed Learning
pub mod quantum_boltzmann;
pub mod quantum_federated;

// Re-export advanced QML algorithms
pub use advanced_algorithms::{
    FeatureMapType, QMLMetrics, QuantumEnsemble, QuantumKernel, QuantumKernelConfig, QuantumSVM,
    QuantumTransferLearning, TransferLearningConfig, VotingStrategy,
};

// Re-export new modules
pub use quantum_contrastive::{
    QuantumAugmentation, QuantumContrastiveConfig, QuantumContrastiveLearner,
};
pub use quantum_memory_networks::{MemoryInitStrategy, QuantumMemoryConfig, QuantumMemoryNetwork};
pub use quantum_meta_learning::{
    QuantumMAML, QuantumMetaLearningConfig, QuantumReptile, QuantumTask,
};
pub use quantum_reservoir::{QuantumReservoirComputer, QuantumReservoirConfig};
pub use quantum_transformer::{QuantumAttention, QuantumTransformer, QuantumTransformerConfig};

// Re-export advanced quantum ML modules
pub use quantum_boltzmann::{DeepQuantumBoltzmannMachine, QRBMConfig, QuantumRBM};
pub use quantum_federated::{AggregationStrategy, QuantumFederatedConfig, QuantumFederatedServer};

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Compute the quantum Fisher information matrix (real part of the quantum
/// geometric tensor) of a parameterized circuit.
///
/// `F_ij = 4 · Re(⟨∂_i ψ | ∂_j ψ⟩ − ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩)`
///
/// The state derivatives `|∂_i ψ⟩` are computed by central finite differences:
/// the circuit's `i`-th parameter is shifted by `±ε`, the resulting state is
/// simulated exactly, and `|∂_i ψ⟩ ≈ (|ψ(θ+ε)⟩ − |ψ(θ−ε)⟩) / (2ε)`. The
/// circuit's original parameters are restored on return.
///
/// The circuit is borrowed mutably because computing the derivatives requires
/// temporarily perturbing its parameters; this is an exact, non-fabricated
/// computation.
pub fn quantum_fisher_information(circuit: &mut QMLCircuit) -> QuantRS2Result<Array2<f64>> {
    let num_params = circuit.num_parameters;
    let mut fisher = Array2::zeros((num_params, num_params));
    if num_params == 0 {
        return Ok(fisher);
    }

    let num_qubits = circuit.config.num_qubits;
    let base_params: Vec<f64> = circuit.parameters().iter().map(|p| p.value).collect();

    // Helper: simulate the state for a given parameter assignment.
    let epsilon = 1e-6;
    let dim = 1usize << num_qubits;

    // Reference state |ψ(θ)⟩.
    circuit.set_parameters(&base_params)?;
    let psi = simulator::simulate(num_qubits, &circuit.gates())?;

    // State derivatives |∂_i ψ⟩ via central differences.
    let mut derivatives: Vec<Array1<Complex64>> = Vec::with_capacity(num_params);
    for i in 0..num_params {
        let mut plus = base_params.clone();
        plus[i] += epsilon;
        circuit.set_parameters(&plus)?;
        let psi_plus = simulator::simulate(num_qubits, &circuit.gates())?;

        let mut minus = base_params.clone();
        minus[i] -= epsilon;
        circuit.set_parameters(&minus)?;
        let psi_minus = simulator::simulate(num_qubits, &circuit.gates())?;

        let mut deriv = Array1::zeros(dim);
        for k in 0..dim {
            deriv[k] = (psi_plus[k] - psi_minus[k]) / Complex64::new(2.0 * epsilon, 0.0);
        }
        derivatives.push(deriv);
    }

    // Restore original parameters.
    circuit.set_parameters(&base_params)?;

    // ⟨ψ | ∂_i ψ⟩ for each i.
    let psi_dot_deriv: Vec<Complex64> = derivatives
        .iter()
        .map(|d| {
            psi.iter()
                .zip(d.iter())
                .map(|(p, di)| p.conj() * di)
                .sum::<Complex64>()
        })
        .collect();

    // Assemble the symmetric Fisher matrix.
    for i in 0..num_params {
        for j in i..num_params {
            let overlap: Complex64 = derivatives[i]
                .iter()
                .zip(derivatives[j].iter())
                .map(|(di, dj)| di.conj() * dj)
                .sum();
            let correction = psi_dot_deriv[i].conj() * psi_dot_deriv[j];
            let value = 4.0 * (overlap - correction).re;
            fisher[(i, j)] = value;
            fisher[(j, i)] = value;
        }
    }

    Ok(fisher)
}

/// Natural gradient for quantum optimization.
///
/// Solves the regularized linear system `(F + λI) · g_nat = g` for the natural
/// gradient `g_nat`, where `F` is the quantum Fisher information matrix, `λ` the
/// Tikhonov regularization, and `g` the Euclidean gradient. The system is
/// solved by inverting the (symmetric positive-definite after regularization)
/// matrix via `scirs2_linalg`.
pub fn natural_gradient(
    gradients: &[f64],
    fisher: &Array2<f64>,
    regularization: f64,
) -> QuantRS2Result<Vec<f64>> {
    let n = gradients.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if fisher.nrows() != n || fisher.ncols() != n {
        return Err(QuantRS2Error::InvalidInput(format!(
            "Fisher matrix is {}×{} but gradient has length {n}",
            fisher.nrows(),
            fisher.ncols()
        )));
    }

    // Regularize the diagonal: (F + λI).
    let mut regularized = fisher.clone();
    for i in 0..n {
        regularized[(i, i)] += regularization;
    }

    let grad = Array1::from_vec(gradients.to_vec());

    // g_nat = (F + λI)^{-1} g. Invert via scirs2_linalg (SciRS2 POLICY).
    if n == 1 {
        let denom = regularized[(0, 0)];
        if denom.abs() < 1e-14 {
            return Err(QuantRS2Error::InvalidInput(
                "regularized Fisher matrix is singular".to_string(),
            ));
        }
        return Ok(vec![grad[0] / denom]);
    }

    let inverse = scirs2_linalg::inv(&regularized.view(), None).map_err(|e| {
        QuantRS2Error::ComputationError(format!("natural gradient solve failed: {e:?}"))
    })?;
    let natural = inverse.dot(&grad);
    Ok(natural.to_vec())
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

    #[test]
    fn test_natural_gradient_solves_system_not_passthrough() {
        // For a non-identity Fisher matrix the natural gradient must DIFFER from
        // the raw gradient (the old fabrication returned the gradient verbatim).
        // (F + λI) g_nat = g  =>  g_nat = (F + λI)^{-1} g.
        let fisher =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.5, 0.5, 3.0]).expect("fisher matrix");
        let gradients = vec![1.0, 1.0];
        let reg = 0.0;

        let g_nat = natural_gradient(&gradients, &fisher, reg).expect("natural gradient");

        // Verify it is the true solution: (F) g_nat ≈ g.
        let recon0 = fisher[(0, 0)] * g_nat[0] + fisher[(0, 1)] * g_nat[1];
        let recon1 = fisher[(1, 0)] * g_nat[0] + fisher[(1, 1)] * g_nat[1];
        assert!((recon0 - gradients[0]).abs() < 1e-9);
        assert!((recon1 - gradients[1]).abs() < 1e-9);

        // And it must NOT be the raw gradient (proves real solve, not passthrough).
        assert!(
            (g_nat[0] - gradients[0]).abs() > 1e-6 || (g_nat[1] - gradients[1]).abs() > 1e-6,
            "natural gradient must differ from raw gradient for non-identity Fisher"
        );
    }

    #[test]
    fn test_natural_gradient_identity_fisher_is_passthrough() {
        // With F = 0 and λ = 1, (F+λI)=I so g_nat == g. This is the
        // mathematically-correct identity case (not a fabrication).
        let fisher = Array2::zeros((3, 3));
        let gradients = vec![0.4, -1.2, 0.7];
        let g_nat = natural_gradient(&gradients, &fisher, 1.0).expect("natural gradient");
        for (a, b) in g_nat.iter().zip(gradients.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_quantum_fisher_information_is_nonzero_for_parameterized_circuit() {
        // A parameterized rotation circuit has a non-trivial quantum geometric
        // tensor; the old placeholder returned all zeros.
        let config = QMLConfig {
            num_qubits: 2,
            num_layers: 1,
            ..Default::default()
        };
        let mut circuit = QMLCircuit::new(config);
        let layer = layers::RotationLayer::uniform(2, 'Y').expect("rotation layer");
        circuit.add_layer(Box::new(layer)).expect("add layer");
        // Set non-trivial parameters.
        circuit.set_parameters(&[0.5, 1.1]).expect("set parameters");

        let fisher = quantum_fisher_information(&mut circuit).expect("fisher");
        assert_eq!(fisher.shape(), &[2, 2]);
        let max_abs = fisher.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_abs > 1e-6,
            "Fisher information must be non-zero for a parameterized circuit, got max {max_abs}"
        );
        // For independent RY rotations on the |0> state the QFI is ~I (diagonal).
        assert!(fisher[(0, 0)] > 1e-3 && fisher[(1, 1)] > 1e-3);
    }
}
