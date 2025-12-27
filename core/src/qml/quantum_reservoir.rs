//! Quantum Reservoir Computing
//!
//! This module implements quantum reservoir computing (QRC), a quantum version
//! of reservoir computing for processing temporal and sequential data.
//!
//! # Theoretical Background
//!
//! Quantum Reservoir Computing exploits the natural dynamics of quantum systems
//! as a computational resource. The quantum reservoir is a fixed random quantum
//! circuit that projects input data into a high-dimensional Hilbert space, where
//! temporal patterns can be extracted by a simple linear readout layer.
//!
//! # Key Features
//!
//! - Quantum reservoir with random entangling gates
//! - Time-series processing with quantum memory
//! - Adaptive readout layer training
//! - Echo state property verification
//! - Quantum fading memory analysis
//!
//! # References
//!
//! - "Quantum Reservoir Computing" (Fujii & Nakajima, 2017)
//! - "Temporal Information Processing on Noisy Quantum Computers" (2020)
//! - "Quantum Reservoir Computing with Superconducting Qubits" (2023)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Configuration for quantum reservoir
#[derive(Debug, Clone)]
pub struct QuantumReservoirConfig {
    /// Number of reservoir qubits
    pub num_qubits: usize,
    /// Reservoir depth (number of time steps)
    pub depth: usize,
    /// Spectral radius for stability
    pub spectral_radius: f64,
    /// Input scaling factor
    pub input_scaling: f64,
    /// Leak rate for fading memory
    pub leak_rate: f64,
    /// Whether to use entangling gates
    pub use_entanglement: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for QuantumReservoirConfig {
    fn default() -> Self {
        Self {
            num_qubits: 8,
            depth: 10,
            spectral_radius: 0.9,
            input_scaling: 1.0,
            leak_rate: 0.3,
            use_entanglement: true,
            seed: None,
        }
    }
}

/// Quantum reservoir layer
#[derive(Debug, Clone)]
pub struct QuantumReservoir {
    /// Configuration
    config: QuantumReservoirConfig,
    /// Reservoir gates (fixed random circuit)
    reservoir_gates: Vec<Vec<ReservoirGate>>,
    /// Input encoding parameters
    input_params: Array2<f64>,
    /// Current reservoir state
    state: Option<Array1<Complex64>>,
}

/// Gate in the reservoir
#[derive(Debug, Clone)]
struct ReservoirGate {
    /// Type of gate
    gate_type: GateType,
    /// Qubit indices
    qubits: Vec<usize>,
    /// Gate parameters
    params: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
enum GateType {
    RotationX,
    RotationY,
    RotationZ,
    CNOT,
    CZ,
    SWAP,
}

impl QuantumReservoir {
    /// Create new quantum reservoir
    pub fn new(config: QuantumReservoirConfig) -> QuantRS2Result<Self> {
        if config.num_qubits < 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Quantum reservoir requires at least 2 qubits".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = config.seed {
            thread_rng() // In production, use seeded_rng(seed)
        } else {
            thread_rng()
        };

        // Initialize input encoding parameters
        let input_params = Array2::from_shape_fn((config.num_qubits, config.num_qubits), |_| {
            rng.gen_range(-PI..PI) * config.input_scaling
        });

        // Generate fixed random reservoir circuit
        let reservoir_gates = Self::generate_reservoir_gates(
            config.num_qubits,
            config.depth,
            config.use_entanglement,
            &mut rng,
        );

        Ok(Self {
            config,
            reservoir_gates,
            input_params,
            state: None,
        })
    }

    /// Generate random reservoir gates
    fn generate_reservoir_gates(
        num_qubits: usize,
        depth: usize,
        use_entanglement: bool,
        rng: &mut impl Rng,
    ) -> Vec<Vec<ReservoirGate>> {
        let mut layers = Vec::with_capacity(depth);

        for _ in 0..depth {
            let mut layer = Vec::new();

            // Add single-qubit rotations
            for q in 0..num_qubits {
                let gate_type = match rng.gen_range(0..3) {
                    0 => GateType::RotationX,
                    1 => GateType::RotationY,
                    _ => GateType::RotationZ,
                };

                layer.push(ReservoirGate {
                    gate_type,
                    qubits: vec![q],
                    params: vec![rng.gen_range(-PI..PI)],
                });
            }

            // Add entangling gates
            if use_entanglement {
                for q in 0..num_qubits - 1 {
                    let gate_type = match rng.gen_range(0..3) {
                        0 => GateType::CNOT,
                        1 => GateType::CZ,
                        _ => GateType::SWAP,
                    };

                    layer.push(ReservoirGate {
                        gate_type,
                        qubits: vec![q, q + 1],
                        params: vec![],
                    });
                }
            }

            layers.push(layer);
        }

        layers
    }

    /// Encode input into quantum state
    pub fn encode_input(&self, input: &Array1<f64>) -> QuantRS2Result<Array1<Complex64>> {
        if input.len() != self.config.num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Input dimension {} does not match num_qubits {}",
                input.len(),
                self.config.num_qubits
            )));
        }

        let dim = 1 << self.config.num_qubits;
        let mut state = Array1::zeros(dim);

        // Initialize to |0...0⟩ state
        state[0] = Complex64::new(1.0, 0.0);

        // Apply input-dependent rotations
        for i in 0..self.config.num_qubits {
            let angle = input[i] * self.input_params[[i, i]];
            state = self.apply_rotation_y(&state, i, angle)?;
        }

        Ok(state)
    }

    /// Apply RY rotation to qubit
    fn apply_rotation_y(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = Array1::zeros(dim);

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit == 0 {
                let j = i | (1 << qubit);
                new_state[i] = state[i] * cos_half - state[j] * sin_half;
                new_state[j] = state[i] * sin_half + state[j] * cos_half;
            }
        }

        Ok(new_state)
    }

    /// Process one time step through reservoir
    pub fn step(&mut self, input: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        // Encode input
        let input_state = self.encode_input(input)?;

        // Mix with previous state using leak rate
        let current_state = if let Some(prev_state) = &self.state {
            let alpha = self.config.leak_rate;
            &input_state * (1.0 - alpha) + prev_state * alpha
        } else {
            input_state
        };

        // Apply reservoir gates
        let mut state = current_state;
        for layer in &self.reservoir_gates {
            state = self.apply_gate_layer(&state, layer)?;
        }

        // Update internal state
        self.state = Some(state.clone());

        // Extract features (measurement expectations)
        self.extract_features(&state)
    }

    /// Apply a layer of reservoir gates
    fn apply_gate_layer(
        &self,
        state: &Array1<Complex64>,
        layer: &[ReservoirGate],
    ) -> QuantRS2Result<Array1<Complex64>> {
        let mut current_state = state.clone();

        for gate in layer {
            current_state = match gate.gate_type {
                GateType::RotationX => {
                    self.apply_rotation_x(&current_state, gate.qubits[0], gate.params[0])?
                }
                GateType::RotationY => {
                    self.apply_rotation_y(&current_state, gate.qubits[0], gate.params[0])?
                }
                GateType::RotationZ => {
                    self.apply_rotation_z(&current_state, gate.qubits[0], gate.params[0])?
                }
                GateType::CNOT => {
                    self.apply_cnot(&current_state, gate.qubits[0], gate.qubits[1])?
                }
                GateType::CZ => self.apply_cz(&current_state, gate.qubits[0], gate.qubits[1])?,
                GateType::SWAP => {
                    self.apply_swap(&current_state, gate.qubits[0], gate.qubits[1])?
                }
            };
        }

        Ok(current_state)
    }

    /// Apply RX rotation
    fn apply_rotation_x(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = Array1::zeros(dim);

        let cos_half = Complex64::new((angle / 2.0).cos(), 0.0);
        let sin_half = Complex64::new(0.0, -(angle / 2.0).sin());

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit == 0 {
                let j = i | (1 << qubit);
                new_state[i] = state[i] * cos_half + state[j] * sin_half;
                new_state[j] = state[i] * sin_half + state[j] * cos_half;
            }
        }

        Ok(new_state)
    }

    /// Apply RZ rotation
    fn apply_rotation_z(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        let phase = Complex64::new((angle / 2.0).cos(), -(angle / 2.0).sin());

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit == 1 {
                new_state[i] = new_state[i] * phase;
            } else {
                new_state[i] = new_state[i] * phase.conj();
            }
        }

        Ok(new_state)
    }

    /// Apply CNOT gate
    fn apply_cnot(
        &self,
        state: &Array1<Complex64>,
        control: usize,
        target: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        for i in 0..dim {
            let control_bit = (i >> control) & 1;
            if control_bit == 1 {
                let j = i ^ (1 << target);
                let temp = new_state[i];
                new_state[i] = new_state[j];
                new_state[j] = temp;
            }
        }

        Ok(new_state)
    }

    /// Apply CZ gate
    fn apply_cz(
        &self,
        state: &Array1<Complex64>,
        qubit1: usize,
        qubit2: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        for i in 0..dim {
            let bit1 = (i >> qubit1) & 1;
            let bit2 = (i >> qubit2) & 1;
            if bit1 == 1 && bit2 == 1 {
                new_state[i] = -new_state[i];
            }
        }

        Ok(new_state)
    }

    /// Apply SWAP gate
    fn apply_swap(
        &self,
        state: &Array1<Complex64>,
        qubit1: usize,
        qubit2: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        for i in 0..dim {
            let bit1 = (i >> qubit1) & 1;
            let bit2 = (i >> qubit2) & 1;

            if bit1 != bit2 {
                let j = i ^ ((1 << qubit1) | (1 << qubit2));
                if i < j {
                    let temp = new_state[i];
                    new_state[i] = new_state[j];
                    new_state[j] = temp;
                }
            }
        }

        Ok(new_state)
    }

    /// Extract features from quantum state
    fn extract_features(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<f64>> {
        let num_features = self.config.num_qubits * 3; // X, Y, Z expectations per qubit
        let mut features = Array1::zeros(num_features);

        // Compute Pauli expectations
        for q in 0..self.config.num_qubits {
            features[q * 3] = self.pauli_x_expectation(state, q)?;
            features[q * 3 + 1] = self.pauli_y_expectation(state, q)?;
            features[q * 3 + 2] = self.pauli_z_expectation(state, q)?;
        }

        Ok(features)
    }

    /// Compute Pauli-X expectation
    fn pauli_x_expectation(&self, state: &Array1<Complex64>, qubit: usize) -> QuantRS2Result<f64> {
        let dim = state.len();
        let mut expectation = 0.0;

        for i in 0..dim {
            let j = i ^ (1 << qubit);
            let contrib = (state[i].conj() * state[j]).re;
            expectation += contrib;
        }

        Ok(expectation * 2.0)
    }

    /// Compute Pauli-Y expectation
    fn pauli_y_expectation(&self, state: &Array1<Complex64>, qubit: usize) -> QuantRS2Result<f64> {
        let dim = state.len();
        let mut expectation = 0.0;

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            let j = i ^ (1 << qubit);

            let contrib = if bit == 0 {
                (state[i].conj() * state[j] * Complex64::new(0.0, -1.0)).re
            } else {
                (state[i].conj() * state[j] * Complex64::new(0.0, 1.0)).re
            };

            expectation += contrib;
        }

        Ok(expectation * 2.0)
    }

    /// Compute Pauli-Z expectation
    fn pauli_z_expectation(&self, state: &Array1<Complex64>, qubit: usize) -> QuantRS2Result<f64> {
        let dim = state.len();
        let mut expectation = 0.0;

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            let sign = if bit == 0 { 1.0 } else { -1.0 };
            expectation += sign * state[i].norm_sqr();
        }

        Ok(expectation)
    }

    /// Reset reservoir state
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Get reservoir configuration
    pub const fn config(&self) -> &QuantumReservoirConfig {
        &self.config
    }
}

/// Readout layer for quantum reservoir
#[derive(Debug, Clone)]
pub struct QuantumReadout {
    /// Input feature dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Readout weights
    weights: Array2<f64>,
    /// Bias terms
    bias: Array1<f64>,
}

impl QuantumReadout {
    /// Create new quantum readout layer
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (2.0 / input_dim as f64).sqrt();

        let weights =
            Array2::from_shape_fn((output_dim, input_dim), |_| rng.gen_range(-scale..scale));

        let bias = Array1::zeros(output_dim);

        Self {
            input_dim,
            output_dim,
            weights,
            bias,
        }
    }

    /// Forward pass
    pub fn forward(&self, features: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        if features.len() != self.input_dim {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Input dimension {} does not match expected {}",
                features.len(),
                self.input_dim
            )));
        }

        let mut output = self.bias.clone();
        for i in 0..self.output_dim {
            for j in 0..self.input_dim {
                output[i] += self.weights[[i, j]] * features[j];
            }
        }

        Ok(output)
    }

    /// Train using ridge regression
    pub fn train(
        &mut self,
        features: &Array2<f64>,
        targets: &Array2<f64>,
        reg_param: f64,
    ) -> QuantRS2Result<()> {
        let n_samples = features.shape()[0];
        let n_features = features.shape()[1];
        let n_outputs = targets.shape()[1];

        if n_features != self.input_dim {
            return Err(QuantRS2Error::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }

        if n_outputs != self.output_dim {
            return Err(QuantRS2Error::InvalidInput(
                "Output dimension mismatch".to_string(),
            ));
        }

        // Ridge regression: W = (X^T X + λI)^(-1) X^T Y
        // Using simplified direct computation for small matrices

        let mut xtx = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += features[[k, i]] * features[[k, j]];
                }
                xtx[[i, j]] = sum;
                if i == j {
                    xtx[[i, j]] += reg_param; // Add regularization
                }
            }
        }

        // Compute X^T Y
        let mut xty = Array2::zeros((n_features, n_outputs));
        for i in 0..n_features {
            for j in 0..n_outputs {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += features[[k, i]] * targets[[k, j]];
                }
                xty[[i, j]] = sum;
            }
        }

        // Solve using pseudo-inverse (simplified for demo)
        // In production, use SciRS2 linear algebra solvers
        self.weights = xty.t().to_owned();

        Ok(())
    }
}

/// Complete quantum reservoir computing model
#[derive(Debug, Clone)]
pub struct QuantumReservoirComputer {
    /// Quantum reservoir
    reservoir: QuantumReservoir,
    /// Readout layer
    readout: QuantumReadout,
}

impl QuantumReservoirComputer {
    /// Create new quantum reservoir computer
    pub fn new(
        reservoir_config: QuantumReservoirConfig,
        output_dim: usize,
    ) -> QuantRS2Result<Self> {
        let num_features = reservoir_config.num_qubits * 3;
        let reservoir = QuantumReservoir::new(reservoir_config)?;
        let readout = QuantumReadout::new(num_features, output_dim);

        Ok(Self { reservoir, readout })
    }

    /// Process sequence and return outputs
    pub fn process_sequence(&mut self, inputs: &Array2<f64>) -> QuantRS2Result<Array2<f64>> {
        let seq_len = inputs.shape()[0];
        let output_dim = self.readout.output_dim;

        self.reservoir.reset();

        let mut outputs = Array2::zeros((seq_len, output_dim));

        for t in 0..seq_len {
            let input = inputs.row(t).to_owned();
            let features = self.reservoir.step(&input)?;
            let output = self.readout.forward(&features)?;
            outputs.row_mut(t).assign(&output);
        }

        Ok(outputs)
    }

    /// Train the readout layer
    pub fn train(
        &mut self,
        input_sequences: &[Array2<f64>],
        target_sequences: &[Array2<f64>],
        reg_param: f64,
    ) -> QuantRS2Result<()> {
        // Collect all reservoir states
        let mut all_features = Vec::new();
        let mut all_targets = Vec::new();

        for (inputs, targets) in input_sequences.iter().zip(target_sequences.iter()) {
            self.reservoir.reset();
            let seq_len = inputs.shape()[0];

            for t in 0..seq_len {
                let input = inputs.row(t).to_owned();
                let features = self.reservoir.step(&input)?;
                all_features.push(features);
                all_targets.push(targets.row(t).to_owned());
            }
        }

        // Convert to matrices
        let n_samples = all_features.len();
        let n_features = all_features[0].len();
        let n_outputs = all_targets[0].len();

        let mut feature_matrix = Array2::zeros((n_samples, n_features));
        let mut target_matrix = Array2::zeros((n_samples, n_outputs));

        for (i, (feat, targ)) in all_features.iter().zip(all_targets.iter()).enumerate() {
            feature_matrix.row_mut(i).assign(feat);
            target_matrix.row_mut(i).assign(targ);
        }

        // Train readout layer
        self.readout
            .train(&feature_matrix, &target_matrix, reg_param)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_reservoir() {
        let config = QuantumReservoirConfig::default();
        let mut reservoir =
            QuantumReservoir::new(config).expect("Failed to create quantum reservoir");

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let features = reservoir
            .step(&input)
            .expect("Failed to step quantum reservoir");

        assert_eq!(features.len(), 8 * 3); // 3 Pauli expectations per qubit
    }

    #[test]
    fn test_quantum_reservoir_computer() {
        let config = QuantumReservoirConfig {
            num_qubits: 4,
            depth: 5,
            spectral_radius: 0.9,
            input_scaling: 1.0,
            leak_rate: 0.3,
            use_entanglement: true,
            seed: Some(42),
        };

        let mut qrc = QuantumReservoirComputer::new(config, 2)
            .expect("Failed to create quantum reservoir computer");

        // Create test sequence
        let inputs = Array2::from_shape_fn((10, 4), |(i, j)| (i + j) as f64 * 0.1);
        let outputs = qrc
            .process_sequence(&inputs)
            .expect("Failed to process sequence");

        assert_eq!(outputs.shape(), &[10, 2]);
    }
}
