//! # QuantumReservoirComputer - connections Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::Result;
use crate::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::{HashMap, VecDeque};

use super::types::{
    InputEncoding, OutputMeasurement, QuantumReservoirArchitecture, QuantumReservoirConfig,
    QuantumReservoirState, ReservoirMetrics, ReservoirTrainingData, TrainingResult,
};

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Create new quantum reservoir computer
    pub fn new(config: QuantumReservoirConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;
        let simulator = StateVectorSimulator::new();
        let reservoir_state = QuantumReservoirState::new(config.num_qubits, config.memory_capacity);
        let reservoir_circuit = Self::generate_reservoir_circuit(&config)?;
        let input_coupling_circuit = Self::generate_input_coupling_circuit(&config)?;
        let output_size = match config.output_measurement {
            OutputMeasurement::PauliExpectation => config.num_qubits * 3,
            OutputMeasurement::Probability => 1 << config.num_qubits,
            OutputMeasurement::Correlations => config.num_qubits * config.num_qubits,
            OutputMeasurement::Entanglement => config.num_qubits,
            OutputMeasurement::Fidelity => 1,
            OutputMeasurement::QuantumFisherInformation => config.num_qubits,
            OutputMeasurement::Variance => config.num_qubits,
            OutputMeasurement::HigherOrderMoments => config.num_qubits * 4,
            OutputMeasurement::SpectralProperties => config.num_qubits,
            OutputMeasurement::QuantumCoherence => config.num_qubits,
            _ => config.num_qubits,
        };
        let feature_size = Self::calculate_feature_size(&config);
        let mut output_weights = Array2::zeros((output_size, feature_size));
        let scale = (2.0 / (output_size + feature_size) as f64).sqrt();
        for elem in &mut output_weights {
            *elem = (thread_rng().random::<f64>() - 0.5) * 2.0 * scale;
        }
        Ok(Self {
            config,
            reservoir_state,
            reservoir_circuit,
            input_coupling_circuit,
            output_weights,
            simulator,
            circuit_interface,
            metrics: ReservoirMetrics::default(),
            training_history: VecDeque::with_capacity(10_000),
        })
    }
    /// Generate reservoir circuit based on architecture
    pub(super) fn generate_reservoir_circuit(
        config: &QuantumReservoirConfig,
    ) -> Result<InterfaceCircuit> {
        let mut circuit = InterfaceCircuit::new(config.num_qubits, 0);
        match config.architecture {
            QuantumReservoirArchitecture::RandomCircuit => {
                Self::generate_random_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::SpinChain => {
                Self::generate_spin_chain_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::TransverseFieldIsing => {
                Self::generate_tfim_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::SmallWorld => {
                Self::generate_small_world_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::FullyConnected => {
                Self::generate_fully_connected_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::Custom => {
                Self::generate_random_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::ScaleFree => {
                Self::generate_small_world_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::HierarchicalModular => {
                Self::generate_random_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::AdaptiveTopology => {
                Self::generate_random_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::QuantumCellularAutomaton => {
                Self::generate_spin_chain_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::Ring => {
                Self::generate_spin_chain_circuit(&mut circuit, config)?;
            }
            _ => {
                Self::generate_random_circuit(&mut circuit, config)?;
            }
        }
        Ok(circuit)
    }
    /// Generate random quantum circuit
    pub(super) fn generate_random_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let depth = config.evolution_steps;
        for _ in 0..depth {
            for qubit in 0..config.num_qubits {
                let angle = thread_rng().random::<f64>() * 2.0 * std::f64::consts::PI;
                let gate_type = match thread_rng().random_range(0..3) {
                    0 => InterfaceGateType::RX(angle),
                    1 => InterfaceGateType::RY(angle),
                    _ => InterfaceGateType::RZ(angle),
                };
                circuit.add_gate(InterfaceGate::new(gate_type, vec![qubit]));
            }
            for _ in 0..(config.num_qubits / 2) {
                let qubit1 = thread_rng().random_range(0..config.num_qubits);
                let qubit2 = thread_rng().random_range(0..config.num_qubits);
                if qubit1 != qubit2 {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::CNOT,
                        vec![qubit1, qubit2],
                    ));
                }
            }
        }
        Ok(())
    }
    /// Generate spin chain circuit
    pub(super) fn generate_spin_chain_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;
        for _ in 0..config.evolution_steps {
            for i in 0..config.num_qubits - 1 {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step),
                    vec![i],
                ));
                circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, i + 1]));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step),
                    vec![i + 1],
                ));
                circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, i + 1]));
            }
        }
        Ok(())
    }
    /// Generate transverse field Ising model circuit
    pub(super) fn generate_tfim_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;
        let field = coupling * 0.5;
        for _ in 0..config.evolution_steps {
            for qubit in 0..config.num_qubits {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RX(field * config.time_step),
                    vec![qubit],
                ));
            }
            for i in 0..config.num_qubits - 1 {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                    vec![i],
                ));
                circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, i + 1]));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step),
                    vec![i + 1],
                ));
                circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, i + 1]));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                    vec![i],
                ));
            }
        }
        Ok(())
    }
    /// Generate small-world network circuit
    pub(super) fn generate_small_world_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;
        let rewiring_prob = 0.1;
        for _ in 0..config.evolution_steps {
            for i in 0..config.num_qubits {
                let next = (i + 1) % config.num_qubits;
                let target = if thread_rng().random::<f64>() < rewiring_prob {
                    thread_rng().random_range(0..config.num_qubits)
                } else {
                    next
                };
                if target != i {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                        vec![i],
                    ));
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, target]));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step),
                        vec![target],
                    ));
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, target]));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                        vec![i],
                    ));
                }
            }
        }
        Ok(())
    }
    /// Generate fully connected circuit
    pub(super) fn generate_fully_connected_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength / config.num_qubits as f64;
        for _ in 0..config.evolution_steps {
            for i in 0..config.num_qubits {
                for j in i + 1..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                        vec![i],
                    ));
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, j]));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step),
                        vec![j],
                    ));
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, j]));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                        vec![i],
                    ));
                }
            }
        }
        Ok(())
    }
    /// Generate input coupling circuit
    pub(super) fn generate_input_coupling_circuit(
        config: &QuantumReservoirConfig,
    ) -> Result<InterfaceCircuit> {
        let mut circuit = InterfaceCircuit::new(config.num_qubits, 0);
        match config.input_encoding {
            InputEncoding::Amplitude => {
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.0), vec![qubit]));
                }
            }
            InputEncoding::Phase => {
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::RZ(0.0), vec![qubit]));
                }
            }
            InputEncoding::BasisState => {
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::X, vec![qubit]));
                }
            }
            _ => {
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.0), vec![qubit]));
                }
            }
        }
        Ok(circuit)
    }
    /// Calculate feature size based on configuration
    pub(super) fn calculate_feature_size(config: &QuantumReservoirConfig) -> usize {
        match config.output_measurement {
            OutputMeasurement::PauliExpectation => config.num_qubits * 3,
            OutputMeasurement::Probability => 1 << config.num_qubits.min(10),
            OutputMeasurement::Correlations => config.num_qubits * config.num_qubits,
            OutputMeasurement::Entanglement => config.num_qubits,
            OutputMeasurement::Fidelity => 1,
            OutputMeasurement::QuantumFisherInformation => config.num_qubits,
            OutputMeasurement::Variance => config.num_qubits,
            OutputMeasurement::HigherOrderMoments => config.num_qubits * 4,
            OutputMeasurement::SpectralProperties => config.num_qubits,
            OutputMeasurement::QuantumCoherence => config.num_qubits,
            _ => config.num_qubits,
        }
    }
    /// Apply single qubit rotation
    pub(super) fn apply_single_qubit_rotation(
        &mut self,
        qubit: usize,
        gate_type: InterfaceGateType,
    ) -> Result<()> {
        let mut temp_circuit = InterfaceCircuit::new(self.config.num_qubits, 0);
        temp_circuit.add_gate(InterfaceGate::new(gate_type, vec![qubit]));
        self.simulator.apply_interface_circuit(&temp_circuit)?;
        Ok(())
    }
    /// Apply single qubit gate
    pub(super) fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        gate_type: InterfaceGateType,
    ) -> Result<()> {
        let mut temp_circuit = InterfaceCircuit::new(self.config.num_qubits, 0);
        temp_circuit.add_gate(InterfaceGate::new(gate_type, vec![qubit]));
        self.simulator.apply_interface_circuit(&temp_circuit)?;
        Ok(())
    }
    /// Apply decoherence to the reservoir state
    pub(super) fn apply_decoherence(&mut self) -> Result<()> {
        let decoherence_rate = self.config.noise_level;
        for amplitude in &mut self.reservoir_state.state_vector {
            let phase_noise = (thread_rng().random::<f64>() - 0.5)
                * decoherence_rate
                * 2.0
                * std::f64::consts::PI;
            *amplitude *= Complex64::new(0.0, phase_noise).exp();
            let damping = (1.0 - decoherence_rate).sqrt();
            *amplitude *= damping;
        }
        let norm: f64 = self
            .reservoir_state
            .state_vector
            .iter()
            .map(scirs2_core::Complex::norm_sqr)
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            self.reservoir_state.state_vector.mapv_inplace(|x| x / norm);
        }
        Ok(())
    }
    /// Measure Pauli expectation values
    pub(super) fn measure_pauli_expectations(&self) -> Result<Array1<f64>> {
        let mut expectations = Vec::new();
        for qubit in 0..self.config.num_qubits {
            let x_exp = self.calculate_single_qubit_expectation(
                qubit,
                &[
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )?;
            expectations.push(x_exp);
            let y_exp = self.calculate_single_qubit_expectation(
                qubit,
                &[
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )?;
            expectations.push(y_exp);
            let z_exp = self.calculate_single_qubit_expectation(
                qubit,
                &[
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )?;
            expectations.push(z_exp);
        }
        Ok(Array1::from_vec(expectations))
    }
    /// Measure two-qubit correlations
    pub(super) fn measure_correlations(&mut self) -> Result<Array1<f64>> {
        let mut correlations = Vec::new();
        for i in 0..self.config.num_qubits {
            for j in 0..self.config.num_qubits {
                if i == j {
                    correlations.push(1.0);
                    self.reservoir_state.correlations[[i, j]] = 1.0;
                } else {
                    let corr = self.calculate_two_qubit_correlation(i, j)?;
                    correlations.push(corr);
                    self.reservoir_state.correlations[[i, j]] = corr;
                }
            }
        }
        Ok(Array1::from_vec(correlations))
    }
    /// Measure entanglement metrics
    pub(super) fn measure_entanglement(&self) -> Result<Array1<f64>> {
        let mut entanglement_measures = Vec::new();
        for qubit in 0..self.config.num_qubits {
            let entropy = self.calculate_von_neumann_entropy(qubit)?;
            entanglement_measures.push(entropy);
        }
        Ok(Array1::from_vec(entanglement_measures))
    }
    /// Train the reservoir computer
    pub fn train(&mut self, training_data: &ReservoirTrainingData) -> Result<TrainingResult> {
        let start_time = std::time::Instant::now();
        let mut all_features = Vec::new();
        let mut all_targets = Vec::new();
        for i in 0..self.config.washout_period.min(training_data.inputs.len()) {
            let _ = self.process_input(&training_data.inputs[i])?;
        }
        for i in self.config.washout_period..training_data.inputs.len() {
            let features = self.process_input(&training_data.inputs[i])?;
            all_features.push(features);
            if i < training_data.targets.len() {
                all_targets.push(training_data.targets[i].clone());
            }
        }
        self.train_output_weights(&all_features, &all_targets)?;
        let (training_error, test_error) =
            self.evaluate_performance(&all_features, &all_targets)?;
        let training_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.metrics.training_examples += all_features.len();
        self.metrics.generalization_error = test_error;
        Ok(TrainingResult {
            training_error,
            test_error,
            training_time_ms: training_time,
            num_examples: all_features.len(),
            echo_state_property: self.estimate_echo_state_property()?,
        })
    }
    /// Reset reservoir computer
    pub fn reset(&mut self) -> Result<()> {
        self.reservoir_state =
            QuantumReservoirState::new(self.config.num_qubits, self.config.memory_capacity);
        self.metrics = ReservoirMetrics::default();
        self.training_history.clear();
        Ok(())
    }
}
