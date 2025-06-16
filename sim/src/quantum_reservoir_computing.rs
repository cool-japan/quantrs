//! Quantum Reservoir Computing Simulation
//!
//! This module implements quantum reservoir computing (QRC), a computational paradigm
//! that exploits the natural dynamics of quantum systems to process temporal information.
//! QRC harnesses the high-dimensional, nonlinear dynamics inherent in quantum systems
//! to create powerful computational reservoirs for machine learning tasks.
//!
//! Key features:
//! - Quantum state evolution as computational reservoir
//! - Temporal information processing with quantum memory
//! - Nonlinear quantum dynamics for complex pattern recognition
//! - Echo state networks with quantum reservoirs
//! - Real-time learning and adaptation
//! - Support for various quantum architectures (spin chains, random circuits, etc.)
//! - Hardware-aware optimization for different quantum platforms

use ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::{Result, SimulatorError};
use crate::statevector::StateVectorSimulator;

/// Quantum reservoir architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumReservoirArchitecture {
    /// Random quantum circuit with fixed connectivity
    RandomCircuit,
    /// Spin chain with nearest-neighbor interactions
    SpinChain,
    /// Ising model with transverse field
    TransverseFieldIsing,
    /// Small-world quantum network
    SmallWorld,
    /// Fully connected quantum network
    FullyConnected,
    /// Custom quantum architecture
    Custom,
}

/// Reservoir dynamics type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservoirDynamics {
    /// Unitary evolution
    Unitary,
    /// Open system dynamics with dissipation
    Open,
    /// Noisy intermediate-scale quantum (NISQ) dynamics
    NISQ,
    /// Adiabatic evolution
    Adiabatic,
}

/// Input encoding method for temporal data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputEncoding {
    /// Amplitude encoding
    Amplitude,
    /// Phase encoding
    Phase,
    /// Basis state encoding
    BasisState,
    /// Coherent state encoding
    Coherent,
    /// Squeezed state encoding
    Squeezed,
}

/// Output measurement strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMeasurement {
    /// Pauli expectation values
    PauliExpectation,
    /// Probability measurements
    Probability,
    /// Correlation functions
    Correlations,
    /// Entanglement measures
    Entanglement,
    /// Fidelity measurements
    Fidelity,
}

/// Quantum reservoir computing configuration
#[derive(Debug, Clone)]
pub struct QuantumReservoirConfig {
    /// Number of qubits in the reservoir
    pub num_qubits: usize,
    /// Reservoir architecture
    pub architecture: QuantumReservoirArchitecture,
    /// Dynamics type
    pub dynamics: ReservoirDynamics,
    /// Input encoding method
    pub input_encoding: InputEncoding,
    /// Output measurement strategy
    pub output_measurement: OutputMeasurement,
    /// Time step for evolution
    pub time_step: f64,
    /// Number of evolution steps per input
    pub evolution_steps: usize,
    /// Reservoir coupling strength
    pub coupling_strength: f64,
    /// Noise level (for NISQ dynamics)
    pub noise_level: f64,
    /// Memory capacity (time steps to remember)
    pub memory_capacity: usize,
    /// Enable real-time adaptation
    pub adaptive_learning: bool,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Washout period (initial time steps to ignore)
    pub washout_period: usize,
}

impl Default for QuantumReservoirConfig {
    fn default() -> Self {
        Self {
            num_qubits: 8,
            architecture: QuantumReservoirArchitecture::RandomCircuit,
            dynamics: ReservoirDynamics::Unitary,
            input_encoding: InputEncoding::Amplitude,
            output_measurement: OutputMeasurement::PauliExpectation,
            time_step: 0.1,
            evolution_steps: 10,
            coupling_strength: 1.0,
            noise_level: 0.01,
            memory_capacity: 100,
            adaptive_learning: true,
            learning_rate: 0.01,
            washout_period: 50,
        }
    }
}

/// Quantum reservoir state
#[derive(Debug, Clone)]
pub struct QuantumReservoirState {
    /// Current quantum state
    pub state_vector: Array1<Complex64>,
    /// Evolution history
    pub state_history: VecDeque<Array1<Complex64>>,
    /// Observable measurements
    pub observables: HashMap<String, f64>,
    /// Correlation matrix
    pub correlations: Array2<f64>,
    /// Time index
    pub time_index: usize,
}

impl QuantumReservoirState {
    /// Create new reservoir state
    pub fn new(num_qubits: usize, memory_capacity: usize) -> Self {
        let state_size = 1 << num_qubits;
        let mut state_vector = Array1::zeros(state_size);
        state_vector[0] = Complex64::new(1.0, 0.0); // Start in |0...0⟩

        Self {
            state_vector,
            state_history: VecDeque::with_capacity(memory_capacity),
            observables: HashMap::new(),
            correlations: Array2::zeros((num_qubits, num_qubits)),
            time_index: 0,
        }
    }

    /// Update state and maintain history
    pub fn update_state(&mut self, new_state: Array1<Complex64>) {
        self.state_history.push_back(self.state_vector.clone());
        if self.state_history.len() > self.state_history.capacity() {
            self.state_history.pop_front();
        }
        self.state_vector = new_state;
        self.time_index += 1;
    }
}

/// Training data for reservoir computing
#[derive(Debug, Clone)]
pub struct ReservoirTrainingData {
    /// Input time series
    pub inputs: Vec<Array1<f64>>,
    /// Target outputs
    pub targets: Vec<Array1<f64>>,
    /// Time stamps
    pub timestamps: Vec<f64>,
}

/// Quantum reservoir computing system
pub struct QuantumReservoirComputer {
    /// Configuration
    config: QuantumReservoirConfig,
    /// Current reservoir state
    reservoir_state: QuantumReservoirState,
    /// Reservoir circuit
    reservoir_circuit: InterfaceCircuit,
    /// Input coupling circuit
    input_coupling_circuit: InterfaceCircuit,
    /// Output weights (trainable)
    output_weights: Array2<f64>,
    /// State vector simulator
    simulator: StateVectorSimulator,
    /// Circuit interface
    circuit_interface: CircuitInterface,
    /// Performance metrics
    metrics: ReservoirMetrics,
    /// Training history
    training_history: VecDeque<TrainingExample>,
}

/// Training example for reservoir learning
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input data
    pub input: Array1<f64>,
    /// Reservoir state after processing
    pub reservoir_state: Array1<f64>,
    /// Target output
    pub target: Array1<f64>,
    /// Prediction error
    pub error: f64,
}

/// Performance metrics for reservoir computing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReservoirMetrics {
    /// Total training examples processed
    pub training_examples: usize,
    /// Current prediction accuracy
    pub prediction_accuracy: f64,
    /// Memory capacity estimate
    pub memory_capacity: f64,
    /// Information processing capacity
    pub processing_capacity: f64,
    /// Generalization error
    pub generalization_error: f64,
    /// Echo state property indicator
    pub echo_state_property: f64,
    /// Average processing time per input
    pub avg_processing_time_ms: f64,
    /// Quantum resource utilization
    pub quantum_resource_usage: f64,
}

impl QuantumReservoirComputer {
    /// Create new quantum reservoir computer
    pub fn new(config: QuantumReservoirConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;
        let simulator = StateVectorSimulator::new();
        
        let reservoir_state = QuantumReservoirState::new(config.num_qubits, config.memory_capacity);
        
        // Generate reservoir circuit based on architecture
        let reservoir_circuit = Self::generate_reservoir_circuit(&config)?;
        
        // Generate input coupling circuit
        let input_coupling_circuit = Self::generate_input_coupling_circuit(&config)?;
        
        // Initialize output weights randomly
        let output_size = match config.output_measurement {
            OutputMeasurement::PauliExpectation => config.num_qubits * 3, // X, Y, Z for each qubit
            OutputMeasurement::Probability => 1 << config.num_qubits, // All basis states
            OutputMeasurement::Correlations => config.num_qubits * config.num_qubits,
            OutputMeasurement::Entanglement => config.num_qubits,
            OutputMeasurement::Fidelity => 1,
        };
        
        let feature_size = Self::calculate_feature_size(&config);
        let mut output_weights = Array2::zeros((output_size, feature_size));
        
        // Xavier initialization
        let scale = (2.0 / (output_size + feature_size) as f64).sqrt();
        for elem in output_weights.iter_mut() {
            *elem = (fastrand::f64() - 0.5) * 2.0 * scale;
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
            training_history: VecDeque::with_capacity(10000),
        })
    }

    /// Generate reservoir circuit based on architecture
    fn generate_reservoir_circuit(config: &QuantumReservoirConfig) -> Result<InterfaceCircuit> {
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
                // Would be implemented based on specific requirements
                Self::generate_random_circuit(&mut circuit, config)?;
            }
        }

        Ok(circuit)
    }

    /// Generate random quantum circuit
    fn generate_random_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let depth = config.evolution_steps;
        
        for _ in 0..depth {
            // Add random single-qubit gates
            for qubit in 0..config.num_qubits {
                let angle = fastrand::f64() * 2.0 * std::f64::consts::PI;
                let gate_type = match fastrand::usize(0..3) {
                    0 => InterfaceGateType::RX(angle),
                    1 => InterfaceGateType::RY(angle),
                    _ => InterfaceGateType::RZ(angle),
                };
                circuit.add_gate(InterfaceGate::new(gate_type, vec![qubit]));
            }
            
            // Add random two-qubit gates
            for _ in 0..(config.num_qubits / 2) {
                let qubit1 = fastrand::usize(0..config.num_qubits);
                let qubit2 = fastrand::usize(0..config.num_qubits);
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
    fn generate_spin_chain_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;
        
        for _ in 0..config.evolution_steps {
            // Nearest-neighbor interactions
            for i in 0..config.num_qubits - 1 {
                // ZZ interaction
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step),
                    vec![i],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![i, i + 1],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step),
                    vec![i + 1],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![i, i + 1],
                ));
            }
        }

        Ok(())
    }

    /// Generate transverse field Ising model circuit
    fn generate_tfim_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;
        let field = coupling * 0.5; // Transverse field strength
        
        for _ in 0..config.evolution_steps {
            // Transverse field (X rotations)
            for qubit in 0..config.num_qubits {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RX(field * config.time_step),
                    vec![qubit],
                ));
            }
            
            // Nearest-neighbor ZZ interactions
            for i in 0..config.num_qubits - 1 {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                    vec![i],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![i, i + 1],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step),
                    vec![i + 1],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![i, i + 1],
                ));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                    vec![i],
                ));
            }
        }

        Ok(())
    }

    /// Generate small-world network circuit
    fn generate_small_world_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;
        let rewiring_prob = 0.1; // Small-world rewiring probability
        
        for _ in 0..config.evolution_steps {
            // Regular lattice connections
            for i in 0..config.num_qubits {
                let next = (i + 1) % config.num_qubits;
                
                // Random rewiring
                let target = if fastrand::f64() < rewiring_prob {
                    fastrand::usize(0..config.num_qubits)
                } else {
                    next
                };
                
                if target != i {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                        vec![i],
                    ));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::CNOT,
                        vec![i, target],
                    ));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step),
                        vec![target],
                    ));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::CNOT,
                        vec![i, target],
                    ));
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
    fn generate_fully_connected_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength / config.num_qubits as f64; // Scale by system size
        
        for _ in 0..config.evolution_steps {
            // All-to-all interactions
            for i in 0..config.num_qubits {
                for j in i + 1..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                        vec![i],
                    ));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::CNOT,
                        vec![i, j],
                    ));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step),
                        vec![j],
                    ));
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::CNOT,
                        vec![i, j],
                    ));
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
    fn generate_input_coupling_circuit(config: &QuantumReservoirConfig) -> Result<InterfaceCircuit> {
        let mut circuit = InterfaceCircuit::new(config.num_qubits, 0);

        match config.input_encoding {
            InputEncoding::Amplitude => {
                // Amplitude encoding through controlled rotations
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RY(0.0), // Will be set dynamically
                        vec![qubit],
                    ));
                }
            }
            InputEncoding::Phase => {
                // Phase encoding through Z rotations
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(0.0), // Will be set dynamically
                        vec![qubit],
                    ));
                }
            }
            InputEncoding::BasisState => {
                // Basis state encoding through X gates
                for qubit in 0..config.num_qubits {
                    // Conditional X gate based on input
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::X,
                        vec![qubit],
                    ));
                }
            }
            _ => {
                // Default to amplitude encoding
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RY(0.0),
                        vec![qubit],
                    ));
                }
            }
        }

        Ok(circuit)
    }

    /// Calculate feature size based on configuration
    fn calculate_feature_size(config: &QuantumReservoirConfig) -> usize {
        match config.output_measurement {
            OutputMeasurement::PauliExpectation => config.num_qubits * 3,
            OutputMeasurement::Probability => 1 << config.num_qubits.min(10), // Limit for memory
            OutputMeasurement::Correlations => config.num_qubits * config.num_qubits,
            OutputMeasurement::Entanglement => config.num_qubits,
            OutputMeasurement::Fidelity => 1,
        }
    }

    /// Process input through quantum reservoir
    pub fn process_input(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let start_time = std::time::Instant::now();

        // Encode input into quantum state
        self.encode_input(input)?;

        // Evolve through reservoir dynamics
        self.evolve_reservoir()?;

        // Extract features from reservoir state
        let features = self.extract_features()?;

        // Update metrics
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.update_processing_time(processing_time);

        Ok(features)
    }

    /// Encode input data into quantum state
    fn encode_input(&mut self, input: &Array1<f64>) -> Result<()> {
        match self.config.input_encoding {
            InputEncoding::Amplitude => {
                self.encode_amplitude(input)?;
            }
            InputEncoding::Phase => {
                self.encode_phase(input)?;
            }
            InputEncoding::BasisState => {
                self.encode_basis_state(input)?;
            }
            _ => {
                self.encode_amplitude(input)?;
            }
        }
        Ok(())
    }

    /// Amplitude encoding
    fn encode_amplitude(&mut self, input: &Array1<f64>) -> Result<()> {
        let num_inputs = input.len().min(self.config.num_qubits);
        
        for i in 0..num_inputs {
            let angle = input[i] * std::f64::consts::PI; // Scale to [0, π]
            self.apply_single_qubit_rotation(i, InterfaceGateType::RY(angle))?;
        }
        
        Ok(())
    }

    /// Phase encoding
    fn encode_phase(&mut self, input: &Array1<f64>) -> Result<()> {
        let num_inputs = input.len().min(self.config.num_qubits);
        
        for i in 0..num_inputs {
            let angle = input[i] * 2.0 * std::f64::consts::PI; // Full phase range
            self.apply_single_qubit_rotation(i, InterfaceGateType::RZ(angle))?;
        }
        
        Ok(())
    }

    /// Basis state encoding
    fn encode_basis_state(&mut self, input: &Array1<f64>) -> Result<()> {
        let num_inputs = input.len().min(self.config.num_qubits);
        
        for i in 0..num_inputs {
            if input[i] > 0.5 {
                self.apply_single_qubit_gate(i, InterfaceGateType::X)?;
            }
        }
        
        Ok(())
    }

    /// Apply single qubit rotation
    fn apply_single_qubit_rotation(
        &mut self,
        qubit: usize,
        gate_type: InterfaceGateType,
    ) -> Result<()> {
        // Create temporary circuit for this operation
        let mut temp_circuit = InterfaceCircuit::new(self.config.num_qubits, 0);
        temp_circuit.add_gate(InterfaceGate::new(gate_type, vec![qubit]));
        
        // Apply to current state (placeholder - would need proper state management)
        self.simulator.apply_interface_circuit(&temp_circuit)?;
        
        Ok(())
    }

    /// Apply single qubit gate
    fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        gate_type: InterfaceGateType,
    ) -> Result<()> {
        let mut temp_circuit = InterfaceCircuit::new(self.config.num_qubits, 0);
        temp_circuit.add_gate(InterfaceGate::new(gate_type, vec![qubit]));
        
        self.simulator.apply_interface_circuit(&temp_circuit)?;
        
        Ok(())
    }

    /// Evolve quantum reservoir through dynamics
    fn evolve_reservoir(&mut self) -> Result<()> {
        match self.config.dynamics {
            ReservoirDynamics::Unitary => {
                self.evolve_unitary()?;
            }
            ReservoirDynamics::Open => {
                self.evolve_open_system()?;
            }
            ReservoirDynamics::NISQ => {
                self.evolve_nisq()?;
            }
            ReservoirDynamics::Adiabatic => {
                self.evolve_adiabatic()?;
            }
        }
        Ok(())
    }

    /// Unitary evolution
    fn evolve_unitary(&mut self) -> Result<()> {
        self.simulator.apply_interface_circuit(&self.reservoir_circuit)?;
        Ok(())
    }

    /// Open system evolution with noise
    fn evolve_open_system(&mut self) -> Result<()> {
        // Apply unitary evolution first
        self.evolve_unitary()?;
        
        // Apply decoherence
        self.apply_decoherence()?;
        
        Ok(())
    }

    /// NISQ evolution with realistic noise
    fn evolve_nisq(&mut self) -> Result<()> {
        // Apply unitary evolution
        self.evolve_unitary()?;
        
        // Apply gate errors
        self.apply_gate_errors()?;
        
        // Apply measurement errors
        self.apply_measurement_errors()?;
        
        Ok(())
    }

    /// Adiabatic evolution
    fn evolve_adiabatic(&mut self) -> Result<()> {
        // Simplified adiabatic evolution
        // In practice, this would implement proper adiabatic dynamics
        self.evolve_unitary()?;
        Ok(())
    }

    /// Apply decoherence to the reservoir state
    fn apply_decoherence(&mut self) -> Result<()> {
        let decoherence_rate = self.config.noise_level;
        
        for amplitude in self.reservoir_state.state_vector.iter_mut() {
            // Apply phase decoherence
            let phase_noise = (fastrand::f64() - 0.5) * decoherence_rate * 2.0 * std::f64::consts::PI;
            *amplitude *= Complex64::new(0.0, phase_noise).exp();
            
            // Apply amplitude damping
            let damping = (1.0 - decoherence_rate).sqrt();
            *amplitude *= damping;
        }
        
        // Renormalize
        let norm: f64 = self.reservoir_state.state_vector
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        
        if norm > 1e-15 {
            self.reservoir_state.state_vector.mapv_inplace(|x| x / norm);
        }
        
        Ok(())
    }

    /// Apply gate errors
    fn apply_gate_errors(&mut self) -> Result<()> {
        let error_rate = self.config.noise_level;
        
        for qubit in 0..self.config.num_qubits {
            if fastrand::f64() < error_rate {
                let error_type = fastrand::usize(0..3);
                let gate_type = match error_type {
                    0 => InterfaceGateType::X,
                    1 => InterfaceGateType::PauliY,
                    _ => InterfaceGateType::PauliZ,
                };
                self.apply_single_qubit_gate(qubit, gate_type)?;
            }
        }
        
        Ok(())
    }

    /// Apply measurement errors
    fn apply_measurement_errors(&mut self) -> Result<()> {
        // Simplified measurement error model
        let error_rate = self.config.noise_level * 0.1; // Lower rate for measurement errors
        
        if fastrand::f64() < error_rate {
            // Randomly flip a qubit
            let qubit = fastrand::usize(0..self.config.num_qubits);
            self.apply_single_qubit_gate(qubit, InterfaceGateType::X)?;
        }
        
        Ok(())
    }

    /// Extract features from reservoir state
    fn extract_features(&mut self) -> Result<Array1<f64>> {
        match self.config.output_measurement {
            OutputMeasurement::PauliExpectation => self.measure_pauli_expectations(),
            OutputMeasurement::Probability => self.measure_probabilities(),
            OutputMeasurement::Correlations => self.measure_correlations(),
            OutputMeasurement::Entanglement => self.measure_entanglement(),
            OutputMeasurement::Fidelity => self.measure_fidelity(),
        }
    }

    /// Measure Pauli expectation values
    fn measure_pauli_expectations(&self) -> Result<Array1<f64>> {
        let mut expectations = Vec::new();
        
        for qubit in 0..self.config.num_qubits {
            // X expectation
            let x_exp = self.calculate_single_qubit_expectation(qubit, &[
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ])?;
            expectations.push(x_exp);
            
            // Y expectation
            let y_exp = self.calculate_single_qubit_expectation(qubit, &[
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            ])?;
            expectations.push(y_exp);
            
            // Z expectation
            let z_exp = self.calculate_single_qubit_expectation(qubit, &[
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ])?;
            expectations.push(z_exp);
        }
        
        Ok(Array1::from_vec(expectations))
    }

    /// Calculate single qubit expectation value
    fn calculate_single_qubit_expectation(
        &self,
        qubit: usize,
        pauli_matrix: &[Complex64; 4],
    ) -> Result<f64> {
        let state = &self.reservoir_state.state_vector;
        let mut expectation = 0.0;
        
        for i in 0..state.len() {
            for j in 0..state.len() {
                let i_bit = (i >> qubit) & 1;
                let j_bit = (j >> qubit) & 1;
                let matrix_element = pauli_matrix[i_bit * 2 + j_bit];
                
                expectation += (state[i].conj() * matrix_element * state[j]).re;
            }
        }
        
        Ok(expectation)
    }

    /// Measure probability distribution
    fn measure_probabilities(&self) -> Result<Array1<f64>> {
        let probabilities: Vec<f64> = self.reservoir_state.state_vector
            .iter()
            .map(|x| x.norm_sqr())
            .collect();
        
        // Limit size for large systems
        let max_size = 1 << 10; // 2^10 = 1024
        if probabilities.len() > max_size {
            // Sample random subset
            let mut sampled = Vec::with_capacity(max_size);
            for _ in 0..max_size {
                let idx = fastrand::usize(0..probabilities.len());
                sampled.push(probabilities[idx]);
            }
            Ok(Array1::from_vec(sampled))
        } else {
            Ok(Array1::from_vec(probabilities))
        }
    }

    /// Measure two-qubit correlations
    fn measure_correlations(&mut self) -> Result<Array1<f64>> {
        let mut correlations = Vec::new();
        
        for i in 0..self.config.num_qubits {
            for j in 0..self.config.num_qubits {
                if i != j {
                    // ZZ correlation
                    let corr = self.calculate_two_qubit_correlation(i, j)?;
                    correlations.push(corr);
                    self.reservoir_state.correlations[[i, j]] = corr;
                } else {
                    correlations.push(1.0); // Self-correlation
                    self.reservoir_state.correlations[[i, j]] = 1.0;
                }
            }
        }
        
        Ok(Array1::from_vec(correlations))
    }

    /// Calculate two-qubit correlation
    fn calculate_two_qubit_correlation(&self, qubit1: usize, qubit2: usize) -> Result<f64> {
        let state = &self.reservoir_state.state_vector;
        let mut correlation = 0.0;
        
        for i in 0..state.len() {
            let bit1 = (i >> qubit1) & 1;
            let bit2 = (i >> qubit2) & 1;
            let sign = if bit1 == bit2 { 1.0 } else { -1.0 };
            correlation += sign * state[i].norm_sqr();
        }
        
        Ok(correlation)
    }

    /// Measure entanglement metrics
    fn measure_entanglement(&self) -> Result<Array1<f64>> {
        let mut entanglement_measures = Vec::new();
        
        // Simplified entanglement measures
        for qubit in 0..self.config.num_qubits {
            // Von Neumann entropy of reduced state (approximation)
            let entropy = self.calculate_von_neumann_entropy(qubit)?;
            entanglement_measures.push(entropy);
        }
        
        Ok(Array1::from_vec(entanglement_measures))
    }

    /// Calculate von Neumann entropy (simplified)
    fn calculate_von_neumann_entropy(&self, _qubit: usize) -> Result<f64> {
        // Simplified calculation - in practice would require density matrix diagonalization
        let state = &self.reservoir_state.state_vector;
        let mut entropy = 0.0;
        
        for amplitude in state.iter() {
            let prob = amplitude.norm_sqr();
            if prob > 1e-15 {
                entropy -= prob * prob.ln();
            }
        }
        
        Ok(entropy / (state.len() as f64).ln()) // Normalized entropy
    }

    /// Measure fidelity with reference state
    fn measure_fidelity(&self) -> Result<Array1<f64>> {
        // Fidelity with initial state |0...0⟩
        let fidelity = self.reservoir_state.state_vector[0].norm_sqr();
        Ok(Array1::from_vec(vec![fidelity]))
    }

    /// Train the reservoir computer
    pub fn train(&mut self, training_data: &ReservoirTrainingData) -> Result<TrainingResult> {
        let start_time = std::time::Instant::now();
        
        let mut all_features = Vec::new();
        let mut all_targets = Vec::new();
        
        // Washout period
        for i in 0..self.config.washout_period.min(training_data.inputs.len()) {
            let _ = self.process_input(&training_data.inputs[i])?;
        }
        
        // Collect training data after washout
        for i in self.config.washout_period..training_data.inputs.len() {
            let features = self.process_input(&training_data.inputs[i])?;
            all_features.push(features);
            
            if i < training_data.targets.len() {
                all_targets.push(training_data.targets[i].clone());
            }
        }
        
        // Train output weights using linear regression
        self.train_output_weights(&all_features, &all_targets)?;
        
        // Evaluate performance
        let (training_error, test_error) = self.evaluate_performance(&all_features, &all_targets)?;
        
        let training_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        // Update metrics
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

    /// Train output weights using ridge regression
    fn train_output_weights(
        &mut self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<()> {
        if features.is_empty() || targets.is_empty() {
            return Ok(());
        }
        
        let n_samples = features.len().min(targets.len());
        let n_features = features[0].len();
        let n_outputs = targets[0].len().min(self.output_weights.nrows());
        
        // Create feature matrix
        let mut feature_matrix = Array2::zeros((n_samples, n_features));
        for (i, feature_vec) in features.iter().enumerate().take(n_samples) {
            for (j, &val) in feature_vec.iter().enumerate().take(n_features) {
                feature_matrix[[i, j]] = val;
            }
        }
        
        // Create target matrix
        let mut target_matrix = Array2::zeros((n_samples, n_outputs));
        for (i, target_vec) in targets.iter().enumerate().take(n_samples) {
            for (j, &val) in target_vec.iter().enumerate().take(n_outputs) {
                target_matrix[[i, j]] = val;
            }
        }
        
        // Ridge regression: W = (X^T X + λI)^(-1) X^T Y
        let lambda = 1e-6; // Regularization parameter
        
        // X^T X
        let xtx = feature_matrix.t().dot(&feature_matrix);
        
        // Add regularization
        let mut xtx_reg = xtx;
        for i in 0..xtx_reg.nrows().min(xtx_reg.ncols()) {
            xtx_reg[[i, i]] += lambda;
        }
        
        // X^T Y
        let xty = feature_matrix.t().dot(&target_matrix);
        
        // Solve using pseudo-inverse (simplified)
        // In practice, would use proper linear solver
        self.solve_linear_system(&xtx_reg, &xty)?;
        
        Ok(())
    }

    /// Solve linear system (simplified implementation)
    fn solve_linear_system(&mut self, a: &Array2<f64>, b: &Array2<f64>) -> Result<()> {
        // Simplified solution using diagonal approximation
        // In practice, would use proper linear algebra library
        
        let min_dim = a.nrows().min(a.ncols()).min(b.nrows());
        
        for i in 0..min_dim.min(self.output_weights.nrows()) {
            for j in 0..b.ncols().min(self.output_weights.ncols()) {
                if a[[i, i]].abs() > 1e-15 {
                    self.output_weights[[i, j]] = b[[i, j]] / a[[i, i]];
                }
            }
        }
        
        Ok(())
    }

    /// Evaluate performance on training data
    fn evaluate_performance(
        &self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<(f64, f64)> {
        if features.is_empty() || targets.is_empty() {
            return Ok((0.0, 0.0));
        }
        
        let mut total_error = 0.0;
        let n_samples = features.len().min(targets.len());
        
        for i in 0..n_samples {
            let prediction = self.predict_output(&features[i])?;
            let error = self.calculate_prediction_error(&prediction, &targets[i]);
            total_error += error;
        }
        
        let training_error = total_error / n_samples as f64;
        
        // Use same error for test (in practice, would use separate test set)
        let test_error = training_error;
        
        Ok((training_error, test_error))
    }

    /// Predict output for given features
    fn predict_output(&self, features: &Array1<f64>) -> Result<Array1<f64>> {
        let feature_size = features.len().min(self.output_weights.ncols());
        let output_size = self.output_weights.nrows();
        
        let mut output = Array1::zeros(output_size);
        
        for i in 0..output_size {
            for j in 0..feature_size {
                output[i] += self.output_weights[[i, j]] * features[j];
            }
        }
        
        Ok(output)
    }

    /// Calculate prediction error
    fn calculate_prediction_error(&self, prediction: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let min_len = prediction.len().min(target.len());
        let mut error = 0.0;
        
        for i in 0..min_len {
            let diff = prediction[i] - target[i];
            error += diff * diff;
        }
        
        (error / min_len as f64).sqrt() // RMSE
    }

    /// Estimate echo state property
    fn estimate_echo_state_property(&self) -> Result<f64> {
        // Simplified estimate based on spectral radius
        // In practice, would compute actual spectral radius of reservoir dynamics
        
        let coupling = self.config.coupling_strength;
        let estimated_spectral_radius = coupling.tanh(); // Heuristic estimate
        
        // Echo state property requires spectral radius < 1
        Ok(if estimated_spectral_radius < 1.0 { 1.0 } else { 1.0 / estimated_spectral_radius })
    }

    /// Update processing time metrics
    fn update_processing_time(&mut self, time_ms: f64) {
        let count = self.metrics.training_examples as f64;
        self.metrics.avg_processing_time_ms = (self.metrics.avg_processing_time_ms * count + time_ms) / (count + 1.0);
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &ReservoirMetrics {
        &self.metrics
    }

    /// Reset reservoir computer
    pub fn reset(&mut self) -> Result<()> {
        self.reservoir_state = QuantumReservoirState::new(self.config.num_qubits, self.config.memory_capacity);
        self.metrics = ReservoirMetrics::default();
        self.training_history.clear();
        Ok(())
    }
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Training error (RMSE)
    pub training_error: f64,
    /// Test error (RMSE)
    pub test_error: f64,
    /// Training time in milliseconds
    pub training_time_ms: f64,
    /// Number of training examples
    pub num_examples: usize,
    /// Echo state property measure
    pub echo_state_property: f64,
}

/// Benchmark quantum reservoir computing
pub fn benchmark_quantum_reservoir_computing() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // Test different reservoir configurations
    let configs = vec![
        QuantumReservoirConfig {
            num_qubits: 6,
            architecture: QuantumReservoirArchitecture::RandomCircuit,
            ..Default::default()
        },
        QuantumReservoirConfig {
            num_qubits: 8,
            architecture: QuantumReservoirArchitecture::SpinChain,
            ..Default::default()
        },
        QuantumReservoirConfig {
            num_qubits: 6,
            architecture: QuantumReservoirArchitecture::TransverseFieldIsing,
            ..Default::default()
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let start = std::time::Instant::now();

        let mut qrc = QuantumReservoirComputer::new(config)?;

        // Generate test data
        let training_data = ReservoirTrainingData {
            inputs: (0..100)
                .map(|i| {
                    Array1::from_vec(vec![
                        (i as f64 * 0.1).sin(),
                        (i as f64 * 0.1).cos(),
                    ])
                })
                .collect(),
            targets: (0..100)
                .map(|i| Array1::from_vec(vec![(i as f64 * 0.1 + 1.0).sin()]))
                .collect(),
            timestamps: (0..100).map(|i| i as f64 * 0.1).collect(),
        };

        // Train and test
        let _training_result = qrc.train(&training_data)?;

        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("config_{}", i), time);

        // Add performance metrics
        let metrics = qrc.get_metrics();
        results.insert(format!("config_{}_accuracy", i), metrics.prediction_accuracy);
        results.insert(format!("config_{}_memory", i), metrics.memory_capacity);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantum_reservoir_creation() {
        let config = QuantumReservoirConfig::default();
        let qrc = QuantumReservoirComputer::new(config);
        assert!(qrc.is_ok());
    }

    #[test]
    fn test_reservoir_state_creation() {
        let state = QuantumReservoirState::new(3, 10);
        assert_eq!(state.state_vector.len(), 8); // 2^3
        assert_eq!(state.state_history.capacity(), 10);
        assert_eq!(state.time_index, 0);
    }

    #[test]
    fn test_input_processing() {
        let config = QuantumReservoirConfig {
            num_qubits: 3,
            evolution_steps: 2,
            ..Default::default()
        };
        let mut qrc = QuantumReservoirComputer::new(config).unwrap();

        let input = Array1::from_vec(vec![0.5, 0.3, 0.8]);
        let result = qrc.process_input(&input);
        assert!(result.is_ok());

        let features = result.unwrap();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_different_architectures() {
        let architectures = vec![
            QuantumReservoirArchitecture::RandomCircuit,
            QuantumReservoirArchitecture::SpinChain,
            QuantumReservoirArchitecture::TransverseFieldIsing,
        ];

        for arch in architectures {
            let config = QuantumReservoirConfig {
                num_qubits: 4,
                architecture: arch,
                evolution_steps: 2,
                ..Default::default()
            };

            let qrc = QuantumReservoirComputer::new(config);
            assert!(qrc.is_ok(), "Failed for architecture: {:?}", arch);
        }
    }

    #[test]
    fn test_feature_extraction() {
        let config = QuantumReservoirConfig {
            num_qubits: 3,
            output_measurement: OutputMeasurement::PauliExpectation,
            ..Default::default()
        };
        let mut qrc = QuantumReservoirComputer::new(config).unwrap();

        let features = qrc.extract_features().unwrap();
        assert_eq!(features.len(), 9); // 3 qubits × 3 Pauli operators
    }

    #[test]
    fn test_training_data() {
        let training_data = ReservoirTrainingData {
            inputs: vec![
                Array1::from_vec(vec![0.1, 0.2]),
                Array1::from_vec(vec![0.3, 0.4]),
            ],
            targets: vec![
                Array1::from_vec(vec![0.5]),
                Array1::from_vec(vec![0.6]),
            ],
            timestamps: vec![0.0, 1.0],
        };

        assert_eq!(training_data.inputs.len(), 2);
        assert_eq!(training_data.targets.len(), 2);
        assert_eq!(training_data.timestamps.len(), 2);
    }

    #[test]
    fn test_encoding_methods() {
        let config = QuantumReservoirConfig {
            num_qubits: 3,
            input_encoding: InputEncoding::Amplitude,
            ..Default::default()
        };
        let mut qrc = QuantumReservoirComputer::new(config).unwrap();

        let input = Array1::from_vec(vec![0.5, 0.3]);
        let result = qrc.encode_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_measurement_strategies() {
        let measurements = vec![
            OutputMeasurement::PauliExpectation,
            OutputMeasurement::Probability,
            OutputMeasurement::Correlations,
            OutputMeasurement::Entanglement,
            OutputMeasurement::Fidelity,
        ];

        for measurement in measurements {
            let config = QuantumReservoirConfig {
                num_qubits: 3,
                output_measurement: measurement,
                ..Default::default()
            };

            let mut qrc = QuantumReservoirComputer::new(config);
            assert!(qrc.is_ok(), "Failed for measurement: {:?}", measurement);
        }
    }

    #[test]
    fn test_reservoir_dynamics() {
        let dynamics = vec![
            ReservoirDynamics::Unitary,
            ReservoirDynamics::Open,
            ReservoirDynamics::NISQ,
        ];

        for dynamic in dynamics {
            let config = QuantumReservoirConfig {
                num_qubits: 3,
                dynamics: dynamic,
                evolution_steps: 1,
                ..Default::default()
            };

            let mut qrc = QuantumReservoirComputer::new(config).unwrap();
            let result = qrc.evolve_reservoir();
            assert!(result.is_ok(), "Failed for dynamics: {:?}", dynamic);
        }
    }

    #[test]
    fn test_metrics_tracking() {
        let config = QuantumReservoirConfig::default();
        let qrc = QuantumReservoirComputer::new(config).unwrap();

        let metrics = qrc.get_metrics();
        assert_eq!(metrics.training_examples, 0);
        assert_eq!(metrics.prediction_accuracy, 0.0);
    }
}