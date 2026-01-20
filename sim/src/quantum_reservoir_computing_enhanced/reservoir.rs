//! Enhanced Quantum Reservoir Computing Implementation
//!
//! This module provides the main QuantumReservoirComputerEnhanced struct and its implementations.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::Complex64;
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::Result;
use crate::scirs2_integration::SciRS2Backend;
use crate::statevector::StateVectorSimulator;

use super::analysis::MemoryAnalyzer;
use super::config::QuantumReservoirConfig;
use super::state::{
    QuantumReservoirState, ReservoirMetrics, ReservoirTrainingData, TrainingExample, TrainingResult,
};
use super::time_series::TimeSeriesPredictor;
use super::types::{
    IPCFunction, InputEncoding, LearningAlgorithm, OutputMeasurement, QuantumReservoirArchitecture,
    ReservoirDynamics,
};

/// Enhanced quantum reservoir computing system
pub struct QuantumReservoirComputerEnhanced {
    /// Configuration
    pub(crate) config: QuantumReservoirConfig,
    /// Current reservoir state
    pub(crate) reservoir_state: QuantumReservoirState,
    /// Reservoir circuit
    reservoir_circuit: InterfaceCircuit,
    /// Input coupling circuit
    #[allow(dead_code)]
    input_coupling_circuit: InterfaceCircuit,
    /// Output weights (trainable)
    output_weights: Array2<f64>,
    /// Time series predictor
    #[allow(dead_code)]
    time_series_predictor: Option<TimeSeriesPredictor>,
    /// Memory analyzer
    memory_analyzer: MemoryAnalyzer,
    /// State vector simulator
    simulator: StateVectorSimulator,
    /// Performance metrics
    metrics: ReservoirMetrics,
    /// Training history
    #[allow(dead_code)]
    training_history: VecDeque<TrainingExample>,
    /// `SciRS2` backend for advanced computations
    #[allow(dead_code)]
    backend: Option<SciRS2Backend>,
    /// Random number generator
    #[allow(dead_code)]
    rng: Arc<Mutex<scirs2_core::random::CoreRandom>>,
}

impl QuantumReservoirComputerEnhanced {
    /// Create new enhanced quantum reservoir computer
    pub fn new(config: QuantumReservoirConfig) -> Result<Self> {
        let simulator = StateVectorSimulator::new();

        let reservoir_state = QuantumReservoirState::new(config.num_qubits, config.memory_capacity);

        // Generate reservoir circuit based on architecture
        let reservoir_circuit = Self::generate_reservoir_circuit(&config)?;

        // Generate input coupling circuit
        let input_coupling_circuit = Self::generate_input_coupling_circuit(&config)?;

        // Initialize output weights randomly
        let output_size = Self::calculate_output_size(&config);
        let feature_size = Self::calculate_feature_size(&config);
        let mut output_weights = Array2::zeros((output_size, feature_size));

        // Xavier initialization
        let scale = (2.0 / (output_size + feature_size) as f64).sqrt();
        for elem in &mut output_weights {
            *elem = (fastrand::f64() - 0.5) * 2.0 * scale;
        }

        // Initialize time series predictor if enabled
        let time_series_predictor =
            if config.time_series_config.enable_arima || config.time_series_config.enable_nar {
                Some(TimeSeriesPredictor::new(&config.time_series_config))
            } else {
                None
            };

        // Initialize memory analyzer
        let memory_analyzer = MemoryAnalyzer::new(config.memory_config.clone());

        Ok(Self {
            config,
            reservoir_state,
            reservoir_circuit,
            input_coupling_circuit,
            output_weights,
            time_series_predictor,
            memory_analyzer,
            simulator,
            metrics: ReservoirMetrics::default(),
            training_history: VecDeque::with_capacity(10_000),
            backend: None,
            rng: Arc::new(Mutex::new(thread_rng())),
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
            QuantumReservoirArchitecture::ScaleFree => {
                Self::generate_scale_free_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::HierarchicalModular => {
                Self::generate_hierarchical_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::Ring => {
                Self::generate_ring_circuit(&mut circuit, config)?;
            }
            QuantumReservoirArchitecture::Grid => {
                Self::generate_grid_circuit(&mut circuit, config)?;
            }
            _ => {
                // Default to random circuit for other architectures
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
                let angle = fastrand::f64() * 2.0 * PI;
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

    /// Generate scale-free network circuit
    fn generate_scale_free_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        // Implement scale-free topology with preferential attachment
        let mut degree_dist = vec![1; config.num_qubits];
        let coupling = config.coupling_strength;

        for _ in 0..config.evolution_steps {
            // Scale-free connections based on degree distribution
            for i in 0..config.num_qubits {
                // Probability proportional to degree
                let total_degree: usize = degree_dist.iter().sum();
                let prob_threshold = degree_dist[i] as f64 / total_degree as f64;

                if fastrand::f64() < prob_threshold {
                    let j = fastrand::usize(0..config.num_qubits);
                    if i != j {
                        // Add interaction
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

                        // Update degrees
                        degree_dist[i] += 1;
                        degree_dist[j] += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate hierarchical modular circuit
    fn generate_hierarchical_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;
        let module_size = (config.num_qubits as f64).sqrt() as usize;

        for _ in 0..config.evolution_steps {
            // Intra-module connections (stronger)
            for module in 0..(config.num_qubits / module_size) {
                let start = module * module_size;
                let end = ((module + 1) * module_size).min(config.num_qubits);

                for i in start..end {
                    for j in (i + 1)..end {
                        circuit.add_gate(InterfaceGate::new(
                            InterfaceGateType::RZ(coupling * config.time_step),
                            vec![i],
                        ));
                        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, j]));
                    }
                }
            }

            // Inter-module connections (weaker)
            for i in 0..config.num_qubits {
                let j = fastrand::usize(0..config.num_qubits);
                if i / module_size != j / module_size && i != j {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::RZ(coupling * config.time_step * 0.3),
                        vec![i],
                    ));
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, j]));
                }
            }
        }

        Ok(())
    }

    /// Generate ring topology circuit
    fn generate_ring_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;

        for _ in 0..config.evolution_steps {
            // Ring connections
            for i in 0..config.num_qubits {
                let j = (i + 1) % config.num_qubits;

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
            }

            // Long-range connections (sparse)
            if fastrand::f64() < 0.1 {
                let i = fastrand::usize(0..config.num_qubits);
                let j = fastrand::usize(0..config.num_qubits);
                if i != j && (i as i32 - j as i32).abs() > 2 {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, j]));
                }
            }
        }

        Ok(())
    }

    /// Generate grid topology circuit
    fn generate_grid_circuit(
        circuit: &mut InterfaceCircuit,
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let coupling = config.coupling_strength;
        let grid_size = (config.num_qubits as f64).sqrt() as usize;

        for _ in 0..config.evolution_steps {
            // Grid connections (nearest neighbors)
            for i in 0..grid_size {
                for j in 0..grid_size {
                    let current = i * grid_size + j;
                    if current >= config.num_qubits {
                        break;
                    }

                    // Right neighbor
                    if j + 1 < grid_size {
                        let neighbor = i * grid_size + j + 1;
                        if neighbor < config.num_qubits {
                            circuit.add_gate(InterfaceGate::new(
                                InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                                vec![current],
                            ));
                            circuit.add_gate(InterfaceGate::new(
                                InterfaceGateType::CNOT,
                                vec![current, neighbor],
                            ));
                        }
                    }

                    // Bottom neighbor
                    if i + 1 < grid_size {
                        let neighbor = (i + 1) * grid_size + j;
                        if neighbor < config.num_qubits {
                            circuit.add_gate(InterfaceGate::new(
                                InterfaceGateType::RZ(coupling * config.time_step / 2.0),
                                vec![current],
                            ));
                            circuit.add_gate(InterfaceGate::new(
                                InterfaceGateType::CNOT,
                                vec![current, neighbor],
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate input coupling circuit
    fn generate_input_coupling_circuit(
        config: &QuantumReservoirConfig,
    ) -> Result<InterfaceCircuit> {
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
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::X, vec![qubit]));
                }
            }
            InputEncoding::Angle => {
                // Angle encoding with multiple rotation axes
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::RX(0.0), vec![qubit]));
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.0), vec![qubit]));
                }
            }
            _ => {
                // Default to amplitude encoding
                for qubit in 0..config.num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.0), vec![qubit]));
                }
            }
        }

        Ok(circuit)
    }

    /// Calculate output size based on configuration
    pub const fn calculate_output_size(_config: &QuantumReservoirConfig) -> usize {
        // For time series prediction, typically 1 output
        1
    }

    /// Calculate feature size based on configuration
    pub fn calculate_feature_size(config: &QuantumReservoirConfig) -> usize {
        match config.output_measurement {
            OutputMeasurement::PauliExpectation => config.num_qubits * 3,
            OutputMeasurement::Probability => 1 << config.num_qubits.min(10), // Limit for memory
            OutputMeasurement::Correlations => config.num_qubits * config.num_qubits,
            OutputMeasurement::Entanglement => config.num_qubits,
            OutputMeasurement::Fidelity => 1,
            OutputMeasurement::QuantumFisherInformation => config.num_qubits,
            OutputMeasurement::Variance => config.num_qubits * 3,
            OutputMeasurement::HigherOrderMoments => config.num_qubits * 6, // Up to 3rd moments
            OutputMeasurement::SpectralProperties => config.num_qubits,
            OutputMeasurement::QuantumCoherence => config.num_qubits,
            OutputMeasurement::Purity => 1,
            OutputMeasurement::QuantumMutualInformation => config.num_qubits * config.num_qubits,
            OutputMeasurement::ProcessTomography => config.num_qubits * config.num_qubits * 4,
            OutputMeasurement::TemporalCorrelations => config.memory_capacity,
            OutputMeasurement::NonLinearReadout => config.num_qubits * 2,
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

        // Update reservoir state with timestamp
        let timestamp = start_time.elapsed().as_secs_f64();
        self.reservoir_state
            .update_state(self.reservoir_state.state_vector.clone(), timestamp);

        // Update metrics
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.update_processing_time(processing_time);

        Ok(features)
    }

    /// Encode input data into quantum state
    pub fn encode_input(&mut self, input: &Array1<f64>) -> Result<()> {
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
            InputEncoding::Angle => {
                self.encode_angle(input)?;
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
            let angle = input[i] * PI; // Scale to [0, π]
            self.apply_single_qubit_rotation(i, InterfaceGateType::RY(angle))?;
        }

        Ok(())
    }

    /// Phase encoding
    fn encode_phase(&mut self, input: &Array1<f64>) -> Result<()> {
        let num_inputs = input.len().min(self.config.num_qubits);

        for i in 0..num_inputs {
            let angle = input[i] * 2.0 * PI; // Full phase range
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

    /// Angle encoding with multiple rotation axes
    fn encode_angle(&mut self, input: &Array1<f64>) -> Result<()> {
        let num_inputs = input.len().min(self.config.num_qubits);

        for i in 0..num_inputs {
            let angle_x = input[i] * PI;
            let angle_y = if i + 1 < input.len() {
                input[i + 1] * PI
            } else {
                0.0
            };

            self.apply_single_qubit_rotation(i, InterfaceGateType::RX(angle_x))?;
            self.apply_single_qubit_rotation(i, InterfaceGateType::RY(angle_y))?;
        }

        Ok(())
    }

    /// Apply single qubit rotation
    fn apply_single_qubit_rotation(
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
    pub fn evolve_reservoir(&mut self) -> Result<()> {
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
            ReservoirDynamics::Floquet => {
                self.evolve_floquet()?;
            }
            _ => {
                // Default to unitary evolution
                self.evolve_unitary()?;
            }
        }
        Ok(())
    }

    /// Unitary evolution
    fn evolve_unitary(&mut self) -> Result<()> {
        self.simulator
            .apply_interface_circuit(&self.reservoir_circuit)?;
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

    /// Floquet evolution with periodic driving
    fn evolve_floquet(&mut self) -> Result<()> {
        // Apply time-dependent Hamiltonian
        let drive_frequency = 1.0;
        let time = self.reservoir_state.time_index as f64 * self.config.time_step;
        let drive_strength = (drive_frequency * time).sin();

        // Apply driving field
        for qubit in 0..self.config.num_qubits {
            let angle = drive_strength * self.config.time_step;
            self.apply_single_qubit_rotation(qubit, InterfaceGateType::RX(angle))?;
        }

        // Apply base evolution
        self.evolve_unitary()?;

        Ok(())
    }

    /// Apply decoherence to the reservoir state
    fn apply_decoherence(&mut self) -> Result<()> {
        let decoherence_rate = self.config.noise_level;

        for amplitude in &mut self.reservoir_state.state_vector {
            // Apply phase decoherence
            let phase_noise = (fastrand::f64() - 0.5) * decoherence_rate * 2.0 * PI;
            *amplitude *= Complex64::new(0.0, phase_noise).exp();

            // Apply amplitude damping
            let damping = (1.0 - decoherence_rate).sqrt();
            *amplitude *= damping;
        }

        // Renormalize
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
        let error_rate = self.config.noise_level * 0.1; // Lower rate for measurement errors

        if fastrand::f64() < error_rate {
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
            OutputMeasurement::QuantumFisherInformation => {
                self.measure_quantum_fisher_information()
            }
            OutputMeasurement::Variance => self.measure_variance(),
            OutputMeasurement::HigherOrderMoments => self.measure_higher_order_moments(),
            OutputMeasurement::QuantumCoherence => self.measure_quantum_coherence(),
            OutputMeasurement::Purity => self.measure_purity(),
            OutputMeasurement::TemporalCorrelations => self.measure_temporal_correlations(),
            _ => {
                // Default to Pauli expectations
                self.measure_pauli_expectations()
            }
        }
    }

    /// Measure Pauli expectation values
    fn measure_pauli_expectations(&self) -> Result<Array1<f64>> {
        let mut expectations = Vec::new();

        for qubit in 0..self.config.num_qubits {
            // X expectation
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

            // Y expectation
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

            // Z expectation
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
        let probabilities: Vec<f64> = self
            .reservoir_state
            .state_vector
            .iter()
            .map(scirs2_core::Complex::norm_sqr)
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
                if i == j {
                    correlations.push(1.0); // Self-correlation
                    self.reservoir_state.correlations[[i, j]] = 1.0;
                } else {
                    // ZZ correlation
                    let corr = self.calculate_two_qubit_correlation(i, j)?;
                    correlations.push(corr);
                    self.reservoir_state.correlations[[i, j]] = corr;
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
        let state = &self.reservoir_state.state_vector;
        let mut entropy = 0.0;

        for amplitude in state {
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

    /// Measure quantum Fisher information
    fn measure_quantum_fisher_information(&self) -> Result<Array1<f64>> {
        let mut qfi_values = Vec::new();

        for qubit in 0..self.config.num_qubits {
            // Simplified QFI calculation for single qubit observables
            let z_exp = self.calculate_single_qubit_expectation(
                qubit,
                &[
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )?;

            // QFI ≈ 4 * Var(Z) for single qubit
            let qfi = 4.0 * (1.0 - z_exp * z_exp);
            qfi_values.push(qfi);
        }

        Ok(Array1::from_vec(qfi_values))
    }

    /// Measure variance of observables
    fn measure_variance(&self) -> Result<Array1<f64>> {
        let mut variances = Vec::new();

        for qubit in 0..self.config.num_qubits {
            // X, Y, Z variances
            for pauli_idx in 0..3 {
                let pauli_matrix = match pauli_idx {
                    0 => [
                        Complex64::new(0.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                    ],
                    1 => [
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, -1.0),
                        Complex64::new(0.0, 1.0),
                        Complex64::new(0.0, 0.0),
                    ],
                    _ => [
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(-1.0, 0.0),
                    ],
                };

                let expectation = self.calculate_single_qubit_expectation(qubit, &pauli_matrix)?;
                let variance = 1.0 - expectation * expectation; // For Pauli operators
                variances.push(variance);
            }
        }

        Ok(Array1::from_vec(variances))
    }

    /// Measure higher-order moments
    fn measure_higher_order_moments(&self) -> Result<Array1<f64>> {
        let mut moments = Vec::new();

        for qubit in 0..self.config.num_qubits {
            // Calculate moments up to 3rd order for Z observable
            let z_exp = self.calculate_single_qubit_expectation(
                qubit,
                &[
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )?;

            // First moment (mean)
            moments.push(z_exp);

            // Second central moment (variance)
            let variance = 1.0 - z_exp * z_exp;
            moments.push(variance);

            // Third central moment (skewness measure)
            // For Pauli-Z, this is typically zero due to symmetry
            moments.push(0.0);

            // Kurtosis measure
            moments.push(variance * variance);

            // Fifth moment (for more complex characterization)
            moments.push(z_exp * variance);

            // Sixth moment
            moments.push(variance * variance * variance);
        }

        Ok(Array1::from_vec(moments))
    }

    /// Measure quantum coherence
    fn measure_quantum_coherence(&self) -> Result<Array1<f64>> {
        let mut coherence_measures = Vec::new();

        for qubit in 0..self.config.num_qubits {
            // L1 norm of coherence (off-diagonal elements in computational basis)
            let mut coherence = 0.0;
            let state = &self.reservoir_state.state_vector;

            for i in 0..state.len() {
                for j in 0..state.len() {
                    if i != j {
                        let i_bit = (i >> qubit) & 1;
                        let j_bit = (j >> qubit) & 1;
                        if i_bit != j_bit {
                            coherence += (state[i].conj() * state[j]).norm();
                        }
                    }
                }
            }

            coherence_measures.push(coherence);
        }

        Ok(Array1::from_vec(coherence_measures))
    }

    /// Measure purity
    fn measure_purity(&self) -> Result<Array1<f64>> {
        // Purity = Tr(ρ²) for the full state
        let state = &self.reservoir_state.state_vector;
        let purity = state.iter().map(|x| x.norm_sqr().powi(2)).sum::<f64>();

        Ok(Array1::from_vec(vec![purity]))
    }

    /// Measure temporal correlations
    fn measure_temporal_correlations(&self) -> Result<Array1<f64>> {
        let mut correlations = Vec::new();

        // Calculate autocorrelation with past states
        let current_state = &self.reservoir_state.state_vector;

        for past_state in &self.reservoir_state.state_history {
            let correlation = current_state
                .iter()
                .zip(past_state.iter())
                .map(|(a, b)| (a.conj() * b).re)
                .sum::<f64>();
            correlations.push(correlation);
        }

        // Pad with zeros if not enough history
        while correlations.len() < self.config.memory_capacity {
            correlations.push(0.0);
        }

        Ok(Array1::from_vec(correlations))
    }

    /// Train the enhanced reservoir computer
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

        // Train output weights using the specified learning algorithm
        self.train_with_learning_algorithm(&all_features, &all_targets)?;

        // Analyze memory capacity if enabled
        if self.config.memory_config.enable_capacity_estimation {
            self.analyze_memory_capacity(&all_features)?;
        }

        // Evaluate performance
        let (training_error, test_error) =
            self.evaluate_performance(&all_features, &all_targets)?;

        let training_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Update metrics
        self.metrics.training_examples += all_features.len();
        self.metrics.generalization_error = test_error;
        self.metrics.memory_capacity = self.reservoir_state.memory_metrics.total_capacity;

        Ok(TrainingResult {
            training_error,
            test_error,
            training_time_ms: training_time,
            num_examples: all_features.len(),
            echo_state_property: self.estimate_echo_state_property()?,
            memory_capacity: self.reservoir_state.memory_metrics.total_capacity,
            nonlinear_capacity: self.reservoir_state.memory_metrics.nonlinear_capacity,
            processing_capacity: self.reservoir_state.memory_metrics.processing_capacity,
        })
    }

    /// Train using advanced learning algorithms
    fn train_with_learning_algorithm(
        &mut self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<()> {
        match self.config.learning_config.algorithm {
            LearningAlgorithm::Ridge => {
                self.train_ridge_regression(features, targets)?;
            }
            LearningAlgorithm::LASSO => {
                self.train_lasso_regression(features, targets)?;
            }
            LearningAlgorithm::ElasticNet => {
                self.train_elastic_net(features, targets)?;
            }
            LearningAlgorithm::RecursiveLeastSquares => {
                self.train_recursive_least_squares(features, targets)?;
            }
            LearningAlgorithm::KalmanFilter => {
                self.train_kalman_filter(features, targets)?;
            }
            _ => {
                // Default to ridge regression
                self.train_ridge_regression(features, targets)?;
            }
        }

        Ok(())
    }

    /// Train ridge regression
    fn train_ridge_regression(
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
        let lambda = self.config.learning_config.regularization;

        // X^T X
        let xtx = feature_matrix.t().dot(&feature_matrix);

        // Add regularization
        let mut xtx_reg = xtx;
        for i in 0..xtx_reg.nrows().min(xtx_reg.ncols()) {
            xtx_reg[[i, i]] += lambda;
        }

        // X^T Y
        let xty = feature_matrix.t().dot(&target_matrix);

        // Solve using simplified approach (in practice would use proper linear solver)
        self.solve_linear_system(&xtx_reg, &xty)?;

        Ok(())
    }

    /// Train LASSO regression (simplified)
    fn train_lasso_regression(
        &mut self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<()> {
        // Simplified LASSO using coordinate descent
        let lambda = self.config.learning_config.regularization;
        let max_iter = 100;

        for _ in 0..max_iter {
            // Coordinate descent updates
            for j in 0..self.output_weights.ncols().min(features[0].len()) {
                for i in 0..self.output_weights.nrows().min(targets[0].len()) {
                    // Soft thresholding update
                    let old_weight = self.output_weights[[i, j]];
                    let gradient = self.compute_lasso_gradient(features, targets, i, j)?;
                    let update = 0.01f64.mul_add(-gradient, old_weight);

                    // Soft thresholding
                    self.output_weights[[i, j]] = if update > lambda {
                        update - lambda
                    } else if update < -lambda {
                        update + lambda
                    } else {
                        0.0
                    };
                }
            }
        }

        Ok(())
    }

    /// Compute LASSO gradient (simplified)
    fn compute_lasso_gradient(
        &self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
        output_idx: usize,
        feature_idx: usize,
    ) -> Result<f64> {
        let mut gradient = 0.0;

        for (feature_vec, target_vec) in features.iter().zip(targets.iter()) {
            if feature_idx < feature_vec.len() && output_idx < target_vec.len() {
                let prediction = self.predict_single_output(feature_vec, output_idx)?;
                let error = prediction - target_vec[output_idx];
                gradient += error * feature_vec[feature_idx];
            }
        }

        gradient /= features.len() as f64;
        Ok(gradient)
    }

    /// Train Elastic Net regression
    fn train_elastic_net(
        &mut self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<()> {
        let l1_ratio = self.config.learning_config.l1_ratio;

        // Combine Ridge and LASSO with L1 ratio
        if l1_ratio > 0.5 {
            // More L1 regularization
            self.train_lasso_regression(features, targets)?;
        } else {
            // More L2 regularization
            self.train_ridge_regression(features, targets)?;
        }

        Ok(())
    }

    /// Train Recursive Least Squares
    fn train_recursive_least_squares(
        &mut self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<()> {
        let forgetting_factor = self.config.learning_config.forgetting_factor;
        let n_features = features[0].len().min(self.output_weights.ncols());
        let n_outputs = targets[0].len().min(self.output_weights.nrows());

        // Initialize covariance matrix
        let mut p_matrix = Array2::eye(n_features) * 1000.0; // Large initial covariance

        // Online RLS updates
        for (feature_vec, target_vec) in features.iter().zip(targets.iter()) {
            let x = feature_vec.slice(s![..n_features]).to_owned();
            let y = target_vec.slice(s![..n_outputs]).to_owned();

            // Update covariance matrix
            let px = p_matrix.dot(&x);
            let denominator = forgetting_factor + x.dot(&px);

            if denominator > 1e-15 {
                let k = &px / denominator;

                // Update weights for each output
                for output_idx in 0..n_outputs {
                    let prediction = self.predict_single_output(feature_vec, output_idx)?;
                    let error = y[output_idx] - prediction;

                    // RLS weight update
                    for feature_idx in 0..n_features {
                        self.output_weights[[output_idx, feature_idx]] += k[feature_idx] * error;
                    }
                }

                // Update covariance matrix
                let outer_product = k
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&x.view().insert_axis(Axis(0)));
                p_matrix = (p_matrix - outer_product) / forgetting_factor;
            }
        }

        Ok(())
    }

    /// Train Kalman filter
    fn train_kalman_filter(
        &mut self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<()> {
        let process_noise = self.config.learning_config.process_noise;
        let measurement_noise = self.config.learning_config.measurement_noise;

        let n_features = features[0].len().min(self.output_weights.ncols());
        let n_outputs = targets[0].len().min(self.output_weights.nrows());

        // Initialize Kalman filter matrices
        let mut state_covariance = Array2::eye(n_features) * 1.0;
        let process_noise_matrix: Array2<f64> = Array2::eye(n_features) * process_noise;
        let measurement_noise_scalar = measurement_noise;

        // Kalman filter updates
        for (feature_vec, target_vec) in features.iter().zip(targets.iter()) {
            let x = feature_vec.slice(s![..n_features]).to_owned();
            let y = target_vec.slice(s![..n_outputs]).to_owned();

            // Prediction step
            let predicted_covariance = &state_covariance + &process_noise_matrix;

            // Update step for each output
            for output_idx in 0..n_outputs {
                let measurement = y[output_idx];
                let prediction = self.predict_single_output(feature_vec, output_idx)?;

                // Kalman gain
                let s = x.dot(&predicted_covariance.dot(&x)) + measurement_noise_scalar;
                if s > 1e-15 {
                    let k = predicted_covariance.dot(&x) / s;

                    // Update weights
                    let innovation = measurement - prediction;
                    for feature_idx in 0..n_features {
                        self.output_weights[[output_idx, feature_idx]] +=
                            k[feature_idx] * innovation;
                    }

                    // Update covariance
                    let kh = k
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&x.view().insert_axis(Axis(0)));
                    state_covariance = &predicted_covariance - &kh.dot(&predicted_covariance);
                }
            }
        }

        Ok(())
    }

    /// Predict single output value
    fn predict_single_output(&self, features: &Array1<f64>, output_idx: usize) -> Result<f64> {
        let feature_size = features.len().min(self.output_weights.ncols());
        let mut output = 0.0;

        for j in 0..feature_size {
            output += self.output_weights[[output_idx, j]] * features[j];
        }

        Ok(output)
    }

    /// Analyze memory capacity
    fn analyze_memory_capacity(&mut self, features: &[Array1<f64>]) -> Result<()> {
        // Linear memory capacity
        let linear_capacity = self.estimate_linear_memory_capacity(features)?;
        self.reservoir_state.memory_metrics.linear_capacity = linear_capacity;

        // Nonlinear memory capacity
        if self.config.memory_config.enable_nonlinear {
            let nonlinear_capacity = self.estimate_nonlinear_memory_capacity(features)?;
            self.reservoir_state.memory_metrics.nonlinear_capacity = nonlinear_capacity;
        }

        // Total capacity
        self.reservoir_state.memory_metrics.total_capacity =
            self.reservoir_state.memory_metrics.linear_capacity
                + self.reservoir_state.memory_metrics.nonlinear_capacity;

        // Information processing capacity
        if self.config.memory_config.enable_ipc {
            let ipc = self.estimate_information_processing_capacity(features)?;
            self.reservoir_state.memory_metrics.processing_capacity = ipc;
        }

        // Update memory analyzer
        self.memory_analyzer.capacity_estimates.insert(
            "linear".to_string(),
            self.reservoir_state.memory_metrics.linear_capacity,
        );
        self.memory_analyzer.capacity_estimates.insert(
            "nonlinear".to_string(),
            self.reservoir_state.memory_metrics.nonlinear_capacity,
        );
        self.memory_analyzer.capacity_estimates.insert(
            "total".to_string(),
            self.reservoir_state.memory_metrics.total_capacity,
        );

        Ok(())
    }

    /// Estimate linear memory capacity
    fn estimate_linear_memory_capacity(&self, features: &[Array1<f64>]) -> Result<f64> {
        // Use correlation analysis to estimate linear memory
        let mut capacity = 0.0;

        for lag in 1..=20 {
            if lag < features.len() {
                let mut correlation = 0.0;
                let mut count = 0;

                for i in lag..features.len() {
                    for j in 0..features[i].len().min(features[i - lag].len()) {
                        correlation += features[i][j] * features[i - lag][j];
                        count += 1;
                    }
                }

                if count > 0 {
                    correlation /= f64::from(count);
                    capacity += correlation.abs();
                }
            }
        }

        Ok(capacity)
    }

    /// Estimate nonlinear memory capacity
    fn estimate_nonlinear_memory_capacity(&self, features: &[Array1<f64>]) -> Result<f64> {
        let mut nonlinear_capacity = 0.0;

        // Test various nonlinear functions
        for order in &self.config.memory_config.nonlinearity_orders {
            let capacity_order = self.test_nonlinear_order(*order, features)?;
            nonlinear_capacity += capacity_order;
        }

        Ok(nonlinear_capacity)
    }

    /// Test specific nonlinear order
    fn test_nonlinear_order(&self, order: usize, features: &[Array1<f64>]) -> Result<f64> {
        let mut capacity = 0.0;

        // Generate nonlinear target function
        for lag in 1..=10 {
            if lag < features.len() {
                let mut correlation = 0.0;
                let mut count = 0;

                for i in lag..features.len() {
                    for j in 0..features[i].len().min(features[i - lag].len()) {
                        // Nonlinear transformation
                        let current = features[i][j];
                        let past = features[i - lag][j];
                        let nonlinear_target = past.powi(order as i32);

                        correlation += current * nonlinear_target;
                        count += 1;
                    }
                }

                if count > 0 {
                    correlation /= f64::from(count);
                    capacity += correlation.abs() / order as f64; // Normalize by order
                }
            }
        }

        Ok(capacity)
    }

    /// Estimate information processing capacity
    fn estimate_information_processing_capacity(&self, features: &[Array1<f64>]) -> Result<f64> {
        let mut ipc = 0.0;

        for ipc_function in &self.config.memory_config.ipc_functions {
            let capacity_func = self.test_ipc_function(*ipc_function, features)?;
            ipc += capacity_func;
        }

        Ok(ipc)
    }

    /// Test specific IPC function
    fn test_ipc_function(&self, function: IPCFunction, features: &[Array1<f64>]) -> Result<f64> {
        let mut capacity = 0.0;

        for lag in 1..=10 {
            if lag < features.len() {
                let mut correlation = 0.0;
                let mut count = 0;

                for i in lag..features.len() {
                    for j in 0..features[i].len().min(features[i - lag].len()) {
                        let current = features[i][j];
                        let past = features[i - lag][j];

                        let target = match function {
                            IPCFunction::Linear => past,
                            IPCFunction::Quadratic => past * past,
                            IPCFunction::Cubic => past * past * past,
                            IPCFunction::Sine => past.sin(),
                            IPCFunction::Product => {
                                if j > 0 && j - 1 < features[i - lag].len() {
                                    past * features[i - lag][j - 1]
                                } else {
                                    past
                                }
                            }
                            IPCFunction::XOR => {
                                if past > 0.0 {
                                    1.0
                                } else {
                                    -1.0
                                }
                            }
                        };

                        correlation += current * target;
                        count += 1;
                    }
                }

                if count > 0 {
                    correlation /= f64::from(count);
                    capacity += correlation.abs();
                }
            }
        }

        Ok(capacity)
    }

    /// Solve linear system (simplified implementation)
    fn solve_linear_system(&mut self, a: &Array2<f64>, b: &Array2<f64>) -> Result<()> {
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
        let coupling = self.config.coupling_strength;
        let estimated_spectral_radius = coupling.tanh(); // Heuristic estimate

        // Echo state property requires spectral radius < 1
        Ok(if estimated_spectral_radius < 1.0 {
            1.0
        } else {
            1.0 / estimated_spectral_radius
        })
    }

    /// Update processing time metrics
    fn update_processing_time(&mut self, time_ms: f64) {
        let count = self.metrics.training_examples as f64;
        self.metrics.avg_processing_time_ms =
            self.metrics.avg_processing_time_ms.mul_add(count, time_ms) / (count + 1.0);
    }

    /// Get current metrics
    pub const fn get_metrics(&self) -> &ReservoirMetrics {
        &self.metrics
    }

    /// Get memory analysis results
    pub const fn get_memory_analysis(&self) -> &MemoryAnalyzer {
        &self.memory_analyzer
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
