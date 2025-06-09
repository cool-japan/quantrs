//! Device-specific model compilation for quantum machine learning
//!
//! This module provides device-aware compilation of quantum ML models,
//! optimizing circuits for specific hardware characteristics and constraints.

use crate::error::{MLError, Result};
use crate::circuit_integration::{DeviceTopology, QubitProperties};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet, VecDeque};

/// Device-specific model compiler
pub struct DeviceCompiler {
    /// Target device topology
    topology: DeviceTopology,
    /// Compilation options
    options: CompilationOptions,
    /// Device characterization data
    characterization: DeviceCharacterization,
}

/// Compilation options
#[derive(Debug, Clone)]
pub struct CompilationOptions {
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Maximum compilation time (seconds)
    pub max_compilation_time: f64,
    /// Target gate error threshold
    pub error_threshold: f64,
    /// Enable noise-aware compilation
    pub noise_aware: bool,
    /// Enable crosstalk mitigation
    pub crosstalk_mitigation: bool,
    /// Routing algorithm
    pub routing_algorithm: RoutingAlgorithm,
    /// Gate synthesis method
    pub synthesis_method: SynthesisMethod,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            max_compilation_time: 60.0,
            error_threshold: 0.01,
            noise_aware: true,
            crosstalk_mitigation: true,
            routing_algorithm: RoutingAlgorithm::SABRE,
            synthesis_method: SynthesisMethod::Solovay_Kitaev,
        }
    }
}

/// Routing algorithms
#[derive(Debug, Clone, Copy)]
pub enum RoutingAlgorithm {
    /// SABRE routing algorithm
    SABRE,
    /// Lookahead routing
    Lookahead,
    /// Token swapping
    TokenSwapping,
    /// Heuristic routing
    Heuristic,
}

/// Gate synthesis methods
#[derive(Debug, Clone, Copy)]
pub enum SynthesisMethod {
    /// Solovay-Kitaev decomposition
    Solovay_Kitaev,
    /// Shannon decomposition
    Shannon,
    /// KAK decomposition for two-qubit gates
    KAK,
    /// Cartan decomposition
    Cartan,
}

/// Device characterization data
#[derive(Debug, Clone)]
pub struct DeviceCharacterization {
    /// Gate error rates
    pub gate_errors: HashMap<String, f64>,
    /// Two-qubit gate errors (by qubit pair)
    pub two_qubit_errors: HashMap<(usize, usize), f64>,
    /// Readout errors
    pub readout_errors: Array1<f64>,
    /// Crosstalk matrix
    pub crosstalk_matrix: Array2<f64>,
    /// Gate times
    pub gate_times: HashMap<String, f64>,
    /// Calibration timestamp
    pub calibration_time: std::time::SystemTime,
}

impl DeviceCharacterization {
    /// Create default characterization
    pub fn default_for_device(num_qubits: usize) -> Self {
        let mut gate_errors = HashMap::new();
        gate_errors.insert("X".to_string(), 0.001);
        gate_errors.insert("Y".to_string(), 0.001);
        gate_errors.insert("Z".to_string(), 0.0001);
        gate_errors.insert("H".to_string(), 0.002);
        gate_errors.insert("CNOT".to_string(), 0.01);
        
        let mut gate_times = HashMap::new();
        gate_times.insert("X".to_string(), 0.02); // 20 ns
        gate_times.insert("Y".to_string(), 0.02);
        gate_times.insert("Z".to_string(), 0.0);  // Virtual Z
        gate_times.insert("H".to_string(), 0.02);
        gate_times.insert("CNOT".to_string(), 0.2); // 200 ns
        
        Self {
            gate_errors,
            two_qubit_errors: HashMap::new(),
            readout_errors: Array1::from_elem(num_qubits, 0.02),
            crosstalk_matrix: Array2::zeros((num_qubits, num_qubits)),
            gate_times,
            calibration_time: std::time::SystemTime::now(),
        }
    }
    
    /// Update gate error for specific qubits
    pub fn set_gate_error(&mut self, gate: &str, qubits: &[usize], error: f64) {
        if qubits.len() == 2 {
            self.two_qubit_errors.insert((qubits[0], qubits[1]), error);
        } else {
            self.gate_errors.insert(gate.to_string(), error);
        }
    }
    
    /// Get expected error for a gate operation
    pub fn get_gate_error(&self, gate: &str, qubits: &[usize]) -> f64 {
        if qubits.len() == 2 {
            self.two_qubit_errors
                .get(&(qubits[0], qubits[1]))
                .or_else(|| self.two_qubit_errors.get(&(qubits[1], qubits[0])))
                .copied()
                .unwrap_or_else(|| self.gate_errors.get(gate).copied().unwrap_or(0.01))
        } else {
            self.gate_errors.get(gate).copied().unwrap_or(0.001)
        }
    }
}

impl DeviceCompiler {
    /// Create a new device compiler
    pub fn new(topology: DeviceTopology) -> Self {
        let num_qubits = topology.num_qubits();
        Self {
            topology,
            options: CompilationOptions::default(),
            characterization: DeviceCharacterization::default_for_device(num_qubits),
        }
    }
    
    /// Set compilation options
    pub fn with_options(mut self, options: CompilationOptions) -> Self {
        self.options = options;
        self
    }
    
    /// Set device characterization
    pub fn with_characterization(mut self, characterization: DeviceCharacterization) -> Self {
        self.characterization = characterization;
        self
    }
    
    /// Compile quantum ML model for target device
    pub fn compile_model(&self, model: &QuantumMLModel) -> Result<CompiledModel> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Convert model to circuit representation
        let mut circuit = self.model_to_circuit(model)?;
        
        // Step 2: Initial circuit optimization
        circuit = self.initial_optimization(&circuit)?;
        
        // Step 3: Qubit mapping and routing
        let (circuit, qubit_mapping) = self.route_circuit(&circuit)?;
        
        // Step 4: Gate synthesis for native gate set
        circuit = self.synthesize_gates(&circuit)?;
        
        // Step 5: Noise-aware optimization
        if self.options.noise_aware {
            circuit = self.noise_aware_optimization(&circuit)?;
        }
        
        // Step 6: Crosstalk mitigation
        if self.options.crosstalk_mitigation {
            circuit = self.mitigate_crosstalk(&circuit)?;
        }
        
        // Step 7: Final optimization
        circuit = self.final_optimization(&circuit)?;
        
        let compilation_time = start_time.elapsed().as_secs_f64();
        
        // Generate compilation metrics
        let metrics = self.analyze_compiled_circuit(&circuit, compilation_time)?;
        
        Ok(CompiledModel {
            circuit,
            qubit_mapping,
            metrics,
            target_device: self.topology.clone(),
            characterization: self.characterization.clone(),
        })
    }
    
    /// Convert ML model to quantum circuit
    fn model_to_circuit(&self, model: &QuantumMLModel) -> Result<Circuit> {
        let mut builder = CircuitBuilder::new();
        
        // Add model layers to circuit
        for layer in &model.layers {
            match layer {
                ModelLayer::Encoding(encoding_layer) => {
                    self.add_encoding_layer(&mut builder, encoding_layer)?;
                }
                ModelLayer::Variational(var_layer) => {
                    self.add_variational_layer(&mut builder, var_layer)?;
                }
                ModelLayer::Measurement(meas_layer) => {
                    self.add_measurement_layer(&mut builder, meas_layer)?;
                }
            }
        }
        
        builder.build()
    }
    
    /// Add encoding layer to circuit
    fn add_encoding_layer(&self, builder: &mut CircuitBuilder, layer: &EncodingLayer) -> Result<()> {
        match &layer.encoding_type {
            EncodingType::Amplitude => {
                // Add amplitude encoding gates
                for qubit in &layer.qubits {
                    builder.add_ry(*qubit, 0.0)?; // Placeholder parameter
                }
            }
            EncodingType::Angle => {
                // Add angle encoding gates
                for qubit in &layer.qubits {
                    builder.add_rz(*qubit, 0.0)?; // Placeholder parameter
                }
            }
            EncodingType::Basis => {
                // Add basis encoding (no gates needed)
            }
        }
        Ok(())
    }
    
    /// Add variational layer to circuit
    fn add_variational_layer(&self, builder: &mut CircuitBuilder, layer: &VariationalLayer) -> Result<()> {
        match &layer.ansatz_type {
            AnsatzType::Hardware_Efficient => {
                // Add hardware-efficient ansatz
                for qubit in &layer.qubits {
                    builder.add_ry(*qubit, 0.0)?;
                    builder.add_rz(*qubit, 0.0)?;
                }
                
                // Add entangling gates
                for i in 0..layer.qubits.len() - 1 {
                    builder.add_cnot(layer.qubits[i], layer.qubits[i + 1])?;
                }
            }
            AnsatzType::QAOA => {
                // Add QAOA ansatz
                for qubit in &layer.qubits {
                    builder.add_rx(*qubit, 0.0)?; // Mixer
                }
                
                // Problem-specific gates would be added here
            }
            AnsatzType::Custom(gates) => {
                // Add custom gate sequence
                for gate in gates {
                    self.add_custom_gate(builder, gate)?;
                }
            }
        }
        Ok(())
    }
    
    /// Add measurement layer to circuit
    fn add_measurement_layer(&self, builder: &mut CircuitBuilder, layer: &MeasurementLayer) -> Result<()> {
        for qubit in &layer.qubits {
            builder.add_measurement(*qubit)?;
        }
        Ok(())
    }
    
    /// Add custom gate to circuit
    fn add_custom_gate(&self, builder: &mut CircuitBuilder, gate: &CustomGate) -> Result<()> {
        match gate {
            CustomGate::SingleQubit { qubit, gate_type, parameter } => {
                match gate_type.as_str() {
                    "RX" => builder.add_rx(*qubit, *parameter)?,
                    "RY" => builder.add_ry(*qubit, *parameter)?,
                    "RZ" => builder.add_rz(*qubit, *parameter)?,
                    "H" => builder.add_h(*qubit)?,
                    _ => return Err(MLError::InvalidConfiguration(
                        format!("Unknown gate type: {}", gate_type),
                    )),
                }
            }
            CustomGate::TwoQubit { control, target, gate_type, parameter } => {
                match gate_type.as_str() {
                    "CNOT" => builder.add_cnot(*control, *target)?,
                    "CZ" => builder.add_cz(*control, *target)?,
                    "RZZ" => builder.add_rzz(*control, *target, *parameter)?,
                    _ => return Err(MLError::InvalidConfiguration(
                        format!("Unknown two-qubit gate type: {}", gate_type),
                    )),
                }
            }
        }
        Ok(())
    }
    
    /// Initial circuit optimization
    fn initial_optimization(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut optimized = circuit.clone();
        
        if self.options.optimization_level >= 1 {
            // Remove redundant gates
            optimized = self.remove_redundant_gates(&optimized)?;
        }
        
        if self.options.optimization_level >= 2 {
            // Merge rotation gates
            optimized = self.merge_rotations(&optimized)?;
        }
        
        if self.options.optimization_level >= 3 {
            // Advanced optimizations
            optimized = self.commutation_optimization(&optimized)?;
        }
        
        Ok(optimized)
    }
    
    /// Route circuit to device topology
    fn route_circuit(&self, circuit: &Circuit) -> Result<(Circuit, QubitMapping)> {
        match self.options.routing_algorithm {
            RoutingAlgorithm::SABRE => self.sabre_routing(circuit),
            RoutingAlgorithm::Lookahead => self.lookahead_routing(circuit),
            RoutingAlgorithm::TokenSwapping => self.token_swapping_routing(circuit),
            RoutingAlgorithm::Heuristic => self.heuristic_routing(circuit),
        }
    }
    
    /// SABRE routing algorithm
    fn sabre_routing(&self, circuit: &Circuit) -> Result<(Circuit, QubitMapping)> {
        let mut routed_circuit = CircuitBuilder::new();
        let mut mapping = QubitMapping::identity(circuit.num_qubits());
        
        // SABRE algorithm implementation (simplified)
        for gate in circuit.gates() {
            if gate.num_qubits() == 2 {
                let (q1, q2) = (gate.qubits()[0], gate.qubits()[1]);
                if !self.topology.are_connected(mapping.logical_to_physical(q1), mapping.logical_to_physical(q2)) {
                    // Insert SWAP gates to make qubits adjacent
                    let swaps = self.find_swap_path(
                        mapping.logical_to_physical(q1),
                        mapping.logical_to_physical(q2),
                    )?;
                    
                    for (qa, qb) in swaps {
                        routed_circuit.add_swap(qa, qb)?;
                        mapping.apply_swap(qa, qb);
                    }
                }
            }
            
            // Add the original gate with mapped qubits
            self.add_mapped_gate(&mut routed_circuit, &gate, &mapping)?;
        }
        
        Ok((routed_circuit.build()?, mapping))
    }
    
    /// Find shortest path of SWAPs between qubits
    fn find_swap_path(&self, start: usize, end: usize) -> Result<Vec<(usize, usize)>> {
        // Simplified shortest path algorithm
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(current) = queue.pop_front() {
            if current == end {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = end;
                while let Some(&prev) = parent.get(&node) {
                    path.push((prev, node));
                    node = prev;
                }
                path.reverse();
                return Ok(path);
            }
            
            for neighbor in self.topology.neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
        
        Err(MLError::InvalidConfiguration(
            "No path found between qubits".to_string(),
        ))
    }
    
    /// Other routing algorithms (simplified implementations)
    fn lookahead_routing(&self, circuit: &Circuit) -> Result<(Circuit, QubitMapping)> {
        // Placeholder - would implement lookahead routing
        self.sabre_routing(circuit)
    }
    
    fn token_swapping_routing(&self, circuit: &Circuit) -> Result<(Circuit, QubitMapping)> {
        // Placeholder - would implement token swapping
        self.sabre_routing(circuit)
    }
    
    fn heuristic_routing(&self, circuit: &Circuit) -> Result<(Circuit, QubitMapping)> {
        // Placeholder - would implement heuristic routing
        self.sabre_routing(circuit)
    }
    
    /// Add gate with mapped qubits
    fn add_mapped_gate(&self, builder: &mut CircuitBuilder, gate: &dyn GateOp, mapping: &QubitMapping) -> Result<()> {
        let mapped_qubits: Vec<usize> = gate.qubits()
            .iter()
            .map(|&q| mapping.logical_to_physical(q))
            .collect();
        
        match gate.gate_type().as_str() {
            "H" => builder.add_h(mapped_qubits[0])?,
            "X" => builder.add_x(mapped_qubits[0])?,
            "Y" => builder.add_y(mapped_qubits[0])?,
            "Z" => builder.add_z(mapped_qubits[0])?,
            "RX" => builder.add_rx(mapped_qubits[0], gate.parameters()[0])?,
            "RY" => builder.add_ry(mapped_qubits[0], gate.parameters()[0])?,
            "RZ" => builder.add_rz(mapped_qubits[0], gate.parameters()[0])?,
            "CNOT" => builder.add_cnot(mapped_qubits[0], mapped_qubits[1])?,
            "CZ" => builder.add_cz(mapped_qubits[0], mapped_qubits[1])?,
            "SWAP" => builder.add_swap(mapped_qubits[0], mapped_qubits[1])?,
            _ => return Err(MLError::InvalidConfiguration(
                format!("Unknown gate type: {}", gate.gate_type()),
            )),
        }
        
        Ok(())
    }
    
    /// Gate synthesis for native gate set
    fn synthesize_gates(&self, circuit: &Circuit) -> Result<Circuit> {
        match self.options.synthesis_method {
            SynthesisMethod::Solovay_Kitaev => self.solovay_kitaev_synthesis(circuit),
            SynthesisMethod::Shannon => self.shannon_synthesis(circuit),
            SynthesisMethod::KAK => self.kak_synthesis(circuit),
            SynthesisMethod::Cartan => self.cartan_synthesis(circuit),
        }
    }
    
    /// Solovay-Kitaev synthesis
    fn solovay_kitaev_synthesis(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would implement Solovay-Kitaev decomposition
        Ok(circuit.clone())
    }
    
    /// Shannon synthesis
    fn shannon_synthesis(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would implement Shannon decomposition
        Ok(circuit.clone())
    }
    
    /// KAK synthesis
    fn kak_synthesis(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would implement KAK decomposition
        Ok(circuit.clone())
    }
    
    /// Cartan synthesis
    fn cartan_synthesis(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would implement Cartan decomposition
        Ok(circuit.clone())
    }
    
    /// Noise-aware optimization
    fn noise_aware_optimization(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut optimized = circuit.clone();
        
        // Reschedule gates to minimize decoherence
        optimized = self.schedule_for_coherence(&optimized)?;
        
        // Choose error-optimal gates
        optimized = self.select_low_error_gates(&optimized)?;
        
        Ok(optimized)
    }
    
    /// Schedule gates to minimize decoherence
    fn schedule_for_coherence(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would implement coherence-aware scheduling
        Ok(circuit.clone())
    }
    
    /// Select gates with lowest error rates
    fn select_low_error_gates(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would select optimal gates based on characterization
        Ok(circuit.clone())
    }
    
    /// Mitigate crosstalk effects
    fn mitigate_crosstalk(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would implement crosstalk mitigation
        Ok(circuit.clone())
    }
    
    /// Final optimization pass
    fn final_optimization(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut optimized = circuit.clone();
        
        // Final gate merging
        optimized = self.merge_rotations(&optimized)?;
        
        // Remove identity gates
        optimized = self.remove_identity_gates(&optimized)?;
        
        Ok(optimized)
    }
    
    /// Remove redundant gates
    fn remove_redundant_gates(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would implement redundant gate removal
        Ok(circuit.clone())
    }
    
    /// Merge rotation gates
    fn merge_rotations(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would merge consecutive rotation gates
        Ok(circuit.clone())
    }
    
    /// Commutation-based optimization
    fn commutation_optimization(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would implement commutation rules
        Ok(circuit.clone())
    }
    
    /// Remove identity gates
    fn remove_identity_gates(&self, circuit: &Circuit) -> Result<Circuit> {
        // Placeholder - would remove gates with zero rotation angles
        Ok(circuit.clone())
    }
    
    /// Analyze compiled circuit
    fn analyze_compiled_circuit(&self, circuit: &Circuit, compilation_time: f64) -> Result<CompilationMetrics> {
        let gate_count = circuit.num_gates();
        let depth = circuit.depth();
        let two_qubit_gate_count = circuit.gates()
            .iter()
            .filter(|g| g.num_qubits() == 2)
            .count();
        
        // Estimate total error
        let mut total_error = 0.0;
        for gate in circuit.gates() {
            total_error += self.characterization.get_gate_error(
                gate.gate_type(),
                &gate.qubits(),
            );
        }
        
        // Estimate execution time
        let mut execution_time = 0.0;
        for gate in circuit.gates() {
            execution_time += self.characterization.gate_times
                .get(gate.gate_type())
                .copied()
                .unwrap_or(0.1);
        }
        
        Ok(CompilationMetrics {
            gate_count,
            depth,
            two_qubit_gate_count,
            total_error,
            execution_time,
            compilation_time,
            swap_count: 0, // Would count actual SWAPs
        })
    }
}

/// Quantum ML model representation
#[derive(Debug, Clone)]
pub struct QuantumMLModel {
    /// Model layers
    pub layers: Vec<ModelLayer>,
    /// Total number of qubits
    pub num_qubits: usize,
    /// Parameter count
    pub num_parameters: usize,
}

/// Model layer types
#[derive(Debug, Clone)]
pub enum ModelLayer {
    /// Data encoding layer
    Encoding(EncodingLayer),
    /// Variational layer
    Variational(VariationalLayer),
    /// Measurement layer
    Measurement(MeasurementLayer),
}

/// Data encoding layer
#[derive(Debug, Clone)]
pub struct EncodingLayer {
    /// Qubits used for encoding
    pub qubits: Vec<usize>,
    /// Encoding method
    pub encoding_type: EncodingType,
}

/// Data encoding types
#[derive(Debug, Clone)]
pub enum EncodingType {
    /// Amplitude encoding
    Amplitude,
    /// Angle encoding
    Angle,
    /// Basis encoding
    Basis,
}

/// Variational layer
#[derive(Debug, Clone)]
pub struct VariationalLayer {
    /// Qubits in the layer
    pub qubits: Vec<usize>,
    /// Ansatz type
    pub ansatz_type: AnsatzType,
    /// Number of repetitions
    pub repetitions: usize,
}

/// Ansatz types
#[derive(Debug, Clone)]
pub enum AnsatzType {
    /// Hardware-efficient ansatz
    Hardware_Efficient,
    /// QAOA ansatz
    QAOA,
    /// Custom gate sequence
    Custom(Vec<CustomGate>),
}

/// Custom gate definition
#[derive(Debug, Clone)]
pub enum CustomGate {
    /// Single-qubit gate
    SingleQubit {
        qubit: usize,
        gate_type: String,
        parameter: f64,
    },
    /// Two-qubit gate
    TwoQubit {
        control: usize,
        target: usize,
        gate_type: String,
        parameter: f64,
    },
}

/// Measurement layer
#[derive(Debug, Clone)]
pub struct MeasurementLayer {
    /// Qubits to measure
    pub qubits: Vec<usize>,
    /// Measurement basis
    pub basis: MeasurementBasis,
}

/// Measurement basis
#[derive(Debug, Clone)]
pub enum MeasurementBasis {
    /// Computational basis (Z)
    Computational,
    /// X basis
    X,
    /// Y basis
    Y,
    /// Custom Pauli string
    Pauli(String),
}

/// Compiled model
#[derive(Debug, Clone)]
pub struct CompiledModel<const N: usize> {
    /// Compiled circuit
    pub circuit: Circuit<N>,
    /// Qubit mapping
    pub qubit_mapping: QubitMapping,
    /// Compilation metrics
    pub metrics: CompilationMetrics,
    /// Target device
    pub target_device: DeviceTopology,
    /// Device characterization
    pub characterization: DeviceCharacterization,
}

/// Qubit mapping between logical and physical qubits
#[derive(Debug, Clone)]
pub struct QubitMapping {
    /// Logical to physical mapping
    logical_to_physical: Vec<usize>,
    /// Physical to logical mapping
    physical_to_logical: Vec<Option<usize>>,
}

impl QubitMapping {
    /// Create identity mapping
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            logical_to_physical: (0..num_qubits).collect(),
            physical_to_logical: (0..num_qubits).map(Some).collect(),
        }
    }
    
    /// Get physical qubit for logical qubit
    pub fn logical_to_physical(&self, logical: usize) -> usize {
        self.logical_to_physical[logical]
    }
    
    /// Get logical qubit for physical qubit
    pub fn physical_to_logical(&self, physical: usize) -> Option<usize> {
        self.physical_to_logical.get(physical).copied().flatten()
    }
    
    /// Apply SWAP operation to mapping
    pub fn apply_swap(&mut self, q1: usize, q2: usize) {
        // Update logical to physical mapping
        for logical in &mut self.logical_to_physical {
            if *logical == q1 {
                *logical = q2;
            } else if *logical == q2 {
                *logical = q1;
            }
        }
        
        // Update physical to logical mapping
        self.physical_to_logical.swap(q1, q2);
    }
}

/// Compilation metrics
#[derive(Debug, Clone)]
pub struct CompilationMetrics {
    /// Total gate count
    pub gate_count: usize,
    /// Circuit depth
    pub depth: usize,
    /// Two-qubit gate count
    pub two_qubit_gate_count: usize,
    /// Total error estimate
    pub total_error: f64,
    /// Execution time estimate (microseconds)
    pub execution_time: f64,
    /// Compilation time (seconds)
    pub compilation_time: f64,
    /// Number of SWAP gates added
    pub swap_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_integration::DeviceTopology;
    
    #[test]
    fn test_device_compiler_creation() {
        let topology = DeviceTopology::new(5)
            .add_edge(0, 1)
            .add_edge(1, 2)
            .add_edge(2, 3)
            .add_edge(3, 4);
        
        let compiler = DeviceCompiler::new(topology);
        assert_eq!(compiler.options.optimization_level, 2);
    }
    
    #[test]
    fn test_device_characterization() {
        let mut char = DeviceCharacterization::default_for_device(3);
        char.set_gate_error("CNOT", &[0, 1], 0.005);
        
        assert_eq!(char.get_gate_error("CNOT", &[0, 1]), 0.005);
        assert_eq!(char.get_gate_error("X", &[0]), 0.001);
    }
    
    #[test]
    fn test_qubit_mapping() {
        let mut mapping = QubitMapping::identity(3);
        assert_eq!(mapping.logical_to_physical(1), 1);
        
        mapping.apply_swap(0, 2);
        assert_eq!(mapping.logical_to_physical(0), 2);
        assert_eq!(mapping.logical_to_physical(2), 0);
    }
    
    #[test]
    fn test_compilation_options() {
        let options = CompilationOptions {
            optimization_level: 3,
            noise_aware: false,
            routing_algorithm: RoutingAlgorithm::Lookahead,
            ..Default::default()
        };
        
        assert_eq!(options.optimization_level, 3);
        assert!(!options.noise_aware);
    }
}