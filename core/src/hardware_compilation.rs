//! Enhanced Hardware-Specific Compilation Algorithms
//!
//! This module provides advanced quantum gate decomposition and compilation
//! algorithms specifically optimized for different quantum hardware platforms,
//! including superconducting qubits, trapped ions, photonic systems, and neutral atoms.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
    matrix_ops::DenseMatrix,
    synthesis::{decompose_single_qubit_zyz, decompose_two_qubit_kak, KAKDecomposition},
    trapped_ion::TrappedIonSystem,
    neutral_atom::NeutralAtomQC,
    photonic::PhotonicSystem,
    pulse::{PulseSequence, Pulse},
};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

/// Hardware platform types for compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwarePlatform {
    /// Superconducting qubit systems (IBM, Google, Rigetti)
    Superconducting,
    /// Trapped ion systems (IonQ, Honeywell)
    TrappedIon,
    /// Photonic quantum systems (Xanadu, PsiQuantum)
    Photonic,
    /// Neutral atom systems (QuEra, Pasqal)
    NeutralAtom,
    /// Silicon quantum dots (Intel)
    SiliconQuantumDot,
    /// Topological qubits (Microsoft)
    Topological,
    /// Generic universal gate set
    Universal,
}

/// Native gate sets for different hardware platforms
#[derive(Debug, Clone)]
pub struct NativeGateSet {
    /// Single-qubit native gates
    pub single_qubit_gates: Vec<NativeGateType>,
    /// Two-qubit native gates
    pub two_qubit_gates: Vec<NativeGateType>,
    /// Multi-qubit native gates
    pub multi_qubit_gates: Vec<NativeGateType>,
    /// Parametric gates with constraints
    pub parametric_constraints: HashMap<NativeGateType, ParameterConstraints>,
    /// Gate fidelities
    pub gate_fidelities: HashMap<NativeGateType, f64>,
    /// Gate durations
    pub gate_durations: HashMap<NativeGateType, Duration>,
}

/// Native gate types for hardware platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeGateType {
    // Single-qubit gates
    X, Y, Z,
    Rx, Ry, Rz,
    SqrtX, SqrtY,
    H, S, T,
    Phase,
    
    // Two-qubit gates
    CNOT, CZ, CY,
    XX, YY, ZZ,
    MS, // Mølmer-Sørensen
    Iswap, SqrtIswap,
    
    // Multi-qubit gates
    Toffoli,
    Fredkin,
    GlobalPhase,
    
    // Platform-specific
    VirtualZ, // Virtual Z gates for superconducting
    BeamSplitter, // For photonic systems
    Rydberg, // For neutral atoms
}

/// Parameter constraints for native gates
#[derive(Debug, Clone)]
pub struct ParameterConstraints {
    /// Minimum parameter value
    pub min_value: f64,
    /// Maximum parameter value
    pub max_value: f64,
    /// Parameter granularity
    pub granularity: f64,
    /// Allowed discrete values (for calibrated gates)
    pub discrete_values: Option<Vec<f64>>,
}

/// Hardware topology constraints
#[derive(Debug, Clone)]
pub struct HardwareTopology {
    /// Qubit connectivity graph
    pub connectivity: HashMap<QubitId, HashSet<QubitId>>,
    /// Physical qubit coordinates
    pub qubit_positions: HashMap<QubitId, (f64, f64, f64)>,
    /// Coupling strengths between qubits
    pub coupling_strengths: HashMap<(QubitId, QubitId), f64>,
    /// Crosstalk matrix
    pub crosstalk_matrix: Array2<f64>,
    /// Maximum simultaneous operations
    pub max_parallel_ops: usize,
}

/// Hardware-specific compilation configuration
#[derive(Debug, Clone)]
pub struct HardwareCompilationConfig {
    /// Target hardware platform
    pub platform: HardwarePlatform,
    /// Native gate set
    pub native_gates: NativeGateSet,
    /// Hardware topology
    pub topology: HardwareTopology,
    /// Optimization objectives
    pub optimization_objectives: Vec<OptimizationObjective>,
    /// Compilation tolerances
    pub tolerances: CompilationTolerances,
    /// Enable cross-talk mitigation
    pub enable_crosstalk_mitigation: bool,
    /// Use pulse-level optimization
    pub use_pulse_optimization: bool,
}

/// Optimization objectives for compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationObjective {
    /// Minimize gate count
    MinimizeGateCount,
    /// Minimize circuit depth
    MinimizeDepth,
    /// Maximize fidelity
    MaximizeFidelity,
    /// Minimize execution time
    MinimizeTime,
    /// Minimize crosstalk
    MinimizeCrosstalk,
    /// Balance all objectives
    Balanced,
}

/// Compilation tolerances
#[derive(Debug, Clone)]
pub struct CompilationTolerances {
    /// Gate decomposition tolerance
    pub decomposition_tolerance: f64,
    /// Parameter optimization tolerance
    pub parameter_tolerance: f64,
    /// Fidelity threshold
    pub fidelity_threshold: f64,
    /// Maximum compilation time
    pub max_compilation_time: Duration,
}

/// Hardware-specific quantum compiler
#[derive(Debug)]
pub struct HardwareCompiler {
    config: HardwareCompilationConfig,
    decomposition_cache: Arc<RwLock<DecompositionCache>>,
    optimization_engine: Arc<RwLock<HardwareOptimizationEngine>>,
    performance_monitor: Arc<RwLock<CompilationPerformanceMonitor>>,
}

/// Cache for decomposed gates
#[derive(Debug)]
pub struct DecompositionCache {
    /// Cached single-qubit decompositions
    single_qubit_cache: HashMap<String, Vec<CompiledGate>>,
    /// Cached two-qubit decompositions
    two_qubit_cache: HashMap<String, Vec<CompiledGate>>,
    /// Cache hit statistics
    cache_stats: CacheStatistics,
}

/// Compiled gate representation
#[derive(Debug, Clone)]
pub struct CompiledGate {
    /// Native gate type
    pub gate_type: NativeGateType,
    /// Target qubits
    pub qubits: Vec<QubitId>,
    /// Gate parameters
    pub parameters: Vec<f64>,
    /// Estimated fidelity
    pub fidelity: f64,
    /// Estimated duration
    pub duration: Duration,
    /// Pulse sequence (if available)
    pub pulse_sequence: Option<PulseSequence>,
}

/// Hardware optimization engine
#[derive(Debug)]
pub struct HardwareOptimizationEngine {
    /// Platform-specific optimizers
    optimizers: HashMap<HardwarePlatform, Box<dyn PlatformOptimizer>>,
    /// Optimization history
    optimization_history: Vec<OptimizationRecord>,
}

/// Platform-specific optimization trait
pub trait PlatformOptimizer: std::fmt::Debug + Send + Sync {
    /// Optimize gate sequence for specific platform
    fn optimize_sequence(&self, gates: &[CompiledGate], config: &HardwareCompilationConfig) 
        -> QuantRS2Result<OptimizedSequence>;
    
    /// Estimate sequence fidelity
    fn estimate_fidelity(&self, sequence: &[CompiledGate]) -> f64;
    
    /// Get platform-specific constraints
    fn get_constraints(&self) -> PlatformConstraints;
}

/// Optimized gate sequence
#[derive(Debug, Clone)]
pub struct OptimizedSequence {
    /// Optimized gates
    pub gates: Vec<CompiledGate>,
    /// Total estimated fidelity
    pub total_fidelity: f64,
    /// Total execution time
    pub total_time: Duration,
    /// Optimization metrics
    pub metrics: OptimizationMetrics,
}

/// Platform-specific constraints
#[derive(Debug, Clone)]
pub struct PlatformConstraints {
    /// Maximum qubit count
    pub max_qubits: usize,
    /// Gate set limitations
    pub gate_limitations: Vec<GateLimitation>,
    /// Timing constraints
    pub timing_constraints: TimingConstraints,
    /// Error model
    pub error_model: ErrorModel,
}

/// Gate limitations for platforms
#[derive(Debug, Clone)]
pub enum GateLimitation {
    /// Only specific parameter values allowed
    DiscreteParameters(NativeGateType, Vec<f64>),
    /// Gate only available on specific qubit pairs
    RestrictedConnectivity(NativeGateType, Vec<(QubitId, QubitId)>),
    /// Gate has limited coherence time
    CoherenceLimit(NativeGateType, Duration),
    /// Gate requires calibration
    RequiresCalibration(NativeGateType),
}

/// Timing constraints for hardware
#[derive(Debug, Clone)]
pub struct TimingConstraints {
    /// Minimum gate separation
    pub min_gate_separation: Duration,
    /// Maximum parallel operations
    pub max_parallel_ops: usize,
    /// Qubit-specific timing
    pub qubit_timing: HashMap<QubitId, Duration>,
}

/// Error model for hardware
#[derive(Debug, Clone)]
pub struct ErrorModel {
    /// Single-qubit gate errors
    pub single_qubit_errors: HashMap<NativeGateType, f64>,
    /// Two-qubit gate errors
    pub two_qubit_errors: HashMap<NativeGateType, f64>,
    /// Readout errors
    pub readout_errors: HashMap<QubitId, f64>,
    /// Idle decay rates
    pub idle_decay_rates: HashMap<QubitId, f64>,
}

/// Performance monitoring for compilation
#[derive(Debug)]
pub struct CompilationPerformanceMonitor {
    /// Compilation times
    compilation_times: Vec<Duration>,
    /// Gate count reductions
    gate_count_reductions: Vec<f64>,
    /// Fidelity improvements
    fidelity_improvements: Vec<f64>,
    /// Cache hit rates
    cache_hit_rates: Vec<f64>,
}

/// Optimization metrics
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    /// Original gate count
    pub original_gate_count: usize,
    /// Optimized gate count
    pub optimized_gate_count: usize,
    /// Gate count reduction percentage
    pub gate_count_reduction: f64,
    /// Original circuit depth
    pub original_depth: usize,
    /// Optimized circuit depth
    pub optimized_depth: usize,
    /// Depth reduction percentage
    pub depth_reduction: f64,
    /// Estimated fidelity improvement
    pub fidelity_improvement: f64,
    /// Compilation time
    pub compilation_time: Duration,
}

/// Optimization record for history tracking
#[derive(Debug, Clone)]
pub struct OptimizationRecord {
    /// Timestamp
    pub timestamp: Instant,
    /// Platform
    pub platform: HardwarePlatform,
    /// Input gate count
    pub input_gates: usize,
    /// Output gate count
    pub output_gates: usize,
    /// Optimization metrics
    pub metrics: OptimizationMetrics,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Total cache requests
    pub total_requests: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Cache hit rate
    pub hit_rate: f64,
}

impl HardwareCompiler {
    /// Create a new hardware compiler
    pub fn new(config: HardwareCompilationConfig) -> QuantRS2Result<Self> {
        let decomposition_cache = Arc::new(RwLock::new(DecompositionCache::new()));
        let optimization_engine = Arc::new(RwLock::new(HardwareOptimizationEngine::new(&config)?));
        let performance_monitor = Arc::new(RwLock::new(CompilationPerformanceMonitor::new()));

        Ok(Self {
            config,
            decomposition_cache,
            optimization_engine,
            performance_monitor,
        })
    }

    /// Compile a quantum gate for the target hardware
    pub fn compile_gate(&self, gate: &dyn GateOp, qubits: &[QubitId]) -> QuantRS2Result<Vec<CompiledGate>> {
        let start_time = Instant::now();

        // Check cache first
        let cache_key = self.generate_cache_key(gate, qubits);
        if let Some(cached_result) = self.check_cache(&cache_key)? {
            self.record_cache_hit();
            return Ok(cached_result);
        }

        self.record_cache_miss();

        // Perform hardware-specific decomposition
        let compiled_gates = match qubits.len() {
            1 => self.compile_single_qubit_gate(gate, qubits[0])?,
            2 => self.compile_two_qubit_gate(gate, qubits[0], qubits[1])?,
            _ => self.compile_multi_qubit_gate(gate, qubits)?,
        };

        // Optimize for target platform
        let optimized_gates = self.optimize_for_platform(&compiled_gates)?;

        // Cache the result
        self.cache_result(&cache_key, &optimized_gates)?;

        let compilation_time = start_time.elapsed();
        self.record_compilation_time(compilation_time);

        Ok(optimized_gates)
    }

    /// Compile a single-qubit gate
    fn compile_single_qubit_gate(&self, gate: &dyn GateOp, qubit: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 2)?; // Single qubit gate is 2x2
        
        // Check if gate is already in native set
        if let Some(native_gate) = self.find_native_single_qubit_gate(&matrix)? {
            return Ok(vec![native_gate]);
        }

        // Decompose using platform-specific method
        match self.config.platform {
            HardwarePlatform::Superconducting => self.decompose_for_superconducting_single(gate, qubit),
            HardwarePlatform::TrappedIon => self.decompose_for_trapped_ion_single(gate, qubit),
            HardwarePlatform::Photonic => self.decompose_for_photonic_single(gate, qubit),
            HardwarePlatform::NeutralAtom => self.decompose_for_neutral_atom_single(gate, qubit),
            _ => self.decompose_universal_single(gate, qubit),
        }
    }

    /// Compile a two-qubit gate
    fn compile_two_qubit_gate(&self, gate: &dyn GateOp, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        // Check connectivity constraints
        if !self.check_connectivity(qubit1, qubit2)? {
            return self.handle_connectivity_constraint(gate, qubit1, qubit2);
        }

        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 4)?; // Two qubit gate is 4x4
        
        // Check if gate is already in native set
        if let Some(native_gate) = self.find_native_two_qubit_gate(&matrix, qubit1, qubit2)? {
            return Ok(vec![native_gate]);
        }

        // Decompose using platform-specific method
        match self.config.platform {
            HardwarePlatform::Superconducting => self.decompose_for_superconducting_two(gate, qubit1, qubit2),
            HardwarePlatform::TrappedIon => self.decompose_for_trapped_ion_two(gate, qubit1, qubit2),
            HardwarePlatform::Photonic => self.decompose_for_photonic_two(gate, qubit1, qubit2),
            HardwarePlatform::NeutralAtom => self.decompose_for_neutral_atom_two(gate, qubit1, qubit2),
            _ => self.decompose_universal_two(gate, qubit1, qubit2),
        }
    }

    /// Compile a multi-qubit gate
    fn compile_multi_qubit_gate(&self, gate: &dyn GateOp, qubits: &[QubitId]) -> QuantRS2Result<Vec<CompiledGate>> {
        // Check if platform supports multi-qubit gates natively
        if self.config.native_gates.multi_qubit_gates.is_empty() {
            return self.decompose_to_two_qubit_gates(gate, qubits);
        }

        // Try to find native multi-qubit implementation
        match self.config.platform {
            HardwarePlatform::TrappedIon => self.compile_trapped_ion_multi(gate, qubits),
            HardwarePlatform::NeutralAtom => self.compile_neutral_atom_multi(gate, qubits),
            _ => self.decompose_to_two_qubit_gates(gate, qubits),
        }
    }

    /// Decompose for superconducting single-qubit gates
    fn decompose_for_superconducting_single(&self, gate: &dyn GateOp, qubit: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 2)?; // Single qubit gate is 2x2
        
        // Use virtual Z gates when possible for superconducting qubits
        if self.is_z_rotation(&matrix)? {
            let angle = self.extract_z_rotation_angle(&matrix)?;
            return Ok(vec![CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![qubit],
                parameters: vec![angle],
                fidelity: 1.0, // Virtual Z gates are perfect
                duration: Duration::from_nanos(0), // Virtual gates take no time
                pulse_sequence: None,
            }]);
        }

        // Decompose to Rx-Rz sequence
        let decomposition = self.decompose_to_rx_rz(&matrix)?;
        let mut compiled_gates = Vec::new();

        for (gate_type, angle) in decomposition {
            compiled_gates.push(CompiledGate {
                gate_type,
                qubits: vec![qubit],
                parameters: vec![angle],
                fidelity: self.get_gate_fidelity(gate_type),
                duration: self.get_gate_duration(gate_type),
                pulse_sequence: self.generate_pulse_sequence(gate_type, &[angle])?,
            });
        }

        Ok(compiled_gates)
    }

    /// Decompose for trapped ion single-qubit gates
    fn decompose_for_trapped_ion_single(&self, gate: &dyn GateOp, qubit: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 2)?; // Single qubit gate is 2x2
        
        // Trapped ions excel at arbitrary rotations
        let (theta, phi, lambda) = self.extract_euler_angles(&matrix)?;
        
        let mut compiled_gates = Vec::new();
        
        // Rz(lambda)
        if lambda.abs() > self.config.tolerances.parameter_tolerance {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::Rz,
                qubits: vec![qubit],
                parameters: vec![lambda],
                fidelity: self.get_gate_fidelity(NativeGateType::Rz),
                duration: self.get_gate_duration(NativeGateType::Rz),
                pulse_sequence: self.generate_pulse_sequence(NativeGateType::Rz, &[lambda])?,
            });
        }
        
        // Ry(theta)
        if theta.abs() > self.config.tolerances.parameter_tolerance {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::Ry,
                qubits: vec![qubit],
                parameters: vec![theta],
                fidelity: self.get_gate_fidelity(NativeGateType::Ry),
                duration: self.get_gate_duration(NativeGateType::Ry),
                pulse_sequence: self.generate_pulse_sequence(NativeGateType::Ry, &[theta])?,
            });
        }
        
        // Rz(phi)
        if phi.abs() > self.config.tolerances.parameter_tolerance {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::Rz,
                qubits: vec![qubit],
                parameters: vec![phi],
                fidelity: self.get_gate_fidelity(NativeGateType::Rz),
                duration: self.get_gate_duration(NativeGateType::Rz),
                pulse_sequence: self.generate_pulse_sequence(NativeGateType::Rz, &[phi])?,
            });
        }

        Ok(compiled_gates)
    }

    /// Decompose for superconducting two-qubit gates
    fn decompose_for_superconducting_two(&self, gate: &dyn GateOp, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 4)?; // Two qubit gate is 4x4
        
        // Use KAK decomposition
        let kak_decomp = decompose_two_qubit_kak(&matrix.as_array().view())?;
        let mut compiled_gates = Vec::new();

        // Left single-qubit gates (before interaction)
        let (left_gate1, left_gate2) = &kak_decomp.left_gates;
        // For now, we'll use simplified compilation - a real implementation would 
        // convert SingleQubitDecomposition to actual gates
        
        // Native two-qubit interaction based on coefficients
        let interaction_strength = (kak_decomp.interaction.xx.abs() + 
                                   kak_decomp.interaction.yy.abs() + 
                                   kak_decomp.interaction.zz.abs()).max(0.01);
        let native_two_qubit = self.get_native_two_qubit_gate();
        
        // Add interaction gates based on strength (simplified)
        if interaction_strength > 0.01 {
            compiled_gates.push(CompiledGate {
                gate_type: native_two_qubit,
                qubits: vec![qubit1, qubit2],
                parameters: vec![interaction_strength],
                fidelity: self.get_gate_fidelity(native_two_qubit),
                duration: self.get_gate_duration(native_two_qubit),
                pulse_sequence: self.generate_pulse_sequence(native_two_qubit, &[interaction_strength])?,
            });
        }

        // Right single-qubit gates (after interaction)
        let (right_gate1, right_gate2) = &kak_decomp.right_gates;
        // For now, we'll use simplified compilation - a real implementation would 
        // convert SingleQubitDecomposition to actual gates

        Ok(compiled_gates)
    }

    /// Decompose for trapped ion two-qubit gates
    fn decompose_for_trapped_ion_two(&self, gate: &dyn GateOp, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 4)?; // Two qubit gate is 4x4
        
        // Trapped ions can implement arbitrary two-qubit gates with Mølmer-Sørensen gates
        let ms_decomp = self.decompose_to_ms_gates(&matrix)?;
        let mut compiled_gates = Vec::new();

        for ms_gate in ms_decomp {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::MS,
                qubits: vec![qubit1, qubit2],
                parameters: ms_gate.parameters.clone(),
                fidelity: self.get_gate_fidelity(NativeGateType::MS),
                duration: self.get_gate_duration(NativeGateType::MS),
                pulse_sequence: self.generate_pulse_sequence(NativeGateType::MS, &ms_gate.parameters)?,
            });
        }

        Ok(compiled_gates)
    }

    /// Check qubit connectivity
    fn check_connectivity(&self, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<bool> {
        if let Some(neighbors) = self.config.topology.connectivity.get(&qubit1) {
            Ok(neighbors.contains(&qubit2))
        } else {
            Ok(false)
        }
    }

    /// Handle connectivity constraints by inserting SWAP gates
    fn handle_connectivity_constraint(&self, gate: &dyn GateOp, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        // Find shortest path between qubits
        let path = self.find_shortest_path(qubit1, qubit2)?;
        let mut compiled_gates = Vec::new();

        // Insert SWAP gates along the path
        for i in 0..path.len()-2 {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::CNOT, // Implemented as 3 CNOTs
                qubits: vec![path[i], path[i+1]],
                parameters: vec![],
                fidelity: self.get_gate_fidelity(NativeGateType::CNOT).powi(3),
                duration: self.get_gate_duration(NativeGateType::CNOT) * 3,
                pulse_sequence: None,
            });
        }

        // Apply the original gate on adjacent qubits
        let original_gate_compiled = self.compile_two_qubit_gate(gate, path[path.len()-2], path[path.len()-1])?;
        compiled_gates.extend(original_gate_compiled);

        // Uncompute SWAP gates
        for i in (0..path.len()-2).rev() {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::CNOT,
                qubits: vec![path[i], path[i+1]],
                parameters: vec![],
                fidelity: self.get_gate_fidelity(NativeGateType::CNOT).powi(3),
                duration: self.get_gate_duration(NativeGateType::CNOT) * 3,
                pulse_sequence: None,
            });
        }

        Ok(compiled_gates)
    }

    /// Find shortest path between qubits
    fn find_shortest_path(&self, start: QubitId, end: QubitId) -> QuantRS2Result<Vec<QubitId>> {
        // Simple BFS implementation
        use std::collections::VecDeque;
        
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(current) = queue.pop_front() {
            if current == end {
                // Reconstruct path
                let mut path = Vec::new();
                let mut curr = end;
                path.push(curr);
                
                while let Some(&prev) = parent.get(&curr) {
                    path.push(prev);
                    curr = prev;
                }
                
                path.reverse();
                return Ok(path);
            }
            
            if let Some(neighbors) = self.config.topology.connectivity.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        Err(QuantRS2Error::InvalidParameter(
            format!("No path found between qubits {:?} and {:?}", start, end)
        ))
    }

    /// Optimize compiled gates for the target platform
    fn optimize_for_platform(&self, gates: &[CompiledGate]) -> QuantRS2Result<Vec<CompiledGate>> {
        let engine = self.optimization_engine.read().unwrap();
        
        if let Some(optimizer) = engine.optimizers.get(&self.config.platform) {
            let optimized = optimizer.optimize_sequence(gates, &self.config)?;
            Ok(optimized.gates)
        } else {
            // Default optimization
            Ok(gates.to_vec())
        }
    }

    /// Helper methods for gate property extraction
    fn is_z_rotation(&self, matrix: &DenseMatrix) -> QuantRS2Result<bool> {
        // Check if matrix represents a Z rotation
        let tolerance = self.config.tolerances.decomposition_tolerance;
        
        // Z rotation matrix has form [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        let arr = matrix.as_array();
        if arr[(0, 1)].norm() > tolerance || arr[(1, 0)].norm() > tolerance {
            return Ok(false);
        }
        
        Ok(true)
    }

    fn extract_z_rotation_angle(&self, matrix: &DenseMatrix) -> QuantRS2Result<f64> {
        let arr = matrix.as_array();
        let z00 = arr[(0, 0)];
        let z11 = arr[(1, 1)];
        
        // Extract angle from phase difference
        let angle = (z11 / z00).arg();
        Ok(angle)
    }

    fn decompose_to_rx_rz(&self, matrix: &DenseMatrix) -> QuantRS2Result<Vec<(NativeGateType, f64)>> {
        // Simplified Rx-Rz decomposition
        let (theta, phi, lambda) = self.extract_euler_angles(matrix)?;
        
        let mut decomposition = Vec::new();
        
        if lambda.abs() > self.config.tolerances.parameter_tolerance {
            decomposition.push((NativeGateType::Rz, lambda));
        }
        if theta.abs() > self.config.tolerances.parameter_tolerance {
            decomposition.push((NativeGateType::Rx, theta));
        }
        if phi.abs() > self.config.tolerances.parameter_tolerance {
            decomposition.push((NativeGateType::Rz, phi));
        }
        
        Ok(decomposition)
    }

    fn extract_euler_angles(&self, matrix: &DenseMatrix) -> QuantRS2Result<(f64, f64, f64)> {
        // Extract ZYZ Euler angles from 2x2 unitary matrix
        let arr = matrix.as_array();
        let u00 = arr[(0, 0)];
        let u01 = arr[(0, 1)];
        let u10 = arr[(1, 0)];
        let u11 = arr[(1, 1)];
        
        let theta: f64 = 2.0 * (u01.norm()).asin(); // Use asin instead of acos for correct ZYZ decomposition
        let phi = if theta.abs() < 1e-10 {
            0.0
        } else {
            (u11 / u00).arg() + (u01 / (-u10)).arg()
        };
        let lambda = if theta.abs() < 1e-10 {
            (u11 / u00).arg()
        } else {
            (u11 / u00).arg() - (u01 / (-u10)).arg()
        };
        
        Ok((theta, phi, lambda))
    }

    fn get_gate_fidelity(&self, gate_type: NativeGateType) -> f64 {
        self.config.native_gates.gate_fidelities
            .get(&gate_type)
            .copied()
            .unwrap_or(0.999) // Default high fidelity
    }

    fn get_gate_duration(&self, gate_type: NativeGateType) -> Duration {
        self.config.native_gates.gate_durations
            .get(&gate_type)
            .copied()
            .unwrap_or(Duration::from_nanos(100)) // Default 100ns
    }

    fn get_native_two_qubit_gate(&self) -> NativeGateType {
        match self.config.platform {
            HardwarePlatform::Superconducting => NativeGateType::CNOT,
            HardwarePlatform::TrappedIon => NativeGateType::MS,
            HardwarePlatform::Photonic => NativeGateType::CZ,
            HardwarePlatform::NeutralAtom => NativeGateType::CZ,
            _ => NativeGateType::CNOT,
        }
    }

    /// Generate cache key for gate and qubits
    fn generate_cache_key(&self, gate: &dyn GateOp, qubits: &[QubitId]) -> String {
        format!("{}_{:?}", gate.name(), qubits)
    }

    /// Utility methods for other decompositions and optimizations
    fn find_native_single_qubit_gate(&self, matrix: &DenseMatrix) -> QuantRS2Result<Option<CompiledGate>> {
        // Check if matrix matches any native single-qubit gate
        // This is a simplified implementation
        Ok(None)
    }

    fn find_native_two_qubit_gate(&self, matrix: &DenseMatrix, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<Option<CompiledGate>> {
        // Check if matrix matches any native two-qubit gate
        // This is a simplified implementation
        Ok(None)
    }

    fn decompose_for_photonic_single(&self, gate: &dyn GateOp, qubit: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        // Photonic implementation would use beam splitters and phase shifters
        Ok(vec![])
    }

    fn decompose_for_neutral_atom_single(&self, gate: &dyn GateOp, qubit: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        // Neutral atom implementation
        Ok(vec![])
    }

    fn decompose_universal_single(&self, gate: &dyn GateOp, qubit: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        // Universal decomposition using any available gates
        Ok(vec![])
    }

    fn decompose_for_photonic_two(&self, gate: &dyn GateOp, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        // Photonic two-qubit implementation
        Ok(vec![])
    }

    fn decompose_for_neutral_atom_two(&self, gate: &dyn GateOp, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        // Neutral atom two-qubit implementation
        Ok(vec![])
    }

    fn decompose_universal_two(&self, gate: &dyn GateOp, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<Vec<CompiledGate>> {
        // Universal two-qubit decomposition
        Ok(vec![])
    }

    fn compile_trapped_ion_multi(&self, gate: &dyn GateOp, qubits: &[QubitId]) -> QuantRS2Result<Vec<CompiledGate>> {
        // Trapped ion multi-qubit implementation
        Ok(vec![])
    }

    fn compile_neutral_atom_multi(&self, gate: &dyn GateOp, qubits: &[QubitId]) -> QuantRS2Result<Vec<CompiledGate>> {
        // Neutral atom multi-qubit implementation
        Ok(vec![])
    }

    fn decompose_to_two_qubit_gates(&self, gate: &dyn GateOp, qubits: &[QubitId]) -> QuantRS2Result<Vec<CompiledGate>> {
        // Generic decomposition to two-qubit gates
        Ok(vec![])
    }

    fn decompose_to_ms_gates(&self, matrix: &DenseMatrix) -> QuantRS2Result<Vec<CompiledGate>> {
        // Decompose to Mølmer-Sørensen gates
        Ok(vec![])
    }

    fn generate_pulse_sequence(&self, gate_type: NativeGateType, parameters: &[f64]) -> QuantRS2Result<Option<PulseSequence>> {
        if !self.config.use_pulse_optimization {
            return Ok(None);
        }
        
        // Generate platform-specific pulse sequence
        match self.config.platform {
            HardwarePlatform::Superconducting => self.generate_superconducting_pulses(gate_type, parameters),
            HardwarePlatform::TrappedIon => self.generate_trapped_ion_pulses(gate_type, parameters),
            _ => Ok(None),
        }
    }

    fn generate_superconducting_pulses(&self, gate_type: NativeGateType, parameters: &[f64]) -> QuantRS2Result<Option<PulseSequence>> {
        // Generate microwave pulses for superconducting qubits
        Ok(None)
    }

    fn generate_trapped_ion_pulses(&self, gate_type: NativeGateType, parameters: &[f64]) -> QuantRS2Result<Option<PulseSequence>> {
        // Generate laser pulses for trapped ions
        Ok(None)
    }

    // Cache management methods
    fn check_cache(&self, key: &str) -> QuantRS2Result<Option<Vec<CompiledGate>>> {
        let cache = self.decomposition_cache.read().unwrap();
        Ok(cache.single_qubit_cache.get(key).cloned())
    }

    fn cache_result(&self, key: &str, gates: &[CompiledGate]) -> QuantRS2Result<()> {
        let mut cache = self.decomposition_cache.write().unwrap();
        cache.single_qubit_cache.insert(key.to_string(), gates.to_vec());
        cache.cache_stats.total_requests += 1;
        Ok(())
    }

    fn record_cache_hit(&self) {
        let mut cache = self.decomposition_cache.write().unwrap();
        cache.cache_stats.cache_hits += 1;
        cache.cache_stats.hit_rate = cache.cache_stats.cache_hits as f64 / cache.cache_stats.total_requests as f64;
    }

    fn record_cache_miss(&self) {
        let mut cache = self.decomposition_cache.write().unwrap();
        cache.cache_stats.cache_misses += 1;
    }

    fn record_compilation_time(&self, duration: Duration) {
        let mut monitor = self.performance_monitor.write().unwrap();
        monitor.compilation_times.push(duration);
    }

    /// Get compilation performance statistics
    pub fn get_performance_stats(&self) -> CompilationPerformanceStats {
        let monitor = self.performance_monitor.read().unwrap();
        let cache = self.decomposition_cache.read().unwrap();
        
        CompilationPerformanceStats {
            average_compilation_time: monitor.compilation_times.iter().sum::<Duration>() / monitor.compilation_times.len() as u32,
            cache_statistics: cache.cache_stats.clone(),
            total_compilations: monitor.compilation_times.len(),
        }
    }
}

/// Compilation performance statistics
#[derive(Debug, Clone)]
pub struct CompilationPerformanceStats {
    /// Average compilation time
    pub average_compilation_time: Duration,
    /// Cache performance
    pub cache_statistics: CacheStatistics,
    /// Total number of compilations
    pub total_compilations: usize,
}

impl DecompositionCache {
    fn new() -> Self {
        Self {
            single_qubit_cache: HashMap::new(),
            two_qubit_cache: HashMap::new(),
            cache_stats: CacheStatistics::default(),
        }
    }
}

impl HardwareOptimizationEngine {
    fn new(config: &HardwareCompilationConfig) -> QuantRS2Result<Self> {
        let mut optimizers: HashMap<HardwarePlatform, Box<dyn PlatformOptimizer>> = HashMap::new();
        
        // Initialize platform-specific optimizers
        optimizers.insert(HardwarePlatform::Superconducting, Box::new(SuperconductingOptimizer::new()));
        optimizers.insert(HardwarePlatform::TrappedIon, Box::new(TrappedIonOptimizer::new()));
        optimizers.insert(HardwarePlatform::Photonic, Box::new(PhotonicOptimizer::new()));
        optimizers.insert(HardwarePlatform::NeutralAtom, Box::new(NeutralAtomOptimizer::new()));
        
        Ok(Self {
            optimizers,
            optimization_history: Vec::new(),
        })
    }
}

impl CompilationPerformanceMonitor {
    fn new() -> Self {
        Self {
            compilation_times: Vec::new(),
            gate_count_reductions: Vec::new(),
            fidelity_improvements: Vec::new(),
            cache_hit_rates: Vec::new(),
        }
    }
}

// Platform-specific optimizer implementations
#[derive(Debug)]
struct SuperconductingOptimizer;

impl SuperconductingOptimizer {
    fn new() -> Self {
        Self
    }
}

impl PlatformOptimizer for SuperconductingOptimizer {
    fn optimize_sequence(&self, gates: &[CompiledGate], config: &HardwareCompilationConfig) -> QuantRS2Result<OptimizedSequence> {
        // Superconducting-specific optimizations
        // 1. Virtual Z gate fusion
        // 2. CNOT gate reduction
        // 3. Cross-resonance pulse optimization
        
        let optimized_gates = self.fuse_virtual_z_gates(gates)?;
        let total_fidelity = self.estimate_fidelity(&optimized_gates);
        let total_time = optimized_gates.iter().map(|g| g.duration).sum();
        
        Ok(OptimizedSequence {
            gates: optimized_gates,
            total_fidelity,
            total_time,
            metrics: self.calculate_metrics(gates, &[], total_fidelity),
        })
    }
    
    fn estimate_fidelity(&self, sequence: &[CompiledGate]) -> f64 {
        sequence.iter().map(|g| g.fidelity).product()
    }
    
    fn get_constraints(&self) -> PlatformConstraints {
        PlatformConstraints {
            max_qubits: 1000,
            gate_limitations: vec![],
            timing_constraints: TimingConstraints {
                min_gate_separation: Duration::from_nanos(10),
                max_parallel_ops: 100,
                qubit_timing: HashMap::new(),
            },
            error_model: ErrorModel {
                single_qubit_errors: HashMap::new(),
                two_qubit_errors: HashMap::new(),
                readout_errors: HashMap::new(),
                idle_decay_rates: HashMap::new(),
            },
        }
    }
}

impl SuperconductingOptimizer {
    fn fuse_virtual_z_gates(&self, gates: &[CompiledGate]) -> QuantRS2Result<Vec<CompiledGate>> {
        // Fuse consecutive virtual Z gates
        let mut optimized = Vec::new();
        let mut current_z_angle = 0.0;
        let mut current_qubit = None;
        
        for gate in gates {
            match gate.gate_type {
                NativeGateType::VirtualZ => {
                    if Some(gate.qubits[0]) == current_qubit {
                        current_z_angle += gate.parameters[0];
                    } else {
                        if let Some(qubit) = current_qubit {
                            if current_z_angle.abs() > 1e-10 {
                                optimized.push(CompiledGate {
                                    gate_type: NativeGateType::VirtualZ,
                                    qubits: vec![qubit],
                                    parameters: vec![current_z_angle],
                                    fidelity: 1.0,
                                    duration: Duration::from_nanos(0),
                                    pulse_sequence: None,
                                });
                            }
                        }
                        current_qubit = Some(gate.qubits[0]);
                        current_z_angle = gate.parameters[0];
                    }
                }
                _ => {
                    if let Some(qubit) = current_qubit {
                        if current_z_angle.abs() > 1e-10 {
                            optimized.push(CompiledGate {
                                gate_type: NativeGateType::VirtualZ,
                                qubits: vec![qubit],
                                parameters: vec![current_z_angle],
                                fidelity: 1.0,
                                duration: Duration::from_nanos(0),
                                pulse_sequence: None,
                            });
                        }
                        current_qubit = None;
                        current_z_angle = 0.0;
                    }
                    optimized.push(gate.clone());
                }
            }
        }
        
        // Handle final virtual Z gate
        if let Some(qubit) = current_qubit {
            if current_z_angle.abs() > 1e-10 {
                optimized.push(CompiledGate {
                    gate_type: NativeGateType::VirtualZ,
                    qubits: vec![qubit],
                    parameters: vec![current_z_angle],
                    fidelity: 1.0,
                    duration: Duration::from_nanos(0),
                    pulse_sequence: None,
                });
            }
        }
        
        Ok(optimized)
    }
    
    fn calculate_metrics(&self, original: &[CompiledGate], optimized: &[CompiledGate], fidelity: f64) -> OptimizationMetrics {
        OptimizationMetrics {
            original_gate_count: original.len(),
            optimized_gate_count: optimized.len(),
            gate_count_reduction: (original.len() - optimized.len()) as f64 / original.len() as f64 * 100.0,
            original_depth: original.len(), // Simplified
            optimized_depth: optimized.len(), // Simplified
            depth_reduction: 0.0, // Would need proper circuit depth calculation
            fidelity_improvement: fidelity,
            compilation_time: Duration::from_millis(1),
        }
    }
}

// Similar implementations for other platform optimizers
#[derive(Debug)]
struct TrappedIonOptimizer;

impl TrappedIonOptimizer {
    fn new() -> Self {
        Self
    }
}

impl PlatformOptimizer for TrappedIonOptimizer {
    fn optimize_sequence(&self, gates: &[CompiledGate], config: &HardwareCompilationConfig) -> QuantRS2Result<OptimizedSequence> {
        // Trapped ion optimizations would focus on:
        // 1. MS gate optimization
        // 2. Ion chain reordering
        // 3. Parallel operations on non-adjacent ions
        
        let optimized_gates = gates.to_vec(); // Simplified
        let total_fidelity = self.estimate_fidelity(&optimized_gates);
        let total_time = optimized_gates.iter().map(|g| g.duration).sum();
        
        Ok(OptimizedSequence {
            gates: optimized_gates,
            total_fidelity,
            total_time,
            metrics: OptimizationMetrics {
                original_gate_count: gates.len(),
                optimized_gate_count: gates.len(),
                gate_count_reduction: 0.0,
                original_depth: gates.len(),
                optimized_depth: gates.len(),
                depth_reduction: 0.0,
                fidelity_improvement: total_fidelity,
                compilation_time: Duration::from_millis(1),
            },
        })
    }
    
    fn estimate_fidelity(&self, sequence: &[CompiledGate]) -> f64 {
        sequence.iter().map(|g| g.fidelity).product()
    }
    
    fn get_constraints(&self) -> PlatformConstraints {
        PlatformConstraints {
            max_qubits: 100,
            gate_limitations: vec![],
            timing_constraints: TimingConstraints {
                min_gate_separation: Duration::from_micros(1),
                max_parallel_ops: 10,
                qubit_timing: HashMap::new(),
            },
            error_model: ErrorModel {
                single_qubit_errors: HashMap::new(),
                two_qubit_errors: HashMap::new(),
                readout_errors: HashMap::new(),
                idle_decay_rates: HashMap::new(),
            },
        }
    }
}

#[derive(Debug)]
struct PhotonicOptimizer;

impl PhotonicOptimizer {
    fn new() -> Self {
        Self
    }
}

impl PlatformOptimizer for PhotonicOptimizer {
    fn optimize_sequence(&self, gates: &[CompiledGate], config: &HardwareCompilationConfig) -> QuantRS2Result<OptimizedSequence> {
        let optimized_gates = gates.to_vec();
        let total_fidelity = self.estimate_fidelity(&optimized_gates);
        let total_time = optimized_gates.iter().map(|g| g.duration).sum();
        
        Ok(OptimizedSequence {
            gates: optimized_gates,
            total_fidelity,
            total_time,
            metrics: OptimizationMetrics {
                original_gate_count: gates.len(),
                optimized_gate_count: gates.len(),
                gate_count_reduction: 0.0,
                original_depth: gates.len(),
                optimized_depth: gates.len(),
                depth_reduction: 0.0,
                fidelity_improvement: total_fidelity,
                compilation_time: Duration::from_millis(1),
            },
        })
    }
    
    fn estimate_fidelity(&self, sequence: &[CompiledGate]) -> f64 {
        sequence.iter().map(|g| g.fidelity).product()
    }
    
    fn get_constraints(&self) -> PlatformConstraints {
        PlatformConstraints {
            max_qubits: 216, // Xanadu X-Series
            gate_limitations: vec![],
            timing_constraints: TimingConstraints {
                min_gate_separation: Duration::from_nanos(1),
                max_parallel_ops: 50,
                qubit_timing: HashMap::new(),
            },
            error_model: ErrorModel {
                single_qubit_errors: HashMap::new(),
                two_qubit_errors: HashMap::new(),
                readout_errors: HashMap::new(),
                idle_decay_rates: HashMap::new(),
            },
        }
    }
}

#[derive(Debug)]
struct NeutralAtomOptimizer;

impl NeutralAtomOptimizer {
    fn new() -> Self {
        Self
    }
}

impl PlatformOptimizer for NeutralAtomOptimizer {
    fn optimize_sequence(&self, gates: &[CompiledGate], config: &HardwareCompilationConfig) -> QuantRS2Result<OptimizedSequence> {
        let optimized_gates = gates.to_vec();
        let total_fidelity = self.estimate_fidelity(&optimized_gates);
        let total_time = optimized_gates.iter().map(|g| g.duration).sum();
        
        Ok(OptimizedSequence {
            gates: optimized_gates,
            total_fidelity,
            total_time,
            metrics: OptimizationMetrics {
                original_gate_count: gates.len(),
                optimized_gate_count: gates.len(),
                gate_count_reduction: 0.0,
                original_depth: gates.len(),
                optimized_depth: gates.len(),
                depth_reduction: 0.0,
                fidelity_improvement: total_fidelity,
                compilation_time: Duration::from_millis(1),
            },
        })
    }
    
    fn estimate_fidelity(&self, sequence: &[CompiledGate]) -> f64 {
        sequence.iter().map(|g| g.fidelity).product()
    }
    
    fn get_constraints(&self) -> PlatformConstraints {
        PlatformConstraints {
            max_qubits: 256,
            gate_limitations: vec![],
            timing_constraints: TimingConstraints {
                min_gate_separation: Duration::from_micros(1),
                max_parallel_ops: 20,
                qubit_timing: HashMap::new(),
            },
            error_model: ErrorModel {
                single_qubit_errors: HashMap::new(),
                two_qubit_errors: HashMap::new(),
                readout_errors: HashMap::new(),
                idle_decay_rates: HashMap::new(),
            },
        }
    }
}

/// Factory functions for creating hardware-specific compilers
impl HardwareCompiler {
    /// Create a compiler for superconducting quantum processors
    pub fn for_superconducting(topology: HardwareTopology) -> QuantRS2Result<Self> {
        let config = HardwareCompilationConfig {
            platform: HardwarePlatform::Superconducting,
            native_gates: create_superconducting_gate_set(),
            topology,
            optimization_objectives: vec![OptimizationObjective::MinimizeGateCount, OptimizationObjective::MaximizeFidelity],
            tolerances: CompilationTolerances {
                decomposition_tolerance: 1e-12,
                parameter_tolerance: 1e-10,
                fidelity_threshold: 0.99,
                max_compilation_time: Duration::from_secs(60),
            },
            enable_crosstalk_mitigation: true,
            use_pulse_optimization: true,
        };
        
        Self::new(config)
    }
    
    /// Create a compiler for trapped ion systems
    pub fn for_trapped_ion(topology: HardwareTopology) -> QuantRS2Result<Self> {
        let config = HardwareCompilationConfig {
            platform: HardwarePlatform::TrappedIon,
            native_gates: create_trapped_ion_gate_set(),
            topology,
            optimization_objectives: vec![OptimizationObjective::MinimizeTime, OptimizationObjective::MaximizeFidelity],
            tolerances: CompilationTolerances {
                decomposition_tolerance: 1e-14,
                parameter_tolerance: 1e-12,
                fidelity_threshold: 0.995,
                max_compilation_time: Duration::from_secs(120),
            },
            enable_crosstalk_mitigation: false,
            use_pulse_optimization: true,
        };
        
        Self::new(config)
    }
}

/// Helper functions for creating platform-specific gate sets
fn create_superconducting_gate_set() -> NativeGateSet {
    let mut gate_fidelities = HashMap::new();
    gate_fidelities.insert(NativeGateType::VirtualZ, 1.0);
    gate_fidelities.insert(NativeGateType::Rx, 0.9995);
    gate_fidelities.insert(NativeGateType::Ry, 0.9995);
    gate_fidelities.insert(NativeGateType::CNOT, 0.995);
    
    let mut gate_durations = HashMap::new();
    gate_durations.insert(NativeGateType::VirtualZ, Duration::from_nanos(0));
    gate_durations.insert(NativeGateType::Rx, Duration::from_nanos(20));
    gate_durations.insert(NativeGateType::Ry, Duration::from_nanos(20));
    gate_durations.insert(NativeGateType::CNOT, Duration::from_nanos(300));
    
    NativeGateSet {
        single_qubit_gates: vec![NativeGateType::Rx, NativeGateType::Ry, NativeGateType::VirtualZ],
        two_qubit_gates: vec![NativeGateType::CNOT],
        multi_qubit_gates: vec![],
        parametric_constraints: HashMap::new(),
        gate_fidelities,
        gate_durations,
    }
}

fn create_trapped_ion_gate_set() -> NativeGateSet {
    let mut gate_fidelities = HashMap::new();
    gate_fidelities.insert(NativeGateType::Rx, 0.9999);
    gate_fidelities.insert(NativeGateType::Ry, 0.9999);
    gate_fidelities.insert(NativeGateType::Rz, 0.9999);
    gate_fidelities.insert(NativeGateType::MS, 0.998);
    
    let mut gate_durations = HashMap::new();
    gate_durations.insert(NativeGateType::Rx, Duration::from_micros(10));
    gate_durations.insert(NativeGateType::Ry, Duration::from_micros(10));
    gate_durations.insert(NativeGateType::Rz, Duration::from_micros(1));
    gate_durations.insert(NativeGateType::MS, Duration::from_micros(100));
    
    NativeGateSet {
        single_qubit_gates: vec![NativeGateType::Rx, NativeGateType::Ry, NativeGateType::Rz],
        two_qubit_gates: vec![NativeGateType::MS],
        multi_qubit_gates: vec![NativeGateType::MS], // MS can be applied to multiple ions
        parametric_constraints: HashMap::new(),
        gate_fidelities,
        gate_durations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubit::QubitId;
    use std::collections::{HashMap, HashSet};

    fn create_test_topology() -> HardwareTopology {
        let mut connectivity = HashMap::new();
        let mut qubit_positions = HashMap::new();
        
        // Create a simple 4-qubit linear topology
        for i in 0..4 {
            let qubit = QubitId::new(i);
            qubit_positions.insert(qubit, (i as f64, 0.0, 0.0));
            
            let mut neighbors = HashSet::new();
            if i > 0 {
                neighbors.insert(QubitId::new(i - 1));
            }
            if i < 3 {
                neighbors.insert(QubitId::new(i + 1));
            }
            connectivity.insert(qubit, neighbors);
        }
        
        HardwareTopology {
            connectivity,
            qubit_positions,
            coupling_strengths: HashMap::new(),
            crosstalk_matrix: Array2::zeros((4, 4)),
            max_parallel_ops: 2,
        }
    }
    
    #[test]
    fn test_superconducting_compiler_creation() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology);
        assert!(compiler.is_ok());
        
        let compiler = compiler.unwrap();
        assert_eq!(compiler.config.platform, HardwarePlatform::Superconducting);
        assert!(compiler.config.native_gates.single_qubit_gates.contains(&NativeGateType::VirtualZ));
    }
    
    #[test]
    fn test_trapped_ion_compiler_creation() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_trapped_ion(topology);
        assert!(compiler.is_ok());
        
        let compiler = compiler.unwrap();
        assert_eq!(compiler.config.platform, HardwarePlatform::TrappedIon);
        assert!(compiler.config.native_gates.two_qubit_gates.contains(&NativeGateType::MS));
    }
    
    #[test]
    fn test_connectivity_check() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology).unwrap();
        
        // Adjacent qubits should be connected
        assert!(compiler.check_connectivity(QubitId::new(0), QubitId::new(1)).unwrap());
        assert!(compiler.check_connectivity(QubitId::new(1), QubitId::new(2)).unwrap());
        
        // Non-adjacent qubits should not be connected
        assert!(!compiler.check_connectivity(QubitId::new(0), QubitId::new(2)).unwrap());
        assert!(!compiler.check_connectivity(QubitId::new(0), QubitId::new(3)).unwrap());
    }
    
    #[test]
    fn test_shortest_path_finding() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology).unwrap();
        
        // Path between adjacent qubits
        let path = compiler.find_shortest_path(QubitId::new(0), QubitId::new(1)).unwrap();
        assert_eq!(path, vec![QubitId::new(0), QubitId::new(1)]);
        
        // Path between distant qubits
        let path = compiler.find_shortest_path(QubitId::new(0), QubitId::new(3)).unwrap();
        assert_eq!(path, vec![QubitId::new(0), QubitId::new(1), QubitId::new(2), QubitId::new(3)]);
    }
    
    #[test]
    fn test_virtual_z_optimization() {
        let optimizer = SuperconductingOptimizer::new();
        
        let gates = vec![
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.5],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            },
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.3],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            },
            CompiledGate {
                gate_type: NativeGateType::Rx,
                qubits: vec![QubitId::new(0)],
                parameters: vec![1.0],
                fidelity: 0.999,
                duration: Duration::from_nanos(20),
                pulse_sequence: None,
            },
        ];
        
        let optimized = optimizer.fuse_virtual_z_gates(&gates).unwrap();
        assert_eq!(optimized.len(), 2); // Virtual Z gates should be fused
        assert_eq!(optimized[0].gate_type, NativeGateType::VirtualZ);
        assert!((optimized[0].parameters[0] - 0.8).abs() < 1e-10); // 0.5 + 0.3 = 0.8
    }
    
    #[test]
    fn test_gate_fidelity_calculation() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology).unwrap();
        
        // Virtual Z gates should have perfect fidelity
        assert_eq!(compiler.get_gate_fidelity(NativeGateType::VirtualZ), 1.0);
        
        // Single-qubit gates should have high fidelity
        assert!(compiler.get_gate_fidelity(NativeGateType::Rx) > 0.999);
        
        // Two-qubit gates should have lower fidelity
        assert!(compiler.get_gate_fidelity(NativeGateType::CNOT) < compiler.get_gate_fidelity(NativeGateType::Rx));
    }
    
    #[test]
    fn test_platform_constraints() {
        let superconducting_optimizer = SuperconductingOptimizer::new();
        let constraints = superconducting_optimizer.get_constraints();
        
        assert!(constraints.max_qubits >= 100); // Should support many qubits
        assert!(constraints.timing_constraints.min_gate_separation < Duration::from_micros(1));
    }
    
    #[test]
    fn test_compilation_performance_tracking() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology).unwrap();
        
        // Simulate some compilation times
        compiler.record_compilation_time(Duration::from_millis(10));
        compiler.record_compilation_time(Duration::from_millis(15));
        compiler.record_compilation_time(Duration::from_millis(12));
        
        let stats = compiler.get_performance_stats();
        assert_eq!(stats.total_compilations, 3);
        assert!(stats.average_compilation_time > Duration::from_millis(10));
        assert!(stats.average_compilation_time < Duration::from_millis(15));
    }
    
    #[test]
    fn test_cache_functionality() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology).unwrap();
        
        let test_gates = vec![
            CompiledGate {
                gate_type: NativeGateType::Rx,
                qubits: vec![QubitId::new(0)],
                parameters: vec![1.0],
                fidelity: 0.999,
                duration: Duration::from_nanos(20),
                pulse_sequence: None,
            }
        ];
        
        // Cache a result
        let cache_key = "test_gate_0";
        compiler.cache_result(cache_key, &test_gates).unwrap();
        
        // Retrieve from cache
        let cached_result = compiler.check_cache(cache_key).unwrap();
        assert!(cached_result.is_some());
        
        let cached_gates = cached_result.unwrap();
        assert_eq!(cached_gates.len(), 1);
        assert_eq!(cached_gates[0].gate_type, NativeGateType::Rx);
    }
    
    #[test]
    fn test_z_rotation_detection() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology).unwrap();
        
        // Create a Z rotation matrix
        let angle = std::f64::consts::PI / 4.0;
        let mut z_matrix = Array2::zeros((2, 2));
        z_matrix[(0, 0)] = Complex64::from_polar(1.0, -angle / 2.0);
        z_matrix[(1, 1)] = Complex64::from_polar(1.0, angle / 2.0);
        
        let dense_z_matrix = DenseMatrix::new(z_matrix).unwrap();
        assert!(compiler.is_z_rotation(&dense_z_matrix).unwrap());
        
        let extracted_angle = compiler.extract_z_rotation_angle(&dense_z_matrix).unwrap();
        assert!((extracted_angle - angle).abs() < 1e-10);
    }
    
    #[test]
    fn test_euler_angle_extraction() {
        let topology = create_test_topology();
        let compiler = HardwareCompiler::for_superconducting(topology).unwrap();
        
        // Create identity matrix
        let mut identity = Array2::zeros((2, 2));
        identity[(0, 0)] = Complex64::new(1.0, 0.0);
        identity[(1, 1)] = Complex64::new(1.0, 0.0);
        
        let dense_identity = DenseMatrix::new(identity).unwrap();
        let (theta, phi, lambda) = compiler.extract_euler_angles(&dense_identity).unwrap();
        
        // For identity matrix, theta should be close to 0
        assert!(theta.abs() < 1e-10);
    }
    
    #[test]
    fn test_optimization_metrics_calculation() {
        let optimizer = SuperconductingOptimizer::new();
        
        let original_gates = vec![
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.5],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            },
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.3],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            },
        ];
        
        let optimized_gates = vec![
            CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![QubitId::new(0)],
                parameters: vec![0.8],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            }
        ];
        
        let metrics = optimizer.calculate_metrics(&original_gates, &optimized_gates, 1.0);
        
        assert_eq!(metrics.original_gate_count, 2);
        assert_eq!(metrics.optimized_gate_count, 1);
        assert_eq!(metrics.gate_count_reduction, 50.0); // 50% reduction
    }
}