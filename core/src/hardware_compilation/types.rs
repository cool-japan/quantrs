//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    matrix_ops::DenseMatrix,
    pulse::PulseSequence,
    qubit::QubitId,
    synthesis::decompose_two_qubit_kak,
};
use scirs2_core::ndarray::Array2;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use super::functions::{
    create_superconducting_gate_set, create_trapped_ion_gate_set, PlatformOptimizer,
};

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
#[derive(Debug)]
pub struct NeutralAtomOptimizer;
impl NeutralAtomOptimizer {
    pub const fn new() -> Self {
        Self
    }
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
impl DecompositionCache {
    fn new() -> Self {
        Self {
            single_qubit_cache: HashMap::new(),
            two_qubit_cache: HashMap::new(),
            cache_stats: CacheStatistics::default(),
        }
    }
}
/// Hardware optimization engine
#[derive(Debug)]
pub struct HardwareOptimizationEngine {
    /// Platform-specific optimizers
    optimizers: HashMap<HardwarePlatform, Box<dyn PlatformOptimizer>>,
    /// Optimization history
    optimization_history: Vec<OptimizationRecord>,
}
impl HardwareOptimizationEngine {
    fn new(_config: &HardwareCompilationConfig) -> QuantRS2Result<Self> {
        let mut optimizers: HashMap<HardwarePlatform, Box<dyn PlatformOptimizer>> = HashMap::new();
        optimizers.insert(
            HardwarePlatform::Superconducting,
            Box::new(SuperconductingOptimizer::new()),
        );
        optimizers.insert(
            HardwarePlatform::TrappedIon,
            Box::new(TrappedIonOptimizer::new()),
        );
        optimizers.insert(
            HardwarePlatform::Photonic,
            Box::new(PhotonicOptimizer::new()),
        );
        optimizers.insert(
            HardwarePlatform::NeutralAtom,
            Box::new(NeutralAtomOptimizer::new()),
        );
        Ok(Self {
            optimizers,
            optimization_history: Vec::new(),
        })
    }
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
#[derive(Debug)]
pub struct TrappedIonOptimizer;
impl TrappedIonOptimizer {
    pub const fn new() -> Self {
        Self
    }
}
#[derive(Debug)]
pub struct PhotonicOptimizer;
impl PhotonicOptimizer {
    pub const fn new() -> Self {
        Self
    }
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
/// Native gate types for hardware platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeGateType {
    X,
    Y,
    Z,
    Rx,
    Ry,
    Rz,
    SqrtX,
    SqrtY,
    H,
    S,
    T,
    Phase,
    CNOT,
    CZ,
    CY,
    XX,
    YY,
    ZZ,
    MS,
    Iswap,
    SqrtIswap,
    Toffoli,
    Fredkin,
    GlobalPhase,
    VirtualZ,
    BeamSplitter,
    Rydberg,
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
/// Hardware-specific quantum compiler
#[derive(Debug)]
pub struct HardwareCompiler {
    pub config: HardwareCompilationConfig,
    decomposition_cache: Arc<RwLock<DecompositionCache>>,
    optimization_engine: Arc<RwLock<HardwareOptimizationEngine>>,
    performance_monitor: Arc<RwLock<CompilationPerformanceMonitor>>,
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
    pub fn compile_gate(
        &self,
        gate: &dyn GateOp,
        qubits: &[QubitId],
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        let start_time = Instant::now();
        let cache_key = self.generate_cache_key(gate, qubits);
        if let Some(cached_result) = self.check_cache(&cache_key)? {
            self.record_cache_hit();
            return Ok(cached_result);
        }
        self.record_cache_miss();
        let compiled_gates = match qubits.len() {
            1 => self.compile_single_qubit_gate(gate, qubits[0])?,
            2 => self.compile_two_qubit_gate(gate, qubits[0], qubits[1])?,
            _ => self.compile_multi_qubit_gate(gate, qubits)?,
        };
        let optimized_gates = self.optimize_for_platform(&compiled_gates)?;
        self.cache_result(&cache_key, &optimized_gates)?;
        let compilation_time = start_time.elapsed();
        self.record_compilation_time(compilation_time);
        Ok(optimized_gates)
    }
    /// Compile a single-qubit gate
    fn compile_single_qubit_gate(
        &self,
        gate: &dyn GateOp,
        qubit: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 2)?;
        if let Some(native_gate) = self.find_native_single_qubit_gate(&matrix)? {
            return Ok(vec![native_gate]);
        }
        match self.config.platform {
            HardwarePlatform::Superconducting => {
                self.decompose_for_superconducting_single(gate, qubit)
            }
            HardwarePlatform::TrappedIon => self.decompose_for_trapped_ion_single(gate, qubit),
            HardwarePlatform::Photonic => self.decompose_for_photonic_single(gate, qubit),
            HardwarePlatform::NeutralAtom => self.decompose_for_neutral_atom_single(gate, qubit),
            _ => self.decompose_universal_single(gate, qubit),
        }
    }
    /// Compile a two-qubit gate
    fn compile_two_qubit_gate(
        &self,
        gate: &dyn GateOp,
        qubit1: QubitId,
        qubit2: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        if !self.check_connectivity(qubit1, qubit2)? {
            return self.handle_connectivity_constraint(gate, qubit1, qubit2);
        }
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 4)?;
        if let Some(native_gate) = self.find_native_two_qubit_gate(&matrix, qubit1, qubit2)? {
            return Ok(vec![native_gate]);
        }
        match self.config.platform {
            HardwarePlatform::Superconducting => {
                self.decompose_for_superconducting_two(gate, qubit1, qubit2)
            }
            HardwarePlatform::TrappedIon => {
                self.decompose_for_trapped_ion_two(gate, qubit1, qubit2)
            }
            HardwarePlatform::Photonic => self.decompose_for_photonic_two(gate, qubit1, qubit2),
            HardwarePlatform::NeutralAtom => {
                self.decompose_for_neutral_atom_two(gate, qubit1, qubit2)
            }
            _ => self.decompose_universal_two(gate, qubit1, qubit2),
        }
    }
    /// Compile a multi-qubit gate
    fn compile_multi_qubit_gate(
        &self,
        gate: &dyn GateOp,
        qubits: &[QubitId],
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        if self.config.native_gates.multi_qubit_gates.is_empty() {
            return self.decompose_to_two_qubit_gates(gate, qubits);
        }
        match self.config.platform {
            HardwarePlatform::TrappedIon => self.compile_trapped_ion_multi(gate, qubits),
            HardwarePlatform::NeutralAtom => self.compile_neutral_atom_multi(gate, qubits),
            _ => self.decompose_to_two_qubit_gates(gate, qubits),
        }
    }
    /// Decompose for superconducting single-qubit gates
    fn decompose_for_superconducting_single(
        &self,
        gate: &dyn GateOp,
        qubit: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 2)?;
        if self.is_z_rotation(&matrix)? {
            let angle = self.extract_z_rotation_angle(&matrix)?;
            return Ok(vec![CompiledGate {
                gate_type: NativeGateType::VirtualZ,
                qubits: vec![qubit],
                parameters: vec![angle],
                fidelity: 1.0,
                duration: Duration::from_nanos(0),
                pulse_sequence: None,
            }]);
        }
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
    fn decompose_for_trapped_ion_single(
        &self,
        gate: &dyn GateOp,
        qubit: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 2)?;
        let (theta, phi, lambda) = self.extract_euler_angles(&matrix)?;
        let mut compiled_gates = Vec::new();
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
    fn decompose_for_superconducting_two(
        &self,
        gate: &dyn GateOp,
        qubit1: QubitId,
        qubit2: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 4)?;
        let kak_decomp = decompose_two_qubit_kak(&matrix.as_array().view())?;
        let mut compiled_gates = Vec::new();
        let (_left_gate1, _left_gate2) = &kak_decomp.left_gates;
        let interaction_strength = (kak_decomp.interaction.xx.abs()
            + kak_decomp.interaction.yy.abs()
            + kak_decomp.interaction.zz.abs())
        .max(0.01);
        let native_two_qubit = self.get_native_two_qubit_gate();
        if interaction_strength > 0.01 {
            compiled_gates.push(CompiledGate {
                gate_type: native_two_qubit,
                qubits: vec![qubit1, qubit2],
                parameters: vec![interaction_strength],
                fidelity: self.get_gate_fidelity(native_two_qubit),
                duration: self.get_gate_duration(native_two_qubit),
                pulse_sequence: self
                    .generate_pulse_sequence(native_two_qubit, &[interaction_strength])?,
            });
        }
        let (_right_gate1, _right_gate2) = &kak_decomp.right_gates;
        Ok(compiled_gates)
    }
    /// Decompose for trapped ion two-qubit gates
    fn decompose_for_trapped_ion_two(
        &self,
        gate: &dyn GateOp,
        qubit1: QubitId,
        qubit2: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        let matrix_vec = gate.matrix()?;
        let matrix = DenseMatrix::from_vec(matrix_vec, 4)?;
        let ms_decomp = self.decompose_to_ms_gates(&matrix)?;
        let mut compiled_gates = Vec::new();
        for ms_gate in ms_decomp {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::MS,
                qubits: vec![qubit1, qubit2],
                parameters: ms_gate.parameters.clone(),
                fidelity: self.get_gate_fidelity(NativeGateType::MS),
                duration: self.get_gate_duration(NativeGateType::MS),
                pulse_sequence: self
                    .generate_pulse_sequence(NativeGateType::MS, &ms_gate.parameters)?,
            });
        }
        Ok(compiled_gates)
    }
    /// Check qubit connectivity
    pub fn check_connectivity(&self, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<bool> {
        if let Some(neighbors) = self.config.topology.connectivity.get(&qubit1) {
            Ok(neighbors.contains(&qubit2))
        } else {
            Ok(false)
        }
    }
    /// Handle connectivity constraints by inserting SWAP gates
    fn handle_connectivity_constraint(
        &self,
        gate: &dyn GateOp,
        qubit1: QubitId,
        qubit2: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        let path = self.find_shortest_path(qubit1, qubit2)?;
        let mut compiled_gates = Vec::new();
        for i in 0..path.len() - 2 {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::CNOT,
                qubits: vec![path[i], path[i + 1]],
                parameters: vec![],
                fidelity: self.get_gate_fidelity(NativeGateType::CNOT).powi(3),
                duration: self.get_gate_duration(NativeGateType::CNOT) * 3,
                pulse_sequence: None,
            });
        }
        let original_gate_compiled =
            self.compile_two_qubit_gate(gate, path[path.len() - 2], path[path.len() - 1])?;
        compiled_gates.extend(original_gate_compiled);
        for i in (0..path.len() - 2).rev() {
            compiled_gates.push(CompiledGate {
                gate_type: NativeGateType::CNOT,
                qubits: vec![path[i], path[i + 1]],
                parameters: vec![],
                fidelity: self.get_gate_fidelity(NativeGateType::CNOT).powi(3),
                duration: self.get_gate_duration(NativeGateType::CNOT) * 3,
                pulse_sequence: None,
            });
        }
        Ok(compiled_gates)
    }
    /// Find shortest path between qubits
    pub fn find_shortest_path(&self, start: QubitId, end: QubitId) -> QuantRS2Result<Vec<QubitId>> {
        use std::collections::VecDeque;
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(current) = queue.pop_front() {
            if current == end {
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
                    if visited.insert(neighbor) {
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        Err(QuantRS2Error::InvalidParameter(format!(
            "No path found between qubits {start:?} and {end:?}"
        )))
    }
    /// Optimize compiled gates for the target platform
    fn optimize_for_platform(&self, gates: &[CompiledGate]) -> QuantRS2Result<Vec<CompiledGate>> {
        let engine = self
            .optimization_engine
            .read()
            .map_err(|e| QuantRS2Error::RuntimeError(format!("Lock poisoned: {e}")))?;
        if let Some(optimizer) = engine.optimizers.get(&self.config.platform) {
            let optimized = optimizer.optimize_sequence(gates, &self.config)?;
            Ok(optimized.gates)
        } else {
            Ok(gates.to_vec())
        }
    }
    /// Helper methods for gate property extraction
    pub fn is_z_rotation(&self, matrix: &DenseMatrix) -> QuantRS2Result<bool> {
        let tolerance = self.config.tolerances.decomposition_tolerance;
        let arr = matrix.as_array();
        if arr[(0, 1)].norm() > tolerance || arr[(1, 0)].norm() > tolerance {
            return Ok(false);
        }
        Ok(true)
    }
    pub fn extract_z_rotation_angle(&self, matrix: &DenseMatrix) -> QuantRS2Result<f64> {
        let arr = matrix.as_array();
        let z00 = arr[(0, 0)];
        let z11 = arr[(1, 1)];
        let angle = (z11 / z00).arg();
        Ok(angle)
    }
    fn decompose_to_rx_rz(
        &self,
        matrix: &DenseMatrix,
    ) -> QuantRS2Result<Vec<(NativeGateType, f64)>> {
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
    pub fn extract_euler_angles(&self, matrix: &DenseMatrix) -> QuantRS2Result<(f64, f64, f64)> {
        let arr = matrix.as_array();
        let u00 = arr[(0, 0)];
        let u01 = arr[(0, 1)];
        let u10 = arr[(1, 0)];
        let u11 = arr[(1, 1)];
        let theta: f64 = 2.0 * (u01.norm()).asin();
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
    pub fn get_gate_fidelity(&self, gate_type: NativeGateType) -> f64 {
        self.config
            .native_gates
            .gate_fidelities
            .get(&gate_type)
            .copied()
            .unwrap_or(0.999)
    }
    fn get_gate_duration(&self, gate_type: NativeGateType) -> Duration {
        self.config
            .native_gates
            .gate_durations
            .get(&gate_type)
            .copied()
            .unwrap_or(Duration::from_nanos(100))
    }
    const fn get_native_two_qubit_gate(&self) -> NativeGateType {
        match self.config.platform {
            HardwarePlatform::TrappedIon => NativeGateType::MS,
            HardwarePlatform::Photonic | HardwarePlatform::NeutralAtom => NativeGateType::CZ,
            HardwarePlatform::Superconducting | _ => NativeGateType::CNOT,
        }
    }
    /// Generate cache key for gate and qubits
    fn generate_cache_key(&self, gate: &dyn GateOp, qubits: &[QubitId]) -> String {
        format!("{}_{:?}", gate.name(), qubits)
    }
    /// Utility methods for other decompositions and optimizations
    const fn find_native_single_qubit_gate(
        &self,
        _matrix: &DenseMatrix,
    ) -> QuantRS2Result<Option<CompiledGate>> {
        Ok(None)
    }
    const fn find_native_two_qubit_gate(
        &self,
        _matrix: &DenseMatrix,
        _qubit1: QubitId,
        _qubit2: QubitId,
    ) -> QuantRS2Result<Option<CompiledGate>> {
        Ok(None)
    }
    fn decompose_for_photonic_single(
        &self,
        _gate: &dyn GateOp,
        _qubit: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    fn decompose_for_neutral_atom_single(
        &self,
        _gate: &dyn GateOp,
        _qubit: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    fn decompose_universal_single(
        &self,
        _gate: &dyn GateOp,
        _qubit: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    fn decompose_for_photonic_two(
        &self,
        _gate: &dyn GateOp,
        _qubit1: QubitId,
        _qubit2: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    fn decompose_for_neutral_atom_two(
        &self,
        _gate: &dyn GateOp,
        _qubit1: QubitId,
        _qubit2: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    fn decompose_universal_two(
        &self,
        _gate: &dyn GateOp,
        _qubit1: QubitId,
        _qubit2: QubitId,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    fn compile_trapped_ion_multi(
        &self,
        _gate: &dyn GateOp,
        _qubits: &[QubitId],
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    fn compile_neutral_atom_multi(
        &self,
        _gate: &dyn GateOp,
        _qubits: &[QubitId],
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    fn decompose_to_two_qubit_gates(
        &self,
        _gate: &dyn GateOp,
        _qubits: &[QubitId],
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    const fn decompose_to_ms_gates(
        &self,
        _matrix: &DenseMatrix,
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        Ok(vec![])
    }
    const fn generate_pulse_sequence(
        &self,
        gate_type: NativeGateType,
        parameters: &[f64],
    ) -> QuantRS2Result<Option<PulseSequence>> {
        if !self.config.use_pulse_optimization {
            return Ok(None);
        }
        match self.config.platform {
            HardwarePlatform::Superconducting => {
                self.generate_superconducting_pulses(gate_type, parameters)
            }
            HardwarePlatform::TrappedIon => self.generate_trapped_ion_pulses(gate_type, parameters),
            _ => Ok(None),
        }
    }
    const fn generate_superconducting_pulses(
        &self,
        _gate_type: NativeGateType,
        _parameters: &[f64],
    ) -> QuantRS2Result<Option<PulseSequence>> {
        Ok(None)
    }
    const fn generate_trapped_ion_pulses(
        &self,
        _gate_type: NativeGateType,
        _parameters: &[f64],
    ) -> QuantRS2Result<Option<PulseSequence>> {
        Ok(None)
    }
    pub fn check_cache(&self, key: &str) -> QuantRS2Result<Option<Vec<CompiledGate>>> {
        let cache = self
            .decomposition_cache
            .read()
            .map_err(|e| QuantRS2Error::RuntimeError(format!("Cache lock poisoned: {e}")))?;
        Ok(cache.single_qubit_cache.get(key).cloned())
    }
    pub fn cache_result(&self, key: &str, gates: &[CompiledGate]) -> QuantRS2Result<()> {
        let mut cache = self
            .decomposition_cache
            .write()
            .map_err(|e| QuantRS2Error::RuntimeError(format!("Cache lock poisoned: {e}")))?;
        cache
            .single_qubit_cache
            .insert(key.to_string(), gates.to_vec());
        cache.cache_stats.total_requests += 1;
        Ok(())
    }
    fn record_cache_hit(&self) {
        if let Ok(mut cache) = self.decomposition_cache.write() {
            cache.cache_stats.cache_hits += 1;
            cache.cache_stats.hit_rate =
                cache.cache_stats.cache_hits as f64 / cache.cache_stats.total_requests as f64;
        }
    }
    fn record_cache_miss(&self) {
        if let Ok(mut cache) = self.decomposition_cache.write() {
            cache.cache_stats.cache_misses += 1;
        }
    }
    pub fn record_compilation_time(&self, duration: Duration) {
        if let Ok(mut monitor) = self.performance_monitor.write() {
            monitor.compilation_times.push(duration);
        }
    }
    /// Get compilation performance statistics
    pub fn get_performance_stats(&self) -> CompilationPerformanceStats {
        let monitor = self
            .performance_monitor
            .read()
            .expect("performance monitor lock poisoned");
        let cache = self
            .decomposition_cache
            .read()
            .expect("cache lock poisoned");
        let avg_time = if monitor.compilation_times.is_empty() {
            Duration::ZERO
        } else {
            monitor.compilation_times.iter().sum::<Duration>()
                / monitor.compilation_times.len() as u32
        };
        CompilationPerformanceStats {
            average_compilation_time: avg_time,
            cache_statistics: cache.cache_stats.clone(),
            total_compilations: monitor.compilation_times.len(),
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
            optimization_objectives: vec![
                OptimizationObjective::MinimizeGateCount,
                OptimizationObjective::MaximizeFidelity,
            ],
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
            optimization_objectives: vec![
                OptimizationObjective::MinimizeTime,
                OptimizationObjective::MaximizeFidelity,
            ],
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
impl CompilationPerformanceMonitor {
    const fn new() -> Self {
        Self {
            compilation_times: Vec::new(),
            gate_count_reductions: Vec::new(),
            fidelity_improvements: Vec::new(),
            cache_hit_rates: Vec::new(),
        }
    }
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
#[derive(Debug)]
pub struct SuperconductingOptimizer;
impl SuperconductingOptimizer {
    pub const fn new() -> Self {
        Self
    }
}
impl SuperconductingOptimizer {
    pub fn fuse_virtual_z_gates(
        &self,
        gates: &[CompiledGate],
    ) -> QuantRS2Result<Vec<CompiledGate>> {
        let mut optimized = Vec::new();
        let mut current_z_angle = 0.0;
        let mut current_qubit = None;
        for gate in gates {
            if gate.gate_type == NativeGateType::VirtualZ {
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
                    current_qubit = None;
                    current_z_angle = 0.0;
                }
                optimized.push(gate.clone());
            }
        }
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
    pub(super) fn calculate_metrics(
        &self,
        original: &[CompiledGate],
        optimized: &[CompiledGate],
        fidelity: f64,
    ) -> OptimizationMetrics {
        OptimizationMetrics {
            original_gate_count: original.len(),
            optimized_gate_count: optimized.len(),
            gate_count_reduction: (original.len() - optimized.len()) as f64 / original.len() as f64
                * 100.0,
            original_depth: original.len(),
            optimized_depth: optimized.len(),
            depth_reduction: 0.0,
            fidelity_improvement: fidelity,
            compilation_time: Duration::from_millis(1),
        }
    }
}
