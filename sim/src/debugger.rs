//! Quantum algorithm debugger interface.
//!
//! This module provides comprehensive debugging capabilities for quantum algorithms,
//! including step-by-step execution, state inspection, breakpoints, and analysis tools.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{Result, SimulatorError};
#[cfg(feature = "mps")]
use crate::mps_enhanced::{EnhancedMPS, MPSConfig};
use crate::statevector::StateVectorSimulator;
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::gate::GateOp;

// Placeholder for MPSConfig when MPS feature is disabled
#[cfg(not(feature = "mps"))]
#[derive(Debug, Clone, Default)]
pub struct MPSConfig {
    pub max_bond_dim: usize,
    pub tolerance: f64,
}

/// Breakpoint condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakCondition {
    /// Break at specific gate index
    GateIndex(usize),
    /// Break when a qubit reaches a certain state
    QubitState { qubit: usize, state: bool },
    /// Break when entanglement entropy exceeds threshold
    EntanglementThreshold { cut: usize, threshold: f64 },
    /// Break when fidelity with target state drops below threshold
    FidelityThreshold {
        target_state: Vec<Complex64>,
        threshold: f64,
    },
    /// Break when a Pauli observable expectation value crosses threshold
    ObservableThreshold {
        observable: String,
        threshold: f64,
        direction: ThresholdDirection,
    },
    /// Break when circuit depth exceeds limit
    CircuitDepth(usize),
    /// Break when execution time exceeds limit
    ExecutionTime(Duration),
}

/// Threshold crossing direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdDirection {
    Above,
    Below,
    Either,
}

/// Execution snapshot at a specific point
#[derive(Debug, Clone)]
pub struct ExecutionSnapshot {
    /// Gate index in the circuit
    pub gate_index: usize,
    /// Current quantum state
    pub state: Array1<Complex64>,
    /// Timestamp
    pub timestamp: Instant,
    /// Gate that was just executed (None for initial state)
    pub last_gate: Option<Arc<dyn GateOp + Send + Sync>>,
    /// Cumulative gate count by type
    pub gate_counts: HashMap<String, usize>,
    /// Entanglement entropies at different cuts
    pub entanglement_entropies: Vec<f64>,
    /// Circuit depth so far
    pub circuit_depth: usize,
}

/// Performance metrics during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total execution time
    pub total_time: Duration,
    /// Time per gate type
    pub gate_times: HashMap<String, Duration>,
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
    /// Gate execution counts
    pub gate_counts: HashMap<String, usize>,
    /// Average entanglement entropy
    pub avg_entanglement: f64,
    /// Maximum entanglement entropy reached
    pub max_entanglement: f64,
    /// Number of snapshots taken
    pub snapshot_count: usize,
}

/// Memory usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Peak state vector memory (bytes)
    pub peak_statevector_memory: usize,
    /// Current MPS bond dimensions
    pub mps_bond_dims: Vec<usize>,
    /// Peak MPS memory (bytes)
    pub peak_mps_memory: usize,
    /// Debugger overhead (bytes)
    pub debugger_overhead: usize,
}

/// Watchpoint for monitoring specific properties
#[derive(Debug, Clone)]
pub struct Watchpoint {
    /// Unique identifier
    pub id: String,
    /// Description
    pub description: String,
    /// Property to watch
    pub property: WatchProperty,
    /// Logging frequency
    pub frequency: WatchFrequency,
    /// History of watched values
    pub history: VecDeque<(usize, f64)>, // (gate_index, value)
}

/// Properties that can be watched
#[derive(Debug, Clone)]
pub enum WatchProperty {
    /// Total probability (should be 1)
    Normalization,
    /// Entanglement entropy at specific cut
    EntanglementEntropy(usize),
    /// Expectation value of Pauli observable
    PauliExpectation(String),
    /// Fidelity with reference state
    Fidelity(Array1<Complex64>),
    /// Average gate fidelity
    GateFidelity,
    /// Circuit depth
    CircuitDepth,
    /// MPS bond dimension
    MPSBondDimension,
}

/// Watch frequency
#[derive(Debug, Clone)]
pub enum WatchFrequency {
    /// Watch at every gate
    EveryGate,
    /// Watch every N gates
    EveryNGates(usize),
    /// Watch at specific gate indices
    AtGates(HashSet<usize>),
}

/// Debugging session configuration
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Whether to store full state snapshots
    pub store_snapshots: bool,
    /// Maximum number of snapshots to keep
    pub max_snapshots: usize,
    /// Whether to track performance metrics
    pub track_performance: bool,
    /// Whether to enable automatic state validation
    pub validate_state: bool,
    /// Entanglement entropy cut positions to monitor
    pub entropy_cuts: Vec<usize>,
    /// Use MPS representation for large systems
    pub use_mps: bool,
    /// MPS configuration if used
    pub mps_config: Option<MPSConfig>,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            store_snapshots: true,
            max_snapshots: 100,
            track_performance: true,
            validate_state: true,
            entropy_cuts: vec![],
            use_mps: false,
            mps_config: None,
        }
    }
}

/// Main quantum algorithm debugger
pub struct QuantumDebugger<const N: usize> {
    /// Configuration
    config: DebugConfig,
    /// Current circuit being debugged
    circuit: Option<Circuit<N>>,
    /// Active breakpoints
    breakpoints: Vec<BreakCondition>,
    /// Active watchpoints
    watchpoints: HashMap<String, Watchpoint>,
    /// Execution snapshots
    snapshots: VecDeque<ExecutionSnapshot>,
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Current execution state
    execution_state: ExecutionState,
    /// State vector simulator
    simulator: StateVectorSimulator,
    /// MPS simulator (if enabled)
    #[cfg(feature = "mps")]
    mps_simulator: Option<EnhancedMPS>,
    /// Current gate index
    current_gate: usize,
    /// Execution start time
    start_time: Option<Instant>,
}

/// Current execution state
#[derive(Debug, Clone)]
enum ExecutionState {
    /// Not running
    Idle,
    /// Running normally
    Running,
    /// Paused at breakpoint
    Paused { reason: String },
    /// Finished execution
    Finished,
    /// Error occurred
    Error { message: String },
}

impl<const N: usize> QuantumDebugger<N> {
    /// Create a new quantum debugger
    pub fn new(config: DebugConfig) -> Result<Self> {
        let simulator = StateVectorSimulator::new();

        #[cfg(feature = "mps")]
        let mps_simulator = if config.use_mps {
            Some(EnhancedMPS::new(
                N,
                config.mps_config.clone().unwrap_or_default(),
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            circuit: None,
            breakpoints: Vec::new(),
            watchpoints: HashMap::new(),
            snapshots: VecDeque::new(),
            metrics: PerformanceMetrics {
                total_time: Duration::new(0, 0),
                gate_times: HashMap::new(),
                memory_usage: MemoryUsage {
                    peak_statevector_memory: 0,
                    mps_bond_dims: vec![],
                    peak_mps_memory: 0,
                    debugger_overhead: 0,
                },
                gate_counts: HashMap::new(),
                avg_entanglement: 0.0,
                max_entanglement: 0.0,
                snapshot_count: 0,
            },
            execution_state: ExecutionState::Idle,
            simulator,
            #[cfg(feature = "mps")]
            mps_simulator,
            current_gate: 0,
            start_time: None,
        })
    }

    /// Load a circuit for debugging
    pub fn load_circuit(&mut self, circuit: Circuit<N>) -> Result<()> {
        self.circuit = Some(circuit);
        self.reset();
        Ok(())
    }

    /// Reset debugger state
    pub fn reset(&mut self) {
        self.snapshots.clear();
        self.metrics = PerformanceMetrics {
            total_time: Duration::new(0, 0),
            gate_times: HashMap::new(),
            memory_usage: MemoryUsage {
                peak_statevector_memory: 0,
                mps_bond_dims: vec![],
                peak_mps_memory: 0,
                debugger_overhead: 0,
            },
            gate_counts: HashMap::new(),
            avg_entanglement: 0.0,
            max_entanglement: 0.0,
            snapshot_count: 0,
        };
        self.execution_state = ExecutionState::Idle;
        self.current_gate = 0;
        self.start_time = None;

        // Reset simulator to |0...0> state
        self.simulator = StateVectorSimulator::new();
        #[cfg(feature = "mps")]
        if let Some(ref mut mps) = self.mps_simulator {
            *mps = EnhancedMPS::new(N, self.config.mps_config.clone().unwrap_or_default());
        }

        // Clear watchpoint histories
        for watchpoint in self.watchpoints.values_mut() {
            watchpoint.history.clear();
        }
    }

    /// Add a breakpoint
    pub fn add_breakpoint(&mut self, condition: BreakCondition) {
        self.breakpoints.push(condition);
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, index: usize) -> Result<()> {
        if index >= self.breakpoints.len() {
            return Err(SimulatorError::IndexOutOfBounds(index));
        }
        self.breakpoints.remove(index);
        Ok(())
    }

    /// Add a watchpoint
    pub fn add_watchpoint(&mut self, watchpoint: Watchpoint) {
        self.watchpoints.insert(watchpoint.id.clone(), watchpoint);
    }

    /// Remove a watchpoint
    pub fn remove_watchpoint(&mut self, id: &str) -> Result<()> {
        if self.watchpoints.remove(id).is_none() {
            return Err(SimulatorError::InvalidInput(format!(
                "Watchpoint '{id}' not found"
            )));
        }
        Ok(())
    }

    /// Execute the circuit step by step
    pub fn step(&mut self) -> Result<StepResult> {
        let circuit = self
            .circuit
            .as_ref()
            .ok_or_else(|| SimulatorError::InvalidOperation("No circuit loaded".to_string()))?;

        if self.current_gate >= circuit.gates().len() {
            self.execution_state = ExecutionState::Finished;
            return Ok(StepResult::Finished);
        }

        // Check if we're paused
        if let ExecutionState::Paused { .. } = self.execution_state {
            // Continue from pause
            self.execution_state = ExecutionState::Running;
        }

        // Start timing if first step
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
            self.execution_state = ExecutionState::Running;
        }

        // Get gate information before borrowing mutably
        let gate_name = circuit.gates()[self.current_gate].name().to_string();
        let total_gates = circuit.gates().len();

        // Execute the current gate
        let gate_start = Instant::now();

        // Apply gate to appropriate simulator.
        //
        // The MPS backend evolves incrementally and is mutated in place here. The
        // state-vector backend, by contrast, does not need per-gate mutation: its
        // current amplitudes are reconstructed on demand by replaying the executed
        // prefix (gates `0..current_gate`) in `get_current_state`. Advancing
        // `self.current_gate` below is therefore sufficient to expose the correct
        // state for the state-vector path.
        #[cfg(feature = "mps")]
        if let Some(ref mut mps) = self.mps_simulator {
            mps.apply_gate(circuit.gates()[self.current_gate].as_ref())?;
        }

        let gate_time = gate_start.elapsed();

        // Update metrics
        *self
            .metrics
            .gate_times
            .entry(gate_name.clone())
            .or_insert(Duration::new(0, 0)) += gate_time;
        *self.metrics.gate_counts.entry(gate_name).or_insert(0) += 1;

        // Check watchpoints
        self.update_watchpoints()?;

        // Take snapshot if configured
        if self.config.store_snapshots {
            self.take_snapshot()?;
        }

        // Check breakpoints
        if let Some(reason) = self.check_breakpoints()? {
            self.execution_state = ExecutionState::Paused {
                reason: reason.clone(),
            };
            return Ok(StepResult::BreakpointHit { reason });
        }

        self.current_gate += 1;

        if self.current_gate >= total_gates {
            self.execution_state = ExecutionState::Finished;
            if let Some(start) = self.start_time {
                self.metrics.total_time = start.elapsed();
            }
            Ok(StepResult::Finished)
        } else {
            Ok(StepResult::Continue)
        }
    }

    /// Run until next breakpoint or completion
    pub fn run(&mut self) -> Result<StepResult> {
        loop {
            match self.step()? {
                StepResult::Continue => {}
                result => return Ok(result),
            }
        }
    }

    /// Get current quantum state
    ///
    /// Returns the full `2^N` amplitude vector after the gates that have been
    /// executed so far (`self.current_gate` gates from the loaded circuit).
    ///
    /// The state is reconstructed by replaying the executed prefix of the
    /// circuit through the embedded [`StateVectorSimulator`]. This always
    /// reflects the real amplitudes; if no circuit is loaded, the simulator
    /// is in the initial `|0…0⟩` state, which is returned as a genuine state
    /// vector (amplitude 1 on the zero basis state) rather than a fabricated
    /// all-zero vector.
    pub fn get_current_state(&self) -> Result<Array1<Complex64>> {
        #[cfg(feature = "mps")]
        if let Some(ref mps) = self.mps_simulator {
            return mps
                .to_statevector()
                .map_err(|e| SimulatorError::UnsupportedOperation(format!("MPS error: {e}")));
        }

        self.compute_statevector_prefix()
    }

    /// Reconstruct the state-vector amplitudes for the executed circuit prefix.
    ///
    /// Builds a circuit containing only the first `self.current_gate` gates and
    /// runs it through the embedded [`StateVectorSimulator`], returning the real
    /// amplitudes. When no circuit is loaded the result is the initial `|0…0⟩`
    /// state.
    fn compute_statevector_prefix(&self) -> Result<Array1<Complex64>> {
        let dim = 1_usize << N;

        let Some(circuit) = self.circuit.as_ref() else {
            // No circuit loaded: the simulator sits in the |0…0⟩ state.
            let mut amplitudes = Array1::zeros(dim);
            amplitudes[0] = Complex64::new(1.0, 0.0);
            return Ok(amplitudes);
        };

        // Replay only the gates that have already been executed. We clone the
        // shared `Arc` handles (cheap, no gate cloning) into a fresh prefix
        // circuit so the simulator can evolve |0…0⟩ up to the current step.
        let gates = circuit.gates();
        let executed = self.current_gate.min(gates.len());

        let mut prefix: Circuit<N> = Circuit::with_capacity(executed);
        for gate in &gates[..executed] {
            prefix.add_gate_arc(Arc::clone(gate))?;
        }

        let register = self.simulator.run(&prefix)?;
        Ok(Array1::from(register.amplitudes().to_vec()))
    }

    /// Get entanglement entropy at the specified bipartition cut.
    ///
    /// The cut splits the qubits into two contiguous groups; `cut` is the number
    /// of qubits on one side of the partition. The von Neumann entropy (in nats,
    /// consistent with the MPS path) of the reduced density matrix is returned.
    ///
    /// Both the state-vector and MPS backends are handled through
    /// [`Self::get_current_state`], which yields the genuine amplitudes for the
    /// current step (the MPS backend is contracted to a state vector via its
    /// `to_statevector` method). This computes the real entropy in every case
    /// rather than returning a placeholder.
    pub fn get_entanglement_entropy(&self, cut: usize) -> Result<f64> {
        let state = self.get_current_state()?;
        compute_entanglement_entropy(&state, cut, N)
    }

    /// Get expectation value of Pauli observable
    pub fn get_pauli_expectation(&self, pauli_string: &str) -> Result<Complex64> {
        #[cfg(feature = "mps")]
        if let Some(ref mps) = self.mps_simulator {
            return mps
                .expectation_value_pauli(pauli_string)
                .map_err(|e| SimulatorError::UnsupportedOperation(format!("MPS error: {e}")));
        }

        let state = self.get_current_state()?;
        compute_pauli_expectation(&state, pauli_string)
    }

    /// Get performance metrics
    pub const fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Get all snapshots
    pub const fn get_snapshots(&self) -> &VecDeque<ExecutionSnapshot> {
        &self.snapshots
    }

    /// Get watchpoint by ID
    pub fn get_watchpoint(&self, id: &str) -> Option<&Watchpoint> {
        self.watchpoints.get(id)
    }

    /// Get all watchpoints
    pub const fn get_watchpoints(&self) -> &HashMap<String, Watchpoint> {
        &self.watchpoints
    }

    /// Check if execution is finished
    pub const fn is_finished(&self) -> bool {
        matches!(self.execution_state, ExecutionState::Finished)
    }

    /// Check if execution is paused
    pub const fn is_paused(&self) -> bool {
        matches!(self.execution_state, ExecutionState::Paused { .. })
    }

    /// Get current execution state
    pub const fn get_execution_state(&self) -> &ExecutionState {
        &self.execution_state
    }

    /// Generate debugging report
    pub fn generate_report(&self) -> DebugReport {
        DebugReport {
            circuit_summary: self.circuit.as_ref().map(|c| CircuitSummary {
                total_gates: c.gates().len(),
                gate_types: self.metrics.gate_counts.clone(),
                estimated_depth: estimate_circuit_depth(c),
            }),
            performance: self.metrics.clone(),
            entanglement_analysis: self.analyze_entanglement(),
            state_analysis: self.analyze_state(),
            recommendations: self.generate_recommendations(),
        }
    }

    // Private helper methods

    fn take_snapshot(&mut self) -> Result<()> {
        if self.snapshots.len() >= self.config.max_snapshots {
            self.snapshots.pop_front();
        }

        let circuit = self.circuit.as_ref().ok_or_else(|| {
            SimulatorError::InvalidOperation("No circuit loaded for snapshot".to_string())
        })?;
        let state = self.get_current_state()?;

        let snapshot = ExecutionSnapshot {
            gate_index: self.current_gate,
            state,
            timestamp: Instant::now(),
            last_gate: if self.current_gate > 0 {
                Some(circuit.gates()[self.current_gate - 1].clone())
            } else {
                None
            },
            gate_counts: self.metrics.gate_counts.clone(),
            entanglement_entropies: self.compute_all_entanglement_entropies()?,
            circuit_depth: self.current_gate, // Simplified
        };

        self.snapshots.push_back(snapshot);
        self.metrics.snapshot_count += 1;
        Ok(())
    }

    fn check_breakpoints(&self) -> Result<Option<String>> {
        for breakpoint in &self.breakpoints {
            match breakpoint {
                BreakCondition::GateIndex(target) => {
                    if self.current_gate == *target {
                        return Ok(Some(format!("Reached gate index {target}")));
                    }
                }
                BreakCondition::EntanglementThreshold { cut, threshold } => {
                    let entropy = self.get_entanglement_entropy(*cut)?;
                    if entropy > *threshold {
                        return Ok(Some(format!(
                            "Entanglement entropy {entropy:.4} > {threshold:.4} at cut {cut}"
                        )));
                    }
                }
                BreakCondition::ObservableThreshold {
                    observable,
                    threshold,
                    direction,
                } => {
                    let expectation = self.get_pauli_expectation(observable)?.re;
                    let hit = match direction {
                        ThresholdDirection::Above => expectation > *threshold,
                        ThresholdDirection::Below => expectation < *threshold,
                        ThresholdDirection::Either => (expectation - threshold).abs() < 1e-10,
                    };
                    if hit {
                        return Ok(Some(format!(
                            "Observable {observable} = {expectation:.4} crossed threshold {threshold:.4}"
                        )));
                    }
                }
                _ => {
                    // Other breakpoint types would be implemented here
                }
            }
        }
        Ok(None)
    }

    fn update_watchpoints(&mut self) -> Result<()> {
        let current_gate = self.current_gate;

        // Collect watchpoint updates to avoid borrowing issues
        let mut updates = Vec::new();

        for (id, watchpoint) in &self.watchpoints {
            let should_update = match &watchpoint.frequency {
                WatchFrequency::EveryGate => true,
                WatchFrequency::EveryNGates(n) => current_gate % n == 0,
                WatchFrequency::AtGates(gates) => gates.contains(&current_gate),
            };

            if should_update {
                let value = match &watchpoint.property {
                    WatchProperty::EntanglementEntropy(cut) => {
                        self.get_entanglement_entropy(*cut)?
                    }
                    WatchProperty::PauliExpectation(observable) => {
                        self.get_pauli_expectation(observable)?.re
                    }
                    WatchProperty::Normalization => {
                        let state = self.get_current_state()?;
                        state
                            .iter()
                            .map(scirs2_core::Complex::norm_sqr)
                            .sum::<f64>()
                    }
                    _ => 0.0, // Other properties would be implemented
                };

                updates.push((id.clone(), current_gate, value));
            }
        }

        // Apply updates
        for (id, gate, value) in updates {
            if let Some(watchpoint) = self.watchpoints.get_mut(&id) {
                watchpoint.history.push_back((gate, value));

                // Keep history size manageable
                if watchpoint.history.len() > 1000 {
                    watchpoint.history.pop_front();
                }
            }
        }

        Ok(())
    }

    fn compute_all_entanglement_entropies(&self) -> Result<Vec<f64>> {
        let mut entropies = Vec::new();
        for &cut in &self.config.entropy_cuts {
            // Only evaluate genuine bipartitions: `cut` qubits on the left and
            // `N - cut` on the right, both non-empty (1 <= cut <= N - 1).
            if cut >= 1 && cut < N {
                entropies.push(self.get_entanglement_entropy(cut)?);
            }
        }
        Ok(entropies)
    }

    const fn analyze_entanglement(&self) -> EntanglementAnalysis {
        // Analyze entanglement patterns from snapshots and watchpoints
        EntanglementAnalysis {
            max_entropy: self.metrics.max_entanglement,
            avg_entropy: self.metrics.avg_entanglement,
            entropy_evolution: Vec::new(), // Would be filled from watchpoint histories
        }
    }

    const fn analyze_state(&self) -> StateAnalysis {
        // Analyze quantum state properties
        StateAnalysis {
            is_separable: false,      // Would compute this
            schmidt_rank: 1,          // Would compute this
            participation_ratio: 1.0, // Would compute this
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze performance and suggest optimizations
        if self.metrics.max_entanglement > 3.0 {
            recommendations.push(
                "High entanglement detected. Consider using MPS simulation for better scaling."
                    .to_string(),
            );
        }

        if self.metrics.gate_counts.get("CNOT").unwrap_or(&0) > &50 {
            recommendations
                .push("Many CNOT gates detected. Consider gate optimization.".to_string());
        }

        recommendations
    }
}

/// Result of a debugging step
#[derive(Debug, Clone)]
pub enum StepResult {
    /// Continue execution
    Continue,
    /// Breakpoint was hit
    BreakpointHit { reason: String },
    /// Execution finished
    Finished,
}

/// Circuit summary for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitSummary {
    pub total_gates: usize,
    pub gate_types: HashMap<String, usize>,
    pub estimated_depth: usize,
}

/// Entanglement analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementAnalysis {
    pub max_entropy: f64,
    pub avg_entropy: f64,
    pub entropy_evolution: Vec<(usize, f64)>,
}

/// State analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateAnalysis {
    pub is_separable: bool,
    pub schmidt_rank: usize,
    pub participation_ratio: f64,
}

/// Complete debugging report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugReport {
    pub circuit_summary: Option<CircuitSummary>,
    pub performance: PerformanceMetrics,
    pub entanglement_analysis: EntanglementAnalysis,
    pub state_analysis: StateAnalysis,
    pub recommendations: Vec<String>,
}

// Helper functions

/// Compute the bipartite von Neumann entanglement entropy from a state vector.
///
/// The amplitude vector `|ψ⟩` is reshaped into a `2^cut × 2^(num_qubits - cut)`
/// matrix `M`. The Schmidt coefficients of the bipartition are the singular
/// values `σ_i` of `M`, and the reduced-density-matrix eigenvalues are `σ_i²`.
/// The von Neumann entropy is therefore `S = -Σ_i σ_i² ln σ_i²` (natural log,
/// matching the convention used by the MPS backend's `entanglement_entropy`).
///
/// The singular value decomposition is computed with [`scirs2_linalg`] (the
/// SciRS2 complex SVD, which diagonalizes `MᴴM`); `ndarray-linalg` is not used.
fn compute_entanglement_entropy(
    state: &Array1<Complex64>,
    cut: usize,
    num_qubits: usize,
) -> Result<f64> {
    // A valid bipartition needs at least one qubit on each side: `cut` qubits on
    // the left (1..=num_qubits-1) and `num_qubits - cut` on the right. Reject
    // cuts that would leave an empty subsystem or exceed the register.
    if num_qubits < 2 || cut == 0 || cut >= num_qubits {
        return Err(SimulatorError::IndexOutOfBounds(cut));
    }

    let left_dim = 1usize << cut;
    let right_dim = 1usize << (num_qubits - cut);

    // Reshape the state into the bipartite amplitude matrix M[i_left, i_right].
    let state_matrix =
        Array2::from_shape_vec((left_dim, right_dim), state.to_vec()).map_err(|_| {
            SimulatorError::DimensionMismatch("Invalid state vector dimension".to_string())
        })?;

    // Singular values via SciRS2 complex SVD. The squared singular values are the
    // Schmidt probabilities p_i = σ_i².
    let svd = scirs2_linalg::complex::decompositions::complex_svd(&state_matrix.view(), false)
        .map_err(|e| SimulatorError::LinalgError(format!("complex SVD failed: {e}")))?;

    let mut entropy = 0.0_f64;
    for &sigma in &svd.s {
        let p = sigma * sigma;
        // Skip vanishing Schmidt coefficients; lim_{p->0} p ln p = 0.
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }

    // Guard against tiny negative values from floating-point round-off.
    Ok(entropy.max(0.0))
}

/// Compute the expectation value `⟨ψ| P |ψ⟩` of a Pauli string from a state vector.
///
/// `pauli_string` is a sequence of `I`, `X`, `Y`, `Z` characters, one per qubit.
/// The leftmost character corresponds to the highest-index qubit (little-endian
/// basis ordering), matching the convention of the MPS backend's
/// `expectation_value_pauli`. The string length must equal the number of qubits.
///
/// This evaluates the real expectation value by applying the tensor-product
/// Pauli operator to the amplitude vector; it does not return a placeholder.
fn compute_pauli_expectation(state: &Array1<Complex64>, pauli_string: &str) -> Result<Complex64> {
    let dim = state.len();
    let num_qubits = dim.trailing_zeros() as usize;

    if dim != 1usize << num_qubits {
        return Err(SimulatorError::DimensionMismatch(format!(
            "State vector length {dim} is not a power of two"
        )));
    }

    if pauli_string.len() != num_qubits {
        return Err(SimulatorError::InvalidInput(format!(
            "Pauli string length {} doesn't match qubit count {num_qubits}",
            pauli_string.len()
        )));
    }

    let mut result = Complex64::new(0.0, 0.0);

    for (i, amplitude) in state.iter().enumerate() {
        let mut coeff = Complex64::new(1.0, 0.0);
        let mut target_state = i;

        // Leftmost character maps to the highest qubit, so iterate reversed to
        // pair character position with qubit index 0, 1, 2, …
        for (qubit, pauli_char) in pauli_string.chars().rev().enumerate() {
            let bit = (i >> qubit) & 1;
            match pauli_char {
                'I' => {}
                'X' => {
                    target_state ^= 1 << qubit;
                }
                'Y' => {
                    target_state ^= 1 << qubit;
                    coeff *= if bit == 0 {
                        Complex64::new(0.0, 1.0)
                    } else {
                        Complex64::new(0.0, -1.0)
                    };
                }
                'Z' => {
                    if bit == 1 {
                        coeff = -coeff;
                    }
                }
                other => {
                    return Err(SimulatorError::InvalidInput(format!(
                        "Invalid Pauli operator: {other}"
                    )));
                }
            }
        }

        result += amplitude.conj() * coeff * state[target_state];
    }

    Ok(result)
}

/// Estimate circuit depth
fn estimate_circuit_depth<const N: usize>(circuit: &Circuit<N>) -> usize {
    // Simplified depth estimation - would need proper dependency analysis
    circuit.gates().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let config = DebugConfig::default();
        let debugger: QuantumDebugger<3> =
            QuantumDebugger::new(config).expect("Failed to create debugger");
        assert!(matches!(debugger.execution_state, ExecutionState::Idle));
    }

    #[test]
    fn test_breakpoint_management() {
        let config = DebugConfig::default();
        let mut debugger: QuantumDebugger<3> =
            QuantumDebugger::new(config).expect("Failed to create debugger");

        debugger.add_breakpoint(BreakCondition::GateIndex(5));
        assert_eq!(debugger.breakpoints.len(), 1);

        debugger
            .remove_breakpoint(0)
            .expect("Failed to remove breakpoint");
        assert_eq!(debugger.breakpoints.len(), 0);
    }

    #[test]
    fn test_watchpoint_management() {
        let config = DebugConfig::default();
        let mut debugger: QuantumDebugger<3> =
            QuantumDebugger::new(config).expect("Failed to create debugger");

        let watchpoint = Watchpoint {
            id: "test".to_string(),
            description: "Test watchpoint".to_string(),
            property: WatchProperty::Normalization,
            frequency: WatchFrequency::EveryGate,
            history: VecDeque::new(),
        };

        debugger.add_watchpoint(watchpoint);
        assert!(debugger.get_watchpoint("test").is_some());

        debugger
            .remove_watchpoint("test")
            .expect("Failed to remove watchpoint");
        assert!(debugger.get_watchpoint("test").is_none());
    }

    /// Build a debugger with snapshots disabled and a Bell circuit loaded.
    fn bell_debugger() -> QuantumDebugger<2> {
        let config = DebugConfig {
            store_snapshots: false,
            ..DebugConfig::default()
        };
        let mut debugger: QuantumDebugger<2> =
            QuantumDebugger::new(config).expect("Failed to create debugger");

        let mut circuit: Circuit<2> = Circuit::new();
        circuit
            .bell_state(0, 1)
            .expect("Failed to build Bell state");
        debugger
            .load_circuit(circuit)
            .expect("Failed to load circuit");
        debugger
    }

    #[test]
    fn test_get_current_state_initial_is_zero_ket() {
        // With no gates executed yet, the state must be a *genuine* |00> state
        // (amplitude 1 on basis 0), not a fabricated all-zero vector.
        let debugger = bell_debugger();
        let state = debugger
            .get_current_state()
            .expect("Failed to get current state");

        assert_eq!(state.len(), 4);
        assert!((state[0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        for amp in state.iter().skip(1) {
            assert!(amp.norm() < 1e-12);
        }

        // It must NOT be the old fabricated all-zero "dummy" vector.
        let dummy: Array1<Complex64> = Array1::zeros(4);
        assert!(
            state != dummy,
            "state must not be the fabricated all-zero vector"
        );
    }

    #[test]
    fn test_get_current_state_bell_amplitudes() {
        // Execute H(0) then CNOT(0,1); the real Bell state is
        // (|00> + |11>)/sqrt(2) = (1/sqrt2, 0, 0, 1/sqrt2).
        let mut debugger = bell_debugger();
        debugger.step().expect("step 1 failed"); // apply H
        debugger.step().expect("step 2 failed"); // apply CNOT

        let state = debugger
            .get_current_state()
            .expect("Failed to get current state");

        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        assert!((state[0] - Complex64::new(inv_sqrt2, 0.0)).norm() < 1e-10);
        assert!(state[1].norm() < 1e-10);
        assert!(state[2].norm() < 1e-10);
        assert!((state[3] - Complex64::new(inv_sqrt2, 0.0)).norm() < 1e-10);

        // Norm must be 1 (real, normalized state).
        let norm_sq: f64 = state.iter().map(scirs2_core::Complex::norm_sqr).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_current_state_prefix_after_first_gate() {
        // After only H(0), the state is (|00> + |01>)/sqrt2 = (1/sqrt2, 1/sqrt2, 0, 0).
        // Qubit 0 is the low bit (little-endian: index 1 == q0=1), so H(0) populates
        // indices 0 and 1. This verifies the executed *prefix* is reconstructed, not
        // the full circuit.
        let mut debugger = bell_debugger();
        debugger.step().expect("step 1 failed"); // apply H only

        let state = debugger
            .get_current_state()
            .expect("Failed to get current state");

        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        assert!((state[0] - Complex64::new(inv_sqrt2, 0.0)).norm() < 1e-10);
        assert!((state[1] - Complex64::new(inv_sqrt2, 0.0)).norm() < 1e-10);
        assert!(state[2].norm() < 1e-10);
        assert!(state[3].norm() < 1e-10);
    }

    #[test]
    fn test_entanglement_entropy_bell_is_ln2() {
        // A Bell state is maximally entangled across the 1|1 cut: S = ln(2) nats.
        let mut debugger = bell_debugger();
        debugger.run().expect("run failed");

        let entropy = debugger
            .get_entanglement_entropy(1)
            .expect("entropy failed");
        assert!(
            (entropy - std::f64::consts::LN_2).abs() < 1e-10,
            "Bell entropy {entropy} should equal ln(2)"
        );
    }

    #[test]
    fn test_entanglement_entropy_product_state_is_zero() {
        // |+0> = H(0) only is a product state across the 1|1 cut: S = 0.
        let mut debugger = bell_debugger();
        debugger.step().expect("step failed"); // H(0) only

        let entropy = debugger
            .get_entanglement_entropy(1)
            .expect("entropy failed");
        assert!(
            entropy.abs() < 1e-10,
            "product-state entropy {entropy} should be 0"
        );
    }

    #[test]
    fn test_compute_entanglement_entropy_known_schmidt() {
        // Construct a 2-qubit state with known Schmidt coefficients
        // |psi> = sqrt(0.8)|00> + sqrt(0.2)|11>.
        // Reduced density eigenvalues are {0.8, 0.2}; entropy is the binary
        // entropy in nats: -(0.8 ln 0.8 + 0.2 ln 0.2).
        let p0 = 0.8_f64;
        let p1 = 0.2_f64;
        let state = Array1::from(vec![
            Complex64::new(p0.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(p1.sqrt(), 0.0),
        ]);

        let expected = -(p0 * p0.ln() + p1 * p1.ln());
        let entropy = compute_entanglement_entropy(&state, 1, 2).expect("entropy failed");
        assert!(
            (entropy - expected).abs() < 1e-10,
            "entropy {entropy} should equal {expected}"
        );
    }

    #[test]
    fn test_compute_entanglement_entropy_rejects_bad_cut() {
        let state: Array1<Complex64> =
            Array1::from(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        // num_qubits = 1: no valid bipartition exists.
        assert!(compute_entanglement_entropy(&state, 0, 1).is_err());
    }

    #[test]
    fn test_pauli_expectation_z_on_computational_basis() {
        // <Z> on |0> = +1, <Z> on |1> = -1.
        let zero: Array1<Complex64> =
            Array1::from(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let one: Array1<Complex64> =
            Array1::from(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);

        let ez0 = compute_pauli_expectation(&zero, "Z").expect("pauli failed");
        let ez1 = compute_pauli_expectation(&one, "Z").expect("pauli failed");
        assert!((ez0 - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!((ez1 - Complex64::new(-1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_pauli_expectation_x_on_plus_state() {
        // |+> = (|0> + |1>)/sqrt2; <X> = +1, <Z> = 0.
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let plus: Array1<Complex64> = Array1::from(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ]);

        let ex = compute_pauli_expectation(&plus, "X").expect("pauli failed");
        let ez = compute_pauli_expectation(&plus, "Z").expect("pauli failed");
        assert!((ex - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(ez.norm() < 1e-12);
    }

    #[test]
    fn test_pauli_expectation_zz_on_bell() {
        // Bell state (|00>+|11>)/sqrt2: <ZZ> = +1, <XX> = +1, <ZI> = 0.
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let bell: Array1<Complex64> = Array1::from(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ]);

        let ezz = compute_pauli_expectation(&bell, "ZZ").expect("pauli failed");
        let exx = compute_pauli_expectation(&bell, "XX").expect("pauli failed");
        let ezi = compute_pauli_expectation(&bell, "ZI").expect("pauli failed");
        assert!((ezz - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!((exx - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(ezi.norm() < 1e-12);
    }

    #[test]
    fn test_pauli_expectation_length_mismatch_errors() {
        let zero: Array1<Complex64> =
            Array1::from(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        // One qubit but a two-character Pauli string.
        assert!(compute_pauli_expectation(&zero, "ZZ").is_err());
    }

    #[test]
    fn test_pauli_via_debugger_zz_on_bell() {
        // End-to-end through the debugger API on a real executed circuit.
        let mut debugger = bell_debugger();
        debugger.run().expect("run failed");

        let ezz = debugger.get_pauli_expectation("ZZ").expect("pauli failed");
        assert!((ezz - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }
}
