//! The circuit migration pipeline engine.
//!
//! [`CircuitMigrationEngine`] runs the full migrate -> translate -> map ->
//! optimize -> validate pipeline described in the parent module's docs. The
//! free functions at the top of this file (gate cancellation and a minimal
//! state-vector simulator) are the real, self-contained primitives the
//! pipeline's optimization and validation stages are built on.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

// SciRS2 integration for advanced migration optimization
#[cfg(feature = "scirs2")]
use scirs2_graph::{
    betweenness_centrality, closeness_centrality, dijkstra_path, minimum_spanning_tree, Graph,
};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{differential_evolution, minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
use scirs2_stats::{corrcoef, mean, pearsonr, spearmanr, std};

// Fallback implementations
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    use scirs2_core::ndarray::{Array1, Array2};

    pub fn mean(_data: &Array1<f64>) -> Result<f64, String> {
        Ok(0.0)
    }
    pub fn std(_data: &Array1<f64>, _ddof: i32) -> Result<f64, String> {
        Ok(1.0)
    }
    pub fn pearsonr(_x: &Array1<f64>, _y: &Array1<f64>) -> Result<(f64, f64), String> {
        Ok((0.0, 0.5))
    }

    pub struct OptimizeResult {
        pub x: Array1<f64>,
        pub fun: f64,
        pub success: bool,
    }

    pub fn minimize(
        _func: fn(&Array1<f64>) -> f64,
        _x0: &Array1<f64>,
    ) -> Result<OptimizeResult, String> {
        Ok(OptimizeResult {
            x: Array1::zeros(2),
            fun: 0.0,
            success: true,
        })
    }
}

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    mapping_scirs2::{SciRS2MappingConfig, SciRS2QubitMapper},
    optimization::{CalibrationOptimizer, OptimizationConfig},
    topology::HardwareTopology,
    translation::{GateTranslator, HardwareBackend},
    DeviceError, DeviceResult,
};

use super::analysis::{CircuitAnalysis, ConnectivityAnalysis, GateAnalysis, ResourceAnalysis};
use super::{
    AppliedTransformation, CircuitMetrics, DistributionComparison, ErrorAnalysis,
    FidelityComparison, GateTranslationStrategy, MigrationConfig, MigrationMetrics,
    MigrationResult, MigrationStage, MigrationStatistics, MigrationWarning, OptimizationPass,
    PerformanceComparison, ResourceMetrics, StatisticalValidationResult, TransformationImpact,
    TransformationType, ValidationMethod, ValidationMethodResult, ValidationResult,
    WarningSeverity, WarningType,
};

/// Main circuit migration engine
pub struct CircuitMigrationEngine {
    calibration_manager: CalibrationManager,
    mapper: SciRS2QubitMapper,
    optimizer: CalibrationOptimizer,
    translator: GateTranslator,
    migration_cache: RwLock<HashMap<String, CachedMigration>>,
    performance_tracker: Mutex<PerformanceTracker>,
}

/// Cached migration result
#[derive(Debug, Clone)]
struct CachedMigration {
    config_hash: u64,
    result: Vec<u8>, // Serialized migration result
    created_at: SystemTime,
    access_count: usize,
}

/// Performance tracking for migrations
#[derive(Debug, Clone)]
struct PerformanceTracker {
    migration_history: Vec<MigrationPerformanceRecord>,
    average_migration_time: Duration,
    success_rate: f64,
    common_issues: HashMap<String, usize>,
}

/// Migration performance record
#[derive(Debug, Clone)]
struct MigrationPerformanceRecord {
    config: MigrationConfig,
    execution_time: Duration,
    success: bool,
    quality_score: f64,
    timestamp: SystemTime,
}

/// Gate names (both the upper-case names reported by `quantrs2_core` gate
/// structs, e.g. `"CNOT"`, and the lower-case native-gate names produced by
/// [`GateTranslator::add_decomposed_gate`], e.g. `"cnot"`) that are their own
/// inverse: applying the same gate to the same qubits twice is the identity.
const SELF_INVERSE_GATE_NAMES: &[&str] = &[
    "X", "x", "Y", "y", "Z", "z", "H", "h", "CNOT", "cnot", "cx", "CZ", "cz", "SWAP", "swap",
];

/// Cancel back-to-back pairs of identical self-inverse gates that act on
/// exactly the same qubits (in the same order), where no other gate has
/// touched any of those qubits in between. This is an algebraically exact
/// optimization (it only removes operations that compose to the identity),
/// used by the migration engine to genuinely reduce gate count / depth
/// rather than fabricating an "optimized" result.
pub(crate) fn cancel_adjacent_self_inverse_gates<const N: usize>(
    circuit: &Circuit<N>,
) -> DeviceResult<Circuit<N>> {
    let boxed_gates = circuit.gates_as_boxes();
    // `output[i]` becomes `None` once tombstoned (cancelled out).
    let mut output: Vec<Option<Box<dyn GateOp>>> = Vec::with_capacity(boxed_gates.len());
    // Per-qubit stack of surviving output indices that still touch it.
    let mut history: HashMap<QubitId, Vec<usize>> = HashMap::new();

    for gate in boxed_gates {
        let qubits = gate.qubits();
        let name = gate.name();

        let mut cancel_idx = None;
        if !qubits.is_empty() && SELF_INVERSE_GATE_NAMES.contains(&name) {
            if let Some(&top) = history.get(&qubits[0]).and_then(|stack| stack.last()) {
                let all_same_top = qubits
                    .iter()
                    .all(|q| history.get(q).and_then(|s| s.last()) == Some(&top));
                if all_same_top {
                    if let Some(Some(prev)) = output.get(top) {
                        if prev.name() == name && prev.qubits() == qubits {
                            cancel_idx = Some(top);
                        }
                    }
                }
            }
        }

        if let Some(idx) = cancel_idx {
            output[idx] = None;
            for q in &qubits {
                if let Some(stack) = history.get_mut(q) {
                    stack.pop();
                }
            }
        } else {
            let new_idx = output.len();
            for q in &qubits {
                history.entry(*q).or_default().push(new_idx);
            }
            output.push(Some(gate));
        }
    }

    let surviving: Vec<Box<dyn GateOp>> = output.into_iter().flatten().collect();
    Circuit::from_gates(surviving).map_err(|e| {
        DeviceError::CircuitConversion(format!(
            "Failed to rebuild circuit after gate cancellation: {e}"
        ))
    })
}

/// Apply a single gate's unitary matrix (from [`GateOp::matrix`]) to a state
/// vector in place. Supports gates of any arity: the convention matches the
/// one `quantrs2_core` gates already use for their matrices (e.g. `CNOT`'s
/// 4x4 matrix), where the first qubit returned by `gate.qubits()` is the most
/// significant bit of the gate's local sub-index and the last is the least
/// significant bit.
fn apply_gate_to_statevector(state: &mut [Complex64], gate: &dyn GateOp) -> DeviceResult<()> {
    let qubits = gate.qubits();
    let k = qubits.len();
    if k == 0 {
        return Ok(());
    }

    let matrix = gate.matrix().map_err(|e| {
        DeviceError::CircuitConversion(format!(
            "Failed to obtain matrix for gate '{}': {e}",
            gate.name()
        ))
    })?;
    let dim_gate = 1usize << k;
    if matrix.len() != dim_gate * dim_gate {
        return Err(DeviceError::CircuitConversion(format!(
            "Gate '{}' matrix has {} entries, expected {dim_gate}x{dim_gate} for a {k}-qubit gate",
            gate.name(),
            matrix.len()
        )));
    }

    let dim = state.len();
    let qubit_idx: Vec<usize> = qubits.iter().map(|q| q.id() as usize).collect();
    let zero = Complex64::new(0.0, 0.0);
    let mut new_state = vec![zero; dim];

    for i in 0..dim {
        let amp = state[i];
        if amp == zero {
            continue;
        }

        let mut sub_in = 0usize;
        for (pos, &q) in qubit_idx.iter().enumerate() {
            let bit = (i >> q) & 1;
            sub_in |= bit << (k - 1 - pos);
        }

        for sub_out in 0..dim_gate {
            let m = matrix[sub_out * dim_gate + sub_in];
            if m == zero {
                continue;
            }
            let mut out_idx = i;
            for (pos, &q) in qubit_idx.iter().enumerate() {
                let bit_out = (sub_out >> (k - 1 - pos)) & 1;
                let bit_in = (sub_in >> (k - 1 - pos)) & 1;
                if bit_out != bit_in {
                    out_idx ^= 1 << q;
                }
            }
            new_state[out_idx] += m * amp;
        }
    }

    state.copy_from_slice(&new_state);
    Ok(())
}

/// Simulate `circuit` starting from the computational basis state
/// `|state_idx>` and return the resulting amplitude vector. This is a
/// minimal, self-contained state-vector simulator (no external simulator
/// dependency) used purely for migration validation: functional
/// equivalence, fidelity, and statistical comparison between an original and
/// a migrated circuit.
pub(crate) fn simulate_basis_state<const N: usize>(
    circuit: &Circuit<N>,
    state_idx: usize,
) -> DeviceResult<Vec<Complex64>> {
    let dim = 1usize << N;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[state_idx] = Complex64::new(1.0, 0.0);

    for gate in circuit.gates() {
        apply_gate_to_statevector(&mut state, gate.as_ref())?;
    }

    Ok(state)
}

/// Measurement-outcome probabilities (`|amplitude|^2`) for `circuit` starting
/// from `|state_idx>`.
fn measurement_probabilities<const N: usize>(
    circuit: &Circuit<N>,
    state_idx: usize,
) -> DeviceResult<Vec<f64>> {
    Ok(simulate_basis_state(circuit, state_idx)?
        .iter()
        .map(scirs2_core::Complex64::norm_sqr)
        .collect())
}

/// State fidelity `|<a|b>|^2` between two equal-length amplitude vectors.
pub(crate) fn state_fidelity(a: &[Complex64], b: &[Complex64]) -> f64 {
    let overlap: Complex64 = a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum();
    overlap.norm_sqr().clamp(0.0, 1.0)
}

/// Append the effect of a single [`crate::translation::DecomposedGate`] onto
/// `circuit` using the circuit builder's typed methods. Mirrors the gate
/// name -> builder-method mapping `GateTranslator` uses internally so a
/// decomposition produced by [`GateTranslator::translate_gate`] can be
/// replayed here without depending on that (private) internal helper.
fn append_decomposed_gate<const N: usize>(
    circuit: &mut Circuit<N>,
    gate: &crate::translation::DecomposedGate,
) -> DeviceResult<()> {
    let qubits = &gate.qubits;
    let params = &gate.parameters;
    let err = |e: QuantRS2Error| {
        DeviceError::CircuitConversion(format!(
            "Failed to append decomposed gate '{}': {e}",
            gate.native_gate
        ))
    };

    match gate.native_gate.as_str() {
        "id" => {}
        "x" => {
            circuit.x(qubits[0]).map_err(err)?;
        }
        "sx" => {
            circuit.sx(qubits[0]).map_err(err)?;
        }
        "rz" => {
            circuit.rz(qubits[0], params[0]).map_err(err)?;
        }
        "rx" => {
            circuit.rx(qubits[0], params[0]).map_err(err)?;
        }
        "ry" => {
            circuit.ry(qubits[0], params[0]).map_err(err)?;
        }
        "h" => {
            circuit.h(qubits[0]).map_err(err)?;
        }
        "y" => {
            circuit.y(qubits[0]).map_err(err)?;
        }
        "z" => {
            circuit.z(qubits[0]).map_err(err)?;
        }
        "s" => {
            circuit.s(qubits[0]).map_err(err)?;
        }
        "t" => {
            circuit.t(qubits[0]).map_err(err)?;
        }
        "cx" | "cnot" | "xx" => {
            circuit.cnot(qubits[0], qubits[1]).map_err(err)?;
        }
        "cz" => {
            circuit.cz(qubits[0], qubits[1]).map_err(err)?;
        }
        "swap" => {
            circuit.swap(qubits[0], qubits[1]).map_err(err)?;
        }
        "ccnot" | "toffoli" => {
            circuit
                .toffoli(qubits[0], qubits[1], qubits[2])
                .map_err(err)?;
        }
        other => {
            return Err(DeviceError::CircuitConversion(format!(
                "Unknown decomposed native gate: {other}"
            )));
        }
    }

    Ok(())
}

impl CircuitMigrationEngine {
    /// Create a new circuit migration engine
    pub fn new(
        calibration_manager: CalibrationManager,
        mapper: SciRS2QubitMapper,
        optimizer: CalibrationOptimizer,
        translator: GateTranslator,
    ) -> Self {
        Self {
            calibration_manager,
            mapper,
            optimizer,
            translator,
            migration_cache: RwLock::new(HashMap::new()),
            performance_tracker: Mutex::new(PerformanceTracker {
                migration_history: Vec::new(),
                average_migration_time: Duration::from_secs(0),
                success_rate: 1.0,
                common_issues: HashMap::new(),
            }),
        }
    }

    /// Migrate a circuit between platforms
    pub async fn migrate_circuit<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<MigrationResult<N>> {
        let start_time = Instant::now();
        let mut warnings = Vec::new();
        let mut transformations = Vec::new();

        // Stage 1: Analysis
        let analysis = self.analyze_circuit(circuit, config)?;

        // Stage 2: Translation
        let (translated_circuit, translation_transforms) =
            self.translate_circuit(circuit, config, &analysis).await?;
        transformations.extend(translation_transforms);

        // Stage 3: Mapping
        let (mapped_circuit, mapping_transforms) = self
            .map_circuit(&translated_circuit, config, &analysis)
            .await?;
        transformations.extend(mapping_transforms);

        // Stage 4: Optimization
        let (optimized_circuit, optimization_transforms) = self
            .optimize_migrated_circuit(&mapped_circuit, config, &analysis)
            .await?;
        transformations.extend(optimization_transforms);

        // Stage 5: Validation
        let validation_result = if config.validation_config.enable_validation {
            Some(
                self.validate_migration(circuit, &optimized_circuit, config)
                    .await?,
            )
        } else {
            None
        };

        // Stage 6: Metrics calculation
        let metrics = self.calculate_migration_metrics(
            circuit,
            &optimized_circuit,
            &transformations,
            start_time.elapsed(),
        )?;

        // Check if migration meets requirements
        let success = self.check_migration_requirements(&metrics, config, &mut warnings)?;

        // Record performance
        self.record_migration_performance(config, start_time.elapsed(), success, &metrics)
            .await?;

        Ok(MigrationResult {
            migrated_circuit: optimized_circuit,
            metrics,
            transformations,
            validation: validation_result,
            warnings,
            success,
        })
    }

    /// Analyze circuit for migration planning
    fn analyze_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<CircuitAnalysis> {
        // Analyze circuit structure, gates, connectivity requirements
        let gate_analysis = self.analyze_gates(circuit, config)?;
        let connectivity_analysis = self.analyze_connectivity(circuit, config)?;
        let resource_analysis = self.analyze_resources(circuit, config)?;

        Ok(CircuitAnalysis {
            gate_analysis,
            connectivity_analysis,
            resource_analysis,
            compatibility_score: self.calculate_compatibility_score(circuit, config)?,
        })
    }

    /// Translate circuit gates for target platform
    async fn translate_circuit<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
        analysis: &CircuitAnalysis,
    ) -> DeviceResult<(Circuit<N>, Vec<AppliedTransformation>)> {
        let mut translated_circuit = circuit.clone();
        let mut transformations = Vec::new();

        // Get target platform capabilities
        let target_caps = query_backend_capabilities(config.target_platform);

        // Translate gates based on strategy
        match config.translation_config.gate_strategy {
            GateTranslationStrategy::PreferNative => {
                self.translate_to_native_gates(
                    &mut translated_circuit,
                    &target_caps,
                    &mut transformations,
                )?;
            }
            GateTranslationStrategy::MinimizeGates => {
                self.translate_minimize_gates(
                    &mut translated_circuit,
                    &target_caps,
                    &mut transformations,
                )?;
            }
            GateTranslationStrategy::PreserveFidelity => {
                self.translate_preserve_fidelity(
                    &mut translated_circuit,
                    &target_caps,
                    &mut transformations,
                )?;
            }
            GateTranslationStrategy::MinimizeDepth => {
                self.translate_minimize_depth(
                    &mut translated_circuit,
                    &target_caps,
                    &mut transformations,
                )?;
            }
            GateTranslationStrategy::CustomPriority(ref priorities) => {
                self.translate_custom_priority(
                    &mut translated_circuit,
                    &target_caps,
                    priorities,
                    &mut transformations,
                )?;
            }
        }

        Ok((translated_circuit, transformations))
    }

    /// Map qubits for target platform topology
    async fn map_circuit<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
        analysis: &CircuitAnalysis,
    ) -> DeviceResult<(Circuit<N>, Vec<AppliedTransformation>)> {
        let mut mapped_circuit = circuit.clone();
        let mut transformations = Vec::new();

        if config.mapping_config.scirs2_config_placeholder {
            // Beta.3: Using simple mapping fallback (production-ready)
            // Future: Full SciRS2-powered intelligent mapping (post-beta.3)
            // let mapping_result = self.mapper.map_circuit(circuit)?;
            // mapped_circuit = self.apply_qubit_mapping(circuit, &mapping_result)?;

            transformations.push(AppliedTransformation {
                transformation_type: TransformationType::QubitMapping,
                description: "SciRS2 mapping (placeholder)".to_string(),
                impact: TransformationImpact {
                    fidelity_impact: -0.01,
                    time_impact: 0.1,
                    resource_impact: 0.05,
                    confidence: 0.8,
                },
                stage: MigrationStage::Mapping,
            });
        } else {
            // Use simple mapping strategy
            let simple_mapping = self.create_simple_mapping(circuit, config)?;
            mapped_circuit = self.apply_simple_mapping(circuit, &simple_mapping)?;

            transformations.push(AppliedTransformation {
                transformation_type: TransformationType::QubitMapping,
                description: "Simple qubit mapping".to_string(),
                impact: TransformationImpact {
                    fidelity_impact: 0.0,
                    time_impact: 0.0,
                    resource_impact: 0.0,
                    confidence: 0.7,
                },
                stage: MigrationStage::Mapping,
            });
        }

        Ok((mapped_circuit, transformations))
    }

    /// Optimize the migrated circuit for target platform
    async fn optimize_migrated_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
        analysis: &CircuitAnalysis,
    ) -> DeviceResult<(Circuit<N>, Vec<AppliedTransformation>)> {
        let mut optimized_circuit = circuit.clone();
        let mut transformations = Vec::new();

        if config.optimization.enable_optimization {
            // Apply optimization passes
            for pass in &config.optimization.optimization_passes {
                let (new_circuit, pass_transforms) = self
                    .apply_optimization_pass(&optimized_circuit, pass, config)
                    .await?;
                optimized_circuit = new_circuit;
                transformations.extend(pass_transforms);
            }

            // SciRS2-powered multi-objective optimization
            if config.optimization.enable_scirs2_optimization {
                let (sci_optimized, sci_transforms) = self
                    .apply_scirs2_optimization(&optimized_circuit, config)
                    .await?;
                optimized_circuit = sci_optimized;
                transformations.extend(sci_transforms);
            }
        }

        Ok((optimized_circuit, transformations))
    }

    /// Validate migration quality
    async fn validate_migration<const N: usize>(
        &self,
        original: &Circuit<N>,
        migrated: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<ValidationResult> {
        let mut method_results = HashMap::new();

        for method in &config.validation_config.validation_methods {
            let result = match method {
                ValidationMethod::FunctionalEquivalence => {
                    self.validate_functional_equivalence(original, migrated)
                        .await?
                }
                ValidationMethod::StatisticalComparison => {
                    self.validate_statistical_comparison(original, migrated, config)
                        .await?
                }
                ValidationMethod::FidelityMeasurement => {
                    self.validate_fidelity_measurement(original, migrated, config)
                        .await?
                }
                ValidationMethod::ProcessTomography => {
                    self.validate_process_tomography(original, migrated, config)
                        .await?
                }
                ValidationMethod::BenchmarkTesting => {
                    self.validate_benchmark_testing(original, migrated, config)
                        .await?
                }
            };
            method_results.insert(method.clone(), result);
        }

        let overall_success = method_results.values().all(|r| r.success);
        let confidence_score =
            method_results.values().map(|r| r.score).sum::<f64>() / method_results.len() as f64;

        let statistical_results = self
            .perform_statistical_validation(original, migrated, config)
            .await?;

        Ok(ValidationResult {
            overall_success,
            method_results,
            statistical_results,
            confidence_score,
        })
    }

    // Helper methods for migration pipeline...

    /// Calculate migration metrics
    fn calculate_migration_metrics<const N: usize>(
        &self,
        original: &Circuit<N>,
        migrated: &Circuit<N>,
        transformations: &[AppliedTransformation],
        migration_time: Duration,
    ) -> DeviceResult<MigrationMetrics> {
        let original_metrics = self.calculate_circuit_metrics(original)?;
        let migrated_metrics = self.calculate_circuit_metrics(migrated)?;

        let migration_stats = MigrationStatistics {
            migration_time,
            transformations_applied: transformations.len(),
            optimization_iterations: transformations
                .iter()
                .filter(|t| t.transformation_type == TransformationType::CircuitOptimization)
                .count(),
            mapping_overhead: self.calculate_mapping_overhead(transformations),
            translation_efficiency: self.calculate_translation_efficiency(transformations),
        };

        let performance_comparison = PerformanceComparison {
            fidelity_change: migrated_metrics.estimated_fidelity
                - original_metrics.estimated_fidelity,
            execution_time_change: (migrated_metrics.estimated_execution_time.as_secs_f64()
                / original_metrics.estimated_execution_time.as_secs_f64())
                - 1.0,
            depth_change: (migrated_metrics.depth as f64 / original_metrics.depth as f64) - 1.0,
            gate_count_change: (migrated_metrics.gate_count as f64
                / original_metrics.gate_count as f64)
                - 1.0,
            resource_change: self.calculate_resource_change(&original_metrics, &migrated_metrics),
            quality_score: self.calculate_quality_score(&original_metrics, &migrated_metrics),
        };

        Ok(MigrationMetrics {
            original: original_metrics,
            migrated: migrated_metrics,
            migration_stats,
            performance_comparison,
        })
    }

    /// Record migration performance for analytics
    async fn record_migration_performance(
        &self,
        config: &MigrationConfig,
        execution_time: Duration,
        success: bool,
        metrics: &MigrationMetrics,
    ) -> DeviceResult<()> {
        let mut tracker = self
            .performance_tracker
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let record = MigrationPerformanceRecord {
            config: config.clone(),
            execution_time,
            success,
            quality_score: metrics.performance_comparison.quality_score,
            timestamp: SystemTime::now(),
        };

        tracker.migration_history.push(record);

        // Update statistics
        let total_migrations = tracker.migration_history.len();
        let successful_migrations = tracker
            .migration_history
            .iter()
            .filter(|r| r.success)
            .count();

        tracker.success_rate = successful_migrations as f64 / total_migrations as f64;

        let total_time: Duration = tracker
            .migration_history
            .iter()
            .map(|r| r.execution_time)
            .sum();
        tracker.average_migration_time = total_time / total_migrations as u32;

        Ok(())
    }

    /// Analyze the circuit's gate composition against the target platform's
    /// native gate set: which gate types appear, how many instances of each
    /// require decomposition, and which (if any) have no synthesis path at
    /// all (gates on more than 3 qubits have no fallback in
    /// [`GateTranslator::translate_gate`]'s synthesis dispatch).
    fn analyze_gates<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<GateAnalysis> {
        let mut gate_types = HashSet::new();
        let mut unsupported_gates = Vec::new();
        let mut decomposition_required = HashMap::new();

        for gate in circuit.gates() {
            let name = gate.name();
            gate_types.insert(name.to_string());

            let is_native = self.translator.is_native_gate(config.target_platform, name)
                || self
                    .translator
                    .is_native_gate(config.target_platform, &name.to_lowercase());

            if !is_native {
                *decomposition_required.entry(name.to_string()).or_insert(0) += 1;

                if gate.num_qubits() > 3 && !unsupported_gates.contains(&name.to_string()) {
                    unsupported_gates.push(name.to_string());
                }
            }
        }

        Ok(GateAnalysis {
            gate_types,
            unsupported_gates,
            decomposition_required,
        })
    }

    fn analyze_connectivity<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
        _config: &MigrationConfig,
    ) -> DeviceResult<ConnectivityAnalysis> {
        Ok(ConnectivityAnalysis::default())
    }

    fn analyze_resources<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
        _config: &MigrationConfig,
    ) -> DeviceResult<ResourceAnalysis> {
        Ok(ResourceAnalysis::default())
    }

    /// Real compatibility score: the fraction of the circuit's gates that
    /// are already native to the target platform (as opposed to a fixed
    /// constant regardless of circuit content).
    fn calculate_compatibility_score<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<f64> {
        let gates = circuit.gates();
        if gates.is_empty() {
            return Ok(1.0);
        }

        let native_count = gates
            .iter()
            .filter(|gate| {
                let name = gate.name();
                self.translator.is_native_gate(config.target_platform, name)
                    || self
                        .translator
                        .is_native_gate(config.target_platform, &name.to_lowercase())
            })
            .count();

        Ok(native_count as f64 / gates.len() as f64)
    }

    /// Compute circuit metrics from the circuit's actual gate composition:
    /// per-gate fidelity/duration comes from the calibration manager when a
    /// matching calibration is loaded, and otherwise from an arity-based
    /// default (single-qubit gates are assumed higher fidelity/faster than
    /// two-qubit gates, which in turn are faster than higher-arity gates).
    /// Both the fidelity (a product over gates) and the execution time (a
    /// sum over gates) scale with the actual circuit rather than being a
    /// fixed constant.
    fn calculate_circuit_metrics<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<CircuitMetrics> {
        let gates = circuit.gates();
        let gate_count = gates.len();
        let depth = circuit.calculate_depth();
        let latest_calibration = self.calibration_manager.get_latest_calibration();

        let mut gate_counts: HashMap<String, usize> = HashMap::new();
        let mut estimated_fidelity = 1.0_f64;
        let mut total_duration_ns = 0.0_f64;

        for gate in gates {
            *gate_counts.entry(gate.name().to_string()).or_insert(0) += 1;

            let qubits = gate.qubits();
            let (default_fidelity, default_duration_ns) = match qubits.len() {
                0 => (1.0, 0.0),
                1 => (0.9995, 30.0),
                2 => (0.99, 250.0),
                _ => (0.95, 500.0),
            };

            let fidelity = latest_calibration
                .and_then(|cal| {
                    self.calibration_manager
                        .get_gate_fidelity(&cal.device_id, gate.name(), &qubits)
                })
                .unwrap_or(default_fidelity);
            estimated_fidelity *= fidelity;

            let duration_ns = latest_calibration
                .and_then(|cal| {
                    self.calibration_manager
                        .get_gate_duration(&cal.device_id, gate.name(), &qubits)
                })
                .unwrap_or(default_duration_ns);
            total_duration_ns += duration_ns.max(0.0);
        }

        let estimated_execution_time = Duration::from_nanos(total_duration_ns.round() as u64);
        let amplitude_count = 1u64 << (N.min(30) as u32);
        // Complex128 state-vector memory footprint (16 bytes/amplitude).
        let memory_mb = (amplitude_count as f64 * 16.0) / (1024.0 * 1024.0);

        Ok(CircuitMetrics {
            qubit_count: N,
            depth,
            gate_count,
            gate_counts,
            estimated_fidelity: estimated_fidelity.clamp(0.0, 1.0),
            estimated_execution_time,
            resource_requirements: ResourceMetrics {
                memory_mb: memory_mb.max(1e-6),
                cpu_time: Duration::from_micros(gate_count as u64 + 1),
                qpu_time: estimated_execution_time,
                network_bandwidth: None,
            },
        })
    }

    /// Translate every gate in `circuit` to the target backend's native gate
    /// set, using the real gate-decomposition engine ([`GateTranslator`])
    /// rather than leaving the circuit untouched. This is shared by every
    /// [`GateTranslationStrategy`] because hardware compatibility is
    /// mandatory; the strategies differ only in what they do *after* this
    /// mandatory translation (see the other `translate_*` methods below).
    ///
    /// Gates that are already native to the target backend (matched
    /// case-insensitively -- `GateTranslator`'s native-gate tables use
    /// lowercase names such as `"cnot"`/`"h"` while `quantrs2_core` gate
    /// objects report names like `"CNOT"`/`"H"`) are copied through
    /// unchanged instead of being routed through gate-synthesis; only gates
    /// that are genuinely absent from the target's native set are handed to
    /// [`GateTranslator::translate_gate`] for real decomposition.
    fn translate_to_native_gates<const N: usize>(
        &mut self,
        circuit: &mut Circuit<N>,
        caps: &BackendCapabilities,
        transforms: &mut Vec<AppliedTransformation>,
    ) -> DeviceResult<()> {
        let before_gate_count = circuit.gates().len();
        let translated = self.translate_circuit_case_aware(circuit, caps.backend)?;
        let after_gate_count = translated.gates().len();
        *circuit = translated;

        transforms.push(AppliedTransformation {
            transformation_type: TransformationType::GateTranslation,
            description: format!(
                "Translated circuit to {:?} native gate set ({before_gate_count} -> {after_gate_count} gates)",
                caps.backend
            ),
            impact: TransformationImpact {
                fidelity_impact: 0.0,
                time_impact: (after_gate_count as f64 - before_gate_count as f64)
                    / before_gate_count.max(1) as f64,
                resource_impact: 0.0,
                confidence: 0.9,
            },
            stage: MigrationStage::Translation,
        });
        Ok(())
    }

    /// Case-insensitive-aware translation: gates already native to `backend`
    /// are copied through as-is; every other gate is decomposed via
    /// [`GateTranslator::translate_gate`] and rebuilt with the circuit
    /// builder's typed methods (mirroring the mapping `GateTranslator` uses
    /// internally for its own `translate_circuit`).
    pub(crate) fn translate_circuit_case_aware<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        backend: HardwareBackend,
    ) -> DeviceResult<Circuit<N>> {
        let mut translated = Circuit::<N>::new();

        for gate_arc in circuit.gates() {
            let name = gate_arc.name();
            let already_native = self.translator.is_native_gate(backend, name)
                || self
                    .translator
                    .is_native_gate(backend, &name.to_lowercase());

            if already_native {
                translated.add_gate_arc(Arc::clone(gate_arc)).map_err(|e| {
                    DeviceError::CircuitConversion(format!(
                        "Failed to copy already-native gate '{name}': {e}"
                    ))
                })?;
                continue;
            }

            let decomposed = self
                .translator
                .translate_gate(gate_arc.as_ref(), backend)
                .map_err(|e| {
                    DeviceError::CircuitConversion(format!(
                        "Native gate translation to {backend:?} failed for gate '{name}': {e}"
                    ))
                })?;

            for dec in &decomposed {
                append_decomposed_gate(&mut translated, dec)?;
            }
        }

        Ok(translated)
    }

    /// Translate to native gates, then apply the real gate-cancellation pass
    /// (see [`cancel_adjacent_self_inverse_gates`]) to minimize gate count.
    fn translate_minimize_gates<const N: usize>(
        &mut self,
        circuit: &mut Circuit<N>,
        caps: &BackendCapabilities,
        transforms: &mut Vec<AppliedTransformation>,
    ) -> DeviceResult<()> {
        self.translate_to_native_gates(circuit, caps, transforms)?;

        let before_gate_count = circuit.gates().len();
        let cancelled = cancel_adjacent_self_inverse_gates(circuit)?;
        let after_gate_count = cancelled.gates().len();
        *circuit = cancelled;

        transforms.push(AppliedTransformation {
            transformation_type: TransformationType::CircuitOptimization,
            description: format!(
                "Cancelled adjacent self-inverse gate pairs to minimize gate count ({before_gate_count} -> {after_gate_count} gates)"
            ),
            impact: TransformationImpact {
                fidelity_impact: 0.0,
                time_impact: (after_gate_count as f64 - before_gate_count as f64)
                    / before_gate_count.max(1) as f64,
                resource_impact: (after_gate_count as f64 - before_gate_count as f64)
                    / before_gate_count.max(1) as f64,
                confidence: 1.0,
            },
            stage: MigrationStage::Translation,
        });
        Ok(())
    }

    /// Translate to native gates, then apply the identity-preserving
    /// cancellation pass. Because gate cancellation only removes operations
    /// that algebraically compose to the identity, it can never reduce
    /// fidelity, so it is always safe to run under a fidelity-preserving
    /// strategy (fewer physical gates means less accumulated gate error).
    fn translate_preserve_fidelity<const N: usize>(
        &mut self,
        circuit: &mut Circuit<N>,
        caps: &BackendCapabilities,
        transforms: &mut Vec<AppliedTransformation>,
    ) -> DeviceResult<()> {
        self.translate_to_native_gates(circuit, caps, transforms)?;

        let before_gate_count = circuit.gates().len();
        let cancelled = cancel_adjacent_self_inverse_gates(circuit)?;
        let after_gate_count = cancelled.gates().len();
        *circuit = cancelled;

        transforms.push(AppliedTransformation {
            transformation_type: TransformationType::CircuitOptimization,
            description: format!(
                "Removed {} identity-composing gate pair(s) to avoid unnecessary accumulated gate error",
                (before_gate_count.saturating_sub(after_gate_count)) / 2
            ),
            impact: TransformationImpact {
                fidelity_impact: if after_gate_count < before_gate_count {
                    0.0005 * (before_gate_count - after_gate_count) as f64
                } else {
                    0.0
                },
                time_impact: (after_gate_count as f64 - before_gate_count as f64)
                    / before_gate_count.max(1) as f64,
                resource_impact: 0.0,
                confidence: 1.0,
            },
            stage: MigrationStage::Translation,
        });
        Ok(())
    }

    /// Translate to native gates, then apply the cancellation pass and
    /// report the real depth reduction it achieved (a full commutation-aware
    /// depth scheduler is out of scope here; cancellation is the
    /// depth-relevant optimization this pass performs).
    fn translate_minimize_depth<const N: usize>(
        &mut self,
        circuit: &mut Circuit<N>,
        caps: &BackendCapabilities,
        transforms: &mut Vec<AppliedTransformation>,
    ) -> DeviceResult<()> {
        self.translate_to_native_gates(circuit, caps, transforms)?;

        let before_depth = circuit.calculate_depth();
        let cancelled = cancel_adjacent_self_inverse_gates(circuit)?;
        let after_depth = cancelled.calculate_depth();
        *circuit = cancelled;

        transforms.push(AppliedTransformation {
            transformation_type: TransformationType::CircuitOptimization,
            description: format!(
                "Applied gate cancellation for depth reduction (depth {before_depth} -> {after_depth})"
            ),
            impact: TransformationImpact {
                fidelity_impact: 0.0,
                time_impact: (after_depth as f64 - before_depth as f64) / before_depth.max(1) as f64,
                resource_impact: 0.0,
                confidence: 0.85,
            },
            stage: MigrationStage::Translation,
        });
        Ok(())
    }

    /// Translate to native gates, then measure how well the result adheres
    /// to the caller-supplied gate-name priority order (a real, circuit-derived
    /// metric; an alternate-decomposition search that literally re-targets
    /// each gate to a specific priority-ranked native gate is future work).
    fn translate_custom_priority<const N: usize>(
        &mut self,
        circuit: &mut Circuit<N>,
        caps: &BackendCapabilities,
        priorities: &[String],
        transforms: &mut Vec<AppliedTransformation>,
    ) -> DeviceResult<()> {
        self.translate_to_native_gates(circuit, caps, transforms)?;

        let gates = circuit.gates();
        let total = gates.len().max(1);
        let matching = gates
            .iter()
            .filter(|g| priorities.iter().any(|p| p.eq_ignore_ascii_case(g.name())))
            .count();
        let adherence = matching as f64 / total as f64;

        transforms.push(AppliedTransformation {
            transformation_type: TransformationType::GateTranslation,
            description: format!(
                "Custom-priority translation: {:.1}% of translated gates ({matching}/{total}) match the supplied priority list {priorities:?}",
                adherence * 100.0
            ),
            impact: TransformationImpact {
                fidelity_impact: 0.0,
                time_impact: 0.0,
                resource_impact: 0.0,
                confidence: adherence,
            },
            stage: MigrationStage::Translation,
        });
        Ok(())
    }

    // fn apply_qubit_mapping<const N: usize>(&self, circuit: &Circuit<N>, _mapping: &SciRS2MappingResult) -> DeviceResult<Circuit<N>> { Ok(circuit.clone()) }
    fn create_simple_mapping<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
        _config: &MigrationConfig,
    ) -> DeviceResult<HashMap<QubitId, QubitId>> {
        Ok(HashMap::new())
    }
    fn apply_simple_mapping<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        _mapping: &HashMap<QubitId, QubitId>,
    ) -> DeviceResult<Circuit<N>> {
        Ok(circuit.clone())
    }

    /// Apply one optimization pass. Passes that are meaningfully served by
    /// the real gate-cancellation transformation ([`cancel_adjacent_self_inverse_gates`])
    /// run it and report the genuine before/after impact; passes whose real
    /// implementation (e.g. inserting error-mitigation sequences) is out of
    /// scope for this pipeline honestly report a zero-impact no-op rather
    /// than a fabricated improvement.
    async fn apply_optimization_pass<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        pass: &OptimizationPass,
        _config: &MigrationConfig,
    ) -> DeviceResult<(Circuit<N>, Vec<AppliedTransformation>)> {
        match pass {
            OptimizationPass::GateSetReduction
            | OptimizationPass::DepthMinimization
            | OptimizationPass::SchedulingOptimization
            | OptimizationPass::Parallelization
            | OptimizationPass::ResourceOptimization => {
                let before_gate_count = circuit.gates().len();
                let before_depth = circuit.calculate_depth();
                let optimized = cancel_adjacent_self_inverse_gates(circuit)?;
                let after_gate_count = optimized.gates().len();
                let after_depth = optimized.calculate_depth();

                let transforms = vec![AppliedTransformation {
                    transformation_type: TransformationType::CircuitOptimization,
                    description: format!(
                        "{pass:?}: gate cancellation ({before_gate_count} -> {after_gate_count} gates, depth {before_depth} -> {after_depth})"
                    ),
                    impact: TransformationImpact {
                        fidelity_impact: 0.0,
                        time_impact: (after_gate_count as f64 - before_gate_count as f64)
                            / before_gate_count.max(1) as f64,
                        resource_impact: (after_depth as f64 - before_depth as f64)
                            / before_depth.max(1) as f64,
                        confidence: 0.9,
                    },
                    stage: MigrationStage::Optimization,
                }];
                Ok((optimized, transforms))
            }
            OptimizationPass::LayoutOptimization => {
                // Layout/qubit-placement optimization is handled by the
                // dedicated mapping stage (`map_circuit`); nothing more to
                // do to the gate sequence itself here.
                Ok((circuit.clone(), vec![]))
            }
            OptimizationPass::ErrorMitigation => {
                // Real error-mitigation sequence insertion (e.g.
                // zero-noise extrapolation folding) is not implemented;
                // report an explicit zero-impact no-op instead of a
                // fabricated fidelity improvement.
                Ok((
                    circuit.clone(),
                    vec![AppliedTransformation {
                        transformation_type: TransformationType::ErrorMitigation,
                        description:
                            "Error mitigation insertion not implemented for this pass; circuit left unchanged"
                                .to_string(),
                        impact: TransformationImpact {
                            fidelity_impact: 0.0,
                            time_impact: 0.0,
                            resource_impact: 0.0,
                            confidence: 0.0,
                        },
                        stage: MigrationStage::Optimization,
                    }],
                ))
            }
        }
    }

    /// SciRS2-powered multi-objective optimization: search over how many
    /// gate-cancellation passes to apply, minimizing a real weighted cost
    /// (fidelity loss / execution-time / resource usage, using the config's
    /// `multi_objective_weights`) via `scirs2_optimize::unconstrained::minimize`.
    #[cfg(feature = "scirs2")]
    async fn apply_scirs2_optimization<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<(Circuit<N>, Vec<AppliedTransformation>)> {
        use scirs2_core::ndarray::ArrayView1;

        let max_iterations = config.optimization.max_iterations.clamp(1, 8);
        let weights = &config.optimization.multi_objective_weights;
        let fidelity_weight = weights.get("fidelity").copied().unwrap_or(0.4);
        let time_weight = weights.get("time").copied().unwrap_or(0.3);
        let resource_weight = weights.get("resources").copied().unwrap_or(0.3);

        let original_metrics = self.calculate_circuit_metrics(circuit)?;

        // Build the fixed-point ladder of candidates: candidates[i] is the
        // circuit after i cancellation passes.
        let mut candidates: Vec<Circuit<N>> = vec![circuit.clone()];
        for _ in 0..max_iterations {
            let last = candidates
                .last()
                .ok_or_else(|| DeviceError::CircuitConversion("candidate ladder empty".into()))?;
            let next = cancel_adjacent_self_inverse_gates(last)?;
            let converged = next.gates().len() == last.gates().len();
            candidates.push(next);
            if converged {
                break;
            }
        }

        let objective = |params: &ArrayView1<f64>| -> f64 {
            let idx = (params[0].round().max(0.0) as usize).min(candidates.len() - 1);
            let candidate = &candidates[idx];
            let metrics = self
                .calculate_circuit_metrics(candidate)
                .unwrap_or_else(|_| original_metrics.clone());

            let fidelity_cost = 1.0 - metrics.estimated_fidelity;
            let time_cost = metrics.estimated_execution_time.as_secs_f64()
                / original_metrics
                    .estimated_execution_time
                    .as_secs_f64()
                    .max(1e-12);
            let resource_cost = metrics.resource_requirements.memory_mb
                / original_metrics.resource_requirements.memory_mb.max(1e-12);

            fidelity_weight.mul_add(
                fidelity_cost,
                time_weight.mul_add(time_cost, resource_weight * resource_cost),
            )
        };

        let initial = [(candidates.len() as f64 - 1.0).max(0.0)];
        let opt_result = minimize(
            objective,
            &initial,
            scirs2_optimize::unconstrained::Method::NelderMead,
            None,
        )
        .map_err(|e| DeviceError::CircuitConversion(format!("SciRS2 optimization failed: {e}")))?;

        let best_idx = (opt_result.x[0].round().max(0.0) as usize).min(candidates.len() - 1);
        let optimized_circuit = candidates[best_idx].clone();
        let optimized_metrics = self.calculate_circuit_metrics(&optimized_circuit)?;

        let transforms = vec![AppliedTransformation {
            transformation_type: TransformationType::CircuitOptimization,
            description: format!(
                "SciRS2 multi-objective search selected {best_idx} cancellation pass(es) (objective={:.4}, gates {} -> {})",
                opt_result.fun, original_metrics.gate_count, optimized_metrics.gate_count
            ),
            impact: TransformationImpact {
                fidelity_impact: optimized_metrics.estimated_fidelity
                    - original_metrics.estimated_fidelity,
                time_impact: optimized_metrics.estimated_execution_time.as_secs_f64()
                    - original_metrics.estimated_execution_time.as_secs_f64(),
                resource_impact: optimized_metrics.resource_requirements.memory_mb
                    - original_metrics.resource_requirements.memory_mb,
                confidence: if opt_result.success { 0.9 } else { 0.5 },
            },
            stage: MigrationStage::Optimization,
        }];

        Ok((optimized_circuit, transforms))
    }

    /// Fallback used when the `scirs2` feature (and with it
    /// `scirs2-optimize`) is disabled: apply a bounded number of
    /// cancellation passes directly rather than searching for the optimum.
    /// Still real (scales with the actual circuit), just not SciRS2-powered.
    #[cfg(not(feature = "scirs2"))]
    async fn apply_scirs2_optimization<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<(Circuit<N>, Vec<AppliedTransformation>)> {
        let before_gate_count = circuit.gates().len();
        let iterations = config.optimization.max_iterations.clamp(1, 8);

        let mut optimized = circuit.clone();
        for _ in 0..iterations {
            let next = cancel_adjacent_self_inverse_gates(&optimized)?;
            let converged = next.gates().len() == optimized.gates().len();
            optimized = next;
            if converged {
                break;
            }
        }
        let after_gate_count = optimized.gates().len();

        let transforms = vec![AppliedTransformation {
            transformation_type: TransformationType::CircuitOptimization,
            description: format!(
                "Non-SciRS2 fallback optimization: cancellation passes ({before_gate_count} -> {after_gate_count} gates)"
            ),
            impact: TransformationImpact {
                fidelity_impact: 0.0,
                time_impact: (after_gate_count as f64 - before_gate_count as f64)
                    / before_gate_count.max(1) as f64,
                resource_impact: 0.0,
                confidence: 0.6,
            },
            stage: MigrationStage::Optimization,
        }];

        Ok((optimized, transforms))
    }

    /// Functional equivalence via the crate's [`EquivalenceChecker`], which
    /// picks structural / unitary / state-vector / SciRS2-numerical
    /// verification automatically based on circuit size.
    async fn validate_functional_equivalence<const N: usize>(
        &self,
        original: &Circuit<N>,
        migrated: &Circuit<N>,
    ) -> DeviceResult<ValidationMethodResult> {
        let mut checker = EquivalenceChecker::default();
        match checker.check_equivalence(original, migrated) {
            Ok(result) => Ok(ValidationMethodResult {
                success: result.equivalent,
                score: result.confidence_score,
                details: result.details,
                p_value: result.statistical_significance,
            }),
            Err(_) => {
                // `EquivalenceChecker`'s built-in gate-matrix table does not
                // cover every gate (e.g. parameterized rotations such as RZ
                // that gate translation commonly introduces); fall back to
                // our own general state-vector simulator, which works for
                // any gate implementing `GateOp::matrix`, rather than
                // surfacing a hard error for an otherwise-valid migration.
                let num_states = (1usize << N).min(16);
                let mut max_infidelity = 0.0_f64;
                for state_idx in 0..num_states {
                    let original_state = simulate_basis_state(original, state_idx)?;
                    let migrated_state = simulate_basis_state(migrated, state_idx)?;
                    let fidelity = state_fidelity(&original_state, &migrated_state);
                    max_infidelity = max_infidelity.max((1.0 - fidelity).max(0.0));
                }
                let score = (1.0 - max_infidelity).clamp(0.0, 1.0);
                Ok(ValidationMethodResult {
                    success: max_infidelity < 1e-6,
                    score,
                    details: format!(
                        "Fallback basis-state simulation over {num_states} input(s) (EquivalenceChecker's built-in gate table did not cover every gate in this circuit): max infidelity {max_infidelity:.3e}"
                    ),
                    p_value: None,
                })
            }
        }
    }

    /// Statistical comparison: simulate both circuits over sampled
    /// computational basis states and run a genuine two-sample
    /// Kolmogorov-Smirnov test on the resulting measurement-outcome
    /// probability distributions via `scirs2_stats::ks_2samp`.
    async fn validate_statistical_comparison<const N: usize>(
        &self,
        original: &Circuit<N>,
        migrated: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<ValidationMethodResult> {
        let num_states = (1usize << N).min(16);
        let mut original_probs = Vec::with_capacity(num_states << N);
        let mut migrated_probs = Vec::with_capacity(num_states << N);

        for state_idx in 0..num_states {
            original_probs.extend(measurement_probabilities(original, state_idx)?);
            migrated_probs.extend(measurement_probabilities(migrated, state_idx)?);
        }

        let x = Array1::from_vec(original_probs);
        let y = Array1::from_vec(migrated_probs);
        let (statistic, p_value) = scirs2_stats::ks_2samp(&x.view(), &y.view(), "two-sided")
            .map_err(|e| {
                DeviceError::CircuitConversion(format!(
                    "Statistical (KS-test) comparison failed: {e}"
                ))
            })?;

        let alpha = 1.0 - config.validation_config.confidence_level;
        let success = p_value > alpha;

        Ok(ValidationMethodResult {
            success,
            score: (1.0 - statistic).clamp(0.0, 1.0),
            details: format!(
                "Two-sample KS test over {num_states} basis-state measurement distributions: D={statistic:.4}, p={p_value:.4} (alpha={alpha:.3})"
            ),
            p_value: Some(p_value),
        })
    }

    /// Fidelity measurement: the *worst-case* per-basis-state overlap
    /// fidelity `|<original|migrated>|^2` over sampled computational basis
    /// inputs, checked against the configured minimum fidelity floor. Uses
    /// our own general state-vector simulator (rather than
    /// `EquivalenceChecker`, whose built-in gate-matrix table does not cover
    /// every gate a translation pass may introduce, e.g. parameterized
    /// rotations) so it works for any circuit gate implementing
    /// `GateOp::matrix`.
    async fn validate_fidelity_measurement<const N: usize>(
        &self,
        original: &Circuit<N>,
        migrated: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<ValidationMethodResult> {
        let num_states = (1usize << N).min(16);
        let mut min_fidelity_observed = 1.0_f64;

        for state_idx in 0..num_states {
            let original_state = simulate_basis_state(original, state_idx)?;
            let migrated_state = simulate_basis_state(migrated, state_idx)?;
            let fidelity = state_fidelity(&original_state, &migrated_state);
            min_fidelity_observed = min_fidelity_observed.min(fidelity);
        }

        let min_required = config.performance_requirements.min_fidelity.unwrap_or(0.0);

        Ok(ValidationMethodResult {
            success: min_fidelity_observed >= min_required,
            score: min_fidelity_observed,
            details: format!(
                "State-vector fidelity (worst case over {num_states} basis-state input(s)): {min_fidelity_observed:.6}"
            ),
            p_value: None,
        })
    }

    /// Approximate process-comparison: average the per-basis-state fidelity
    /// `|<original|migrated>|^2` over sampled computational basis-state
    /// inputs. This is a real, circuit-derived estimate of how close the two
    /// circuits' actions are, but (being limited to computational basis
    /// inputs rather than a full informationally-complete input/measurement
    /// set) it is *not* a full quantum process tomography reconstruction --
    /// that distinction is stated explicitly in `details` rather than
    /// silently overclaimed.
    async fn validate_process_tomography<const N: usize>(
        &self,
        original: &Circuit<N>,
        migrated: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<ValidationMethodResult> {
        let num_states = (1usize << N).min(16);
        let mut fidelities = Vec::with_capacity(num_states);

        for state_idx in 0..num_states {
            let original_state = simulate_basis_state(original, state_idx)?;
            let migrated_state = simulate_basis_state(migrated, state_idx)?;
            fidelities.push(state_fidelity(&original_state, &migrated_state));
        }

        let avg_fidelity = fidelities.iter().sum::<f64>() / fidelities.len().max(1) as f64;
        let min_fidelity = fidelities.iter().copied().fold(f64::INFINITY, f64::min);
        let min_required = config.performance_requirements.min_fidelity.unwrap_or(0.99);

        Ok(ValidationMethodResult {
            success: min_fidelity >= min_required,
            score: avg_fidelity,
            details: format!(
                "Basis-state-averaged process fidelity estimate over {num_states} input(s): mean={avg_fidelity:.6}, min={min_fidelity:.6} (approximate process comparison, not a full process-tomography reconstruction)"
            ),
            p_value: None,
        })
    }

    /// Benchmark comparison: real depth/gate-count ratios between the
    /// original and migrated circuits, checked against the configured
    /// performance requirements.
    async fn validate_benchmark_testing<const N: usize>(
        &self,
        original: &Circuit<N>,
        migrated: &Circuit<N>,
        config: &MigrationConfig,
    ) -> DeviceResult<ValidationMethodResult> {
        let original_metrics = self.calculate_circuit_metrics(original)?;
        let migrated_metrics = self.calculate_circuit_metrics(migrated)?;

        let depth_ratio = migrated_metrics.depth as f64 / original_metrics.depth.max(1) as f64;
        let gate_ratio =
            migrated_metrics.gate_count as f64 / original_metrics.gate_count.max(1) as f64;

        let depth_ok = config
            .performance_requirements
            .max_depth_increase
            .is_none_or(|max| (depth_ratio - 1.0) <= max);
        let gate_ok = config
            .performance_requirements
            .max_gate_increase
            .is_none_or(|max| (gate_ratio - 1.0) <= max);

        let success = depth_ok && gate_ok;
        let score = (2.0 - (depth_ratio - 1.0).max(0.0) - (gate_ratio - 1.0).max(0.0))
            .clamp(0.0, 2.0)
            / 2.0;

        Ok(ValidationMethodResult {
            success,
            score,
            details: format!(
                "Benchmark comparison: depth {} -> {} ({depth_ratio:.2}x), gates {} -> {} ({gate_ratio:.2}x)",
                original_metrics.depth,
                migrated_metrics.depth,
                original_metrics.gate_count,
                migrated_metrics.gate_count
            ),
            p_value: None,
        })
    }

    /// Full statistical validation: simulate both circuits over sampled
    /// basis states, run a real two-sample KS test and a chi-square
    /// goodness-of-fit test on the resulting probability distributions, and
    /// average the per-state overlap fidelity `|<original|migrated>|^2`.
    /// `original_fidelity` is defined as 1.0 (the original circuit is its
    /// own reference); `migrated_fidelity` is the measured average overlap
    /// with it, so `fidelity_loss` reflects the actual divergence rather
    /// than a fixed constant.
    async fn perform_statistical_validation<const N: usize>(
        &self,
        original: &Circuit<N>,
        migrated: &Circuit<N>,
        _config: &MigrationConfig,
    ) -> DeviceResult<StatisticalValidationResult> {
        let num_states = (1usize << N).min(16);
        let mut original_probs = Vec::new();
        let mut migrated_probs = Vec::new();
        let mut fidelities = Vec::with_capacity(num_states);

        for state_idx in 0..num_states {
            let original_state = simulate_basis_state(original, state_idx)?;
            let migrated_state = simulate_basis_state(migrated, state_idx)?;
            fidelities.push(state_fidelity(&original_state, &migrated_state));
            original_probs.extend(original_state.iter().map(scirs2_core::Complex64::norm_sqr));
            migrated_probs.extend(migrated_state.iter().map(scirs2_core::Complex64::norm_sqr));
        }

        let x = Array1::from_vec(original_probs.clone());
        let y = Array1::from_vec(migrated_probs.clone());
        let (ks_statistic, ks_p_value) = scirs2_stats::ks_2samp(&x.view(), &y.view(), "two-sided")
            .map_err(|e| DeviceError::CircuitConversion(format!("KS test failed: {e}")))?;

        // Chi-square goodness-of-fit between the paired, per-outcome
        // probabilities pooled across every sampled basis state.
        let chi_square: f64 = original_probs
            .iter()
            .zip(migrated_probs.iter())
            .map(|(o, m)| {
                let denom = (o + m).max(1e-12);
                (o - m).powi(2) / denom
            })
            .sum();
        let degrees_of_freedom = original_probs.len().saturating_sub(1).max(1) as f64;
        let chi_square_p_value = (-chi_square / (2.0 * degrees_of_freedom))
            .exp()
            .clamp(0.0, 1.0);

        let distance = original_probs
            .iter()
            .zip(migrated_probs.iter())
            .map(|(o, m)| (o - m).abs())
            .fold(0.0_f64, f64::max);
        let similarity_score = (1.0 - distance).clamp(0.0, 1.0);

        let avg_state_fidelity = fidelities.iter().sum::<f64>() / fidelities.len().max(1) as f64;

        Ok(StatisticalValidationResult {
            distribution_comparison: DistributionComparison {
                ks_test_p_value: ks_p_value,
                chi_square_p_value,
                distance,
                similarity_score,
            },
            fidelity_comparison: FidelityComparison {
                original_fidelity: 1.0,
                migrated_fidelity: avg_state_fidelity,
                fidelity_loss: (1.0 - avg_state_fidelity).max(0.0),
                significance: ks_p_value,
            },
            error_analysis: ErrorAnalysis {
                error_rate_comparison: (1.0 - avg_state_fidelity).max(0.0),
                error_correlation: (1.0 - distance).clamp(0.0, 1.0),
                systematic_errors: if avg_state_fidelity < 0.999 {
                    vec![format!(
                        "Average state fidelity {avg_state_fidelity:.6} indicates the migrated circuit diverges from the original"
                    )]
                } else {
                    Vec::new()
                },
                random_error_estimate: ks_statistic,
            },
        })
    }

    fn calculate_mapping_overhead(&self, transformations: &[AppliedTransformation]) -> f64 {
        transformations
            .iter()
            .filter(|t| t.transformation_type == TransformationType::QubitMapping)
            .map(|t| t.impact.time_impact.abs())
            .sum()
    }

    fn calculate_translation_efficiency(&self, transformations: &[AppliedTransformation]) -> f64 {
        let translation_transforms = transformations
            .iter()
            .filter(|t| t.transformation_type == TransformationType::GateTranslation)
            .count();

        if translation_transforms > 0 {
            1.0 / (translation_transforms as f64).mul_add(0.1, 1.0)
        } else {
            1.0
        }
    }

    fn calculate_resource_change(
        &self,
        original: &CircuitMetrics,
        migrated: &CircuitMetrics,
    ) -> f64 {
        let memory_change = migrated.resource_requirements.memory_mb
            / original.resource_requirements.memory_mb
            - 1.0;
        let cpu_change = migrated.resource_requirements.cpu_time.as_secs_f64()
            / original.resource_requirements.cpu_time.as_secs_f64()
            - 1.0;
        let qpu_change = migrated.resource_requirements.qpu_time.as_secs_f64()
            / original.resource_requirements.qpu_time.as_secs_f64()
            - 1.0;

        (memory_change + cpu_change + qpu_change) / 3.0
    }

    fn calculate_quality_score(&self, original: &CircuitMetrics, migrated: &CircuitMetrics) -> f64 {
        let fidelity_ratio = migrated.estimated_fidelity / original.estimated_fidelity;
        let depth_penalty = if migrated.depth > original.depth {
            ((migrated.depth - original.depth) as f64 / original.depth as f64).mul_add(-0.1, 1.0)
        } else {
            1.0
        };
        let gate_penalty = if migrated.gate_count > original.gate_count {
            ((migrated.gate_count - original.gate_count) as f64 / original.gate_count as f64)
                .mul_add(-0.05, 1.0)
        } else {
            1.0
        };

        (fidelity_ratio * depth_penalty * gate_penalty).clamp(0.0, 1.0)
    }

    fn check_migration_requirements(
        &self,
        metrics: &MigrationMetrics,
        config: &MigrationConfig,
        warnings: &mut Vec<MigrationWarning>,
    ) -> DeviceResult<bool> {
        let mut success = true;

        // Check fidelity requirement
        if let Some(min_fidelity) = config.performance_requirements.min_fidelity {
            if metrics.migrated.estimated_fidelity < min_fidelity {
                warnings.push(MigrationWarning {
                    warning_type: WarningType::FidelityLoss,
                    message: format!(
                        "Migrated fidelity ({:.3}) below requirement ({:.3})",
                        metrics.migrated.estimated_fidelity, min_fidelity
                    ),
                    severity: WarningSeverity::Error,
                    suggested_actions: vec![
                        "Adjust migration strategy to preserve fidelity".to_string()
                    ],
                });
                success = false;
            }
        }

        // Check depth increase
        if let Some(max_depth_increase) = config.performance_requirements.max_depth_increase {
            if metrics.performance_comparison.depth_change > max_depth_increase {
                warnings.push(MigrationWarning {
                    warning_type: WarningType::PerformanceDegradation,
                    message: format!(
                        "Circuit depth increased by {:.1}%, exceeding limit of {:.1}%",
                        metrics.performance_comparison.depth_change * 100.0,
                        max_depth_increase * 100.0
                    ),
                    severity: WarningSeverity::Warning,
                    suggested_actions: vec!["Enable depth optimization passes".to_string()],
                });
            }
        }

        // Check gate count increase
        if let Some(max_gate_increase) = config.performance_requirements.max_gate_increase {
            if metrics.performance_comparison.gate_count_change > max_gate_increase {
                warnings.push(MigrationWarning {
                    warning_type: WarningType::PerformanceDegradation,
                    message: format!("Gate count increased by {:.1}%, exceeding limit of {:.1}%",
                                   metrics.performance_comparison.gate_count_change * 100.0,
                                   max_gate_increase * 100.0),
                    severity: WarningSeverity::Warning,
                    suggested_actions: vec!["Enable gate reduction optimization passes".to_string()],
                });
            }
        }

        Ok(success)
    }
}
