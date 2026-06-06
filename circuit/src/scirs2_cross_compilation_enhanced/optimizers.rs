//! ML-based optimization and compilation helpers
//!
//! This module contains the ML compilation optimizer, feature extractors,
//! and internal helper types for cross-compilation.

use super::config::{EnhancedCrossCompilationConfig, TargetPlatform};
use super::types::{IRGate, IROperation, IROperationType, QuantumIR, SourceCircuit, TargetCode};
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// ML compilation optimizer
pub struct MLCompilationOptimizer {
    config: EnhancedCrossCompilationConfig,
    model: Arc<Mutex<CompilationModel>>,
    feature_extractor: Arc<CompilationFeatureExtractor>,
}

impl MLCompilationOptimizer {
    pub fn new(config: EnhancedCrossCompilationConfig) -> Self {
        Self {
            config,
            model: Arc::new(Mutex::new(CompilationModel::new())),
            feature_extractor: Arc::new(CompilationFeatureExtractor::new()),
        }
    }

    pub fn optimize(&self, ir: &QuantumIR, target: TargetPlatform) -> QuantRS2Result<QuantumIR> {
        let features = self.feature_extractor.extract_features(ir, target)?;

        // Compute strategy, then drop the lock before applying transforms.
        let strategy = {
            let model = self
                .model
                .lock()
                .map_err(|e| QuantRS2Error::RuntimeError(format!("Model lock poisoned: {e}")))?;
            model.predict_strategy(&features)?
        };

        // Apply ML-guided optimizations using the predicted strategy.
        let optimized = Self::apply_ml_optimizations(ir, &strategy)?;

        Ok(optimized)
    }

    /// Apply ML-guided optimization transforms in sequence.
    ///
    /// When the strategy carries no explicit transformations (e.g., the model
    /// is a placeholder and returns an empty list), all four transforms are
    /// applied in a canonical order so this path is never a no-op.
    fn apply_ml_optimizations(
        ir: &QuantumIR,
        strategy: &MLOptimizationStrategy,
    ) -> QuantRS2Result<QuantumIR> {
        if strategy.transformations.is_empty() {
            // Fallback: apply all transforms in canonical order.
            let ir = Self::apply_rotation_merging_transform(ir)?;
            let ir = Self::apply_gate_fusion_transform(&ir)?;
            let ir = Self::apply_commutation_transform(&ir)?;
            let ir = Self::apply_decomposition_transform(&ir)?;
            return Ok(ir);
        }

        let mut current = ir.clone();
        for transform in &strategy.transformations {
            current = match transform.transform_type {
                TransformationType::GateFusion => Self::apply_gate_fusion_transform(&current)?,
                TransformationType::RotationMerging => {
                    Self::apply_rotation_merging_transform(&current)?
                }
                TransformationType::Commutation => Self::apply_commutation_transform(&current)?,
                TransformationType::Decomposition => Self::apply_decomposition_transform(&current)?,
            };
        }
        Ok(current)
    }

    // -----------------------------------------------------------------------
    // Private helpers: gate classification
    // -----------------------------------------------------------------------

    /// Returns true when the gate acts on exactly one qubit (single-qubit gates).
    fn is_single_qubit_gate(gate: &IRGate) -> bool {
        matches!(
            gate,
            IRGate::H
                | IRGate::X
                | IRGate::Y
                | IRGate::Z
                | IRGate::S
                | IRGate::T
                | IRGate::RX(_)
                | IRGate::RY(_)
                | IRGate::RZ(_)
                | IRGate::U1(_)
                | IRGate::U2(_, _)
                | IRGate::U3(_, _, _)
        )
    }

    /// Extract the qubit set for an operation (all qubits involved, including controls).
    fn op_qubits(op: &IROperation) -> Vec<usize> {
        let mut q = op.qubits.clone();
        q.extend_from_slice(&op.controls);
        q.sort_unstable();
        q.dedup();
        q
    }

    /// Returns true when the two operations act on entirely disjoint qubit sets.
    fn qubits_are_disjoint(a: &IROperation, b: &IROperation) -> bool {
        let qa = Self::op_qubits(a);
        let qb = Self::op_qubits(b);
        !qa.iter().any(|q| qb.contains(q))
    }

    // -----------------------------------------------------------------------
    // Transform: RotationMerging
    // -----------------------------------------------------------------------

    /// Combine consecutive same-type rotation gates on the same qubit by
    /// summing their angles (mod 2π).  If the resulting angle is < ε the
    /// gate pair is dropped entirely.
    fn apply_rotation_merging_transform(ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        const EPSILON: f64 = 1e-9;
        let ops = &ir.operations;
        let mut result: Vec<IROperation> = Vec::with_capacity(ops.len());

        for op in ops {
            let merged = if let Some(last) = result.last_mut() {
                // Only merge if both are single-qubit gates on the same single qubit.
                if last.qubits.len() == 1 && op.qubits.len() == 1 && last.qubits[0] == op.qubits[0]
                {
                    Self::try_merge_rotations(&last.operation_type, &op.operation_type)
                } else {
                    None
                }
            } else {
                None
            };

            match merged {
                Some(Some(merged_type)) => {
                    // Replace the last operation with the merged gate.
                    let last = result.last_mut().ok_or_else(|| {
                        QuantRS2Error::RuntimeError("Internal merge error".to_string())
                    })?;
                    last.operation_type = merged_type;
                }
                Some(None) => {
                    // Angle sums to ~0 — remove the last gate entirely.
                    result.pop();
                }
                None => {
                    result.push(op.clone());
                }
            }
        }

        let mut out = ir.clone();
        out.operations = result;
        Ok(out)
    }

    /// Try to merge two consecutive `IROperationType` values into one rotation.
    ///
    /// Returns:
    /// - `Some(Some(merged))` — successfully merged.
    /// - `Some(None)` — angle cancelled to zero; remove both.
    /// - `None` — not mergeable.
    fn try_merge_rotations(
        a: &IROperationType,
        b: &IROperationType,
    ) -> Option<Option<IROperationType>> {
        const EPSILON: f64 = 1e-9;
        let two_pi = 2.0 * PI;

        match (a, b) {
            (IROperationType::Gate(IRGate::RX(t1)), IROperationType::Gate(IRGate::RX(t2))) => {
                let sum = (t1 + t2).rem_euclid(two_pi);
                if sum.abs() < EPSILON || (sum - two_pi).abs() < EPSILON {
                    Some(None)
                } else {
                    Some(Some(IROperationType::Gate(IRGate::RX(sum))))
                }
            }
            (IROperationType::Gate(IRGate::RY(t1)), IROperationType::Gate(IRGate::RY(t2))) => {
                let sum = (t1 + t2).rem_euclid(two_pi);
                if sum.abs() < EPSILON || (sum - two_pi).abs() < EPSILON {
                    Some(None)
                } else {
                    Some(Some(IROperationType::Gate(IRGate::RY(sum))))
                }
            }
            (IROperationType::Gate(IRGate::RZ(t1)), IROperationType::Gate(IRGate::RZ(t2))) => {
                let sum = (t1 + t2).rem_euclid(two_pi);
                if sum.abs() < EPSILON || (sum - two_pi).abs() < EPSILON {
                    Some(None)
                } else {
                    Some(Some(IROperationType::Gate(IRGate::RZ(sum))))
                }
            }
            (IROperationType::Gate(IRGate::U1(t1)), IROperationType::Gate(IRGate::U1(t2))) => {
                let sum = (t1 + t2).rem_euclid(two_pi);
                if sum.abs() < EPSILON || (sum - two_pi).abs() < EPSILON {
                    Some(None)
                } else {
                    Some(Some(IROperationType::Gate(IRGate::U1(sum))))
                }
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Transform: GateFusion
    // -----------------------------------------------------------------------

    /// Fuse consecutive single-qubit gates on the same qubit where possible.
    ///
    /// This is a superset of `RotationMerging`: same-type rotations are merged
    /// by angle addition; other pairs are left as-is (no arbitrary matrix
    /// multiply path exists without a linear-algebra dependency).
    fn apply_gate_fusion_transform(ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        // For same-type rotation gates delegation is sufficient.
        // The rotation merging pass already handles the common case.
        // Here we run it again and additionally handle X–X, Y–Y, Z–Z, H–H
        // (each pair is the identity and can be dropped).
        const EPSILON: f64 = 1e-9;
        let ops = &ir.operations;
        let mut result: Vec<IROperation> = Vec::with_capacity(ops.len());

        for op in ops {
            let action = if let Some(last) = result.last() {
                if last.qubits.len() == 1 && op.qubits.len() == 1 && last.qubits[0] == op.qubits[0]
                {
                    // Try rotation merge first.
                    let rotation_merge =
                        Self::try_merge_rotations(&last.operation_type, &op.operation_type);
                    if rotation_merge.is_some() {
                        rotation_merge.map(|inner| ("rotation", inner))
                    } else {
                        // Check self-inverse pairs: gate ∘ gate = I.
                        Self::try_fuse_self_inverse(&last.operation_type, &op.operation_type)
                            .map(|_| ("cancel", None))
                    }
                } else {
                    None
                }
            } else {
                None
            };

            match action {
                Some(("rotation", Some(merged_type))) => {
                    let last = result.last_mut().ok_or_else(|| {
                        QuantRS2Error::RuntimeError("Internal fusion error".to_string())
                    })?;
                    last.operation_type = merged_type;
                }
                Some((_, None)) => {
                    // Both cancelled — remove the last gate.
                    result.pop();
                }
                _ => {
                    result.push(op.clone());
                }
            }
        }

        let mut out = ir.clone();
        out.operations = result;
        Ok(out)
    }

    /// Returns `Some(())` when `a ∘ b = I` (self-inverse pairs).
    fn try_fuse_self_inverse(a: &IROperationType, b: &IROperationType) -> Option<()> {
        match (a, b) {
            (IROperationType::Gate(IRGate::H), IROperationType::Gate(IRGate::H))
            | (IROperationType::Gate(IRGate::X), IROperationType::Gate(IRGate::X))
            | (IROperationType::Gate(IRGate::Y), IROperationType::Gate(IRGate::Y))
            | (IROperationType::Gate(IRGate::Z), IROperationType::Gate(IRGate::Z))
            | (IROperationType::Gate(IRGate::CNOT), IROperationType::Gate(IRGate::CNOT))
            | (IROperationType::Gate(IRGate::CZ), IROperationType::Gate(IRGate::CZ)) => Some(()),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Transform: Commutation
    // -----------------------------------------------------------------------

    /// Reorder gates where safe to enable downstream fusion passes.
    ///
    /// Single forward pass: for each gate at position i, if it commutes with
    /// the gate immediately before it (disjoint qubit sets) AND swapping would
    /// place it adjacent to an earlier gate of the same type on the same qubit,
    /// swap the pair.  This is deliberately conservative and O(n).
    fn apply_commutation_transform(ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        let mut ops = ir.operations.clone();
        let n = ops.len();

        let mut i = 1;
        while i < n {
            let commutes = Self::qubits_are_disjoint(&ops[i - 1], &ops[i]);
            if commutes {
                // Check if swapping places ops[i] adjacent to a same-type
                // same-qubit gate further back.
                let enables_fusion = i >= 2
                    && ops[i].qubits == ops[i - 2].qubits
                    && std::mem::discriminant(&ops[i].operation_type)
                        == std::mem::discriminant(&ops[i - 2].operation_type);
                if enables_fusion {
                    ops.swap(i - 1, i);
                }
            }
            i += 1;
        }

        let mut out = ir.clone();
        out.operations = ops;
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Transform: Decomposition
    // -----------------------------------------------------------------------

    /// Rewrite compound gates into hardware-primitive sequences.
    ///
    /// Supported decompositions:
    /// - `Toffoli` (CCX, 3-qubit) → 15-gate sequence using H, CNOT, T, U1(−π/4).
    /// - `SWAP` → three CNOT gates.
    /// - `Fredkin` (CSWAP) → CNOT + Toffoli + CNOT (further decomposed inline).
    ///
    /// All other gates pass through unchanged.
    fn apply_decomposition_transform(ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        let mut out_ops: Vec<IROperation> = Vec::new();

        for op in &ir.operations {
            match &op.operation_type {
                IROperationType::Gate(IRGate::Toffoli) if op.qubits.len() >= 3 => {
                    let (c1, c2, t) = (op.qubits[0], op.qubits[1], op.qubits[2]);
                    out_ops.extend(Self::decompose_toffoli(c1, c2, t));
                }
                IROperationType::Gate(IRGate::SWAP) if op.qubits.len() >= 2 => {
                    let (a, b) = (op.qubits[0], op.qubits[1]);
                    out_ops.extend(Self::decompose_swap(a, b));
                }
                IROperationType::Gate(IRGate::Fredkin) if op.qubits.len() >= 3 => {
                    let (ctrl, a, b) = (op.qubits[0], op.qubits[1], op.qubits[2]);
                    out_ops.extend(Self::decompose_fredkin(ctrl, a, b));
                }
                _ => {
                    out_ops.push(op.clone());
                }
            }
        }

        let mut result = ir.clone();
        result.operations = out_ops;
        Ok(result)
    }

    /// Build a simple single-qubit `IROperation` for the given gate.
    fn single_qubit_op(gate: IRGate, qubit: usize) -> IROperation {
        IROperation {
            operation_type: IROperationType::Gate(gate),
            qubits: vec![qubit],
            controls: vec![],
            parameters: vec![],
        }
    }

    /// Build a two-qubit `IROperation` for the given gate.
    fn two_qubit_op(gate: IRGate, q0: usize, q1: usize) -> IROperation {
        IROperation {
            operation_type: IROperationType::Gate(gate),
            qubits: vec![q0, q1],
            controls: vec![],
            parameters: vec![],
        }
    }

    /// Toffoli (CCX) → standard 15-gate decomposition.
    ///
    /// `Tdg` is not a named variant; we represent T† as `U1(−π/4)`.
    /// Layout: qubits = [c1, c2, t]
    fn decompose_toffoli(c1: usize, c2: usize, t: usize) -> Vec<IROperation> {
        let tdg = |q| Self::single_qubit_op(IRGate::U1(-PI / 4.0), q);
        let tgate = |q| Self::single_qubit_op(IRGate::T, q);
        let hgate = |q| Self::single_qubit_op(IRGate::H, q);
        let cnot = |ctrl, tgt| Self::two_qubit_op(IRGate::CNOT, ctrl, tgt);

        vec![
            hgate(t),
            cnot(c2, t),
            tdg(t),
            cnot(c1, t),
            tgate(t),
            cnot(c2, t),
            tdg(t),
            cnot(c1, t),
            tgate(c2),
            tgate(t),
            hgate(t),
            cnot(c1, c2),
            tgate(c1),
            tdg(c2),
            cnot(c1, c2),
        ]
    }

    /// SWAP → three CNOT gates.
    fn decompose_swap(a: usize, b: usize) -> Vec<IROperation> {
        vec![
            Self::two_qubit_op(IRGate::CNOT, a, b),
            Self::two_qubit_op(IRGate::CNOT, b, a),
            Self::two_qubit_op(IRGate::CNOT, a, b),
        ]
    }

    /// Fredkin (CSWAP, ctrl a b) → CNOT(b,a), Toffoli(ctrl,a,b), CNOT(b,a).
    fn decompose_fredkin(ctrl: usize, a: usize, b: usize) -> Vec<IROperation> {
        let mut ops = vec![Self::two_qubit_op(IRGate::CNOT, b, a)];
        ops.extend(Self::decompose_toffoli(ctrl, a, b));
        ops.push(Self::two_qubit_op(IRGate::CNOT, b, a));
        ops
    }
}

/// Compilation monitor
pub struct CompilationMonitor {
    config: EnhancedCrossCompilationConfig,
    metrics: Arc<Mutex<CompilationMetrics>>,
}

impl CompilationMonitor {
    pub fn new(config: EnhancedCrossCompilationConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(Mutex::new(CompilationMetrics::new())),
        }
    }

    pub fn update_optimization_progress(&self, ir: &QuantumIR) -> QuantRS2Result<()> {
        let anomaly = {
            let mut metrics = self
                .metrics
                .lock()
                .map_err(|e| QuantRS2Error::RuntimeError(format!("Metrics lock poisoned: {e}")))?;
            metrics.update(ir)?;
            metrics.detect_anomaly()
        }; // Early drop the lock guard

        // Check for anomalies
        if anomaly {
            // Handle anomaly
        }

        Ok(())
    }
}

/// Compilation validator
pub struct CompilationValidator {
    config: EnhancedCrossCompilationConfig,
}

impl CompilationValidator {
    pub const fn new(config: EnhancedCrossCompilationConfig) -> Self {
        Self { config }
    }

    pub fn validate_compilation(
        &self,
        source: &SourceCircuit,
        target_code: &TargetCode,
        platform: TargetPlatform,
    ) -> QuantRS2Result<super::types::ValidationResult> {
        let mut result = super::types::ValidationResult::new();

        // Semantic validation
        if self.config.base_config.preserve_semantics {
            let semantic_valid = self.validate_semantics(source, target_code)?;
            result.semantic_validation = Some(semantic_valid);
        }

        // Resource validation
        let resource_valid = self.validate_resources(target_code, platform)?;
        result.resource_validation = Some(resource_valid);

        // Fidelity validation
        let fidelity = self.estimate_fidelity(source, target_code)?;
        result.fidelity_estimate = Some(fidelity);

        result.is_valid = result.semantic_validation.unwrap_or(true)
            && result.resource_validation.unwrap_or(true)
            && fidelity >= self.config.base_config.validation_threshold;

        Ok(result)
    }

    pub const fn validate_semantics(
        &self,
        _source: &SourceCircuit,
        _target: &TargetCode,
    ) -> QuantRS2Result<bool> {
        // Semantic validation logic
        Ok(true)
    }

    pub const fn validate_resources(
        &self,
        _target: &TargetCode,
        _platform: TargetPlatform,
    ) -> QuantRS2Result<bool> {
        // Resource validation logic
        Ok(true)
    }

    pub const fn estimate_fidelity(
        &self,
        _source: &SourceCircuit,
        _target: &TargetCode,
    ) -> QuantRS2Result<f64> {
        // Fidelity estimation logic
        Ok(0.99)
    }
}

/// ML optimization strategy
pub struct MLOptimizationStrategy {
    pub transformations: Vec<IRTransformation>,
    pub confidence: f64,
}

/// IR transformation
pub struct IRTransformation {
    pub transform_type: TransformationType,
    pub parameters: HashMap<String, f64>,
}

/// Transformation type
pub enum TransformationType {
    GateFusion,
    RotationMerging,
    Commutation,
    Decomposition,
}

/// Compilation model
pub struct CompilationModel {
    // ML model implementation
}

impl CompilationModel {
    pub const fn new() -> Self {
        Self {}
    }

    pub const fn predict_strategy(
        &self,
        _features: &CompilationFeatures,
    ) -> QuantRS2Result<MLOptimizationStrategy> {
        // Placeholder implementation
        Ok(MLOptimizationStrategy {
            transformations: vec![],
            confidence: 0.9,
        })
    }
}

impl Default for CompilationModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Compilation feature extractor
pub struct CompilationFeatureExtractor {
    // Feature extraction logic
}

impl CompilationFeatureExtractor {
    pub const fn new() -> Self {
        Self {}
    }

    pub const fn extract_features(
        &self,
        _ir: &QuantumIR,
        _target: TargetPlatform,
    ) -> QuantRS2Result<CompilationFeatures> {
        Ok(CompilationFeatures {
            circuit_features: vec![],
            target_features: vec![],
            complexity_features: vec![],
        })
    }
}

impl Default for CompilationFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compilation features
pub struct CompilationFeatures {
    pub circuit_features: Vec<f64>,
    pub target_features: Vec<f64>,
    pub complexity_features: Vec<f64>,
}

/// Compilation metrics
pub struct CompilationMetrics {
    pub gate_count: usize,
    pub circuit_depth: usize,
    pub optimization_count: usize,
}

impl CompilationMetrics {
    pub const fn new() -> Self {
        Self {
            gate_count: 0,
            circuit_depth: 0,
            optimization_count: 0,
        }
    }

    pub fn update(&mut self, ir: &QuantumIR) -> QuantRS2Result<()> {
        self.gate_count = ir.operations.len();
        // Calculate depth and other metrics
        Ok(())
    }

    pub const fn detect_anomaly(&self) -> bool {
        // Simple anomaly detection
        false
    }
}

impl Default for CompilationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Target specification
pub struct TargetSpecification {
    pub native_gates: Vec<IRGate>,
    pub connectivity: Vec<(usize, usize)>,
    pub error_rates: HashMap<String, f64>,
}

/// Compilation cache
pub struct CompilationCache {
    pub cache: HashMap<(String, TargetPlatform), super::types::CrossCompilationResult>,
}

impl CompilationCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
}

impl Default for CompilationCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Build a minimal QuantumIR with the given operations.
    fn build_ir(num_qubits: usize, ops: Vec<IROperation>) -> QuantumIR {
        QuantumIR {
            num_qubits,
            num_classical_bits: 0,
            operations: ops,
            classical_operations: vec![],
            metadata: HashMap::new(),
        }
    }

    // Build a simple single-qubit gate operation.
    fn single_gate(gate: IRGate, qubit: usize) -> IROperation {
        IROperation {
            operation_type: IROperationType::Gate(gate),
            qubits: vec![qubit],
            controls: vec![],
            parameters: vec![],
        }
    }

    // Build a two-qubit gate operation.
    fn two_qubit_gate(gate: IRGate, q0: usize, q1: usize) -> IROperation {
        IROperation {
            operation_type: IROperationType::Gate(gate),
            qubits: vec![q0, q1],
            controls: vec![],
            parameters: vec![],
        }
    }

    // Build a three-qubit gate operation.
    fn three_qubit_gate(gate: IRGate, q0: usize, q1: usize, q2: usize) -> IROperation {
        IROperation {
            operation_type: IROperationType::Gate(gate),
            qubits: vec![q0, q1, q2],
            controls: vec![],
            parameters: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // RotationMerging tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rotation_merging_combines_rx_angles() {
        let ir = build_ir(
            1,
            vec![
                single_gate(IRGate::RX(0.5), 0),
                single_gate(IRGate::RX(0.3), 0),
            ],
        );
        let result = MLCompilationOptimizer::apply_rotation_merging_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            1,
            "two RX gates should merge to one"
        );
        match &result.operations[0].operation_type {
            IROperationType::Gate(IRGate::RX(angle)) => {
                let expected = (0.5f64 + 0.3).rem_euclid(2.0 * std::f64::consts::PI);
                assert!(
                    (angle - expected).abs() < 1e-9,
                    "merged angle should be 0.8, got {angle}"
                );
            }
            other => panic!("expected RX gate, got {other:?}"),
        }
    }

    #[test]
    fn test_rotation_merging_removes_cancelling_rx() {
        let angle = std::f64::consts::PI;
        let ir = build_ir(
            1,
            vec![
                single_gate(IRGate::RX(angle), 0),
                single_gate(IRGate::RX(-angle), 0),
            ],
        );
        let result = MLCompilationOptimizer::apply_rotation_merging_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            0,
            "RX(π) + RX(-π) should cancel to zero gates"
        );
    }

    #[test]
    fn test_rotation_merging_different_qubits_unchanged() {
        let ir = build_ir(
            2,
            vec![
                single_gate(IRGate::RX(0.5), 0),
                single_gate(IRGate::RX(0.5), 1), // different qubit — no merge
            ],
        );
        let result = MLCompilationOptimizer::apply_rotation_merging_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            2,
            "gates on different qubits must not merge"
        );
    }

    #[test]
    fn test_rotation_merging_different_types_unchanged() {
        let ir = build_ir(
            1,
            vec![
                single_gate(IRGate::RX(0.5), 0),
                single_gate(IRGate::RY(0.5), 0), // different type — no merge
            ],
        );
        let result = MLCompilationOptimizer::apply_rotation_merging_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            2,
            "RX + RY on same qubit must not merge"
        );
    }

    // -----------------------------------------------------------------------
    // GateFusion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gate_fusion_reduces_same_type_rotations() {
        let ir = build_ir(
            1,
            vec![
                single_gate(IRGate::RZ(1.0), 0),
                single_gate(IRGate::RZ(0.5), 0),
            ],
        );
        let result = MLCompilationOptimizer::apply_gate_fusion_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            1,
            "consecutive RZ on same qubit should fuse to 1 gate"
        );
    }

    #[test]
    fn test_gate_fusion_cancels_h_h() {
        // H ∘ H = I
        let ir = build_ir(
            1,
            vec![single_gate(IRGate::H, 0), single_gate(IRGate::H, 0)],
        );
        let result = MLCompilationOptimizer::apply_gate_fusion_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            0,
            "H followed by H should cancel to zero gates"
        );
    }

    #[test]
    fn test_gate_fusion_cancels_x_x() {
        let ir = build_ir(
            1,
            vec![single_gate(IRGate::X, 0), single_gate(IRGate::X, 0)],
        );
        let result = MLCompilationOptimizer::apply_gate_fusion_transform(&ir).unwrap();
        assert_eq!(result.operations.len(), 0, "X ∘ X should cancel");
    }

    // -----------------------------------------------------------------------
    // Commutation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_commutation_reorders_disjoint_qubits() {
        // Circuit: RX(q=0), RX(q=1), RX(q=0)
        // Gate at i=1 (q=1) commutes with i=0 (q=0) — disjoint.
        // After swap, i=0 is RX(q=1) and i=1 is RX(q=0), which is NOT i-2 check.
        // After second swap opportunity at i=2, ops[2] (q=0) vs ops[1] (q=0):
        // they don't commute (same qubit).
        // The test verifies that at minimum the function completes without error
        // and returns valid gate count.
        let ir = build_ir(
            2,
            vec![
                single_gate(IRGate::RX(0.5), 0),
                single_gate(IRGate::RX(0.5), 1),
                single_gate(IRGate::RX(0.3), 0),
            ],
        );
        let result = MLCompilationOptimizer::apply_commutation_transform(&ir).unwrap();
        // Gate count is unchanged by commutation.
        assert_eq!(
            result.operations.len(),
            3,
            "commutation preserves gate count"
        );
    }

    #[test]
    fn test_commutation_enables_downstream_fusion() {
        // Circuit: RX(q=0), RX(q=1), RX(q=0)
        // After commutation the RX(q=0) at position 2 should be moved next to
        // RX(q=0) at position 0 (since RX(q=1) commutes with both).
        let ir = build_ir(
            2,
            vec![
                single_gate(IRGate::RX(0.5), 0),
                single_gate(IRGate::RX(0.5), 1), // commutes with neighbors on q=0
                single_gate(IRGate::RX(0.3), 0),
            ],
        );
        let commuted = MLCompilationOptimizer::apply_commutation_transform(&ir).unwrap();
        // After commutation + fusion we should get 2 ops (one merged RX on q=0,
        // one RX on q=1) instead of 3.
        let fused = MLCompilationOptimizer::apply_rotation_merging_transform(&commuted).unwrap();
        assert_eq!(
            fused.operations.len(),
            2,
            "commutation + rotation-merge should collapse two RX(q=0) into one"
        );
    }

    // -----------------------------------------------------------------------
    // Decomposition tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decomposition_toffoli_produces_15_gates() {
        let ir = build_ir(3, vec![three_qubit_gate(IRGate::Toffoli, 0, 1, 2)]);
        let result = MLCompilationOptimizer::apply_decomposition_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            15,
            "Toffoli should decompose into exactly 15 primitive gates"
        );
    }

    #[test]
    fn test_decomposition_swap_produces_3_cnots() {
        let ir = build_ir(2, vec![two_qubit_gate(IRGate::SWAP, 0, 1)]);
        let result = MLCompilationOptimizer::apply_decomposition_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            3,
            "SWAP should decompose into exactly 3 CNOT gates"
        );
        for op in &result.operations {
            assert!(
                matches!(&op.operation_type, IROperationType::Gate(IRGate::CNOT)),
                "each SWAP decomposition gate should be a CNOT, got {:?}",
                op.operation_type
            );
        }
    }

    #[test]
    fn test_decomposition_non_compound_passes_through() {
        let ir = build_ir(
            1,
            vec![single_gate(IRGate::H, 0), single_gate(IRGate::RX(1.0), 0)],
        );
        let result = MLCompilationOptimizer::apply_decomposition_transform(&ir).unwrap();
        assert_eq!(
            result.operations.len(),
            2,
            "non-compound gates should pass through unchanged"
        );
    }

    // -----------------------------------------------------------------------
    // End-to-end apply_ml_optimizations test
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_ml_optimizations_fallback_path() {
        // Verify the fallback (empty strategy) path executes without error.
        let strategy = MLOptimizationStrategy {
            transformations: vec![],
            confidence: 0.9,
        };
        let ir = build_ir(
            1,
            vec![
                single_gate(IRGate::RX(0.5), 0),
                single_gate(IRGate::RX(0.5), 0),
            ],
        );
        let result = MLCompilationOptimizer::apply_ml_optimizations(&ir, &strategy).unwrap();
        // After rotation merging both RX gates should collapse to one.
        assert_eq!(
            result.operations.len(),
            1,
            "fallback path should apply rotation merging and fuse the two RX gates"
        );
    }
}
