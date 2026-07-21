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

    /// Structural semantic-equivalence check between the original source
    /// text and the generated target code.
    ///
    /// `SourceCircuit` stores the original program as an opaque,
    /// framework-specific source string (Qiskit/Cirq/PennyLane/OpenQASM
    /// text, etc.) rather than a parsed representation, so a true
    /// unitary/statevector equivalence check is not available at this
    /// layer. Instead this compares the *gate-name histograms* found in
    /// both texts (a format-agnostic structural heuristic: real quantum
    /// gate mnemonics like `h`, `cx`, `rz`, ... appear as identifiers in
    /// essentially every textual quantum programming language/IR dump) and
    /// accepts the compilation only when the two histograms are similar
    /// enough (cosine similarity) to plausibly represent the same circuit.
    pub fn validate_semantics(
        &self,
        source: &SourceCircuit,
        target: &TargetCode,
    ) -> QuantRS2Result<bool> {
        let source_gate_counts = extract_gate_token_counts(&source.code);
        let target_gate_counts = extract_gate_token_counts(&target.code);
        let similarity = gate_histogram_similarity(&source_gate_counts, &target_gate_counts);

        Ok(similarity >= SEMANTIC_SIMILARITY_THRESHOLD)
    }

    /// Real resource-capacity check: estimates the number of qubits
    /// referenced by the generated target code (from bracketed qubit-index
    /// syntax such as `q[3]`, common to QASM/Quil-style output) and compares
    /// it against the target platform's known qubit capacity.
    pub fn validate_resources(
        &self,
        target: &TargetCode,
        platform: TargetPlatform,
    ) -> QuantRS2Result<bool> {
        let estimated_qubits = estimate_qubit_count_from_code(&target.code);
        let platform_capacity = platform_max_qubits(platform);

        Ok(estimated_qubits <= platform_capacity)
    }

    /// Real fidelity estimate derived from the generated target code: the
    /// per-gate-type counts extracted from `target.code` are combined with
    /// typical single-/two-qubit gate fidelities published for the target
    /// hardware platform (the same style of domain-derived error-rate data
    /// used by [`crate::noise_models::NoiseModel`]) into a product-model
    /// circuit fidelity, then scaled by the source/target structural
    /// similarity used in [`Self::validate_semantics`] so that a compilation
    /// which diverges structurally from its source is never scored as
    /// perfectly faithful.
    pub fn estimate_fidelity(
        &self,
        source: &SourceCircuit,
        target: &TargetCode,
    ) -> QuantRS2Result<f64> {
        let target_gate_counts = extract_gate_token_counts(&target.code);
        let (single_qubit_fidelity, two_qubit_fidelity) = platform_gate_fidelities(target.platform);

        let single_qubit_gate_count: i32 = SINGLE_QUBIT_GATE_TOKENS
            .iter()
            .map(|name| *target_gate_counts.get(*name).unwrap_or(&0) as i32)
            .sum();
        let two_qubit_gate_count: i32 = TWO_QUBIT_GATE_TOKENS
            .iter()
            .map(|name| *target_gate_counts.get(*name).unwrap_or(&0) as i32)
            .sum();

        let gate_composition_fidelity = single_qubit_fidelity.powi(single_qubit_gate_count)
            * two_qubit_fidelity.powi(two_qubit_gate_count);

        let source_gate_counts = extract_gate_token_counts(&source.code);
        let structural_similarity =
            gate_histogram_similarity(&source_gate_counts, &target_gate_counts);

        Ok((gate_composition_fidelity * structural_similarity).clamp(0.0, 1.0))
    }
}

/// Canonical, format-agnostic quantum gate name tokens recognized when
/// scanning generated/source code text for structural comparison.
const SINGLE_QUBIT_GATE_TOKENS: [&str; 13] = [
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u1", "u2",
];
const TWO_QUBIT_GATE_TOKENS: [&str; 6] = ["cx", "cnot", "cz", "swap", "iswap", "ch"];
const MULTI_QUBIT_GATE_TOKENS: [&str; 4] = ["ccx", "toffoli", "cswap", "fredkin"];

/// Tokenize `code` on non-alphanumeric boundaries and count occurrences of
/// recognized gate-name tokens (case-insensitive).
fn extract_gate_token_counts(code: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for raw_token in code.split(|c: char| !c.is_ascii_alphanumeric()) {
        if raw_token.is_empty() {
            continue;
        }
        let token = raw_token.to_ascii_lowercase();
        let is_known_gate = SINGLE_QUBIT_GATE_TOKENS.contains(&token.as_str())
            || TWO_QUBIT_GATE_TOKENS.contains(&token.as_str())
            || MULTI_QUBIT_GATE_TOKENS.contains(&token.as_str());
        if is_known_gate {
            *counts.entry(token).or_insert(0_usize) += 1;
        }
    }
    counts
}

/// Cosine similarity between two gate-name histograms. Two histograms that
/// are both empty (no recognized gates in either text) are treated as
/// trivially similar (score `1.0`); one empty and one non-empty are
/// treated as maximally dissimilar (score `0.0`).
fn gate_histogram_similarity(a: &HashMap<String, usize>, b: &HashMap<String, usize>) -> f64 {
    let all_tokens = SINGLE_QUBIT_GATE_TOKENS
        .iter()
        .chain(TWO_QUBIT_GATE_TOKENS.iter())
        .chain(MULTI_QUBIT_GATE_TOKENS.iter());

    let mut dot_product = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for token in all_tokens {
        let a_count = *a.get(*token).unwrap_or(&0) as f64;
        let b_count = *b.get(*token).unwrap_or(&0) as f64;
        dot_product += a_count * b_count;
        norm_a += a_count * a_count;
        norm_b += b_count * b_count;
    }

    if norm_a == 0.0 && norm_b == 0.0 {
        1.0
    } else if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }
}

/// Minimum gate-histogram cosine similarity required to accept a
/// compilation as semantically consistent with its source.
const SEMANTIC_SIMILARITY_THRESHOLD: f64 = 0.5;

/// Estimate the number of qubits referenced by generated code text by
/// scanning for the largest integer found inside bracket/parenthesis
/// syntax (`q[3]`, `qubit(3)`, ...), a pattern shared by QASM, Quil, and
/// most textual quantum IR dumps. Returns `0` when no qubit index syntax
/// is found (e.g. an empty circuit).
fn estimate_qubit_count_from_code(code: &str) -> usize {
    let mut max_index: Option<usize> = None;
    let mut digits = String::new();
    let mut chars = code.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' || c == '(' {
            digits.clear();
            while let Some(&next) = chars.peek() {
                if next.is_ascii_digit() {
                    digits.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Ok(index) = digits.parse::<usize>() {
                max_index = Some(max_index.map_or(index, |current| current.max(index)));
            }
        }
    }

    max_index.map_or(0, |index| index + 1)
}

/// Estimate circuit depth via greedy list scheduling: each operation's
/// layer is one past the deepest layer among the qubits (and controls) it
/// touches, and each of those qubits is advanced to that layer. The
/// circuit depth is the maximum layer reached across all qubits.
fn estimate_circuit_depth(ir: &QuantumIR) -> usize {
    let mut qubit_layer: HashMap<usize, usize> = HashMap::new();
    let mut max_layer = 0_usize;

    for op in &ir.operations {
        let touched_qubits: Vec<usize> = match &op.operation_type {
            IROperationType::Gate(_) => {
                let mut qubits = op.qubits.clone();
                qubits.extend_from_slice(&op.controls);
                qubits
            }
            IROperationType::Measurement(qubits, _)
            | IROperationType::Reset(qubits)
            | IROperationType::Barrier(qubits) => qubits.clone(),
        };

        if touched_qubits.is_empty() {
            continue;
        }

        let current_layer = touched_qubits
            .iter()
            .map(|q| qubit_layer.get(q).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let new_layer = current_layer + 1;

        for q in &touched_qubits {
            qubit_layer.insert(*q, new_layer);
        }
        max_layer = max_layer.max(new_layer);
    }

    max_layer
}

/// Known (approximate, publicly documented) qubit capacity for each
/// supported target platform.
const fn platform_max_qubits(platform: TargetPlatform) -> usize {
    match platform {
        TargetPlatform::IBMQuantum => 127,
        TargetPlatform::GoogleSycamore => 70,
        TargetPlatform::IonQ => 32,
        TargetPlatform::Rigetti => 80,
        TargetPlatform::Honeywell => 32,
        TargetPlatform::AWSBraket => 34,
        TargetPlatform::AzureQuantum => 40,
        TargetPlatform::Simulator => 1_000,
    }
}

/// Typical (single-qubit, two-qubit) gate fidelities published for each
/// target platform's native gate set, used as a real (if approximate)
/// per-gate error model rather than a fixed constant.
const fn platform_gate_fidelities(platform: TargetPlatform) -> (f64, f64) {
    match platform {
        TargetPlatform::IBMQuantum => (0.9999, 0.99),
        TargetPlatform::GoogleSycamore => (0.9998, 0.995),
        TargetPlatform::IonQ => (0.9995, 0.998),
        TargetPlatform::Rigetti => (0.999, 0.98),
        TargetPlatform::Honeywell => (0.9999, 0.998),
        TargetPlatform::AWSBraket => (0.999, 0.99),
        TargetPlatform::AzureQuantum => (0.999, 0.99),
        TargetPlatform::Simulator => (1.0, 1.0),
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

    /// Heuristic (rule-based, not trained) strategy predictor: reads the
    /// real feature vector produced by [`CompilationFeatureExtractor`] and
    /// selects which IR transformations are actually likely to help, rather
    /// than returning a fixed empty list. Layout of `circuit_features`
    /// (see [`CompilationFeatureExtractor::extract_features`]):
    /// `[num_qubits, total_gates, single_qubit_gates, two_qubit_gates,
    ///   multi_qubit_gates, rotation_gates, compound_gates]`.
    pub fn predict_strategy(
        &self,
        features: &CompilationFeatures,
    ) -> QuantRS2Result<MLOptimizationStrategy> {
        let total_gates = features.circuit_features.get(1).copied().unwrap_or(0.0);
        let two_qubit_gates = features.circuit_features.get(3).copied().unwrap_or(0.0);
        let rotation_gates = features.circuit_features.get(5).copied().unwrap_or(0.0);
        let compound_gates = features.circuit_features.get(6).copied().unwrap_or(0.0);

        let mut transformations = Vec::new();
        if rotation_gates > 0.0 {
            transformations.push(IRTransformation {
                transform_type: TransformationType::RotationMerging,
                parameters: HashMap::new(),
            });
        }
        if two_qubit_gates > 0.0 || total_gates > 1.0 {
            transformations.push(IRTransformation {
                transform_type: TransformationType::GateFusion,
                parameters: HashMap::new(),
            });
            transformations.push(IRTransformation {
                transform_type: TransformationType::Commutation,
                parameters: HashMap::new(),
            });
        }
        if compound_gates > 0.0 {
            transformations.push(IRTransformation {
                transform_type: TransformationType::Decomposition,
                parameters: HashMap::new(),
            });
        }

        // Confidence reflects how much real evidence backed the decision: a
        // circuit with no gates gives a low-confidence (uninformed) empty
        // strategy, while a larger, richer gate set saturates toward (but
        // never reaches) full confidence.
        let confidence = if total_gates <= 0.0 {
            0.5
        } else {
            (0.5 + 0.5 * (total_gates / (total_gates + 10.0))).min(0.99)
        };

        Ok(MLOptimizationStrategy {
            transformations,
            confidence,
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

    /// Extract a real feature vector from the actual IR and target
    /// platform, consumed by [`CompilationModel::predict_strategy`].
    ///
    /// `circuit_features` layout: `[num_qubits, total_gates,
    /// single_qubit_gates, two_qubit_gates, multi_qubit_gates,
    /// rotation_gates, compound_gates]`.
    /// `target_features` layout: `[platform_qubit_capacity,
    /// single_qubit_gate_fidelity, two_qubit_gate_fidelity]`.
    /// `complexity_features` layout: `[gate_density, two_qubit_ratio,
    /// estimated_depth]`.
    pub fn extract_features(
        &self,
        ir: &QuantumIR,
        target: TargetPlatform,
    ) -> QuantRS2Result<CompilationFeatures> {
        let mut single_qubit_gates = 0.0_f64;
        let mut two_qubit_gates = 0.0_f64;
        let mut multi_qubit_gates = 0.0_f64;
        let mut rotation_gates = 0.0_f64;
        let mut compound_gates = 0.0_f64;
        let mut total_gates = 0.0_f64;

        for op in &ir.operations {
            if let IROperationType::Gate(gate) = &op.operation_type {
                total_gates += 1.0;
                match op.qubits.len() {
                    1 => single_qubit_gates += 1.0,
                    2 => two_qubit_gates += 1.0,
                    _ => multi_qubit_gates += 1.0,
                }
                if matches!(
                    gate,
                    IRGate::RX(_)
                        | IRGate::RY(_)
                        | IRGate::RZ(_)
                        | IRGate::U1(_)
                        | IRGate::U2(_, _)
                        | IRGate::U3(_, _, _)
                ) {
                    rotation_gates += 1.0;
                }
                if matches!(gate, IRGate::Toffoli | IRGate::Fredkin | IRGate::SWAP) {
                    compound_gates += 1.0;
                }
            }
        }

        let estimated_depth = estimate_circuit_depth(ir) as f64;
        let (single_qubit_fidelity, two_qubit_fidelity) = platform_gate_fidelities(target);
        let platform_capacity = platform_max_qubits(target) as f64;

        let gate_density = if ir.num_qubits > 0 {
            total_gates / ir.num_qubits as f64
        } else {
            0.0
        };
        let two_qubit_ratio = if total_gates > 0.0 {
            two_qubit_gates / total_gates
        } else {
            0.0
        };

        Ok(CompilationFeatures {
            circuit_features: vec![
                ir.num_qubits as f64,
                total_gates,
                single_qubit_gates,
                two_qubit_gates,
                multi_qubit_gates,
                rotation_gates,
                compound_gates,
            ],
            target_features: vec![platform_capacity, single_qubit_fidelity, two_qubit_fidelity],
            complexity_features: vec![gate_density, two_qubit_ratio, estimated_depth],
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

    // -----------------------------------------------------------------------
    // CompilationValidator tests: validate_semantics / validate_resources /
    // estimate_fidelity must now depend on their inputs, not return
    // hardcoded true/true/0.99 for every circuit.
    // -----------------------------------------------------------------------

    fn make_source(code: &str) -> SourceCircuit {
        SourceCircuit {
            framework: crate::scirs2_cross_compilation_enhanced::QuantumFramework::OpenQASM,
            code: code.to_string(),
            metadata: HashMap::new(),
        }
    }

    fn make_target(code: &str, platform: TargetPlatform) -> TargetCode {
        TargetCode {
            platform,
            code: code.to_string(),
            format: crate::scirs2_cross_compilation_enhanced::CodeFormat::QASM,
            metadata: HashMap::new(),
        }
    }

    fn validator() -> CompilationValidator {
        CompilationValidator::new(EnhancedCrossCompilationConfig::default())
    }

    #[test]
    fn test_validate_semantics_rejects_dissimilar_gate_content() {
        let v = validator();

        // Source has gate content; target is an entirely empty compiled program.
        let source = make_source("h q[0]; cx q[0], q[1]; h q[1];");
        let target = make_target("OPENQASM 2.0;\nqreg q[2];\n", TargetPlatform::IBMQuantum);

        let valid = v.validate_semantics(&source, &target).unwrap();
        assert!(
            !valid,
            "empty target code must not be judged semantically equivalent to a non-trivial source"
        );
    }

    #[test]
    fn test_validate_semantics_accepts_matching_gate_content() {
        let v = validator();

        let source = make_source("h q[0]; cx q[0], q[1];");
        let target = make_target(
            "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n",
            TargetPlatform::IBMQuantum,
        );

        let valid = v.validate_semantics(&source, &target).unwrap();
        assert!(
            valid,
            "matching gate histograms between source and target should validate as semantically consistent"
        );
    }

    #[test]
    fn test_validate_semantics_both_empty_is_trivially_valid() {
        let v = validator();
        let source = make_source("// comment only, no gates");
        let target = make_target("// no gates emitted", TargetPlatform::Simulator);

        let valid = v.validate_semantics(&source, &target).unwrap();
        assert!(valid, "two gateless programs are trivially consistent");
    }

    #[test]
    fn test_validate_resources_rejects_oversized_circuit_for_platform() {
        let v = validator();
        // IonQ has a much smaller qubit capacity than an index of 99 implies.
        let target = make_target("qreg q[100];\nh q[99];\n", TargetPlatform::IonQ);

        let valid = v.validate_resources(&target, TargetPlatform::IonQ).unwrap();
        assert!(
            !valid,
            "a circuit using 100 qubits must be rejected for a 32-qubit platform"
        );
    }

    #[test]
    fn test_validate_resources_accepts_small_circuit() {
        let v = validator();
        let target = make_target(
            "qreg q[2];\nh q[0];\ncx q[0], q[1];\n",
            TargetPlatform::IonQ,
        );

        let valid = v.validate_resources(&target, TargetPlatform::IonQ).unwrap();
        assert!(
            valid,
            "a 2-qubit circuit must fit within any supported platform"
        );
    }

    #[test]
    fn test_estimate_fidelity_decreases_with_more_two_qubit_gates() {
        let v = validator();
        let source = make_source("h q[0]; cx q[0], q[1];");

        let few_two_qubit_gates =
            make_target("h q[0];\ncx q[0], q[1];\n", TargetPlatform::IBMQuantum);
        let many_two_qubit_gates = make_target(
            "h q[0];\ncx q[0], q[1];\ncx q[1], q[0];\ncx q[0], q[1];\ncx q[1], q[0];\ncx q[0], q[1];\n",
            TargetPlatform::IBMQuantum,
        );

        let fidelity_few = v.estimate_fidelity(&source, &few_two_qubit_gates).unwrap();
        let fidelity_many = v.estimate_fidelity(&source, &many_two_qubit_gates).unwrap();

        assert!((0.0..=1.0).contains(&fidelity_few));
        assert!((0.0..=1.0).contains(&fidelity_many));
        assert!(
            fidelity_many < fidelity_few,
            "more two-qubit gates must yield a lower fidelity estimate: few={fidelity_few}, many={fidelity_many}"
        );
    }

    #[test]
    fn test_estimate_fidelity_not_hardcoded_constant() {
        let v = validator();
        let source = make_source("h q[0];");
        let target = make_target("h q[0];\ncx q[0], q[1];\n", TargetPlatform::IonQ);

        let fidelity = v.estimate_fidelity(&source, &target).unwrap();
        assert!(
            (fidelity - 0.99).abs() > 1e-9,
            "fidelity must be computed from the actual gate composition, not the old hardcoded 0.99"
        );
    }

    #[test]
    fn test_validate_compilation_can_actually_fail() {
        let v = validator();
        let source = make_source("h q[0]; cx q[0], q[1]; h q[1]; cx q[1], q[0];");
        // Deliberately mismatched / oversized target for IonQ.
        let target = make_target("qreg q[64];\n", TargetPlatform::IonQ);

        let result = v
            .validate_compilation(&source, &target, TargetPlatform::IonQ)
            .unwrap();
        assert!(
            !result.is_valid,
            "comprehensive validation must be able to reject a broken/mismatched compilation"
        );
    }

    // -----------------------------------------------------------------------
    // CompilationFeatureExtractor / CompilationModel tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_features_reflects_actual_circuit_content() {
        let extractor = CompilationFeatureExtractor::new();
        let ir = build_ir(
            2,
            vec![
                single_gate(IRGate::H, 0),
                single_gate(IRGate::RX(0.5), 0),
                two_qubit_gate(IRGate::CNOT, 0, 1),
            ],
        );

        let features = extractor
            .extract_features(&ir, TargetPlatform::IBMQuantum)
            .unwrap();

        assert_eq!(features.circuit_features.len(), 7);
        assert_eq!(features.circuit_features[0], 2.0, "num_qubits");
        assert_eq!(features.circuit_features[1], 3.0, "total_gates");
        assert_eq!(features.circuit_features[3], 1.0, "two_qubit_gates");
        assert_eq!(features.circuit_features[5], 1.0, "rotation_gates");
        assert!(!features.target_features.is_empty());
        assert!(!features.complexity_features.is_empty());

        let empty_ir = build_ir(1, vec![]);
        let empty_features = extractor
            .extract_features(&empty_ir, TargetPlatform::IBMQuantum)
            .unwrap();
        assert_ne!(
            features.circuit_features, empty_features.circuit_features,
            "feature vectors must depend on the actual circuit content"
        );
    }

    #[test]
    fn test_predict_strategy_selects_real_transformations() {
        let model = CompilationModel::new();
        let extractor = CompilationFeatureExtractor::new();

        let ir_with_rotations = build_ir(
            1,
            vec![
                single_gate(IRGate::RX(0.5), 0),
                single_gate(IRGate::RX(0.5), 0),
            ],
        );
        let features = extractor
            .extract_features(&ir_with_rotations, TargetPlatform::IBMQuantum)
            .unwrap();
        let strategy = model.predict_strategy(&features).unwrap();
        assert!(
            !strategy.transformations.is_empty(),
            "a circuit with rotation gates must yield a non-empty strategy"
        );
        assert!(
            strategy
                .transformations
                .iter()
                .any(|t| matches!(t.transform_type, TransformationType::RotationMerging)),
            "rotation-heavy circuits should select RotationMerging"
        );

        let empty_ir = build_ir(1, vec![]);
        let empty_features = extractor
            .extract_features(&empty_ir, TargetPlatform::IBMQuantum)
            .unwrap();
        let empty_strategy = model.predict_strategy(&empty_features).unwrap();
        assert!(
            empty_strategy.transformations.is_empty(),
            "an empty circuit should not select any transformations"
        );
        assert!((empty_strategy.confidence - 0.5).abs() < 1e-9);
    }
}
