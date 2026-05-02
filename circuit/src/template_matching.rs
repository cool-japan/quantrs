//! Template Matching Pass for quantum circuit gate-count reduction.
//!
//! Implements a convergent rewriting pass that applies precomputed equivalence
//! patterns (templates) to reduce the number of gates in a circuit.  Each
//! template maps a longer gate sequence to a shorter (or empty) equivalent one.
//!
//! Unlike the existing [`crate::optimization::TemplateMatching`] which is
//! coupled to the abstract cost-model framework, this pass works directly on
//! `Arc<dyn GateOp + Send + Sync>` slices and supports parametric templates
//! (e.g. RZ(a)·RZ(b) → RZ(a+b)).
//!
//! # Usage
//!
//! ```rust
//! use quantrs2_circuit::template_matching::TemplateMatchingPass;
//! use quantrs2_core::gate::single::{Hadamard, RotationZ};
//! use quantrs2_core::qubit::QubitId;
//! use std::sync::Arc;
//! use quantrs2_core::gate::GateOp;
//!
//! let q = QubitId::new(0);
//! let gates: Vec<Arc<dyn GateOp + Send + Sync>> = vec![
//!     Arc::new(Hadamard { target: q }),
//!     Arc::new(Hadamard { target: q }),
//! ];
//!
//! let pass = TemplateMatchingPass::with_standard_templates();
//! let result = pass.run(&gates);
//! assert!(result.is_empty(), "H·H should cancel to identity");
//! ```

use quantrs2_core::gate::{
    multi::{CNOT, CZ, SWAP},
    single::{Hadamard, PauliX, PauliY, PauliZ, Phase, PhaseDagger, RotationX, RotationY,
             RotationZ, TDagger, T},
    GateOp,
};
use quantrs2_core::qubit::QubitId;
use std::sync::Arc;

// ─── Template types ──────────────────────────────────────────────────────────

/// A single gate entry in a template pattern or replacement.
///
/// Qubit indices are *relative*: 0 = first qubit in the matched window,
/// 1 = second distinct qubit, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateGate {
    /// Gate name as returned by `GateOp::name()` (e.g. "H", "CNOT", "RZ").
    pub gate_name: String,
    /// Relative qubit indices used by this gate in the pattern.
    pub qubits: Vec<usize>,
    /// Numeric parameters (e.g. rotation angle). Empty for non-parametric gates.
    pub params: Vec<f64>,
}

impl TemplateGate {
    fn new(name: impl Into<String>, qubits: Vec<usize>) -> Self {
        Self {
            gate_name: name.into(),
            qubits,
            params: vec![],
        }
    }

    fn with_params(name: impl Into<String>, qubits: Vec<usize>, params: Vec<f64>) -> Self {
        Self {
            gate_name: name.into(),
            qubits,
            params,
        }
    }
}

// ─── Pattern kind ────────────────────────────────────────────────────────────

/// How a template matches and what it produces.
#[derive(Clone)]
enum TemplateKind {
    /// Fixed gate list → fixed (shorter) gate list.  Qubits are mapped
    /// positionally using `TemplateGate::qubits`.
    Fixed {
        pattern: Vec<TemplateGate>,
        replacement: Vec<TemplateGate>,
    },
    /// Merging two rotation gates of the same type on the same qubit.
    /// E.g. RZ(a)·RZ(b) → RZ(a+b).
    RotationMerge {
        gate_name: &'static str,
    },
}

// ─── Template ────────────────────────────────────────────────────────────────

/// A named rewrite rule (pattern → replacement).
#[derive(Clone)]
pub struct GateTemplate {
    /// Human-readable name for debugging.
    pub name: &'static str,
    kind: TemplateKind,
}

impl GateTemplate {
    /// Create a fixed-reduction template.
    pub fn fixed(
        name: &'static str,
        pattern: Vec<TemplateGate>,
        replacement: Vec<TemplateGate>,
    ) -> Self {
        Self {
            name,
            kind: TemplateKind::Fixed { pattern, replacement },
        }
    }

    /// Create a rotation-merging template.
    pub fn rotation_merge(name: &'static str, gate_name: &'static str) -> Self {
        Self {
            name,
            kind: TemplateKind::RotationMerge { gate_name },
        }
    }
}

// ─── Standard template library ───────────────────────────────────────────────

/// Build the standard set of 20+ gate-rewrite templates.
///
/// Separated from `TemplateMatchingPass::with_standard_templates` so that the
/// Vec can be constructed as a single `vec![…]` literal, avoiding the
/// `clippy::vec_init_then_push` pattern.
fn standard_templates() -> Vec<GateTemplate> {
    vec![
        // ── Single-qubit self-inverse cancellations ──────────────────────────
        // H·H = I
        GateTemplate::fixed(
            "H·H = I",
            vec![TemplateGate::new("H", vec![0]), TemplateGate::new("H", vec![0])],
            vec![],
        ),
        // X·X = I
        GateTemplate::fixed(
            "X·X = I",
            vec![TemplateGate::new("X", vec![0]), TemplateGate::new("X", vec![0])],
            vec![],
        ),
        // Y·Y = I
        GateTemplate::fixed(
            "Y·Y = I",
            vec![TemplateGate::new("Y", vec![0]), TemplateGate::new("Y", vec![0])],
            vec![],
        ),
        // Z·Z = I
        GateTemplate::fixed(
            "Z·Z = I",
            vec![TemplateGate::new("Z", vec![0]), TemplateGate::new("Z", vec![0])],
            vec![],
        ),
        // ── Two-qubit self-inverse cancellations ─────────────────────────────
        // CNOT(c,t)·CNOT(c,t) = I
        GateTemplate::fixed(
            "CNOT·CNOT = I",
            vec![
                TemplateGate::new("CNOT", vec![0, 1]),
                TemplateGate::new("CNOT", vec![0, 1]),
            ],
            vec![],
        ),
        // CZ(c,t)·CZ(c,t) = I
        GateTemplate::fixed(
            "CZ·CZ = I",
            vec![
                TemplateGate::new("CZ", vec![0, 1]),
                TemplateGate::new("CZ", vec![0, 1]),
            ],
            vec![],
        ),
        // SWAP·SWAP = I
        GateTemplate::fixed(
            "SWAP·SWAP = I",
            vec![
                TemplateGate::new("SWAP", vec![0, 1]),
                TemplateGate::new("SWAP", vec![0, 1]),
            ],
            vec![],
        ),
        // ── Gate composition rules ────────────────────────────────────────────
        // S·S = Z
        GateTemplate::fixed(
            "S·S = Z",
            vec![TemplateGate::new("S", vec![0]), TemplateGate::new("S", vec![0])],
            vec![TemplateGate::new("Z", vec![0])],
        ),
        // T·T = S
        GateTemplate::fixed(
            "T·T = S",
            vec![TemplateGate::new("T", vec![0]), TemplateGate::new("T", vec![0])],
            vec![TemplateGate::new("S", vec![0])],
        ),
        // T†·T† = S†
        GateTemplate::fixed(
            "T†·T† = S†",
            vec![
                TemplateGate::new("T†", vec![0]),
                TemplateGate::new("T†", vec![0]),
            ],
            vec![TemplateGate::new("S†", vec![0])],
        ),
        // S†·S† = Z  (since (S†)² = Z† = Z for self-adjoint Z)
        GateTemplate::fixed(
            "S†·S† = Z",
            vec![
                TemplateGate::new("S†", vec![0]),
                TemplateGate::new("S†", vec![0]),
            ],
            vec![TemplateGate::new("Z", vec![0])],
        ),
        // S·S† = I
        GateTemplate::fixed(
            "S·S† = I",
            vec![TemplateGate::new("S", vec![0]), TemplateGate::new("S†", vec![0])],
            vec![],
        ),
        // S†·S = I
        GateTemplate::fixed(
            "S†·S = I",
            vec![TemplateGate::new("S†", vec![0]), TemplateGate::new("S", vec![0])],
            vec![],
        ),
        // T·T† = I
        GateTemplate::fixed(
            "T·T† = I",
            vec![TemplateGate::new("T", vec![0]), TemplateGate::new("T†", vec![0])],
            vec![],
        ),
        // T†·T = I
        GateTemplate::fixed(
            "T†·T = I",
            vec![TemplateGate::new("T†", vec![0]), TemplateGate::new("T", vec![0])],
            vec![],
        ),
        // T·T·T·T = Z  (since T^8 = I, T^4 = Z up to global phase)
        GateTemplate::fixed(
            "T⁴ = Z",
            vec![
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
            ],
            vec![TemplateGate::new("Z", vec![0])],
        ),
        // T⁸ = I
        GateTemplate::fixed(
            "T⁸ = I",
            vec![
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
                TemplateGate::new("T", vec![0]),
            ],
            vec![],
        ),
        // ── Conjugation identities (single-qubit) ────────────────────────────
        // H·X·H = Z
        GateTemplate::fixed(
            "H·X·H = Z",
            vec![
                TemplateGate::new("H", vec![0]),
                TemplateGate::new("X", vec![0]),
                TemplateGate::new("H", vec![0]),
            ],
            vec![TemplateGate::new("Z", vec![0])],
        ),
        // H·Z·H = X
        GateTemplate::fixed(
            "H·Z·H = X",
            vec![
                TemplateGate::new("H", vec![0]),
                TemplateGate::new("Z", vec![0]),
                TemplateGate::new("H", vec![0]),
            ],
            vec![TemplateGate::new("X", vec![0])],
        ),
        // H·Y·H = Y  (up to global phase: H·Y·H = −Y, phase ignored → Y)
        GateTemplate::fixed(
            "H·Y·H = Y (global phase)",
            vec![
                TemplateGate::new("H", vec![0]),
                TemplateGate::new("Y", vec![0]),
                TemplateGate::new("H", vec![0]),
            ],
            vec![TemplateGate::new("Y", vec![0])],
        ),
        // X·Z·X = Z  (up to global phase)
        GateTemplate::fixed(
            "X·Z·X = Z (global phase)",
            vec![
                TemplateGate::new("X", vec![0]),
                TemplateGate::new("Z", vec![0]),
                TemplateGate::new("X", vec![0]),
            ],
            vec![TemplateGate::new("Z", vec![0])],
        ),
        // Z·X·Z = X  (up to global phase)
        GateTemplate::fixed(
            "Z·X·Z = X (global phase)",
            vec![
                TemplateGate::new("Z", vec![0]),
                TemplateGate::new("X", vec![0]),
                TemplateGate::new("Z", vec![0]),
            ],
            vec![TemplateGate::new("X", vec![0])],
        ),
        // ── Rotation merging ──────────────────────────────────────────────────
        // RZ(a)·RZ(b) = RZ(a+b)
        GateTemplate::rotation_merge("RZ·RZ = RZ(a+b)", "RZ"),
        // RX(a)·RX(b) = RX(a+b)
        GateTemplate::rotation_merge("RX·RX = RX(a+b)", "RX"),
        // RY(a)·RY(b) = RY(a+b)
        GateTemplate::rotation_merge("RY·RY = RY(a+b)", "RY"),
    ]
}

// ─── TemplateMatchingPass ────────────────────────────────────────────────────

/// Gate-reduction pass using precomputed equivalence templates.
///
/// Iterates through the gate list, tries to match each template at each
/// position, replaces with the shorter equivalent, and repeats until
/// convergence (no further reductions found).
pub struct TemplateMatchingPass {
    templates: Vec<GateTemplate>,
}

impl TemplateMatchingPass {
    /// Create a pass with the provided set of templates.
    pub fn new(templates: Vec<GateTemplate>) -> Self {
        Self { templates }
    }

    /// Create a pass with the standard library of 20+ templates.
    ///
    /// Includes:
    /// - Self-inverse cancellations: H·H, X·X, Y·Y, Z·Z, CNOT·CNOT, CZ·CZ, SWAP·SWAP
    /// - Gate composition rules: S·S→Z, T·T→S, T†·T†→S†, Z·Z→I
    /// - Conjugation identities: H·X·H→Z, H·Z·H→X, H·Y·H→Y (global phase)
    /// - Rotation merging: RZ(a)·RZ(b)→RZ(a+b), RX(a)·RX(b)→RX(a+b), RY(a)·RY(b)→RY(a+b)
    pub fn with_standard_templates() -> Self {
        Self {
            templates: standard_templates(),
        }
    }

    /// Run the pass on a gate list, returning the reduced gate list.
    ///
    /// Iterates until no further reductions are found (convergence).
    pub fn run(&self, gates: &[Arc<dyn GateOp + Send + Sync>]) -> Vec<Arc<dyn GateOp + Send + Sync>> {
        let mut current: Vec<Arc<dyn GateOp + Send + Sync>> = gates.to_vec();

        loop {
            let reduced = self.single_pass(&current);
            if reduced.len() == current.len() {
                // No reduction occurred — converged.
                break;
            }
            current = reduced;
        }

        current
    }

    /// Apply one pass over the gate list, trying all templates.
    fn single_pass(
        &self,
        gates: &[Arc<dyn GateOp + Send + Sync>],
    ) -> Vec<Arc<dyn GateOp + Send + Sync>> {
        let mut result: Vec<Arc<dyn GateOp + Send + Sync>> = Vec::with_capacity(gates.len());
        let mut i = 0;

        'outer: while i < gates.len() {
            // Try every template at position i
            for template in &self.templates {
                if let Some((replacement, consumed)) =
                    self.try_apply_template(template, gates, i)
                {
                    result.extend(replacement);
                    i += consumed;
                    continue 'outer;
                }
            }
            // No template matched — keep this gate
            result.push(gates[i].clone());
            i += 1;
        }

        result
    }

    /// Try to apply `template` starting at index `start` in `gates`.
    ///
    /// Returns `Some((replacement_gates, gates_consumed))` on a match, or `None`.
    fn try_apply_template(
        &self,
        template: &GateTemplate,
        gates: &[Arc<dyn GateOp + Send + Sync>],
        start: usize,
    ) -> Option<(Vec<Arc<dyn GateOp + Send + Sync>>, usize)> {
        match &template.kind {
            TemplateKind::Fixed { pattern, replacement } => {
                self.try_match_fixed(pattern, replacement, gates, start)
            }
            TemplateKind::RotationMerge { gate_name } => {
                self.try_merge_rotation(gate_name, gates, start)
            }
        }
    }

    /// Match a fixed pattern and produce a fixed replacement.
    fn try_match_fixed(
        &self,
        pattern: &[TemplateGate],
        replacement: &[TemplateGate],
        gates: &[Arc<dyn GateOp + Send + Sync>],
        start: usize,
    ) -> Option<(Vec<Arc<dyn GateOp + Send + Sync>>, usize)> {
        if start + pattern.len() > gates.len() {
            return None;
        }

        // Build relative-qubit → concrete-QubitId mapping
        let mut qubit_map: Vec<Option<QubitId>> = Vec::new();

        for (pat_gate, real_gate) in pattern.iter().zip(gates[start..].iter()) {
            // Check gate name
            if real_gate.name() != pat_gate.gate_name {
                return None;
            }

            let real_qubits = real_gate.qubits();

            // Check arity
            if real_qubits.len() != pat_gate.qubits.len() {
                return None;
            }

            // Extend qubit_map if needed and verify consistency
            for (rel_idx, &concrete) in pat_gate.qubits.iter().zip(real_qubits.iter()) {
                // Grow mapping
                while qubit_map.len() <= *rel_idx {
                    qubit_map.push(None);
                }

                match qubit_map[*rel_idx] {
                    None => qubit_map[*rel_idx] = Some(concrete),
                    Some(existing) => {
                        if existing != concrete {
                            return None; // Inconsistent qubit mapping
                        }
                    }
                }
            }

            // For two-qubit gates with two distinct relative indices, ensure
            // the two concrete qubits are actually distinct.
            if pat_gate.qubits.len() == 2 {
                let r0 = pat_gate.qubits[0];
                let r1 = pat_gate.qubits[1];
                if r0 != r1 {
                    // Already stored: verify they're distinct concrete qubits
                    if qubit_map.get(r0).copied().flatten()
                        == qubit_map.get(r1).copied().flatten()
                    {
                        return None;
                    }
                }
            }
        }

        // All gates in pattern must act on the same concrete qubit set —
        // for single-qubit patterns this is naturally enforced above.
        // For the two-qubit patterns (CNOT, CZ, SWAP) the mapping already
        // captures control/target ordering correctly.

        // Build replacement gates
        let mut result: Vec<Arc<dyn GateOp + Send + Sync>> = Vec::new();
        for rep_gate in replacement {
            let concrete_qubits: Vec<QubitId> = rep_gate
                .qubits
                .iter()
                .filter_map(|&rel| qubit_map.get(rel).copied().flatten())
                .collect();

            if concrete_qubits.len() != rep_gate.qubits.len() {
                return None; // Qubit not in mapping (shouldn't happen with well-formed templates)
            }

            let gate_arc = make_gate(&rep_gate.gate_name, &concrete_qubits, &rep_gate.params)?;
            result.push(gate_arc);
        }

        Some((result, pattern.len()))
    }

    /// Try to merge two consecutive rotation gates of the same type on the same qubit.
    ///
    /// Produces a single merged rotation gate (or identity if the total angle ≈ 0).
    fn try_merge_rotation(
        &self,
        gate_name: &'static str,
        gates: &[Arc<dyn GateOp + Send + Sync>],
        start: usize,
    ) -> Option<(Vec<Arc<dyn GateOp + Send + Sync>>, usize)> {
        if start + 1 >= gates.len() {
            return None;
        }

        let g0 = &gates[start];
        let g1 = &gates[start + 1];

        // Both must have the same gate name
        if g0.name() != gate_name || g1.name() != gate_name {
            return None;
        }

        // Both must act on the same single qubit
        let q0 = g0.qubits();
        let q1 = g1.qubits();
        if q0.len() != 1 || q1.len() != 1 || q0[0] != q1[0] {
            return None;
        }

        // Extract rotation angles via downcast
        let theta0 = extract_rotation_angle(g0.as_ref(), gate_name)?;
        let theta1 = extract_rotation_angle(g1.as_ref(), gate_name)?;
        let combined = theta0 + theta1;

        let qubit = q0[0];

        // If the combined angle is effectively zero (mod 2π), produce identity
        let angle_mod = combined.rem_euclid(2.0 * std::f64::consts::PI);
        if angle_mod < 1e-9 || (2.0 * std::f64::consts::PI - angle_mod) < 1e-9 {
            return Some((vec![], 2));
        }

        // Otherwise produce merged rotation
        let merged = make_gate(gate_name, &[qubit], &[combined])?;
        Some((vec![merged], 2))
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Extract the rotation angle from a gate known to be one of "RX", "RY", "RZ".
fn extract_rotation_angle(gate: &dyn GateOp, gate_name: &str) -> Option<f64> {
    match gate_name {
        "RX" => gate
            .as_any()
            .downcast_ref::<RotationX>()
            .map(|g| g.theta),
        "RY" => gate
            .as_any()
            .downcast_ref::<RotationY>()
            .map(|g| g.theta),
        "RZ" => gate
            .as_any()
            .downcast_ref::<RotationZ>()
            .map(|g| g.theta),
        _ => None,
    }
}

/// Construct an `Arc<dyn GateOp + Send + Sync>` from a name, concrete qubits, and params.
///
/// Returns `None` if the gate name is unrecognised or arity is wrong.
fn make_gate(
    name: &str,
    qubits: &[QubitId],
    params: &[f64],
) -> Option<Arc<dyn GateOp + Send + Sync>> {
    match (name, qubits.len()) {
        ("H", 1) => Some(Arc::new(Hadamard { target: qubits[0] })),
        ("X", 1) => Some(Arc::new(PauliX { target: qubits[0] })),
        ("Y", 1) => Some(Arc::new(PauliY { target: qubits[0] })),
        ("Z", 1) => Some(Arc::new(PauliZ { target: qubits[0] })),
        ("S", 1) => Some(Arc::new(Phase { target: qubits[0] })),
        ("S†", 1) => Some(Arc::new(PhaseDagger { target: qubits[0] })),
        ("T", 1) => Some(Arc::new(T { target: qubits[0] })),
        ("T†", 1) => Some(Arc::new(TDagger { target: qubits[0] })),
        ("CNOT", 2) => Some(Arc::new(CNOT {
            control: qubits[0],
            target: qubits[1],
        })),
        ("CZ", 2) => Some(Arc::new(CZ {
            control: qubits[0],
            target: qubits[1],
        })),
        ("SWAP", 2) => Some(Arc::new(SWAP {
            qubit1: qubits[0],
            qubit2: qubits[1],
        })),
        ("RX", 1) if !params.is_empty() => Some(Arc::new(RotationX {
            target: qubits[0],
            theta: params[0],
        })),
        ("RY", 1) if !params.is_empty() => Some(Arc::new(RotationY {
            target: qubits[0],
            theta: params[0],
        })),
        ("RZ", 1) if !params.is_empty() => Some(Arc::new(RotationZ {
            target: qubits[0],
            theta: params[0],
        })),
        _ => None,
    }
}

// ─── TemplateGate constructors (public API) ──────────────────────────────────

impl TemplateGate {
    /// Single-qubit gate without parameters.
    pub fn single(gate_name: impl Into<String>) -> Self {
        Self::new(gate_name, vec![0])
    }

    /// Two-qubit gate (control=0, target=1 by default).
    pub fn two_qubit(gate_name: impl Into<String>) -> Self {
        Self::new(gate_name, vec![0, 1])
    }

    /// Rotation gate with an explicit angle.
    pub fn rotation(gate_name: impl Into<String>, angle: f64) -> Self {
        Self::with_params(gate_name, vec![0], vec![angle])
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::{
        multi::{CNOT, CZ, SWAP},
        single::{Hadamard, PauliX, PauliY, PauliZ, Phase, RotationX, RotationY, RotationZ, T},
        GateOp,
    };
    use quantrs2_core::qubit::QubitId;
    use std::sync::Arc;

    fn q(id: u32) -> QubitId {
        QubitId::new(id)
    }

    fn arc<G: GateOp + Send + Sync + 'static>(g: G) -> Arc<dyn GateOp + Send + Sync> {
        Arc::new(g)
    }

    fn pass() -> TemplateMatchingPass {
        TemplateMatchingPass::with_standard_templates()
    }

    // ── Cancellation tests ───────────────────────────────────────────────────

    #[test]
    fn test_hh_cancellation() {
        let q0 = q(0);
        let gates = vec![arc(Hadamard { target: q0 }), arc(Hadamard { target: q0 })];
        let result = pass().run(&gates);
        assert!(
            result.is_empty(),
            "H·H should cancel to identity (0 gates), got {}",
            result.len()
        );
    }

    #[test]
    fn test_xx_cancellation() {
        let q0 = q(0);
        let gates = vec![arc(PauliX { target: q0 }), arc(PauliX { target: q0 })];
        let result = pass().run(&gates);
        assert!(result.is_empty(), "X·X should cancel");
    }

    #[test]
    fn test_yy_cancellation() {
        let q0 = q(0);
        let gates = vec![arc(PauliY { target: q0 }), arc(PauliY { target: q0 })];
        let result = pass().run(&gates);
        assert!(result.is_empty(), "Y·Y should cancel");
    }

    #[test]
    fn test_zz_cancellation() {
        let q0 = q(0);
        let gates = vec![arc(PauliZ { target: q0 }), arc(PauliZ { target: q0 })];
        let result = pass().run(&gates);
        assert!(result.is_empty(), "Z·Z should cancel");
    }

    #[test]
    fn test_cnot_cancellation() {
        let (c, t) = (q(0), q(1));
        let gates = vec![
            arc(CNOT { control: c, target: t }),
            arc(CNOT { control: c, target: t }),
        ];
        let result = pass().run(&gates);
        assert!(
            result.is_empty(),
            "CNOT·CNOT should cancel to identity, got {} gates",
            result.len()
        );
    }

    #[test]
    fn test_cz_cancellation() {
        let (c, t) = (q(0), q(1));
        let gates = vec![
            arc(CZ { control: c, target: t }),
            arc(CZ { control: c, target: t }),
        ];
        let result = pass().run(&gates);
        assert!(result.is_empty(), "CZ·CZ should cancel");
    }

    #[test]
    fn test_swap_cancellation() {
        let (a, b) = (q(0), q(1));
        let gates = vec![
            arc(SWAP { qubit1: a, qubit2: b }),
            arc(SWAP { qubit1: a, qubit2: b }),
        ];
        let result = pass().run(&gates);
        assert!(result.is_empty(), "SWAP·SWAP should cancel");
    }

    // ── Composition tests ────────────────────────────────────────────────────

    #[test]
    fn test_ss_to_z() {
        let q0 = q(0);
        let gates = vec![arc(Phase { target: q0 }), arc(Phase { target: q0 })];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 1, "S·S should produce one gate");
        assert_eq!(result[0].name(), "Z", "S·S should produce Z");
    }

    #[test]
    fn test_tt_to_s() {
        let q0 = q(0);
        let gates = vec![arc(T { target: q0 }), arc(T { target: q0 })];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 1, "T·T should produce one gate");
        assert_eq!(result[0].name(), "S", "T·T should produce S");
    }

    // ── Rotation merging tests ───────────────────────────────────────────────

    #[test]
    fn test_rz_merging() {
        let q0 = q(0);
        let gates = vec![
            arc(RotationZ { target: q0, theta: 0.3 }),
            arc(RotationZ { target: q0, theta: 0.7 }),
        ];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 1, "RZ(0.3)·RZ(0.7) should merge to one gate");
        assert_eq!(result[0].name(), "RZ");
        let merged = result[0]
            .as_any()
            .downcast_ref::<RotationZ>()
            .expect("should downcast to RotationZ");
        assert!(
            (merged.theta - 1.0).abs() < 1e-9,
            "merged angle should be 1.0, got {}",
            merged.theta
        );
    }

    #[test]
    fn test_rx_merging() {
        let q0 = q(0);
        let gates = vec![
            arc(RotationX { target: q0, theta: 0.5 }),
            arc(RotationX { target: q0, theta: 0.5 }),
        ];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 1, "RX(0.5)·RX(0.5) should merge");
        let merged = result[0]
            .as_any()
            .downcast_ref::<RotationX>()
            .expect("should downcast to RotationX");
        assert!(
            (merged.theta - 1.0).abs() < 1e-9,
            "merged angle should be 1.0"
        );
    }

    #[test]
    fn test_ry_merging() {
        let q0 = q(0);
        let gates = vec![
            arc(RotationY { target: q0, theta: 0.2 }),
            arc(RotationY { target: q0, theta: 0.8 }),
        ];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 1, "RY(0.2)·RY(0.8) should merge");
    }

    // ── No false reduction tests ─────────────────────────────────────────────

    #[test]
    fn test_no_false_reduction_different_qubits() {
        // H on q0 and H on q1 — must NOT cancel (different qubits)
        let gates = vec![
            arc(Hadamard { target: q(0) }),
            arc(Hadamard { target: q(1) }),
        ];
        let result = pass().run(&gates);
        assert_eq!(
            result.len(),
            2,
            "H q[0]; H q[1]; must stay (different qubits)"
        );
    }

    #[test]
    fn test_no_false_reduction_different_gates() {
        // H then X — must stay
        let q0 = q(0);
        let gates = vec![arc(Hadamard { target: q0 }), arc(PauliX { target: q0 })];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 2, "H·X must not reduce");
    }

    #[test]
    fn test_cnot_different_controls_no_cancel() {
        // CNOT(0,2) and CNOT(1,2) — different control qubits, must NOT cancel
        let gates = vec![
            arc(CNOT { control: q(0), target: q(2) }),
            arc(CNOT { control: q(1), target: q(2) }),
        ];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 2, "CNOT with different controls must not cancel");
    }

    #[test]
    fn test_cnot_different_targets_no_cancel() {
        // CNOT(0,1) and CNOT(0,2) — different target qubits, must NOT cancel
        let gates = vec![
            arc(CNOT { control: q(0), target: q(1) }),
            arc(CNOT { control: q(0), target: q(2) }),
        ];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 2, "CNOT with different targets must not cancel");
    }

    // ── Conjugation identity tests ───────────────────────────────────────────

    #[test]
    fn test_hxh_to_z() {
        let q0 = q(0);
        let gates = vec![
            arc(Hadamard { target: q0 }),
            arc(PauliX { target: q0 }),
            arc(Hadamard { target: q0 }),
        ];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 1, "H·X·H should reduce to one gate");
        assert_eq!(result[0].name(), "Z", "H·X·H = Z");
    }

    #[test]
    fn test_hzh_to_x() {
        let q0 = q(0);
        let gates = vec![
            arc(Hadamard { target: q0 }),
            arc(PauliZ { target: q0 }),
            arc(Hadamard { target: q0 }),
        ];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 1, "H·Z·H should reduce to one gate");
        assert_eq!(result[0].name(), "X", "H·Z·H = X");
    }

    // ── Convergence test ─────────────────────────────────────────────────────

    #[test]
    fn test_multi_pass_convergence() {
        // H·H·H·H = I (4 H gates cancel in two passes)
        let q0 = q(0);
        let gates = vec![
            arc(Hadamard { target: q0 }),
            arc(Hadamard { target: q0 }),
            arc(Hadamard { target: q0 }),
            arc(Hadamard { target: q0 }),
        ];
        let result = pass().run(&gates);
        assert!(result.is_empty(), "H⁴ should converge to identity");
    }

    // ── RZ identity (zero angle) ─────────────────────────────────────────────

    #[test]
    fn test_rz_cancels_to_identity_when_total_is_2pi() {
        let q0 = q(0);
        let two_pi = 2.0 * std::f64::consts::PI;
        let gates = vec![
            arc(RotationZ { target: q0, theta: two_pi * 0.6 }),
            arc(RotationZ { target: q0, theta: two_pi * 0.4 }),
        ];
        let result = pass().run(&gates);
        // Combined angle = 2π, which should cancel to identity
        assert!(
            result.is_empty(),
            "RZ(0.6·2π)·RZ(0.4·2π) should cancel to identity, got {} gates",
            result.len()
        );
    }

    // ── Rz merging across multiple pairs ─────────────────────────────────────

    #[test]
    fn test_rz_merging_three_gates() {
        // RZ(0.3)·RZ(0.3)·RZ(0.3) — first two merge, then we get RZ(0.6)·RZ(0.3) = RZ(0.9)
        let q0 = q(0);
        let gates = vec![
            arc(RotationZ { target: q0, theta: 0.3 }),
            arc(RotationZ { target: q0, theta: 0.3 }),
            arc(RotationZ { target: q0, theta: 0.3 }),
        ];
        let result = pass().run(&gates);
        assert_eq!(result.len(), 1, "RZ·RZ·RZ should merge to one gate");
        let merged = result[0]
            .as_any()
            .downcast_ref::<RotationZ>()
            .expect("should be RZ");
        assert!(
            (merged.theta - 0.9).abs() < 1e-9,
            "merged angle should be 0.9, got {}",
            merged.theta
        );
    }
}
