//! Rewriting optimization passes: peephole, template matching, circuit rewriting, and parallelization.

use crate::optimization::cost_model::CostModel;
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use quantrs2_core::gate::{
    multi,
    single::{self},
    GateOp,
};
use quantrs2_core::qubit::QubitId;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use super::OptimizationPass;

/// Peephole optimization - looks at small windows of gates for local optimizations
pub struct PeepholeOptimization {
    window_size: usize,
    patterns: Vec<PeepholePattern>,
}

#[derive(Clone)]
pub struct PeepholePattern {
    name: String,
    window_size: usize,
    matcher: fn(&[Box<dyn GateOp>]) -> Option<Vec<Box<dyn GateOp>>>,
}

impl PeepholeOptimization {
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        let patterns = vec![
            // Pattern: X-Y-X = -Y
            PeepholePattern {
                name: "X-Y-X to -Y".to_string(),
                window_size: 3,
                matcher: |gates| {
                    if gates.len() >= 3 {
                        let g0 = &gates[0];
                        let g1 = &gates[1];
                        let g2 = &gates[2];

                        if g0.name() == "X"
                            && g2.name() == "X"
                            && g1.name() == "Y"
                            && g0.qubits() == g1.qubits()
                            && g1.qubits() == g2.qubits()
                        {
                            // X-Y-X = -Y, we can return Y with a phase
                            return Some(vec![g1.clone()]);
                        }
                    }
                    None
                },
            },
            // Pattern: H-S-H = X·RZ(π/2)·X
            PeepholePattern {
                name: "H-S-H simplification".to_string(),
                window_size: 3,
                matcher: |gates| {
                    if gates.len() >= 3 {
                        let g0 = &gates[0];
                        let g1 = &gates[1];
                        let g2 = &gates[2];

                        if g0.name() == "H"
                            && g2.name() == "H"
                            && g1.name() == "S"
                            && g0.qubits() == g1.qubits()
                            && g1.qubits() == g2.qubits()
                        {
                            let target = g0.qubits()[0];
                            return Some(vec![
                                Box::new(single::PauliX { target }) as Box<dyn GateOp>,
                                Box::new(single::RotationZ {
                                    target,
                                    theta: PI / 2.0,
                                }) as Box<dyn GateOp>,
                                Box::new(single::PauliX { target }) as Box<dyn GateOp>,
                            ]);
                        }
                    }
                    None
                },
            },
            // Pattern: RZ-RX-RZ (Euler decomposition check)
            PeepholePattern {
                name: "Euler angle optimization".to_string(),
                window_size: 3,
                matcher: |gates| {
                    if gates.len() >= 3 {
                        let g0 = &gates[0];
                        let g1 = &gates[1];
                        let g2 = &gates[2];

                        if g0.name() == "RZ"
                            && g1.name() == "RX"
                            && g2.name() == "RZ"
                            && g0.qubits() == g1.qubits()
                            && g1.qubits() == g2.qubits()
                        {
                            if let (Some(rz1), Some(rx), Some(rz2)) = (
                                g0.as_any().downcast_ref::<single::RotationZ>(),
                                g1.as_any().downcast_ref::<single::RotationX>(),
                                g2.as_any().downcast_ref::<single::RotationZ>(),
                            ) {
                                // If middle rotation is small, might be numerical error
                                if rx.theta.abs() < 1e-10 {
                                    let combined_angle = rz1.theta + rz2.theta;
                                    if combined_angle.abs() < 1e-10 {
                                        return Some(vec![]); // Identity
                                    }
                                    return Some(vec![Box::new(single::RotationZ {
                                        target: rz1.target,
                                        theta: combined_angle,
                                    })
                                        as Box<dyn GateOp>]);
                                }
                            }
                        }
                    }
                    None
                },
            },
            // Pattern: CNOT-RZ-CNOT (phase gadget)
            PeepholePattern {
                name: "Phase gadget optimization".to_string(),
                window_size: 3,
                matcher: |gates| {
                    if gates.len() >= 3 {
                        let g0 = &gates[0];
                        let g1 = &gates[1];
                        let g2 = &gates[2];

                        if g0.name() == "CNOT" && g2.name() == "CNOT" && g1.name() == "RZ" {
                            if let (Some(cnot1), Some(rz), Some(cnot2)) = (
                                g0.as_any().downcast_ref::<multi::CNOT>(),
                                g1.as_any().downcast_ref::<single::RotationZ>(),
                                g2.as_any().downcast_ref::<multi::CNOT>(),
                            ) {
                                if cnot1.control == cnot2.control
                                    && cnot1.target == cnot2.target
                                    && rz.target == cnot1.target
                                {
                                    // This is a controlled-RZ, keep as is for now
                                    return None;
                                }
                            }
                        }
                    }
                    None
                },
            },
            // Pattern: Hadamard ladder reduction
            PeepholePattern {
                name: "Hadamard ladder".to_string(),
                window_size: 4,
                matcher: |gates| {
                    if gates.len() >= 4 {
                        // H-CNOT-H-CNOT pattern
                        if gates[0].name() == "H"
                            && gates[1].name() == "CNOT"
                            && gates[2].name() == "H"
                            && gates[3].name() == "CNOT"
                        {
                            let h1_target = gates[0].qubits()[0];
                            let h2_target = gates[2].qubits()[0];

                            if let (Some(cnot1), Some(cnot2)) = (
                                gates[1].as_any().downcast_ref::<multi::CNOT>(),
                                gates[3].as_any().downcast_ref::<multi::CNOT>(),
                            ) {
                                if h1_target == cnot1.control
                                    && h2_target == cnot2.control
                                    && cnot1.target == cnot2.target
                                {
                                    return None; // Keep for now, needs deeper analysis
                                }
                            }
                        }
                    }
                    None
                },
            },
        ];

        Self {
            window_size,
            patterns,
        }
    }

    /// Apply patterns to a window of gates
    fn apply_patterns(&self, window: &[Box<dyn GateOp>]) -> Option<Vec<Box<dyn GateOp>>> {
        for pattern in &self.patterns {
            if window.len() >= pattern.window_size {
                if let Some(replacement) = (pattern.matcher)(window) {
                    return Some(replacement);
                }
            }
        }
        None
    }
}

impl OptimizationPass for PeepholeOptimization {
    fn name(&self) -> &'static str {
        "Peephole Optimization"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            let mut matched = false;

            for window_size in (2..=self.window_size).rev() {
                if i + window_size <= gates.len() {
                    let window = &gates[i..i + window_size];

                    if let Some(replacement) = self.apply_patterns(window) {
                        optimized.extend(replacement);
                        i += window_size;
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                optimized.push(gates[i].clone());
                i += 1;
            }
        }

        Ok(optimized)
    }
}

/// Template matching optimization
pub struct TemplateMatching {
    templates: Vec<CircuitTemplate>,
}

#[derive(Clone)]
pub struct CircuitTemplate {
    name: String,
    pattern: Vec<String>,
    replacement: Vec<String>,
    cost_reduction: f64,
}

impl TemplateMatching {
    #[must_use]
    pub fn new() -> Self {
        let templates = vec![
            CircuitTemplate {
                name: "H-Z-H to X".to_string(),
                pattern: vec!["H".to_string(), "Z".to_string(), "H".to_string()],
                replacement: vec!["X".to_string()],
                cost_reduction: 2.0,
            },
            CircuitTemplate {
                name: "H-X-H to Z".to_string(),
                pattern: vec!["H".to_string(), "X".to_string(), "H".to_string()],
                replacement: vec!["Z".to_string()],
                cost_reduction: 2.0,
            },
            CircuitTemplate {
                name: "CNOT-H-CNOT to CZ".to_string(),
                pattern: vec!["CNOT".to_string(), "H".to_string(), "CNOT".to_string()],
                replacement: vec!["CZ".to_string()],
                cost_reduction: 1.5,
            },
            CircuitTemplate {
                name: "Double CNOT elimination".to_string(),
                pattern: vec!["CNOT".to_string(), "CNOT".to_string()],
                replacement: vec![],
                cost_reduction: 2.0,
            },
            CircuitTemplate {
                name: "S-S to Z".to_string(),
                pattern: vec!["S".to_string(), "S".to_string()],
                replacement: vec!["Z".to_string()],
                cost_reduction: 1.0,
            },
        ];

        Self { templates }
    }

    #[must_use]
    pub const fn with_templates(templates: Vec<CircuitTemplate>) -> Self {
        Self { templates }
    }

    /// Create an advanced template matcher with more sophisticated patterns
    #[must_use]
    pub fn with_advanced_templates() -> Self {
        let templates = vec![
            CircuitTemplate {
                name: "H-Z-H to X".to_string(),
                pattern: vec!["H".to_string(), "Z".to_string(), "H".to_string()],
                replacement: vec!["X".to_string()],
                cost_reduction: 2.0,
            },
            CircuitTemplate {
                name: "H-X-H to Z".to_string(),
                pattern: vec!["H".to_string(), "X".to_string(), "H".to_string()],
                replacement: vec!["Z".to_string()],
                cost_reduction: 2.0,
            },
            CircuitTemplate {
                name: "CNOT-CNOT elimination".to_string(),
                pattern: vec!["CNOT".to_string(), "CNOT".to_string()],
                replacement: vec![],
                cost_reduction: 2.0,
            },
            CircuitTemplate {
                name: "S-S to Z".to_string(),
                pattern: vec!["S".to_string(), "S".to_string()],
                replacement: vec!["Z".to_string()],
                cost_reduction: 1.0,
            },
            CircuitTemplate {
                name: "T-T-T-T to Identity".to_string(),
                pattern: vec![
                    "T".to_string(),
                    "T".to_string(),
                    "T".to_string(),
                    "T".to_string(),
                ],
                replacement: vec![],
                cost_reduction: 4.0,
            },
            CircuitTemplate {
                name: "CNOT-H-CNOT to CZ".to_string(),
                pattern: vec!["CNOT".to_string(), "H".to_string(), "CNOT".to_string()],
                replacement: vec!["CZ".to_string()],
                cost_reduction: 1.0,
            },
            CircuitTemplate {
                name: "SWAP via 3 CNOTs".to_string(),
                pattern: vec!["CNOT".to_string(), "CNOT".to_string(), "CNOT".to_string()],
                replacement: vec!["SWAP".to_string()],
                cost_reduction: 0.5,
            },
        ];

        Self { templates }
    }

    /// Create a template matcher for specific hardware
    #[must_use]
    pub fn for_hardware(hardware: &str) -> Self {
        let templates = match hardware {
            "ibm" => vec![
                CircuitTemplate {
                    name: "H-Z-H to X".to_string(),
                    pattern: vec!["H".to_string(), "Z".to_string(), "H".to_string()],
                    replacement: vec!["X".to_string()],
                    cost_reduction: 2.0,
                },
                CircuitTemplate {
                    name: "CNOT-CNOT elimination".to_string(),
                    pattern: vec!["CNOT".to_string(), "CNOT".to_string()],
                    replacement: vec![],
                    cost_reduction: 2.0,
                },
            ],
            "google" => vec![CircuitTemplate {
                name: "CNOT to CZ with Hadamards".to_string(),
                pattern: vec!["CNOT".to_string()],
                replacement: vec!["H".to_string(), "CZ".to_string(), "H".to_string()],
                cost_reduction: -0.5,
            }],
            _ => Self::new().templates,
        };

        Self { templates }
    }
}

impl OptimizationPass for TemplateMatching {
    fn name(&self) -> &'static str {
        "Template Matching"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = gates;
        let mut changed = true;

        while changed {
            changed = false;
            let original_cost = cost_model.gates_cost(&optimized);

            for template in &self.templates {
                let result = self.apply_template(template, optimized.clone())?;
                let new_cost = cost_model.gates_cost(&result);

                if new_cost < original_cost {
                    optimized = result;
                    changed = true;
                    break;
                }
            }
        }

        Ok(optimized)
    }
}

impl TemplateMatching {
    fn apply_template(
        &self,
        template: &CircuitTemplate,
        gates: Vec<Box<dyn GateOp>>,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            if let Some(replacement) = self.match_pattern_at_position(template, &gates, i)? {
                result.extend(replacement);
                i += template.pattern.len();
            } else {
                result.push(gates[i].clone());
                i += 1;
            }
        }

        Ok(result)
    }

    fn match_pattern_at_position(
        &self,
        template: &CircuitTemplate,
        gates: &[Box<dyn GateOp>],
        start: usize,
    ) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        if start + template.pattern.len() > gates.len() {
            return Ok(None);
        }

        let mut qubit_mapping = HashMap::new();
        let mut all_qubits = Vec::new();
        let mut is_match = true;

        for (i, pattern_gate) in template.pattern.iter().enumerate() {
            let gate = &gates[start + i];

            if !self.gate_matches_pattern(gate.as_ref(), pattern_gate, &qubit_mapping) {
                is_match = false;
                break;
            }

            for qubit in gate.qubits() {
                if !all_qubits.contains(&qubit) {
                    all_qubits.push(qubit);
                }
            }
        }

        if !is_match {
            return Ok(None);
        }

        if template
            .pattern
            .iter()
            .all(|p| p == "H" || p == "X" || p == "Y" || p == "Z" || p == "S" || p == "T")
        {
            let first_qubit = gates[start].qubits();
            if first_qubit.len() != 1 {
                return Ok(None);
            }

            for i in 1..template.pattern.len() {
                let gate_qubits = gates[start + i].qubits();
                if gate_qubits != first_qubit {
                    return Ok(None);
                }
            }
        }

        qubit_mapping.insert("qubits".to_string(), all_qubits);
        self.generate_replacement_gates(template, &qubit_mapping)
    }

    fn gate_matches_pattern(
        &self,
        gate: &dyn GateOp,
        pattern: &str,
        _qubit_mapping: &HashMap<String, Vec<QubitId>>,
    ) -> bool {
        gate.name() == pattern
    }

    fn generate_replacement_gates(
        &self,
        template: &CircuitTemplate,
        qubit_mapping: &HashMap<String, Vec<QubitId>>,
    ) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        let mut replacement_gates = Vec::new();

        let qubits: Vec<QubitId> = qubit_mapping
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        let mut unique_qubits: Vec<QubitId> = Vec::new();
        for qubit in qubits {
            if !unique_qubits.contains(&qubit) {
                unique_qubits.push(qubit);
            }
        }

        for replacement_pattern in &template.replacement {
            if let Some(gate) = self.create_simple_gate(replacement_pattern, &unique_qubits)? {
                replacement_gates.push(gate);
            }
        }

        Ok(Some(replacement_gates))
    }

    fn create_simple_gate(
        &self,
        pattern: &str,
        qubits: &[QubitId],
    ) -> QuantRS2Result<Option<Box<dyn GateOp>>> {
        if qubits.is_empty() {
            return Ok(None);
        }

        match pattern {
            "H" => Ok(Some(Box::new(single::Hadamard { target: qubits[0] }))),
            "X" => Ok(Some(Box::new(single::PauliX { target: qubits[0] }))),
            "Y" => Ok(Some(Box::new(single::PauliY { target: qubits[0] }))),
            "Z" => Ok(Some(Box::new(single::PauliZ { target: qubits[0] }))),
            "S" => Ok(Some(Box::new(single::Phase { target: qubits[0] }))),
            "T" => Ok(Some(Box::new(single::T { target: qubits[0] }))),
            "CNOT" if qubits.len() >= 2 => Ok(Some(Box::new(multi::CNOT {
                control: qubits[0],
                target: qubits[1],
            }))),
            "CZ" if qubits.len() >= 2 => Ok(Some(Box::new(multi::CZ {
                control: qubits[0],
                target: qubits[1],
            }))),
            "SWAP" if qubits.len() >= 2 => Ok(Some(Box::new(multi::SWAP {
                qubit1: qubits[0],
                qubit2: qubits[1],
            }))),
            _ => Ok(None),
        }
    }

    /// Create a gate instance from name and qubits
    fn create_gate(&self, gate_name: &str, qubits: &[QubitId]) -> QuantRS2Result<Box<dyn GateOp>> {
        match (gate_name, qubits.len()) {
            ("H", 1) => Ok(Box::new(single::Hadamard { target: qubits[0] })),
            ("X", 1) => Ok(Box::new(single::PauliX { target: qubits[0] })),
            ("Y", 1) => Ok(Box::new(single::PauliY { target: qubits[0] })),
            ("Z", 1) => Ok(Box::new(single::PauliZ { target: qubits[0] })),
            ("S", 1) => Ok(Box::new(single::Phase { target: qubits[0] })),
            ("T", 1) => Ok(Box::new(single::T { target: qubits[0] })),
            ("CNOT", 2) => Ok(Box::new(multi::CNOT {
                control: qubits[0],
                target: qubits[1],
            })),
            ("CZ", 2) => Ok(Box::new(multi::CZ {
                control: qubits[0],
                target: qubits[1],
            })),
            ("SWAP", 2) => Ok(Box::new(multi::SWAP {
                qubit1: qubits[0],
                qubit2: qubits[1],
            })),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Cannot create gate {} with {} qubits",
                gate_name,
                qubits.len()
            ))),
        }
    }
}

impl Default for TemplateMatching {
    fn default() -> Self {
        Self::new()
    }
}

// ─── helpers shared by CircuitRewriting rule closures ───────────────────────

/// Return the single qubit targeted by a gate, or `None` if it is multi-qubit.
pub fn single_qubit_of(g: &dyn GateOp) -> Option<QubitId> {
    let qs = g.qubits();
    if qs.len() == 1 {
        Some(qs[0])
    } else {
        None
    }
}

/// `true` when every gate in the slice targets exactly the same single qubit.
pub fn all_same_single_qubit(gates: &[Box<dyn GateOp>]) -> bool {
    let first = match single_qubit_of(gates[0].as_ref()) {
        Some(q) => q,
        None => return false,
    };
    gates[1..]
        .iter()
        .all(|g| single_qubit_of(g.as_ref()) == Some(first))
}

/// Extract the rotation angle from an RZ gate.
pub fn extract_rz_angle(g: &dyn GateOp) -> Option<f64> {
    if g.name() != "RZ" {
        return None;
    }
    g.matrix().ok().map(|m| 2.0 * m[0].arg())
}

/// Extract the rotation angle from an RX gate.
pub fn extract_rx_angle(g: &dyn GateOp) -> Option<f64> {
    if g.name() != "RX" {
        return None;
    }
    g.matrix().ok().map(|m| {
        let sin_half = -m[1].im;
        let cos_half = m[0].re;
        2.0 * sin_half.atan2(cos_half)
    })
}

/// Extract the rotation angle from an RY gate.
pub fn extract_ry_angle(g: &dyn GateOp) -> Option<f64> {
    if g.name() != "RY" {
        return None;
    }
    g.matrix().ok().map(|m| {
        let sin_half = -m[1].re;
        let cos_half = m[0].re;
        2.0 * sin_half.atan2(cos_half)
    })
}

/// Normalise an angle into `(-π, π]`.
pub fn normalise_angle(theta: f64) -> f64 {
    use std::f64::consts::TAU;
    let t = theta % TAU;
    if t > PI {
        t - TAU
    } else if t <= -PI {
        t + TAU
    } else {
        t
    }
}

/// `true` when `normalise_angle(theta)` is within `eps` of zero.
pub fn is_identity_angle(theta: f64, eps: f64) -> bool {
    normalise_angle(theta).abs() < eps
}

fn default_rewrite_rules() -> Vec<RewriteRule> {
    // 3-gate rules — listed first so they fire before their 2-gate sub-patterns

    // H X H → Z
    let hxh_to_z = RewriteRule {
        name: "H-X-H to Z".to_string(),
        window_size: 3,
        condition: |w| {
            w[0].name() == "H"
                && w[1].name() == "X"
                && w[2].name() == "H"
                && all_same_single_qubit(w)
        },
        rewrite: |w| {
            let t = single_qubit_of(w[0].as_ref()).unwrap_or_else(|| w[0].qubits()[0]);
            vec![Box::new(single::PauliZ { target: t }) as Box<dyn GateOp>]
        },
    };

    // H Z H → X
    let hzh_to_x = RewriteRule {
        name: "H-Z-H to X".to_string(),
        window_size: 3,
        condition: |w| {
            w[0].name() == "H"
                && w[1].name() == "Z"
                && w[2].name() == "H"
                && all_same_single_qubit(w)
        },
        rewrite: |w| {
            let t = single_qubit_of(w[0].as_ref()).unwrap_or_else(|| w[0].qubits()[0]);
            vec![Box::new(single::PauliX { target: t }) as Box<dyn GateOp>]
        },
    };

    // H Y H → Y  (H Y H = −Y; global phase dropped → Y)
    let hyh_to_y = RewriteRule {
        name: "H-Y-H to Y (global phase)".to_string(),
        window_size: 3,
        condition: |w| {
            w[0].name() == "H"
                && w[1].name() == "Y"
                && w[2].name() == "H"
                && all_same_single_qubit(w)
        },
        rewrite: |w| {
            let t = single_qubit_of(w[1].as_ref()).unwrap_or_else(|| w[1].qubits()[0]);
            vec![Box::new(single::PauliY { target: t }) as Box<dyn GateOp>]
        },
    };

    // X Z X → Z  (X Z X = −Z; global phase dropped → Z)
    let xzx_to_z = RewriteRule {
        name: "X-Z-X to Z (global phase)".to_string(),
        window_size: 3,
        condition: |w| {
            w[0].name() == "X"
                && w[1].name() == "Z"
                && w[2].name() == "X"
                && all_same_single_qubit(w)
        },
        rewrite: |w| {
            let t = single_qubit_of(w[1].as_ref()).unwrap_or_else(|| w[1].qubits()[0]);
            vec![Box::new(single::PauliZ { target: t }) as Box<dyn GateOp>]
        },
    };

    // Z X Z → X  (Z X Z = −X; global phase dropped → X)
    let zxz_to_x = RewriteRule {
        name: "Z-X-Z to X (global phase)".to_string(),
        window_size: 3,
        condition: |w| {
            w[0].name() == "Z"
                && w[1].name() == "X"
                && w[2].name() == "Z"
                && all_same_single_qubit(w)
        },
        rewrite: |w| {
            let t = single_qubit_of(w[1].as_ref()).unwrap_or_else(|| w[1].qubits()[0]);
            vec![Box::new(single::PauliX { target: t }) as Box<dyn GateOp>]
        },
    };

    // 2-gate Pauli cancellation rules
    let hh = RewriteRule {
        name: "H-H cancel".to_string(),
        window_size: 2,
        condition: |w| w[0].name() == "H" && w[1].name() == "H" && all_same_single_qubit(w),
        rewrite: |_w| vec![],
    };
    let xx = RewriteRule {
        name: "X-X cancel".to_string(),
        window_size: 2,
        condition: |w| w[0].name() == "X" && w[1].name() == "X" && all_same_single_qubit(w),
        rewrite: |_w| vec![],
    };
    let yy = RewriteRule {
        name: "Y-Y cancel".to_string(),
        window_size: 2,
        condition: |w| w[0].name() == "Y" && w[1].name() == "Y" && all_same_single_qubit(w),
        rewrite: |_w| vec![],
    };
    let zz = RewriteRule {
        name: "Z-Z cancel".to_string(),
        window_size: 2,
        condition: |w| w[0].name() == "Z" && w[1].name() == "Z" && all_same_single_qubit(w),
        rewrite: |_w| vec![],
    };

    // S² = Z
    let ss_to_z = RewriteRule {
        name: "S-S to Z".to_string(),
        window_size: 2,
        condition: |w| w[0].name() == "S" && w[1].name() == "S" && all_same_single_qubit(w),
        rewrite: |w| {
            let t = single_qubit_of(w[0].as_ref()).unwrap_or_else(|| w[0].qubits()[0]);
            vec![Box::new(single::PauliZ { target: t }) as Box<dyn GateOp>]
        },
    };

    // T² = S
    let tt_to_s = RewriteRule {
        name: "T-T to S".to_string(),
        window_size: 2,
        condition: |w| w[0].name() == "T" && w[1].name() == "T" && all_same_single_qubit(w),
        rewrite: |w| {
            let t = single_qubit_of(w[0].as_ref()).unwrap_or_else(|| w[0].qubits()[0]);
            vec![Box::new(single::Phase { target: t }) as Box<dyn GateOp>]
        },
    };

    // CNOT CNOT (same control+target) → identity
    let cnot_cnot = RewriteRule {
        name: "CNOT-CNOT cancel".to_string(),
        window_size: 2,
        condition: |w| {
            if w[0].name() != "CNOT" || w[1].name() != "CNOT" {
                return false;
            }
            match (
                w[0].as_any().downcast_ref::<multi::CNOT>(),
                w[1].as_any().downcast_ref::<multi::CNOT>(),
            ) {
                (Some(c1), Some(c2)) => c1.control == c2.control && c1.target == c2.target,
                _ => false,
            }
        },
        rewrite: |_w| vec![],
    };

    // CZ CZ → identity (CZ is self-inverse and symmetric)
    let cz_cz = RewriteRule {
        name: "CZ-CZ cancel".to_string(),
        window_size: 2,
        condition: |w| {
            if w[0].name() != "CZ" || w[1].name() != "CZ" {
                return false;
            }
            let q0 = w[0].qubits();
            let q1 = w[1].qubits();
            if q0.len() != 2 || q1.len() != 2 {
                return false;
            }
            (q0[0] == q1[0] && q0[1] == q1[1]) || (q0[0] == q1[1] && q0[1] == q1[0])
        },
        rewrite: |_w| vec![],
    };

    // Rotation merging rules
    let rx_merge = RewriteRule {
        name: "RX-RX merge".to_string(),
        window_size: 2,
        condition: |w| {
            w[0].name() == "RX"
                && w[1].name() == "RX"
                && all_same_single_qubit(w)
                && extract_rx_angle(w[0].as_ref()).is_some()
                && extract_rx_angle(w[1].as_ref()).is_some()
        },
        rewrite: |w| {
            let t = single_qubit_of(w[0].as_ref()).unwrap_or_else(|| w[0].qubits()[0]);
            let a = extract_rx_angle(w[0].as_ref()).unwrap_or(0.0);
            let b = extract_rx_angle(w[1].as_ref()).unwrap_or(0.0);
            let sum = normalise_angle(a + b);
            if is_identity_angle(sum, 1e-10) {
                vec![]
            } else {
                vec![Box::new(single::RotationX {
                    target: t,
                    theta: sum,
                }) as Box<dyn GateOp>]
            }
        },
    };

    let ry_merge = RewriteRule {
        name: "RY-RY merge".to_string(),
        window_size: 2,
        condition: |w| {
            w[0].name() == "RY"
                && w[1].name() == "RY"
                && all_same_single_qubit(w)
                && extract_ry_angle(w[0].as_ref()).is_some()
                && extract_ry_angle(w[1].as_ref()).is_some()
        },
        rewrite: |w| {
            let t = single_qubit_of(w[0].as_ref()).unwrap_or_else(|| w[0].qubits()[0]);
            let a = extract_ry_angle(w[0].as_ref()).unwrap_or(0.0);
            let b = extract_ry_angle(w[1].as_ref()).unwrap_or(0.0);
            let sum = normalise_angle(a + b);
            if is_identity_angle(sum, 1e-10) {
                vec![]
            } else {
                vec![Box::new(single::RotationY {
                    target: t,
                    theta: sum,
                }) as Box<dyn GateOp>]
            }
        },
    };

    let rz_merge = RewriteRule {
        name: "RZ-RZ merge".to_string(),
        window_size: 2,
        condition: |w| {
            w[0].name() == "RZ"
                && w[1].name() == "RZ"
                && all_same_single_qubit(w)
                && extract_rz_angle(w[0].as_ref()).is_some()
                && extract_rz_angle(w[1].as_ref()).is_some()
        },
        rewrite: |w| {
            let t = single_qubit_of(w[0].as_ref()).unwrap_or_else(|| w[0].qubits()[0]);
            let a = extract_rz_angle(w[0].as_ref()).unwrap_or(0.0);
            let b = extract_rz_angle(w[1].as_ref()).unwrap_or(0.0);
            let sum = normalise_angle(a + b);
            if is_identity_angle(sum, 1e-10) {
                vec![]
            } else {
                vec![Box::new(single::RotationZ {
                    target: t,
                    theta: sum,
                }) as Box<dyn GateOp>]
            }
        },
    };

    vec![
        hxh_to_z, hzh_to_x, hyh_to_y, xzx_to_z, zxz_to_x, hh, xx, yy, zz, ss_to_z, tt_to_s,
        cnot_cnot, cz_cz, rx_merge, ry_merge, rz_merge,
    ]
}

/// Circuit rewriting using equivalence rules
pub struct CircuitRewriting {
    rules: Vec<RewriteRule>,
    max_rewrites: usize,
}

/// A single rewrite rule: a window size, a pattern condition, and a replacement producer.
#[derive(Clone)]
pub struct RewriteRule {
    /// Human-readable label for debugging
    name: String,
    /// Number of consecutive gates this rule inspects
    window_size: usize,
    /// Returns `true` when the window matches this rule's LHS pattern
    condition: fn(&[Box<dyn GateOp>]) -> bool,
    /// Returns the replacement gates (may be empty for cancellation rules)
    rewrite: fn(&[Box<dyn GateOp>]) -> Vec<Box<dyn GateOp>>,
}

impl CircuitRewriting {
    /// Create a `CircuitRewriting` pass with the default rule set.
    #[must_use]
    pub fn new(max_rewrites: usize) -> Self {
        Self {
            rules: default_rewrite_rules(),
            max_rewrites,
        }
    }

    /// Create a `CircuitRewriting` pass with a caller-supplied rule set.
    #[must_use]
    pub fn with_rules(rules: Vec<RewriteRule>, max_rewrites: usize) -> Self {
        Self {
            rules,
            max_rewrites,
        }
    }

    /// Try every rule at position `pos` in `gates`.
    fn try_apply_at(
        &self,
        gates: &[Box<dyn GateOp>],
        pos: usize,
    ) -> Option<(usize, Vec<Box<dyn GateOp>>)> {
        for rule in &self.rules {
            let end = pos + rule.window_size;
            if end > gates.len() {
                continue;
            }
            let window = &gates[pos..end];
            if (rule.condition)(window) {
                return Some((rule.window_size, (rule.rewrite)(window)));
            }
        }
        None
    }

    /// One left-to-right scan. Returns `(new_gates, did_fire)`.
    fn scan_once(&self, gates: Vec<Box<dyn GateOp>>) -> (Vec<Box<dyn GateOp>>, bool) {
        let mut out: Vec<Box<dyn GateOp>> = Vec::with_capacity(gates.len());
        let mut i = 0;
        let mut fired = false;

        while i < gates.len() {
            if let Some((consumed, replacement)) = self.try_apply_at(&gates, i) {
                out.extend(replacement);
                i += consumed;
                fired = true;
            } else {
                out.push(gates[i].clone());
                i += 1;
            }
        }

        (out, fired)
    }
}

impl OptimizationPass for CircuitRewriting {
    fn name(&self) -> &'static str {
        "Circuit Rewriting"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut current = gates;
        let mut passes_done = 0;

        loop {
            if passes_done >= self.max_rewrites {
                break;
            }
            let (next, fired) = self.scan_once(current);
            current = next;
            if !fired {
                break;
            }
            passes_done += 1;
        }

        Ok(current)
    }
}

/// ASAP gate scheduling pass — reorders gates to minimise circuit depth.
///
/// Uses Kahn's topological-sort algorithm. Two gates are dependent if they
/// share at least one qubit. Gates in each layer are emitted in original
/// program order for determinism.
pub struct ParallelizationPass;

impl ParallelizationPass {
    /// Create a new `ParallelizationPass`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for ParallelizationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for ParallelizationPass {
    fn name(&self) -> &'static str {
        "Parallelization (ASAP)"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        Ok(parallelize_gates(gates))
    }
}

/// Reorder `gates` with ASAP scheduling so independent gates appear as early
/// as possible, minimising circuit depth.
///
/// Gate *j* depends on gate *i* (i < j) if and only if they share at least one
/// qubit. Kahn's algorithm is used to emit gates in topological order layer by
/// layer; within each layer the original index order is preserved.
#[must_use]
pub fn parallelize_gates(gates: Vec<Box<dyn GateOp>>) -> Vec<Box<dyn GateOp>> {
    let n = gates.len();
    if n == 0 {
        return gates;
    }

    // Build the set of qubit ids for every gate.
    let qubit_sets: Vec<HashSet<u32>> = gates
        .iter()
        .map(|g| g.qubits().into_iter().map(|q| q.id()).collect())
        .collect();

    let mut in_degree = vec![0usize; n];
    let mut predecessors: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];

    let mut last_gate_on_qubit: HashMap<u32, usize> = HashMap::new();

    for j in 0..n {
        for &qid in &qubit_sets[j] {
            if let Some(&i) = last_gate_on_qubit.get(&qid) {
                if predecessors[j].insert(i) {
                    successors[i].push(j);
                    in_degree[j] += 1;
                }
            }
        }
        for &qid in &qubit_sets[j] {
            last_gate_on_qubit.insert(qid, j);
        }
    }

    // Kahn's algorithm
    let mut result: Vec<Box<dyn GateOp>> = Vec::with_capacity(n);
    let mut ready: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();

    while !ready.is_empty() {
        ready.sort_unstable();
        let layer = std::mem::take(&mut ready);
        for &idx in &layer {
            result.push(gates[idx].clone());
            for &succ in &successors[idx] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    ready.push(succ);
                }
            }
        }
    }

    // If the graph had a cycle (impossible for sequential gate list, but be safe)
    // fall back to original order.
    if result.len() != n {
        gates
    } else {
        result
    }
}

/// Helper functions for optimization passes
pub mod utils {
    use super::{GateOp, HashMap};
    use crate::optimization::gate_properties::get_gate_properties;

    /// Check if two gates cancel each other
    pub fn gates_cancel(gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        if gate1.name() != gate2.name() || gate1.qubits() != gate2.qubits() {
            return false;
        }

        let props = get_gate_properties(gate1);
        props.is_self_inverse
    }

    /// Check if a gate is effectively identity
    pub fn is_identity_gate(gate: &dyn GateOp, tolerance: f64) -> bool {
        match gate.name() {
            "RX" | "RY" | "RZ" => {
                if let Ok(matrix) = gate.matrix() {
                    (matrix[0].re - 1.0).abs() < tolerance && matrix[0].im.abs() < tolerance
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Calculate circuit depth
    #[must_use]
    pub fn calculate_depth(gates: &[Box<dyn GateOp>]) -> usize {
        let mut qubit_depths: HashMap<u32, usize> = HashMap::new();
        let mut max_depth = 0;

        for gate in gates {
            let gate_qubits = gate.qubits();
            let current_depth = gate_qubits
                .iter()
                .map(|q| qubit_depths.get(&q.id()).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);

            let new_depth = current_depth + 1;
            for qubit in gate_qubits {
                qubit_depths.insert(qubit.id(), new_depth);
            }

            max_depth = max_depth.max(new_depth);
        }

        max_depth
    }
}

// ─── unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod rewriting_tests {
    use super::*;
    use quantrs2_core::gate::single::{
        Hadamard, PauliX, PauliY, PauliZ, Phase, RotationX, RotationY, RotationZ, T,
    };
    use quantrs2_core::qubit::QubitId;

    fn q(id: u32) -> QubitId {
        QubitId::new(id)
    }

    fn cost() -> crate::optimization::cost_model::AbstractCostModel {
        crate::optimization::cost_model::AbstractCostModel::new(
            crate::optimization::cost_model::CostWeights::default(),
        )
    }

    #[test]
    fn test_hh_cancels() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: q0 }),
            Box::new(Hadamard { target: q0 }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert!(result.is_empty(), "H-H should cancel to identity");
    }

    #[test]
    fn test_hh_different_qubits_no_cancel() {
        let pass = CircuitRewriting::new(10);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: q(0) }),
            Box::new(Hadamard { target: q(1) }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 2, "H on different qubits must not cancel");
    }

    #[test]
    fn test_ss_to_z() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Phase { target: q0 }),
            Box::new(Phase { target: q0 }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 1, "S-S should produce one gate");
        assert_eq!(result[0].name(), "Z", "S-S should produce Z");
    }

    #[test]
    fn test_tt_to_s() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> =
            vec![Box::new(T { target: q0 }), Box::new(T { target: q0 })];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 1, "T-T should produce one gate");
        assert_eq!(result[0].name(), "S", "T-T should produce S");
    }

    #[test]
    fn test_rz_rz_merge() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let theta1 = PI / 4.0;
        let theta2 = PI / 4.0;
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationZ {
                target: q0,
                theta: theta1,
            }),
            Box::new(RotationZ {
                target: q0,
                theta: theta2,
            }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 1, "RZ-RZ should merge to one gate");
        assert_eq!(result[0].name(), "RZ");
        let merged = result[0]
            .as_any()
            .downcast_ref::<RotationZ>()
            .expect("should be RotationZ");
        let expected = normalise_angle(theta1 + theta2);
        assert!(
            (merged.theta - expected).abs() < 1e-9,
            "merged angle {:.6} != expected {:.6}",
            merged.theta,
            expected
        );
    }

    #[test]
    fn test_rz_rz_cancel() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationZ {
                target: q0,
                theta: PI,
            }),
            Box::new(RotationZ {
                target: q0,
                theta: PI,
            }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert!(result.is_empty(), "RZ(π)+RZ(π) should cancel");
    }

    #[test]
    fn test_rx_rx_merge() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationX {
                target: q0,
                theta: PI / 3.0,
            }),
            Box::new(RotationX {
                target: q0,
                theta: PI / 6.0,
            }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 1, "RX-RX should merge");
        assert_eq!(result[0].name(), "RX");
        let merged = result[0]
            .as_any()
            .downcast_ref::<RotationX>()
            .expect("RotationX");
        let expected = normalise_angle(PI / 3.0 + PI / 6.0);
        assert!((merged.theta - expected).abs() < 1e-9);
    }

    #[test]
    fn test_ry_ry_cancel() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationY {
                target: q0,
                theta: PI / 2.0,
            }),
            Box::new(RotationY {
                target: q0,
                theta: -PI / 2.0,
            }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert!(result.is_empty(), "RY(π/2)+RY(-π/2) should cancel");
    }

    #[test]
    fn test_hxh_to_z() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: q0 }),
            Box::new(PauliX { target: q0 }),
            Box::new(Hadamard { target: q0 }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name(), "Z", "H-X-H → Z");
    }

    #[test]
    fn test_hzh_to_x() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: q0 }),
            Box::new(PauliZ { target: q0 }),
            Box::new(Hadamard { target: q0 }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name(), "X", "H-Z-H → X");
    }

    #[test]
    fn test_hyh_to_y() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: q0 }),
            Box::new(PauliY { target: q0 }),
            Box::new(Hadamard { target: q0 }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name(), "Y", "H-Y-H → Y");
    }

    #[test]
    fn test_hhh_converges_to_h() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: q0 }),
            Box::new(Hadamard { target: q0 }),
            Box::new(Hadamard { target: q0 }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert_eq!(result.len(), 1, "H-H-H should converge to one H");
        assert_eq!(result[0].name(), "H");
    }

    #[test]
    fn test_xxyy_cancel() {
        let pass = CircuitRewriting::new(10);
        let q0 = q(0);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(PauliX { target: q0 }),
            Box::new(PauliX { target: q0 }),
            Box::new(PauliY { target: q0 }),
            Box::new(PauliY { target: q0 }),
        ];
        let result = pass.apply_to_gates(gates, &cost()).expect("apply failed");
        assert!(result.is_empty(), "X-X-Y-Y should cancel");
    }
}
