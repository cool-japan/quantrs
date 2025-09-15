//! Quantum Gate Fusion Engine
//!
//! Optimizes quantum circuits by fusing adjacent gates into efficient sequences,
//! reducing the total number of matrix multiplications and improving performance.

use num_complex::Complex64;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, OnceLock};
use crate::error::{QuantRS2Error, QuantRS2Result};

/// Types of gates that can be fused
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GateType {
    // Single-qubit gates
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    Phase(u64), // Quantized angle
    RX(u64),    // Quantized angle
    RY(u64),    // Quantized angle
    RZ(u64),    // Quantized angle
    S,
    T,

    // Two-qubit gates
    CNOT,
    CZ,
    SWAP,
    CRZ(u64), // Controlled RZ with quantized angle

    // Multi-qubit gates
    Toffoli,
    Fredkin,
}

/// A quantum gate with target qubits
#[derive(Debug, Clone)]
pub struct QuantumGate {
    pub gate_type: GateType,
    pub qubits: Vec<usize>,
    pub matrix: Vec<Complex64>,
}

impl QuantumGate {
    /// Create a new quantum gate
    pub fn new(gate_type: GateType, qubits: Vec<usize>) -> QuantRS2Result<Self> {
        let matrix = Self::compute_matrix(&gate_type)?;
        Ok(Self {
            gate_type,
            qubits,
            matrix,
        })
    }

    /// Compute the unitary matrix for a gate type
    fn compute_matrix(gate_type: &GateType) -> QuantRS2Result<Vec<Complex64>> {
        use std::f64::consts::{PI, FRAC_1_SQRT_2};

        let matrix = match gate_type {
            GateType::PauliX => vec![
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ],
            GateType::PauliY => vec![
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            ],
            GateType::PauliZ => vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ],
            GateType::Hadamard => vec![
                Complex64::new(FRAC_1_SQRT_2, 0.0), Complex64::new(FRAC_1_SQRT_2, 0.0),
                Complex64::new(FRAC_1_SQRT_2, 0.0), Complex64::new(-FRAC_1_SQRT_2, 0.0),
            ],
            GateType::S => vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0),
            ],
            GateType::T => vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2),
            ],
            GateType::RX(quantized_angle) => {
                let angle = (*quantized_angle as f64) / 1_000_000.0;
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();
                vec![
                    Complex64::new(cos_half, 0.0), Complex64::new(0.0, -sin_half),
                    Complex64::new(0.0, -sin_half), Complex64::new(cos_half, 0.0),
                ]
            },
            GateType::RY(quantized_angle) => {
                let angle = (*quantized_angle as f64) / 1_000_000.0;
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();
                vec![
                    Complex64::new(cos_half, 0.0), Complex64::new(-sin_half, 0.0),
                    Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0),
                ]
            },
            GateType::RZ(quantized_angle) => {
                let angle = (*quantized_angle as f64) / 1_000_000.0;
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();
                vec![
                    Complex64::new(cos_half, -sin_half), Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0), Complex64::new(cos_half, sin_half),
                ]
            },
            GateType::Phase(quantized_angle) => {
                let angle = (*quantized_angle as f64) / 1_000_000.0;
                vec![
                    Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0), Complex64::new(angle.cos(), angle.sin()),
                ]
            },
            GateType::CNOT => vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ],
            _ => return Err(QuantRS2Error::UnsupportedGate(format!("{:?}", gate_type))),
        };

        Ok(matrix)
    }

    /// Get the number of qubits this gate acts on
    pub fn num_qubits(&self) -> usize {
        match self.gate_type {
            GateType::PauliX | GateType::PauliY | GateType::PauliZ |
            GateType::Hadamard | GateType::Phase(_) | GateType::RX(_) |
            GateType::RY(_) | GateType::RZ(_) | GateType::S | GateType::T => 1,

            GateType::CNOT | GateType::CZ | GateType::SWAP | GateType::CRZ(_) => 2,

            GateType::Toffoli | GateType::Fredkin => 3,
        }
    }
}

/// Rule for fusing gates
#[derive(Debug, Clone)]
pub struct FusionRule {
    pub pattern: Vec<GateType>,
    pub replacement: Vec<GateType>,
    pub efficiency_gain: f64, // Expected speedup
}

impl FusionRule {
    /// Create common fusion rules
    pub fn common_rules() -> Vec<FusionRule> {
        vec![
            // X * X = I (eliminate double X)
            FusionRule {
                pattern: vec![GateType::PauliX, GateType::PauliX],
                replacement: vec![], // Identity = no gates
                efficiency_gain: 2.0,
            },

            // Y * Y = I
            FusionRule {
                pattern: vec![GateType::PauliY, GateType::PauliY],
                replacement: vec![],
                efficiency_gain: 2.0,
            },

            // Z * Z = I
            FusionRule {
                pattern: vec![GateType::PauliZ, GateType::PauliZ],
                replacement: vec![],
                efficiency_gain: 2.0,
            },

            // H * H = I
            FusionRule {
                pattern: vec![GateType::Hadamard, GateType::Hadamard],
                replacement: vec![],
                efficiency_gain: 2.0,
            },

            // S * S = Z
            FusionRule {
                pattern: vec![GateType::S, GateType::S],
                replacement: vec![GateType::PauliZ],
                efficiency_gain: 2.0,
            },

            // T * T * T * T = I
            FusionRule {
                pattern: vec![GateType::T, GateType::T, GateType::T, GateType::T],
                replacement: vec![],
                efficiency_gain: 4.0,
            },

            // Commute Z and RZ (can be parallelized)
            // This would be handled by specialized logic
        ]
    }
}

/// A sequence of fused gates
#[derive(Debug, Clone)]
pub struct FusedGateSequence {
    pub gates: Vec<QuantumGate>,
    pub fused_matrix: Vec<Complex64>,
    pub target_qubits: Vec<usize>,
    pub efficiency_gain: f64,
}

impl FusedGateSequence {
    /// Create a fused sequence from individual gates
    pub fn from_gates(gates: Vec<QuantumGate>) -> QuantRS2Result<Self> {
        if gates.is_empty() {
            return Err(QuantRS2Error::InvalidInput("Empty gate sequence".to_string()));
        }

        // All gates must act on the same qubits for fusion
        let target_qubits = gates[0].qubits.clone();
        for gate in &gates {
            if gate.qubits != target_qubits {
                return Err(QuantRS2Error::InvalidInput(
                    "All gates must act on the same qubits for fusion".to_string()
                ));
            }
        }

        // Compute fused matrix by multiplying individual matrices
        let matrix_size = gates[0].matrix.len();
        let sqrt_size = (matrix_size as f64).sqrt() as usize;

        let mut fused_matrix = Self::identity_matrix(sqrt_size);

        // Multiply matrices in reverse order (gates are applied left to right)
        for gate in gates.iter().rev() {
            fused_matrix = Self::matrix_multiply(&fused_matrix, &gate.matrix, sqrt_size)?;
        }

        let efficiency_gain = gates.len() as f64; // Each gate fusion saves one matrix multiplication

        Ok(Self {
            gates,
            fused_matrix,
            target_qubits,
            efficiency_gain,
        })
    }

    /// Create identity matrix
    fn identity_matrix(size: usize) -> Vec<Complex64> {
        let mut matrix = vec![Complex64::new(0.0, 0.0); size * size];
        for i in 0..size {
            matrix[i * size + i] = Complex64::new(1.0, 0.0);
        }
        matrix
    }

    /// Check if matrix is approximately identity
    fn is_identity_matrix(&self) -> bool {
        let size = (self.fused_matrix.len() as f64).sqrt() as usize;
        let identity = Self::identity_matrix(size);

        for (a, b) in self.fused_matrix.iter().zip(identity.iter()) {
            if (a - b).norm() > 1e-10 {
                return false;
            }
        }
        true
    }

    /// Multiply two matrices
    fn matrix_multiply(
        a: &[Complex64],
        b: &[Complex64],
        size: usize,
    ) -> QuantRS2Result<Vec<Complex64>> {
        if a.len() != size * size || b.len() != size * size {
            return Err(QuantRS2Error::InvalidInput("Matrix size mismatch".to_string()));
        }

        let mut result = vec![Complex64::new(0.0, 0.0); size * size];

        for i in 0..size {
            for j in 0..size {
                for k in 0..size {
                    result[i * size + j] += a[i * size + k] * b[k * size + j];
                }
            }
        }

        Ok(result)
    }
}

/// Gate fusion engine
pub struct GateFusionEngine {
    rules: Vec<FusionRule>,
    statistics: Arc<RwLock<FusionStatistics>>,
}

/// Fusion performance statistics
#[derive(Debug, Clone, Default)]
pub struct FusionStatistics {
    pub total_fusions: u64,
    pub gates_eliminated: u64,
    pub total_efficiency_gain: f64,
    pub fusion_types: HashMap<String, u64>,
}

impl GateFusionEngine {
    /// Create a new fusion engine
    pub fn new() -> Self {
        Self {
            rules: FusionRule::common_rules(),
            statistics: Arc::new(RwLock::new(FusionStatistics::default())),
        }
    }

    /// Add a custom fusion rule
    pub fn add_rule(&mut self, rule: FusionRule) {
        self.rules.push(rule);
    }

    /// Fuse a sequence of gates
    pub fn fuse_gates(&self, gates: Vec<QuantumGate>) -> QuantRS2Result<Vec<FusedGateSequence>> {
        if gates.is_empty() {
            return Ok(vec![]);
        }

        let mut fused_sequences = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            let gate = &gates[i];

            // Try to find fusable patterns
            if let Some(fusion_length) = self.find_fusion_pattern(&gates[i..]) {
                // Found a fusable pattern
                let fusion_gates = gates[i..i + fusion_length].to_vec();
                let fused_sequence = FusedGateSequence::from_gates(fusion_gates)?;

                // Only add non-identity sequences
                if !fused_sequence.is_identity_matrix() {
                    // Update statistics
                    {
                        let mut stats = self.statistics.write().unwrap();
                        stats.total_fusions += 1;
                        stats.gates_eliminated += (fusion_length - 1) as u64;
                        stats.total_efficiency_gain += fused_sequence.efficiency_gain;

                        let fusion_type = format!("{:?}_fusion", gate.gate_type);
                        *stats.fusion_types.entry(fusion_type).or_insert(0) += 1;
                    }

                    fused_sequences.push(fused_sequence);
                } else {
                    // Identity matrix - gates cancelled out, count them as eliminated
                    let mut stats = self.statistics.write().unwrap();
                    stats.total_fusions += 1;
                    stats.gates_eliminated += fusion_length as u64; // All gates eliminated
                }
                i += fusion_length;
            } else {
                // No fusion pattern found, group consecutive gates on the same qubit
                let mut gate_group = vec![gate.clone()];
                let mut j = i + 1;

                // Collect consecutive gates on the same qubit
                while j < gates.len() && gates[j].qubits == gate.qubits {
                    gate_group.push(gates[j].clone());
                    j += 1;
                }

                // Create a fused sequence for the group
                let fused_sequence = FusedGateSequence::from_gates(gate_group)?;
                fused_sequences.push(fused_sequence);
                i = j;
            }
        }

        Ok(fused_sequences)
    }

    /// Find fusion patterns in gate sequence
    fn find_fusion_pattern(&self, gates: &[QuantumGate]) -> Option<usize> {
        for rule in &self.rules {
            if gates.len() >= rule.pattern.len() {
                let matches = gates[..rule.pattern.len()]
                    .iter()
                    .zip(&rule.pattern)
                    .all(|(gate, pattern_gate)| gate.gate_type == *pattern_gate);

                // Also check that all gates in the pattern act on the same qubits
                let same_qubits = if rule.pattern.len() > 1 {
                    let first_qubits = &gates[0].qubits;
                    gates[1..rule.pattern.len()]
                        .iter()
                        .all(|gate| gate.qubits == *first_qubits)
                } else {
                    true // Single gate patterns always match
                };

                if matches && same_qubits {
                    return Some(rule.pattern.len());
                }
            }
        }

        // Check for consecutive identical single-qubit gates on the same qubits (can be optimized)
        if gates.len() >= 2 {
            let first = &gates[0];
            if first.num_qubits() == 1 {
                let mut count = 1;
                for gate in gates.iter().skip(1) {
                    if gate.gate_type == first.gate_type && gate.qubits == first.qubits {
                        count += 1;
                    } else {
                        break;
                    }
                }
                if count > 1 {
                    return Some(count); // Found consecutive identical gates on same qubits
                }
            }
        }

        None
    }

    /// Get fusion statistics
    pub fn get_statistics(&self) -> FusionStatistics {
        self.statistics.read().unwrap().clone()
    }

    /// Get global fusion statistics
    pub fn get_global_statistics() -> FusionStatistics {
        if let Some(engine) = GLOBAL_FUSION_ENGINE.get() {
            engine.get_statistics()
        } else {
            FusionStatistics::default()
        }
    }

    /// Reset statistics
    pub fn reset_statistics(&self) {
        let mut stats = self.statistics.write().unwrap();
        *stats = FusionStatistics::default();
    }
}

impl Default for GateFusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Global gate fusion engine
static GLOBAL_FUSION_ENGINE: OnceLock<GateFusionEngine> = OnceLock::new();

/// Get the global gate fusion engine
pub fn get_global_fusion_engine() -> &'static GateFusionEngine {
    GLOBAL_FUSION_ENGINE.get_or_init(GateFusionEngine::new)
}

/// Apply gate fusion to a circuit
pub fn apply_gate_fusion(gates: Vec<QuantumGate>) -> QuantRS2Result<Vec<FusedGateSequence>> {
    let engine = get_global_fusion_engine();
    engine.fuse_gates(gates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_x_fusion() {
        let gates = vec![
            QuantumGate::new(GateType::PauliX, vec![0]).unwrap(),
            QuantumGate::new(GateType::PauliX, vec![0]).unwrap(),
        ];

        let engine = GateFusionEngine::new();
        let fused = engine.fuse_gates(gates).unwrap();

        // Should eliminate both X gates (X*X = I)
        assert_eq!(fused.len(), 0);

        let stats = engine.get_statistics();
        assert_eq!(stats.gates_eliminated, 2);
    }

    #[test]
    fn test_hadamard_fusion() {
        let gates = vec![
            QuantumGate::new(GateType::Hadamard, vec![0]).unwrap(),
            QuantumGate::new(GateType::Hadamard, vec![0]).unwrap(),
        ];

        let engine = GateFusionEngine::new();
        let fused = engine.fuse_gates(gates).unwrap();

        // Should eliminate both H gates (H*H = I)
        assert_eq!(fused.len(), 0);
    }

    #[test]
    fn test_mixed_gate_fusion() {
        let gates = vec![
            QuantumGate::new(GateType::PauliX, vec![0]).unwrap(),
            QuantumGate::new(GateType::PauliY, vec![0]).unwrap(),
            QuantumGate::new(GateType::PauliZ, vec![0]).unwrap(),
        ];

        let engine = GateFusionEngine::new();
        let fused = engine.fuse_gates(gates).unwrap();

        // Should create one fused sequence with all three gates
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].gates.len(), 3);
    }

    #[test]
    fn test_no_fusion_different_qubits() {
        let gates = vec![
            QuantumGate::new(GateType::PauliX, vec![0]).unwrap(),
            QuantumGate::new(GateType::PauliX, vec![1]).unwrap(), // Different qubit
        ];

        let engine = GateFusionEngine::new();
        let fused = engine.fuse_gates(gates).unwrap();

        // Should create two separate sequences
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_matrix_multiplication() {
        // Test identity multiplication
        let identity = vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        ];
        let pauli_x = vec![
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ];

        let result = FusedGateSequence::matrix_multiply(&identity, &pauli_x, 2).unwrap();

        // I * X should equal X
        for (a, b) in result.iter().zip(pauli_x.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_efficiency_gain_calculation() {
        let gates = vec![
            QuantumGate::new(GateType::S, vec![0]).unwrap(),
            QuantumGate::new(GateType::T, vec![0]).unwrap(),
            QuantumGate::new(GateType::Hadamard, vec![0]).unwrap(),
        ];

        let fused = FusedGateSequence::from_gates(gates).unwrap();
        assert_eq!(fused.efficiency_gain, 3.0); // Three gates fused into one
    }
}