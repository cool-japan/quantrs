//! Commutation analysis for quantum gate reordering.
//!
//! This module provides functionality to analyze which quantum gates commute
//! with each other, enabling optimizations like gate reordering and parallelization.

use ndarray::Array2;
use num_complex::Complex64;
use std::collections::{HashMap, HashSet};

use quantrs2_core::gate::GateOp;
use quantrs2_core::qubit::QubitId;

/// Type of gate for commutation analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GateType {
    /// Single-qubit X rotation
    Rx(String), // parameter as string for hashing
    /// Single-qubit Y rotation
    Ry(String),
    /// Single-qubit Z rotation
    Rz(String),
    /// Hadamard gate
    H,
    /// Pauli-X gate
    X,
    /// Pauli-Y gate
    Y,
    /// Pauli-Z gate
    Z,
    /// Phase gate
    S,
    /// T gate
    T,
    /// CNOT gate
    CNOT,
    /// CZ gate
    CZ,
    /// SWAP gate
    SWAP,
    /// Toffoli gate
    Toffoli,
    /// Measurement
    Measure,
    /// Custom gate
    Custom(String),
}

/// Result of commutation check
#[derive(Debug, Clone, PartialEq)]
pub enum CommutationResult {
    /// Gates commute exactly
    Commute,
    /// Gates anti-commute (commute up to a phase)
    AntiCommute(Complex64),
    /// Gates don't commute
    NonCommute,
    /// Gates commute under certain conditions
    ConditionalCommute(String),
}

/// Commutation rules database
pub struct CommutationRules {
    /// Cached commutation results
    cache: HashMap<(GateType, GateType), CommutationResult>,
    /// Custom commutation rules
    custom_rules: HashMap<(String, String), CommutationResult>,
}

impl CommutationRules {
    /// Create a new commutation rules database with standard rules
    pub fn new() -> Self {
        let mut rules = Self {
            cache: HashMap::new(),
            custom_rules: HashMap::new(),
        };
        rules.initialize_standard_rules();
        rules
    }

    /// Initialize standard commutation rules
    fn initialize_standard_rules(&mut self) {
        use CommutationResult::*;
        use GateType::*;

        // Pauli commutation rules
        self.add_rule(X, X, Commute);
        self.add_rule(Y, Y, Commute);
        self.add_rule(Z, Z, Commute);
        self.add_rule(X, Y, NonCommute);
        self.add_rule(X, Z, NonCommute);
        self.add_rule(Y, Z, NonCommute);

        // Hadamard commutation
        self.add_rule(H, H, Commute);
        self.add_rule(H, X, NonCommute);
        self.add_rule(H, Y, NonCommute);
        self.add_rule(H, Z, NonCommute);

        // Phase gates
        self.add_rule(S, S, Commute);
        self.add_rule(T, T, Commute);
        self.add_rule(S, T, Commute);
        self.add_rule(S, Z, Commute);
        self.add_rule(T, Z, Commute);

        // Z-basis rotations commute
        self.add_rule(Z, Rz("any".to_string()), Commute);
        self.add_rule(S, Rz("any".to_string()), Commute);
        self.add_rule(T, Rz("any".to_string()), Commute);
        self.add_rule(Rz("any1".to_string()), Rz("any2".to_string()), Commute);

        // CNOT commutation rules
        self.add_rule(
            CNOT,
            CNOT,
            ConditionalCommute("Same control and target".to_string()),
        );
        self.add_rule(CZ, CZ, ConditionalCommute("Same qubits".to_string()));

        // Measurements don't commute with most gates
        self.add_rule(Measure, X, NonCommute);
        self.add_rule(Measure, Y, NonCommute);
        self.add_rule(Measure, H, NonCommute);
        self.add_rule(Measure, Z, Commute); // Z-basis measurement commutes with Z
    }

    /// Add a commutation rule
    pub fn add_rule(&mut self, gate1: GateType, gate2: GateType, result: CommutationResult) {
        self.cache
            .insert((gate1.clone(), gate2.clone()), result.clone());
        // Commutation is symmetric for most cases
        if matches!(
            result,
            CommutationResult::Commute | CommutationResult::NonCommute
        ) {
            self.cache.insert((gate2, gate1), result);
        }
    }

    /// Add a custom commutation rule
    pub fn add_custom_rule(&mut self, gate1: String, gate2: String, result: CommutationResult) {
        self.custom_rules
            .insert((gate1.clone(), gate2.clone()), result.clone());
        if matches!(
            result,
            CommutationResult::Commute | CommutationResult::NonCommute
        ) {
            self.custom_rules.insert((gate2, gate1), result);
        }
    }

    /// Check if two gate types commute
    pub fn check_commutation(&self, gate1: &GateType, gate2: &GateType) -> CommutationResult {
        // Check cache first
        if let Some(result) = self.cache.get(&(gate1.clone(), gate2.clone())) {
            return result.clone();
        }

        // Check custom rules
        if let (GateType::Custom(name1), GateType::Custom(name2)) = (gate1, gate2) {
            if let Some(result) = self.custom_rules.get(&(name1.clone(), name2.clone())) {
                return result.clone();
            }
        }

        // Default: assume non-commuting
        CommutationResult::NonCommute
    }
}

impl Default for CommutationRules {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer for gate commutation in circuits
pub struct CommutationAnalyzer {
    rules: CommutationRules,
}

impl CommutationAnalyzer {
    /// Create a new commutation analyzer
    pub fn new() -> Self {
        Self {
            rules: CommutationRules::new(),
        }
    }

    /// Create with custom rules
    pub fn with_rules(rules: CommutationRules) -> Self {
        Self { rules }
    }

    /// Convert a gate operation to a gate type
    pub fn gate_to_type(gate: &dyn GateOp) -> GateType {
        match gate.name() {
            "H" => GateType::H,
            "X" => GateType::X,
            "Y" => GateType::Y,
            "Z" => GateType::Z,
            "S" => GateType::S,
            "T" => GateType::T,
            "RX" => GateType::Rx("generic".to_string()),
            "RY" => GateType::Ry("generic".to_string()),
            "RZ" => GateType::Rz("generic".to_string()),
            "CNOT" => GateType::CNOT,
            "CZ" => GateType::CZ,
            "SWAP" => GateType::SWAP,
            "Toffoli" => GateType::Toffoli,
            "Measure" => GateType::Measure,
            name => GateType::Custom(name.to_string()),
        }
    }

    /// Check if two gates commute considering their qubit assignments
    pub fn gates_commute(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        let qubits1: HashSet<_> = gate1.qubits().iter().map(|q| q.id()).collect();
        let qubits2: HashSet<_> = gate2.qubits().iter().map(|q| q.id()).collect();

        // Gates on disjoint qubits always commute
        if qubits1.is_disjoint(&qubits2) {
            return true;
        }

        // Check gate types
        let type1 = Self::gate_to_type(gate1);
        let type2 = Self::gate_to_type(gate2);

        match self.rules.check_commutation(&type1, &type2) {
            CommutationResult::Commute => true,
            CommutationResult::AntiCommute(_) => true, // Still commute, just with phase
            CommutationResult::NonCommute => false,
            CommutationResult::ConditionalCommute(condition) => {
                // Check specific conditions
                self.check_conditional_commutation(gate1, gate2, &condition)
            }
        }
    }

    /// Check conditional commutation
    fn check_conditional_commutation(
        &self,
        gate1: &dyn GateOp,
        gate2: &dyn GateOp,
        condition: &str,
    ) -> bool {
        match condition {
            "Same control and target" => {
                // For CNOT gates
                if gate1.name() == "CNOT" && gate2.name() == "CNOT" {
                    let qubits1 = gate1.qubits();
                    let qubits2 = gate2.qubits();
                    return qubits1[0] == qubits2[0] && qubits1[1] == qubits2[1];
                }
                false
            }
            "Same qubits" => {
                // Check if gates operate on exactly the same qubits
                let qubits1: HashSet<_> = gate1.qubits().iter().map(|q| q.id()).collect();
                let qubits2: HashSet<_> = gate2.qubits().iter().map(|q| q.id()).collect();
                qubits1 == qubits2
            }
            _ => false,
        }
    }

    /// Find all gates that commute with a given gate in a list
    pub fn find_commuting_gates(
        &self,
        target_gate: &dyn GateOp,
        gates: &[Box<dyn GateOp>],
    ) -> Vec<usize> {
        gates
            .iter()
            .enumerate()
            .filter(|(_, gate)| self.gates_commute(target_gate, gate.as_ref()))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Build a commutation matrix for a list of gates
    pub fn build_commutation_matrix(&self, gates: &[Box<dyn GateOp>]) -> Array2<bool> {
        let n = gates.len();
        let mut matrix = Array2::from_elem((n, n), false);

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[[i, j]] = true; // Gate commutes with itself
                } else {
                    matrix[[i, j]] = self.gates_commute(gates[i].as_ref(), gates[j].as_ref());
                }
            }
        }

        matrix
    }

    /// Find independent gate sets that can be executed in parallel
    pub fn find_parallel_sets(&self, gates: &[Box<dyn GateOp>]) -> Vec<Vec<usize>> {
        let n = gates.len();
        let mut remaining: HashSet<usize> = (0..n).collect();
        let mut parallel_sets = Vec::new();

        while !remaining.is_empty() {
            let mut current_set = Vec::new();
            let mut current_qubits = HashSet::new();

            let mut indices_to_check: Vec<usize> = remaining.iter().copied().collect();
            indices_to_check.sort(); // Process in order for deterministic results

            for idx in indices_to_check {
                let gate_qubits: HashSet<_> = gates[idx].qubits().iter().map(|q| q.id()).collect();

                // Check if this gate can be added to current set
                let can_add = if current_set.is_empty() {
                    true
                } else if !current_qubits.is_disjoint(&gate_qubits) {
                    false
                } else {
                    // Check commutation with all gates in current set
                    current_set.iter().all(|&other_idx| {
                        let gate1: &Box<dyn GateOp> = &gates[idx];
                        let gate2: &Box<dyn GateOp> = &gates[other_idx];
                        self.gates_commute(gate1.as_ref(), gate2.as_ref())
                    })
                };

                if can_add {
                    current_set.push(idx);
                    current_qubits.extend(gate_qubits);
                    remaining.remove(&idx);
                }
            }

            if !current_set.is_empty() {
                parallel_sets.push(current_set);
            }
        }

        parallel_sets
    }
}

impl Default for CommutationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension methods for circuit optimization using commutation
pub trait CommutationOptimization {
    /// Reorder gates to maximize parallelism
    fn optimize_gate_order(&mut self, analyzer: &CommutationAnalyzer);

    /// Group commuting gates together
    fn group_commuting_gates(&mut self, analyzer: &CommutationAnalyzer);
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX, PauliZ};

    #[test]
    fn test_basic_commutation() {
        let analyzer = CommutationAnalyzer::new();

        // Test Pauli commutation
        let x1 = PauliX { target: QubitId(0) };
        let x2 = PauliX { target: QubitId(0) };
        let z = PauliZ { target: QubitId(0) };

        assert!(analyzer.gates_commute(&x1, &x2)); // X commutes with X
        assert!(!analyzer.gates_commute(&x1, &z)); // X doesn't commute with Z
    }

    #[test]
    fn test_disjoint_qubits() {
        let analyzer = CommutationAnalyzer::new();

        // Gates on different qubits always commute
        let h0 = Hadamard { target: QubitId(0) };
        let h1 = Hadamard { target: QubitId(1) };

        assert!(analyzer.gates_commute(&h0, &h1));
    }

    #[test]
    fn test_cnot_commutation() {
        let analyzer = CommutationAnalyzer::new();

        // Same CNOT gates commute
        let cnot1 = CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };
        let cnot2 = CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };
        assert!(analyzer.gates_commute(&cnot1, &cnot2));

        // Different CNOT gates may not commute
        let cnot3 = CNOT {
            control: QubitId(1),
            target: QubitId(0),
        };
        assert!(!analyzer.gates_commute(&cnot1, &cnot3));
    }

    #[test]
    fn test_commutation_matrix() {
        let analyzer = CommutationAnalyzer::new();

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(Hadamard { target: QubitId(1) }),
            Box::new(PauliX { target: QubitId(0) }),
        ];

        let matrix = analyzer.build_commutation_matrix(&gates);

        // Check expected commutations
        assert!(matrix[[0, 0]]); // H0 with itself
        assert!(matrix[[0, 1]]); // H0 with H1 (different qubits)
        assert!(!matrix[[0, 2]]); // H0 with X0 (don't commute)
    }

    #[test]
    fn test_parallel_sets() {
        let analyzer = CommutationAnalyzer::new();

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(Hadamard { target: QubitId(1) }),
            Box::new(Hadamard { target: QubitId(2) }),
            Box::new(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            }),
        ];

        let parallel_sets = analyzer.find_parallel_sets(&gates);

        // First three H gates can be parallel
        assert_eq!(parallel_sets.len(), 2);
        assert_eq!(parallel_sets[0].len(), 3); // All H gates
        assert_eq!(parallel_sets[1].len(), 1); // CNOT alone
    }
}
