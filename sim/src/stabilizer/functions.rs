//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::GateOp;
use quantrs2_core::prelude::*;
use scirs2_core::random::prelude::*;
use std::sync::Arc;

use super::types::{
    CliffordCircuitBuilder, StabilizerGate, StabilizerSimulator, StabilizerTableau,
};

/// Phase encoding for Stim compatibility
/// 0 = +1, 1 = +i, 2 = -1, 3 = -i
pub type StabilizerPhase = u8;
/// Phase constants for clarity
pub mod phase {
    /// Phase +1
    pub const PLUS_ONE: u8 = 0;
    /// Phase +i
    pub const PLUS_I: u8 = 1;
    /// Phase -1
    pub const MINUS_ONE: u8 = 2;
    /// Phase -i
    pub const MINUS_I: u8 = 3;
}
/// Check if a circuit can be simulated by the stabilizer simulator
#[must_use]
pub fn is_clifford_circuit<const N: usize>(circuit: &Circuit<N>) -> bool {
    circuit.gates().iter().all(|gate| {
        matches!(
            gate.name(),
            "H" | "S" | "S†" | "CNOT" | "X" | "Y" | "Z" | "CZ" | "Phase" | "PhaseDagger"
        )
    })
}
/// Convert a gate operation to a stabilizer gate
pub(super) fn gate_to_stabilizer(gate: &Arc<dyn GateOp + Send + Sync>) -> Option<StabilizerGate> {
    let gate_name = gate.name();
    let qubits = gate.qubits();
    match gate_name {
        "H" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::H(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "S" | "Phase" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::S(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "X" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::X(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "Y" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::Y(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "Z" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::Z(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "CNOT" => {
            if qubits.len() == 2 {
                Some(StabilizerGate::CNOT(
                    qubits[0].0 as usize,
                    qubits[1].0 as usize,
                ))
            } else {
                None
            }
        }
        "CZ" => {
            if qubits.len() == 2 {
                None
            } else {
                None
            }
        }
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stabilizer_init() {
        let sim = StabilizerSimulator::new(3);
        let stabs = sim.get_stabilizers();
        assert_eq!(stabs.len(), 3);
        assert_eq!(stabs[0], "+ZII");
        assert_eq!(stabs[1], "+IZI");
        assert_eq!(stabs[2], "+IIZ");
    }
    #[test]
    fn test_hadamard_gate() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+X");
    }
    #[test]
    fn test_bell_state() {
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(0, 1))
            .expect("CNOT gate application should succeed");
        let stabs = sim.get_stabilizers();
        assert!(stabs.contains(&"+XX".to_string()));
        assert!(stabs.contains(&"+ZZ".to_string()));
    }
    #[test]
    fn test_ghz_state() {
        let mut sim = StabilizerSimulator::new(3);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(0, 1))
            .expect("CNOT gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(1, 2))
            .expect("CNOT gate application should succeed");
        let stabs = sim.get_stabilizers();
        assert!(stabs.contains(&"+XXX".to_string()));
        assert!(stabs.contains(&"+ZZI".to_string()));
        assert!(stabs.contains(&"+IZZ".to_string()));
    }
    #[test]
    fn test_s_dag_gate() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::S(0))
            .expect("S gate application should succeed");
        sim.apply_gate(StabilizerGate::SDag(0))
            .expect("S† gate application should succeed");
        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+Z");
    }
    #[test]
    fn test_sqrt_x_gate() {
        let mut sim1 = StabilizerSimulator::new(1);
        sim1.apply_gate(StabilizerGate::SqrtX(0))
            .expect("√X gate application should succeed");
        sim1.apply_gate(StabilizerGate::SqrtX(0))
            .expect("√X gate application should succeed");
        let stabs1 = sim1.get_stabilizers();
        let mut sim2 = StabilizerSimulator::new(1);
        sim2.apply_gate(StabilizerGate::X(0))
            .expect("X gate application should succeed");
        let stabs2 = sim2.get_stabilizers();
        assert!(stabs1[0] == "+Z" || stabs1[0] == "-Z");
        assert!(stabs2[0] == "+Z" || stabs2[0] == "-Z");
    }
    #[test]
    fn test_sqrt_y_gate() {
        let mut sim1 = StabilizerSimulator::new(1);
        sim1.apply_gate(StabilizerGate::SqrtY(0))
            .expect("√Y gate application should succeed");
        sim1.apply_gate(StabilizerGate::SqrtY(0))
            .expect("√Y gate application should succeed");
        let stabs1 = sim1.get_stabilizers();
        let mut sim2 = StabilizerSimulator::new(1);
        sim2.apply_gate(StabilizerGate::Y(0))
            .expect("Y gate application should succeed");
        let stabs2 = sim2.get_stabilizers();
        assert_eq!(stabs1[0], stabs2[0]);
    }
    #[test]
    fn test_cz_gate() {
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::H(1))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CZ(0, 1))
            .expect("CZ gate application should succeed");
        let stabs = sim.get_stabilizers();
        assert!(stabs.len() == 2);
    }
    #[test]
    fn test_cy_gate() {
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CY(0, 1))
            .expect("CY gate application should succeed");
        let stabs = sim.get_stabilizers();
        assert!(stabs.len() == 2);
    }
    #[test]
    fn test_swap_gate() {
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::X(1))
            .expect("X gate application should succeed");
        sim.apply_gate(StabilizerGate::SWAP(0, 1))
            .expect("SWAP gate application should succeed");
        let stabs = sim.get_stabilizers();
        assert!(stabs.len() == 2);
    }
    #[test]
    fn test_builder_pattern_new_gates() {
        let sim = CliffordCircuitBuilder::new(2)
            .h(0)
            .s_dag(0)
            .sqrt_x(1)
            .cz(0, 1)
            .run()
            .expect("Circuit execution should succeed");
        let stabs = sim.get_stabilizers();
        assert!(stabs.len() == 2);
    }
    #[test]
    fn test_large_clifford_circuit() {
        let mut sim = StabilizerSimulator::new(100);
        for i in 0..100 {
            sim.apply_gate(StabilizerGate::H(i))
                .expect("Hadamard gate application should succeed");
        }
        for i in 0..99 {
            sim.apply_gate(StabilizerGate::CNOT(i, i + 1))
                .expect("CNOT gate application should succeed");
        }
        let stabs = sim.get_stabilizers();
        assert_eq!(stabs.len(), 100);
    }
    #[test]
    fn test_measurement_randomness() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        let mut outcomes = Vec::new();
        for _ in 0..10 {
            let mut test_sim = StabilizerSimulator::new(1);
            test_sim
                .apply_gate(StabilizerGate::H(0))
                .expect("Hadamard gate application should succeed");
            let outcome = test_sim.measure(0).expect("Measurement should succeed");
            outcomes.push(outcome);
        }
        let first = outcomes[0];
        let all_same = outcomes.iter().all(|&x| x == first);
        assert!(
            !all_same || outcomes.len() < 5,
            "Measurements should show randomness"
        );
    }
    #[test]
    fn test_measure_x_basis() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        let outcome = sim
            .tableau
            .measure_x(0)
            .expect("X-basis measurement should succeed");
        assert!(!outcome);
    }
    #[test]
    fn test_measure_y_basis() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0)).unwrap();
        sim.apply_gate(StabilizerGate::S(0)).unwrap();
        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+Y");
        let outcome = sim
            .tableau
            .measure_y(0)
            .expect("Y-basis measurement should succeed");
        assert!(!outcome);
    }
    #[test]
    fn test_reset_operation() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::X(0))
            .expect("X gate application should succeed");
        sim.tableau.reset(0).expect("Reset should succeed");
        let outcome = sim.measure(0).expect("Measurement should succeed");
        assert!(!outcome);
        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+Z");
    }
    #[test]
    fn test_reset_from_superposition() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.tableau.reset(0).expect("Reset should succeed");
        let outcome = sim.measure(0).expect("Measurement should succeed");
        assert!(!outcome);
    }
    #[test]
    fn test_x_y_measurements_commute() {
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0)).unwrap();
        sim.apply_gate(StabilizerGate::H(1)).unwrap();
        sim.apply_gate(StabilizerGate::S(1)).unwrap();
        let _outcome_x = sim.tableau.measure_x(0).unwrap();
        let _outcome_y = sim.tableau.measure_y(1).unwrap();
    }
    #[test]
    fn test_imaginary_phase_tracking() {
        let mut tableau = StabilizerTableau::new(1);
        tableau.apply_h(0).unwrap();
        tableau.apply_s(0).unwrap();
        let stabs = tableau.get_stabilizers();
        assert_eq!(stabs[0], "+Y");
    }
    #[test]
    fn test_imaginary_phase_with_s_dag() {
        let mut tableau = StabilizerTableau::new(1);
        tableau.apply_h(0).unwrap();
        tableau.apply_s_dag(0).unwrap();
        let stabs = tableau.get_stabilizers();
        assert_eq!(stabs[0], "-Y");
    }
    #[test]
    fn test_stim_format_identity() {
        let mut tableau = StabilizerTableau::with_format(2, true);
        let stabs = tableau.get_stabilizers();
        assert_eq!(stabs[0], "+Z_");
        assert_eq!(stabs[1], "+_Z");
        tableau.apply_h(0).unwrap();
        let stabs = tableau.get_stabilizers();
        assert_eq!(stabs[0], "+X_");
        assert_eq!(stabs[1], "+_Z");
    }
    #[test]
    fn test_standard_format_identity() {
        let tableau = StabilizerTableau::with_format(2, false);
        let stabs = tableau.get_stabilizers();
        assert_eq!(stabs[0], "+ZI");
        assert_eq!(stabs[1], "+IZ");
    }
    #[test]
    fn test_destabilizers_output() {
        let mut tableau = StabilizerTableau::new(2);
        let destabs = tableau.get_destabilizers();
        assert_eq!(destabs[0], "+XI");
        assert_eq!(destabs[1], "+IX");
        tableau.apply_h(0).unwrap();
        let destabs = tableau.get_destabilizers();
        assert_eq!(destabs[0], "+ZI");
        assert_eq!(destabs[1], "+IX");
    }
    #[test]
    fn test_phase_constants() {
        assert_eq!(phase::PLUS_ONE, 0);
        assert_eq!(phase::PLUS_I, 1);
        assert_eq!(phase::MINUS_ONE, 2);
        assert_eq!(phase::MINUS_I, 3);
    }
    #[test]
    fn test_phase_arithmetic() {
        assert_eq!(
            StabilizerTableau::negate_phase(phase::PLUS_ONE),
            phase::MINUS_ONE
        );
        assert_eq!(
            StabilizerTableau::negate_phase(phase::PLUS_I),
            phase::MINUS_I
        );
        assert_eq!(
            StabilizerTableau::negate_phase(phase::MINUS_ONE),
            phase::PLUS_ONE
        );
        assert_eq!(
            StabilizerTableau::negate_phase(phase::MINUS_I),
            phase::PLUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_i(phase::PLUS_ONE),
            phase::PLUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_i(phase::PLUS_I),
            phase::MINUS_ONE
        );
        assert_eq!(
            StabilizerTableau::multiply_by_i(phase::MINUS_ONE),
            phase::MINUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_i(phase::MINUS_I),
            phase::PLUS_ONE
        );
        assert_eq!(
            StabilizerTableau::multiply_by_minus_i(phase::PLUS_ONE),
            phase::MINUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_minus_i(phase::PLUS_I),
            phase::PLUS_ONE
        );
        assert_eq!(
            StabilizerTableau::multiply_by_minus_i(phase::MINUS_ONE),
            phase::PLUS_I
        );
        assert_eq!(
            StabilizerTableau::multiply_by_minus_i(phase::MINUS_I),
            phase::MINUS_ONE
        );
    }
    #[test]
    fn test_y_gate_phase_tracking() {
        let mut tableau = StabilizerTableau::new(1);
        tableau.apply_y(0).unwrap();
        let stabs = tableau.get_stabilizers();
        assert_eq!(stabs[0], "-Z");
    }
    #[test]
    fn test_sqrt_gates_produce_imaginary_phases() {
        let mut tableau = StabilizerTableau::new(1);
        tableau.apply_sqrt_y(0).unwrap();
        let stabs = tableau.get_stabilizers();
        assert_eq!(stabs[0], "-X");
    }
}
