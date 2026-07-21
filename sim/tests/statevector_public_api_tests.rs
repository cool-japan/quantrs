//! Regression tests for the imperative public gate API on
//! [`StateVectorSimulator`] (initialize_state / get_state / set_state /
//! apply_h / apply_x / apply_z_public / apply_cnot_public / apply_toffoli /
//! apply_fredkin / apply_interface_circuit).
//!
//! These methods previously did nothing to any real state and `get_state`
//! returned a hardcoded `vec![1+0i]`. They now evolve a genuine amplitude
//! vector stored inside the simulator.

use quantrs2_core::qubit::QubitId;
use quantrs2_sim::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::Complex64;

const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn approx(a: Complex64, re: f64, im: f64) -> bool {
    (a.re - re).abs() < 1e-9 && (a.im - im).abs() < 1e-9
}

#[test]
fn initialize_state_allocates_ground_state() {
    let mut sim = StateVectorSimulator::new();
    sim.initialize_state(3).expect("initialize");
    let state = sim.get_state();
    assert_eq!(state.len(), 8);
    assert!(approx(state[0], 1.0, 0.0));
    for amp in &state[1..] {
        assert!(approx(*amp, 0.0, 0.0));
    }
}

#[test]
fn apply_x_flips_single_qubit() {
    let mut sim = StateVectorSimulator::new();
    sim.initialize_state(1).expect("initialize");
    sim.apply_x(0).expect("x");
    let state = sim.get_state();
    assert_eq!(state.len(), 2);
    assert!(approx(state[0], 0.0, 0.0));
    assert!(approx(state[1], 1.0, 0.0));
}

#[test]
fn apply_h_then_cnot_builds_bell_state() {
    let mut sim = StateVectorSimulator::new();
    sim.initialize_state(2).expect("initialize");
    sim.apply_h(0).expect("h");
    sim.apply_cnot_public(0, 1).expect("cnot");
    let state = sim.get_state();
    // (|00> + |11>) / sqrt(2): indices 0 and 3 populated, 1 and 2 empty.
    assert!(approx(state[0], INV_SQRT2, 0.0));
    assert!(approx(state[1], 0.0, 0.0));
    assert!(approx(state[2], 0.0, 0.0));
    assert!(approx(state[3], INV_SQRT2, 0.0));
    let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-9);
}

#[test]
fn apply_z_flips_phase_of_one_component() {
    let mut sim = StateVectorSimulator::new();
    sim.initialize_state(1).expect("initialize");
    sim.apply_h(0).expect("h");
    sim.apply_z_public(0).expect("z");
    // H|0> = |+>, Z|+> = |->  =>  (|0> - |1>)/sqrt(2)
    let state = sim.get_state();
    assert!(approx(state[0], INV_SQRT2, 0.0));
    assert!(approx(state[1], -INV_SQRT2, 0.0));
}

#[test]
fn apply_toffoli_flips_target_only_when_both_controls_set() {
    let mut sim = StateVectorSimulator::new();
    sim.initialize_state(3).expect("initialize");
    // Only one control set -> target must stay 0.
    sim.apply_x(0).expect("x");
    sim.apply_toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2))
        .expect("toffoli");
    let state = sim.get_state();
    // Still |001> (index 1).
    assert!(approx(state[1], 1.0, 0.0));

    // Now set the second control -> target flips.
    sim.apply_x(1).expect("x");
    sim.apply_toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2))
        .expect("toffoli");
    let state = sim.get_state();
    // |111> => index 0b111 = 7.
    assert!(approx(state[7], 1.0, 0.0));
    let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-9);
}

#[test]
fn apply_fredkin_swaps_targets_under_control() {
    let mut sim = StateVectorSimulator::new();
    sim.initialize_state(3).expect("initialize");
    // control=qubit0, targets qubit1 & qubit2. Prepare control=1, t1(qubit1)=1.
    sim.apply_x(0).expect("x");
    sim.apply_x(1).expect("x");
    // State |011> in bit order (bit0=1, bit1=1, bit2=0) -> index 3.
    sim.apply_fredkin(QubitId::new(0), QubitId::new(1), QubitId::new(2))
        .expect("fredkin");
    let state = sim.get_state();
    // Swap qubit1<->qubit2: bit1 0, bit2 1 -> index bit0=1,bit2=4 => 5.
    assert!(approx(state[5], 1.0, 0.0));
    assert!(approx(state[3], 0.0, 0.0));
}

#[test]
fn get_state_and_set_state_round_trip() {
    let mut sim = StateVectorSimulator::new();
    sim.initialize_state(2).expect("initialize");
    let custom = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(-0.5, 0.0),
    ];
    sim.set_state(custom.clone()).expect("set_state");
    assert_eq!(sim.get_state(), custom);
    // get_state_mut returns an owned copy that can be edited and written back.
    let mut edited = sim.get_state_mut();
    edited[0] = Complex64::new(0.0, 0.0);
    edited[3] = Complex64::new(0.5, 0.0);
    // Renormalise-agnostic: just verify the write-back path.
    sim.set_state(edited.clone()).expect("set_state");
    assert_eq!(sim.get_state(), edited);
}

#[test]
fn set_state_rejects_non_power_of_two() {
    let mut sim = StateVectorSimulator::new();
    let bad = vec![Complex64::new(1.0, 0.0); 3];
    assert!(sim.set_state(bad).is_err());
}

#[test]
fn gates_before_initialization_error() {
    let mut sim = StateVectorSimulator::new();
    assert!(sim.apply_h(0).is_err());
    assert!(sim
        .apply_toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2))
        .is_err());
}

#[test]
fn apply_interface_circuit_x_gate() {
    let mut sim = StateVectorSimulator::new();
    let mut circuit = InterfaceCircuit::new(1, 0);
    circuit.add_gate(InterfaceGate::new(InterfaceGateType::X, vec![0]));
    // apply_interface_circuit lazily initialises the state to |0>.
    sim.apply_interface_circuit(&circuit)
        .expect("apply circuit");
    let state = sim.get_state();
    assert_eq!(state.len(), 2);
    assert!(approx(state[0], 0.0, 0.0));
    assert!(approx(state[1], 1.0, 0.0));
}

#[test]
fn apply_interface_circuit_entangling_sequence() {
    let mut sim = StateVectorSimulator::new();
    let mut circuit = InterfaceCircuit::new(2, 0);
    circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
    sim.apply_interface_circuit(&circuit)
        .expect("apply circuit");
    let state = sim.get_state();
    // Bell state again, this time through the interface-circuit path.
    assert!(approx(state[0], INV_SQRT2, 0.0));
    assert!(approx(state[3], INV_SQRT2, 0.0));
    assert!(approx(state[1], 0.0, 0.0));
    assert!(approx(state[2], 0.0, 0.0));
}

#[test]
fn apply_interface_circuit_rejects_measurement() {
    let mut sim = StateVectorSimulator::new();
    let mut circuit = InterfaceCircuit::new(1, 1);
    circuit.add_gate(InterfaceGate::new(InterfaceGateType::Measure, vec![0]));
    assert!(sim.apply_interface_circuit(&circuit).is_err());
}
