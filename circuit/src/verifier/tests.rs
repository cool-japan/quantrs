//! Tests for the verifier module

use super::*;
use quantrs2_core::gate::multi::CNOT;
use quantrs2_core::gate::single::Hadamard;
use quantrs2_core::qubit::QubitId;

#[test]
fn test_verifier_creation() {
    let circuit = Circuit::<2>::new();
    let verifier = QuantumVerifier::new(circuit);
    assert_eq!(verifier.config.enable_property_verification, true);
}

#[test]
fn test_property_addition() {
    let circuit = Circuit::<2>::new();
    let mut verifier = QuantumVerifier::new(circuit);

    let property = QuantumProperty::Unitarity { tolerance: 1e-12 };
    verifier.add_property(property).unwrap();
}

#[test]
fn test_invariant_addition() {
    let circuit = Circuit::<2>::new();
    let mut verifier = QuantumVerifier::new(circuit);

    let invariant = CircuitInvariant::QubitCount { expected_count: 2 };
    verifier.add_invariant(invariant).unwrap();
}

#[test]
fn test_verification_process() {
    let mut circuit = Circuit::<2>::new();
    circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
    circuit
        .add_gate(CNOT {
            control: QubitId(0),
            target: QubitId(1),
        })
        .unwrap();

    let mut verifier = QuantumVerifier::new(circuit);

    verifier
        .add_property(QuantumProperty::Unitarity { tolerance: 1e-12 })
        .unwrap();
    verifier
        .add_invariant(CircuitInvariant::QubitCount { expected_count: 2 })
        .unwrap();

    let result = verifier.verify_circuit().unwrap();
    assert!(matches!(
        result.status,
        VerificationStatus::Verified | VerificationStatus::Unknown
    ));
}

#[test]
fn test_property_checker() {
    let circuit = Circuit::<2>::new();
    let checker = PropertyChecker::new();
    let config = VerifierConfig::default();

    let results = checker.verify_all_properties(&circuit, &config).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_invariant_checker() {
    let circuit = Circuit::<2>::new();
    let checker = InvariantChecker::new();
    let config = VerifierConfig::default();

    let results = checker.check_all_invariants(&circuit, &config).unwrap();
    assert!(results.is_empty());
}
