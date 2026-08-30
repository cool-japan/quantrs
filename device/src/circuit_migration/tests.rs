use super::engine::{cancel_adjacent_self_inverse_gates, simulate_basis_state, state_fidelity};
use super::*;
use crate::translation::HardwareBackend;
use quantrs2_circuit::prelude::*;
use quantrs2_core::qubit::QubitId;
use std::collections::HashMap;
use std::time::Duration;

#[test]
fn test_migration_config_default() {
    let config = MigrationConfig::default();
    assert_eq!(config.source_platform, HardwareBackend::IBMQuantum);
    assert_eq!(config.target_platform, HardwareBackend::AmazonBraket);
    assert_eq!(config.strategy, MigrationStrategy::Optimized);
    assert!(config.optimization.enable_optimization);
    assert!(config.validation_config.enable_validation);
}

#[test]
fn test_migration_strategy_custom() {
    let strategy = MigrationStrategy::Custom {
        fidelity_weight: 0.5,
        time_weight: 0.3,
        resource_weight: 0.2,
    };

    match strategy {
        MigrationStrategy::Custom {
            fidelity_weight,
            time_weight,
            resource_weight,
        } => {
            assert_eq!(fidelity_weight, 0.5);
            assert_eq!(time_weight, 0.3);
            assert_eq!(resource_weight, 0.2);
        }
        _ => panic!("Expected Custom strategy"),
    }
}

#[test]
fn test_warning_severity_ordering() {
    assert!(WarningSeverity::Info < WarningSeverity::Warning);
    assert!(WarningSeverity::Warning < WarningSeverity::Error);
    assert!(WarningSeverity::Error < WarningSeverity::Critical);
}

#[test]
fn test_circuit_metrics_calculation() {
    // This would test the circuit metrics calculation
    // Placeholder for actual implementation
    let metrics = CircuitMetrics {
        qubit_count: 5,
        depth: 10,
        gate_count: 25,
        gate_counts: HashMap::new(),
        estimated_fidelity: 0.95,
        estimated_execution_time: Duration::from_millis(100),
        resource_requirements: ResourceMetrics {
            memory_mb: 128.0,
            cpu_time: Duration::from_millis(50),
            qpu_time: Duration::from_millis(10),
            network_bandwidth: Some(1.0),
        },
    };

    assert_eq!(metrics.qubit_count, 5);
    assert_eq!(metrics.depth, 10);
    assert_eq!(metrics.gate_count, 25);
}

// --- Regression tests for the migration-pipeline fixes -------------------

/// Build a `CircuitMigrationEngine` with default/no-op collaborators, purely
/// for exercising the migration pipeline in tests (no real calibration data,
/// no real hardware topology).
fn test_engine() -> CircuitMigrationEngine {
    use crate::calibration::CalibrationManager;
    use crate::mapping_scirs2::{SciRS2MappingConfig, SciRS2QubitMapper};
    use crate::optimization::{CalibrationOptimizer, OptimizationConfig};
    use crate::topology::HardwareTopology;
    use crate::translation::GateTranslator;

    CircuitMigrationEngine::new(
        CalibrationManager::new(),
        SciRS2QubitMapper::new(
            SciRS2MappingConfig::default(),
            HardwareTopology::default(),
            None,
        ),
        CalibrationOptimizer::new(CalibrationManager::new(), OptimizationConfig::default()),
        GateTranslator::new(),
    )
}

#[test]
fn test_cancel_adjacent_self_inverse_gates_removes_identity_pairs() {
    // H;H on the same qubit is the identity and should fully cancel.
    let mut circuit = Circuit::<1>::new();
    circuit.h(0).expect("add H");
    circuit.h(0).expect("add H");
    assert_eq!(circuit.gates().len(), 2);

    let cancelled = cancel_adjacent_self_inverse_gates(&circuit).expect("cancellation succeeds");
    assert_eq!(
        cancelled.gates().len(),
        0,
        "adjacent H;H pair should cancel to the identity"
    );
}

#[test]
fn test_cancel_adjacent_self_inverse_gates_preserves_non_adjacent_pairs() {
    // H;X;H: the two H gates are not *adjacent* on qubit 0 (an X sits
    // between them acting on the same qubit), so nothing should cancel.
    let mut circuit = Circuit::<1>::new();
    circuit.h(0).expect("add H");
    circuit.x(0).expect("add X");
    circuit.h(0).expect("add H");

    let cancelled = cancel_adjacent_self_inverse_gates(&circuit).expect("cancellation succeeds");
    assert_eq!(
        cancelled.gates().len(),
        3,
        "non-adjacent identical gates must not be cancelled"
    );
}

#[test]
fn test_cancel_adjacent_self_inverse_gates_preserves_cnot_on_disjoint_qubits() {
    // CNOT(0,1) then H(0) then CNOT(0,1): the CNOTs are not adjacent because
    // an H touches qubit 0 in between, so the pair must survive intact.
    let mut circuit = Circuit::<2>::new();
    circuit.cnot(0, 1).expect("add CNOT");
    circuit.h(0).expect("add H");
    circuit.cnot(0, 1).expect("add CNOT");

    let cancelled = cancel_adjacent_self_inverse_gates(&circuit).expect("cancellation succeeds");
    assert_eq!(cancelled.gates().len(), 3);
}

#[test]
fn test_state_fidelity_self_overlap_is_one() {
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).expect("add H");
    circuit.cnot(0, 1).expect("add CNOT");

    let state = simulate_basis_state(&circuit, 0).expect("simulation succeeds");
    let fidelity = state_fidelity(&state, &state);
    assert!(
        (fidelity - 1.0).abs() < 1e-9,
        "a state's fidelity with itself must be 1.0, got {fidelity}"
    );
}

#[test]
fn test_state_fidelity_distinguishes_orthogonal_states() {
    // |0> and X|0> = |1> are orthogonal: fidelity must be ~0.
    let identity_circuit = Circuit::<1>::new();
    let mut x_circuit = Circuit::<1>::new();
    x_circuit.x(0).expect("add X");

    let state0 = simulate_basis_state(&identity_circuit, 0).expect("simulate |0>");
    let state1 = simulate_basis_state(&x_circuit, 0).expect("simulate X|0>");

    let fidelity = state_fidelity(&state0, &state1);
    assert!(
        fidelity < 1e-9,
        "orthogonal states must have near-zero fidelity, got {fidelity}"
    );
}

#[tokio::test]
async fn test_migrate_circuit_bell_state_end_to_end() {
    // Regression test for the circuit-migration pipeline: every stage
    // (translation, mapping, optimization, validation, metrics) must now
    // operate on the *real* circuit instead of fabricating results.
    let mut engine = test_engine();
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).expect("add H");
    circuit.cnot(0, 1).expect("add CNOT");

    let config = MigrationConfig::default();
    let result = engine
        .migrate_circuit(&circuit, &config)
        .await
        .expect("migration should succeed for a simple Bell-state circuit");

    // The migrated circuit must still contain real gates (translation is not
    // a no-op) and its metrics must be derived from the actual circuit size.
    assert!(
        !result.migrated_circuit.gates().is_empty(),
        "migrated circuit should not be empty"
    );
    assert_eq!(result.metrics.original.qubit_count, 2);
    assert!(result.metrics.original.gate_count > 0);
    assert!(result.metrics.migrated.gate_count > 0);

    // Validation must have actually run and produced a real, in-range score
    // rather than a fixed placeholder like 0.95/0.92/0.94.
    let validation = result
        .validation
        .expect("validation is enabled by MigrationConfig::default()");
    assert!(validation.confidence_score.is_finite());
    assert!((0.0..=1.0).contains(&validation.confidence_score));
    assert!(!validation.method_results.is_empty());

    for method_result in validation.method_results.values() {
        assert!(method_result.score.is_finite());
        assert!(!method_result.details.is_empty());
    }
}

#[tokio::test]
async fn test_migrate_circuit_functional_equivalence_holds() {
    // The migrated Bell-state circuit must remain functionally equivalent to
    // the original (translation to native gates must not change semantics).
    let mut engine = test_engine();
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).expect("add H");
    circuit.cnot(0, 1).expect("add CNOT");

    let mut config = MigrationConfig::default();
    config.validation_config.validation_methods = vec![ValidationMethod::FunctionalEquivalence];

    let result = engine
        .migrate_circuit(&circuit, &config)
        .await
        .expect("migration should succeed");

    let validation = result.validation.expect("validation enabled");
    let functional_result = validation
        .method_results
        .get(&ValidationMethod::FunctionalEquivalence)
        .expect("functional equivalence result present");

    assert!(
        functional_result.success,
        "translated Bell-state circuit must remain functionally equivalent: {}",
        functional_result.details
    );
}

#[test]
fn test_translate_circuit_case_aware_preserves_entanglement() {
    // Regression test for a real translation bug this fix uncovered:
    // `GateTranslator::is_native_gate` compares gate names case-sensitively,
    // so a circuit's "CNOT"/"H" gates never match a backend's lowercase
    // native-gate tables ("cnot"/"h"), which previously forced *every* gate
    // through `GateTranslator`'s KAK/ZYZ synthesis fallback -- and that
    // fallback was observed to drop the entangling operation entirely for a
    // simple Bell-state circuit, turning it into a separable product state.
    // `translate_circuit_case_aware` avoids the bug by passing already-native
    // gates through unchanged (matched case-insensitively) instead of
    // routing them through synthesis.
    let mut engine = test_engine();
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).expect("add H");
    circuit.cnot(0, 1).expect("add CNOT");

    let translated = engine
        .translate_circuit_case_aware(&circuit, HardwareBackend::AmazonBraket)
        .expect("case-aware translation should succeed");

    let original_state = simulate_basis_state(&circuit, 0).expect("simulate original");
    let translated_state = simulate_basis_state(&translated, 0).expect("simulate translated");

    let fidelity = state_fidelity(&original_state, &translated_state);
    assert!(
        (fidelity - 1.0).abs() < 1e-9,
        "translation must preserve the Bell state's entanglement (fidelity={fidelity}); \
         a fidelity near 0.25 indicates the CNOT was dropped and the state became separable"
    );
}
