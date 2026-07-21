//! Regression tests for the error-mitigation module.

use super::*;
use std::collections::HashMap;

fn minimal_noise_model() -> NoiseModel {
    let mut gate_errors = HashMap::new();
    gate_errors.insert(
        "RY".to_string(),
        GateErrorModel {
            error_rate: 0.01,
            error_type: ErrorType::Depolarizing { strength: 0.01 },
            coherence_limited: false,
            gate_time: 20e-9,
            fidelity_model: FidelityModel,
        },
    );

    NoiseModel {
        gate_errors,
        measurement_errors: MeasurementErrorModel {
            readout_fidelity: 0.95,
            assignment_matrix: Array2::eye(2),
            state_preparation_errors: Array1::zeros(2),
            measurement_crosstalk: Array2::eye(2),
        },
        coherence_times: CoherenceTimeModel {
            t1_times: Array1::from_elem(2, 50e-6),
            t2_times: Array1::from_elem(2, 30e-6),
            t2_echo_times: Array1::from_elem(2, 40e-6),
            temporal_fluctuations: TemporalFluctuation,
        },
        crosstalk_matrix: Array2::eye(2),
        temporal_correlations: TemporalCorrelationModel {
            correlation_function: CorrelationFunction::Exponential,
            correlation_time: 1e-6,
            noise_spectrum: NoiseSpectrum,
        },
    }
}

fn simple_rotation_circuit() -> QuantumCircuit {
    QuantumCircuit {
        gates: vec![
            QuantumGate {
                name: "H".to_string(),
                qubits: vec![0],
                parameters: Array1::from_vec(vec![]),
            },
            QuantumGate {
                name: "RY".to_string(),
                qubits: vec![0],
                parameters: Array1::from_vec(vec![0.3]),
            },
            QuantumGate {
                name: "RY".to_string(),
                qubits: vec![1],
                parameters: Array1::from_vec(vec![0.7]),
            },
        ],
        qubits: 2,
    }
}

/// Regression test for the fabricated-gradient bug: `with_parameters`
/// used to always return `self.clone()`, silently ignoring the
/// requested parameter values entirely, which made
/// `apply_gradient_mitigation`'s parameter-shift evaluation measure the
/// identical (unshifted) circuit for both `+pi/2` and `-pi/2`, so every
/// mitigated gradient came out as exactly zero regardless of input.
#[test]
fn test_with_parameters_actually_changes_gate_angles() {
    let circuit = simple_rotation_circuit();
    let new_params = Array1::from_vec(vec![1.234, -0.5]);
    let shifted = circuit
        .with_parameters(&new_params)
        .expect("with_parameters should succeed");

    // The two RY gates' angles must reflect the new parameter vector,
    // not the original circuit's angles.
    assert!((shifted.gates[1].parameters[0] - 1.234).abs() < 1e-12);
    assert!((shifted.gates[2].parameters[0] - (-0.5)).abs() < 1e-12);
    // The un-parameterized H gate must be left alone.
    assert_eq!(shifted.gates[0].parameters.len(), 0);
    // The original circuit must not have been mutated.
    assert!((circuit.gates[1].parameters[0] - 0.3).abs() < 1e-12);
}

/// Regression test: `apply_gradient_mitigation` must produce a nonzero
/// gradient once `with_parameters` genuinely shifts the circuit (before
/// the fix, every entry was silently 0.0).
#[test]
fn test_gradient_mitigation_is_not_all_zero() {
    let mitigator = QuantumMLErrorMitigator::new(
        MitigationStrategy::ReadoutErrorMitigation {
            calibration_matrix: Array2::eye(2),
            correction_method: ReadoutCorrectionMethod::MatrixInversion,
            regularization: 1e-6,
        },
        minimal_noise_model(),
    )
    .expect("mitigator construction should succeed");

    let circuit = simple_rotation_circuit();
    let parameters = Array1::from_vec(vec![0.3, 0.7]);
    let gradients = Array1::from_vec(vec![0.0, 0.0]);

    let mitigated = mitigator
        .apply_gradient_mitigation(&circuit, &parameters, &gradients)
        .expect("gradient mitigation should succeed");

    assert!(
        mitigated.iter().any(|&g| g.abs() > 1e-9),
        "all mitigated gradients were (near) zero: {mitigated:?}"
    );
}

/// Regression test: `PerformanceMetrics::current_performance()` used to
/// return a hardcoded `0.85` regardless of the data passed to
/// `update()`. It must now be a real function of the observed
/// measurements/gradients, and clearly noisy/unstable data must score
/// lower than clean/stable data.
#[test]
fn test_performance_metrics_are_data_driven() {
    let mut clean_metrics = PerformanceMetrics::new();
    let clean_measurements =
        Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("valid shape");
    let small_gradients = Array1::from_vec(vec![0.001, 0.001]);
    clean_metrics
        .update(&clean_measurements, &small_gradients)
        .expect("update should succeed");

    let mut noisy_metrics = PerformanceMetrics::new();
    let noisy_measurements =
        Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.5, 0.5]).expect("valid shape");
    let large_gradients = Array1::from_vec(vec![10.0, 10.0]);
    noisy_metrics
        .update(&noisy_measurements, &large_gradients)
        .expect("update should succeed");

    assert!(
        clean_metrics.current_performance() > noisy_metrics.current_performance(),
        "clean data ({}) should score higher than noisy data ({})",
        clean_metrics.current_performance(),
        noisy_metrics.current_performance()
    );
    // Neither score should be the old hardcoded placeholder.
    assert!((clean_metrics.current_performance() - 0.85).abs() > 1e-9);
}

/// Regression test: `switch_to_best_performing_strategy`/
/// `switch_to_resource_optimal_strategy` used to be no-ops (`Ok(())`
/// that changed nothing). They must now really escalate/relax the
/// active strategy's hyperparameters in place.
#[test]
fn test_strategy_adaptation_actually_changes_hyperparameters() {
    let mut mitigator = QuantumMLErrorMitigator::new(
        MitigationStrategy::ZNE {
            scale_factors: vec![1.0, 3.0],
            extrapolation_method: ExtrapolationMethod::Polynomial { degree: 1 },
            circuit_folding: CircuitFoldingMethod::GlobalFolding,
        },
        minimal_noise_model(),
    )
    .expect("mitigator construction should succeed");

    mitigator
        .switch_to_best_performing_strategy()
        .expect("escalation should succeed");
    let escalated_len = match &mitigator.mitigation_strategy {
        MitigationStrategy::ZNE { scale_factors, .. } => scale_factors.len(),
        _ => panic!("strategy variant changed unexpectedly"),
    };
    assert_eq!(
        escalated_len, 3,
        "escalating should add a new, higher scale factor"
    );

    mitigator
        .switch_to_resource_optimal_strategy()
        .expect("relaxation should succeed");
    let relaxed_len = match &mitigator.mitigation_strategy {
        MitigationStrategy::ZNE { scale_factors, .. } => scale_factors.len(),
        _ => panic!("strategy variant changed unexpectedly"),
    };
    assert_eq!(
        relaxed_len, 2,
        "relaxing should drop the most expensive scale factor"
    );

    // Readout correction method escalation.
    let mut readout_mitigator = QuantumMLErrorMitigator::new(
        MitigationStrategy::ReadoutErrorMitigation {
            calibration_matrix: Array2::eye(2),
            correction_method: ReadoutCorrectionMethod::MatrixInversion,
            regularization: 1e-6,
        },
        minimal_noise_model(),
    )
    .expect("mitigator construction should succeed");

    readout_mitigator
        .switch_to_best_performing_strategy()
        .expect("escalation should succeed");
    match &readout_mitigator.mitigation_strategy {
        MitigationStrategy::ReadoutErrorMitigation {
            correction_method, ..
        } => assert!(matches!(
            correction_method,
            ReadoutCorrectionMethod::ConstrainedLeastSquares
        )),
        _ => panic!("strategy variant changed unexpectedly"),
    }
}

/// Regression test: `StrategySelectionPolicy::select_strategy` used to
/// unconditionally return `strategies[0]`, ignoring what kind of
/// strategy it actually was. It must now prefer a fully implemented
/// strategy (`ReadoutErrorMitigation`) over a not-yet-implemented one
/// (`CDR`) even when the not-yet-implemented one comes first.
#[test]
fn test_strategy_selection_prefers_implemented_strategy() {
    let policy = StrategySelectionPolicy;
    let circuit = simple_rotation_circuit();
    let metrics = PerformanceMetrics::new();

    let strategies = vec![
        MitigationStrategy::CDR {
            training_circuits: vec![],
            regression_model: CDRModel,
            feature_extraction: FeatureExtractionMethod::CircuitDepth,
        },
        MitigationStrategy::ReadoutErrorMitigation {
            calibration_matrix: Array2::eye(2),
            correction_method: ReadoutCorrectionMethod::MatrixInversion,
            regularization: 1e-6,
        },
    ];

    let selected = policy
        .select_strategy(&circuit, &metrics, &strategies)
        .expect("selection should succeed");

    assert!(matches!(
        selected,
        MitigationStrategy::ReadoutErrorMitigation { .. }
    ));
}

/// Regression test: the readout-error-mitigation matrix-inversion path
/// must actually transform the measurements via the calibration matrix,
/// not silently pass them through unchanged.
#[test]
fn test_readout_matrix_inversion_actually_corrects() {
    let calibration_matrix =
        Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.1, 0.9]).expect("valid calibration matrix");
    let mitigator = QuantumMLErrorMitigator::new(
        MitigationStrategy::ReadoutErrorMitigation {
            calibration_matrix: calibration_matrix.clone(),
            correction_method: ReadoutCorrectionMethod::MatrixInversion,
            regularization: 1e-6,
        },
        minimal_noise_model(),
    )
    .expect("mitigator construction should succeed");

    let circuit = simple_rotation_circuit();
    let measurements =
        Array2::from_shape_vec((1, 2), vec![0.86, 0.18]).expect("valid measurement shape");

    let corrected = mitigator
        .apply_measurement_mitigation(&circuit, &measurements)
        .expect("readout correction should succeed");

    // corrected = M^-1 @ observed must differ from the raw input.
    let diff: f64 = corrected
        .iter()
        .zip(measurements.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "matrix-inversion correction left measurements unchanged: {corrected:?}"
    );
}

/// Regression test: strategies that are honestly not yet implemented
/// (CDR, symmetry verification, virtual distillation, ML mitigation,
/// hybrid correction) must return `MLError::NotSupported` from the
/// public mitigation entry points rather than silently fabricating a
/// cloned/no-op "corrected" result.
#[test]
fn test_unimplemented_strategies_honestly_error() {
    let mut mitigator = QuantumMLErrorMitigator::new(
        MitigationStrategy::CDR {
            training_circuits: vec![],
            regression_model: CDRModel,
            feature_extraction: FeatureExtractionMethod::CircuitDepth,
        },
        minimal_noise_model(),
    )
    .expect("mitigator construction should succeed");

    let circuit = simple_rotation_circuit();
    let measurements =
        Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).expect("valid measurement shape");
    let parameters = Array1::from_vec(vec![0.3, 0.7]);
    let gradients = Array1::from_vec(vec![0.1, 0.1]);

    let result =
        mitigator.mitigate_training_errors(&circuit, &parameters, &measurements, &gradients);
    assert!(matches!(result, Err(MLError::NotSupported(_))));
}
