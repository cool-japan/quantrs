//! Tests for quantum circuit formatter

use super::*;
use crate::builder::Circuit;
use quantrs2_core::gate::multi::CNOT;
use quantrs2_core::gate::single::Hadamard;
use quantrs2_core::qubit::QubitId;

#[test]
fn test_formatter_creation() {
    let circuit = Circuit::<2>::new();
    let formatter = QuantumFormatter::new(circuit);
    assert_eq!(formatter.config.max_line_length, 100);
}

#[test]
fn test_formatting_process() {
    let mut circuit = Circuit::<2>::new();
    circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
    circuit
        .add_gate(CNOT {
            control: QubitId(0),
            target: QubitId(1),
        })
        .unwrap();

    let mut formatter = QuantumFormatter::new(circuit);
    let result = formatter.format_circuit().unwrap();

    assert!(!result.formatted_circuit.code.is_empty());
    assert!(result.statistics.total_lines > 0);
}

#[test]
fn test_config_defaults() {
    let config = FormatterConfig::default();
    assert_eq!(config.max_line_length, 100);
    assert_eq!(config.indentation.spaces_per_level, 4);
    assert!(config.spacing.around_operators);
}

#[test]
fn test_style_compliance() {
    let circuit = Circuit::<2>::new();
    let formatter = QuantumFormatter::new(circuit);

    let style_info = StyleInformation {
        applied_rules: Vec::new(),
        violations_fixed: Vec::new(),
        compliance_score: 0.9,
        consistency_metrics: ConsistencyMetrics {
            naming_consistency: 0.9,
            indentation_consistency: 0.9,
            spacing_consistency: 0.9,
            comment_consistency: 0.9,
            overall_consistency: 0.9,
        },
    };

    let compliance = formatter.assess_style_compliance(&style_info).unwrap();
    assert!(matches!(
        compliance.compliance_level,
        ComplianceLevel::Excellent
    ));
}
