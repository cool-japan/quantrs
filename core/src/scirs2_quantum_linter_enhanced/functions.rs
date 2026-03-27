//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::gate_translation::GateType;
use crate::parallel_ops_stubs::*;
use crate::scirs2_quantum_linter::{LintSeverity, LintingConfig, QuantumGate};

use super::types::{
    CustomLintRule, EnhancedLintingConfig, EnhancedQuantumLinter, GatePatternMatcher,
    HardwareArchitecture, LintFindingType, LintPattern,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_enhanced_linter_creation() {
        let linter = EnhancedQuantumLinter::new();
        assert!(linter.config.enable_ml_pattern_detection);
    }
    #[test]
    fn test_redundant_gates_detection() {
        let linter = EnhancedQuantumLinter::new();
        let gates = vec![
            QuantumGate::new(GateType::X, vec![0], None),
            QuantumGate::new(GateType::X, vec![0], None),
        ];
        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report
            .findings
            .iter()
            .any(|f| f.finding_type == LintFindingType::RedundantGates));
    }
    #[test]
    fn test_pattern_matching() {
        let linter = EnhancedQuantumLinter::new();
        let gates = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::H, vec![0], None),
        ];
        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(!report.findings.is_empty());
    }
    #[test]
    fn test_complexity_analysis() {
        let config = EnhancedLintingConfig {
            enable_complexity_analysis: true,
            ..Default::default()
        };
        let linter = EnhancedQuantumLinter::with_config(config);
        let mut gates = Vec::new();
        for i in 0..60 {
            gates.push(QuantumGate::new(GateType::T, vec![i % 5], None));
        }
        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report.findings.iter().any(|f| matches!(f.finding_type,
            LintFindingType::CustomRule(ref s) if s.contains("T-count"))));
    }
    #[test]
    fn test_hardware_compatibility() {
        let config = EnhancedLintingConfig {
            enable_hardware_specific_linting: true,
            target_architectures: vec![HardwareArchitecture::IBMQ],
            ..Default::default()
        };
        let linter = EnhancedQuantumLinter::with_config(config);
        let gates = vec![QuantumGate::new(
            GateType::Controlled(Box::new(GateType::Controlled(Box::new(GateType::X)))),
            vec![2],
            Some(vec![0, 1]),
        )];
        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report
            .findings
            .iter()
            .any(|f| f.finding_type == LintFindingType::UnsupportedGateSet));
    }
    #[test]
    fn test_custom_rules() {
        let custom_rule = CustomLintRule {
            name: "No X after Z".to_string(),
            description: "X gate should not follow Z gate".to_string(),
            pattern: LintPattern::GateSequence(vec![
                GatePatternMatcher {
                    gate_type: Some(GateType::Z),
                    qubit_count: None,
                    is_controlled: None,
                    is_parameterized: None,
                },
                GatePatternMatcher {
                    gate_type: Some(GateType::X),
                    qubit_count: None,
                    is_controlled: None,
                    is_parameterized: None,
                },
            ]),
            severity: LintSeverity::Warning,
            fix_suggestion: Some("Reorder gates".to_string()),
        };
        let config = EnhancedLintingConfig {
            custom_rules: vec![custom_rule],
            ..Default::default()
        };
        let linter = EnhancedQuantumLinter::with_config(config);
        let gates = vec![
            QuantumGate::new(GateType::Z, vec![0], None),
            QuantumGate::new(GateType::X, vec![0], None),
        ];
        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report.findings.iter().any(|f| matches!(f.finding_type,
            LintFindingType::CustomRule(ref s) if s.contains("No X after Z"))));
    }
    #[test]
    fn test_quality_metrics() {
        let linter = EnhancedQuantumLinter::new();
        let gates = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::CNOT, vec![0, 1], None),
        ];
        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report.metrics.circuit_quality_score > 0.0);
        assert!(report.metrics.performance_score > 0.0);
        assert!(report.metrics.hardware_readiness_score > 0.0);
        assert!(report.metrics.maintainability_score > 0.0);
    }
}
