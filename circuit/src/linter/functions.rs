//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::builder::Circuit;

use super::types::{
    ComplexityAnalyzer, LinterConfig, PatternDetector, QuantumLinter, StyleChecker,
};

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;
    use quantrs2_core::qubit::QubitId;
    #[test]
    fn test_linter_creation() {
        let circuit = Circuit::<2>::new();
        let linter = QuantumLinter::new(circuit);
        assert!(linter.config.enable_pattern_detection);
    }
    #[test]
    fn test_linting_process() {
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("add H gate to circuit");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("add CNOT gate to circuit");
        let mut linter = QuantumLinter::new(circuit);
        let result = linter.lint_circuit().expect("lint_circuit should succeed");
        assert!(result.quality_score >= 0.0 && result.quality_score <= 1.0);
        assert!(result.metadata.analysis_scope.total_gates > 0);
    }
    #[test]
    fn test_pattern_detector() {
        let circuit = Circuit::<2>::new();
        let detector = PatternDetector::new();
        let config = LinterConfig::default();
        let result = detector
            .detect_all_patterns(&circuit, &config)
            .expect("detect_all_patterns should succeed");
        assert!(result.pattern_score >= 0.0 && result.pattern_score <= 1.0);
    }
    #[test]
    fn test_style_checker() {
        let circuit = Circuit::<2>::new();
        let checker = StyleChecker::new();
        let config = LinterConfig::default();
        let (issues, analysis) = checker
            .check_all_styles(&circuit, &config)
            .expect("check_all_styles should succeed");
        assert!(analysis.overall_score >= 0.0 && analysis.overall_score <= 1.0);
    }
    #[test]
    fn test_complexity_analyzer() {
        let circuit = Circuit::<2>::new();
        let analyzer = ComplexityAnalyzer::new();
        let config = LinterConfig::default();
        let metrics = analyzer
            .analyze_complexity(&circuit, &config)
            .expect("analyze_complexity should succeed");
        assert!(metrics.overall_complexity >= 0.0);
    }
}
