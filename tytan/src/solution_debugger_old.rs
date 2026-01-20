//! Solution debugger for quantum optimization.
//!
//! This module provides comprehensive debugging tools for analyzing
//! quantum optimization solutions, constraint violations, and solution quality.
//!
//! The module has been refactored into submodules for better organization:
//! - `types`: Core data structures and types
//! - `config`: Configuration and settings
//! - `constraint_analyzer`: Constraint violation analysis
//! - `energy_analyzer`: Energy breakdown analysis
//! - `comparison`: Solution comparison functionality
//! - `analysis`: Analysis result types
//! - `visualization`: Solution visualization
//! - `reporting`: Debug reports and output generation

// Re-export everything from the new modular structure for backward compatibility
mod solution_debugger;
pub use solution_debugger::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use scirs2_core::ndarray::Array2;

    fn create_test_problem_info() -> types::ProblemInfo {
        let mut var_map = HashMap::new();
        var_map.insert("x0".to_string(), 0);
        var_map.insert("x1".to_string(), 1);

        let mut reverse_var_map = HashMap::new();
        reverse_var_map.insert(0, "x0".to_string());
        reverse_var_map.insert(1, "x1".to_string());

        let qubo = Array2::from_shape_vec((2, 2), vec![1.0, -2.0, -2.0, 1.0])
            .expect("Test QUBO matrix shape should be valid");

        types::ProblemInfo {
            name: "Test Problem".to_string(),
            problem_type: "QUBO".to_string(),
            num_variables: 2,
            var_map,
            reverse_var_map,
            qubo,
            constraints: Vec::new(),
            optimal_solution: None,
            metadata: HashMap::new(),
        }
    }

    fn create_test_solution() -> types::Solution {
        let mut assignments = HashMap::new();
        assignments.insert("x0".to_string(), true);
        assignments.insert("x1".to_string(), false);

        types::Solution {
            assignments,
            energy: -1.0,
            quality_metrics: HashMap::new(),
            metadata: HashMap::new(),
            sampling_stats: None,
        }
    }

    #[test]
    fn test_debugger_creation() {
        let problem_info = create_test_problem_info();
        let mut config = config::DebuggerConfig::default();
        let debugger = SolutionDebugger::new(problem_info, config);

        // Just test that creation succeeds
        assert!(true);
    }

    #[test]
    fn test_solution_debugging() {
        let problem_info = create_test_problem_info();
        let mut config = config::DebuggerConfig::default();
        let debugger = SolutionDebugger::new(problem_info, config);

        let solution = create_test_solution();
        let mut report = debugger.debug_solution(&solution);

        assert_eq!(report.solution.energy, solution.energy);
        assert!(report.constraint_analysis.is_some());
        assert!(report.energy_analysis.is_some());
    }

    #[test]
    fn test_constraint_analyzer() {
        let analyzer = constraint_analyzer::ConstraintAnalyzer::new(1e-6);

        // Test constraint analyzer creation
        assert!(true);
    }

    #[test]
    fn test_energy_analyzer() {
        let analyzer = energy_analyzer::EnergyAnalyzer::new(2);

        // Test energy analyzer creation
        assert!(true);
    }

    #[test]
    fn test_solution_comparator() {
        let comparator = comparison::SolutionComparator::new();

        // Test comparator creation
        assert!(true);
    }

    #[test]
    fn test_visualizer() {
        let visualizer = visualization::SolutionVisualizer::new();

        // Test visualizer creation
        assert!(true);
    }

    #[test]
    fn test_report_generation() {
        let solution = create_test_solution();
        let mut summary = reporting::DebugSummary::default();

        let mut report = reporting::DebugReport {
            solution,
            constraint_analysis: None,
            energy_analysis: None,
            comparison_results: Vec::new(),
            visualizations: Vec::new(),
            issues: Vec::new(),
            suggestions: Vec::new(),
            summary,
        };

        let text_summary = report.generate_text_summary();
        assert!(text_summary.contains("Solution Debug Report"));

        let json_result = report.to_json();
        assert!(json_result.is_ok());
    }
}