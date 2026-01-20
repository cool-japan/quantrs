//! Automated testing framework for quantum optimization.
//!
//! This module provides comprehensive testing tools for QUBO problems,
//! including test case generation, validation, and benchmarking.

#![allow(dead_code)]

mod config;
mod framework;
mod generators;
mod reports;
mod results;
mod types;
mod validators;

// Re-export all public types
pub use config::*;
pub use framework::*;
pub use generators::*;
pub use reports::*;
pub use results::*;
pub use types::*;
pub use validators::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SASampler;
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    #[ignore]
    fn test_testing_framework() {
        let config = TestConfig {
            seed: Some(42),
            cases_per_category: 5,
            problem_sizes: vec![5, 10],
            samplers: vec![SamplerConfig {
                name: "SA".to_string(),
                num_samples: 100,
                parameters: HashMap::new(),
            }],
            timeout: Duration::from_secs(10),
            validation: ValidationConfig {
                check_constraints: true,
                check_objective: true,
                statistical_tests: false,
                tolerance: 1e-6,
                min_quality: 0.0,
            },
            output: OutputConfig {
                generate_report: true,
                format: ReportFormat::Text,
                output_dir: "/tmp".to_string(),
                verbosity: VerbosityLevel::Info,
            },
        };

        let mut framework = TestingFramework::new(config);

        // Add test categories
        framework.add_category(TestCategory {
            name: "Graph Problems".to_string(),
            description: "Graph-based optimization problems".to_string(),
            problem_types: vec![ProblemType::MaxCut, ProblemType::GraphColoring],
            difficulties: vec![Difficulty::Easy, Difficulty::Medium],
            tags: vec!["graph".to_string()],
        });

        // Generate test suite
        let result = framework.generate_suite();
        assert!(result.is_ok());
        assert!(!framework.suite.test_cases.is_empty());

        // Run tests
        let sampler = SASampler::new(Some(42));
        let result = framework.run_suite(&sampler);
        assert!(result.is_ok());

        // Check results
        assert!(framework.results.summary.total_tests > 0);
        assert!(framework.results.summary.success_rate >= 0.0);

        // Generate report
        let report = framework.generate_report();
        assert!(report.is_ok());
    }
}
