//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::applications::{ApplicationError, ApplicationResult};
use scirs2_core::random::prelude::*;
use std::time::{Duration, Instant};

use super::types::*;

/// Problem generator trait
pub trait ProblemGenerator: Send + Sync {
    /// Generate problem instance
    fn generate_instance(
        &self,
        size: usize,
        params: &GenerationParameters,
    ) -> ApplicationResult<ProblemInstance>;
    /// Get generator name
    fn get_name(&self) -> &str;
    /// Get supported problem category
    fn get_category(&self) -> ProblemCategory;
}
/// Classical solver trait
pub trait ClassicalSolver: Send + Sync {
    /// Solve optimization problem
    fn solve(
        &self,
        problem: &ProblemRepresentation,
        time_limit: Duration,
    ) -> ApplicationResult<ClassicalSolutionResult>;
    /// Get algorithm name
    fn get_algorithm_name(&self) -> ClassicalAlgorithm;
    /// Tune algorithm parameters
    fn tune_parameters(&mut self, instances: &[ProblemInstance]) -> ApplicationResult<()>;
}
/// Statistical test trait
pub trait StatisticalTest: Send + Sync {
    /// Perform statistical test
    fn perform_test(&self, data: &StatisticalTestData) -> ApplicationResult<TestResult>;
    /// Get test name
    fn get_test_name(&self) -> &str;
    /// Get test assumptions
    fn get_assumptions(&self) -> Vec<String>;
}
/// Effect size calculator trait
pub trait EffectSizeCalculator: Send + Sync {
    /// Calculate effect size
    fn calculate_effect_size(&self, data: &StatisticalTestData) -> ApplicationResult<f64>;
    /// Get effect size name
    fn get_effect_size_name(&self) -> &str;
    /// Get interpretation guidelines
    fn get_interpretation(&self, effect_size: f64) -> EffectSizeInterpretation;
}
/// Certification criterion evaluator
pub trait CertificationCriterionEvaluator: Send + Sync {
    /// Evaluate criterion
    fn evaluate(
        &self,
        result: &AdvantageDemonstrationResult,
    ) -> ApplicationResult<CriterionEvaluation>;
    /// Get criterion name
    fn get_criterion_name(&self) -> &str;
    /// Get weight in overall evaluation
    fn get_weight(&self) -> f64;
}
/// Evidence evaluator
pub trait EvidenceEvaluator: Send + Sync {
    /// Evaluate evidence quality
    fn evaluate_evidence(&self, evidence: &Evidence) -> ApplicationResult<EvidenceQuality>;
    /// Get evaluator name
    fn get_evaluator_name(&self) -> &str;
}
/// Create example quantum advantage demonstrator
pub fn create_example_advantage_demonstrator() -> ApplicationResult<QuantumAdvantageDemonstrator> {
    let config = AdvantageConfig::default();
    Ok(QuantumAdvantageDemonstrator::new(config))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_advantage_demonstrator_creation() {
        let demonstrator =
            create_example_advantage_demonstrator().expect("Demonstrator creation should succeed");
        assert_eq!(demonstrator.config.confidence_level, 0.95);
        assert_eq!(demonstrator.config.num_repetitions, 100);
    }
    #[test]
    fn test_advantage_config_defaults() {
        let config = AdvantageConfig::default();
        assert!(config
            .classical_algorithms
            .contains(&ClassicalAlgorithm::SimulatedAnnealing));
        assert!(config
            .quantum_devices
            .contains(&QuantumDevice::DWaveAdvantage));
        assert!(config
            .advantage_metrics
            .contains(&AdvantageMetric::TimeToSolution));
    }
    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert!(suite.config.include_standard_benchmarks);
        assert!(suite.config.include_random_instances);
        assert_eq!(suite.metadata.version, "1.0.0");
    }
    #[test]
    fn test_certification_levels() {
        let levels = vec![
            CertificationLevel::NoAdvantage,
            CertificationLevel::WeakEvidence,
            CertificationLevel::ModerateEvidence,
            CertificationLevel::StrongEvidence,
            CertificationLevel::DefinitiveAdvantage,
        ];
        assert_eq!(levels.len(), 5);
    }
    #[test]
    fn test_advantage_metrics() {
        let metrics = vec![
            AdvantageMetric::TimeToSolution,
            AdvantageMetric::SolutionQuality,
            AdvantageMetric::EnergyConsumption,
            AdvantageMetric::CostEfficiency,
            AdvantageMetric::Scalability,
        ];
        assert_eq!(metrics.len(), 5);
    }
}
