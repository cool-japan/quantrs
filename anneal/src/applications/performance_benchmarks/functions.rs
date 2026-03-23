//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::applications::ApplicationResult;
use crate::simulator::{
    AnnealingParams, AnnealingResult, AnnealingSolution, ClassicalAnnealingSimulator,
    QuantumAnnealingSimulator,
};

use super::types::{BenchmarkConfiguration, PerformanceBenchmarkSuite, TemperatureProfile};
use crate::applications::unified::SolverType;

/// Run the complete performance benchmark suite with default configuration
pub fn run_performance_benchmarks() -> ApplicationResult<()> {
    let config = BenchmarkConfiguration::default();
    let mut benchmark_suite = PerformanceBenchmarkSuite::new(config);
    benchmark_suite.run_all_benchmarks()?;
    Ok(())
}
/// Run performance benchmarks with custom configuration
pub fn run_performance_benchmarks_with_config(
    config: BenchmarkConfiguration,
) -> ApplicationResult<()> {
    let mut benchmark_suite = PerformanceBenchmarkSuite::new(config);
    benchmark_suite.run_all_benchmarks()?;
    Ok(())
}
/// Create a quick performance benchmark for a specific industry
pub fn quick_benchmark(industry: &str, sizes: Vec<usize>) -> ApplicationResult<String> {
    let mut config = BenchmarkConfiguration::default();
    config.benchmark_industries = vec![industry.to_string()];
    config.problem_sizes = sizes;
    config.repetitions = 3;
    let mut benchmark_suite = PerformanceBenchmarkSuite::new(config);
    benchmark_suite.run_all_benchmarks()?;
    benchmark_suite.generate_performance_report()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfiguration::default();
        let suite = PerformanceBenchmarkSuite::new(config);
        assert_eq!(suite.results.len(), 0);
        assert!(!suite.config.benchmark_industries.is_empty());
    }
    #[test]
    fn test_benchmark_configuration() {
        let config = BenchmarkConfiguration::default();
        assert!(config.benchmark_industries.contains(&"finance".to_string()));
        assert!(config.problem_sizes.contains(&10));
        assert!(config.solver_types.contains(&SolverType::Classical));
    }
    #[test]
    fn test_temperature_profile_application() {
        let suite = PerformanceBenchmarkSuite::new(BenchmarkConfiguration::default());
        let mut params = AnnealingParams::default();
        suite.apply_temperature_profile(&mut params, &TemperatureProfile::Linear);
    }
    #[test]
    fn test_system_info_gathering() {
        let system_info = PerformanceBenchmarkSuite::gather_system_info();
        assert!(!system_info.os.is_empty());
        assert!(system_info.cpu_info.num_cores > 0);
        assert!(system_info.memory_info.total_memory_gb > 0.0);
    }
    #[test]
    fn test_benchmark_problem_config_creation() {
        let suite = PerformanceBenchmarkSuite::new(BenchmarkConfiguration::default());
        let finance_config = suite
            .create_benchmark_problem_config("finance", 10)
            .expect("Failed to create finance benchmark problem config");
        assert!(finance_config.contains_key("num_assets"));
        let logistics_config = suite
            .create_benchmark_problem_config("logistics", 8)
            .expect("Failed to create logistics benchmark problem config");
        assert!(logistics_config.contains_key("num_vehicles"));
    }
}
