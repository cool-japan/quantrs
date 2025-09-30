//! Utility functions and test helpers for advanced testing framework

use scirs2_core::random::prelude::*;
use super::*;

/// Create standard test configuration
pub fn create_standard_test_config(test_type: &str) -> ApplicationResult<TestingConfig> {
    match test_type {
        "performance" => Ok(TestingConfig {
            enable_parallel: true,
            max_concurrent_tests: 4,
            test_timeout: Duration::from_secs(600),
            performance_tolerance: 0.05,
            significance_level: 0.05,
            data_retention: Duration::from_secs(14 * 24 * 3600),
            detailed_logging: true,
            stress_test_sizes: vec![50, 100, 200, 500],
        }),
        "regression" => Ok(TestingConfig {
            enable_parallel: false,
            max_concurrent_tests: 1,
            test_timeout: Duration::from_secs(300),
            performance_tolerance: 0.1,
            significance_level: 0.01,
            data_retention: Duration::from_secs(60 * 24 * 3600),
            detailed_logging: true,
            stress_test_sizes: vec![10, 50, 100],
        }),
        "stress" => Ok(TestingConfig {
            enable_parallel: true,
            max_concurrent_tests: 8,
            test_timeout: Duration::from_secs(1200),
            performance_tolerance: 0.2,
            significance_level: 0.05,
            data_retention: Duration::from_secs(30 * 24 * 3600),
            detailed_logging: false,
            stress_test_sizes: vec![100, 500, 1000, 2000, 5000, 10000],
        }),
        "property" => Ok(TestingConfig {
            enable_parallel: true,
            max_concurrent_tests: 6,
            test_timeout: Duration::from_secs(180),
            performance_tolerance: 0.1,
            significance_level: 0.05,
            data_retention: Duration::from_secs(21 * 24 * 3600),
            detailed_logging: true,
            stress_test_sizes: vec![10, 25, 50, 100],
        }),
        _ => Err(ApplicationError::ConfigurationError(format!(
            "Unknown test type: {}",
            test_type
        ))),
    }
}

/// Create test problem with specific characteristics
pub fn create_test_problem(
    problem_type: ProblemType,
    size: usize,
    density: f64,
    seed: Option<u64>,
) -> ApplicationResult<IsingModel> {
    let mut problem = IsingModel::new(size);

    // Set random seed if provided
    let mut rng_seed = seed.unwrap_or_else(|| thread_rng().gen());

    match problem_type {
        ProblemType::RandomIsing => create_random_ising_problem(&mut problem, density, rng_seed)?,
        ProblemType::MaxCut => create_max_cut_problem(&mut problem, density, rng_seed)?,
        ProblemType::VertexCover => create_vertex_cover_problem(&mut problem, density, rng_seed)?,
        ProblemType::TSP => create_tsp_problem(&mut problem, density, rng_seed)?,
        ProblemType::Portfolio => create_portfolio_problem(&mut problem, density, rng_seed)?,
        ProblemType::Custom(ref name) => {
            return Err(ApplicationError::ConfigurationError(format!(
                "Custom problem type not implemented: {}",
                name
            )));
        }
    }

    Ok(problem)
}

/// Create random Ising model problem
fn create_random_ising_problem(
    problem: &mut IsingModel,
    density: f64,
    seed: u64,
) -> ApplicationResult<()> {
    let size = problem.num_qubits;

    // Use seed for reproducibility
    let mut local_seed = seed;

    // Set random biases
    for i in 0..size {
        local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let bias = ((local_seed % 2000) as f64 / 1000.0) - 1.0; // Range [-1, 1]
        problem.set_bias(i, bias)?;
    }

    // Set random couplings based on density
    let max_edges = size * (size - 1) / 2;
    let target_edges = (max_edges as f64 * density) as usize;

    let mut edges_added = 0;
    for i in 0..size {
        for j in (i + 1)..size {
            if edges_added >= target_edges {
                break;
            }

            local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
            if (local_seed % 1000) < (density * 1000.0) as u64 {
                local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
                let coupling = ((local_seed % 2000) as f64 / 1000.0) - 1.0; // Range [-1, 1]
                problem.set_coupling(i, j, coupling)?;
                edges_added += 1;
            }
        }
        if edges_added >= target_edges {
            break;
        }
    }

    Ok(())
}

/// Create Max-Cut problem instance
fn create_max_cut_problem(
    problem: &mut IsingModel,
    density: f64,
    seed: u64,
) -> ApplicationResult<()> {
    let size = problem.num_qubits;
    let mut local_seed = seed;

    // Max-Cut: no biases, only edge weights
    for i in 0..size {
        problem.set_bias(i, 0.0)?;
    }

    // Add edges with weights
    let max_edges = size * (size - 1) / 2;
    let target_edges = (max_edges as f64 * density) as usize;

    let mut edges_added = 0;
    for i in 0..size {
        for j in (i + 1)..size {
            if edges_added >= target_edges {
                break;
            }

            local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
            if (local_seed % 1000) < (density * 1000.0) as u64 {
                local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
                let weight = (local_seed % 100) as f64 / 100.0; // Range [0, 1]
                                                                // For Max-Cut, use negative coupling (ferromagnetic)
                problem.set_coupling(i, j, -weight)?;
                edges_added += 1;
            }
        }
        if edges_added >= target_edges {
            break;
        }
    }

    Ok(())
}

/// Create Vertex Cover problem instance
fn create_vertex_cover_problem(
    problem: &mut IsingModel,
    density: f64,
    seed: u64,
) -> ApplicationResult<()> {
    let size = problem.num_qubits;
    let mut local_seed = seed;

    // Vertex Cover: penalty for not covering edges
    // Set biases to encourage smaller covers
    for i in 0..size {
        problem.set_bias(i, 1.0)?; // Cost of including vertex
    }

    // Add edge constraints
    let max_edges = size * (size - 1) / 2;
    let target_edges = (max_edges as f64 * density) as usize;

    let mut edges_added = 0;
    for i in 0..size {
        for j in (i + 1)..size {
            if edges_added >= target_edges {
                break;
            }

            local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
            if (local_seed % 1000) < (density * 1000.0) as u64 {
                // Large penalty if neither vertex is in cover
                let penalty = 10.0;
                problem.set_coupling(i, j, penalty)?;
                edges_added += 1;
            }
        }
        if edges_added >= target_edges {
            break;
        }
    }

    Ok(())
}

/// Create TSP problem instance (simplified)
fn create_tsp_problem(problem: &mut IsingModel, _density: f64, seed: u64) -> ApplicationResult<()> {
    let size = problem.num_qubits;
    let mut local_seed = seed;

    // TSP encoding requires careful mapping - simplified version here
    // Set biases to encourage valid tours
    for i in 0..size {
        problem.set_bias(i, 0.0)?;
    }

    // Add constraints for TSP (simplified)
    for i in 0..size {
        for j in (i + 1)..size {
            local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let distance = (local_seed % 100) as f64 / 10.0; // Distance between cities
            problem.set_coupling(i, j, distance)?;
        }
    }

    Ok(())
}

/// Create Portfolio Optimization problem instance
fn create_portfolio_problem(
    problem: &mut IsingModel,
    _density: f64,
    seed: u64,
) -> ApplicationResult<()> {
    let size = problem.num_qubits;
    let mut local_seed = seed;

    // Portfolio: expected returns (biases) and correlations (couplings)
    for i in 0..size {
        local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let expected_return = ((local_seed % 200) as f64 / 1000.0) + 0.05; // 5-25% return
        problem.set_bias(i, -expected_return)?; // Negative because we want to maximize
    }

    // Add correlation matrix (risk)
    for i in 0..size {
        for j in (i + 1)..size {
            local_seed = local_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let correlation = ((local_seed % 100) as f64 / 500.0) - 0.1; // Range [-0.1, 0.1]
            problem.set_coupling(i, j, correlation)?;
        }
    }

    Ok(())
}

/// Validate test framework configuration
pub fn validate_framework_config(config: &TestingConfig) -> ApplicationResult<()> {
    if config.max_concurrent_tests == 0 {
        return Err(ApplicationError::ConfigurationError(
            "max_concurrent_tests must be greater than 0".to_string(),
        ));
    }

    if config.test_timeout.as_secs() == 0 {
        return Err(ApplicationError::ConfigurationError(
            "test_timeout must be greater than 0".to_string(),
        ));
    }

    if config.performance_tolerance < 0.0 || config.performance_tolerance > 1.0 {
        return Err(ApplicationError::ConfigurationError(
            "performance_tolerance must be between 0.0 and 1.0".to_string(),
        ));
    }

    if config.significance_level < 0.0 || config.significance_level > 1.0 {
        return Err(ApplicationError::ConfigurationError(
            "significance_level must be between 0.0 and 1.0".to_string(),
        ));
    }

    if config.stress_test_sizes.is_empty() {
        return Err(ApplicationError::ConfigurationError(
            "stress_test_sizes cannot be empty".to_string(),
        ));
    }

    // Check that stress test sizes are in ascending order
    for i in 1..config.stress_test_sizes.len() {
        if config.stress_test_sizes[i] <= config.stress_test_sizes[i - 1] {
            return Err(ApplicationError::ConfigurationError(
                "stress_test_sizes must be in ascending order".to_string(),
            ));
        }
    }

    Ok(())
}

/// Calculate test quality metrics
pub fn calculate_test_quality_metrics(results: &[TestExecutionResult]) -> TestQualityMetrics {
    if results.is_empty() {
        return TestQualityMetrics {
            mean_quality: 0.0,
            std_dev_quality: 0.0,
            min_quality: 0.0,
            max_quality: 0.0,
            median_quality: 0.0,
            success_rate: 0.0,
            mean_execution_time: Duration::default(),
            std_dev_execution_time: Duration::default(),
        };
    }

    let qualities: Vec<f64> = results.iter().map(|r| r.solution_quality).collect();
    let execution_times: Vec<Duration> = results.iter().map(|r| r.execution_time).collect();

    // Quality statistics
    let mean_quality = qualities.iter().sum::<f64>() / qualities.len() as f64;
    let variance_quality = qualities
        .iter()
        .map(|q| (q - mean_quality).powi(2))
        .sum::<f64>()
        / (qualities.len() - 1).max(1) as f64;
    let std_dev_quality = variance_quality.sqrt();

    let mut sorted_qualities = qualities.clone();
    sorted_qualities.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_quality = if sorted_qualities.len() % 2 == 0 {
        (sorted_qualities[sorted_qualities.len() / 2 - 1]
            + sorted_qualities[sorted_qualities.len() / 2])
            / 2.0
    } else {
        sorted_qualities[sorted_qualities.len() / 2]
    };

    let min_quality = sorted_qualities[0];
    let max_quality = sorted_qualities[sorted_qualities.len() - 1];

    // Success rate (assuming convergence_achieved indicates success)
    let successful_tests = results.iter().filter(|r| r.convergence_achieved).count();
    let success_rate = successful_tests as f64 / results.len() as f64;

    // Execution time statistics
    let total_time: Duration = execution_times.iter().sum();
    let mean_execution_time = total_time / execution_times.len() as u32;

    let mean_time_secs = mean_execution_time.as_secs_f64();
    let variance_time = execution_times
        .iter()
        .map(|t| (t.as_secs_f64() - mean_time_secs).powi(2))
        .sum::<f64>()
        / (execution_times.len() - 1).max(1) as f64;
    let std_dev_execution_time = Duration::from_secs_f64(variance_time.sqrt());

    TestQualityMetrics {
        mean_quality,
        std_dev_quality,
        min_quality,
        max_quality,
        median_quality,
        success_rate,
        mean_execution_time,
        std_dev_execution_time,
    }
}

/// Generate test report summary
pub fn generate_test_summary(results: &TestSuiteResults) -> String {
    let mut summary = String::new();

    summary.push_str(&format!("# Test Suite Summary\n\n"));
    summary.push_str(&format!(
        "**Execution Time:** {:?}\n",
        results.execution_time
    ));
    summary.push_str(&format!(
        "**Overall Success:** {}\n\n",
        if results.overall_success {
            "✅"
        } else {
            "❌"
        }
    ));

    // Scenario tests summary
    if !results.scenario_results.is_empty() {
        let scenario_success = results
            .scenario_results
            .iter()
            .filter(|r| r.success)
            .count();
        summary.push_str(&format!(
            "## Scenario Tests\n- **Results:** {}/{} passed\n- **Success Rate:** {:.1}%\n\n",
            scenario_success,
            results.scenario_results.len(),
            (scenario_success as f64 / results.scenario_results.len() as f64) * 100.0
        ));
    }

    // Regression tests summary
    if !results.regression_results.is_empty() {
        let regressions_detected = results
            .regression_results
            .iter()
            .filter(|r| r.regression_detected)
            .count();
        summary.push_str(&format!(
            "## Regression Tests\n- **Regressions Detected:** {}\n- **Tests Analyzed:** {}\n\n",
            regressions_detected,
            results.regression_results.len()
        ));
    }

    // Platform tests summary
    if !results.platform_results.is_empty() {
        let avg_compatibility = results
            .platform_results
            .iter()
            .map(|r| r.compatibility_score)
            .sum::<f64>()
            / results.platform_results.len() as f64;

        summary.push_str(&format!(
            "## Platform Tests\n- **Platforms Tested:** {}\n- **Average Compatibility:** {:.2}\n\n",
            results.platform_results.len(),
            avg_compatibility
        ));
    }

    // Stress tests summary
    if !results.stress_results.is_empty() {
        let avg_success_rate = results
            .stress_results
            .iter()
            .map(|r| r.success_rate)
            .sum::<f64>()
            / results.stress_results.len() as f64;

        summary.push_str(&format!(
            "## Stress Tests\n- **Tests Completed:** {}\n- **Average Success Rate:** {:.1}%\n\n",
            results.stress_results.len(),
            avg_success_rate * 100.0
        ));
    }

    // Property tests summary
    if !results.property_results.is_empty() {
        let total_cases = results
            .property_results
            .iter()
            .map(|r| r.cases_tested)
            .sum::<usize>();
        let total_passed = results
            .property_results
            .iter()
            .map(|r| r.cases_passed)
            .sum::<usize>();

        summary.push_str(&format!(
            "## Property Tests\n- **Properties Tested:** {}\n- **Test Cases:** {} total, {} passed\n- **Overall Confidence:** {:.1}%\n\n",
            results.property_results.len(),
            total_cases,
            total_passed,
            if total_cases > 0 { (total_passed as f64 / total_cases as f64) * 100.0 } else { 0.0 }
        ));
    }

    summary.push_str(&format!(
        "---\n*Generated at: {}*\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    ));

    summary
}

/// Compare two test results
pub fn compare_test_results(
    result1: &TestExecutionResult,
    result2: &TestExecutionResult,
    tolerance: f64,
) -> TestComparisonResult {
    let quality_diff = (result1.solution_quality - result2.solution_quality).abs();
    let quality_similar = quality_diff <= tolerance;

    let time_diff = if result1.execution_time > result2.execution_time {
        result1.execution_time - result2.execution_time
    } else {
        result2.execution_time - result1.execution_time
    };

    let energy_diff = (result1.final_energy - result2.final_energy).abs();
    let energy_similar = energy_diff <= tolerance * result1.final_energy.abs().max(1.0);

    TestComparisonResult {
        quality_difference: quality_diff,
        quality_similar,
        time_difference: time_diff,
        energy_difference: energy_diff,
        energy_similar,
        overall_similar: quality_similar && energy_similar,
    }
}

/// Test quality metrics
#[derive(Debug, Clone)]
pub struct TestQualityMetrics {
    /// Mean solution quality
    pub mean_quality: f64,
    /// Standard deviation of quality
    pub std_dev_quality: f64,
    /// Minimum quality observed
    pub min_quality: f64,
    /// Maximum quality observed
    pub max_quality: f64,
    /// Median quality
    pub median_quality: f64,
    /// Success rate (convergence achieved)
    pub success_rate: f64,
    /// Mean execution time
    pub mean_execution_time: Duration,
    /// Standard deviation of execution time
    pub std_dev_execution_time: Duration,
}

/// Test comparison result
#[derive(Debug, Clone)]
pub struct TestComparisonResult {
    /// Absolute difference in solution quality
    pub quality_difference: f64,
    /// Whether qualities are similar within tolerance
    pub quality_similar: bool,
    /// Difference in execution time
    pub time_difference: Duration,
    /// Absolute difference in final energy
    pub energy_difference: f64,
    /// Whether energies are similar within tolerance
    pub energy_similar: bool,
    /// Overall similarity assessment
    pub overall_similar: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_config_creation() {
        let config = create_standard_test_config("performance").unwrap();
        assert!(config.enable_parallel);
        assert_eq!(config.max_concurrent_tests, 4);
        assert_eq!(config.performance_tolerance, 0.05);
    }

    #[test]
    fn test_config_validation() {
        let mut config = TestingConfig::default();
        assert!(validate_framework_config(&config).is_ok());

        config.max_concurrent_tests = 0;
        assert!(validate_framework_config(&config).is_err());

        config.max_concurrent_tests = 4;
        config.performance_tolerance = 1.5;
        assert!(validate_framework_config(&config).is_err());
    }

    #[test]
    fn test_problem_creation() {
        let problem = create_test_problem(ProblemType::RandomIsing, 10, 0.3, Some(42));
        assert!(problem.is_ok());

        let ising = problem.unwrap();
        assert_eq!(ising.num_qubits, 10);
    }

    #[test]
    fn test_quality_metrics_calculation() {
        let results = vec![
            TestExecutionResult {
                solution_quality: 0.9,
                execution_time: Duration::from_millis(100),
                final_energy: -0.9,
                best_solution: vec![1, -1],
                convergence_achieved: true,
                memory_used: 1024,
            },
            TestExecutionResult {
                solution_quality: 0.8,
                execution_time: Duration::from_millis(150),
                final_energy: -0.8,
                best_solution: vec![-1, 1],
                convergence_achieved: true,
                memory_used: 1024,
            },
        ];

        let metrics = calculate_test_quality_metrics(&results);
        assert!((metrics.mean_quality - 0.85).abs() < 1e-10);
        assert_eq!(metrics.success_rate, 1.0);
        assert_eq!(metrics.min_quality, 0.8);
        assert_eq!(metrics.max_quality, 0.9);
    }

    #[test]
    fn test_test_comparison() {
        let result1 = TestExecutionResult {
            solution_quality: 0.9,
            execution_time: Duration::from_millis(100),
            final_energy: -0.9,
            best_solution: vec![1],
            convergence_achieved: true,
            memory_used: 1024,
        };

        let result2 = TestExecutionResult {
            solution_quality: 0.92,
            execution_time: Duration::from_millis(110),
            final_energy: -0.91,
            best_solution: vec![1],
            convergence_achieved: true,
            memory_used: 1024,
        };

        let comparison = compare_test_results(&result1, &result2, 0.05);
        assert!(comparison.quality_similar);
        assert!(comparison.energy_similar);
        assert!(comparison.overall_similar);
    }

    #[test]
    fn test_max_cut_problem_creation() {
        let problem = create_test_problem(ProblemType::MaxCut, 5, 0.5, Some(123));
        assert!(problem.is_ok());

        let max_cut = problem.unwrap();
        assert_eq!(max_cut.num_qubits, 5);

        // Check that biases are zero (Max-Cut characteristic)
        for i in 0..5 {
            assert_eq!(max_cut.get_bias(i).unwrap(), 0.0);
        }
    }

    #[test]
    fn test_test_summary_generation() {
        let results = TestSuiteResults {
            scenario_results: vec![ScenarioTestResult {
                scenario_id: "test1".to_string(),
                execution_time: Duration::from_millis(100),
                test_result: TestExecutionResult {
                    solution_quality: 0.9,
                    execution_time: Duration::from_millis(100),
                    final_energy: -0.9,
                    best_solution: vec![1],
                    convergence_achieved: true,
                    memory_used: 1024,
                },
                validation_results: Vec::new(),
                success: true,
            }],
            regression_results: Vec::new(),
            platform_results: Vec::new(),
            stress_results: Vec::new(),
            property_results: Vec::new(),
            execution_time: Duration::from_secs(1),
            overall_success: true,
        };

        let summary = generate_test_summary(&results);
        assert!(summary.contains("Test Suite Summary"));
        assert!(summary.contains("1/1 passed"));
        assert!(summary.contains("✅"));
    }
}
