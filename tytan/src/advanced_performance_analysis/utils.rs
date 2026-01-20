//! Utility functions for performance analysis

use super::*;

/// Utility functions for testing the performance analysis system
pub mod test_utils {
    use super::*;

    /// Create a test benchmark configuration
    pub fn create_test_benchmark_config() -> BenchmarkConfig {
        BenchmarkConfig {
            iterations: 5,
            warmup_iterations: 1,
            problem_sizes: vec![10, 20, 50],
            time_limit: Duration::from_secs(30),
            memory_limit: 1024 * 1024 * 1024, // 1GB
            detailed_profiling: true,
        }
    }

    /// Create test problem characteristics
    pub fn create_test_problem_characteristics() -> ProblemCharacteristics {
        ProblemCharacteristics {
            problem_size: 100,
            density: 0.5,
            structure: ProblemStructure::Random,
            symmetries: vec![],
            hardness_indicators: HashMap::new(),
        }
    }

    /// Create mock training data
    pub fn create_mock_training_data() -> Vec<TrainingExample> {
        vec![TrainingExample {
            features: {
                let mut features = HashMap::new();
                features.insert("problem_size".to_string(), 100.0);
                features.insert("density".to_string(), 0.5);
                features
            },
            targets: {
                let mut targets = HashMap::new();
                targets.insert("execution_time".to_string(), 1.2);
                targets.insert("memory_usage".to_string(), 0.8);
                targets
            },
            metadata: HashMap::new(),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = create_comprehensive_analyzer();
        assert!(analyzer.config.real_time_monitoring);
        assert_eq!(analyzer.config.monitoring_frequency, 1.0);
    }

    #[test]
    fn test_lightweight_analyzer() {
        let analyzer = create_lightweight_analyzer();
        assert_eq!(analyzer.config.collection_level, MetricsLevel::Basic);
        assert_eq!(analyzer.config.analysis_depth, AnalysisDepth::Surface);
        assert!(!analyzer.config.comparative_analysis);
    }

    #[test]
    fn test_trend_calculation() {
        let analyzer = create_comprehensive_analyzer();

        // Test improving trend
        let mut improving_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            analyzer.calculate_trend(&improving_values),
            TrendDirection::Improving
        );

        // Test degrading trend
        let mut degrading_values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(
            analyzer.calculate_trend(&degrading_values),
            TrendDirection::Degrading
        );

        // Test stable trend
        let mut stable_values = vec![3.0, 3.01, 2.99, 3.0, 3.01];
        assert_eq!(
            analyzer.calculate_trend(&stable_values),
            TrendDirection::Stable
        );
    }

    #[test]
    fn test_system_info_collection() {
        let system_info = SystemInfo::collect();
        assert!(!system_info.os.is_empty());
        assert!(system_info.cpu.cores > 0);
        assert!(system_info.memory.total_memory > 0.0);
    }

    #[test]
    fn test_monitor_functionality() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = CpuMonitor::new();
        assert!(!monitor.is_active());

        monitor.start_monitoring()?;
        assert!(monitor.is_active());

        let metrics = monitor.get_current_metrics()?;
        assert!(metrics.contains_key("cpu_utilization"));

        monitor.stop_monitoring()?;
        assert!(!monitor.is_active());

        Ok(())
    }

    #[test]
    fn test_benchmark_execution() -> Result<(), Box<dyn std::error::Error>> {
        let benchmark = QuboEvaluationBenchmark::new();
        let config = create_test_benchmark_config();

        let result = benchmark.run_benchmark(&config)?;
        assert_eq!(result.execution_times.len(), 5);
        assert_eq!(result.memory_usage.len(), 5);
        assert_eq!(result.solution_quality.len(), 5);

        Ok(())
    }

    #[test]
    fn test_prediction_model() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = LinearRegressionModel::new();

        let characteristics = create_test_problem_characteristics();

        let predictions = model.predict_performance(&characteristics)?;
        assert!(predictions.contains_key("execution_time"));
        assert!(predictions.contains_key("memory_usage"));
        assert!(predictions.contains_key("solution_quality"));

        // Test training
        let training_data = create_mock_training_data();
        model.train(&training_data)?;
        assert!(model.get_accuracy() > 0.0);

        Ok(())
    }

    #[test]
    fn test_config_creation() {
        let mut config = create_default_analysis_config();
        assert!(config.real_time_monitoring);
        assert_eq!(config.monitoring_frequency, 1.0);
        assert_eq!(config.collection_level, MetricsLevel::Detailed);

        let lightweight_config = create_lightweight_config();
        assert_eq!(lightweight_config.collection_level, MetricsLevel::Basic);
        assert_eq!(lightweight_config.analysis_depth, AnalysisDepth::Surface);
    }
}
