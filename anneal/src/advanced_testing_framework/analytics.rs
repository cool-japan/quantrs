//! Test result analytics and reporting

use super::*;

/// Test result analytics
#[derive(Debug)]
pub struct TestAnalytics {
    /// Result database
    pub result_database: TestResultDatabase,
    /// Analytics engines
    pub analytics_engines: Vec<AnalyticsEngine>,
    /// Report generators
    pub report_generators: Vec<ReportGenerator>,
    /// Visualization tools
    pub visualization_tools: Vec<VisualizationTool>,
}

/// Test result database
#[derive(Debug)]
pub struct TestResultDatabase {
    /// Execution records
    pub execution_records: HashMap<String, Vec<TestExecutionRecord>>,
    /// Performance trends
    pub performance_trends: HashMap<String, PerformanceTrend>,
    /// Failure patterns
    pub failure_patterns: HashMap<String, FailurePattern>,
    /// Database statistics
    pub statistics: DatabaseStatistics,
}

/// Test execution record
#[derive(Debug, Clone)]
pub struct TestExecutionRecord {
    /// Record identifier
    pub id: String,
    /// Test identifier
    pub test_id: String,
    /// Execution timestamp
    pub timestamp: Instant,
    /// Execution result
    pub result: TestExecutionResult,
    /// Test configuration
    pub config: HashMap<String, String>,
    /// Environment information
    pub environment: HashMap<String, String>,
}

/// Performance trend analysis
#[derive(Debug)]
pub struct PerformanceTrend {
    /// Metric being tracked
    pub metric: String,
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend magnitude
    pub trend_magnitude: f64,
    /// Confidence level
    pub confidence: f64,
    /// Data points
    pub data_points: VecDeque<(Instant, f64)>,
}

/// Failure pattern analysis
#[derive(Debug)]
pub struct FailurePattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern type
    pub pattern_type: FailurePatternType,
    /// Occurrence frequency
    pub frequency: f64,
    /// Pattern conditions
    pub conditions: Vec<PatternCondition>,
    /// Associated failures
    pub failures: Vec<FailureInstance>,
}

/// Conditions for pattern matching
#[derive(Debug, Clone)]
pub struct PatternCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition value
    pub value: PropertyValue,
    /// Condition operator
    pub operator: ConditionOperator,
}

/// Instance of failure occurrence
#[derive(Debug, Clone)]
pub struct FailureInstance {
    /// Failure timestamp
    pub timestamp: Instant,
    /// Test identifier
    pub test_id: String,
    /// Failure details
    pub details: TestError,
    /// Context information
    pub context: HashMap<String, String>,
}

/// Test error information
#[derive(Debug, Clone)]
pub struct TestError {
    /// Error type
    pub error_type: TestErrorType,
    /// Error message
    pub message: String,
    /// Error code
    pub code: Option<i32>,
    /// Error location
    pub location: Option<String>,
    /// Stack trace
    pub stack_trace: Option<String>,
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStatistics {
    /// Total test executions
    pub total_executions: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Data retention policy
    pub retention_policy: RetentionPolicy,
}

/// Analytics engine for test data
#[derive(Debug)]
pub struct AnalyticsEngine {
    /// Engine identifier
    pub id: String,
    /// Engine type
    pub engine_type: AnalyticsEngineType,
    /// Analysis algorithms
    pub algorithms: Vec<AnalysisAlgorithm>,
    /// Output format
    pub output_format: AnalyticsOutputFormat,
}

/// Analysis algorithm
#[derive(Debug)]
pub struct AnalysisAlgorithm {
    /// Algorithm identifier
    pub id: String,
    /// Algorithm type
    pub algorithm_type: String,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
    /// Required input data types
    pub input_types: Vec<String>,
}

/// Report generator
#[derive(Debug)]
pub struct ReportGenerator {
    /// Generator identifier
    pub id: String,
    /// Report type
    pub report_type: ReportType,
    /// Template configuration
    pub template_config: ReportTemplate,
    /// Output format
    pub output_format: ReportFormat,
}

/// Report types
#[derive(Debug, Clone, PartialEq)]
pub enum ReportType {
    /// Performance summary
    PerformanceSummary,
    /// Failure analysis
    FailureAnalysis,
    /// Trend analysis
    TrendAnalysis,
    /// Comparison report
    Comparison,
    /// Custom report
    Custom(String),
}

/// Report template
#[derive(Debug)]
pub struct ReportTemplate {
    /// Template identifier
    pub id: String,
    /// Template content
    pub content: String,
    /// Template variables
    pub variables: HashMap<String, String>,
    /// Styling configuration
    pub styling: ReportStyling,
}

/// Report styling configuration
#[derive(Debug)]
pub struct ReportStyling {
    /// Color scheme
    pub color_scheme: String,
    /// Font configuration
    pub font_config: HashMap<String, String>,
    /// Layout settings
    pub layout_settings: HashMap<String, String>,
}

/// Visualization tool
#[derive(Debug)]
pub struct VisualizationTool {
    /// Tool identifier
    pub id: String,
    /// Chart types supported
    pub supported_charts: Vec<ChartType>,
    /// Rendering engine
    pub rendering_engine: RenderingEngine,
    /// Interactive features
    pub interactive_features: Vec<InteractiveFeature>,
}

/// Rendering engine configuration
#[derive(Debug)]
pub struct RenderingEngine {
    /// Engine type
    pub engine_type: RenderingEngineType,
    /// Configuration parameters
    pub config: HashMap<String, String>,
    /// Performance settings
    pub performance_settings: HashMap<String, f64>,
}

/// Interactive features for visualizations
#[derive(Debug, Clone, PartialEq)]
pub enum InteractiveFeature {
    /// Zoom functionality
    Zoom,
    /// Pan functionality
    Pan,
    /// Hover tooltips
    Tooltips,
    /// Drill-down capability
    DrillDown,
    /// Data filtering
    Filtering,
    /// Export functionality
    Export,
}

impl TestAnalytics {
    pub fn new() -> Self {
        Self {
            result_database: TestResultDatabase {
                execution_records: HashMap::new(),
                performance_trends: HashMap::new(),
                failure_patterns: HashMap::new(),
                statistics: DatabaseStatistics {
                    total_executions: 0,
                    success_rate: 0.0,
                    avg_execution_time: Duration::default(),
                    retention_policy: RetentionPolicy {
                        retention_period: Duration::from_secs(30 * 24 * 3600),
                        cleanup_frequency: Duration::from_secs(7 * 24 * 3600),
                        archive_policy: None,
                    },
                },
            },
            analytics_engines: Self::create_default_engines(),
            report_generators: Self::create_default_generators(),
            visualization_tools: Self::create_default_visualization_tools(),
        }
    }

    /// Create default analytics engines
    fn create_default_engines() -> Vec<AnalyticsEngine> {
        vec![
            AnalyticsEngine {
                id: "statistical_analyzer".to_string(),
                engine_type: AnalyticsEngineType::Statistical,
                algorithms: vec![
                    AnalysisAlgorithm {
                        id: "descriptive_stats".to_string(),
                        algorithm_type: "descriptive_statistics".to_string(),
                        parameters: HashMap::new(),
                        input_types: vec!["numerical".to_string()],
                    },
                    AnalysisAlgorithm {
                        id: "correlation_analysis".to_string(),
                        algorithm_type: "correlation".to_string(),
                        parameters: HashMap::new(),
                        input_types: vec!["time_series".to_string()],
                    },
                ],
                output_format: AnalyticsOutputFormat::JSON,
            },
            AnalyticsEngine {
                id: "trend_analyzer".to_string(),
                engine_type: AnalyticsEngineType::TimeSeries,
                algorithms: vec![AnalysisAlgorithm {
                    id: "trend_detection".to_string(),
                    algorithm_type: "linear_regression".to_string(),
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("window_size".to_string(), 30.0);
                        params
                    },
                    input_types: vec!["time_series".to_string()],
                }],
                output_format: AnalyticsOutputFormat::CSV,
            },
            AnalyticsEngine {
                id: "pattern_recognizer".to_string(),
                engine_type: AnalyticsEngineType::PatternRecognition,
                algorithms: vec![AnalysisAlgorithm {
                    id: "failure_clustering".to_string(),
                    algorithm_type: "k_means".to_string(),
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("num_clusters".to_string(), 5.0);
                        params
                    },
                    input_types: vec!["failure_data".to_string()],
                }],
                output_format: AnalyticsOutputFormat::JSON,
            },
        ]
    }

    /// Create default report generators
    fn create_default_generators() -> Vec<ReportGenerator> {
        vec![
            ReportGenerator {
                id: "performance_reporter".to_string(),
                report_type: ReportType::PerformanceSummary,
                template_config: ReportTemplate {
                    id: "performance_template".to_string(),
                    content:
                        "# Performance Summary\n\n{{performance_metrics}}\n\n{{trend_analysis}}"
                            .to_string(),
                    variables: HashMap::new(),
                    styling: ReportStyling {
                        color_scheme: "blue".to_string(),
                        font_config: HashMap::new(),
                        layout_settings: HashMap::new(),
                    },
                },
                output_format: ReportFormat::HTML,
            },
            ReportGenerator {
                id: "failure_reporter".to_string(),
                report_type: ReportType::FailureAnalysis,
                template_config: ReportTemplate {
                    id: "failure_template".to_string(),
                    content: "# Failure Analysis\n\n{{failure_patterns}}\n\n{{recommendations}}"
                        .to_string(),
                    variables: HashMap::new(),
                    styling: ReportStyling {
                        color_scheme: "red".to_string(),
                        font_config: HashMap::new(),
                        layout_settings: HashMap::new(),
                    },
                },
                output_format: ReportFormat::PDF,
            },
        ]
    }

    /// Create default visualization tools
    fn create_default_visualization_tools() -> Vec<VisualizationTool> {
        vec![
            VisualizationTool {
                id: "chart_generator".to_string(),
                supported_charts: vec![
                    ChartType::Line,
                    ChartType::Bar,
                    ChartType::Scatter,
                    ChartType::Histogram,
                ],
                rendering_engine: RenderingEngine {
                    engine_type: RenderingEngineType::SVG,
                    config: HashMap::new(),
                    performance_settings: HashMap::new(),
                },
                interactive_features: vec![
                    InteractiveFeature::Zoom,
                    InteractiveFeature::Tooltips,
                    InteractiveFeature::Export,
                ],
            },
            VisualizationTool {
                id: "heatmap_generator".to_string(),
                supported_charts: vec![ChartType::Heatmap],
                rendering_engine: RenderingEngine {
                    engine_type: RenderingEngineType::Canvas,
                    config: HashMap::new(),
                    performance_settings: HashMap::new(),
                },
                interactive_features: vec![
                    InteractiveFeature::Tooltips,
                    InteractiveFeature::DrillDown,
                ],
            },
        ]
    }

    /// Process test results
    pub fn process_test_results(&mut self, results: &TestSuiteResults) -> ApplicationResult<()> {
        println!("Processing test results for analytics");

        // Process scenario results
        for scenario_result in &results.scenario_results {
            self.process_scenario_result(scenario_result)?;
        }

        // Process regression results
        for regression_result in &results.regression_results {
            self.process_regression_result(regression_result)?;
        }

        // Process platform results
        for platform_result in &results.platform_results {
            self.process_platform_result(platform_result)?;
        }

        // Process stress results
        for stress_result in &results.stress_results {
            self.process_stress_result(stress_result)?;
        }

        // Process property results
        for property_result in &results.property_results {
            self.process_property_result(property_result)?;
        }

        // Update database statistics
        self.update_database_statistics()?;

        // Analyze trends and patterns
        self.analyze_trends_and_patterns()?;

        Ok(())
    }

    /// Process scenario test result
    fn process_scenario_result(&mut self, result: &ScenarioTestResult) -> ApplicationResult<()> {
        let record = TestExecutionRecord {
            id: format!(
                "scenario_{}_{}",
                result.scenario_id,
                Instant::now().elapsed().as_nanos()
            ),
            test_id: result.scenario_id.clone(),
            timestamp: Instant::now(),
            result: result.test_result.clone(),
            config: HashMap::new(),
            environment: HashMap::new(),
        };

        self.result_database
            .execution_records
            .entry(result.scenario_id.clone())
            .or_insert_with(Vec::new)
            .push(record);

        Ok(())
    }

    /// Process regression test result
    fn process_regression_result(
        &mut self,
        result: &RegressionTestResult,
    ) -> ApplicationResult<()> {
        if result.regression_detected {
            // Record as potential failure pattern
            let failure = FailureInstance {
                timestamp: Instant::now(),
                test_id: result.test_id.clone(),
                details: TestError {
                    error_type: TestErrorType::RuntimeError,
                    message: "Performance regression detected".to_string(),
                    code: None,
                    location: None,
                    stack_trace: None,
                },
                context: {
                    let mut context = HashMap::new();
                    context.insert("confidence".to_string(), result.confidence.to_string());
                    context.insert("p_value".to_string(), result.p_value.to_string());
                    context
                },
            };

            let pattern_id = format!("regression_{}", result.test_id);
            let pattern = self
                .result_database
                .failure_patterns
                .entry(pattern_id.clone())
                .or_insert_with(|| FailurePattern {
                    id: pattern_id,
                    pattern_type: FailurePatternType::Temporal,
                    frequency: 0.0,
                    conditions: Vec::new(),
                    failures: Vec::new(),
                });

            pattern.failures.push(failure);
            pattern.frequency = pattern.failures.len() as f64;
        }

        Ok(())
    }

    /// Process platform test result
    fn process_platform_result(&mut self, result: &PlatformTestResult) -> ApplicationResult<()> {
        // Store platform compatibility data
        let record_id = format!(
            "platform_{}_{}",
            result.platform_id,
            Instant::now().elapsed().as_nanos()
        );

        // Create a synthetic execution result for platform test
        let execution_result = TestExecutionResult {
            solution_quality: result.compatibility_score,
            execution_time: Duration::from_secs(1),
            final_energy: -result.compatibility_score,
            best_solution: vec![1],
            convergence_achieved: result.compatibility_score > 0.9,
            memory_used: 1024,
        };

        let record = TestExecutionRecord {
            id: record_id,
            test_id: result.platform_id.clone(),
            timestamp: Instant::now(),
            result: execution_result,
            config: HashMap::new(),
            environment: {
                let mut env = HashMap::new();
                env.insert("test_type".to_string(), "platform_validation".to_string());
                env.insert(
                    "compatibility_score".to_string(),
                    result.compatibility_score.to_string(),
                );
                env
            },
        };

        self.result_database
            .execution_records
            .entry(result.platform_id.clone())
            .or_insert_with(Vec::new)
            .push(record);

        Ok(())
    }

    /// Process stress test result
    fn process_stress_result(&mut self, result: &StressTestResult) -> ApplicationResult<()> {
        let record_id = format!(
            "stress_{}_{}",
            result.test_id,
            Instant::now().elapsed().as_nanos()
        );

        let execution_result = TestExecutionResult {
            solution_quality: result.success_rate,
            execution_time: Duration::from_secs(60), // Simplified
            final_energy: -result.success_rate,
            best_solution: vec![1],
            convergence_achieved: result.success_rate > 0.9,
            memory_used: 2048,
        };

        let record = TestExecutionRecord {
            id: record_id,
            test_id: result.test_id.clone(),
            timestamp: Instant::now(),
            result: execution_result,
            config: HashMap::new(),
            environment: {
                let mut env = HashMap::new();
                env.insert("test_type".to_string(), "stress_test".to_string());
                env.insert("max_load".to_string(), result.max_load.to_string());
                env.insert("throughput".to_string(), result.throughput.to_string());
                env
            },
        };

        self.result_database
            .execution_records
            .entry(result.test_id.clone())
            .or_insert_with(Vec::new)
            .push(record);

        Ok(())
    }

    /// Process property test result
    fn process_property_result(&mut self, result: &PropertyTestResult) -> ApplicationResult<()> {
        let record_id = format!(
            "property_{}_{}",
            result.property_id,
            Instant::now().elapsed().as_nanos()
        );

        let execution_result = TestExecutionResult {
            solution_quality: result.confidence,
            execution_time: result.execution_time,
            final_energy: -result.confidence,
            best_solution: vec![1],
            convergence_achieved: result.confidence > 0.95,
            memory_used: 512,
        };

        let record = TestExecutionRecord {
            id: record_id,
            test_id: result.property_id.clone(),
            timestamp: Instant::now(),
            result: execution_result,
            config: HashMap::new(),
            environment: {
                let mut env = HashMap::new();
                env.insert("test_type".to_string(), "property_test".to_string());
                env.insert("cases_tested".to_string(), result.cases_tested.to_string());
                env.insert("cases_passed".to_string(), result.cases_passed.to_string());
                env
            },
        };

        self.result_database
            .execution_records
            .entry(result.property_id.clone())
            .or_insert_with(Vec::new)
            .push(record);

        Ok(())
    }

    /// Update database statistics
    fn update_database_statistics(&mut self) -> ApplicationResult<()> {
        let mut total_executions = 0;
        let mut successful_executions = 0;
        let mut total_time = Duration::default();

        for records in self.result_database.execution_records.values() {
            for record in records {
                total_executions += 1;
                total_time += record.result.execution_time;

                if record.result.convergence_achieved {
                    successful_executions += 1;
                }
            }
        }

        self.result_database.statistics.total_executions = total_executions;
        self.result_database.statistics.success_rate = if total_executions > 0 {
            successful_executions as f64 / total_executions as f64
        } else {
            0.0
        };

        self.result_database.statistics.avg_execution_time = if total_executions > 0 {
            total_time / total_executions as u32
        } else {
            Duration::default()
        };

        Ok(())
    }

    /// Analyze trends and patterns
    fn analyze_trends_and_patterns(&mut self) -> ApplicationResult<()> {
        // Analyze performance trends for each test type
        for (test_id, records) in &self.result_database.execution_records {
            if records.len() >= 5 {
                // Need minimum data points
                let trend = self.calculate_performance_trend(test_id, records)?;
                self.result_database
                    .performance_trends
                    .insert(test_id.clone(), trend);
            }
        }

        // Analyze failure patterns
        self.analyze_failure_patterns()?;

        Ok(())
    }

    /// Calculate performance trend
    fn calculate_performance_trend(
        &self,
        test_id: &str,
        records: &[TestExecutionRecord],
    ) -> ApplicationResult<PerformanceTrend> {
        let mut data_points = VecDeque::new();

        // Extract quality data points
        for record in records.iter().rev().take(50) {
            // Last 50 records
            data_points.push_back((record.timestamp, record.result.solution_quality));
        }

        // Simple trend analysis
        let values: Vec<f64> = data_points.iter().map(|(_, v)| *v).collect();
        let n = values.len() as f64;

        if n < 2.0 {
            return Ok(PerformanceTrend {
                metric: "solution_quality".to_string(),
                trend_direction: TrendDirection::Stable,
                trend_magnitude: 0.0,
                confidence: 0.0,
                data_points,
            });
        }

        // Calculate linear trend
        let x_sum = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let y_sum = values.iter().sum::<f64>();
        let xy_sum = values
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let x2_sum = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2));

        let trend_direction = if slope > 0.01 {
            TrendDirection::Improving
        } else if slope < -0.01 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        Ok(PerformanceTrend {
            metric: "solution_quality".to_string(),
            trend_direction,
            trend_magnitude: slope.abs(),
            confidence: 0.8, // Simplified
            data_points,
        })
    }

    /// Analyze failure patterns
    fn analyze_failure_patterns(&mut self) -> ApplicationResult<()> {
        // Update frequency for existing patterns
        for pattern in self.result_database.failure_patterns.values_mut() {
            // Calculate recent frequency (last 30 days)
            let cutoff = Instant::now() - Duration::from_secs(30 * 24 * 3600);
            let recent_failures = pattern
                .failures
                .iter()
                .filter(|f| f.timestamp > cutoff)
                .count();

            pattern.frequency = recent_failures as f64;
        }

        Ok(())
    }

    /// Generate reports
    pub fn generate_reports(&mut self) -> ApplicationResult<()> {
        println!("Generating test reports");

        for generator in &self.report_generators {
            let report = self.generate_report_with_generator(generator)?;
            println!(
                "Generated {:?} report: {} bytes",
                generator.report_type,
                report.len()
            );
        }

        Ok(())
    }

    /// Generate report with specific generator
    fn generate_report_with_generator(
        &self,
        generator: &ReportGenerator,
    ) -> ApplicationResult<String> {
        match generator.report_type {
            ReportType::PerformanceSummary => self.generate_performance_summary(),
            ReportType::FailureAnalysis => self.generate_failure_analysis(),
            ReportType::TrendAnalysis => self.generate_trend_analysis(),
            _ => Ok("Report type not implemented".to_string()),
        }
    }

    /// Generate performance summary report
    fn generate_performance_summary(&self) -> ApplicationResult<String> {
        let mut report = String::new();
        report.push_str("# Performance Summary Report\n\n");

        // Overall statistics
        report.push_str(&format!(
            "## Overall Statistics\n- Total Executions: {}\n- Success Rate: {:.2}%\n- Average Execution Time: {:?}\n\n",
            self.result_database.statistics.total_executions,
            self.result_database.statistics.success_rate * 100.0,
            self.result_database.statistics.avg_execution_time
        ));

        // Performance by test type
        report.push_str("## Performance by Test Type\n");
        for (test_id, records) in &self.result_database.execution_records {
            if !records.is_empty() {
                let avg_quality = records
                    .iter()
                    .map(|r| r.result.solution_quality)
                    .sum::<f64>()
                    / records.len() as f64;

                report.push_str(&format!(
                    "- {}: {:.3} average quality ({} executions)\n",
                    test_id,
                    avg_quality,
                    records.len()
                ));
            }
        }

        report.push_str("\n");

        // Trend analysis
        if !self.result_database.performance_trends.is_empty() {
            report.push_str("## Performance Trends\n");
            for (test_id, trend) in &self.result_database.performance_trends {
                report.push_str(&format!(
                    "- {}: {:?} trend (magnitude: {:.4})\n",
                    test_id, trend.trend_direction, trend.trend_magnitude
                ));
            }
        }

        Ok(report)
    }

    /// Generate failure analysis report
    fn generate_failure_analysis(&self) -> ApplicationResult<String> {
        let mut report = String::new();
        report.push_str("# Failure Analysis Report\n\n");

        if self.result_database.failure_patterns.is_empty() {
            report.push_str("No failure patterns detected.\n");
            return Ok(report);
        }

        report.push_str("## Detected Failure Patterns\n");
        for pattern in self.result_database.failure_patterns.values() {
            report.push_str(&format!(
                "### Pattern: {}\n- Type: {:?}\n- Frequency: {:.1}\n- Failures: {}\n\n",
                pattern.id,
                pattern.pattern_type,
                pattern.frequency,
                pattern.failures.len()
            ));
        }

        Ok(report)
    }

    /// Generate trend analysis report
    fn generate_trend_analysis(&self) -> ApplicationResult<String> {
        let mut report = String::new();
        report.push_str("# Trend Analysis Report\n\n");

        if self.result_database.performance_trends.is_empty() {
            report.push_str("No trends detected.\n");
            return Ok(report);
        }

        for (test_id, trend) in &self.result_database.performance_trends {
            report.push_str(&format!(
                "## {}\n- Metric: {}\n- Direction: {:?}\n- Magnitude: {:.4}\n- Confidence: {:.2}\n- Data Points: {}\n\n",
                test_id,
                trend.metric,
                trend.trend_direction,
                trend.trend_magnitude,
                trend.confidence,
                trend.data_points.len()
            ));
        }

        Ok(report)
    }

    /// Get analytics summary
    pub fn get_analytics_summary(&self) -> AnalyticsSummary {
        AnalyticsSummary {
            total_tests: self.result_database.statistics.total_executions,
            success_rate: self.result_database.statistics.success_rate,
            avg_execution_time: self.result_database.statistics.avg_execution_time,
            active_trends: self.result_database.performance_trends.len(),
            detected_patterns: self.result_database.failure_patterns.len(),
            data_retention_days: self
                .result_database
                .statistics
                .retention_policy
                .retention_period
                .as_secs()
                / (24 * 3600),
        }
    }
}

/// Analytics summary information
#[derive(Debug, Clone)]
pub struct AnalyticsSummary {
    /// Total number of tests executed
    pub total_tests: usize,
    /// Overall success rate
    pub success_rate: f64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Number of active performance trends
    pub active_trends: usize,
    /// Number of detected failure patterns
    pub detected_patterns: usize,
    /// Data retention period in days
    pub data_retention_days: u64,
}
