//! Test result types and result storage management

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, SystemTime};

use super::config::TestStorageConfig;

/// Integration test result
#[derive(Debug, Clone)]
pub struct IntegrationTestResult {
    /// Test case ID
    pub test_case_id: String,
    /// Execution timestamp
    pub timestamp: SystemTime,
    /// Test outcome
    pub outcome: TestOutcome,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Validation results
    pub validation_results: ValidationResults,
    /// Error information (if failed)
    pub error_info: Option<ErrorInfo>,
    /// Test artifacts
    pub artifacts: Vec<TestArtifact>,
}

/// Test outcome
#[derive(Debug, Clone, PartialEq)]
pub enum TestOutcome {
    Passed,
    Failed,
    Skipped,
    Timeout,
    Error,
}

/// Performance metrics for test execution
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Execution duration
    pub execution_duration: Duration,
    /// Setup duration
    pub setup_duration: Duration,
    /// Cleanup duration
    pub cleanup_duration: Duration,
    /// Memory usage peak
    pub peak_memory_usage: usize,
    /// CPU usage average
    pub avg_cpu_usage: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Overall validation status
    pub status: ValidationStatus,
    /// Individual validations
    pub validations: Vec<IndividualValidation>,
    /// Validation summary
    pub summary: ValidationSummary,
}

/// Validation status
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Passed,
    Failed,
    Partial,
    NotExecuted,
}

/// Individual validation result
#[derive(Debug, Clone)]
pub struct IndividualValidation {
    /// Validation name
    pub name: String,
    /// Validation status
    pub status: ValidationStatus,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// Validation summary
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total validations
    pub total: usize,
    /// Passed validations
    pub passed: usize,
    /// Failed validations
    pub failed: usize,
    /// Skipped validations
    pub skipped: usize,
}

/// Error information for failed tests
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// Error code
    pub error_code: String,
    /// Error message
    pub message: String,
    /// Error category
    pub category: ErrorCategory,
    /// Stack trace
    pub stack_trace: Option<String>,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Error categories
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    Setup,
    Execution,
    Validation,
    Cleanup,
    Infrastructure,
    Timeout,
    Resource,
    Configuration,
    Custom(String),
}

/// Test artifact
#[derive(Debug, Clone)]
pub struct TestArtifact {
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// Artifact path
    pub path: String,
    /// Artifact size
    pub size: usize,
    /// Artifact metadata
    pub metadata: HashMap<String, String>,
}

/// Artifact types
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactType {
    Log,
    Screenshot,
    Report,
    Data,
    Configuration,
    Custom(String),
}

/// Integration validation result
#[derive(Debug, Clone)]
pub struct IntegrationValidationResult {
    /// Component integration results
    pub component_results: ComponentIntegrationResults,
    /// System integration results
    pub system_results: SystemIntegrationResults,
    /// Performance integration results
    pub performance_results: PerformanceIntegrationResults,
    /// Overall validation status
    pub overall_status: ValidationStatus,
}

/// Component integration results
#[derive(Debug, Clone)]
pub struct ComponentIntegrationResults {
    /// Individual component results
    pub components: HashMap<String, ComponentResult>,
    /// Integration matrix
    pub integration_matrix: Vec<Vec<IntegrationStatus>>,
}

/// System integration results
#[derive(Debug, Clone)]
pub struct SystemIntegrationResults {
    /// End-to-end test results
    pub end_to_end_results: Vec<EndToEndResult>,
    /// System health metrics
    pub system_health: SystemHealthMetrics,
}

/// Performance integration results
#[derive(Debug, Clone)]
pub struct PerformanceIntegrationResults {
    /// Performance benchmarks
    pub benchmarks: HashMap<String, BenchmarkResult>,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Performance regressions
    pub regressions: Vec<PerformanceRegression>,
}

/// Component result
#[derive(Debug, Clone)]
pub struct ComponentResult {
    /// Component name
    pub name: String,
    /// Test status
    pub status: ValidationStatus,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Error details
    pub error_details: Option<ErrorInfo>,
}

/// Integration status between components
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationStatus {
    Compatible,
    Incompatible,
    Warning,
    NotTested,
}

/// End-to-end test result
#[derive(Debug, Clone)]
pub struct EndToEndResult {
    /// Test scenario name
    pub scenario: String,
    /// Test status
    pub status: ValidationStatus,
    /// Execution time
    pub execution_time: Duration,
    /// Steps executed
    pub steps: Vec<StepResult>,
}

/// Step result
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Step name
    pub name: String,
    /// Step status
    pub status: ValidationStatus,
    /// Step duration
    pub duration: Duration,
    /// Step output
    pub output: Option<String>,
}

/// System health metrics
#[derive(Debug, Clone)]
pub struct SystemHealthMetrics {
    /// Overall health score
    pub health_score: f64,
    /// Component health
    pub component_health: HashMap<String, f64>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization
    pub cpu: f64,
    /// Memory utilization
    pub memory: f64,
    /// Disk utilization
    pub disk: f64,
    /// Network utilization
    pub network: f64,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Benchmark score
    pub score: f64,
    /// Baseline comparison
    pub baseline_comparison: Option<f64>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Performance trends
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Execution time trend
    pub execution_time_trend: Vec<(SystemTime, Duration)>,
    /// Memory usage trend
    pub memory_trend: Vec<(SystemTime, usize)>,
    /// Success rate trend
    pub success_rate_trend: Vec<(SystemTime, f64)>,
}

/// Performance regression
#[derive(Debug, Clone)]
pub struct PerformanceRegression {
    /// Metric name
    pub metric: String,
    /// Previous value
    pub previous_value: f64,
    /// Current value
    pub current_value: f64,
    /// Regression percentage
    pub regression_percentage: f64,
    /// Severity
    pub severity: RegressionSeverity,
}

/// Regression severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Test result storage system
pub struct TestResultStorage {
    /// Storage configuration
    pub storage_config: TestStorageConfig,
    /// In-memory result cache
    pub result_cache: HashMap<String, super::execution::TestExecutionResult>,
    /// Result index
    pub result_index: BTreeMap<SystemTime, String>,
    /// Storage statistics
    pub storage_stats: StorageStatistics,
}

impl TestResultStorage {
    pub fn new(config: TestStorageConfig) -> Self {
        Self {
            storage_config: config,
            result_cache: HashMap::new(),
            result_index: BTreeMap::new(),
            storage_stats: StorageStatistics::default(),
        }
    }

    /// Store a test execution result
    pub fn store_result(
        &mut self,
        result: super::execution::TestExecutionResult,
    ) -> Result<(), String> {
        let id = result.execution_id.clone();
        let timestamp = result.start_time;

        // Update statistics
        self.storage_stats.total_results += 1;
        self.storage_stats.storage_size += std::mem::size_of_val(&result);

        // Store in cache
        self.result_cache.insert(id.clone(), result);

        // Update index
        self.result_index.insert(timestamp, id);

        // Check if cleanup is needed
        if self.should_cleanup() {
            self.cleanup_old_results();
        }

        Ok(())
    }

    /// Retrieve a test result by execution ID
    pub fn get_result(&self, execution_id: &str) -> Option<&super::execution::TestExecutionResult> {
        self.result_cache.get(execution_id)
    }

    /// Get results by time range
    pub fn get_results_by_time_range(
        &self,
        start: SystemTime,
        end: SystemTime,
    ) -> Vec<&super::execution::TestExecutionResult> {
        self.result_index
            .range(start..=end)
            .filter_map(|(_, id)| self.result_cache.get(id))
            .collect()
    }

    /// Get the most recent N results
    pub fn get_recent_results(&self, count: usize) -> Vec<&super::execution::TestExecutionResult> {
        self.result_index
            .iter()
            .rev()
            .take(count)
            .filter_map(|(_, id)| self.result_cache.get(id))
            .collect()
    }

    /// Get all results for a specific test case
    pub fn get_results_for_test_case(
        &self,
        test_case_id: &str,
    ) -> Vec<&super::execution::TestExecutionResult> {
        self.result_cache
            .values()
            .filter(|r| r.test_case_id == test_case_id)
            .collect()
    }

    /// Clear all stored results
    pub fn clear_all(&mut self) {
        self.result_cache.clear();
        self.result_index.clear();
        self.storage_stats = StorageStatistics::default();
    }

    /// Get storage statistics
    pub fn get_statistics(&self) -> &StorageStatistics {
        &self.storage_stats
    }

    /// Check if cleanup is needed
    fn should_cleanup(&self) -> bool {
        match &self.storage_config.retention_policy {
            super::config::RetentionPolicy::KeepLast(max_results) => {
                self.storage_stats.total_results > *max_results
            }
            super::config::RetentionPolicy::KeepForDuration(_) => {
                // Check if we should do time-based cleanup
                true
            }
            super::config::RetentionPolicy::KeepAll => false,
            super::config::RetentionPolicy::Custom(_) => false,
        }
    }

    /// Cleanup old results
    fn cleanup_old_results(&mut self) {
        match &self.storage_config.retention_policy {
            super::config::RetentionPolicy::KeepLast(max_results) => {
                // Remove oldest results if we exceed the limit
                while self.storage_stats.total_results > *max_results {
                    if let Some((time, id)) = self
                        .result_index
                        .iter()
                        .next()
                        .map(|(t, i)| (*t, i.clone()))
                    {
                        self.result_index.remove(&time);
                        if let Some(result) = self.result_cache.remove(&id) {
                            self.storage_stats.total_results =
                                self.storage_stats.total_results.saturating_sub(1);
                            self.storage_stats.storage_size = self
                                .storage_stats
                                .storage_size
                                .saturating_sub(std::mem::size_of_val(&result));
                        }
                    } else {
                        break;
                    }
                }
            }
            super::config::RetentionPolicy::KeepForDuration(duration) => {
                let cutoff_time = SystemTime::now()
                    .checked_sub(*duration)
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                // Remove old entries from index and cache
                let old_ids: Vec<String> = self
                    .result_index
                    .range(..cutoff_time)
                    .map(|(_, id)| id.clone())
                    .collect();

                for id in old_ids {
                    if let Some(result) = self.result_cache.remove(&id) {
                        self.result_index.remove(&result.start_time);
                        self.storage_stats.total_results =
                            self.storage_stats.total_results.saturating_sub(1);
                        self.storage_stats.storage_size = self
                            .storage_stats
                            .storage_size
                            .saturating_sub(std::mem::size_of_val(&result));
                    }
                }
            }
            _ => {}
        }

        self.storage_stats.last_cleanup = SystemTime::now();
    }

    /// Export results to a file (placeholder)
    pub fn export_results(&self, _file_path: &str) -> Result<(), String> {
        // Placeholder for file export functionality
        Ok(())
    }

    /// Import results from a file (placeholder)
    pub fn import_results(&mut self, _file_path: &str) -> Result<usize, String> {
        // Placeholder for file import functionality
        Ok(0)
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    /// Total stored results
    pub total_results: usize,
    /// Storage size in bytes
    pub storage_size: usize,
    /// Last cleanup time
    pub last_cleanup: SystemTime,
    /// Compression ratio
    pub compression_ratio: f64,
}

impl Default for StorageStatistics {
    fn default() -> Self {
        Self {
            total_results: 0,
            storage_size: 0,
            last_cleanup: SystemTime::now(),
            compression_ratio: 1.0,
        }
    }
}
