//! Test result types and result storage management

use std::collections::{BTreeMap, HashMap};
use std::fs;
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    #[must_use]
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
    #[must_use]
    pub fn get_result(&self, execution_id: &str) -> Option<&super::execution::TestExecutionResult> {
        self.result_cache.get(execution_id)
    }

    /// Get results by time range
    #[must_use]
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
    #[must_use]
    pub fn get_recent_results(&self, count: usize) -> Vec<&super::execution::TestExecutionResult> {
        self.result_index
            .iter()
            .rev()
            .take(count)
            .filter_map(|(_, id)| self.result_cache.get(id))
            .collect()
    }

    /// Get all results for a specific test case
    #[must_use]
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
    #[must_use]
    pub const fn get_statistics(&self) -> &StorageStatistics {
        &self.storage_stats
    }

    /// Check if cleanup is needed
    const fn should_cleanup(&self) -> bool {
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

    /// Export all currently cached results to a JSON file at `file_path`.
    ///
    /// Each cached [`super::execution::TestExecutionResult`] is serialized as
    /// a JSON object capturing its execution id, test case id, status,
    /// start/end times, outcome, performance metrics, validation summary and
    /// metadata. Fields that are not persisted here (individual validations,
    /// error details, artifacts) are honestly omitted rather than fabricated;
    /// [`Self::import_results`] restores them as empty/`None`.
    pub fn export_results(&self, file_path: &str) -> Result<(), String> {
        let mut entries = Vec::with_capacity(self.result_cache.len());
        for result in self.result_cache.values() {
            entries.push(execution_result_to_json(result));
        }
        let document = serde_json::Value::Array(entries);
        let text = serde_json::to_string_pretty(&document)
            .map_err(|e| format!("failed to serialize results: {e}"))?;
        fs::write(file_path, text)
            .map_err(|e| format!("failed to write results to '{file_path}': {e}"))
    }

    /// Import results previously written by [`Self::export_results`], merging
    /// them into this storage's cache/index and returning the number of
    /// records actually imported.
    pub fn import_results(&mut self, file_path: &str) -> Result<usize, String> {
        let text = fs::read_to_string(file_path)
            .map_err(|e| format!("failed to read results from '{file_path}': {e}"))?;
        let document: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("failed to parse results file '{file_path}': {e}"))?;
        let array = document
            .as_array()
            .ok_or_else(|| format!("results file '{file_path}' is not a JSON array"))?;

        let mut imported = 0usize;
        for entry in array {
            let result = execution_result_from_json(entry)
                .map_err(|e| format!("malformed result entry in '{file_path}': {e}"))?;
            self.store_result(result)?;
            imported += 1;
        }
        Ok(imported)
    }
}

/// Serialize a [`super::execution::TestExecutionResult`] into a self-describing
/// JSON value. Used by [`TestResultStorage::export_results`].
fn execution_result_to_json(result: &super::execution::TestExecutionResult) -> serde_json::Value {
    use super::execution::ExecutionStatus;

    let status = match &result.status {
        ExecutionStatus::Success => serde_json::json!({"kind": "Success"}),
        ExecutionStatus::Failure(msg) => serde_json::json!({"kind": "Failure", "message": msg}),
        ExecutionStatus::Timeout => serde_json::json!({"kind": "Timeout"}),
        ExecutionStatus::Cancelled => serde_json::json!({"kind": "Cancelled"}),
        ExecutionStatus::Error(msg) => serde_json::json!({"kind": "Error", "message": msg}),
    };

    let outcome = match result.result.outcome {
        TestOutcome::Passed => "Passed",
        TestOutcome::Failed => "Failed",
        TestOutcome::Skipped => "Skipped",
        TestOutcome::Timeout => "Timeout",
        TestOutcome::Error => "Error",
    };

    let validation_status = match result.result.validation_results.status {
        ValidationStatus::Passed => "Passed",
        ValidationStatus::Failed => "Failed",
        ValidationStatus::Partial => "Partial",
        ValidationStatus::NotExecuted => "NotExecuted",
    };

    serde_json::json!({
        "execution_id": result.execution_id,
        "test_case_id": result.test_case_id,
        "status": status,
        "start_time_secs": system_time_to_secs(result.start_time),
        "end_time_secs": system_time_to_secs(result.end_time),
        "outcome": outcome,
        "performance_metrics": {
            "execution_duration_secs": result.result.performance_metrics.execution_duration.as_secs_f64(),
            "setup_duration_secs": result.result.performance_metrics.setup_duration.as_secs_f64(),
            "cleanup_duration_secs": result.result.performance_metrics.cleanup_duration.as_secs_f64(),
            "peak_memory_usage": result.result.performance_metrics.peak_memory_usage,
            "avg_cpu_usage": result.result.performance_metrics.avg_cpu_usage,
            "custom_metrics": result.result.performance_metrics.custom_metrics,
        },
        "validation_status": validation_status,
        "validation_summary": {
            "total": result.result.validation_results.summary.total,
            "passed": result.result.validation_results.summary.passed,
            "failed": result.result.validation_results.summary.failed,
            "skipped": result.result.validation_results.summary.skipped,
        },
        "metadata": result.metadata,
    })
}

/// Reconstruct a [`super::execution::TestExecutionResult`] from a JSON value
/// produced by [`execution_result_to_json`]. Fields not captured by the
/// export format (individual validations, error details, artifacts) are
/// honestly restored empty rather than fabricated.
fn execution_result_from_json(
    value: &serde_json::Value,
) -> Result<super::execution::TestExecutionResult, String> {
    use super::execution::{ExecutionStatus, TestExecutionResult};

    let get_str = |key: &str| -> Result<String, String> {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(std::string::ToString::to_string)
            .ok_or_else(|| format!("missing or invalid field '{key}'"))
    };
    let get_u64 = |key: &str| -> Result<u64, String> {
        value
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("missing or invalid field '{key}'"))
    };

    let execution_id = get_str("execution_id")?;
    let test_case_id = get_str("test_case_id")?;
    let start_time = secs_to_system_time(get_u64("start_time_secs")?);
    let end_time = secs_to_system_time(get_u64("end_time_secs")?);

    let status_value = value
        .get("status")
        .ok_or_else(|| "missing field 'status'".to_string())?;
    let status_kind = status_value
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "missing field 'status.kind'".to_string())?;
    let status = match status_kind {
        "Success" => ExecutionStatus::Success,
        "Timeout" => ExecutionStatus::Timeout,
        "Cancelled" => ExecutionStatus::Cancelled,
        "Failure" => ExecutionStatus::Failure(
            status_value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        "Error" => ExecutionStatus::Error(
            status_value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        other => return Err(format!("unknown status kind '{other}'")),
    };

    let outcome = match get_str("outcome")?.as_str() {
        "Passed" => TestOutcome::Passed,
        "Failed" => TestOutcome::Failed,
        "Skipped" => TestOutcome::Skipped,
        "Timeout" => TestOutcome::Timeout,
        "Error" => TestOutcome::Error,
        other => return Err(format!("unknown outcome '{other}'")),
    };

    let validation_status = match get_str("validation_status")?.as_str() {
        "Passed" => ValidationStatus::Passed,
        "Failed" => ValidationStatus::Failed,
        "Partial" => ValidationStatus::Partial,
        "NotExecuted" => ValidationStatus::NotExecuted,
        other => return Err(format!("unknown validation status '{other}'")),
    };

    let pm = value
        .get("performance_metrics")
        .ok_or_else(|| "missing field 'performance_metrics'".to_string())?;
    let f64_field = |key: &str| -> Result<f64, String> {
        pm.get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("missing or invalid field 'performance_metrics.{key}'"))
    };
    let performance_metrics = PerformanceMetrics {
        execution_duration: Duration::from_secs_f64(f64_field("execution_duration_secs")?),
        setup_duration: Duration::from_secs_f64(f64_field("setup_duration_secs")?),
        cleanup_duration: Duration::from_secs_f64(f64_field("cleanup_duration_secs")?),
        peak_memory_usage: pm
            .get("peak_memory_usage")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                "missing or invalid field 'performance_metrics.peak_memory_usage'".to_string()
            })? as usize,
        avg_cpu_usage: f64_field("avg_cpu_usage")?,
        custom_metrics: pm
            .get("custom_metrics")
            .and_then(|v| serde_json::from_value::<HashMap<String, f64>>(v.clone()).ok())
            .unwrap_or_default(),
    };

    let vs = value
        .get("validation_summary")
        .ok_or_else(|| "missing field 'validation_summary'".to_string())?;
    let usize_field = |key: &str| -> Result<usize, String> {
        vs.get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .ok_or_else(|| format!("missing or invalid field 'validation_summary.{key}'"))
    };
    let validation_summary = ValidationSummary {
        total: usize_field("total")?,
        passed: usize_field("passed")?,
        failed: usize_field("failed")?,
        skipped: usize_field("skipped")?,
    };

    let metadata = value
        .get("metadata")
        .and_then(|v| serde_json::from_value::<HashMap<String, String>>(v.clone()).ok())
        .unwrap_or_default();

    let integration_result = IntegrationTestResult {
        test_case_id: test_case_id.clone(),
        timestamp: end_time,
        outcome,
        performance_metrics,
        validation_results: ValidationResults {
            status: validation_status,
            validations: Vec::new(),
            summary: validation_summary,
        },
        error_info: None,
        artifacts: Vec::new(),
    };

    Ok(TestExecutionResult {
        execution_id,
        test_case_id,
        status,
        start_time,
        end_time,
        result: integration_result,
        metadata,
    })
}

fn system_time_to_secs(time: SystemTime) -> u64 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn secs_to_system_time(secs: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
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

#[cfg(test)]
mod tests {
    use super::super::execution::{ExecutionStatus, TestExecutionResult};
    use super::*;

    fn make_execution_result(id: &str, outcome: TestOutcome) -> TestExecutionResult {
        let now = SystemTime::now();
        let mut metadata = HashMap::new();
        metadata.insert("suite".to_string(), "regression".to_string());

        TestExecutionResult {
            execution_id: id.to_string(),
            test_case_id: format!("{id}_case"),
            status: ExecutionStatus::Success,
            start_time: now,
            end_time: now + Duration::from_secs(3),
            result: IntegrationTestResult {
                test_case_id: format!("{id}_case"),
                timestamp: now,
                outcome,
                performance_metrics: PerformanceMetrics {
                    execution_duration: Duration::from_secs(3),
                    setup_duration: Duration::from_millis(200),
                    cleanup_duration: Duration::from_millis(50),
                    peak_memory_usage: 2048,
                    avg_cpu_usage: 0.42,
                    custom_metrics: HashMap::new(),
                },
                validation_results: ValidationResults {
                    status: ValidationStatus::Passed,
                    validations: vec![],
                    summary: ValidationSummary {
                        total: 2,
                        passed: 2,
                        failed: 0,
                        skipped: 0,
                    },
                },
                error_info: None,
                artifacts: vec![],
            },
            metadata,
        }
    }

    #[test]
    fn export_and_import_results_round_trip_real_data() {
        let mut storage = TestResultStorage::new(TestStorageConfig::default());
        storage
            .store_result(make_execution_result("exec_a", TestOutcome::Passed))
            .expect("store should succeed");
        storage
            .store_result(make_execution_result("exec_b", TestOutcome::Failed))
            .expect("store should succeed");

        let mut path = std::env::temp_dir();
        path.push(format!(
            "quantrs2_anneal_results_export_test_{}.json",
            std::process::id()
        ));
        let path_str = path.to_str().expect("path should be valid UTF-8");

        storage
            .export_results(path_str)
            .expect("export should succeed");

        let written = std::fs::read_to_string(&path).expect("exported file should exist");
        // The file must contain the real outcome/id data, not a fabricated placeholder.
        assert!(written.contains("exec_a"));
        assert!(written.contains("exec_b"));
        assert!(written.contains("Failed"));

        let mut restored = TestResultStorage::new(TestStorageConfig::default());
        let imported = restored
            .import_results(path_str)
            .expect("import should succeed");
        assert_eq!(imported, 2);
        assert_eq!(restored.result_cache.len(), 2);

        let restored_b = restored
            .get_result("exec_b")
            .expect("exec_b should have been restored");
        assert_eq!(restored_b.result.outcome, TestOutcome::Failed);
        assert_eq!(
            restored_b.metadata.get("suite").map(String::as_str),
            Some("regression")
        );

        std::fs::remove_file(&path).expect("cleanup should succeed");
    }

    #[test]
    fn import_results_errors_on_missing_file() {
        let mut storage = TestResultStorage::new(TestStorageConfig::default());
        let mut path = std::env::temp_dir();
        path.push(format!(
            "quantrs2_anneal_results_missing_{}.json",
            std::process::id()
        ));
        let result = storage.import_results(path.to_str().unwrap());
        assert!(result.is_err());
    }
}
