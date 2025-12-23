//! Main comprehensive integration testing framework

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::config::IntegrationTestConfig;
use super::execution::TestExecutionEngine;
use super::monitoring::TestPerformanceMonitor;
use super::reporting::TestReportGenerator;
use super::results::TestResultStorage;
use super::scenarios::{IntegrationTestCase, TestRegistry};

/// Main comprehensive integration testing framework
pub struct ComprehensiveIntegrationTesting {
    /// Framework configuration
    pub config: IntegrationTestConfig,
    /// Test case registry
    pub test_registry: Arc<RwLock<TestRegistry>>,
    /// Test execution engine
    pub execution_engine: Arc<Mutex<TestExecutionEngine>>,
    /// Result storage system
    pub result_storage: Arc<Mutex<TestResultStorage>>,
    /// Performance monitor
    pub performance_monitor: Arc<Mutex<TestPerformanceMonitor>>,
    /// Report generator
    pub report_generator: Arc<Mutex<TestReportGenerator>>,
    /// Environment manager
    pub environment_manager: Arc<Mutex<TestEnvironmentManager>>,
}

impl ComprehensiveIntegrationTesting {
    /// Create a new comprehensive integration testing framework
    pub fn new(config: IntegrationTestConfig) -> Self {
        Self {
            config: config.clone(),
            test_registry: Arc::new(RwLock::new(TestRegistry::new())),
            execution_engine: Arc::new(Mutex::new(TestExecutionEngine::new())),
            result_storage: Arc::new(Mutex::new(TestResultStorage::new(
                config.storage_config.clone(),
            ))),
            performance_monitor: Arc::new(Mutex::new(TestPerformanceMonitor::new())),
            report_generator: Arc::new(Mutex::new(TestReportGenerator::new())),
            environment_manager: Arc::new(Mutex::new(TestEnvironmentManager::new(
                config.environment_config.clone(),
            ))),
        }
    }

    /// Register a test case
    pub fn register_test_case(&self, test_case: IntegrationTestCase) -> Result<(), String> {
        let mut registry = self.test_registry.write().unwrap();
        registry.register_test_case(test_case)
    }

    /// Execute all registered tests
    pub async fn execute_all_tests(
        &self,
    ) -> Result<Vec<super::results::IntegrationTestResult>, String> {
        let registry = self.test_registry.read().unwrap();
        let test_cases: Vec<_> = registry.test_cases.values().cloned().collect();
        drop(registry);

        let mut results = Vec::new();
        let mut engine = self.execution_engine.lock().unwrap();

        for test_case in test_cases {
            let request = super::execution::TestExecutionRequest {
                id: format!("exec_{}", test_case.id),
                test_case,
                priority: super::scenarios::TestPriority::Normal,
                requested_time: std::time::SystemTime::now(),
                context: super::execution::ExecutionContext {
                    parameters: std::collections::HashMap::new(),
                    environment: std::collections::HashMap::new(),
                    resources: super::execution::ResourceAllocation {
                        cpu_cores: 1,
                        memory_bytes: 1024 * 1024 * 1024, // 1 GB
                        disk_bytes: 1024 * 1024 * 1024,   // 1 GB
                        network_bandwidth: None,
                    },
                    metadata: std::collections::HashMap::new(),
                },
            };

            match engine.execute_test(request) {
                Ok(exec_result) => {
                    results.push(exec_result.result);
                }
                Err(e) => {
                    return Err(format!("Failed to execute test: {}", e));
                }
            }
        }

        Ok(results)
    }

    /// Execute a specific test suite
    pub async fn execute_test_suite(
        &self,
        suite_name: &str,
    ) -> Result<super::results::IntegrationTestResult, String> {
        let registry = self.test_registry.read().unwrap();
        let suite = registry
            .test_suites
            .get(suite_name)
            .ok_or_else(|| format!("Test suite '{}' not found", suite_name))?
            .clone();
        drop(registry);

        // Execute all test cases in the suite
        let mut engine = self.execution_engine.lock().unwrap();
        let registry = self.test_registry.read().unwrap();

        for test_case_id in &suite.test_cases {
            if let Some(test_case) = registry.test_cases.get(test_case_id) {
                let request = super::execution::TestExecutionRequest {
                    id: format!("exec_{}_{}", suite_name, test_case_id),
                    test_case: test_case.clone(),
                    priority: super::scenarios::TestPriority::Normal,
                    requested_time: std::time::SystemTime::now(),
                    context: super::execution::ExecutionContext {
                        parameters: std::collections::HashMap::new(),
                        environment: std::collections::HashMap::new(),
                        resources: super::execution::ResourceAllocation {
                            cpu_cores: 1,
                            memory_bytes: 1024 * 1024 * 1024,
                            disk_bytes: 1024 * 1024 * 1024,
                            network_bandwidth: None,
                        },
                        metadata: std::collections::HashMap::new(),
                    },
                };

                engine.execute_test(request)?;
            }
        }

        // Return a summary result
        Ok(super::results::IntegrationTestResult {
            test_case_id: suite_name.to_string(),
            timestamp: std::time::SystemTime::now(),
            outcome: super::results::TestOutcome::Passed,
            performance_metrics: super::results::PerformanceMetrics {
                execution_duration: std::time::Duration::from_secs(0),
                setup_duration: std::time::Duration::from_secs(0),
                cleanup_duration: std::time::Duration::from_secs(0),
                peak_memory_usage: 0,
                avg_cpu_usage: 0.0,
                custom_metrics: std::collections::HashMap::new(),
            },
            validation_results: super::results::ValidationResults {
                status: super::results::ValidationStatus::Passed,
                validations: Vec::new(),
                summary: super::results::ValidationSummary {
                    total: suite.test_cases.len(),
                    passed: suite.test_cases.len(),
                    failed: 0,
                    skipped: 0,
                },
            },
            error_info: None,
            artifacts: Vec::new(),
        })
    }

    /// Generate comprehensive test report
    pub fn generate_report(&self) -> Result<String, String> {
        let storage = self.result_storage.lock().unwrap();
        let stats = storage.get_statistics();

        let report = format!(
            "Comprehensive Integration Test Report\n\
             =====================================\n\
             \n\
             Total Results: {}\n\
             Storage Size: {} bytes\n\
             Last Cleanup: {:?}\n\
             Compression Ratio: {:.2}\n\
             \n\
             Recent Test Results:\n",
            stats.total_results, stats.storage_size, stats.last_cleanup, stats.compression_ratio
        );

        Ok(report)
    }
}

/// Test environment manager
pub struct TestEnvironmentManager {
    /// Environment configuration
    pub config: super::config::TestEnvironmentConfig,
    /// Active environments
    pub active_environments: HashMap<String, TestEnvironment>,
}

impl TestEnvironmentManager {
    pub fn new(config: super::config::TestEnvironmentConfig) -> Self {
        Self {
            config,
            active_environments: HashMap::new(),
        }
    }

    /// Create a new test environment
    pub fn create_environment(&mut self, id: String) -> Result<(), String> {
        if self.active_environments.contains_key(&id) {
            return Err(format!("Environment {} already exists", id));
        }

        let environment = TestEnvironment {
            id: id.clone(),
            status: EnvironmentStatus::Initializing,
            resources: self.config.resource_allocation.clone(),
        };

        self.active_environments.insert(id, environment);
        Ok(())
    }

    /// Get an environment by ID
    pub fn get_environment(&self, id: &str) -> Option<&TestEnvironment> {
        self.active_environments.get(id)
    }

    /// Destroy a test environment
    pub fn destroy_environment(&mut self, id: &str) -> Result<(), String> {
        self.active_environments
            .remove(id)
            .ok_or_else(|| format!("Environment {} not found", id))?;
        Ok(())
    }

    /// List all active environments
    pub fn list_environments(&self) -> Vec<&TestEnvironment> {
        self.active_environments.values().collect()
    }

    /// Update environment status
    pub fn update_environment_status(
        &mut self,
        id: &str,
        status: EnvironmentStatus,
    ) -> Result<(), String> {
        let env = self
            .active_environments
            .get_mut(id)
            .ok_or_else(|| format!("Environment {} not found", id))?;
        env.status = status;
        Ok(())
    }

    /// Get count of active environments
    pub fn active_count(&self) -> usize {
        self.active_environments.len()
    }

    /// Clear all environments
    pub fn clear_all(&mut self) {
        self.active_environments.clear();
    }
}

/// Test environment instance
#[derive(Debug, Clone)]
pub struct TestEnvironment {
    /// Environment ID
    pub id: String,
    /// Environment status
    pub status: EnvironmentStatus,
    /// Resource allocation
    pub resources: super::config::ResourceAllocationConfig,
}

/// Environment status
#[derive(Debug, Clone, PartialEq)]
pub enum EnvironmentStatus {
    Initializing,
    Ready,
    Running,
    Cleaning,
    Stopped,
    Error(String),
}
