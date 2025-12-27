//! Comprehensive Integration Testing Framework for Quantum Annealing Systems
//!
//! This module implements a sophisticated integration testing framework that validates
//! the seamless interaction between all quantum annealing components including quantum
//! error correction, advanced algorithms, multi-chip systems, hybrid execution engines,
//! and scientific computing applications. It provides automated testing, performance
//! validation, stress testing, and comprehensive system verification.
//!
//! Key Features:
//! - Multi-level integration testing (unit, component, system, end-to-end)
//! - Automated test generation and execution
//! - Performance regression testing and benchmarking
//! - Stress testing and fault injection
//! - Cross-component interaction validation
//! - Scientific application workflow testing
//! - Real-time monitoring and reporting
//! - Test result analysis and optimization recommendations

pub mod config;
pub mod execution;
pub mod framework;
pub mod monitoring;
pub mod reporting;
pub mod results;
pub mod scenarios;
pub mod validation;

// Re-export commonly used types
pub use config::{
    AlertChannel, BenchmarkConfig, BenchmarkSuite, FaultInjectionConfig, FaultType,
    IntegrationTestConfig, MonitoredMetric, MonitoringConfig, ReportFormat, StatisticalTest,
    StorageFormat, StressScenario, StressTestConfig, TestEnvironmentConfig, TestStorageConfig,
};

pub use execution::{ExecutionStatus, TestExecutionEngine, TestExecutionResult};
pub use framework::ComprehensiveIntegrationTesting;
pub use results::{
    ComponentIntegrationResults, IntegrationTestResult, IntegrationValidationResult,
    SystemIntegrationResults, ValidationStatus,
};
pub use scenarios::{
    IntegrationTestCase, TestCategory, TestMetadata, TestPriority, TestRegistry, TestSuite,
};

// Additional types that may be referenced in lib.rs
pub use config::TestEnvironmentConfig as EnvironmentRequirements;
pub use execution::TestExecutionRequest as TestExecutionSpec;
pub use results::PerformanceMetrics as PerformanceTestResult;
pub use results::ValidationResults as StressTestResult;
pub use scenarios::ExpectedResults as ExpectedOutcomes;

// Placeholder for create_example_integration_testing function
#[must_use]
pub fn create_example_integration_testing() -> ComprehensiveIntegrationTesting {
    ComprehensiveIntegrationTesting::new(IntegrationTestConfig::default())
}

// Import types from parent modules
use crate::applications::{ApplicationError, ApplicationResult};
use std::time::Duration;
