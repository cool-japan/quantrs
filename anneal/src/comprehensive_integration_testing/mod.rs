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
pub mod scenarios;
pub mod execution;
pub mod monitoring;
pub mod reporting;
pub mod results;
pub mod framework;
pub mod validation;

// Re-export commonly used types
pub use config::{
    IntegrationTestConfig, TestStorageConfig, BenchmarkConfig, StressTestConfig,
    FaultInjectionConfig, MonitoringConfig, TestEnvironmentConfig,
    StorageFormat, BenchmarkSuite, StatisticalTest, StressScenario, FaultType,
    MonitoredMetric, AlertChannel, ReportFormat,
};

pub use framework::ComprehensiveIntegrationTesting;
pub use scenarios::{
    IntegrationTestCase, TestSuite, TestCategory, TestPriority, TestMetadata, TestRegistry,
};
pub use execution::{TestExecutionEngine, TestExecutionResult, ExecutionStatus};
pub use results::{
    IntegrationTestResult, IntegrationValidationResult, ComponentIntegrationResults,
    SystemIntegrationResults, ValidationStatus,
};

// Additional types that may be referenced in lib.rs
pub use scenarios::{ExpectedResults as ExpectedOutcomes};
pub use execution::{TestExecutionRequest as TestExecutionSpec};
pub use results::{PerformanceMetrics as PerformanceTestResult};
pub use config::{TestEnvironmentConfig as EnvironmentRequirements};
pub use results::{ValidationResults as StressTestResult};

// Placeholder for create_example_integration_testing function
pub fn create_example_integration_testing() -> ComprehensiveIntegrationTesting {
    ComprehensiveIntegrationTesting::new(IntegrationTestConfig::default())
}

// Import types from parent modules
use crate::applications::{ApplicationError, ApplicationResult};
use std::time::Duration;