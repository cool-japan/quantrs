//! Configuration types for the testing framework.
//!
//! This module defines configuration structures for tests, samplers,
//! validation, and output settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Main test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Random seed
    pub seed: Option<u64>,
    /// Number of test cases per category
    pub cases_per_category: usize,
    /// Problem sizes to test
    pub problem_sizes: Vec<usize>,
    /// Samplers to test
    pub samplers: Vec<SamplerConfig>,
    /// Timeout per test
    pub timeout: Duration,
    /// Validation settings
    pub validation: ValidationConfig,
    /// Output settings
    pub output: OutputConfig,
}

/// Sampler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// Sampler name
    pub name: String,
    /// Number of samples
    pub num_samples: usize,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Check constraint satisfaction
    pub check_constraints: bool,
    /// Check objective improvement
    pub check_objective: bool,
    /// Statistical validation
    pub statistical_tests: bool,
    /// Tolerance for floating point comparisons
    pub tolerance: f64,
    /// Minimum solution quality
    pub min_quality: f64,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Generate report
    pub generate_report: bool,
    /// Report format
    pub format: ReportFormat,
    /// Output directory
    pub output_dir: String,
    /// Verbosity level
    pub verbosity: VerbosityLevel,
}

/// Report format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Plain text
    Text,
    /// JSON
    Json,
    /// HTML
    Html,
    /// Markdown
    Markdown,
    /// CSV
    Csv,
}

/// Verbosity level for output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerbosityLevel {
    /// Only errors
    Error,
    /// Warnings and errors
    Warning,
    /// Info messages
    Info,
    /// Debug information
    Debug,
}
