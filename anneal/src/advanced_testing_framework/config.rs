//! Configuration types for advanced testing framework

use super::{Duration, HashMap};

/// Configuration for the testing framework
#[derive(Debug, Clone)]
pub struct TestingConfig {
    /// Enable parallel test execution
    pub enable_parallel: bool,
    /// Maximum concurrent tests
    pub max_concurrent_tests: usize,
    /// Test timeout duration
    pub test_timeout: Duration,
    /// Performance threshold tolerance
    pub performance_tolerance: f64,
    /// Statistical significance level
    pub significance_level: f64,
    /// Test data retention period
    pub data_retention: Duration,
    /// Enable detailed logging
    pub detailed_logging: bool,
    /// Stress test problem sizes
    pub stress_test_sizes: Vec<usize>,
}

impl Default for TestingConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            max_concurrent_tests: 8,
            test_timeout: Duration::from_secs(300),
            performance_tolerance: 0.05,
            significance_level: 0.05,
            data_retention: Duration::from_secs(30 * 24 * 3600),
            detailed_logging: true,
            stress_test_sizes: vec![100, 500, 1000, 2000, 5000],
        }
    }
}

/// Alert thresholds for regression detection
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Performance degradation threshold
    pub degradation_threshold: f64,
    /// Statistical significance level
    pub significance_level: f64,
    /// Minimum sample size for detection
    pub min_sample_size: usize,
    /// Alert cooldown period
    pub cooldown_period: Duration,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            degradation_threshold: 0.1,
            significance_level: 0.05,
            min_sample_size: 10,
            cooldown_period: Duration::from_secs(3600),
        }
    }
}

/// Data retention policy
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Retention period
    pub retention_period: Duration,
    /// Cleanup frequency
    pub cleanup_frequency: Duration,
    /// Archive policy
    pub archive_policy: Option<ArchivePolicy>,
}

/// Archive policy for old data
#[derive(Debug, Clone)]
pub struct ArchivePolicy {
    /// Archive location
    pub archive_location: String,
    /// Compression enabled
    pub compression: bool,
    /// Archive format
    pub format: ArchiveFormat,
}

/// Archive format options
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchiveFormat {
    /// JSON format
    JSON,
    /// Binary format
    Binary,
    /// Compressed format
    Compressed,
    /// Custom format
    Custom(String),
}

/// Stress test resource constraints
#[derive(Debug, Clone)]
pub struct StressResourceConstraints {
    /// Maximum memory usage (MB)
    pub max_memory: Option<usize>,
    /// Maximum CPU usage (0.0-1.0)
    pub max_cpu: Option<f64>,
    /// Maximum execution time
    pub max_time: Option<Duration>,
    /// Maximum concurrent tests
    pub max_concurrent: Option<usize>,
}

/// Platform configuration
#[derive(Debug, Clone)]
pub struct PlatformConfig {
    /// Platform identifier
    pub platform_id: String,
    /// Connection parameters
    pub connection_params: HashMap<String, String>,
    /// Authentication settings
    pub auth_settings: Option<AuthSettings>,
    /// Platform-specific options
    pub platform_options: HashMap<String, String>,
}

/// Authentication settings
#[derive(Debug, Clone)]
pub struct AuthSettings {
    /// Authentication type
    pub auth_type: AuthType,
    /// Credentials
    pub credentials: HashMap<String, String>,
    /// Token expiry
    pub token_expiry: Option<Duration>,
}

/// Authentication types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthType {
    /// API key authentication
    ApiKey,
    /// OAuth authentication
    OAuth,
    /// Certificate authentication
    Certificate,
    /// Custom authentication
    Custom(String),
}
