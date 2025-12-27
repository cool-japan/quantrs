//! # Global Configuration for QuantRS2
//!
//! This module provides global configuration settings for the QuantRS2 framework,
//! allowing users to customize behavior across all subcrates.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use quantrs2::config::{Config, LogLevel};
//!
//! // Get the global configuration
//! let config = Config::global();
//!
//! // Configure settings
//! config.set_num_threads(8);
//! config.set_log_level(LogLevel::Debug);
//! config.set_memory_limit_gb(16);
//!
//! // Or use the builder pattern
//! Config::builder()
//!     .num_threads(8)
//!     .log_level(LogLevel::Info)
//!     .memory_limit_gb(32)
//!     .apply();
//! ```
//!
//! ## Environment Variables
//!
//! Configuration can also be set via environment variables:
//! - `QUANTRS2_NUM_THREADS`: Number of threads for parallel operations
//! - `QUANTRS2_LOG_LEVEL`: Logging level (trace, debug, info, warn, error)
//! - `QUANTRS2_MEMORY_LIMIT_GB`: Memory limit in gigabytes
//! - `QUANTRS2_BACKEND`: Default backend (cpu, gpu, tensor_network)

use std::str::FromStr;
use std::sync::{Arc, RwLock};

/// Global logging level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Trace-level logging (most verbose)
    Trace,
    /// Debug-level logging
    Debug,
    /// Info-level logging
    Info,
    /// Warning-level logging
    Warn,
    /// Error-level logging (least verbose)
    Error,
    /// Disable all logging
    Off,
}

impl LogLevel {
    /// Get string representation
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Off => "off",
        }
    }
}

impl FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "trace" => Ok(Self::Trace),
            "debug" => Ok(Self::Debug),
            "info" => Ok(Self::Info),
            "warn" | "warning" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            "off" | "none" => Ok(Self::Off),
            _ => Err(format!("Invalid log level: {s}")),
        }
    }
}

/// Default simulation backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultBackend {
    /// CPU-based state vector simulation
    Cpu,
    /// GPU-accelerated simulation (if available)
    Gpu,
    /// Tensor network simulation
    TensorNetwork,
    /// Stabilizer simulation
    Stabilizer,
    /// Auto-select best backend
    Auto,
}

impl DefaultBackend {
    /// Get string representation
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::TensorNetwork => "tensor_network",
            Self::Stabilizer => "stabilizer",
            Self::Auto => "auto",
        }
    }
}

impl FromStr for DefaultBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "tensor_network" | "tensor-network" | "tn" => Ok(Self::TensorNetwork),
            "stabilizer" => Ok(Self::Stabilizer),
            "auto" => Ok(Self::Auto),
            _ => Err(format!("Invalid backend: {s}")),
        }
    }
}

/// Global configuration settings
#[derive(Debug, Clone)]
pub struct ConfigData {
    /// Number of threads for parallel operations
    pub num_threads: Option<usize>,
    /// Logging level
    pub log_level: LogLevel,
    /// Memory limit in bytes (None = unlimited)
    pub memory_limit_bytes: Option<usize>,
    /// Default simulation backend
    pub default_backend: DefaultBackend,
    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable telemetry collection
    pub enable_telemetry: bool,
    /// Cache directory for compiled circuits
    pub cache_dir: Option<String>,
    /// Maximum cache size in bytes
    pub max_cache_size_bytes: Option<usize>,
}

impl Default for ConfigData {
    fn default() -> Self {
        Self {
            num_threads: None, // Use system default
            log_level: LogLevel::Warn,
            memory_limit_bytes: None, // Unlimited
            default_backend: DefaultBackend::Auto,
            enable_gpu: true,
            enable_simd: true,
            enable_telemetry: false,
            cache_dir: None,
            max_cache_size_bytes: Some(1024 * 1024 * 1024), // 1 GB default
        }
    }
}

/// Global configuration instance
pub struct Config {
    data: Arc<RwLock<ConfigData>>,
}

impl Config {
    /// Create a new configuration with default settings
    fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(ConfigData::default())),
        }
    }

    /// Get the global configuration instance
    pub fn global() -> &'static Self {
        use std::sync::OnceLock;
        static CONFIG: OnceLock<Config> = OnceLock::new();

        CONFIG.get_or_init(|| {
            let config = Self::new();
            config.load_from_env();
            config
        })
    }

    /// Create a configuration builder
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder {
            data: ConfigData::default(),
        }
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> Option<usize> {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .num_threads
    }

    /// Set the number of threads
    pub fn set_num_threads(&self, num_threads: usize) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .num_threads = Some(num_threads);
    }

    /// Get the logging level
    pub fn log_level(&self) -> LogLevel {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .log_level
    }

    /// Set the logging level
    pub fn set_log_level(&self, level: LogLevel) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .log_level = level;
    }

    /// Get the memory limit in bytes
    pub fn memory_limit_bytes(&self) -> Option<usize> {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .memory_limit_bytes
    }

    /// Set the memory limit in bytes
    pub fn set_memory_limit_bytes(&self, limit: usize) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .memory_limit_bytes = Some(limit);
    }

    /// Set the memory limit in gigabytes
    pub fn set_memory_limit_gb(&self, limit_gb: usize) {
        self.set_memory_limit_bytes(limit_gb * 1024 * 1024 * 1024);
    }

    /// Get the default backend
    pub fn default_backend(&self) -> DefaultBackend {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .default_backend
    }

    /// Set the default backend
    pub fn set_default_backend(&self, backend: DefaultBackend) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .default_backend = backend;
    }

    /// Check if GPU acceleration is enabled
    pub fn is_gpu_enabled(&self) -> bool {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .enable_gpu
    }

    /// Enable or disable GPU acceleration
    pub fn set_gpu_enabled(&self, enabled: bool) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .enable_gpu = enabled;
    }

    /// Check if SIMD optimizations are enabled
    pub fn is_simd_enabled(&self) -> bool {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .enable_simd
    }

    /// Enable or disable SIMD optimizations
    pub fn set_simd_enabled(&self, enabled: bool) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .enable_simd = enabled;
    }

    /// Check if telemetry is enabled
    pub fn is_telemetry_enabled(&self) -> bool {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .enable_telemetry
    }

    /// Enable or disable telemetry
    pub fn set_telemetry_enabled(&self, enabled: bool) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .enable_telemetry = enabled;
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> Option<String> {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .cache_dir
            .clone()
    }

    /// Set the cache directory
    pub fn set_cache_dir(&self, dir: impl Into<String>) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .cache_dir = Some(dir.into());
    }

    /// Get the maximum cache size in bytes
    pub fn max_cache_size_bytes(&self) -> Option<usize> {
        self.data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .max_cache_size_bytes
    }

    /// Set the maximum cache size in bytes
    pub fn set_max_cache_size_bytes(&self, size: usize) {
        self.data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .max_cache_size_bytes = Some(size);
    }

    /// Set the maximum cache size in megabytes
    pub fn set_max_cache_size_mb(&self, size_mb: usize) {
        self.set_max_cache_size_bytes(size_mb * 1024 * 1024);
    }

    /// Load configuration from environment variables
    fn load_from_env(&self) {
        // Number of threads
        if let Ok(val) = std::env::var("QUANTRS2_NUM_THREADS") {
            if let Ok(num) = val.parse::<usize>() {
                self.set_num_threads(num);
            }
        }

        // Log level
        if let Ok(val) = std::env::var("QUANTRS2_LOG_LEVEL") {
            if let Ok(level) = val.parse::<LogLevel>() {
                self.set_log_level(level);
            }
        }

        // Memory limit
        if let Ok(val) = std::env::var("QUANTRS2_MEMORY_LIMIT_GB") {
            if let Ok(limit) = val.parse::<usize>() {
                self.set_memory_limit_gb(limit);
            }
        }

        // Backend
        if let Ok(val) = std::env::var("QUANTRS2_BACKEND") {
            if let Ok(backend) = val.parse::<DefaultBackend>() {
                self.set_default_backend(backend);
            }
        }

        // GPU
        if let Ok(val) = std::env::var("QUANTRS2_ENABLE_GPU") {
            if let Ok(enabled) = val.parse::<bool>() {
                self.set_gpu_enabled(enabled);
            }
        }

        // SIMD
        if let Ok(val) = std::env::var("QUANTRS2_ENABLE_SIMD") {
            if let Ok(enabled) = val.parse::<bool>() {
                self.set_simd_enabled(enabled);
            }
        }

        // Telemetry
        if let Ok(val) = std::env::var("QUANTRS2_ENABLE_TELEMETRY") {
            if let Ok(enabled) = val.parse::<bool>() {
                self.set_telemetry_enabled(enabled);
            }
        }

        // Cache directory
        if let Ok(val) = std::env::var("QUANTRS2_CACHE_DIR") {
            self.set_cache_dir(val);
        }

        // Max cache size
        if let Ok(val) = std::env::var("QUANTRS2_MAX_CACHE_SIZE_MB") {
            if let Ok(size) = val.parse::<usize>() {
                self.set_max_cache_size_mb(size);
            }
        }
    }

    /// Reset configuration to defaults
    pub fn reset(&self) {
        *self.data.write().unwrap_or_else(|e| e.into_inner()) = ConfigData::default();
    }

    /// Get a copy of the current configuration data
    pub fn snapshot(&self) -> ConfigData {
        self.data.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Builder for configuration
pub struct ConfigBuilder {
    data: ConfigData,
}

impl ConfigBuilder {
    /// Set the number of threads
    #[must_use]
    pub const fn num_threads(mut self, num_threads: usize) -> Self {
        self.data.num_threads = Some(num_threads);
        self
    }

    /// Set the logging level
    #[must_use]
    pub const fn log_level(mut self, level: LogLevel) -> Self {
        self.data.log_level = level;
        self
    }

    /// Set the memory limit in bytes
    #[must_use]
    pub const fn memory_limit_bytes(mut self, limit: usize) -> Self {
        self.data.memory_limit_bytes = Some(limit);
        self
    }

    /// Set the memory limit in gigabytes
    #[must_use]
    pub const fn memory_limit_gb(self, limit_gb: usize) -> Self {
        self.memory_limit_bytes(limit_gb * 1024 * 1024 * 1024)
    }

    /// Set the default backend
    #[must_use]
    pub const fn default_backend(mut self, backend: DefaultBackend) -> Self {
        self.data.default_backend = backend;
        self
    }

    /// Enable or disable GPU acceleration
    #[must_use]
    pub const fn enable_gpu(mut self, enabled: bool) -> Self {
        self.data.enable_gpu = enabled;
        self
    }

    /// Enable or disable SIMD optimizations
    #[must_use]
    pub const fn enable_simd(mut self, enabled: bool) -> Self {
        self.data.enable_simd = enabled;
        self
    }

    /// Enable or disable telemetry
    #[must_use]
    pub const fn enable_telemetry(mut self, enabled: bool) -> Self {
        self.data.enable_telemetry = enabled;
        self
    }

    /// Set the cache directory
    #[must_use]
    pub fn cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.data.cache_dir = Some(dir.into());
        self
    }

    /// Set the maximum cache size in bytes
    #[must_use]
    pub const fn max_cache_size_bytes(mut self, size: usize) -> Self {
        self.data.max_cache_size_bytes = Some(size);
        self
    }

    /// Set the maximum cache size in megabytes
    #[must_use]
    pub const fn max_cache_size_mb(self, size_mb: usize) -> Self {
        self.max_cache_size_bytes(size_mb * 1024 * 1024)
    }

    /// Apply this configuration as the global configuration
    pub fn apply(self) {
        let config = Config::global();
        *config.data.write().unwrap_or_else(|e| e.into_inner()) = self.data;
    }

    /// Build the configuration without applying it globally
    pub fn build(self) -> ConfigData {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_parsing() {
        assert_eq!("trace".parse::<LogLevel>(), Ok(LogLevel::Trace));
        assert_eq!("DEBUG".parse::<LogLevel>(), Ok(LogLevel::Debug));
        assert_eq!("Info".parse::<LogLevel>(), Ok(LogLevel::Info));
        assert_eq!("warn".parse::<LogLevel>(), Ok(LogLevel::Warn));
        assert_eq!("error".parse::<LogLevel>(), Ok(LogLevel::Error));
        assert_eq!("off".parse::<LogLevel>(), Ok(LogLevel::Off));
        assert!("invalid".parse::<LogLevel>().is_err());
    }

    #[test]
    fn test_backend_parsing() {
        assert_eq!("cpu".parse::<DefaultBackend>(), Ok(DefaultBackend::Cpu));
        assert_eq!("GPU".parse::<DefaultBackend>(), Ok(DefaultBackend::Gpu));
        assert_eq!(
            "tensor-network".parse::<DefaultBackend>(),
            Ok(DefaultBackend::TensorNetwork)
        );
        assert_eq!(
            "stabilizer".parse::<DefaultBackend>(),
            Ok(DefaultBackend::Stabilizer)
        );
        assert_eq!("auto".parse::<DefaultBackend>(), Ok(DefaultBackend::Auto));
        assert!("invalid".parse::<DefaultBackend>().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config_data = Config::builder()
            .num_threads(8)
            .log_level(LogLevel::Debug)
            .memory_limit_gb(16)
            .default_backend(DefaultBackend::Gpu)
            .enable_gpu(true)
            .enable_simd(true)
            .enable_telemetry(false)
            .cache_dir("/tmp/quantrs2")
            .max_cache_size_mb(512)
            .build();

        assert_eq!(config_data.num_threads, Some(8));
        assert_eq!(config_data.log_level, LogLevel::Debug);
        assert_eq!(
            config_data.memory_limit_bytes,
            Some(16 * 1024 * 1024 * 1024)
        );
        assert_eq!(config_data.default_backend, DefaultBackend::Gpu);
        assert!(config_data.enable_gpu);
        assert!(config_data.enable_simd);
        assert!(!config_data.enable_telemetry);
        assert_eq!(config_data.cache_dir, Some("/tmp/quantrs2".to_string()));
        assert_eq!(config_data.max_cache_size_bytes, Some(512 * 1024 * 1024));
    }

    #[test]
    fn test_global_config() {
        let config = Config::global();

        // Test setters and getters
        config.set_num_threads(4);
        assert_eq!(config.num_threads(), Some(4));

        config.set_log_level(LogLevel::Info);
        assert_eq!(config.log_level(), LogLevel::Info);

        config.set_memory_limit_gb(8);
        assert_eq!(config.memory_limit_bytes(), Some(8 * 1024 * 1024 * 1024));

        config.set_default_backend(DefaultBackend::TensorNetwork);
        assert_eq!(config.default_backend(), DefaultBackend::TensorNetwork);

        config.set_gpu_enabled(false);
        assert!(!config.is_gpu_enabled());

        config.set_simd_enabled(false);
        assert!(!config.is_simd_enabled());

        config.set_telemetry_enabled(true);
        assert!(config.is_telemetry_enabled());

        config.set_cache_dir("/test/cache");
        assert_eq!(config.cache_dir(), Some("/test/cache".to_string()));

        config.set_max_cache_size_mb(256);
        assert_eq!(config.max_cache_size_bytes(), Some(256 * 1024 * 1024));
    }

    #[test]
    fn test_config_snapshot() {
        let config = Config::global();

        // Reset to ensure clean state
        config.reset();

        // Set known values for this test
        config.set_num_threads(6);
        config.set_log_level(LogLevel::Warn);

        let snapshot = config.snapshot();

        // Verify the snapshot captured the current state
        assert_eq!(snapshot.num_threads, config.num_threads());
        assert_eq!(snapshot.log_level, config.log_level());
    }
}
