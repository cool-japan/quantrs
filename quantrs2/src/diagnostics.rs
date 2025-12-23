//! # System Diagnostics and Health Checks
//!
//! This module provides diagnostic tools for checking the QuantRS2 environment,
//! validating hardware capabilities, and troubleshooting issues.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use quantrs2::diagnostics;
//!
//! // Run full system diagnostics
//! let report = diagnostics::run_diagnostics();
//! println!("{}", report);
//!
//! // Check if system is ready for quantum simulation
//! if !diagnostics::is_ready() {
//!     eprintln!("System is not ready for quantum simulation");
//!     diagnostics::print_issues();
//! }
//! ```

use crate::config::Config;
use crate::version::{check_compatibility, VersionInfo};
use std::fmt;

/// Diagnostic severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational message
    Info,
    /// Warning - system will work but may not be optimal
    Warning,
    /// Error - critical issue that will prevent operation
    Error,
}

/// Diagnostic issue
#[derive(Debug, Clone)]
pub struct DiagnosticIssue {
    /// Severity level
    pub severity: Severity,
    /// Component or feature affected
    pub component: String,
    /// Description of the issue
    pub message: String,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
}

impl DiagnosticIssue {
    /// Create a new diagnostic issue
    pub fn new(
        severity: Severity,
        component: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            component: component.into(),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Add a suggestion for fixing the issue
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

impl fmt::Display for DiagnosticIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let severity_str = match self.severity {
            Severity::Info => "INFO",
            Severity::Warning => "WARN",
            Severity::Error => "ERROR",
        };

        write!(f, "[{}] {}: {}", severity_str, self.component, self.message)?;

        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  Suggestion: {suggestion}")?;
        }

        Ok(())
    }
}

/// Enabled feature flags
#[derive(Debug, Clone, Copy)]
pub struct FeatureInfo {
    /// Circuit feature enabled
    pub circuit: bool,
    /// Simulation feature enabled
    pub sim: bool,
    /// Machine learning feature enabled
    pub ml: bool,
    /// Device/hardware feature enabled
    pub device: bool,
    /// Annealing feature enabled
    pub anneal: bool,
    /// Tytan feature enabled
    pub tytan: bool,
    /// SymEngine feature enabled
    pub symengine: bool,
}

impl FeatureInfo {
    /// Detect enabled features based on compile-time feature flags
    pub const fn detect() -> Self {
        Self {
            circuit: cfg!(feature = "circuit"),
            sim: cfg!(feature = "sim"),
            ml: cfg!(feature = "ml"),
            device: cfg!(feature = "device"),
            anneal: cfg!(feature = "anneal"),
            tytan: cfg!(feature = "tytan"),
            symengine: cfg!(feature = "symengine"),
        }
    }

    /// Count the number of enabled features
    pub const fn count_enabled(&self) -> usize {
        let mut count = 0;
        if self.circuit {
            count += 1;
        }
        if self.sim {
            count += 1;
        }
        if self.ml {
            count += 1;
        }
        if self.device {
            count += 1;
        }
        if self.anneal {
            count += 1;
        }
        if self.tytan {
            count += 1;
        }
        if self.symengine {
            count += 1;
        }
        count
    }
}

/// System capabilities
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Total memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    /// GPU available
    pub has_gpu: bool,
    /// AVX2 support
    pub has_avx2: bool,
    /// AVX-512 support
    pub has_avx512: bool,
    /// ARM NEON support
    pub has_neon: bool,
}

impl SystemCapabilities {
    /// Detect system capabilities
    pub fn detect() -> Self {
        // Detect CPU cores
        let cpu_cores = num_cpus::get();

        // Detect memory
        let (total_memory_bytes, available_memory_bytes) = Self::detect_memory();

        // Detect SIMD capabilities
        let has_avx2 = Self::detect_avx2();
        let has_avx512 = Self::detect_avx512();
        let has_neon = cfg!(target_arch = "aarch64");

        // Detect GPU (check for CUDA/OpenCL/Metal availability)
        let has_gpu = Self::detect_gpu();

        Self {
            cpu_cores,
            total_memory_bytes,
            available_memory_bytes,
            has_gpu,
            has_avx2,
            has_avx512,
            has_neon,
        }
    }

    /// Detect total and available memory
    fn detect_memory() -> (u64, u64) {
        // Platform-specific memory detection
        #[cfg(target_os = "macos")]
        {
            Self::detect_memory_macos()
        }

        #[cfg(target_os = "linux")]
        {
            Self::detect_memory_linux()
        }

        #[cfg(target_os = "windows")]
        {
            Self::detect_memory_windows()
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            (0, 0)
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_memory_macos() -> (u64, u64) {
        use std::process::Command;

        // Get total memory using sysctl
        let total = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .and_then(|s| s.trim().parse::<u64>().ok())
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // For available memory, we'll use vm_stat and parse it
        // This is a rough estimate based on free + inactive pages
        let available = Command::new("vm_stat")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    let output_str = String::from_utf8(output.stdout).ok()?;
                    let mut free_pages = 0u64;
                    let mut inactive_pages = 0u64;
                    let page_size = 4096u64; // Default page size on macOS

                    for line in output_str.lines() {
                        if line.contains("Pages free:") {
                            free_pages = line
                                .split(':')
                                .nth(1)
                                .and_then(|s| s.trim().trim_end_matches('.').parse().ok())
                                .unwrap_or(0);
                        } else if line.contains("Pages inactive:") {
                            inactive_pages = line
                                .split(':')
                                .nth(1)
                                .and_then(|s| s.trim().trim_end_matches('.').parse().ok())
                                .unwrap_or(0);
                        }
                    }

                    Some((free_pages + inactive_pages) * page_size)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        (total, available)
    }

    #[cfg(target_os = "linux")]
    fn detect_memory_linux() -> (u64, u64) {
        use std::fs;

        let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let mut total = 0u64;
        let mut available = 0u64;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0)
                    * 1024; // Convert kB to bytes
            } else if line.starts_with("MemAvailable:") {
                available = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0)
                    * 1024; // Convert kB to bytes
            }
        }

        (total, available)
    }

    #[cfg(target_os = "windows")]
    fn detect_memory_windows() -> (u64, u64) {
        // Windows memory detection would require winapi calls
        // For now, return placeholder values
        // A complete implementation would use GetPhysicallyInstalledSystemMemory
        // and GlobalMemoryStatusEx from windows-sys crate
        (0, 0)
    }

    /// Detect AVX2 support
    #[allow(clippy::missing_const_for_fn)] // Uses runtime feature detection
    fn detect_avx2() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Check if compiled with AVX2 or detect at runtime
            if cfg!(target_feature = "avx2") {
                return true;
            }
            // Runtime detection using is_x86_feature_detected!
            // Both x86 and x86_64 support this macro
            std::arch::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Detect AVX-512 support
    #[allow(clippy::missing_const_for_fn)] // Uses runtime feature detection
    fn detect_avx512() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if cfg!(target_feature = "avx512f") {
                return true;
            }
            // Runtime detection using is_x86_feature_detected!
            // Both x86 and x86_64 support this macro
            std::arch::is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Detect GPU availability
    #[allow(clippy::missing_const_for_fn)] // Uses runtime detection
    #[allow(clippy::needless_return)] // Required for conditional compilation clarity
    fn detect_gpu() -> bool {
        // Check for Metal on macOS (Apple Silicon and Intel Macs with Metal support)
        #[cfg(target_os = "macos")]
        {
            // Metal is available on macOS 10.11+ which is all modern Macs
            // For a more thorough check, we would use the Metal framework
            // For now, assume Metal is available on macOS
            true
        }

        // Check for CUDA on Linux/Windows
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            use std::process::Command;
            // Check if nvidia-smi is available (indicates NVIDIA GPU with drivers)
            Command::new("nvidia-smi")
                .arg("--query-gpu=name")
                .arg("--format=csv,noheader")
                .output()
                .map(|output| output.status.success())
                .unwrap_or(false)
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            false
        }
    }
}

/// Diagnostic report
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// Version information
    pub version: VersionInfo,
    /// System capabilities
    pub capabilities: SystemCapabilities,
    /// Enabled features
    pub features: FeatureInfo,
    /// Configuration snapshot
    pub config: crate::config::ConfigData,
    /// List of issues found
    pub issues: Vec<DiagnosticIssue>,
}

impl DiagnosticReport {
    /// Check if the system is ready (no errors)
    pub fn is_ready(&self) -> bool {
        !self.has_errors()
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == Severity::Error)
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == Severity::Warning)
    }

    /// Get all errors
    pub fn errors(&self) -> Vec<&DiagnosticIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Error)
            .collect()
    }

    /// Get all warnings
    pub fn warnings(&self) -> Vec<&DiagnosticIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Warning)
            .collect()
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let errors = self.errors().len();
        let warnings = self.warnings().len();
        let info = self
            .issues
            .iter()
            .filter(|i| i.severity == Severity::Info)
            .count();

        format!("Diagnostic Summary: {errors} errors, {warnings} warnings, {info} info messages")
    }
}

impl fmt::Display for DiagnosticReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== QuantRS2 Diagnostic Report ===")?;
        writeln!(f)?;
        writeln!(f, "Version: {}", self.version.version_string())?;
        writeln!(f, "Platform: {}", self.version.target)?;
        writeln!(f, "Build Profile: {}", self.version.profile)?;
        writeln!(f)?;

        writeln!(f, "Enabled Features ({}):", self.features.count_enabled())?;
        writeln!(f, "  circuit: {}", self.features.circuit)?;
        writeln!(f, "  sim: {}", self.features.sim)?;
        writeln!(f, "  ml: {}", self.features.ml)?;
        writeln!(f, "  device: {}", self.features.device)?;
        writeln!(f, "  anneal: {}", self.features.anneal)?;
        writeln!(f, "  tytan: {}", self.features.tytan)?;
        writeln!(f, "  symengine: {}", self.features.symengine)?;
        writeln!(f)?;

        writeln!(f, "System Capabilities:")?;
        writeln!(f, "  CPU Cores: {}", self.capabilities.cpu_cores)?;
        writeln!(
            f,
            "  Total Memory: {:.2} GB",
            self.capabilities.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )?;
        writeln!(
            f,
            "  GPU Available: {}",
            if self.capabilities.has_gpu {
                "Yes"
            } else {
                "No"
            }
        )?;
        writeln!(f, "  AVX2: {}", self.capabilities.has_avx2)?;
        writeln!(f, "  AVX-512: {}", self.capabilities.has_avx512)?;
        writeln!(f, "  ARM NEON: {}", self.capabilities.has_neon)?;
        writeln!(f)?;

        writeln!(f, "Configuration:")?;
        writeln!(
            f,
            "  Threads: {}",
            self.config
                .num_threads
                .map_or_else(|| "auto".to_string(), |n| n.to_string())
        )?;
        writeln!(f, "  Log Level: {}", self.config.log_level.as_str())?;
        writeln!(
            f,
            "  Default Backend: {}",
            self.config.default_backend.as_str()
        )?;
        writeln!(f, "  GPU Enabled: {}", self.config.enable_gpu)?;
        writeln!(f, "  SIMD Enabled: {}", self.config.enable_simd)?;
        writeln!(f)?;

        if !self.issues.is_empty() {
            writeln!(f, "Issues Found:")?;
            for issue in &self.issues {
                writeln!(f, "  {issue}")?;
            }
            writeln!(f)?;
        }

        writeln!(f, "{}", self.summary())?;

        if self.is_ready() {
            writeln!(f)?;
            writeln!(f, "System is READY for quantum simulation!")?;
        } else if self.has_errors() {
            writeln!(f)?;
            writeln!(
                f,
                "System has CRITICAL ERRORS - please fix before proceeding"
            )?;
        } else if self.has_warnings() {
            writeln!(f)?;
            writeln!(
                f,
                "System is ready but has WARNINGS - review for optimal performance"
            )?;
        }

        Ok(())
    }
}

/// Run comprehensive system diagnostics
pub fn run_diagnostics() -> DiagnosticReport {
    let version = VersionInfo::current();
    let capabilities = SystemCapabilities::detect();
    let features = FeatureInfo::detect();
    let config = Config::global().snapshot();
    let mut issues = Vec::new();

    // Check version compatibility
    if let Err(compat_issues) = check_compatibility() {
        for issue in compat_issues {
            issues.push(
                DiagnosticIssue::new(Severity::Error, "Compatibility", format!("{issue}"))
                    .with_suggestion("Update Rust version or fix feature flags"),
            );
        }
    }

    // Check CPU capabilities
    if capabilities.cpu_cores < 2 {
        issues.push(
            DiagnosticIssue::new(
                Severity::Warning,
                "CPU",
                format!("Only {} CPU core(s) available", capabilities.cpu_cores),
            )
            .with_suggestion("Multi-core CPU recommended for better performance"),
        );
    }

    // Check SIMD support
    if !capabilities.has_avx2 && !capabilities.has_neon {
        issues.push(
            DiagnosticIssue::new(
                Severity::Warning,
                "SIMD",
                "No advanced SIMD support detected (AVX2/NEON)",
            )
            .with_suggestion("Performance may be limited without SIMD acceleration"),
        );
    }

    // Check GPU configuration
    if config.enable_gpu && !capabilities.has_gpu {
        issues.push(
            DiagnosticIssue::new(
                Severity::Warning,
                "GPU",
                "GPU acceleration enabled but no GPU detected",
            )
            .with_suggestion("Disable GPU or install GPU drivers"),
        );
    }

    // Check memory configuration
    if let Some(limit) = config.memory_limit_bytes {
        if limit < 1024 * 1024 * 1024 {
            // Less than 1 GB
            issues.push(
                DiagnosticIssue::new(
                    Severity::Warning,
                    "Memory",
                    format!("Memory limit is very low ({} MB)", limit / 1024 / 1024),
                )
                .with_suggestion("Increase memory limit for larger quantum systems"),
            );
        }
    }

    // Add info messages
    issues.push(DiagnosticIssue::new(
        Severity::Info,
        "QuantRS2",
        format!("Running version {}", version.quantrs2),
    ));

    issues.push(DiagnosticIssue::new(
        Severity::Info,
        "SciRS2",
        format!("Using SciRS2 v{}", version.scirs2),
    ));

    DiagnosticReport {
        version,
        capabilities,
        features,
        config,
        issues,
    }
}

/// Check if the system is ready for quantum simulation
pub fn is_ready() -> bool {
    let report = run_diagnostics();
    report.is_ready()
}

/// Print diagnostic issues to stderr
pub fn print_issues() {
    let report = run_diagnostics();
    for issue in &report.issues {
        eprintln!("{issue}");
    }
}

/// Print full diagnostic report to stdout
pub fn print_report() {
    let report = run_diagnostics();
    println!("{report}");
}

/// Validate environment and panic if critical issues are found
///
/// This is useful to call at the start of your application to ensure
/// all requirements are met before proceeding.
pub fn validate_or_panic() {
    let report = run_diagnostics();
    if !report.is_ready() {
        eprintln!("{report}");
        panic!("System validation failed - cannot continue");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_issue() {
        let issue = DiagnosticIssue::new(Severity::Warning, "Test", "Test message")
            .with_suggestion("Fix it");

        assert_eq!(issue.severity, Severity::Warning);
        assert_eq!(issue.component, "Test");
        assert_eq!(issue.message, "Test message");
        assert_eq!(issue.suggestion, Some("Fix it".to_string()));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn test_system_capabilities() {
        let caps = SystemCapabilities::detect();
        assert!(caps.cpu_cores > 0);
    }

    #[test]
    fn test_run_diagnostics() {
        let report = run_diagnostics();
        assert!(!report.version.quantrs2.is_empty());
        assert!(report.capabilities.cpu_cores > 0);
    }

    #[test]
    fn test_is_ready() {
        // This should generally pass unless there are compatibility issues
        let ready = is_ready();
        // We don't assert true/false as it depends on the environment
        println!("System ready: {}", ready);
    }

    #[test]
    fn test_diagnostic_report_summary() {
        let report = run_diagnostics();
        let summary = report.summary();
        assert!(summary.contains("Diagnostic Summary"));
    }
}
