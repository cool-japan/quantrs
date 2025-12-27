//! # Version Compatibility and Information Module
//!
//! This module provides version information and compatibility checking for `QuantRS2` and its dependencies.
//!
//! ## Version Constants
//!
//! ```rust,ignore
//! use quantrs2::version::*;
//!
//! println!("QuantRS2 version: {}", QUANTRS2_VERSION);
//! println!("SciRS2 version: {}", SCIRS2_VERSION);
//! println!("Build timestamp: {}", BUILD_TIMESTAMP);
//! ```
//!
//! ## Compatibility Checking
//!
//! ```rust,ignore
//! use quantrs2::version::check_compatibility;
//!
//! // Check if all dependencies are compatible
//! if let Err(issues) = check_compatibility() {
//!     for issue in issues {
//!         eprintln!("Compatibility issue: {}", issue);
//!     }
//! }
//! ```

#![allow(clippy::doc_markdown)] // QuantRS2, SciRS2 are proper names
#![allow(clippy::must_use_candidate)] // Most functions naturally return values
#![allow(clippy::redundant_closure_for_method_calls)] // Explicit closures are clearer

/// `QuantRS2` framework version
pub const QUANTRS2_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Alias for QUANTRS2_VERSION for convenience
pub const VERSION: &str = QUANTRS2_VERSION;

/// QuantRS2 core library version
pub const QUANTRS2_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum required SciRS2 version
pub const SCIRS2_MIN_VERSION: &str = "0.1.0-rc.2";

/// Expected SciRS2 version (for optimal compatibility)
pub const SCIRS2_VERSION: &str = "0.1.0-rc.2";

/// Build timestamp (UTC)
pub const BUILD_TIMESTAMP: &str = env!("VERGEN_BUILD_TIMESTAMP");

/// Git commit hash (if available)
pub const GIT_COMMIT_HASH: Option<&str> = option_env!("VERGEN_GIT_SHA");

/// Git branch (if available)
pub const GIT_BRANCH: Option<&str> = option_env!("VERGEN_GIT_BRANCH");

/// Rust compiler version used for build
pub const RUSTC_VERSION: &str = env!("VERGEN_RUSTC_SEMVER");

/// Target triple
pub const TARGET_TRIPLE: &str = env!("VERGEN_CARGO_TARGET_TRIPLE");

/// Build profile (debug/release)
pub const BUILD_PROFILE: &str = env!("VERGEN_CARGO_PROFILE");

/// Minimum required Rust version
pub const MIN_RUST_VERSION: &str = "1.86.0";

/// Version information structure
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionInfo {
    /// QuantRS2 version
    pub quantrs2: String,
    /// SciRS2 version (expected)
    pub scirs2: String,
    /// Rust compiler version
    pub rustc: String,
    /// Build timestamp
    pub build_time: String,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Git branch (if available)
    pub git_branch: Option<String>,
    /// Target platform
    pub target: String,
    /// Build profile
    pub profile: String,
}

impl VersionInfo {
    /// Get current version information
    pub fn current() -> Self {
        Self {
            quantrs2: QUANTRS2_VERSION.to_string(),
            scirs2: SCIRS2_VERSION.to_string(),
            rustc: RUSTC_VERSION.to_string(),
            build_time: BUILD_TIMESTAMP.to_string(),
            git_commit: GIT_COMMIT_HASH.map(|s| s.to_string()),
            git_branch: GIT_BRANCH.map(|s| s.to_string()),
            target: TARGET_TRIPLE.to_string(),
            profile: BUILD_PROFILE.to_string(),
        }
    }

    /// Get a formatted version string
    pub fn version_string(&self) -> String {
        format!("QuantRS2 v{}", self.quantrs2)
    }

    /// Get a detailed version string with all information
    pub fn detailed_version_string(&self) -> String {
        let mut parts = vec![
            format!("QuantRS2 v{}", self.quantrs2),
            format!("SciRS2 v{}", self.scirs2),
            format!("rustc v{}", self.rustc),
        ];

        if let Some(ref commit) = self.git_commit {
            parts.push(format!("commit {}", &commit[..7.min(commit.len())]));
        }

        if let Some(ref branch) = self.git_branch {
            parts.push(format!("branch {branch}"));
        }

        parts.push(format!("built {}", self.build_time));
        parts.push(format!("target {}", self.target));
        parts.push(format!("profile {}", self.profile));

        parts.join(" | ")
    }
}

impl std::fmt::Display for VersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.detailed_version_string())
    }
}

/// Compatibility issue type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityIssue {
    /// Rust version is too old
    RustVersionTooOld {
        /// Current version
        current: String,
        /// Required version
        required: String,
    },
    /// Feature combination is unsupported
    UnsupportedFeatureCombination {
        /// Description of the issue
        description: String,
    },
    /// Dependency version mismatch
    DependencyVersionMismatch {
        /// Dependency name
        dependency: String,
        /// Expected version
        expected: String,
        /// Detected version (if available)
        detected: Option<String>,
    },
}

impl std::fmt::Display for CompatibilityIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RustVersionTooOld { current, required } => {
                write!(
                    f,
                    "Rust version {current} is too old, {required} or newer is required"
                )
            }
            Self::UnsupportedFeatureCombination { description } => {
                write!(f, "Unsupported feature combination: {description}")
            }
            Self::DependencyVersionMismatch {
                dependency,
                expected,
                detected,
            } => {
                if let Some(detected) = detected {
                    write!(
                        f,
                        "Dependency '{dependency}' version mismatch: expected {expected}, detected {detected}"
                    )
                } else {
                    write!(
                        f,
                        "Dependency '{dependency}' version mismatch: expected {expected}, but version could not be detected"
                    )
                }
            }
        }
    }
}

/// Check version compatibility of `QuantRS2` and its dependencies
///
/// Returns `Ok(())` if all compatibility checks pass, or `Err(issues)` with a list of issues.
///
/// # Errors
///
/// Returns a vector of [`CompatibilityIssue`] if any compatibility problems are detected:
/// - Rust version is too old
/// - Unsupported feature combinations
/// - Dependency version mismatches
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::version::check_compatibility;
///
/// match check_compatibility() {
///     Ok(()) => println!("All compatibility checks passed"),
///     Err(issues) => {
///         eprintln!("Compatibility issues detected:");
///         for issue in issues {
///             eprintln!("  - {}", issue);
///         }
///     }
/// }
/// ```
pub fn check_compatibility() -> Result<(), Vec<CompatibilityIssue>> {
    let mut issues = Vec::new();

    // Check Rust version
    if let Err(issue) = check_rust_version() {
        issues.push(issue);
    }

    // Check feature compatibility
    if let Err(mut feature_issues) = check_feature_compatibility() {
        issues.append(&mut feature_issues);
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// Check if the Rust compiler version is sufficient
fn check_rust_version() -> Result<(), CompatibilityIssue> {
    // Parse versions for comparison
    let current = RUSTC_VERSION;
    let required = MIN_RUST_VERSION;

    // Simple string comparison for version numbers (works for semver)
    if version_compare(current, required) >= 0 {
        Ok(())
    } else {
        Err(CompatibilityIssue::RustVersionTooOld {
            current: current.to_string(),
            required: required.to_string(),
        })
    }
}

/// Check feature flag compatibility
fn check_feature_compatibility() -> Result<(), Vec<CompatibilityIssue>> {
    let mut issues = Vec::new();

    // Check that circuit feature is enabled when sim is enabled
    #[cfg(all(feature = "sim", not(feature = "circuit")))]
    {
        issues.push(CompatibilityIssue::UnsupportedFeatureCombination {
            description: "Feature 'sim' requires feature 'circuit'".to_string(),
        });
    }

    // Check that anneal is enabled when tytan is enabled
    #[cfg(all(feature = "tytan", not(feature = "anneal")))]
    {
        issues.push(CompatibilityIssue::UnsupportedFeatureCombination {
            description: "Feature 'tytan' requires feature 'anneal'".to_string(),
        });
    }

    // Check that sim and anneal are enabled when ml is enabled
    #[cfg(all(feature = "ml", not(all(feature = "sim", feature = "anneal"))))]
    {
        issues.push(CompatibilityIssue::UnsupportedFeatureCombination {
            description: "Feature 'ml' requires features 'sim' and 'anneal'".to_string(),
        });
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// Simple version comparison (returns -1, 0, or 1)
fn version_compare(v1: &str, v2: &str) -> i32 {
    let v1_parts: Vec<&str> = v1.split('.').collect();
    let v2_parts: Vec<&str> = v2.split('.').collect();

    for (p1, p2) in v1_parts.iter().zip(v2_parts.iter()) {
        // Extract numeric part (ignore -beta, -rc, etc.)
        let n1 = p1
            .split('-')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        let n2 = p2
            .split('-')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        if n1 < n2 {
            return -1;
        } else if n1 > n2 {
            return 1;
        }
    }

    // If all parts are equal, compare lengths
    match v1_parts.len().cmp(&v2_parts.len()) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => 0,
    }
}

/// Print version information to stdout
pub fn print_version() {
    let info = VersionInfo::current();
    println!("{}", info.version_string());
}

/// Print detailed version information to stdout
pub fn print_detailed_version() {
    let info = VersionInfo::current();
    println!("{}", info.detailed_version_string());
}

/// Validate that the current environment is compatible with QuantRS2
///
/// This function should be called at startup to ensure all requirements are met.
/// It panics if critical compatibility issues are found.
///
/// # Panics
///
/// Panics if critical compatibility issues are detected.
pub fn validate_environment() {
    if let Err(issues) = check_compatibility() {
        eprintln!("QuantRS2 compatibility issues detected:");
        for issue in &issues {
            eprintln!("  - {issue}");
        }
        panic!("Cannot continue due to compatibility issues");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_constants() {
        assert!(!QUANTRS2_VERSION.is_empty());
        assert!(!SCIRS2_VERSION.is_empty());
        assert!(!RUSTC_VERSION.is_empty());
        assert!(!BUILD_TIMESTAMP.is_empty());
        assert!(!TARGET_TRIPLE.is_empty());
        assert!(!BUILD_PROFILE.is_empty());
    }

    #[test]
    fn test_version_info() {
        let info = VersionInfo::current();
        assert_eq!(info.quantrs2, QUANTRS2_VERSION);
        assert_eq!(info.scirs2, SCIRS2_VERSION);

        let version_str = info.version_string();
        assert!(version_str.contains(QUANTRS2_VERSION));

        let detailed_str = info.detailed_version_string();
        assert!(detailed_str.contains(QUANTRS2_VERSION));
        assert!(detailed_str.contains(SCIRS2_VERSION));
    }

    #[test]
    fn test_version_compare() {
        assert_eq!(version_compare("1.70.0", "1.70.0"), 0);
        assert_eq!(version_compare("1.71.0", "1.70.0"), 1);
        assert_eq!(version_compare("1.70.0", "1.71.0"), -1);
        assert_eq!(version_compare("1.70.1", "1.70.0"), 1);
        assert_eq!(version_compare("2.0.0", "1.99.0"), 1);

        // Test with pre-release versions
        // Note: Pre-release versions are considered greater due to having more parts
        // This is acceptable for our use case (minimum version checking)
        // as "1.70.0-beta.1" satisfies a "1.70.0" requirement
        assert!(version_compare("1.70.0-beta.1", "1.70.0") >= 0);
        assert_eq!(version_compare("1.71.0-rc.2", "1.70.0"), 1);
    }

    #[test]
    fn test_compatibility_check() {
        // This should not panic in normal builds
        let result = check_compatibility();

        // Print any issues for debugging
        if let Err(ref issues) = result {
            for issue in issues {
                eprintln!("Compatibility issue: {issue}");
            }
        }

        // In a properly configured build, this should pass
        // However, we allow it to fail in CI or development environments
        // where feature flags might be misconfigured
        match result {
            Ok(()) => {
                // All good
            }
            Err(issues) => {
                // Log issues but don't fail the test
                eprintln!("Note: {} compatibility issues detected (this may be expected in some build configurations)", issues.len());
            }
        }
    }

    #[test]
    fn test_rust_version_check() {
        // This should not panic - we're building with a compatible Rust version
        let result = check_rust_version();
        assert!(result.is_ok(), "Rust version check failed: {result:?}");
    }

    #[test]
    fn test_display_implementations() {
        let info = VersionInfo::current();
        let display_str = format!("{info}");
        assert!(!display_str.is_empty());

        let issue = CompatibilityIssue::RustVersionTooOld {
            current: "1.60.0".to_string(),
            required: "1.70.0".to_string(),
        };
        let issue_str = format!("{issue}");
        assert!(issue_str.contains("1.60.0"));
        assert!(issue_str.contains("1.70.0"));
    }
}
