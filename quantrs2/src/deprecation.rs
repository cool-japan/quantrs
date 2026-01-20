//! # Deprecation Framework for QuantRS2
//!
//! This module provides utilities and infrastructure for managing API deprecations,
//! sunset timelines, and migration guidance.
//!
//! ## Overview
//!
//! The deprecation framework helps maintainers and users:
//! - Track deprecated items with clear migration paths
//! - Understand sunset timelines for alpha/beta APIs
//! - Receive runtime warnings about deprecated usage
//! - Plan migrations to stable APIs
//!
//! ## Usage
//!
//! ### Checking Deprecation Status
//!
//! ```rust,ignore
//! use quantrs2::deprecation::{DeprecationStatus, is_deprecated};
//!
//! // Check if a feature is deprecated
//! if is_deprecated("old_feature_name") {
//!     // Use alternative
//! }
//! ```
//!
//! ### Querying Migration Information
//!
//! ```rust,ignore
//! use quantrs2::deprecation::get_migration_info;
//!
//! if let Some(info) = get_migration_info("old_api") {
//!     println!("Migration: {}", info.migration_guide);
//!     println!("Alternative: {}", info.alternative);
//! }
//! ```

#![allow(clippy::doc_markdown)]

use std::collections::HashMap;
use std::sync::OnceLock;

/// Deprecation status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeprecationStatus {
    /// Not deprecated - stable API
    Stable,
    /// Will be deprecated in a future version
    PendingDeprecation,
    /// Currently deprecated - migration recommended
    Deprecated,
    /// Removed in this version (for documentation purposes)
    Removed,
}

impl std::fmt::Display for DeprecationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "Stable"),
            Self::PendingDeprecation => write!(f, "Pending Deprecation"),
            Self::Deprecated => write!(f, "Deprecated"),
            Self::Removed => write!(f, "Removed"),
        }
    }
}

/// API stability level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StabilityLevel {
    /// Experimental API - may change without notice
    Experimental,
    /// Unstable API - may change between minor versions
    Unstable,
    /// Stable API - will not change until next major version
    Stable,
}

impl std::fmt::Display for StabilityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Experimental => write!(f, "Experimental"),
            Self::Unstable => write!(f, "Unstable"),
            Self::Stable => write!(f, "Stable"),
        }
    }
}

/// Information about a deprecated item
#[derive(Debug, Clone)]
pub struct DeprecationInfo {
    /// Name of the deprecated item
    pub name: String,
    /// Deprecation status
    pub status: DeprecationStatus,
    /// Version when deprecation was introduced
    pub deprecated_since: Option<String>,
    /// Version when item will be/was removed
    pub removal_version: Option<String>,
    /// Reason for deprecation
    pub reason: String,
    /// Migration guide or instructions
    pub migration_guide: String,
    /// Alternative to use instead
    pub alternative: Option<String>,
    /// Documentation link
    pub doc_link: Option<String>,
}

impl DeprecationInfo {
    /// Create a new deprecation info entry
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: DeprecationStatus::Deprecated,
            deprecated_since: None,
            removal_version: None,
            reason: String::new(),
            migration_guide: String::new(),
            alternative: None,
            doc_link: None,
        }
    }

    /// Set deprecation status
    #[must_use]
    pub const fn status(mut self, status: DeprecationStatus) -> Self {
        self.status = status;
        self
    }

    /// Set the version when deprecation was introduced
    #[must_use]
    pub fn since(mut self, version: impl Into<String>) -> Self {
        self.deprecated_since = Some(version.into());
        self
    }

    /// Set the planned removal version
    #[must_use]
    pub fn removal(mut self, version: impl Into<String>) -> Self {
        self.removal_version = Some(version.into());
        self
    }

    /// Set the deprecation reason
    #[must_use]
    pub fn reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Set the migration guide
    #[must_use]
    pub fn guide(mut self, guide: impl Into<String>) -> Self {
        self.migration_guide = guide.into();
        self
    }

    /// Set the alternative to use
    #[must_use]
    pub fn alternative(mut self, alt: impl Into<String>) -> Self {
        self.alternative = Some(alt.into());
        self
    }

    /// Set documentation link
    #[must_use]
    pub fn doc(mut self, link: impl Into<String>) -> Self {
        self.doc_link = Some(link.into());
        self
    }

    /// Format a deprecation warning message
    #[allow(clippy::format_push_string)] // Simple string building, write! adds complexity
    pub fn warning_message(&self) -> String {
        let mut msg = format!("'{}' is {}", self.name, self.status);

        if let Some(ref since) = self.deprecated_since {
            msg.push_str(&format!(" since v{since}"));
        }

        if let Some(ref removal) = self.removal_version {
            msg.push_str(&format!(", scheduled for removal in v{removal}"));
        }

        if !self.reason.is_empty() {
            msg.push_str(&format!(". Reason: {}", self.reason));
        }

        if let Some(ref alt) = self.alternative {
            msg.push_str(&format!(". Use '{alt}' instead"));
        }

        msg
    }
}

impl std::fmt::Display for DeprecationInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.warning_message())
    }
}

/// Module stability information
#[derive(Debug, Clone)]
pub struct ModuleStability {
    /// Module name
    pub module: String,
    /// Stability level
    pub stability: StabilityLevel,
    /// Additional notes
    pub notes: Option<String>,
}

impl ModuleStability {
    /// Create new module stability info
    pub fn new(module: impl Into<String>, stability: StabilityLevel) -> Self {
        Self {
            module: module.into(),
            stability,
            notes: None,
        }
    }

    /// Add notes
    #[must_use]
    pub fn notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

// Global deprecation registry
static DEPRECATION_REGISTRY: OnceLock<DeprecationRegistry> = OnceLock::new();

/// Registry holding all deprecation information
struct DeprecationRegistry {
    items: HashMap<String, DeprecationInfo>,
    modules: HashMap<String, ModuleStability>,
}

impl DeprecationRegistry {
    fn new() -> Self {
        let mut registry = Self {
            items: HashMap::new(),
            modules: HashMap::new(),
        };
        registry.register_known_deprecations();
        registry.register_module_stability();
        registry
    }

    fn register_known_deprecations(&mut self) {
        // Register known deprecations for v0.1.0
        // Currently none - this is a placeholder for future deprecations

        // Example deprecation (commented out for reference):
        // self.items.insert(
        //     "old_api_name".to_string(),
        //     DeprecationInfo::new("old_api_name")
        //         .since("0.1.0")
        //         .removal("0.2.0")
        //         .reason("Replaced with more efficient implementation")
        //         .alternative("new_api_name")
        //         .guide("Replace calls to old_api_name() with new_api_name()"),
        // );

        // Alpha APIs that may change
        self.items.insert(
            "symengine_experimental".to_string(),
            DeprecationInfo::new("symengine_experimental")
                .status(DeprecationStatus::PendingDeprecation)
                .since("0.1.0")
                .removal("0.2.0")
                .reason("SymEngine integration API is still experimental and may change")
                .guide(
                    "Review updated API before upgrading to 0.2.0. \
                     Check CHANGELOG.md for migration instructions.",
                ),
        );
    }

    fn register_module_stability(&mut self) {
        // Register module stability levels
        self.modules.insert(
            "quantrs2::core".to_string(),
            ModuleStability::new("quantrs2::core", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::circuit".to_string(),
            ModuleStability::new("quantrs2::circuit", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::sim".to_string(),
            ModuleStability::new("quantrs2::sim", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::device".to_string(),
            ModuleStability::new("quantrs2::device", StabilityLevel::Unstable)
                .notes("Hardware provider APIs may change based on provider updates"),
        );

        self.modules.insert(
            "quantrs2::ml".to_string(),
            ModuleStability::new("quantrs2::ml", StabilityLevel::Unstable)
                .notes("Machine learning APIs are stabilizing but may change in minor versions"),
        );

        self.modules.insert(
            "quantrs2::anneal".to_string(),
            ModuleStability::new("quantrs2::anneal", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::tytan".to_string(),
            ModuleStability::new("quantrs2::tytan", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::symengine".to_string(),
            ModuleStability::new("quantrs2::symengine", StabilityLevel::Experimental)
                .notes("SymEngine integration is experimental and API may change significantly"),
        );

        // Facade modules
        self.modules.insert(
            "quantrs2::config".to_string(),
            ModuleStability::new("quantrs2::config", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::diagnostics".to_string(),
            ModuleStability::new("quantrs2::diagnostics", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::utils".to_string(),
            ModuleStability::new("quantrs2::utils", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::testing".to_string(),
            ModuleStability::new("quantrs2::testing", StabilityLevel::Stable),
        );

        self.modules.insert(
            "quantrs2::bench".to_string(),
            ModuleStability::new("quantrs2::bench", StabilityLevel::Stable),
        );
    }
}

/// Get the global deprecation registry
fn get_registry() -> &'static DeprecationRegistry {
    DEPRECATION_REGISTRY.get_or_init(DeprecationRegistry::new)
}

/// Check if an item is deprecated
///
/// # Arguments
///
/// * `name` - The name of the item to check
///
/// # Returns
///
/// `true` if the item is deprecated or pending deprecation
///
/// # Example
///
/// ```rust
/// use quantrs2::deprecation::is_deprecated;
///
/// if is_deprecated("some_api") {
///     // Handle deprecated API usage
/// }
/// ```
pub fn is_deprecated(name: &str) -> bool {
    get_registry().items.get(name).is_some_and(|info| {
        matches!(
            info.status,
            DeprecationStatus::Deprecated | DeprecationStatus::PendingDeprecation
        )
    })
}

/// Get migration information for a deprecated item
///
/// # Arguments
///
/// * `name` - The name of the deprecated item
///
/// # Returns
///
/// `Some(DeprecationInfo)` if the item has deprecation information, `None` otherwise
///
/// # Example
///
/// ```rust
/// use quantrs2::deprecation::get_migration_info;
///
/// if let Some(info) = get_migration_info("old_api") {
///     println!("Migration guide: {}", info.migration_guide);
/// }
/// ```
pub fn get_migration_info(name: &str) -> Option<&'static DeprecationInfo> {
    get_registry().items.get(name)
}

/// Get stability information for a module
///
/// # Arguments
///
/// * `module` - The module path (e.g., "quantrs2::sim")
///
/// # Returns
///
/// `Some(ModuleStability)` if the module has stability information
///
/// # Example
///
/// ```rust
/// use quantrs2::deprecation::get_module_stability;
///
/// if let Some(stability) = get_module_stability("quantrs2::ml") {
///     println!("Stability level: {}", stability.stability);
/// }
/// ```
pub fn get_module_stability(module: &str) -> Option<&'static ModuleStability> {
    get_registry().modules.get(module)
}

/// Get all deprecated items
///
/// # Returns
///
/// An iterator over all items that are deprecated or pending deprecation
pub fn list_deprecated() -> impl Iterator<Item = &'static DeprecationInfo> {
    get_registry().items.values().filter(|info| {
        matches!(
            info.status,
            DeprecationStatus::Deprecated | DeprecationStatus::PendingDeprecation
        )
    })
}

/// Get all module stability information
///
/// # Returns
///
/// An iterator over all module stability entries
pub fn list_modules() -> impl Iterator<Item = &'static ModuleStability> {
    get_registry().modules.values()
}

/// Check if any deprecations are pending
///
/// # Returns
///
/// `true` if there are items pending deprecation
pub fn has_pending_deprecations() -> bool {
    get_registry()
        .items
        .values()
        .any(|info| info.status == DeprecationStatus::PendingDeprecation)
}

/// Print deprecation warnings for all deprecated items
///
/// This is useful for startup checks or migration planning.
///
/// # Example
///
/// ```rust
/// use quantrs2::deprecation::print_deprecation_warnings;
///
/// // Print all deprecation warnings at startup
/// print_deprecation_warnings();
/// ```
pub fn print_deprecation_warnings() {
    for info in list_deprecated() {
        eprintln!("[DEPRECATION WARNING] {}", info.warning_message());
    }
}

/// Generate a migration report for all deprecated items
///
/// # Returns
///
/// A formatted string containing all deprecation information and migration guides
///
/// # Example
///
/// ```rust
/// use quantrs2::deprecation::migration_report;
///
/// let report = migration_report();
/// println!("{}", report);
/// ```
pub fn migration_report() -> String {
    use std::fmt::Write;

    let mut report = String::from("# QuantRS2 Migration Report\n\n");

    report.push_str("## Module Stability\n\n");
    report.push_str("| Module | Stability | Notes |\n");
    report.push_str("|--------|-----------|-------|\n");

    for stability in list_modules() {
        let notes = stability.notes.as_deref().unwrap_or("-");
        let _ = writeln!(
            report,
            "| {} | {} | {} |",
            stability.module, stability.stability, notes
        );
    }

    report.push_str("\n## Deprecations\n\n");

    let deprecated: Vec<_> = list_deprecated().collect();
    if deprecated.is_empty() {
        report.push_str("No deprecated items in current version.\n");
    } else {
        for info in deprecated {
            let _ = writeln!(report, "### {}\n", info.name);
            let _ = writeln!(report, "**Status**: {}", info.status);

            if let Some(ref since) = info.deprecated_since {
                let _ = writeln!(report, "**Deprecated Since**: v{since}");
            }

            if let Some(ref removal) = info.removal_version {
                let _ = writeln!(report, "**Removal Version**: v{removal}");
            }

            if !info.reason.is_empty() {
                let _ = writeln!(report, "\n**Reason**: {}", info.reason);
            }

            if let Some(ref alt) = info.alternative {
                let _ = writeln!(report, "\n**Alternative**: `{alt}`");
            }

            if !info.migration_guide.is_empty() {
                let _ = writeln!(report, "\n**Migration Guide**:\n{}", info.migration_guide);
            }

            report.push('\n');
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deprecation_status_display() {
        assert_eq!(format!("{}", DeprecationStatus::Stable), "Stable");
        assert_eq!(
            format!("{}", DeprecationStatus::PendingDeprecation),
            "Pending Deprecation"
        );
        assert_eq!(format!("{}", DeprecationStatus::Deprecated), "Deprecated");
        assert_eq!(format!("{}", DeprecationStatus::Removed), "Removed");
    }

    #[test]
    fn test_stability_level_display() {
        assert_eq!(format!("{}", StabilityLevel::Experimental), "Experimental");
        assert_eq!(format!("{}", StabilityLevel::Unstable), "Unstable");
        assert_eq!(format!("{}", StabilityLevel::Stable), "Stable");
    }

    #[test]
    fn test_deprecation_info_builder() {
        let info = DeprecationInfo::new("test_api")
            .since("0.1.0")
            .removal("0.2.0")
            .reason("Testing")
            .alternative("new_api")
            .guide("Use new_api instead");

        assert_eq!(info.name, "test_api");
        assert_eq!(info.deprecated_since, Some("0.1.0".to_string()));
        assert_eq!(info.removal_version, Some("0.2.0".to_string()));
        assert_eq!(info.alternative, Some("new_api".to_string()));
    }

    #[test]
    fn test_warning_message() {
        let info = DeprecationInfo::new("old_function")
            .since("0.1.0")
            .removal("0.2.0")
            .reason("Replaced")
            .alternative("new_function");

        let msg = info.warning_message();
        assert!(msg.contains("old_function"));
        assert!(msg.contains("0.1.0"));
        assert!(msg.contains("0.2.0"));
        assert!(msg.contains("new_function"));
    }

    #[test]
    fn test_registry_initialization() {
        // Should not panic
        let _ = get_registry();
    }

    #[test]
    fn test_module_stability() {
        let stability = get_module_stability("quantrs2::core");
        assert!(stability.is_some());

        let stability = stability.expect("quantrs2::core module stability should exist");
        assert_eq!(stability.stability, StabilityLevel::Stable);
    }

    #[test]
    fn test_list_modules() {
        let modules: Vec<_> = list_modules().collect();
        assert!(!modules.is_empty());

        // Check that core modules are present
        let module_names: Vec<_> = modules.iter().map(|m| m.module.as_str()).collect();
        assert!(module_names.contains(&"quantrs2::core"));
        assert!(module_names.contains(&"quantrs2::config"));
    }

    #[test]
    fn test_migration_report() {
        let report = migration_report();
        assert!(report.contains("Migration Report"));
        assert!(report.contains("Module Stability"));
    }

    #[test]
    fn test_deprecation_status_ordering() {
        assert!(DeprecationStatus::Stable < DeprecationStatus::PendingDeprecation);
        assert!(DeprecationStatus::PendingDeprecation < DeprecationStatus::Deprecated);
        assert!(DeprecationStatus::Deprecated < DeprecationStatus::Removed);
    }

    #[test]
    fn test_stability_level_ordering() {
        assert!(StabilityLevel::Experimental < StabilityLevel::Unstable);
        assert!(StabilityLevel::Unstable < StabilityLevel::Stable);
    }
}
