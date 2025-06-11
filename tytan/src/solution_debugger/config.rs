//! Configuration types for the solution debugger.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct DebuggerConfig {
    /// Enable detailed analysis
    pub detailed_analysis: bool,
    /// Check constraint violations
    pub check_constraints: bool,
    /// Analyze energy breakdown
    pub analyze_energy: bool,
    /// Compare with known solutions
    pub compare_solutions: bool,
    /// Generate visualizations
    pub generate_visuals: bool,
    /// Output format
    pub output_format: DebugOutputFormat,
    /// Verbosity level
    pub verbosity: VerbosityLevel,
}

#[derive(Debug, Clone, Serialize)]
pub enum DebugOutputFormat {
    /// Console output
    Console,
    /// HTML report
    Html,
    /// JSON data
    Json,
    /// Markdown report
    Markdown,
}

#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq, Serialize)]
pub enum VerbosityLevel {
    /// Minimal output
    Minimal,
    /// Normal output
    Normal,
    /// Detailed output
    Detailed,
    /// Debug-level output
    Debug,
}

impl Default for DebuggerConfig {
    fn default() -> Self {
        Self {
            detailed_analysis: true,
            check_constraints: true,
            analyze_energy: true,
            compare_solutions: false,
            generate_visuals: false,
            output_format: DebugOutputFormat::Console,
            verbosity: VerbosityLevel::Normal,
        }
    }
}
