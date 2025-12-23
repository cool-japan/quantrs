//! Result types and state structures for formatter

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Complete formatting result
#[derive(Debug, Clone)]
pub struct FormattingResult {
    /// Formatted circuit code
    pub formatted_circuit: FormattedCircuit,
    /// Statistics
    pub statistics: FormattingStatistics,
    /// Applied changes
    pub changes: Vec<FormattingChange>,
    /// Style information
    pub style_information: StyleInformation,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Formatting duration
    pub duration: Duration,
}

/// Formatted circuit representation
#[derive(Debug, Clone)]
pub struct FormattedCircuit {
    /// Formatted code
    pub code: String,
    /// Line count
    pub line_count: usize,
    /// Character count
    pub char_count: usize,
}

/// Formatting statistics
#[derive(Debug, Clone, Default)]
pub struct FormattingStatistics {
    /// Total lines processed
    pub total_lines: usize,
    /// Lines modified
    pub lines_modified: usize,
    /// Characters added
    pub characters_added: usize,
    /// Characters removed
    pub characters_removed: usize,
    /// Formatting time
    pub formatting_time: Duration,
}

/// Formatting change
#[derive(Debug, Clone)]
pub struct FormattingChange {
    /// Change type
    pub change_type: ChangeType,
    /// Start position
    pub start: Position,
    /// End position
    pub end: Position,
    /// Old text
    pub old_text: String,
    /// New text
    pub new_text: String,
}

/// Change types
#[derive(Debug, Clone)]
pub enum ChangeType {
    Indentation,
    Spacing,
    LineBreak,
    Alignment,
    Comment,
    Organization,
}

/// Position in code
#[derive(Debug, Clone)]
pub struct Position {
    pub line: usize,
    pub column: usize,
}

/// Style information
#[derive(Debug, Clone)]
pub struct StyleInformation {
    /// Applied rules
    pub applied_rules: Vec<String>,
    /// Violations fixed
    pub violations_fixed: Vec<String>,
    /// Compliance score
    pub compliance_score: f64,
    /// Consistency metrics
    pub consistency_metrics: ConsistencyMetrics,
}

/// Consistency metrics
#[derive(Debug, Clone)]
pub struct ConsistencyMetrics {
    pub naming_consistency: f64,
    pub indentation_consistency: f64,
    pub spacing_consistency: f64,
    pub comment_consistency: f64,
    pub overall_consistency: f64,
}

/// Quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Readability score
    pub readability_score: f64,
    /// Maintainability score
    pub maintainability_score: f64,
    /// Complexity score
    pub complexity_score: f64,
    /// Overall quality score
    pub overall_quality: f64,
}

/// Style compliance result
#[derive(Debug, Clone)]
pub struct StyleCompliance {
    /// Compliance level
    pub compliance_level: ComplianceLevel,
    /// Issues
    pub issues: Vec<StyleIssue>,
    /// Overall score
    pub score: f64,
}

/// Compliance levels
#[derive(Debug, Clone)]
pub enum ComplianceLevel {
    Excellent,
    Good,
    Fair,
    Poor,
}

/// Style issue
#[derive(Debug, Clone)]
pub struct StyleIssue {
    /// Issue type
    pub issue_type: String,
    /// Description
    pub description: String,
    /// Severity
    pub severity: String,
    /// Location
    pub location: Option<Position>,
}

/// Code structure for analysis
#[derive(Debug, Clone, Default)]
pub struct CodeStructure {
    /// Sections
    pub sections: Vec<CodeSection>,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Code section
#[derive(Debug, Clone)]
pub struct CodeSection {
    /// Section name
    pub name: String,
    /// Section content
    pub content: String,
    /// Section start line
    pub start_line: usize,
    /// Section end line
    pub end_line: usize,
}

/// Whitespace statistics
#[derive(Debug, Clone)]
pub struct WhitespaceStatistics {
    pub total_whitespace: usize,
    pub indentation_chars: usize,
    pub spacing_chars: usize,
    pub line_breaks: usize,
    pub consistency_score: f64,
}

/// Alignment statistics
#[derive(Debug, Clone)]
pub struct AlignmentStatistics {
    pub total_alignments: usize,
    pub successful_alignments: usize,
    pub average_quality: f64,
    pub consistency_score: f64,
}

/// Whitespace state
#[derive(Debug, Clone)]
pub struct WhitespaceState {
    pub indentation_level: usize,
    pub line_length: usize,
    pub pending_changes: Vec<FormattingChange>,
    pub statistics: WhitespaceStatistics,
}

/// Alignment state
#[derive(Debug, Clone)]
pub struct AlignmentState {
    pub active_alignments: Vec<AlignmentGroup>,
    pub alignment_columns: Vec<usize>,
    pub statistics: AlignmentStatistics,
}

/// Alignment group
#[derive(Debug, Clone)]
pub struct AlignmentGroup {
    pub items: Vec<AlignmentItem>,
    pub column: usize,
}

/// Alignment item
#[derive(Debug, Clone)]
pub struct AlignmentItem {
    pub text: String,
    pub position: Position,
}

/// Whitespace optimization settings
#[derive(Debug, Clone)]
pub struct WhitespaceOptimization {
    pub remove_trailing: bool,
    pub normalize_indentation: bool,
    pub optimize_line_breaks: bool,
    pub compress_empty_lines: bool,
    pub target_compression: f64,
}

/// Alignment optimization settings
#[derive(Debug, Clone)]
pub struct AlignmentOptimization {
    pub auto_detect: bool,
    pub quality_threshold: f64,
    pub max_distance: usize,
    pub prefer_compact: bool,
}

/// Comment formatting state
#[derive(Debug, Clone)]
pub struct CommentFormatterState {
    pub rules: Vec<CommentRule>,
    pub templates: HashMap<String, String>,
    pub quality_threshold: f64,
}

/// Comment rule
#[derive(Debug, Clone)]
pub struct CommentRule {
    pub name: String,
    pub pattern: String,
    pub replacement: String,
}
