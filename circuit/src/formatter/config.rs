//! Configuration types for quantum circuit formatter

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Formatter configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatterConfig {
    /// Maximum line length
    pub max_line_length: usize,
    /// Indentation style
    pub indentation: IndentationConfig,
    /// Spacing configuration
    pub spacing: SpacingConfig,
    /// Alignment settings
    pub alignment: AlignmentConfig,
    /// Comment formatting
    pub comments: CommentConfig,
    /// Code organization
    pub organization: OrganizationConfig,
    /// Optimization settings
    pub optimization: OptimizationConfig,
    /// Style enforcement
    pub style_enforcement: StyleEnforcementConfig,
    /// `SciRS2` analysis integration
    pub scirs2_analysis: SciRS2AnalysisConfig,
    /// Auto-correction settings
    pub auto_correction: AutoCorrectionConfig,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            max_line_length: 100,
            indentation: IndentationConfig::default(),
            spacing: SpacingConfig::default(),
            alignment: AlignmentConfig::default(),
            comments: CommentConfig::default(),
            organization: OrganizationConfig::default(),
            optimization: OptimizationConfig::default(),
            style_enforcement: StyleEnforcementConfig::default(),
            scirs2_analysis: SciRS2AnalysisConfig::default(),
            auto_correction: AutoCorrectionConfig::default(),
        }
    }
}

/// Indentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndentationConfig {
    pub style: IndentationStyle,
    pub spaces_per_level: usize,
    pub tab_size: usize,
    pub continuation_indent: usize,
    pub align_closing_brackets: bool,
}

impl Default for IndentationConfig {
    fn default() -> Self {
        Self {
            style: IndentationStyle::Spaces,
            spaces_per_level: 4,
            tab_size: 4,
            continuation_indent: 4,
            align_closing_brackets: true,
        }
    }
}

/// Indentation styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndentationStyle {
    Spaces,
    Tabs,
    Smart,
}

/// Spacing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacingConfig {
    pub around_operators: bool,
    pub after_commas: bool,
    pub around_parentheses: SpacingStyle,
    pub around_brackets: SpacingStyle,
    pub around_braces: SpacingStyle,
    pub before_function_calls: bool,
    pub in_empty_parentheses: bool,
    pub blank_lines_between_sections: usize,
    pub blank_lines_around_classes: usize,
}

impl Default for SpacingConfig {
    fn default() -> Self {
        Self {
            around_operators: true,
            after_commas: true,
            around_parentheses: SpacingStyle::Outside,
            around_brackets: SpacingStyle::None,
            around_braces: SpacingStyle::Inside,
            before_function_calls: false,
            in_empty_parentheses: false,
            blank_lines_between_sections: 2,
            blank_lines_around_classes: 2,
        }
    }
}

/// Spacing styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpacingStyle {
    None,
    Inside,
    Outside,
    Both,
}

/// Alignment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig {
    pub align_gate_parameters: bool,
    pub align_comments: bool,
    pub align_variable_declarations: bool,
    pub align_circuit_definitions: bool,
    pub column_alignment_threshold: usize,
    pub max_alignment_columns: usize,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            align_gate_parameters: true,
            align_comments: true,
            align_variable_declarations: true,
            align_circuit_definitions: true,
            column_alignment_threshold: 3,
            max_alignment_columns: 10,
        }
    }
}

/// Comment formatting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentConfig {
    pub format_block_comments: bool,
    pub format_inline_comments: bool,
    pub comment_line_length: usize,
    pub comment_alignment: CommentAlignment,
    pub preserve_formatting: bool,
    pub auto_generate_comments: bool,
    pub target_comment_density: f64,
}

impl Default for CommentConfig {
    fn default() -> Self {
        Self {
            format_block_comments: true,
            format_inline_comments: true,
            comment_line_length: 80,
            comment_alignment: CommentAlignment::Left,
            preserve_formatting: false,
            auto_generate_comments: false,
            target_comment_density: 0.2,
        }
    }
}

/// Comment alignment styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentAlignment {
    Left,
    Right,
    Center,
    Column,
}

/// Code organization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationConfig {
    pub group_related_gates: bool,
    pub sort_imports: bool,
    pub organize_functions: bool,
    pub grouping_strategy: GroupingStrategy,
    pub section_ordering: Vec<String>,
    pub enforce_section_separation: bool,
}

impl Default for OrganizationConfig {
    fn default() -> Self {
        Self {
            group_related_gates: true,
            sort_imports: true,
            organize_functions: true,
            grouping_strategy: GroupingStrategy::Logical,
            section_ordering: vec![
                "imports".to_string(),
                "constants".to_string(),
                "variables".to_string(),
                "gates".to_string(),
                "measurements".to_string(),
            ],
            enforce_section_separation: true,
        }
    }
}

/// Grouping strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupingStrategy {
    Logical,
    ByQubit,
    ByGateType,
    ByDepth,
    None,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub optimize_line_breaks: bool,
    pub optimize_whitespace: bool,
    pub minimize_changes: bool,
    pub preserve_structure: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            optimize_line_breaks: true,
            optimize_whitespace: true,
            minimize_changes: false,
            preserve_structure: true,
        }
    }
}

/// Style enforcement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleEnforcementConfig {
    pub enforce_naming_conventions: bool,
    pub enforce_spacing_rules: bool,
    pub enforce_alignment_rules: bool,
    pub severity_level: SeverityLevel,
}

impl Default for StyleEnforcementConfig {
    fn default() -> Self {
        Self {
            enforce_naming_conventions: true,
            enforce_spacing_rules: true,
            enforce_alignment_rules: true,
            severity_level: SeverityLevel::Warning,
        }
    }
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    Info,
    Warning,
    Error,
}

/// `SciRS2` analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciRS2AnalysisConfig {
    pub enable_graph_analysis: bool,
    pub enable_pattern_recognition: bool,
    pub analysis_depth: usize,
}

impl Default for SciRS2AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_graph_analysis: true,
            enable_pattern_recognition: true,
            analysis_depth: 3,
        }
    }
}

/// Auto-correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoCorrectionConfig {
    pub auto_fix_indentation: bool,
    pub auto_fix_spacing: bool,
    pub auto_fix_alignment: bool,
    pub suggest_improvements: bool,
}

impl Default for AutoCorrectionConfig {
    fn default() -> Self {
        Self {
            auto_fix_indentation: true,
            auto_fix_spacing: true,
            auto_fix_alignment: true,
            suggest_improvements: true,
        }
    }
}
