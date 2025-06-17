//! Reporting functionality for the solution debugger.

use super::analysis::{ConstraintAnalysis, EnergyAnalysis};
use super::comparison::ComparisonResult;
use super::types::Solution;
use super::visualization::Visualization;
use serde::Serialize;

/// Debug report
#[derive(Debug, Clone, Serialize)]
pub struct DebugReport {
    /// Solution being debugged
    pub solution: Solution,
    /// Constraint analysis results
    pub constraint_analysis: Option<ConstraintAnalysis>,
    /// Energy analysis results
    pub energy_analysis: Option<EnergyAnalysis>,
    /// Comparison results
    pub comparison_results: Vec<ComparisonResult>,
    /// Generated visualizations
    pub visualizations: Vec<Visualization>,
    /// Identified issues
    pub issues: Vec<Issue>,
    /// Suggested improvements
    pub suggestions: Vec<Suggestion>,
    /// Summary
    pub summary: DebugSummary,
}

/// Debug summary
#[derive(Debug, Clone, Serialize)]
pub struct DebugSummary {
    /// Overall solution score (0.0 to 1.0)
    pub overall_score: f64,
    /// Total energy
    pub total_energy: f64,
    /// Constraint satisfaction rate
    pub constraint_satisfaction_rate: f64,
    /// Number of issues found
    pub total_issues: usize,
    /// Number of critical issues
    pub critical_issues: usize,
    /// Number of suggestions
    pub suggestions_count: usize,
    /// Improvement potential (0.0 to 1.0)
    pub improvement_potential: f64,
    /// Analysis timestamp
    pub timestamp: String,
}

/// Issue identified in solution
#[derive(Debug, Clone, Serialize)]
pub struct Issue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: String,
    /// Issue description
    pub description: String,
    /// Location where issue was found
    pub location: String,
    /// Suggested action to fix
    pub suggested_action: String,
}

#[derive(Debug, Clone, Serialize)]
pub enum IssueSeverity {
    /// Critical issue that prevents solution validity
    Critical,
    /// High priority issue
    High,
    /// Medium priority issue
    Medium,
    /// Low priority issue
    Low,
    /// Informational
    Info,
}

/// Suggestion for improvement
#[derive(Debug, Clone, Serialize)]
pub struct Suggestion {
    /// Suggestion category
    pub category: String,
    /// Description of suggestion
    pub description: String,
    /// Expected impact (higher is better)
    pub impact: f64,
    /// Feasibility of implementing (0.0 to 1.0)
    pub feasibility: f64,
    /// Specific action steps
    pub action_steps: Vec<String>,
}

impl Default for DebugSummary {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            total_energy: 0.0,
            constraint_satisfaction_rate: 0.0,
            total_issues: 0,
            critical_issues: 0,
            suggestions_count: 0,
            improvement_potential: 0.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

impl DebugReport {
    /// Generate text summary of the debug report
    pub fn generate_text_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("=== Solution Debug Report ===\n"));
        summary.push_str(&format!(
            "Overall Score: {:.2}\n",
            self.summary.overall_score
        ));
        summary.push_str(&format!("Total Energy: {:.4}\n", self.summary.total_energy));
        summary.push_str(&format!(
            "Constraint Satisfaction: {:.1}%\n",
            self.summary.constraint_satisfaction_rate * 100.0
        ));
        summary.push_str(&format!(
            "Issues Found: {} (Critical: {})\n",
            self.summary.total_issues, self.summary.critical_issues
        ));
        summary.push_str(&format!(
            "Suggestions: {}\n",
            self.summary.suggestions_count
        ));
        summary.push_str(&format!(
            "Improvement Potential: {:.1}%\n\n",
            self.summary.improvement_potential * 100.0
        ));

        // Add constraint analysis
        if let Some(ref constraint_analysis) = self.constraint_analysis {
            summary.push_str("=== Constraint Analysis ===\n");
            summary.push_str(&format!(
                "Total Constraints: {}\n",
                constraint_analysis.total_constraints
            ));
            summary.push_str(&format!("Satisfied: {}\n", constraint_analysis.satisfied));
            summary.push_str(&format!("Violated: {}\n", constraint_analysis.violated));
            summary.push_str(&format!(
                "Penalty Incurred: {:.4}\n\n",
                constraint_analysis.penalty_incurred
            ));
        }

        // Add energy analysis
        if let Some(ref energy_analysis) = self.energy_analysis {
            summary.push_str("=== Energy Analysis ===\n");
            summary.push_str(&format!(
                "Total Energy: {:.4}\n",
                energy_analysis.total_energy
            ));
            summary.push_str(&format!(
                "Linear Terms: {:.4}\n",
                energy_analysis.breakdown.linear_terms
            ));
            summary.push_str(&format!(
                "Quadratic Terms: {:.4}\n",
                energy_analysis.breakdown.quadratic_terms
            ));
            summary.push_str(&format!(
                "Critical Variables: {}\n",
                energy_analysis.critical_variables.len()
            ));
            summary.push_str(&format!(
                "Critical Interactions: {}\n\n",
                energy_analysis.critical_interactions.len()
            ));
        }

        // Add issues
        if !self.issues.is_empty() {
            summary.push_str("=== Issues Found ===\n");
            for (i, issue) in self.issues.iter().enumerate() {
                summary.push_str(&format!(
                    "{}. [{:?}] {}: {}\n",
                    i + 1,
                    issue.severity,
                    issue.category,
                    issue.description
                ));
            }
            summary.push('\n');
        }

        // Add suggestions
        if !self.suggestions.is_empty() {
            summary.push_str("=== Suggestions ===\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                summary.push_str(&format!(
                    "{}. {}: {} (Impact: {:.2}, Feasibility: {:.2})\n",
                    i + 1,
                    suggestion.category,
                    suggestion.description,
                    suggestion.impact,
                    suggestion.feasibility
                ));
            }
            summary.push('\n');
        }

        summary
    }

    /// Generate JSON report
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Save report to file
    pub fn save_to_file(
        &self,
        filename: &str,
        format: &super::config::DebugOutputFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let content = match format {
            super::config::DebugOutputFormat::Json => self.to_json()?,
            super::config::DebugOutputFormat::Console => self.generate_text_summary(),
            super::config::DebugOutputFormat::Html => self.generate_html_report(),
            super::config::DebugOutputFormat::Markdown => self.generate_markdown_report(),
        };

        let mut file = std::fs::File::create(filename)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    /// Generate HTML report
    fn generate_html_report(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Solution Debug Report</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str(".summary { background: #f0f0f0; padding: 10px; border-radius: 5px; }\n");
        html.push_str(
            ".issue { margin: 10px 0; padding: 10px; border-left: 4px solid #ff0000; }\n",
        );
        html.push_str(
            ".suggestion { margin: 10px 0; padding: 10px; border-left: 4px solid #00aa00; }\n",
        );
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str("<h1>Solution Debug Report</h1>\n");

        // Summary section
        html.push_str("<div class='summary'>\n");
        html.push_str(&format!("<h2>Summary</h2>\n"));
        html.push_str(&format!(
            "<p><strong>Overall Score:</strong> {:.2}</p>\n",
            self.summary.overall_score
        ));
        html.push_str(&format!(
            "<p><strong>Total Energy:</strong> {:.4}</p>\n",
            self.summary.total_energy
        ));
        html.push_str(&format!(
            "<p><strong>Constraint Satisfaction:</strong> {:.1}%</p>\n",
            self.summary.constraint_satisfaction_rate * 100.0
        ));
        html.push_str("</div>\n");

        // Issues section
        if !self.issues.is_empty() {
            html.push_str("<h2>Issues</h2>\n");
            for issue in &self.issues {
                html.push_str("<div class='issue'>\n");
                html.push_str(&format!(
                    "<h3>{:?}: {}</h3>\n",
                    issue.severity, issue.category
                ));
                html.push_str(&format!("<p>{}</p>\n", issue.description));
                html.push_str(&format!(
                    "<p><em>Suggested Action:</em> {}</p>\n",
                    issue.suggested_action
                ));
                html.push_str("</div>\n");
            }
        }

        // Suggestions section
        if !self.suggestions.is_empty() {
            html.push_str("<h2>Suggestions</h2>\n");
            for suggestion in &self.suggestions {
                html.push_str("<div class='suggestion'>\n");
                html.push_str(&format!("<h3>{}</h3>\n", suggestion.category));
                html.push_str(&format!("<p>{}</p>\n", suggestion.description));
                html.push_str(&format!(
                    "<p><em>Impact:</em> {:.2}, <em>Feasibility:</em> {:.2}</p>\n",
                    suggestion.impact, suggestion.feasibility
                ));
                html.push_str("</div>\n");
            }
        }

        html.push_str("</body>\n</html>");
        html
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self) -> String {
        let mut md = String::new();

        md.push_str("# Solution Debug Report\n\n");

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "- **Overall Score:** {:.2}\n",
            self.summary.overall_score
        ));
        md.push_str(&format!(
            "- **Total Energy:** {:.4}\n",
            self.summary.total_energy
        ));
        md.push_str(&format!(
            "- **Constraint Satisfaction:** {:.1}%\n",
            self.summary.constraint_satisfaction_rate * 100.0
        ));
        md.push_str(&format!(
            "- **Issues Found:** {} (Critical: {})\n",
            self.summary.total_issues, self.summary.critical_issues
        ));
        md.push_str(&format!(
            "- **Suggestions:** {}\n",
            self.summary.suggestions_count
        ));
        md.push_str(&format!(
            "- **Improvement Potential:** {:.1}%\n\n",
            self.summary.improvement_potential * 100.0
        ));

        // Issues
        if !self.issues.is_empty() {
            md.push_str("## Issues\n\n");
            for (i, issue) in self.issues.iter().enumerate() {
                md.push_str(&format!(
                    "### {}. [{:?}] {}\n\n",
                    i + 1,
                    issue.severity,
                    issue.category
                ));
                md.push_str(&format!("{}\n\n", issue.description));
                md.push_str(&format!(
                    "**Suggested Action:** {}\n\n",
                    issue.suggested_action
                ));
            }
        }

        // Suggestions
        if !self.suggestions.is_empty() {
            md.push_str("## Suggestions\n\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                md.push_str(&format!("### {}. {}\n\n", i + 1, suggestion.category));
                md.push_str(&format!("{}\n\n", suggestion.description));
                md.push_str(&format!("- **Impact:** {:.2}\n", suggestion.impact));
                md.push_str(&format!(
                    "- **Feasibility:** {:.2}\n\n",
                    suggestion.feasibility
                ));
            }
        }

        md
    }
}
