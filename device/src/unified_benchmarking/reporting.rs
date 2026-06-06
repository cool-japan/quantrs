//! Report generator for automated reporting

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

use super::config::ExportDestination;

/// Report generator for automated reporting
pub struct ReportGenerator {
    report_templates: HashMap<String, ReportTemplate>,
    export_handlers: HashMap<ExportDestination, Box<dyn ExportHandler + Send + Sync>>,
    visualization_engine: VisualizationEngine,
}

pub trait ExportHandler {
    fn export(&self, report: &GeneratedReport) -> Result<String, String>;
}

#[derive(Debug, Clone)]
pub struct ReportTemplate {
    pub template_id: String,
    pub template_name: String,
    pub sections: Vec<ReportSection>,
    pub styling: ReportStyling,
}

#[derive(Debug, Clone)]
pub struct ReportSection {
    pub section_id: String,
    pub title: String,
    pub content_type: SectionContentType,
    pub data_queries: Vec<DataQuery>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionContentType {
    Text,
    Table,
    Chart,
    Statistical,
    Comparison,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct DataQuery {
    pub query_id: String,
    pub query_type: QueryType,
    pub filters: HashMap<String, String>,
    pub aggregations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    PerformanceMetrics,
    CostAnalysis,
    TrendAnalysis,
    Comparison,
    Statistical,
}

#[derive(Debug, Clone)]
pub struct ReportStyling {
    pub theme: String,
    pub color_scheme: Vec<String>,
    pub font_family: String,
    pub custom_css: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedReport {
    pub report_id: String,
    pub report_type: String,
    pub generation_time: SystemTime,
    pub content: ReportContent,
    pub metadata: ReportMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportContent {
    pub sections: Vec<ReportSectionContent>,
    pub attachments: Vec<ReportAttachment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSectionContent {
    pub section_id: String,
    pub title: String,
    pub content: String,
    pub visualizations: Vec<VisualizationData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportAttachment {
    pub attachment_id: String,
    pub filename: String,
    pub content_type: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub report_id: String,
    pub title: String,
    pub description: String,
    pub author: String,
    pub keywords: Vec<String>,
    pub data_sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub visualization_id: String,
    pub visualization_type: String,
    pub data: Vec<u8>,
    pub format: String,
}

#[derive(Debug, Clone)]
pub struct VisualizationEngine {
    pub chart_library: String,
    pub default_width: u32,
    pub default_height: u32,
    pub color_palette: Vec<String>,
    pub font_size: f32,
    pub marker_size: f32,
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportGenerator {
    pub fn new() -> Self {
        Self {
            report_templates: HashMap::new(),
            export_handlers: HashMap::new(),
            visualization_engine: VisualizationEngine {
                chart_library: "plotters".to_string(),
                default_width: 800,
                default_height: 600,
                color_palette: vec![
                    "#1f77b4".to_string(),
                    "#ff7f0e".to_string(),
                    "#2ca02c".to_string(),
                    "#d62728".to_string(),
                ],
                font_size: 12.0,
                marker_size: 5.0,
            },
        }
    }

    /// Generate a benchmark report from a `UnifiedBenchmarkResult`.
    ///
    /// Returns a `GeneratedReport` containing a Markdown-formatted summary of
    /// the benchmark run.  The report includes a header section, a per-platform
    /// performance summary table, timing statistics, and a list of optimisation
    /// recommendations.
    pub fn generate_report(
        &self,
        result: &super::results::UnifiedBenchmarkResult,
    ) -> GeneratedReport {
        let report_id = format!("report_{}", result.execution_id);
        let mut sections = Vec::new();

        // ── Section 1: header / overview ────────────────────────────────────
        let overview_md = format!(
            "# Unified Quantum Benchmark Report\n\n\
             **Execution ID:** `{}`  \n\
             **Platforms tested:** {}  \n\
             **Benchmarks executed:** {}  \n\
             **Total duration:** {:.3}s\n",
            result.execution_id,
            result.execution_metadata.platforms_tested.len(),
            result.execution_metadata.benchmarks_executed,
            result.execution_metadata.total_duration.as_secs_f64(),
        );
        sections.push(ReportSectionContent {
            section_id: "overview".to_string(),
            title: "Overview".to_string(),
            content: overview_md,
            visualizations: vec![],
        });

        // ── Section 2: per-platform performance table ────────────────────────
        let mut table_md = String::from(
            "## Platform Performance Summary\n\n\
             | Platform | Fidelity | Error Rate | Throughput | Availability |\n\
             |---|---|---|---|---|\n",
        );
        for (platform, pr) in &result.platform_results {
            let m = &pr.performance_metrics;
            table_md.push_str(&Self::generate_markdown_table_row(
                &format!("{platform:?}"),
                m,
            ));
        }
        sections.push(ReportSectionContent {
            section_id: "platform_summary".to_string(),
            title: "Platform Summary".to_string(),
            content: table_md,
            visualizations: vec![],
        });

        // ── Section 3: cross-platform analysis ───────────────────────────────
        let mut cpa_md = String::from(
            "## Cross-Platform Analysis\n\n| Platform | Composite Score |\n|---|---|\n",
        );
        for (label, score) in &result.cross_platform_analysis.platform_comparison {
            cpa_md.push_str(&format!("| {label} | {score:.4} |\n"));
        }
        if let Some((metric, platform)) = result
            .cross_platform_analysis
            .best_platform_per_metric
            .iter()
            .next()
        {
            cpa_md.push_str(&format!(
                "\nBest platform for **{metric}**: `{platform:?}`\n"
            ));
        }
        sections.push(ReportSectionContent {
            section_id: "cross_platform".to_string(),
            title: "Cross-Platform Analysis".to_string(),
            content: cpa_md,
            visualizations: vec![],
        });

        // ── Section 4: optimisation recommendations ──────────────────────────
        let mut rec_md = String::from("## Optimisation Recommendations\n\n");
        for (i, rec) in result.optimization_recommendations.iter().enumerate() {
            rec_md.push_str(&format!(
                "{}. **{}** (priority {}): {}  \n   Expected improvement: {:.4}  \n   Effort: {}\n\n",
                i + 1,
                rec.recommendation_type,
                rec.priority,
                rec.description,
                rec.expected_improvement,
                rec.implementation_effort,
            ));
        }
        sections.push(ReportSectionContent {
            section_id: "recommendations".to_string(),
            title: "Optimisation Recommendations".to_string(),
            content: rec_md,
            visualizations: vec![],
        });

        // ── Section 5: resource and cost summary ─────────────────────────────
        let cost_md = format!(
            "## Cost Analysis\n\n\
             **Total estimated cost:** ${:.4}  \n\
             **Potential savings:** ${:.4}\n",
            result.cost_analysis.total_cost,
            result.cost_analysis.cost_optimization.potential_savings,
        );
        sections.push(ReportSectionContent {
            section_id: "cost".to_string(),
            title: "Cost Analysis".to_string(),
            content: cost_md,
            visualizations: vec![],
        });

        GeneratedReport {
            report_id: report_id.clone(),
            report_type: "unified_benchmark".to_string(),
            generation_time: std::time::SystemTime::now(),
            content: ReportContent {
                sections,
                attachments: vec![],
            },
            metadata: ReportMetadata {
                report_id,
                title: format!("Benchmark Report – {}", result.execution_id),
                description: "Automated unified quantum benchmark report".to_string(),
                author: "quantrs2-device".to_string(),
                keywords: vec!["quantum".to_string(), "benchmark".to_string()],
                data_sources: vec![result.execution_id.clone()],
            },
        }
    }

    /// Build a Markdown table row from a single platform label and its metrics.
    fn generate_markdown_table_row(
        label: &str,
        m: &super::results::PlatformPerformanceMetrics,
    ) -> String {
        format!(
            "| {} | {:.4} | {:.4} | {:.2} ops/s | {:.3} |\n",
            label, m.overall_fidelity, m.error_rate, m.throughput, m.availability,
        )
    }

    /// Convenience wrapper: render the full Markdown text from a report.
    pub fn report_to_markdown(report: &GeneratedReport) -> String {
        report
            .content
            .sections
            .iter()
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_benchmarking::results::PlatformPerformanceMetrics;
    use std::time::Duration;

    #[test]
    fn test_report_generator_new_is_default() {
        let g1 = ReportGenerator::new();
        let g2 = ReportGenerator::default();
        assert_eq!(
            g1.visualization_engine.chart_library,
            g2.visualization_engine.chart_library
        );
    }

    #[test]
    fn test_generate_markdown_table_row_format() {
        let m = PlatformPerformanceMetrics {
            overall_fidelity: 0.99,
            average_execution_time: Duration::from_millis(50),
            throughput: 100.0,
            error_rate: 0.01,
            availability: 0.995,
        };
        let row = ReportGenerator::generate_markdown_table_row("TestPlatform", &m);
        assert!(
            row.contains("0.9900"),
            "row should contain formatted fidelity"
        );
        assert!(
            row.contains("TestPlatform"),
            "row should contain platform label"
        );
    }

    #[test]
    fn test_report_to_markdown_combines_sections() {
        let report = GeneratedReport {
            report_id: "test_r1".to_string(),
            report_type: "test".to_string(),
            generation_time: std::time::SystemTime::UNIX_EPOCH,
            content: ReportContent {
                sections: vec![
                    ReportSectionContent {
                        section_id: "s1".to_string(),
                        title: "Section 1".to_string(),
                        content: "Hello world".to_string(),
                        visualizations: vec![],
                    },
                    ReportSectionContent {
                        section_id: "s2".to_string(),
                        title: "Section 2".to_string(),
                        content: "Second part".to_string(),
                        visualizations: vec![],
                    },
                ],
                attachments: vec![],
            },
            metadata: ReportMetadata {
                report_id: "test_r1".to_string(),
                title: "Test".to_string(),
                description: "desc".to_string(),
                author: "test".to_string(),
                keywords: vec![],
                data_sources: vec![],
            },
        };
        let md = ReportGenerator::report_to_markdown(&report);
        assert!(
            md.contains("Hello world"),
            "combined markdown must include section 1"
        );
        assert!(
            md.contains("Second part"),
            "combined markdown must include section 2"
        );
    }
}
