//! Test reporting and report generation

use std::collections::HashMap;
use std::fs;
use std::time::SystemTime;

use super::config::ReportFormat;
use super::results::TestOutcome;

use std::fmt::Write;

/// Real pass/fail/skip/timeout/error counts computed from `ReportData::test_results`.
#[derive(Debug, Clone, Copy, Default)]
struct TestOutcomeCounts {
    passed: usize,
    failed: usize,
    skipped: usize,
    timeout: usize,
    error: usize,
}

impl TestOutcomeCounts {
    fn from_data(data: &ReportData) -> Self {
        let mut counts = Self::default();
        for result in &data.test_results {
            match result.outcome {
                TestOutcome::Passed => counts.passed += 1,
                TestOutcome::Failed => counts.failed += 1,
                TestOutcome::Skipped => counts.skipped += 1,
                TestOutcome::Timeout => counts.timeout += 1,
                TestOutcome::Error => counts.error += 1,
            }
        }
        counts
    }

    const fn total(&self) -> usize {
        self.passed + self.failed + self.skipped + self.timeout + self.error
    }
}
/// Test report generator
pub struct TestReportGenerator {
    /// Report templates
    pub templates: HashMap<String, ReportTemplate>,
    /// Generated reports
    pub generated_reports: Vec<GeneratedReport>,
    /// Report configuration
    pub config: super::config::ReportingConfig,
}

impl TestReportGenerator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            generated_reports: vec![],
            config: super::config::ReportingConfig::default(),
        }
    }

    /// Register a report template
    pub fn register_template(&mut self, template: ReportTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Generate a report from a template
    pub fn generate_report(
        &mut self,
        template_name: &str,
        data: &ReportData,
    ) -> Result<GeneratedReport, String> {
        let template = self
            .templates
            .get(template_name)
            .ok_or_else(|| format!("Template '{template_name}' not found"))?;

        // Generate report content based on format
        let content = match template.format {
            ReportFormat::HTML => self.generate_html_report(template, data)?,
            ReportFormat::JSON => self.generate_json_report(template, data)?,
            ReportFormat::XML => self.generate_xml_report(template, data)?,
            ReportFormat::PDF => self.generate_pdf_report(template, data)?,
            ReportFormat::CSV => self.generate_csv_report(template, data)?,
        };

        let report = GeneratedReport {
            id: format!(
                "report_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("system time before UNIX_EPOCH")
                    .as_secs()
            ),
            name: template.name.clone(),
            format: template.format.clone(),
            generated_at: SystemTime::now(),
            content: content.clone(),
            metadata: template.metadata.clone(),
            size: content.len(),
        };

        self.generated_reports.push(report.clone());
        Ok(report)
    }

    /// Generate HTML format report
    fn generate_html_report(
        &self,
        template: &ReportTemplate,
        data: &ReportData,
    ) -> Result<String, String> {
        let mut html = String::from("<html><head><title>");
        html.push_str(&template.metadata.title);
        html.push_str("</title></head><body>");

        for section in &template.sections {
            write!(html, "<h2>{}</h2>", section.name).expect("failed to write to string");
            match &section.content {
                SectionContent::Text(text) => {
                    write!(html, "<p>{text}</p>").expect("failed to write to string");
                }
                SectionContent::Table(_) => {
                    let counts = TestOutcomeCounts::from_data(data);
                    html.push_str("<table><tr><th>Metric</th><th>Value</th></tr>");
                    write!(
                        html,
                        "<tr><td>Tests Passed</td><td>{}</td></tr>",
                        counts.passed
                    )
                    .expect("failed to write to string");
                    write!(
                        html,
                        "<tr><td>Tests Failed</td><td>{}</td></tr>",
                        counts.failed
                    )
                    .expect("failed to write to string");
                    write!(
                        html,
                        "<tr><td>Tests Skipped</td><td>{}</td></tr>",
                        counts.skipped
                    )
                    .expect("failed to write to string");
                    write!(
                        html,
                        "<tr><td>Tests Timed Out</td><td>{}</td></tr>",
                        counts.timeout
                    )
                    .expect("failed to write to string");
                    write!(
                        html,
                        "<tr><td>Tests Errored</td><td>{}</td></tr>",
                        counts.error
                    )
                    .expect("failed to write to string");
                    write!(
                        html,
                        "<tr><td>Total Tests</td><td>{}</td></tr>",
                        counts.total()
                    )
                    .expect("failed to write to string");
                    for (metric_name, metric_value) in &data.performance_metrics {
                        write!(
                            html,
                            "<tr><td>{metric_name}</td><td>{metric_value}</td></tr>"
                        )
                        .expect("failed to write to string");
                    }
                    html.push_str("</table>");
                }
                _ => {
                    html.push_str("<p>Content not implemented</p>");
                }
            }
        }

        html.push_str("</body></html>");
        Ok(html)
    }

    /// Generate JSON format report
    fn generate_json_report(
        &self,
        template: &ReportTemplate,
        data: &ReportData,
    ) -> Result<String, String> {
        let counts = TestOutcomeCounts::from_data(data);
        let value = serde_json::json!({
            "title": template.metadata.title,
            "description": template.metadata.description,
            "summary": {
                "tests_passed": counts.passed,
                "tests_failed": counts.failed,
                "tests_skipped": counts.skipped,
                "tests_timeout": counts.timeout,
                "tests_error": counts.error,
                "total_tests": counts.total(),
            },
            "performance_metrics": data.performance_metrics,
            "additional_data": data.additional_data,
        });
        serde_json::to_string(&value).map_err(|e| format!("failed to serialize JSON report: {e}"))
    }

    /// Generate XML format report
    fn generate_xml_report(
        &self,
        template: &ReportTemplate,
        data: &ReportData,
    ) -> Result<String, String> {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        write!(xml, "<report title=\"{}\">\n", template.metadata.title)
            .expect("failed to write to string");
        write!(
            xml,
            "  <description>{}</description>\n",
            template.metadata.description
        )
        .expect("failed to write to string");

        for section in &template.sections {
            write!(xml, "  <section name=\"{}\">\n", section.name)
                .expect("failed to write to string");
            match &section.content {
                SectionContent::Text(text) => {
                    writeln!(xml, "    <content>{text}</content>")
                        .expect("failed to write to string");
                }
                SectionContent::Table(_) => {
                    let counts = TestOutcomeCounts::from_data(data);
                    xml.push_str("    <table>\n");
                    writeln!(
                        xml,
                        "      <row><cell>Tests Passed</cell><cell>{}</cell></row>",
                        counts.passed
                    )
                    .expect("failed to write to string");
                    writeln!(
                        xml,
                        "      <row><cell>Tests Failed</cell><cell>{}</cell></row>",
                        counts.failed
                    )
                    .expect("failed to write to string");
                    writeln!(
                        xml,
                        "      <row><cell>Tests Skipped</cell><cell>{}</cell></row>",
                        counts.skipped
                    )
                    .expect("failed to write to string");
                    writeln!(
                        xml,
                        "      <row><cell>Tests Timed Out</cell><cell>{}</cell></row>",
                        counts.timeout
                    )
                    .expect("failed to write to string");
                    writeln!(
                        xml,
                        "      <row><cell>Tests Errored</cell><cell>{}</cell></row>",
                        counts.error
                    )
                    .expect("failed to write to string");
                    writeln!(
                        xml,
                        "      <row><cell>Total Tests</cell><cell>{}</cell></row>",
                        counts.total()
                    )
                    .expect("failed to write to string");
                    xml.push_str("    </table>\n");
                }
                _ => {
                    xml.push_str("    <content>Content not implemented</content>\n");
                }
            }
            xml.push_str("  </section>\n");
        }

        xml.push_str("</report>");
        Ok(xml)
    }

    /// Generate PDF format report (placeholder)
    fn generate_pdf_report(
        &self,
        template: &ReportTemplate,
        _data: &ReportData,
    ) -> Result<String, String> {
        Ok(format!("PDF Report: {}", template.metadata.title))
    }

    /// Generate CSV format report
    fn generate_csv_report(
        &self,
        _template: &ReportTemplate,
        data: &ReportData,
    ) -> Result<String, String> {
        let counts = TestOutcomeCounts::from_data(data);
        let mut csv = String::from("Metric,Value\n");
        writeln!(csv, "Tests Passed,{}", counts.passed).expect("failed to write to string");
        writeln!(csv, "Tests Failed,{}", counts.failed).expect("failed to write to string");
        writeln!(csv, "Tests Skipped,{}", counts.skipped).expect("failed to write to string");
        writeln!(csv, "Tests Timed Out,{}", counts.timeout).expect("failed to write to string");
        writeln!(csv, "Tests Errored,{}", counts.error).expect("failed to write to string");
        writeln!(csv, "Total Tests,{}", counts.total()).expect("failed to write to string");
        for (metric_name, metric_value) in &data.performance_metrics {
            writeln!(csv, "{metric_name},{metric_value}").expect("failed to write to string");
        }
        Ok(csv)
    }

    /// Get a generated report by ID
    #[must_use]
    pub fn get_report(&self, report_id: &str) -> Option<&GeneratedReport> {
        self.generated_reports.iter().find(|r| r.id == report_id)
    }

    /// List all generated reports
    #[must_use]
    pub fn list_reports(&self) -> Vec<&GeneratedReport> {
        self.generated_reports.iter().collect()
    }

    /// Export a previously generated report to disk at `file_path`.
    ///
    /// The report's already-rendered `content` (HTML/JSON/XML/CSV, or the
    /// textual PDF placeholder) is written verbatim to the given path.
    /// Returns an error if the report id is unknown or if the write fails.
    pub fn export_report(&self, report_id: &str, file_path: &str) -> Result<(), String> {
        let report = self
            .get_report(report_id)
            .ok_or_else(|| format!("Report {report_id} not found"))?;
        fs::write(file_path, &report.content)
            .map_err(|e| format!("failed to write report to '{file_path}': {e}"))
    }

    /// Clear all generated reports
    pub fn clear_reports(&mut self) {
        self.generated_reports.clear();
    }

    /// Get report count
    #[must_use]
    pub fn report_count(&self) -> usize {
        self.generated_reports.len()
    }
}

/// Report data container
#[derive(Debug, Clone)]
pub struct ReportData {
    /// Test results
    pub test_results: Vec<super::results::IntegrationTestResult>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Additional data
    pub additional_data: HashMap<String, String>,
}

/// Report template
#[derive(Debug, Clone)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,
    /// Template format
    pub format: ReportFormat,
    /// Template sections
    pub sections: Vec<ReportSection>,
    /// Template metadata
    pub metadata: ReportMetadata,
}

/// Report section
#[derive(Debug, Clone)]
pub struct ReportSection {
    /// Section name
    pub name: String,
    /// Section type
    pub section_type: SectionType,
    /// Section content
    pub content: SectionContent,
    /// Section formatting
    pub formatting: SectionFormatting,
}

/// Section types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionType {
    Summary,
    TestResults,
    PerformanceMetrics,
    ErrorAnalysis,
    Recommendations,
    Custom(String),
}

/// Section content
#[derive(Debug, Clone)]
pub enum SectionContent {
    /// Static text
    Text(String),
    /// Dynamic data
    Data(DataQuery),
    /// Chart/visualization
    Chart(ChartDefinition),
    /// Table
    Table(TableDefinition),
    /// Custom content
    Custom(String),
}

/// Data query for dynamic content
#[derive(Debug, Clone)]
pub struct DataQuery {
    /// Query type
    pub query_type: QueryType,
    /// Query parameters
    pub parameters: HashMap<String, String>,
    /// Data transformation
    pub transformation: Option<DataTransformation>,
}

/// Query types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    TestResults,
    PerformanceMetrics,
    ErrorCounts,
    TrendData,
    ComparisonData,
    Custom(String),
}

/// Data transformation
#[derive(Debug, Clone)]
pub struct DataTransformation {
    /// Transformation type
    pub transformation_type: TransformationType,
    /// Transformation parameters
    pub parameters: HashMap<String, String>,
}

/// Transformation types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformationType {
    Aggregate,
    Filter,
    Sort,
    Group,
    Calculate,
    Custom(String),
}

/// Chart definition
#[derive(Debug, Clone)]
pub struct ChartDefinition {
    /// Chart type
    pub chart_type: ChartType,
    /// Chart data source
    pub data_source: DataQuery,
    /// Chart configuration
    pub configuration: ChartConfiguration,
}

/// Chart types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Scatter,
    Histogram,
    Heatmap,
    Custom(String),
}

/// Chart configuration
#[derive(Debug, Clone)]
pub struct ChartConfiguration {
    /// Chart title
    pub title: String,
    /// X-axis label
    pub x_axis_label: String,
    /// Y-axis label
    pub y_axis_label: String,
    /// Chart dimensions
    pub dimensions: (u32, u32),
    /// Color scheme
    pub color_scheme: Vec<String>,
}

/// Table definition
#[derive(Debug, Clone)]
pub struct TableDefinition {
    /// Table columns
    pub columns: Vec<TableColumn>,
    /// Table data source
    pub data_source: DataQuery,
    /// Table formatting
    pub formatting: TableFormatting,
}

/// Table column
#[derive(Debug, Clone)]
pub struct TableColumn {
    /// Column name
    pub name: String,
    /// Column type
    pub column_type: ColumnType,
    /// Column formatting
    pub formatting: ColumnFormatting,
}

/// Column types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnType {
    Text,
    Number,
    DateTime,
    Boolean,
    Duration,
    Custom(String),
}

/// Column formatting
#[derive(Debug, Clone)]
pub struct ColumnFormatting {
    /// Number format
    pub number_format: Option<NumberFormat>,
    /// Date format
    pub date_format: Option<String>,
    /// Text alignment
    pub alignment: TextAlignment,
}

/// Number formatting
#[derive(Debug, Clone)]
pub struct NumberFormat {
    /// Decimal places
    pub decimal_places: usize,
    /// Use thousands separator
    pub thousands_separator: bool,
    /// Unit suffix
    pub unit: Option<String>,
}

/// Text alignment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
}

/// Table formatting
#[derive(Debug, Clone)]
pub struct TableFormatting {
    /// Show headers
    pub show_headers: bool,
    /// Alternate row colors
    pub alternate_rows: bool,
    /// Border style
    pub border_style: BorderStyle,
}

/// Border styles
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BorderStyle {
    None,
    Simple,
    Double,
    Rounded,
    Custom(String),
}

/// Section formatting
#[derive(Debug, Clone)]
pub struct SectionFormatting {
    /// Font size
    pub font_size: u8,
    /// Font weight
    pub font_weight: FontWeight,
    /// Text color
    pub text_color: String,
    /// Background color
    pub background_color: Option<String>,
    /// Padding
    pub padding: (u8, u8, u8, u8),
}

/// Font weights
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
    ExtraBold,
}

/// Generated report
#[derive(Debug, Clone)]
pub struct GeneratedReport {
    /// Report ID
    pub id: String,
    /// Report name
    pub name: String,
    /// Report format
    pub format: ReportFormat,
    /// Generation timestamp
    pub generated_at: SystemTime,
    /// Report content
    pub content: String,
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Report size
    pub size: usize,
}

/// Report metadata
#[derive(Debug, Clone)]
pub struct ReportMetadata {
    /// Report title
    pub title: String,
    /// Report description
    pub description: String,
    /// Report author
    pub author: String,
    /// Report version
    pub version: String,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::super::results::{
        IntegrationTestResult, PerformanceMetrics, ValidationResults, ValidationStatus,
        ValidationSummary,
    };
    use super::*;
    use std::time::Duration;

    fn make_result(outcome: TestOutcome) -> IntegrationTestResult {
        IntegrationTestResult {
            test_case_id: "case".to_string(),
            timestamp: SystemTime::now(),
            outcome,
            performance_metrics: PerformanceMetrics {
                execution_duration: Duration::from_secs(1),
                setup_duration: Duration::from_millis(10),
                cleanup_duration: Duration::from_millis(5),
                peak_memory_usage: 1024,
                avg_cpu_usage: 0.5,
                custom_metrics: HashMap::new(),
            },
            validation_results: ValidationResults {
                status: ValidationStatus::Passed,
                validations: vec![],
                summary: ValidationSummary {
                    total: 1,
                    passed: 1,
                    failed: 0,
                    skipped: 0,
                },
            },
            error_info: None,
            artifacts: vec![],
        }
    }

    fn make_data() -> ReportData {
        ReportData {
            test_results: vec![
                make_result(TestOutcome::Passed),
                make_result(TestOutcome::Passed),
                make_result(TestOutcome::Failed),
                make_result(TestOutcome::Skipped),
            ],
            performance_metrics: HashMap::new(),
            additional_data: HashMap::new(),
        }
    }

    fn make_template(format: ReportFormat) -> ReportTemplate {
        ReportTemplate {
            name: "regression_template".to_string(),
            format,
            sections: vec![ReportSection {
                name: "Summary".to_string(),
                section_type: SectionType::Summary,
                content: SectionContent::Table(TableDefinition {
                    columns: vec![],
                    data_source: DataQuery {
                        query_type: QueryType::TestResults,
                        parameters: HashMap::new(),
                        transformation: None,
                    },
                    formatting: TableFormatting {
                        show_headers: true,
                        alternate_rows: false,
                        border_style: BorderStyle::Simple,
                    },
                }),
                formatting: SectionFormatting {
                    font_size: 12,
                    font_weight: FontWeight::Normal,
                    text_color: "#000".to_string(),
                    background_color: None,
                    padding: (0, 0, 0, 0),
                },
            }],
            metadata: ReportMetadata {
                title: "Regression Report".to_string(),
                description: "test".to_string(),
                author: "quantrs2".to_string(),
                version: "1".to_string(),
                custom: HashMap::new(),
            },
        }
    }

    #[test]
    fn html_report_reflects_real_outcome_counts() {
        let mut generator = TestReportGenerator::new();
        generator.register_template(make_template(ReportFormat::HTML));
        let data = make_data();
        let report = generator
            .generate_report("regression_template", &data)
            .expect("report generation should succeed");
        assert!(report.content.contains("<td>Tests Passed</td><td>2</td>"));
        assert!(report.content.contains("<td>Tests Failed</td><td>1</td>"));
        assert!(report.content.contains("<td>Tests Skipped</td><td>1</td>"));
        assert!(report.content.contains("<td>Total Tests</td><td>4</td>"));
    }

    #[test]
    fn csv_report_reflects_real_outcome_counts() {
        let mut generator = TestReportGenerator::new();
        generator.register_template(make_template(ReportFormat::CSV));
        let data = make_data();
        let report = generator
            .generate_report("regression_template", &data)
            .expect("report generation should succeed");
        assert!(report.content.contains("Tests Passed,2"));
        assert!(report.content.contains("Tests Failed,1"));
        assert!(report.content.contains("Total Tests,4"));
    }

    #[test]
    fn export_report_actually_writes_the_file() {
        let mut generator = TestReportGenerator::new();
        generator.register_template(make_template(ReportFormat::CSV));
        let data = make_data();
        let report = generator
            .generate_report("regression_template", &data)
            .expect("report generation should succeed");

        let mut path = std::env::temp_dir();
        path.push(format!(
            "quantrs2_anneal_export_report_test_{}.csv",
            report.id
        ));
        let path_str = path.to_str().expect("path should be valid UTF-8");

        generator
            .export_report(&report.id, path_str)
            .expect("export should succeed");

        let written = std::fs::read_to_string(&path).expect("exported file should exist");
        assert_eq!(written, report.content);

        std::fs::remove_file(&path).expect("cleanup should succeed");
    }

    #[test]
    fn export_report_errors_on_unknown_report_id() {
        let generator = TestReportGenerator::new();
        let mut path = std::env::temp_dir();
        path.push("quantrs2_anneal_export_report_missing.csv");
        let result = generator.export_report("does-not-exist", path.to_str().unwrap());
        assert!(result.is_err());
    }
}
