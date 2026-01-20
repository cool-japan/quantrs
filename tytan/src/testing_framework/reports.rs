//! Report generation for test results.
//!
//! This module provides functions for generating test reports in various formats
//! including Text, JSON, HTML, Markdown, and CSV.

use std::fmt::Write;
use std::fs::File;
use std::io::Write as IoWrite;

use super::config::ReportFormat;
use super::results::TestResults;
use super::types::TestSuite;

/// Generate report based on format
pub fn generate_report(
    format: &ReportFormat,
    results: &TestResults,
    suite: &TestSuite,
) -> Result<String, String> {
    match format {
        ReportFormat::Text => generate_text_report(results),
        ReportFormat::Json => generate_json_report(results),
        ReportFormat::Html => generate_html_report(results),
        ReportFormat::Markdown => generate_markdown_report(results),
        ReportFormat::Csv => generate_csv_report(results),
    }
}

/// Generate text report
pub fn generate_text_report(results: &TestResults) -> Result<String, String> {
    let mut report = String::new();

    report.push_str("=== Quantum Optimization Test Report ===\n\n");

    report.push_str(&format!("Total Tests: {}\n", results.summary.total_tests));
    report.push_str(&format!("Passed: {}\n", results.summary.passed));
    report.push_str(&format!("Failed: {}\n", results.summary.failed));
    report.push_str(&format!(
        "Success Rate: {:.2}%\n",
        results.summary.success_rate * 100.0
    ));
    report.push_str(&format!(
        "Average Runtime: {:?}\n\n",
        results.summary.avg_runtime
    ));

    report.push_str("Quality Metrics:\n");
    report.push_str(&format!(
        "  Average Quality: {:.4}\n",
        results.summary.quality_metrics.avg_quality
    ));
    report.push_str(&format!(
        "  Best Quality: {:.4}\n",
        results.summary.quality_metrics.best_quality
    ));
    report.push_str(&format!(
        "  Worst Quality: {:.4}\n",
        results.summary.quality_metrics.worst_quality
    ));
    report.push_str(&format!(
        "  Std Dev: {:.4}\n",
        results.summary.quality_metrics.std_dev
    ));
    report.push_str(&format!(
        "  Constraint Satisfaction: {:.2}%\n\n",
        results.summary.quality_metrics.constraint_satisfaction_rate * 100.0
    ));

    if !results.failures.is_empty() {
        report.push_str("Failures:\n");
        for failure in &results.failures {
            report.push_str(&format!(
                "  - {} ({:?}): {}\n",
                failure.test_id, failure.failure_type, failure.message
            ));
        }
    }

    Ok(report)
}

/// Generate JSON report
pub fn generate_json_report(results: &TestResults) -> Result<String, String> {
    // Helper to convert fmt::Error to String
    fn write_err(e: std::fmt::Error) -> String {
        format!("JSON write error: {e}")
    }

    let mut json = String::new();

    // Build JSON manually (avoiding serde dependency issues)
    json.push_str("{\n");

    // Summary section
    json.push_str("  \"summary\": {\n");
    writeln!(
        &mut json,
        "    \"total_tests\": {},",
        results.summary.total_tests
    )
    .map_err(write_err)?;
    writeln!(&mut json, "    \"passed\": {},", results.summary.passed).map_err(write_err)?;
    writeln!(&mut json, "    \"failed\": {},", results.summary.failed).map_err(write_err)?;
    writeln!(&mut json, "    \"skipped\": {},", results.summary.skipped).map_err(write_err)?;
    writeln!(
        &mut json,
        "    \"success_rate\": {},",
        results.summary.success_rate
    )
    .map_err(write_err)?;
    writeln!(
        &mut json,
        "    \"avg_runtime_ms\": {}",
        results.summary.avg_runtime.as_millis()
    )
    .map_err(write_err)?;
    json.push_str("  },\n");

    // Quality metrics
    json.push_str("  \"quality_metrics\": {\n");
    writeln!(
        &mut json,
        "    \"avg_quality\": {},",
        results.summary.quality_metrics.avg_quality
    )
    .map_err(write_err)?;
    writeln!(
        &mut json,
        "    \"best_quality\": {},",
        results.summary.quality_metrics.best_quality
    )
    .map_err(write_err)?;
    writeln!(
        &mut json,
        "    \"worst_quality\": {},",
        results.summary.quality_metrics.worst_quality
    )
    .map_err(write_err)?;
    writeln!(
        &mut json,
        "    \"std_dev\": {},",
        results.summary.quality_metrics.std_dev
    )
    .map_err(write_err)?;
    writeln!(
        &mut json,
        "    \"constraint_satisfaction_rate\": {}",
        results.summary.quality_metrics.constraint_satisfaction_rate
    )
    .map_err(write_err)?;
    json.push_str("  },\n");

    // Performance data
    json.push_str("  \"performance\": {\n");
    writeln!(
        &mut json,
        "    \"total_time_ms\": {},",
        results.performance.runtime_stats.total_time.as_millis()
    )
    .map_err(write_err)?;
    writeln!(
        &mut json,
        "    \"solving_time_ms\": {},",
        results.performance.runtime_stats.solving_time.as_millis()
    )
    .map_err(write_err)?;
    writeln!(
        &mut json,
        "    \"validation_time_ms\": {}",
        results
            .performance
            .runtime_stats
            .validation_time
            .as_millis()
    )
    .map_err(write_err)?;
    json.push_str("  },\n");

    // Test results
    json.push_str("  \"test_results\": [\n");
    for (i, result) in results.test_results.iter().enumerate() {
        json.push_str("    {\n");
        writeln!(&mut json, "      \"test_id\": \"{}\",", result.test_id).map_err(write_err)?;
        writeln!(&mut json, "      \"sampler\": \"{}\",", result.sampler).map_err(write_err)?;
        writeln!(
            &mut json,
            "      \"objective_value\": {},",
            result.objective_value
        )
        .map_err(write_err)?;
        writeln!(
            &mut json,
            "      \"constraints_satisfied\": {},",
            result.constraints_satisfied
        )
        .map_err(write_err)?;
        writeln!(
            &mut json,
            "      \"runtime_ms\": {},",
            result.runtime.as_millis()
        )
        .map_err(write_err)?;
        writeln!(
            &mut json,
            "      \"is_valid\": {}",
            result.validation.is_valid
        )
        .map_err(write_err)?;
        json.push_str("    }");
        if i < results.test_results.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("  ],\n");

    // Failures
    json.push_str("  \"failures\": [\n");
    for (i, failure) in results.failures.iter().enumerate() {
        json.push_str("    {\n");
        writeln!(&mut json, "      \"test_id\": \"{}\",", failure.test_id).map_err(write_err)?;
        writeln!(
            &mut json,
            "      \"failure_type\": \"{:?}\",",
            failure.failure_type
        )
        .map_err(write_err)?;
        writeln!(
            &mut json,
            "      \"message\": \"{}\"",
            failure.message.replace('"', "\\\"")
        )
        .map_err(write_err)?;
        json.push_str("    }");
        if i < results.failures.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("  ]\n");

    json.push_str("}\n");

    Ok(json)
}

/// Generate HTML report
pub fn generate_html_report(results: &TestResults) -> Result<String, String> {
    let mut html = String::new();

    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("<title>Quantum Optimization Test Report</title>\n");
    html.push_str(
        "<style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .passed { color: green; }
        .failed { color: red; }
    </style>\n",
    );
    html.push_str("</head>\n<body>\n");

    html.push_str("<h1>Quantum Optimization Test Report</h1>\n");

    // Summary
    html.push_str("<h2>Summary</h2>\n");
    html.push_str("<table>\n");
    html.push_str(&format!(
        "<tr><td>Total Tests</td><td>{}</td></tr>\n",
        results.summary.total_tests
    ));
    html.push_str(&format!(
        "<tr><td>Passed</td><td class='passed'>{}</td></tr>\n",
        results.summary.passed
    ));
    html.push_str(&format!(
        "<tr><td>Failed</td><td class='failed'>{}</td></tr>\n",
        results.summary.failed
    ));
    html.push_str(&format!(
        "<tr><td>Success Rate</td><td>{:.2}%</td></tr>\n",
        results.summary.success_rate * 100.0
    ));
    html.push_str("</table>\n");

    html.push_str("</body>\n</html>");

    Ok(html)
}

/// Generate Markdown report
pub fn generate_markdown_report(results: &TestResults) -> Result<String, String> {
    let mut md = String::new();

    md.push_str("# Quantum Optimization Test Report\n\n");

    md.push_str("## Summary\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!(
        "| Total Tests | {} |\n",
        results.summary.total_tests
    ));
    md.push_str(&format!("| Passed | {} |\n", results.summary.passed));
    md.push_str(&format!("| Failed | {} |\n", results.summary.failed));
    md.push_str(&format!(
        "| Success Rate | {:.2}% |\n",
        results.summary.success_rate * 100.0
    ));
    md.push_str(&format!(
        "| Average Runtime | {:?} |\n\n",
        results.summary.avg_runtime
    ));

    md.push_str("## Quality Metrics\n\n");
    md.push_str(&format!(
        "- **Average Quality**: {:.4}\n",
        results.summary.quality_metrics.avg_quality
    ));
    md.push_str(&format!(
        "- **Best Quality**: {:.4}\n",
        results.summary.quality_metrics.best_quality
    ));
    md.push_str(&format!(
        "- **Worst Quality**: {:.4}\n",
        results.summary.quality_metrics.worst_quality
    ));
    md.push_str(&format!(
        "- **Standard Deviation**: {:.4}\n",
        results.summary.quality_metrics.std_dev
    ));
    md.push_str(&format!(
        "- **Constraint Satisfaction Rate**: {:.2}%\n\n",
        results.summary.quality_metrics.constraint_satisfaction_rate * 100.0
    ));

    Ok(md)
}

/// Generate CSV report
pub fn generate_csv_report(results: &TestResults) -> Result<String, String> {
    let mut csv = String::new();

    csv.push_str("test_id,sampler,objective_value,constraints_satisfied,runtime_ms,valid\n");

    for result in &results.test_results {
        csv.push_str(&format!(
            "{},{},{},{},{},{}\n",
            result.test_id,
            result.sampler,
            result.objective_value,
            result.constraints_satisfied,
            result.runtime.as_millis(),
            result.validation.is_valid
        ));
    }

    Ok(csv)
}

/// Export results as CSV with problem type and size
pub fn export_csv(results: &TestResults, suite: &TestSuite) -> Result<String, String> {
    let mut csv = String::new();
    csv.push_str("test_id,problem_type,size,sampler,objective_value,runtime_ms,constraints_satisfied,valid\n");

    for result in &results.test_results {
        // Find corresponding test case for additional info
        if let Some(test_case) = suite.test_cases.iter().find(|tc| tc.id == result.test_id) {
            csv.push_str(&format!(
                "{},{:?},{},{},{},{},{},{}\n",
                result.test_id,
                test_case.problem_type,
                test_case.size,
                result.sampler,
                result.objective_value,
                result.runtime.as_millis(),
                result.constraints_satisfied,
                result.validation.is_valid
            ));
        }
    }

    Ok(csv)
}

/// Export results as XML
pub fn export_xml(results: &TestResults) -> Result<String, String> {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<test_results>\n");
    xml.push_str(&format!(
        "  <summary total=\"{}\" passed=\"{}\" failed=\"{}\" success_rate=\"{:.2}\"/>\n",
        results.summary.total_tests,
        results.summary.passed,
        results.summary.failed,
        results.summary.success_rate
    ));

    xml.push_str("  <tests>\n");
    for result in &results.test_results {
        xml.push_str(&format!(
            "    <test id=\"{}\" sampler=\"{}\" objective=\"{}\" runtime_ms=\"{}\" valid=\"{}\"/>\n",
            result.test_id,
            result.sampler,
            result.objective_value,
            result.runtime.as_millis(),
            result.validation.is_valid
        ));
    }
    xml.push_str("  </tests>\n");
    xml.push_str("</test_results>\n");

    Ok(xml)
}

/// Save report to file
pub fn save_report(report: &str, filename: &str) -> Result<(), String> {
    let mut file = File::create(filename).map_err(|e| format!("Failed to create file: {e}"))?;
    file.write_all(report.as_bytes())
        .map_err(|e| format!("Failed to write file: {e}"))?;
    Ok(())
}
