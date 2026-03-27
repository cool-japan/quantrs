//! # EnhancedResourceEstimator - export_report_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;

use super::types::{EnhancedResourceEstimate, ReportFormat};

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Export estimation report
    pub fn export_report(
        &self,
        estimate: &EnhancedResourceEstimate,
        format: ReportFormat,
    ) -> Result<String, QuantRS2Error> {
        match format {
            ReportFormat::JSON => self.export_json_report(estimate),
            ReportFormat::HTML => self.export_html_report(estimate),
            ReportFormat::PDF => self.export_pdf_report(estimate),
            ReportFormat::Markdown => self.export_markdown_report(estimate),
            ReportFormat::LaTeX => self.export_latex_report(estimate),
            ReportFormat::YAML => Err(QuantRS2Error::UnsupportedOperation(
                "Format not supported".into(),
            )),
        }
    }
    /// Export JSON report
    pub(super) fn export_json_report(
        &self,
        estimate: &EnhancedResourceEstimate,
    ) -> Result<String, QuantRS2Error> {
        serde_json::to_string_pretty(estimate)
            .map_err(|e| QuantRS2Error::ComputationError(format!("JSON serialization failed: {e}")))
    }
    /// Export PDF report
    pub(super) fn export_pdf_report(
        &self,
        _estimate: &EnhancedResourceEstimate,
    ) -> Result<String, QuantRS2Error> {
        Ok("PDF export not implemented".to_string())
    }
    /// Export LaTeX report
    pub(super) fn export_latex_report(
        &self,
        _estimate: &EnhancedResourceEstimate,
    ) -> Result<String, QuantRS2Error> {
        Ok(
            "\\documentclass{article}\n\\begin{document}\nResource Estimation Report\n\\end{document}"
                .to_string(),
        )
    }
}
