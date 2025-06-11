//! Reporting Configuration Types

use serde::{Deserialize, Serialize};

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Enable automated reporting
    pub enable_automated_reports: bool,
    /// Report generation schedule
    pub report_schedule: ReportSchedule,
    /// Report formats to generate
    pub report_formats: Vec<ReportFormat>,
    /// Report distribution settings
    pub distribution_config: DistributionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchedule {
    pub frequency: ReportFrequency,
    pub time_of_day: String,
    pub time_zone: String,
    pub custom_schedule: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportFormat {
    PDF,
    HTML,
    JSON,
    CSV,
    Excel,
    PowerPoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionConfig {
    pub email_recipients: Vec<String>,
    pub file_storage_locations: Vec<String>,
    pub api_endpoints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportFrequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Custom,
}
