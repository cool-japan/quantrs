//! # SecurityMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `SecurityMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{ComplianceFramework, SecurityMonitoringConfig, ThreatSensitivity};

impl Default for SecurityMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_security_monitoring: true,
            threat_detection_sensitivity: ThreatSensitivity::Medium,
            enable_behavioral_analysis: true,
            compliance_frameworks: vec![
                ComplianceFramework::SOC2,
                ComplianceFramework::ISO27001,
                ComplianceFramework::GDPR,
            ],
            security_event_retention: Duration::from_secs(90 * 86_400),
        }
    }
}
