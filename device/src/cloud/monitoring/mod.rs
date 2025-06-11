//! Cloud Monitoring and Analytics Configuration
//!
//! This module provides comprehensive cloud monitoring capabilities for quantum computing
//! workloads, including performance monitoring, resource tracking, cost management,
//! security monitoring, alerting, and analytics.
//!
//! The module has been organized into focused submodules:
//! - `performance`: Performance metrics and monitoring configuration
//! - `resource`: Resource monitoring and usage tracking
//! - `cost`: Cost monitoring, budget tracking, and optimization
//! - `security`: Security monitoring and threat detection
//! - `alerting`: Alert configuration and notification systems
//! - `analytics`: Analytics, reporting, and anomaly detection
//! - `ml`: Machine learning and AutoML configurations

pub mod alerting;
pub mod analytics;
pub mod cost;
pub mod ml;
pub mod performance;
pub mod resource;
pub mod security;

use serde::{Deserialize, Serialize};
use std::time::Duration;

// Re-export main types for backward compatibility
pub use alerting::*;
pub use analytics::*;
pub use cost::*;
pub use ml::*;
pub use performance::*;
pub use resource::*;
pub use security::*;

/// Main cloud monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudMonitoringConfig {
    /// Performance monitoring
    pub performance_monitoring: performance::CloudPerformanceMonitoringConfig,
    /// Resource monitoring
    pub resource_monitoring: resource::CloudResourceMonitoringConfig,
    /// Cost monitoring
    pub cost_monitoring: cost::CloudCostMonitoringConfig,
    /// Security monitoring
    pub security_monitoring: security::CloudSecurityMonitoringConfig,
    /// Alerting configuration
    pub alerting: alerting::CloudAlertingConfig,
    /// Analytics and reporting
    pub analytics: analytics::CloudAnalyticsConfig,
}

impl Default for CloudMonitoringConfig {
    fn default() -> Self {
        Self {
            performance_monitoring: performance::CloudPerformanceMonitoringConfig::default(),
            resource_monitoring: resource::CloudResourceMonitoringConfig::default(),
            cost_monitoring: cost::CloudCostMonitoringConfig::default(),
            security_monitoring: security::CloudSecurityMonitoringConfig::default(),
            alerting: alerting::CloudAlertingConfig::default(),
            analytics: analytics::CloudAnalyticsConfig::default(),
        }
    }
}
