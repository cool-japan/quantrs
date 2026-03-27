//! # BusinessMetricsConfig - Trait Implementations
//!
//! This module contains trait implementations for `BusinessMetricsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{
    BusinessMetricsConfig, PerformanceKpi, UsageMetricsConfig, UserAnalyticsConfig,
};

impl Default for BusinessMetricsConfig {
    fn default() -> Self {
        Self {
            enable_business_metrics: true,
            user_analytics: UserAnalyticsConfig::default(),
            usage_metrics: UsageMetricsConfig::default(),
            performance_kpis: vec![
                PerformanceKpi::ResponseTime,
                PerformanceKpi::Throughput,
                PerformanceKpi::ErrorRate,
                PerformanceKpi::UserSatisfaction,
            ],
            dashboard_refresh_rate: Duration::from_secs(300),
        }
    }
}
