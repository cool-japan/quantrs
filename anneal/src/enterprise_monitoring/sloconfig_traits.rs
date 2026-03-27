//! # SloConfig - Trait Implementations
//!
//! This module contains trait implementations for `SloConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{BurnRateThreshold, SloConfig};

impl Default for SloConfig {
    fn default() -> Self {
        Self {
            enable_slo_monitoring: true,
            default_error_budget: 0.001,
            evaluation_window: Duration::from_secs(24 * 3600),
            alert_on_breach: true,
            burn_rate_thresholds: vec![
                BurnRateThreshold {
                    window: Duration::from_secs(5 * 60),
                    threshold: 14.4,
                },
                BurnRateThreshold {
                    window: Duration::from_secs(1 * 3600),
                    threshold: 6.0,
                },
                BurnRateThreshold {
                    window: Duration::from_secs(6 * 3600),
                    threshold: 1.0,
                },
            ],
        }
    }
}
