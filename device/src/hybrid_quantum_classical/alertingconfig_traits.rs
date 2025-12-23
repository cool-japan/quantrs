//! # Alertingconfig - Trait Implementations
//!
//! This module contains trait implementations for `Alertingconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::collections::HashMap;

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            thresholds: HashMap::new(),
            notification_channels: vec![NotificationChannel::Log],
        }
    }
}
