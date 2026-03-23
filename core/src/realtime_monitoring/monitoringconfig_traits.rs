//! # MonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `MonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
    sync::{Arc, RwLock},
    thread,
    time::{Duration, SystemTime},
};

use super::types::{AlertThresholds, ExportSettings, MonitoringConfig};

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(1),
            data_retention_period: Duration::from_secs(24 * 3600),
            alert_thresholds: AlertThresholds::default(),
            enabled_metrics: HashSet::new(),
            platform_configs: HashMap::new(),
            export_settings: ExportSettings::default(),
        }
    }
}
