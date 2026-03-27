//! # TimeCollector - Trait Implementations
//!
//! This module contains trait implementations for `TimeCollector`.
//!
//! ## Implemented Traits
//!
//! - `MetricsCollector`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use super::functions::MetricsCollector;
use super::types::{MetricType, MetricsSample, TimeCollector};

impl MetricsCollector for TimeCollector {
    fn collect(&self) -> Result<MetricsSample, String> {
        Ok(MetricsSample {
            timestamp: Instant::now(),
            values: HashMap::new(),
        })
    }
    fn name(&self) -> &'static str {
        "TimeCollector"
    }
    fn supported_metrics(&self) -> Vec<MetricType> {
        vec![MetricType::Time]
    }
}
