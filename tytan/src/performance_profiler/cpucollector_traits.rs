//! # CPUCollector - Trait Implementations
//!
//! This module contains trait implementations for `CPUCollector`.
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
use super::types::{CPUCollector, MetricType, MetricsSample};

impl MetricsCollector for CPUCollector {
    fn collect(&self) -> Result<MetricsSample, String> {
        let mut values = HashMap::new();
        values.insert(MetricType::CPU, 0.0);
        Ok(MetricsSample {
            timestamp: Instant::now(),
            values,
        })
    }
    fn name(&self) -> &'static str {
        "CPUCollector"
    }
    fn supported_metrics(&self) -> Vec<MetricType> {
        vec![MetricType::CPU]
    }
}
