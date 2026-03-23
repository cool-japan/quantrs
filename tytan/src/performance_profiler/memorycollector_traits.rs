//! # MemoryCollector - Trait Implementations
//!
//! This module contains trait implementations for `MemoryCollector`.
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
use super::types::{MemoryCollector, MetricType, MetricsSample};

impl MetricsCollector for MemoryCollector {
    fn collect(&self) -> Result<MetricsSample, String> {
        let mut values = HashMap::new();
        values.insert(MetricType::Memory, 0.0);
        Ok(MetricsSample {
            timestamp: Instant::now(),
            values,
        })
    }
    fn name(&self) -> &'static str {
        "MemoryCollector"
    }
    fn supported_metrics(&self) -> Vec<MetricType> {
        vec![MetricType::Memory]
    }
}
