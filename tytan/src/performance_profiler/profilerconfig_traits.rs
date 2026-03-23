//! # ProfilerConfig - Trait Implementations
//!
//! This module contains trait implementations for `ProfilerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::time::{Duration, Instant};

use super::types::{MetricType, OutputFormat, ProfilerConfig};

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_interval: Duration::from_millis(100),
            metrics: vec![MetricType::Time, MetricType::Memory],
            profile_memory: true,
            profile_cpu: true,
            profile_gpu: false,
            detailed_timing: false,
            output_format: OutputFormat::Json,
            auto_save_interval: Some(Duration::from_secs(60)),
        }
    }
}
