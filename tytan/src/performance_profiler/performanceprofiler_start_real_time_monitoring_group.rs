//! # PerformanceProfiler - start_real_time_monitoring_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::types::{AnalysisConfig, PerformanceAnalyzer, ProfilerConfig, RealTimeMonitor};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Start real-time monitoring
    pub fn start_real_time_monitoring(&mut self) -> Result<RealTimeMonitor, String> {
        if !self.config.enabled {
            return Err("Profiling not enabled".to_string());
        }
        let monitor = RealTimeMonitor::new(
            self.config.sampling_interval,
            self.collectors
                .iter()
                .map(|c| c.name().to_string())
                .collect(),
        )?;
        Ok(monitor)
    }
    /// Create new profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            profiles: Vec::new(),
            current_profile: None,
            collectors: Self::default_collectors(),
            analyzer: PerformanceAnalyzer::new(AnalysisConfig {
                detect_bottlenecks: true,
                suggest_optimizations: true,
                detect_anomalies: true,
                detect_regressions: false,
                baseline: None,
            }),
        }
    }
}
