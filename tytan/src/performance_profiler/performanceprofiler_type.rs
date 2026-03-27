//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::functions::MetricsCollector;
use super::types::{PerformanceAnalyzer, Profile, ProfileContext, ProfilerConfig};

/// Performance profiler
pub struct PerformanceProfiler {
    /// Configuration
    pub(super) config: ProfilerConfig,
    /// Profile data
    pub(super) profiles: Vec<Profile>,
    /// Current profile
    pub(super) current_profile: Option<ProfileContext>,
    /// Metrics collectors
    pub(super) collectors: Vec<Box<dyn MetricsCollector>>,
    /// Analysis engine
    pub(super) analyzer: PerformanceAnalyzer,
}
