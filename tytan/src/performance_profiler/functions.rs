//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::time::{Duration, Instant};

use super::performanceprofiler_type::PerformanceProfiler;
use super::types::{MetricType, MetricsSample, OutputFormat, ProfilerConfig};

/// Metrics collector trait
pub trait MetricsCollector: Send + Sync {
    /// Collect metrics
    fn collect(&self) -> Result<MetricsSample, String>;
    /// Collector name
    fn name(&self) -> &str;
    /// Supported metrics
    fn supported_metrics(&self) -> Vec<MetricType>;
}
/// Profiling macros
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr) => {
        $profiler.enter_function($name)
    };
}
#[macro_export]
macro_rules! time_it {
    ($profiler:expr, $name:expr, $code:block) => {{
        $profiler.start_timer($name);
        let mut result = $code;
        $profiler.stop_timer($name);
        result
    }};
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    #[test]
    fn test_performance_profiler() {
        let mut config = ProfilerConfig {
            enabled: true,
            sampling_interval: Duration::from_millis(10),
            metrics: vec![MetricType::Time, MetricType::Memory],
            profile_memory: true,
            profile_cpu: true,
            profile_gpu: false,
            detailed_timing: true,
            output_format: OutputFormat::Json,
            auto_save_interval: None,
        };
        let mut profiler = PerformanceProfiler::new(config);
        let mut result = profiler.start_profile("test_profile");
        assert!(result.is_ok());
        {
            let _guard = profiler.enter_function("test_function");
            profiler.start_timer("computation");
            thread::sleep(Duration::from_millis(10));
            profiler.stop_timer("computation");
            profiler.record_allocation(1024);
            profiler.record_solution_quality(0.5);
            profiler.record_deallocation(1024);
        }
        let profile = profiler.stop_profile();
        assert!(profile.is_ok());
        let profile = profile.expect("Failed to stop profiling in test_performance_profiler");
        assert!(!profile.events.is_empty());
        assert!(profile.metrics.time_metrics.total_time > Duration::from_secs(0));
        let mut report = profiler.analyze_profile(&profile);
        assert!(report.summary.total_time > Duration::from_secs(0));
        let json_report = profiler.generate_report(&profile, &OutputFormat::Json);
        assert!(json_report.is_ok());
    }
}
