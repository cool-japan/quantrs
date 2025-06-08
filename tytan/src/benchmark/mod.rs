//! Hardware benchmarking suite with SciRS2 analysis
//!
//! This module provides comprehensive benchmarking capabilities for quantum annealing
//! hardware and simulation backends, with optional SciRS2 optimizations.

pub mod analysis;
pub mod hardware;
pub mod metrics;
pub mod runner;
pub mod visualization;

pub use self::analysis::PerformanceReport;
pub use self::hardware::{HardwareBackend, BackendCapabilities};
pub use self::metrics::{BenchmarkMetrics, MetricType};
pub use self::runner::{BenchmarkRunner, BenchmarkConfig};
pub use self::visualization::BenchmarkVisualizer;

/// Prelude for common benchmark imports
pub mod prelude {
    pub use super::{
        BenchmarkRunner, BenchmarkConfig, BenchmarkMetrics,
        HardwareBackend, BackendCapabilities,
        PerformanceReport,
    };
}

/// Run a complete benchmark suite
pub fn run_benchmark_suite(config: BenchmarkConfig) -> Result<PerformanceReport, Box<dyn std::error::Error>> {
    let runner = BenchmarkRunner::new(config);
    runner.run_complete_suite()
}