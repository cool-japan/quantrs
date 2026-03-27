//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::buffer_pool::BufferPool;
use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex};

use super::types::{
    AdaptiveBenchmarkController, BenchmarkCache, ComparativeAnalyzer, EnhancedBenchmarkConfig,
    MLPerformancePredictor, RealtimeMonitor, StatisticalAnalysis, VisualAnalyzer,
};

/// Enhanced hardware benchmarking system
pub struct EnhancedHardwareBenchmark {
    pub config: EnhancedBenchmarkConfig,
    pub(super) statistical_analyzer: Arc<StatisticalAnalysis>,
    pub(super) ml_predictor: Option<Arc<MLPerformancePredictor>>,
    pub(super) comparative_analyzer: Arc<ComparativeAnalyzer>,
    pub(super) realtime_monitor: Arc<RealtimeMonitor>,
    pub(super) adaptive_controller: Arc<AdaptiveBenchmarkController>,
    pub(super) visual_analyzer: Arc<VisualAnalyzer>,
    pub(super) buffer_pool: BufferPool<f64>,
    pub(super) cache: Arc<Mutex<BenchmarkCache>>,
}
