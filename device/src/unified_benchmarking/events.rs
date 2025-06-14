//! Event handling for the unified benchmarking system

use std::collections::HashMap;
use std::time::SystemTime;

use super::types::QuantumPlatform;
use super::results::{PlatformBenchmarkResult, UnifiedBenchmarkResult};

/// Benchmark events for real-time monitoring
#[derive(Debug, Clone)]
pub enum BenchmarkEvent {
    BenchmarkStarted {
        execution_id: String,
        platforms: Vec<QuantumPlatform>,
        timestamp: SystemTime,
    },
    PlatformBenchmarkCompleted {
        execution_id: String,
        platform: QuantumPlatform,
        result: PlatformBenchmarkResult,
        timestamp: SystemTime,
    },
    BenchmarkCompleted {
        execution_id: String,
        result: UnifiedBenchmarkResult,
        timestamp: SystemTime,
    },
    BenchmarkFailed {
        execution_id: String,
        error: String,
        timestamp: SystemTime,
    },
    PerformanceAlert {
        metric: String,
        current_value: f64,
        threshold: f64,
        timestamp: SystemTime,
    },
    OptimizationCompleted {
        execution_id: String,
        improvements: HashMap<String, f64>,
        timestamp: SystemTime,
    },
}