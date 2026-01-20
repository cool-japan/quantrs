//! Runtime profiling for JIT compilation
//!
//! This module provides runtime profiling and statistics tracking.

use std::collections::VecDeque;
use std::time::Duration;

/// Runtime profiler for performance monitoring
pub struct RuntimeProfiler {
    /// Execution time tracking
    execution_times: VecDeque<Duration>,
    /// Memory usage tracking
    memory_usage: VecDeque<usize>,
    /// Performance statistics
    stats: RuntimeProfilerStats,
}

impl Default for RuntimeProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeProfiler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            execution_times: VecDeque::new(),
            memory_usage: VecDeque::new(),
            stats: RuntimeProfilerStats::default(),
        }
    }

    /// Record execution time
    pub fn record_execution_time(&mut self, duration: Duration) {
        self.execution_times.push_back(duration);
        if self.execution_times.len() > 1000 {
            self.execution_times.pop_front();
        }
        self.update_stats();
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, usage: usize) {
        self.memory_usage.push_back(usage);
        if self.memory_usage.len() > 1000 {
            self.memory_usage.pop_front();
        }
        self.update_stats();
    }

    /// Update performance statistics
    fn update_stats(&mut self) {
        if !self.execution_times.is_empty() {
            let total_time: Duration = self.execution_times.iter().sum();
            self.stats.average_execution_time = total_time / self.execution_times.len() as u32;

            self.stats.min_execution_time = self
                .execution_times
                .iter()
                .min()
                .copied()
                .unwrap_or(Duration::from_secs(0));
            self.stats.max_execution_time = self
                .execution_times
                .iter()
                .max()
                .copied()
                .unwrap_or(Duration::from_secs(0));
        }

        if !self.memory_usage.is_empty() {
            self.stats.average_memory_usage =
                self.memory_usage.iter().sum::<usize>() / self.memory_usage.len();
            self.stats.peak_memory_usage = self.memory_usage.iter().max().copied().unwrap_or(0);
        }

        self.stats.sample_count = self.execution_times.len();
    }

    /// Get current statistics
    #[must_use]
    pub const fn get_stats(&self) -> &RuntimeProfilerStats {
        &self.stats
    }
}

/// Runtime profiler statistics
#[derive(Debug, Clone)]
pub struct RuntimeProfilerStats {
    /// Average execution time
    pub average_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Average memory usage
    pub average_memory_usage: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Number of samples
    pub sample_count: usize,
}

impl Default for RuntimeProfilerStats {
    fn default() -> Self {
        Self {
            average_execution_time: Duration::from_secs(0),
            min_execution_time: Duration::from_secs(0),
            max_execution_time: Duration::from_secs(0),
            average_memory_usage: 0,
            peak_memory_usage: 0,
            sample_count: 0,
        }
    }
}

/// JIT compiler statistics
#[derive(Debug, Clone)]
pub struct JITCompilerStats {
    /// Total number of compilations
    pub total_compilations: usize,
    /// Total compilation time
    pub total_compilation_time: Duration,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of cache clears
    pub cache_clears: usize,
    /// Average compilation time
    pub average_compilation_time: Duration,
    /// Total patterns analyzed
    pub patterns_analyzed: usize,
    /// Successful compilations
    pub successful_compilations: usize,
    /// Failed compilations
    pub failed_compilations: usize,
}

impl Default for JITCompilerStats {
    fn default() -> Self {
        Self {
            total_compilations: 0,
            total_compilation_time: Duration::from_secs(0),
            cache_hits: 0,
            cache_misses: 0,
            cache_clears: 0,
            average_compilation_time: Duration::from_secs(0),
            patterns_analyzed: 0,
            successful_compilations: 0,
            failed_compilations: 0,
        }
    }
}

/// JIT simulator statistics
#[derive(Debug, Clone)]
pub struct JITSimulatorStats {
    /// Number of compiled executions
    pub compiled_executions: usize,
    /// Number of interpreted executions
    pub interpreted_executions: usize,
    /// Total time spent in compiled execution
    pub total_compiled_time: Duration,
    /// Total time spent in interpreted execution
    pub total_interpreted_time: Duration,
    /// JIT compilation speedup factor
    pub speedup_factor: f64,
}

impl Default for JITSimulatorStats {
    fn default() -> Self {
        Self {
            compiled_executions: 0,
            interpreted_executions: 0,
            total_compiled_time: Duration::from_secs(0),
            total_interpreted_time: Duration::from_secs(0),
            speedup_factor: 1.0,
        }
    }
}

impl JITSimulatorStats {
    /// Update speedup factor
    pub fn update_speedup_factor(&mut self) {
        if self.compiled_executions > 0 && self.interpreted_executions > 0 {
            let avg_compiled =
                self.total_compiled_time.as_secs_f64() / self.compiled_executions as f64;
            let avg_interpreted =
                self.total_interpreted_time.as_secs_f64() / self.interpreted_executions as f64;

            if avg_compiled > 0.0 {
                self.speedup_factor = avg_interpreted / avg_compiled;
            }
        }
    }
}
