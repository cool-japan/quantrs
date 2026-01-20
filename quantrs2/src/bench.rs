//! # Benchmarking Utilities for QuantRS2
//!
//! This module provides utilities for benchmarking quantum algorithms, circuits,
//! and simulations with standardized measurement and reporting.
//!
//! ## Features
//!
//! - Timer utilities with high-precision measurements
//! - Statistical aggregation of multiple runs
//! - Memory usage tracking
//! - Throughput calculations
//! - Benchmark report generation
//!
//! ## Example Usage
//!
//! ```rust
//! use quantrs2::bench::{BenchmarkTimer, BenchmarkStats};
//!
//! // Simple timing
//! let timer = BenchmarkTimer::start();
//! // ... perform operation ...
//! let elapsed = timer.stop();
//! println!("Operation took {:?}", elapsed);
//!
//! // Statistical benchmarking
//! let mut stats = BenchmarkStats::new("quantum_operation");
//! for _ in 0..100 {
//!     let timer = BenchmarkTimer::start();
//!     // ... perform operation ...
//!     stats.record(timer.stop());
//! }
//! println!("{}", stats.report());
//! ```

#![allow(clippy::must_use_candidate)]

use std::time::{Duration, Instant};

/// High-precision benchmark timer
#[derive(Debug)]
pub struct BenchmarkTimer {
    start: Instant,
    label: Option<String>,
}

impl BenchmarkTimer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            label: None,
        }
    }

    /// Start a new timer with a label
    pub fn start_labeled(label: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            label: Some(label.into()),
        }
    }

    /// Stop the timer and return elapsed duration
    pub fn stop(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return elapsed time in milliseconds
    pub fn stop_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Stop the timer and return elapsed time in microseconds
    pub fn stop_us(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1_000_000.0
    }

    /// Stop the timer and return elapsed time in nanoseconds
    pub fn stop_ns(&self) -> u128 {
        self.start.elapsed().as_nanos()
    }

    /// Get the label if set
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

/// Statistical aggregation of benchmark measurements
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    /// Name of the benchmark
    pub name: String,
    /// All recorded durations
    samples: Vec<Duration>,
    /// Number of operations per sample (for throughput calculation)
    ops_per_sample: usize,
}

impl BenchmarkStats {
    /// Create a new benchmark stats collector
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            samples: Vec::new(),
            ops_per_sample: 1,
        }
    }

    /// Set the number of operations per sample (for throughput calculation)
    pub const fn set_ops_per_sample(&mut self, ops: usize) {
        self.ops_per_sample = ops;
    }

    /// Record a single duration sample
    pub fn record(&mut self, duration: Duration) {
        self.samples.push(duration);
    }

    /// Record a sample in milliseconds
    pub fn record_ms(&mut self, ms: f64) {
        self.samples.push(Duration::from_secs_f64(ms / 1000.0));
    }

    /// Get the number of samples recorded
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Calculate the mean duration
    pub fn mean(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let total: Duration = self.samples.iter().sum();
        Some(total / self.samples.len() as u32)
    }

    /// Calculate the median duration
    pub fn median(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted = self.samples.clone();
        sorted.sort();
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            Some((sorted[mid - 1] + sorted[mid]) / 2)
        } else {
            Some(sorted[mid])
        }
    }

    /// Calculate the minimum duration
    pub fn min(&self) -> Option<Duration> {
        self.samples.iter().min().copied()
    }

    /// Calculate the maximum duration
    pub fn max(&self) -> Option<Duration> {
        self.samples.iter().max().copied()
    }

    /// Calculate the standard deviation
    pub fn std_dev(&self) -> Option<Duration> {
        if self.samples.len() < 2 {
            return None;
        }
        let mean = self.mean()?.as_secs_f64();
        let variance: f64 = self
            .samples
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.samples.len() - 1) as f64;
        Some(Duration::from_secs_f64(variance.sqrt()))
    }

    /// Calculate throughput in operations per second
    pub fn throughput(&self) -> Option<f64> {
        let mean = self.mean()?;
        let ops_per_sec = self.ops_per_sample as f64 / mean.as_secs_f64();
        Some(ops_per_sec)
    }

    /// Calculate the percentile duration
    pub fn percentile(&self, p: f64) -> Option<Duration> {
        if self.samples.is_empty() || !(0.0..=100.0).contains(&p) {
            return None;
        }
        let mut sorted = self.samples.clone();
        sorted.sort();
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        Some(sorted[idx])
    }

    /// Generate a human-readable benchmark report
    pub fn report(&self) -> String {
        use std::fmt::Write;

        if self.samples.is_empty() {
            return format!("Benchmark '{}': No samples recorded", self.name);
        }

        let mut report = String::new();
        let _ = writeln!(report, "Benchmark: {}", self.name);
        let _ = writeln!(report, "  Samples: {}", self.count());

        if let Some(mean) = self.mean() {
            let _ = writeln!(report, "  Mean: {mean:?}");
        }
        if let Some(median) = self.median() {
            let _ = writeln!(report, "  Median: {median:?}");
        }
        if let Some(std_dev) = self.std_dev() {
            let _ = writeln!(report, "  Std Dev: {std_dev:?}");
        }
        if let Some(min) = self.min() {
            let _ = writeln!(report, "  Min: {min:?}");
        }
        if let Some(max) = self.max() {
            let _ = writeln!(report, "  Max: {max:?}");
        }
        if let Some(p99) = self.percentile(99.0) {
            let _ = writeln!(report, "  P99: {p99:?}");
        }
        if let Some(throughput) = self.throughput() {
            if throughput > 1000.0 {
                let _ = writeln!(report, "  Throughput: {:.2} K ops/s", throughput / 1000.0);
            } else {
                let _ = writeln!(report, "  Throughput: {throughput:.2} ops/s");
            }
        }

        report
    }

    /// Clear all recorded samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Measure execution time of a closure
///
/// # Arguments
///
/// * `f` - The closure to measure
///
/// # Returns
///
/// Tuple of (result, duration)
///
/// # Example
///
/// ```rust
/// use quantrs2::bench::measure;
///
/// let (result, duration) = measure(|| {
///     // expensive operation
///     42
/// });
/// println!("Result: {}, Duration: {:?}", result, duration);
/// ```
pub fn measure<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let timer = BenchmarkTimer::start();
    let result = f();
    (result, timer.stop())
}

/// Measure average execution time over multiple iterations
///
/// # Arguments
///
/// * `iterations` - Number of iterations to run
/// * `f` - The closure to measure
///
/// # Returns
///
/// `BenchmarkStats` containing all measurements
///
/// # Example
///
/// ```rust
/// use quantrs2::bench::measure_iterations;
///
/// let stats = measure_iterations(100, || {
///     // operation to benchmark
/// });
/// println!("{}", stats.report());
/// ```
pub fn measure_iterations<F, R>(iterations: usize, mut f: F) -> BenchmarkStats
where
    F: FnMut() -> R,
{
    let mut stats = BenchmarkStats::new("benchmark");
    for _ in 0..iterations {
        let timer = BenchmarkTimer::start();
        let _ = f();
        stats.record(timer.stop());
    }
    stats
}

/// Measure with warmup iterations
///
/// Performs warmup iterations that are not recorded, then measures the actual iterations.
///
/// # Arguments
///
/// * `warmup` - Number of warmup iterations
/// * `iterations` - Number of measured iterations
/// * `f` - The closure to measure
///
/// # Returns
///
/// `BenchmarkStats` containing measurements (excluding warmup)
///
/// # Example
///
/// ```rust
/// use quantrs2::bench::measure_with_warmup;
///
/// let stats = measure_with_warmup(10, 100, || {
///     // operation to benchmark
/// });
/// println!("{}", stats.report());
/// ```
pub fn measure_with_warmup<F, R>(warmup: usize, iterations: usize, mut f: F) -> BenchmarkStats
where
    F: FnMut() -> R,
{
    // Warmup phase
    for _ in 0..warmup {
        let _ = f();
    }

    // Measurement phase
    measure_iterations(iterations, f)
}

/// Memory usage estimation
#[derive(Debug, Clone, Copy)]
pub struct MemoryUsage {
    /// Bytes allocated
    pub bytes: usize,
}

impl MemoryUsage {
    /// Create from bytes
    pub const fn from_bytes(bytes: usize) -> Self {
        Self { bytes }
    }

    /// Get size in kilobytes
    pub fn kb(&self) -> f64 {
        self.bytes as f64 / 1024.0
    }

    /// Get size in megabytes
    pub fn mb(&self) -> f64 {
        self.bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get size in gigabytes
    pub fn gb(&self) -> f64 {
        self.bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        crate::utils::format_memory(self.bytes)
    }
}

/// Estimate memory for a state vector with given qubit count
pub const fn estimate_statevector_memory(num_qubits: u32) -> MemoryUsage {
    MemoryUsage::from_bytes(crate::utils::estimate_statevector_memory(num_qubits))
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measured iterations
    pub measure_iterations: usize,
    /// Operations per iteration (for throughput)
    pub ops_per_iteration: usize,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measure_iterations: 100,
            ops_per_iteration: 1,
            verbose: false,
        }
    }
}

impl BenchmarkConfig {
    /// Create a quick benchmark configuration
    pub const fn quick() -> Self {
        Self {
            warmup_iterations: 5,
            measure_iterations: 20,
            ops_per_iteration: 1,
            verbose: false,
        }
    }

    /// Create a thorough benchmark configuration
    pub const fn thorough() -> Self {
        Self {
            warmup_iterations: 50,
            measure_iterations: 1000,
            ops_per_iteration: 1,
            verbose: false,
        }
    }

    /// Run benchmark with this configuration
    pub fn run<F, R>(&self, name: &str, mut f: F) -> BenchmarkStats
    where
        F: FnMut() -> R,
    {
        if self.verbose {
            eprintln!(
                "Running benchmark '{}' with {} warmup + {} iterations...",
                name, self.warmup_iterations, self.measure_iterations
            );
        }

        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = f();
        }

        // Measure
        let mut stats = BenchmarkStats::new(name);
        stats.set_ops_per_sample(self.ops_per_iteration);

        for _ in 0..self.measure_iterations {
            let timer = BenchmarkTimer::start();
            let _ = f();
            stats.record(timer.stop());
        }

        if self.verbose {
            eprintln!("{}", stats.report());
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_benchmark_timer() {
        let timer = BenchmarkTimer::start();
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(9));
        assert!(elapsed < Duration::from_millis(50));
    }

    #[test]
    fn test_benchmark_timer_labeled() {
        let timer = BenchmarkTimer::start_labeled("test_op");
        assert_eq!(timer.label(), Some("test_op"));
        let _ = timer.stop();
    }

    #[test]
    fn test_benchmark_stats() {
        let mut stats = BenchmarkStats::new("test");

        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(20));
        stats.record(Duration::from_millis(30));

        assert_eq!(stats.count(), 3);
        assert_eq!(stats.mean(), Some(Duration::from_millis(20)));
        assert_eq!(stats.median(), Some(Duration::from_millis(20)));
        assert_eq!(stats.min(), Some(Duration::from_millis(10)));
        assert_eq!(stats.max(), Some(Duration::from_millis(30)));
    }

    #[test]
    fn test_benchmark_stats_empty() {
        let stats = BenchmarkStats::new("empty");
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.median(), None);
    }

    #[test]
    fn test_benchmark_stats_throughput() {
        let mut stats = BenchmarkStats::new("throughput_test");
        stats.set_ops_per_sample(100);
        stats.record(Duration::from_secs(1));

        let throughput = stats
            .throughput()
            .expect("throughput should be calculable with one sample");
        assert!((throughput - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_measure() {
        let (result, duration) = measure(|| {
            thread::sleep(Duration::from_millis(5));
            42
        });
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(4));
    }

    #[test]
    fn test_measure_iterations() {
        let stats = measure_iterations(10, || {
            thread::sleep(Duration::from_millis(1));
        });
        assert_eq!(stats.count(), 10);
    }

    #[test]
    fn test_memory_usage() {
        let mem = MemoryUsage::from_bytes(1024 * 1024);
        assert!((mem.kb() - 1024.0).abs() < 0.01);
        assert!((mem.mb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_statevector_memory() {
        let mem = estimate_statevector_memory(10);
        // 2^10 = 1024 complex numbers * 16 bytes = 16384 bytes
        assert_eq!(mem.bytes, 16384);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::quick();
        let stats = config.run("quick_test", || {
            // Minimal operation
            let _ = 1 + 1;
        });
        assert_eq!(stats.count(), 20);
    }

    #[test]
    fn test_percentile() {
        let mut stats = BenchmarkStats::new("percentile_test");
        for i in 1..=100 {
            stats.record(Duration::from_millis(i));
        }

        let p50 = stats
            .percentile(50.0)
            .expect("p50 should be calculable with 100 samples");
        assert!(p50 >= Duration::from_millis(49) && p50 <= Duration::from_millis(51));

        let p99 = stats
            .percentile(99.0)
            .expect("p99 should be calculable with 100 samples");
        assert!(p99 >= Duration::from_millis(98));
    }

    #[test]
    fn test_benchmark_report() {
        let mut stats = BenchmarkStats::new("report_test");
        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(20));

        let report = stats.report();
        assert!(report.contains("report_test"));
        assert!(report.contains("Samples: 2"));
        assert!(report.contains("Mean:"));
    }
}
