//! Benchmark metrics collection and analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Types of metrics to collect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Execution time
    ExecutionTime,
    /// Memory usage
    MemoryUsage,
    /// Solution quality
    SolutionQuality,
    /// Energy efficiency
    EnergyEfficiency,
    /// Throughput
    Throughput,
    /// Scalability
    Scalability,
    /// Cache efficiency
    CacheEfficiency,
    /// Convergence rate
    ConvergenceRate,
}

/// Benchmark metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Problem size (number of variables)
    pub problem_size: usize,
    /// Problem density (fraction of non-zero coefficients)
    pub problem_density: f64,
    /// Timing measurements
    pub timings: TimingMetrics,
    /// Memory measurements
    pub memory: MemoryMetrics,
    /// Solution quality metrics
    pub quality: QualityMetrics,
    /// Hardware utilization
    pub utilization: UtilizationMetrics,
    /// Additional custom metrics
    pub custom: HashMap<String, f64>,
}

/// Timing-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Total execution time
    pub total_time: Duration,
    /// Setup/initialization time
    pub setup_time: Duration,
    /// Actual computation time
    pub compute_time: Duration,
    /// Post-processing time
    pub postprocess_time: Duration,
    /// Time per sample
    pub time_per_sample: Duration,
    /// Time to first solution
    pub time_to_solution: Option<Duration>,
}

/// Memory-related metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryMetrics {
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Average memory usage
    pub avg_memory: usize,
    /// Memory allocated
    pub allocated: usize,
    /// Memory deallocated
    pub deallocated: usize,
    /// Cache misses
    pub cache_misses: Option<u64>,
}

/// Solution quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Best energy found
    pub best_energy: f64,
    /// Average energy across samples
    pub avg_energy: f64,
    /// Standard deviation of energies
    pub energy_std: f64,
    /// Success probability (finding optimal or near-optimal)
    pub success_probability: f64,
    /// Time to reach target quality
    pub time_to_target: Option<Duration>,
    /// Number of unique solutions found
    pub unique_solutions: usize,
}

/// Hardware utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationMetrics {
    /// CPU utilization percentage
    pub cpu_usage: f64,
    /// GPU utilization percentage (if applicable)
    pub gpu_usage: Option<f64>,
    /// Memory bandwidth utilization
    pub memory_bandwidth: f64,
    /// Cache hit rate
    pub cache_hit_rate: Option<f64>,
    /// Power consumption (if measurable)
    pub power_consumption: Option<f64>,
}

impl BenchmarkMetrics {
    /// Create new metrics collection
    pub fn new(problem_size: usize, problem_density: f64) -> Self {
        Self {
            problem_size,
            problem_density,
            timings: TimingMetrics::default(),
            memory: MemoryMetrics::default(),
            quality: QualityMetrics::default(),
            utilization: UtilizationMetrics::default(),
            custom: HashMap::new(),
        }
    }

    /// Calculate efficiency metrics
    pub fn calculate_efficiency(&self) -> EfficiencyMetrics {
        EfficiencyMetrics {
            samples_per_second: self.calculate_throughput(),
            energy_per_sample: self.calculate_energy_efficiency(),
            memory_efficiency: self.calculate_memory_efficiency(),
            scalability_factor: self.calculate_scalability(),
        }
    }

    /// Calculate throughput (samples per second)
    fn calculate_throughput(&self) -> f64 {
        if self.timings.total_time.as_secs_f64() > 0.0 {
            1.0 / self.timings.time_per_sample.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate energy efficiency
    fn calculate_energy_efficiency(&self) -> Option<f64> {
        self.utilization
            .power_consumption
            .map(|power| power * self.timings.time_per_sample.as_secs_f64())
    }

    /// Calculate memory efficiency
    fn calculate_memory_efficiency(&self) -> f64 {
        if self.memory.peak_memory > 0 {
            (self.problem_size as f64) / (self.memory.peak_memory as f64)
        } else {
            0.0
        }
    }

    /// Calculate scalability factor
    fn calculate_scalability(&self) -> f64 {
        // Simple O(n) check - ideal would be linear scaling
        let expected_time = self.problem_size as f64 * 1e-6; // Microseconds per variable
        let actual_time = self.timings.compute_time.as_secs_f64();

        if actual_time > 0.0 {
            expected_time / actual_time
        } else {
            0.0
        }
    }
}

/// Efficiency metrics derived from raw measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Samples generated per second
    pub samples_per_second: f64,
    /// Energy consumed per sample (if available)
    pub energy_per_sample: Option<f64>,
    /// Memory efficiency (problem size / memory used)
    pub memory_efficiency: f64,
    /// Scalability factor (1.0 = perfect linear scaling)
    pub scalability_factor: f64,
}

/// Default implementations
impl Default for TimingMetrics {
    fn default() -> Self {
        Self {
            total_time: Duration::ZERO,
            setup_time: Duration::ZERO,
            compute_time: Duration::ZERO,
            postprocess_time: Duration::ZERO,
            time_per_sample: Duration::ZERO,
            time_to_solution: None,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            best_energy: f64::INFINITY,
            avg_energy: 0.0,
            energy_std: 0.0,
            success_probability: 0.0,
            time_to_target: None,
            unique_solutions: 0,
        }
    }
}

impl Default for UtilizationMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            gpu_usage: None,
            memory_bandwidth: 0.0,
            cache_hit_rate: None,
            power_consumption: None,
        }
    }
}

/// Metrics aggregation utilities
pub mod aggregation {
    use super::*;

    /// Aggregate multiple metric collections
    pub fn aggregate_metrics(metrics: &[BenchmarkMetrics]) -> AggregatedMetrics {
        if metrics.is_empty() {
            return AggregatedMetrics::default();
        }

        let mut aggregated = AggregatedMetrics {
            num_runs: metrics.len(),
            problem_sizes: metrics.iter().map(|m| m.problem_size).collect(),
            ..Default::default()
        };

        // Timing aggregation
        let total_times: Vec<f64> = metrics
            .iter()
            .map(|m| m.timings.total_time.as_secs_f64())
            .collect();
        aggregated.avg_total_time =
            Duration::from_secs_f64(total_times.iter().sum::<f64>() / total_times.len() as f64);
        aggregated.min_total_time =
            Duration::from_secs_f64(total_times.iter().copied().fold(f64::INFINITY, f64::min));
        aggregated.max_total_time =
            Duration::from_secs_f64(total_times.iter().copied().fold(0.0, f64::max));

        // Quality aggregation
        aggregated.best_energy_overall = metrics
            .iter()
            .map(|m| m.quality.best_energy)
            .fold(f64::INFINITY, f64::min);
        aggregated.avg_success_rate = metrics
            .iter()
            .map(|m| m.quality.success_probability)
            .sum::<f64>()
            / metrics.len() as f64;

        // Memory aggregation
        aggregated.peak_memory_overall = metrics
            .iter()
            .map(|m| m.memory.peak_memory)
            .max()
            .unwrap_or(0);

        aggregated
    }

    /// Aggregated metrics across multiple runs
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct AggregatedMetrics {
        pub num_runs: usize,
        pub problem_sizes: Vec<usize>,
        pub avg_total_time: Duration,
        pub min_total_time: Duration,
        pub max_total_time: Duration,
        pub best_energy_overall: f64,
        pub avg_success_rate: f64,
        pub peak_memory_overall: usize,
    }
}

/// Statistical analysis utilities
pub mod statistics {
    use super::*;

    /// Calculate statistical measures for a set of values
    pub fn calculate_statistics(values: &[f64]) -> Statistics {
        if values.is_empty() {
            return Statistics::default();
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            f64::midpoint(sorted[sorted.len() / 2 - 1], sorted[sorted.len() / 2])
        } else {
            sorted[sorted.len() / 2]
        };

        Statistics {
            mean,
            median,
            std_dev,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            percentile_25: percentile(&sorted, 0.25),
            percentile_75: percentile(&sorted, 0.75),
            percentile_95: percentile(&sorted, 0.95),
        }
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        let k = (sorted.len() as f64 - 1.0) * p;
        let f = k.floor() as usize;
        let c = f + 1;

        if c >= sorted.len() {
            sorted[sorted.len() - 1]
        } else {
            (k - f as f64).mul_add(sorted[c] - sorted[f], sorted[f])
        }
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct Statistics {
        pub mean: f64,
        pub median: f64,
        pub std_dev: f64,
        pub min: f64,
        pub max: f64,
        pub percentile_25: f64,
        pub percentile_75: f64,
        pub percentile_95: f64,
    }
}
