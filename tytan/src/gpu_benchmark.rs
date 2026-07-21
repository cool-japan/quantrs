//! GPU benchmarking framework for performance testing and analysis.
//!
//! This module provides comprehensive benchmarking tools for GPU samplers,
//! including automated testing, scaling analysis, and energy efficiency metrics.

#![allow(dead_code)]

use crate::gpu_performance::GpuProfiler;
use crate::sampler::Sampler;
use scirs2_core::ndarray::Array2;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

#[cfg(feature = "scirs")]
use scirs2_core::gpu;

// Stub functions for missing GPU functionality
#[cfg(feature = "scirs")]
const fn get_device_count() -> usize {
    // Placeholder - in reality this would query the GPU backend
    1
}

#[cfg(feature = "scirs")]
struct GpuContext;

#[cfg(feature = "scirs")]
impl GpuContext {
    fn new(_device_id: u32) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }
}

#[cfg(feature = "scirs")]
use crate::scirs_stub::scirs2_plot::{Bar, Line, Plot, Scatter};

/// Benchmark configuration
#[derive(Clone)]
pub struct BenchmarkConfig {
    /// Problem sizes to test
    pub problem_sizes: Vec<usize>,
    /// Number of samples per problem
    pub samples_per_problem: usize,
    /// Number of repetitions for timing
    pub repetitions: usize,
    /// Test different batch sizes
    pub batch_sizes: Vec<usize>,
    /// Test different temperature schedules
    pub temperature_schedules: Vec<(f64, f64)>,
    /// Enable energy measurement
    pub measure_energy: bool,
    /// Output directory for results
    pub output_dir: String,
    /// Verbose output
    pub verbose: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            problem_sizes: vec![10, 50, 100, 250, 500, 1000],
            samples_per_problem: 1000,
            repetitions: 5,
            batch_sizes: vec![32, 64, 128, 256, 512, 1024],
            temperature_schedules: vec![(10.0, 0.01), (5.0, 0.1), (1.0, 0.01)],
            measure_energy: false,
            output_dir: "benchmark_results".to_string(),
            verbose: true,
        }
    }
}

/// Benchmark results
#[derive(Clone)]
pub struct BenchmarkResults {
    /// Results by problem size
    pub size_results: HashMap<usize, SizeResults>,
    /// Results by batch size
    pub batch_results: HashMap<usize, BatchResults>,
    /// Temperature schedule comparison
    pub temp_results: HashMap<String, TempResults>,
    /// Energy efficiency metrics
    pub energy_metrics: Option<EnergyMetrics>,
    /// Device information
    pub device_info: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone)]
pub struct SizeResults {
    /// Average execution time
    pub avg_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Throughput (samples/second)
    pub throughput: f64,
    /// Solution quality (best energy found)
    pub best_energy: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
}

#[derive(Clone)]
pub struct BatchResults {
    /// Execution time
    pub exec_time: Duration,
    /// GPU utilization, when real hardware telemetry is available.
    /// `None` (rather than a fabricated constant) when running on the
    /// CPU-fallback `scirs2_core::gpu` backend, which exposes no real
    /// utilization counters.
    pub gpu_utilization: Option<f64>,
    /// Memory bandwidth utilization; see [`Self::gpu_utilization`] for the
    /// same "no fabricated values" contract.
    pub bandwidth_util: Option<f64>,
}

#[derive(Clone)]
pub struct TempResults {
    /// Convergence time
    pub convergence_time: Duration,
    /// Final solution quality
    pub final_quality: f64,
    /// Number of internal solver iterations to convergence, when the
    /// `Sampler` implementation reports it. The generic `Sampler` trait
    /// does not expose an iteration count, so this is honestly `None`
    /// rather than a fabricated constant.
    pub iterations: Option<usize>,
}

#[derive(Clone)]
pub struct EnergyMetrics {
    /// Power consumption (watts)
    pub avg_power: f64,
    /// Energy per sample (joules)
    pub energy_per_sample: f64,
    /// Performance per watt
    pub perf_per_watt: f64,
}

/// GPU benchmark runner
pub struct GpuBenchmark<S: Sampler> {
    /// Sampler to benchmark
    sampler: S,
    /// Configuration
    config: BenchmarkConfig,
    /// Performance profiler
    profiler: GpuProfiler,
}

impl<S: Sampler> GpuBenchmark<S> {
    /// Create new benchmark
    pub fn new(sampler: S, config: BenchmarkConfig) -> Self {
        Self {
            sampler,
            config,
            profiler: GpuProfiler::new(),
        }
    }

    /// Run complete benchmark suite
    pub fn run_benchmark(&mut self) -> Result<BenchmarkResults, String> {
        if self.config.verbose {
            println!("Starting GPU benchmark...");
        }

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)
            .map_err(|e| format!("Failed to create output directory: {e}"))?;

        let mut results = BenchmarkResults {
            size_results: HashMap::new(),
            batch_results: HashMap::new(),
            temp_results: HashMap::new(),
            energy_metrics: None,
            device_info: self.get_device_info(),
            timestamp: chrono::Utc::now(),
        };

        // Run problem size scaling tests
        self.benchmark_problem_sizes(&mut results)?;

        // Run batch size optimization tests
        self.benchmark_batch_sizes(&mut results)?;

        // Run temperature schedule comparison
        self.benchmark_temperature_schedules(&mut results)?;

        // Measure energy efficiency if enabled
        if self.config.measure_energy {
            self.benchmark_energy_efficiency(&mut results)?;
        }

        // Generate report
        self.generate_report(&results)?;

        Ok(results)
    }

    /// Benchmark different problem sizes
    fn benchmark_problem_sizes(&mut self, results: &mut BenchmarkResults) -> Result<(), String> {
        if self.config.verbose {
            println!("\nBenchmarking problem size scaling...");
        }

        for &size in &self.config.problem_sizes {
            if self.config.verbose {
                println!("  Testing size {size}...");
            }

            // Generate random QUBO problem
            let (qubo, var_map) = generate_random_qubo(size);

            let mut times = Vec::new();
            let mut best_energy = f64::INFINITY;

            for rep in 0..self.config.repetitions {
                let start = Instant::now();

                let solutions = self
                    .sampler
                    .run_qubo(
                        &(qubo.clone(), var_map.clone()),
                        self.config.samples_per_problem,
                    )
                    .map_err(|e| e.to_string())?;

                let elapsed = start.elapsed();
                times.push(elapsed);

                if let Some(best) = solutions.first() {
                    best_energy = best_energy.min(best.energy);
                }

                if self.config.verbose && rep == 0 {
                    println!("    First run: {elapsed:?}");
                }
            }

            // Calculate statistics
            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let variance = times
                .iter()
                .map(|&t| {
                    let diff = if t > avg_time {
                        t.checked_sub(avg_time).unwrap_or_default().as_secs_f64()
                    } else {
                        avg_time.checked_sub(t).unwrap_or_default().as_secs_f64()
                    };
                    diff * diff
                })
                .sum::<f64>()
                / times.len() as f64;
            let std_dev = Duration::from_secs_f64(variance.sqrt());

            let throughput = self.config.samples_per_problem as f64 / avg_time.as_secs_f64();

            results.size_results.insert(
                size,
                SizeResults {
                    avg_time,
                    std_dev,
                    throughput,
                    best_energy,
                    memory_usage: estimate_memory_usage(size, self.config.samples_per_problem),
                },
            );
        }

        Ok(())
    }

    /// Benchmark different batch sizes
    fn benchmark_batch_sizes(&mut self, results: &mut BenchmarkResults) -> Result<(), String> {
        if self.config.verbose {
            println!("\nBenchmarking batch size optimization...");
        }

        // Use a fixed medium-sized problem
        let test_size = 100;
        let (qubo, var_map) = generate_random_qubo(test_size);

        for &batch_size in &self.config.batch_sizes {
            if self.config.verbose {
                println!("  Testing batch size {batch_size}...");
            }

            // Configure sampler with batch size (if supported)
            // This would need to be implemented in the actual sampler

            let start = Instant::now();

            let _solutions = self
                .sampler
                .run_qubo(&(qubo.clone(), var_map.clone()), batch_size)
                .map_err(|e| e.to_string())?;

            let elapsed = start.elapsed();

            // Record real timing/throughput with the profiler. GPU
            // utilization and memory-bandwidth utilization genuinely
            // require hardware performance counters that the CPU-fallback
            // `scirs2_core::gpu` backend used here does not provide, so
            // those two metrics are honestly reported as unavailable
            // (`None`) instead of fabricated constants.
            self.profiler.record_kernel_time("run_qubo", elapsed);
            self.profiler.update_throughput(batch_size, elapsed);

            results.batch_results.insert(
                batch_size,
                BatchResults {
                    exec_time: elapsed,
                    gpu_utilization: None,
                    bandwidth_util: None,
                },
            );
        }

        Ok(())
    }

    /// Benchmark temperature schedules
    fn benchmark_temperature_schedules(
        &mut self,
        results: &mut BenchmarkResults,
    ) -> Result<(), String> {
        if self.config.verbose {
            println!("\nBenchmarking temperature schedules...");
        }

        let test_size = 50;
        let (qubo, var_map) = generate_random_qubo(test_size);

        for &(initial, final_) in &self.config.temperature_schedules {
            let schedule_name = format!("{initial:.1}-{final_:.2}");

            if self.config.verbose {
                println!("  Testing schedule {schedule_name}...");
            }

            // Would need to configure sampler with temperature schedule

            let start = Instant::now();

            let solutions = self
                .sampler
                .run_qubo(
                    &(qubo.clone(), var_map.clone()),
                    self.config.samples_per_problem,
                )
                .map_err(|e| e.to_string())?;

            let elapsed = start.elapsed();

            let final_quality = solutions.first().map_or(f64::INFINITY, |s| s.energy);

            results.temp_results.insert(
                schedule_name,
                TempResults {
                    convergence_time: elapsed,
                    final_quality,
                    // The generic `Sampler` trait does not report its
                    // internal iteration count, so this is honestly `None`
                    // rather than a fabricated constant.
                    iterations: None,
                },
            );
        }

        Ok(())
    }

    /// Benchmark energy efficiency
    ///
    /// Real power-draw measurement requires genuine GPU/board power-monitor
    /// hardware (e.g. NVML/`rocm-smi`), which is not available through the
    /// CPU-fallback `scirs2_core::gpu` backend used by this benchmark
    /// runner. Rather than fabricate a plausible-looking wattage (as the
    /// previous implementation did with a hardcoded `150.0 W`), this
    /// honestly reports that energy metrics are unavailable. Callers that
    /// explicitly opt in via `BenchmarkConfig::measure_energy` therefore see
    /// a clear error instead of invented numbers.
    fn benchmark_energy_efficiency(
        &mut self,
        _results: &mut BenchmarkResults,
    ) -> Result<(), String> {
        if self.config.verbose {
            println!("\nMeasuring energy efficiency...");
        }

        Err(
            "Energy efficiency measurement requires real GPU/board power-monitor hardware \
             (e.g. NVML or rocm-smi), which is not available through this benchmark's \
             CPU-fallback backend; set BenchmarkConfig::measure_energy = false to skip this step."
                .to_string(),
        )
    }

    /// Get device information
    fn get_device_info(&self) -> String {
        #[cfg(feature = "scirs")]
        {
            // The `scirs` stub backend exposes a GpuContext but no physical device,
            // so honestly report that detailed specs are unavailable rather than
            // fabricating them (the previous code returned hardcoded 8192 MB / 64
            // CU / 1500 MHz). A real backend would query genuine device specs here.
            if GpuContext::new(0).is_ok() {
                return "GPU device info unavailable (scirs stub backend has no physical device)"
                    .to_string();
            }
        }

        "GPU information not available".to_string()
    }

    /// Generate benchmark report
    fn generate_report(&self, results: &BenchmarkResults) -> Result<(), String> {
        // Generate plots
        self.plot_scaling_results(results)?;
        self.plot_batch_optimization(results)?;
        self.plot_temperature_comparison(results)?;

        // Generate text report
        let report_path = format!("{}/benchmark_report.txt", self.config.output_dir);
        let mut file =
            File::create(&report_path).map_err(|e| format!("Failed to create report file: {e}"))?;

        writeln!(file, "GPU Benchmark Report")
            .map_err(|e| format!("Failed to write report: {e}"))?;
        writeln!(file, "====================")
            .map_err(|e| format!("Failed to write report: {e}"))?;
        writeln!(file, "Timestamp: {}", results.timestamp)
            .map_err(|e| format!("Failed to write report: {e}"))?;
        writeln!(file, "Device: {}", results.device_info)
            .map_err(|e| format!("Failed to write report: {e}"))?;
        writeln!(file).map_err(|e| format!("Failed to write report: {e}"))?;

        writeln!(file, "Problem Size Scaling:")
            .map_err(|e| format!("Failed to write report: {e}"))?;
        for (size, res) in &results.size_results {
            writeln!(
                file,
                "  Size {}: {:.2} ms avg, {:.0} samples/sec",
                size,
                res.avg_time.as_secs_f64() * 1000.0,
                res.throughput
            )
            .map_err(|e| format!("Failed to write report: {e}"))?;
        }

        if let Some(energy) = &results.energy_metrics {
            writeln!(file).map_err(|e| format!("Failed to write report: {e}"))?;
            writeln!(file, "Energy Efficiency:")
                .map_err(|e| format!("Failed to write report: {e}"))?;
            writeln!(file, "  Average Power: {:.1} W", energy.avg_power)
                .map_err(|e| format!("Failed to write report: {e}"))?;
            writeln!(
                file,
                "  Energy per Sample: {:.3} mJ",
                energy.energy_per_sample * 1000.0
            )
            .map_err(|e| format!("Failed to write report: {e}"))?;
            writeln!(
                file,
                "  Performance per Watt: {:.1} samples/J",
                energy.perf_per_watt
            )
            .map_err(|e| format!("Failed to write report: {e}"))?;
        }

        if self.config.verbose {
            println!("\nReport saved to: {report_path}");
        }

        Ok(())
    }

    /// Plot scaling results
    fn plot_scaling_results(&self, results: &BenchmarkResults) -> Result<(), String> {
        #[cfg(feature = "scirs")]
        {
            let mut plot = Plot::new();

            let mut sizes = Vec::new();
            let mut times = Vec::new();
            let mut throughputs = Vec::new();

            for (size, res) in &results.size_results {
                sizes.push(*size as f64);
                times.push(res.avg_time.as_secs_f64() * 1000.0);
                throughputs.push(res.throughput);
            }

            // Sort by size
            let mut indices: Vec<usize> = (0..sizes.len()).collect();
            indices.sort_by_key(|&i| sizes[i] as usize);

            let sizes: Vec<f64> = indices.iter().map(|&i| sizes[i]).collect();
            let times: Vec<f64> = indices.iter().map(|&i| times[i]).collect();
            let throughputs: Vec<f64> = indices.iter().map(|&i| throughputs[i]).collect();

            let time_line = Line::new(sizes.clone(), times).name("Execution Time (ms)");
            let throughput_line = Line::new(sizes, throughputs).name("Throughput (samples/sec)");

            plot.add_trace(time_line);
            plot.add_trace(throughput_line);
            plot.set_title("GPU Performance Scaling");
            plot.set_xlabel("Problem Size");
            plot.set_ylabel("Performance");

            let plot_path = format!("{}/scaling_plot.html", self.config.output_dir);
            plot.save(&plot_path).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Plot batch size optimization
    fn plot_batch_optimization(&self, results: &BenchmarkResults) -> Result<(), String> {
        #[cfg(feature = "scirs")]
        {
            let mut plot = Plot::new();

            let mut batch_sizes = Vec::new();
            let mut exec_times = Vec::new();

            for (batch, res) in &results.batch_results {
                batch_sizes.push(*batch as f64);
                exec_times.push(res.exec_time.as_secs_f64() * 1000.0);
            }

            let time_bar = Bar::new(
                batch_sizes.iter().map(|&b| b.to_string()).collect(),
                exec_times,
            )
            .name("Execution Time (ms)");

            plot.add_trace(time_bar);
            // GPU utilization is honestly `None` on this benchmark's
            // CPU-fallback backend (see `BatchResults::gpu_utilization`), so
            // there is no real utilization series to plot here.
            plot.set_title("Batch Size Optimization");
            plot.set_xlabel("Batch Size");

            let plot_path = format!("{}/batch_optimization.html", self.config.output_dir);
            plot.save(&plot_path).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Plot temperature schedule comparison
    fn plot_temperature_comparison(&self, results: &BenchmarkResults) -> Result<(), String> {
        #[cfg(feature = "scirs")]
        {
            let mut plot = Plot::new();

            let schedules: Vec<String> = results.temp_results.keys().cloned().collect();
            let qualities: Vec<f64> = schedules
                .iter()
                .map(|s| results.temp_results[s].final_quality)
                .collect();

            let bar = Bar::new(schedules, qualities).name("Final Solution Quality");

            plot.add_trace(bar);
            plot.set_title("Temperature Schedule Comparison");
            plot.set_xlabel("Schedule (Initial-Final)");
            plot.set_ylabel("Solution Quality");

            let plot_path = format!("{}/temperature_comparison.html", self.config.output_dir);
            plot.save(&plot_path).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

/// Generate random QUBO problem for benchmarking
fn generate_random_qubo(size: usize) -> (Array2<f64>, HashMap<String, usize>) {
    use scirs2_core::random::prelude::*;
    let mut rng = thread_rng();

    let mut qubo = Array2::zeros((size, size));

    // Generate random coefficients
    for i in 0..size {
        // Linear terms
        qubo[[i, i]] = rng.random_range(-1.0..1.0);

        // Quadratic terms
        for j in i + 1..size {
            let value = rng.random_range(-2.0..2.0);
            qubo[[i, j]] = value;
            qubo[[j, i]] = value;
        }
    }

    // Create variable map
    let mut var_map = HashMap::new();
    for i in 0..size {
        var_map.insert(format!("x{i}"), i);
    }

    (qubo, var_map)
}

/// Estimate memory usage in MB
fn estimate_memory_usage(problem_size: usize, batch_size: usize) -> f64 {
    // QUBO matrix: n^2 * 8 bytes
    let matrix_size = problem_size * problem_size * 8;

    // States: batch_size * n bytes
    let states_size = batch_size * problem_size;

    // Additional overhead
    let overhead = matrix_size / 10;

    (matrix_size + states_size + overhead) as f64 / (1024.0 * 1024.0)
}

/// Compare multiple GPU implementations
pub struct GpuComparison {
    /// Configurations to compare
    configs: Vec<ComparisonConfig>,
    /// Benchmark configuration
    benchmark_config: BenchmarkConfig,
}

struct ComparisonConfig {
    name: String,
    sampler: Box<dyn Sampler>,
}

impl GpuComparison {
    /// Create new comparison
    pub const fn new(benchmark_config: BenchmarkConfig) -> Self {
        Self {
            configs: Vec::new(),
            benchmark_config,
        }
    }

    /// Add implementation to compare
    pub fn add_implementation(&mut self, name: &str, sampler: Box<dyn Sampler>) {
        self.configs.push(ComparisonConfig {
            name: name.to_string(),
            sampler,
        });
    }

    /// Run comparison
    ///
    /// Actually runs each registered implementation's sampler on the same
    /// random QUBO problem and measures its real wall-clock throughput and
    /// best solution energy. `configs` already stores samplers as
    /// `Box<dyn Sampler>`, so no additional trait-object plumbing is needed
    /// (the previous implementation's "would need trait object support"
    /// comment was stale: it inserted the exact same hardcoded
    /// `ImplementationResult` for every registered implementation without
    /// ever invoking `config.sampler`).
    pub fn run_comparison(&mut self) -> Result<ComparisonResults, String> {
        let mut results = ComparisonResults {
            implementations: HashMap::new(),
            best_performer: String::new(),
        };

        let test_size = self
            .benchmark_config
            .problem_sizes
            .first()
            .copied()
            .unwrap_or(50);
        let (qubo, var_map) = generate_random_qubo(test_size);
        let shots = self.benchmark_config.samples_per_problem.max(1);
        let repetitions = self.benchmark_config.repetitions.max(1);

        for config in &mut self.configs {
            println!("\nBenchmarking {}...", config.name);

            let mut times = Vec::with_capacity(repetitions);
            let mut best_quality = f64::INFINITY;

            for _ in 0..repetitions {
                let start = Instant::now();
                let solutions = config
                    .sampler
                    .run_qubo(&(qubo.clone(), var_map.clone()), shots)
                    .map_err(|e| e.to_string())?;
                times.push(start.elapsed());

                if let Some(best) = solutions.first() {
                    best_quality = best_quality.min(best.energy);
                }
            }

            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let avg_performance = if avg_time.as_secs_f64() > 0.0 {
                shots as f64 / avg_time.as_secs_f64()
            } else {
                0.0
            };

            // A real (data-dependent), if simplified, memory-efficiency
            // proxy: how much smaller the actual estimated working set is
            // relative to a generous 1 GB budget, bounded to [0, 1].
            let estimated_mb = estimate_memory_usage(test_size, shots);
            let memory_efficiency = (1.0 - (estimated_mb / 1024.0).min(1.0)).max(0.0);

            results.implementations.insert(
                config.name.clone(),
                ImplementationResult {
                    avg_performance,
                    best_quality,
                    memory_efficiency,
                },
            );
        }

        // Determine best performer
        results.best_performer = results
            .implementations
            .iter()
            .max_by(|a, b| {
                a.1.avg_performance
                    .partial_cmp(&b.1.avg_performance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.clone())
            .unwrap_or_default();

        Ok(results)
    }
}

/// Comparison results
pub struct ComparisonResults {
    pub implementations: HashMap<String, ImplementationResult>,
    pub best_performer: String,
}

pub struct ImplementationResult {
    pub avg_performance: f64,
    pub best_quality: f64,
    pub memory_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config() {
        let mut config = BenchmarkConfig::default();
        assert!(!config.problem_sizes.is_empty());
        assert!(config.samples_per_problem > 0);
    }

    #[test]
    fn test_generate_random_qubo() {
        let (qubo, var_map) = generate_random_qubo(10);
        assert_eq!(qubo.shape(), &[10, 10]);
        assert_eq!(var_map.len(), 10);
    }

    #[test]
    fn test_memory_estimation() {
        let mem = estimate_memory_usage(100, 1000);
        assert!(mem > 0.0);
        assert!(mem < 1000.0); // Should be reasonable
    }

    #[test]
    fn test_run_comparison_actually_benchmarks_each_sampler() {
        use crate::sampler::SASampler;

        let mut bench_config = BenchmarkConfig::default();
        bench_config.problem_sizes = vec![8];
        bench_config.samples_per_problem = 5;
        bench_config.repetitions = 2;

        let mut comparison = GpuComparison::new(bench_config);
        comparison.add_implementation("sa-1", Box::new(SASampler::new(Some(1))));
        comparison.add_implementation("sa-2", Box::new(SASampler::new(Some(2))));

        let results = comparison
            .run_comparison()
            .expect("comparison should succeed");

        // The old fabricated implementation inserted the exact same
        // ImplementationResult{avg_performance: 1000.0, best_quality: -100.0,
        // memory_efficiency: 0.8} for every registered implementation
        // without ever calling the sampler. Now every implementation must
        // have actually run and produced a real (non-placeholder) energy.
        assert_eq!(results.implementations.len(), 2);
        for (name, result) in &results.implementations {
            assert!(
                result.best_quality.is_finite(),
                "{name} should have produced a real, finite best energy"
            );
            assert!(result.avg_performance > 0.0);
        }
        assert!(!results.best_performer.is_empty());
    }

    #[test]
    fn test_benchmark_energy_efficiency_is_honest_error_not_fabrication() {
        use crate::sampler::SASampler;

        let mut config = BenchmarkConfig::default();
        config.problem_sizes = vec![8];
        config.samples_per_problem = 5;
        config.repetitions = 1;
        config.measure_energy = true;
        config.verbose = false;

        let mut benchmark = GpuBenchmark::new(SASampler::new(Some(1)), config);
        let mut results = BenchmarkResults {
            size_results: HashMap::new(),
            batch_results: HashMap::new(),
            temp_results: HashMap::new(),
            energy_metrics: None,
            device_info: String::new(),
            timestamp: chrono::Utc::now(),
        };

        // The old fabricated implementation always returned
        // Ok(EnergyMetrics { avg_power: 150.0, .. }) regardless of hardware;
        // real power-monitoring hardware is not available here, so this
        // must now be an honest error instead of invented wattage.
        let outcome = benchmark.benchmark_energy_efficiency(&mut results);
        assert!(outcome.is_err());
        assert!(results.energy_metrics.is_none());
    }

    #[test]
    fn test_batch_results_do_not_fabricate_gpu_utilization() {
        use crate::sampler::SASampler;
        use quantrs2_anneal::simulator::AnnealingParams;

        let mut config = BenchmarkConfig::default();
        config.batch_sizes = vec![4, 8];
        config.verbose = false;

        // `benchmark_batch_sizes` uses a fixed 100-variable test problem
        // internally; keep the annealer's own workload tiny so this test
        // runs quickly regardless.
        let params = AnnealingParams {
            num_sweeps: 10,
            num_repetitions: 1,
            ..AnnealingParams::new()
        };
        let mut benchmark = GpuBenchmark::new(SASampler::with_params(Some(1), params), config);
        let mut results = BenchmarkResults {
            size_results: HashMap::new(),
            batch_results: HashMap::new(),
            temp_results: HashMap::new(),
            energy_metrics: None,
            device_info: String::new(),
            timestamp: chrono::Utc::now(),
        };

        benchmark
            .benchmark_batch_sizes(&mut results)
            .expect("batch benchmarking should succeed");

        // The old fabricated implementation always reported
        // gpu_utilization=0.75 / bandwidth_util=0.60 regardless of hardware.
        assert!(!results.batch_results.is_empty());
        for res in results.batch_results.values() {
            assert!(res.gpu_utilization.is_none());
            assert!(res.bandwidth_util.is_none());
        }
    }
}
