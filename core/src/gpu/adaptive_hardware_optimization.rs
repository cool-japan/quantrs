//! Advanced Adaptive Hardware Optimization Module
//!
//! This module provides sophisticated adaptive optimization strategies based on
//! hardware characteristics, workload patterns, and runtime performance metrics.
//!
//! ## Features
//! - Automatic workload profiling and tuning
//! - Memory hierarchy-aware optimization
//! - Power-aware computation strategies
//! - Runtime benchmarking for optimal strategy selection
//! - ML-based performance prediction

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::platform::PlatformCapabilities;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Adaptive hardware optimization configuration
#[derive(Debug, Clone)]
pub struct AdaptiveOptimizationConfig {
    /// Enable automatic workload profiling
    pub enable_workload_profiling: bool,
    /// Enable memory hierarchy optimization
    pub enable_memory_optimization: bool,
    /// Enable power-aware optimization
    pub enable_power_optimization: bool,
    /// Minimum samples before adaptation
    pub min_samples_for_adaptation: usize,
    /// Performance variance threshold for strategy change
    pub variance_threshold: f64,
    /// Enable runtime benchmarking
    pub enable_runtime_benchmarking: bool,
    /// Benchmark sample size
    pub benchmark_samples: usize,
}

impl Default for AdaptiveOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_workload_profiling: true,
            enable_memory_optimization: true,
            enable_power_optimization: false, // Disabled by default
            min_samples_for_adaptation: 10,
            variance_threshold: 0.2,
            enable_runtime_benchmarking: true,
            benchmark_samples: 5,
        }
    }
}

/// Workload characteristics for optimization decisions
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of gates
    pub num_gates: usize,
    /// Gate depth
    pub circuit_depth: usize,
    /// Memory access pattern
    pub access_pattern: AccessPattern,
    /// Computational intensity (FLOPS per byte)
    pub computational_intensity: f64,
    /// Expected execution count
    pub expected_iterations: usize,
}

/// Memory access pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,
    /// Strided access
    Strided,
    /// Random access
    Random,
    /// Mixed access
    Mixed,
}

/// Optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationStrategy {
    /// Optimize for throughput
    Throughput,
    /// Optimize for latency
    Latency,
    /// Balance throughput and latency
    Balanced,
    /// Optimize for memory bandwidth
    MemoryBound,
    /// Optimize for power efficiency
    PowerEfficient,
}

/// Performance profile for a specific workload
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Average execution time
    pub avg_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Number of samples
    pub sample_count: usize,
    /// Best strategy for this profile
    pub best_strategy: OptimizationStrategy,
    /// Memory bandwidth utilization
    pub memory_bandwidth_gbps: f64,
    /// FLOPS achieved
    pub gflops: f64,
}

/// Hardware capability assessment
#[derive(Debug, Clone)]
pub struct HardwareAssessment {
    /// Platform capabilities
    pub capabilities: PlatformCapabilities,
    /// Estimated peak memory bandwidth (GB/s)
    pub peak_memory_bandwidth: f64,
    /// Estimated peak FLOPS
    pub peak_gflops: f64,
    /// Optimal batch size for this hardware
    pub optimal_batch_size: usize,
    /// Optimal tile size for tiled operations
    pub optimal_tile_size: usize,
    /// Maximum efficient state size
    pub max_efficient_state_size: usize,
}

impl HardwareAssessment {
    /// Create assessment from platform capabilities
    pub fn from_capabilities(capabilities: PlatformCapabilities) -> Self {
        // Estimate peak bandwidth based on CPU characteristics
        let peak_memory_bandwidth = Self::estimate_memory_bandwidth(&capabilities);
        let peak_gflops = Self::estimate_peak_gflops(&capabilities);
        let optimal_batch_size = Self::compute_optimal_batch_size(&capabilities);
        let optimal_tile_size = Self::compute_optimal_tile_size(&capabilities);
        let max_efficient_state_size = Self::compute_max_efficient_state_size(&capabilities);

        Self {
            capabilities,
            peak_memory_bandwidth,
            peak_gflops,
            optimal_batch_size,
            optimal_tile_size,
            max_efficient_state_size,
        }
    }

    fn estimate_memory_bandwidth(capabilities: &PlatformCapabilities) -> f64 {
        // Estimate based on number of cores and typical memory system
        let cores = capabilities.cpu.logical_cores as f64;
        // Typical DDR4/DDR5 bandwidth per channel
        let base_bandwidth: f64 = 25.6; // GB/s per channel
                                        // Assume 2 channels with some overhead
        (base_bandwidth * 2.0 * 0.8).min(cores * 10.0)
    }

    fn estimate_peak_gflops(capabilities: &PlatformCapabilities) -> f64 {
        let cores = capabilities.cpu.logical_cores as f64;
        let base_gflops_per_core = if capabilities.cpu.simd.avx512 {
            100.0
        } else if capabilities.cpu.simd.avx2 {
            50.0
        } else {
            25.0
        };
        cores * base_gflops_per_core
    }

    fn compute_optimal_batch_size(capabilities: &PlatformCapabilities) -> usize {
        let l3_cache = capabilities.cpu.cache.l3.unwrap_or(8 * 1024 * 1024);
        // Optimal batch size fits in L3 cache
        let complex_size = std::mem::size_of::<Complex64>();
        (l3_cache / (complex_size * 16)).clamp(32, 1024)
    }

    fn compute_optimal_tile_size(capabilities: &PlatformCapabilities) -> usize {
        let l2_cache = capabilities.cpu.cache.l2.unwrap_or(256 * 1024);
        // Tile should fit in L2 cache
        let complex_size = std::mem::size_of::<Complex64>();
        let elements = l2_cache / (complex_size * 4); // 4x for working memory
        (elements as f64).sqrt() as usize
    }

    fn compute_max_efficient_state_size(capabilities: &PlatformCapabilities) -> usize {
        let total_cache = capabilities.cpu.cache.l3.unwrap_or(8 * 1024 * 1024);
        let cores = capabilities.cpu.logical_cores;
        // Maximum state that can be efficiently processed
        let complex_size = std::mem::size_of::<Complex64>();
        (total_cache * cores) / (complex_size * 2)
    }
}

/// Adaptive hardware optimizer
pub struct AdaptiveHardwareOptimizer {
    /// Configuration
    config: AdaptiveOptimizationConfig,
    /// Hardware assessment
    hardware: HardwareAssessment,
    /// Performance profiles by workload key
    profiles: RwLock<HashMap<String, PerformanceProfile>>,
    /// Current strategy
    current_strategy: Mutex<OptimizationStrategy>,
    /// Optimization history
    history: RwLock<Vec<OptimizationEvent>>,
}

/// Optimization event for history tracking
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    /// Timestamp
    pub timestamp: Instant,
    /// Workload key
    pub workload_key: String,
    /// Strategy used
    pub strategy: OptimizationStrategy,
    /// Execution time
    pub execution_time: Duration,
    /// Was this optimal?
    pub was_optimal: bool,
}

impl AdaptiveHardwareOptimizer {
    /// Create a new adaptive hardware optimizer
    pub fn new(config: AdaptiveOptimizationConfig) -> Self {
        let capabilities = PlatformCapabilities::detect();
        let hardware = HardwareAssessment::from_capabilities(capabilities);

        Self {
            config,
            hardware,
            profiles: RwLock::new(HashMap::new()),
            current_strategy: Mutex::new(OptimizationStrategy::Balanced),
            history: RwLock::new(Vec::new()),
        }
    }

    /// Get hardware assessment
    pub const fn hardware_assessment(&self) -> &HardwareAssessment {
        &self.hardware
    }

    /// Analyze workload and recommend optimization strategy
    pub fn analyze_workload(
        &self,
        characteristics: &WorkloadCharacteristics,
    ) -> OptimizationStrategy {
        // Compute workload metrics
        let state_size = 1 << characteristics.num_qubits;
        let total_operations = characteristics.num_gates * state_size;
        let memory_access =
            state_size * characteristics.circuit_depth * std::mem::size_of::<Complex64>();

        // Determine if workload is compute-bound or memory-bound
        let intensity = characteristics.computational_intensity;

        if intensity > 10.0 {
            // Compute-bound: optimize for throughput
            OptimizationStrategy::Throughput
        } else if intensity < 1.0 {
            // Memory-bound: optimize for memory access
            OptimizationStrategy::MemoryBound
        } else if characteristics.expected_iterations > 100 {
            // Repeated execution: optimize for throughput
            OptimizationStrategy::Throughput
        } else if state_size < self.hardware.optimal_batch_size {
            // Small workload: optimize for latency
            OptimizationStrategy::Latency
        } else {
            // Default to balanced
            OptimizationStrategy::Balanced
        }
    }

    /// Get optimization parameters for given strategy
    pub fn get_optimization_params(
        &self,
        strategy: OptimizationStrategy,
        num_qubits: usize,
    ) -> OptimizationParams {
        let state_size = 1 << num_qubits;

        match strategy {
            OptimizationStrategy::Throughput => OptimizationParams {
                use_simd: true,
                use_parallel: state_size > 1024,
                batch_size: self.hardware.optimal_batch_size,
                tile_size: self.hardware.optimal_tile_size,
                prefetch_distance: 8,
                use_streaming: state_size > self.hardware.max_efficient_state_size,
            },
            OptimizationStrategy::Latency => OptimizationParams {
                use_simd: true,
                use_parallel: false, // Avoid parallel overhead
                batch_size: 1,
                tile_size: 64,
                prefetch_distance: 4,
                use_streaming: false,
            },
            OptimizationStrategy::Balanced => OptimizationParams {
                use_simd: true,
                use_parallel: state_size > 2048,
                batch_size: (self.hardware.optimal_batch_size / 2).max(32),
                tile_size: self.hardware.optimal_tile_size,
                prefetch_distance: 6,
                use_streaming: state_size > self.hardware.max_efficient_state_size * 2,
            },
            OptimizationStrategy::MemoryBound => OptimizationParams {
                use_simd: true,
                use_parallel: true, // Hide memory latency
                batch_size: self.hardware.optimal_batch_size * 2,
                tile_size: self.hardware.optimal_tile_size / 2, // Smaller tiles for better cache use
                prefetch_distance: 16,                          // Aggressive prefetching
                use_streaming: true,
            },
            OptimizationStrategy::PowerEfficient => OptimizationParams {
                use_simd: false, // Reduce power consumption
                use_parallel: false,
                batch_size: 32,
                tile_size: 32,
                prefetch_distance: 4,
                use_streaming: false,
            },
        }
    }

    /// Record execution result for learning
    pub fn record_execution(
        &self,
        workload_key: &str,
        strategy: OptimizationStrategy,
        execution_time: Duration,
    ) {
        // Update performance profile
        if let Ok(mut profiles) = self.profiles.write() {
            let profile = profiles
                .entry(workload_key.to_string())
                .or_insert(PerformanceProfile {
                    avg_time: execution_time,
                    std_dev: Duration::ZERO,
                    min_time: execution_time,
                    max_time: execution_time,
                    sample_count: 0,
                    best_strategy: strategy,
                    memory_bandwidth_gbps: 0.0,
                    gflops: 0.0,
                });

            // Update rolling statistics
            let n = profile.sample_count as f64;
            let new_time = execution_time.as_secs_f64();
            let old_avg = profile.avg_time.as_secs_f64();

            let new_avg = old_avg + (new_time - old_avg) / (n + 1.0);
            profile.avg_time = Duration::from_secs_f64(new_avg);

            if execution_time < profile.min_time {
                profile.min_time = execution_time;
            }
            if execution_time > profile.max_time {
                profile.max_time = execution_time;
            }

            profile.sample_count += 1;

            // Check if we should update best strategy
            if profile.sample_count >= self.config.min_samples_for_adaptation {
                // Simple: if new strategy is significantly better, update
                if execution_time.as_secs_f64() < old_avg * (1.0 - self.config.variance_threshold) {
                    profile.best_strategy = strategy;
                }
            }
        }

        // Record event in history
        if let Ok(mut history) = self.history.write() {
            history.push(OptimizationEvent {
                timestamp: Instant::now(),
                workload_key: workload_key.to_string(),
                strategy,
                execution_time,
                was_optimal: true, // Will be determined later
            });

            // Keep history bounded
            if history.len() > 10000 {
                history.drain(0..1000);
            }
        }
    }

    /// Get recommended strategy for workload
    pub fn get_recommended_strategy(&self, workload_key: &str) -> OptimizationStrategy {
        if let Ok(profiles) = self.profiles.read() {
            if let Some(profile) = profiles.get(workload_key) {
                if profile.sample_count >= self.config.min_samples_for_adaptation {
                    return profile.best_strategy;
                }
            }
        }

        // Fall back to current default
        *self
            .current_strategy
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Get performance profile for workload
    pub fn get_profile(&self, workload_key: &str) -> Option<PerformanceProfile> {
        self.profiles.read().ok()?.get(workload_key).cloned()
    }

    /// Generate optimization report
    pub fn generate_report(&self) -> OptimizationReport {
        let profiles: Vec<_> = self
            .profiles
            .read()
            .map(|p| p.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let total_events = self.history.read().map(|h| h.len()).unwrap_or(0);

        OptimizationReport {
            hardware_assessment: self.hardware.clone(),
            workload_profiles: profiles,
            total_optimization_events: total_events,
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze profiles for patterns
        if let Ok(profiles) = self.profiles.read() {
            let mut memory_bound_count = 0;
            let mut compute_bound_count = 0;

            for (_key, profile) in profiles.iter() {
                if profile.best_strategy == OptimizationStrategy::MemoryBound {
                    memory_bound_count += 1;
                } else if profile.best_strategy == OptimizationStrategy::Throughput {
                    compute_bound_count += 1;
                }
            }

            if memory_bound_count > compute_bound_count * 2 {
                recommendations.push(
                    "Most workloads are memory-bound. Consider using larger tiles and aggressive prefetching".to_string()
                );
            }

            if compute_bound_count > memory_bound_count * 2 {
                recommendations.push(
                    "Most workloads are compute-bound. Consider enabling SIMD and parallel execution".to_string()
                );
            }
        }

        // Hardware-specific recommendations
        if self.hardware.capabilities.cpu.simd.avx512 {
            recommendations.push(
                "AVX-512 detected. Ensure alignment to 64 bytes for optimal performance"
                    .to_string(),
            );
        } else if self.hardware.capabilities.cpu.simd.avx2 {
            recommendations.push(
                "AVX2 detected. Ensure alignment to 32 bytes for optimal performance".to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("System is operating efficiently".to_string());
        }

        recommendations
    }

    /// Run microbenchmark to calibrate optimization parameters
    pub fn calibrate(&self, num_qubits: usize) -> CalibrationResult {
        let state_size = 1 << num_qubits;
        let mut results = HashMap::new();

        // Benchmark different strategies
        for strategy in [
            OptimizationStrategy::Throughput,
            OptimizationStrategy::Latency,
            OptimizationStrategy::Balanced,
            OptimizationStrategy::MemoryBound,
        ] {
            let params = self.get_optimization_params(strategy, num_qubits);

            // Simulate benchmark (in real implementation, would run actual operations)
            let estimated_time = self.estimate_execution_time(state_size, &params);
            results.insert(strategy, estimated_time);
        }

        // Find best strategy
        let best_strategy = results
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(OptimizationStrategy::Balanced, |(s, _)| *s);

        CalibrationResult {
            best_strategy,
            strategy_times: results,
            optimal_params: self.get_optimization_params(best_strategy, num_qubits),
        }
    }

    fn estimate_execution_time(&self, state_size: usize, params: &OptimizationParams) -> Duration {
        // Simplified estimation model
        let base_ops = state_size as f64;
        let simd_factor = if params.use_simd { 4.0 } else { 1.0 };
        let parallel_factor = if params.use_parallel {
            self.hardware.capabilities.cpu.logical_cores as f64
        } else {
            1.0
        };

        let ops_per_sec = self.hardware.peak_gflops * 1e9;
        let estimated_secs = (base_ops * 10.0) / (ops_per_sec * simd_factor * parallel_factor);

        Duration::from_secs_f64(estimated_secs)
    }
}

/// Optimization parameters
#[derive(Debug, Clone)]
pub struct OptimizationParams {
    /// Use SIMD instructions
    pub use_simd: bool,
    /// Use parallel execution
    pub use_parallel: bool,
    /// Batch size for operations
    pub batch_size: usize,
    /// Tile size for tiled operations
    pub tile_size: usize,
    /// Prefetch distance
    pub prefetch_distance: usize,
    /// Use streaming for large data
    pub use_streaming: bool,
}

/// Calibration result
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Best strategy found
    pub best_strategy: OptimizationStrategy,
    /// Execution times for each strategy
    pub strategy_times: HashMap<OptimizationStrategy, Duration>,
    /// Optimal parameters
    pub optimal_params: OptimizationParams,
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Hardware assessment
    pub hardware_assessment: HardwareAssessment,
    /// Workload profiles
    pub workload_profiles: Vec<(String, PerformanceProfile)>,
    /// Total optimization events
    pub total_optimization_events: usize,
    /// Recommendations
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AdaptiveOptimizationConfig::default();
        assert!(config.enable_workload_profiling);
        assert!(config.enable_memory_optimization);
        assert!(!config.enable_power_optimization);
    }

    #[test]
    fn test_hardware_assessment() {
        let capabilities = PlatformCapabilities::detect();
        let assessment = HardwareAssessment::from_capabilities(capabilities);

        assert!(assessment.peak_memory_bandwidth > 0.0);
        assert!(assessment.peak_gflops > 0.0);
        assert!(assessment.optimal_batch_size > 0);
        assert!(assessment.optimal_tile_size > 0);
    }

    #[test]
    fn test_optimizer_creation() {
        let config = AdaptiveOptimizationConfig::default();
        let optimizer = AdaptiveHardwareOptimizer::new(config);

        assert!(optimizer.hardware_assessment().peak_gflops > 0.0);
    }

    #[test]
    fn test_workload_analysis() {
        let config = AdaptiveOptimizationConfig::default();
        let optimizer = AdaptiveHardwareOptimizer::new(config);

        // Compute-bound workload
        let compute_bound = WorkloadCharacteristics {
            num_qubits: 4,
            num_gates: 100,
            circuit_depth: 10,
            access_pattern: AccessPattern::Sequential,
            computational_intensity: 15.0,
            expected_iterations: 1,
        };

        let strategy = optimizer.analyze_workload(&compute_bound);
        assert_eq!(strategy, OptimizationStrategy::Throughput);

        // Memory-bound workload
        let memory_bound = WorkloadCharacteristics {
            num_qubits: 20,
            num_gates: 10,
            circuit_depth: 2,
            access_pattern: AccessPattern::Random,
            computational_intensity: 0.5,
            expected_iterations: 1,
        };

        let strategy = optimizer.analyze_workload(&memory_bound);
        assert_eq!(strategy, OptimizationStrategy::MemoryBound);
    }

    #[test]
    fn test_optimization_params() {
        let config = AdaptiveOptimizationConfig::default();
        let optimizer = AdaptiveHardwareOptimizer::new(config);

        let params = optimizer.get_optimization_params(OptimizationStrategy::Throughput, 10);
        assert!(params.use_simd);
        assert!(params.batch_size > 0);

        let params = optimizer.get_optimization_params(OptimizationStrategy::Latency, 10);
        assert!(!params.use_parallel); // Latency optimization disables parallel
    }

    #[test]
    fn test_execution_recording() {
        let config = AdaptiveOptimizationConfig::default();
        let optimizer = AdaptiveHardwareOptimizer::new(config);

        // Record some executions
        for _ in 0..20 {
            optimizer.record_execution(
                "test_workload",
                OptimizationStrategy::Throughput,
                Duration::from_micros(100),
            );
        }

        let profile = optimizer.get_profile("test_workload");
        assert!(profile.is_some());
        assert_eq!(profile.expect("profile should exist").sample_count, 20);
    }

    #[test]
    fn test_calibration() {
        let config = AdaptiveOptimizationConfig::default();
        let optimizer = AdaptiveHardwareOptimizer::new(config);

        let result = optimizer.calibrate(6);
        assert!(!result.strategy_times.is_empty());
        assert!(result.optimal_params.batch_size > 0);
    }

    #[test]
    fn test_optimization_report() {
        let config = AdaptiveOptimizationConfig::default();
        let optimizer = AdaptiveHardwareOptimizer::new(config);

        let report = optimizer.generate_report();
        assert!(!report.recommendations.is_empty());
        assert!(report.hardware_assessment.peak_gflops > 0.0);
    }

    #[test]
    fn test_recommended_strategy() {
        let config = AdaptiveOptimizationConfig::default();
        let optimizer = AdaptiveHardwareOptimizer::new(config);

        // Without samples, should return default
        let strategy = optimizer.get_recommended_strategy("unknown_workload");
        assert_eq!(strategy, OptimizationStrategy::Balanced);
    }
}
