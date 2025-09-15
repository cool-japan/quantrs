//! GPU performance optimization and profiling.
//!
//! This module provides tools for optimizing GPU performance including
//! memory access patterns, kernel fusion, and performance profiling.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use scirs2_core::gpu::{
    GpuBackend, GpuContext, GpuDevice, GpuError,
};

/// Performance metrics for GPU operations
#[derive(Default, Clone, Debug)]
pub struct GpuPerformanceMetrics {
    /// Kernel execution times
    pub kernel_times: HashMap<String, Vec<Duration>>,
    /// Memory transfer times
    pub transfer_times: HashMap<String, Vec<Duration>>,
    /// Device utilization percentage
    pub device_utilization: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_util: f64,
    /// Compute throughput (GFLOPS)
    pub compute_throughput: f64,
    /// Energy efficiency (solutions per watt)
    pub energy_efficiency: f64,
    /// Queue depth
    pub queue_depth: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// GPU performance profiler using SciRS2 GPU abstractions
pub struct GpuProfiler {
    /// Current metrics
    metrics: Arc<Mutex<GpuPerformanceMetrics>>,
    /// GPU context handle
    context: Arc<GpuContext>,
    /// GPU backend type
    backend: GpuBackend,
    /// Profiling enabled
    enabled: bool,
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProfiler {
    /// Create new profiler
    pub fn new() -> Self {
        // Use CPU backend as default fallback
        let backend = GpuBackend::Cpu;
        let context = GpuContext::new(backend).unwrap_or_else(|_| {
            // This should never fail for CPU backend
            panic!("Failed to create CPU context")
        });

        Self {
            metrics: Arc::new(Mutex::new(GpuPerformanceMetrics::default())),
            context: Arc::new(context),
            backend,
            enabled: true,
        }
    }

    /// Initialize with device context
    pub fn with_context(ctx: GpuContext) -> Self {
        let backend = ctx.backend();
        Self {
            metrics: Arc::new(Mutex::new(GpuPerformanceMetrics::default())),
            context: Arc::new(ctx),
            backend,
            enabled: true,
        }
    }

    /// Enable/disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Record kernel execution time
    pub fn record_kernel_time(&self, kernel_name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        let mut metrics = self.metrics.lock().unwrap();
        metrics
            .kernel_times
            .entry(kernel_name.to_string())
            .or_default()
            .push(duration);
    }

    /// Record memory transfer time
    pub fn record_transfer_time(&self, operation: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        let mut metrics = self.metrics.lock().unwrap();
        metrics
            .transfer_times
            .entry(operation.to_string())
            .or_default()
            .push(duration);
    }

    /// Update device utilization
    pub fn update_utilization(&self, utilization: f64) {
        if !self.enabled {
            return;
        }

        let mut metrics = self.metrics.lock().unwrap();
        metrics.device_utilization = utilization;
    }

    /// Calculate and update throughput metrics
    pub fn update_throughput(&self, operations: usize, duration: Duration) {
        if !self.enabled {
            return;
        }

        let mut metrics = self.metrics.lock().unwrap();
        let seconds = duration.as_secs_f64();
        metrics.compute_throughput = (operations as f64) / seconds / 1e9; // GFLOPS
    }

    /// Get performance report
    pub fn get_report(&self) -> PerformanceReport {
        let metrics = self.metrics.lock().unwrap();

        // Calculate kernel statistics
        let mut kernel_stats = HashMap::new();
        for (name, times) in &metrics.kernel_times {
            let stats = calculate_stats(times);
            kernel_stats.insert(name.clone(), stats);
        }

        // Calculate transfer statistics
        let mut transfer_stats = HashMap::new();
        for (name, times) in &metrics.transfer_times {
            let stats = calculate_stats(times);
            transfer_stats.insert(name.clone(), stats);
        }

        PerformanceReport {
            kernel_stats,
            transfer_stats,
            device_utilization: metrics.device_utilization,
            memory_bandwidth_util: metrics.memory_bandwidth_util,
            compute_throughput: metrics.compute_throughput,
            energy_efficiency: metrics.energy_efficiency,
            recommendations: self.generate_recommendations(&metrics),
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self, metrics: &GpuPerformanceMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check device utilization
        if metrics.device_utilization < 0.7 {
            recommendations.push(
                "Low GPU utilization detected. Consider increasing batch size or workload."
                    .to_string(),
            );
        }

        // Check memory bandwidth
        if metrics.memory_bandwidth_util > 0.9 {
            recommendations.push(
                "High memory bandwidth usage. Consider memory access optimization or compression."
                    .to_string(),
            );
        }

        // Check kernel performance
        for (kernel, times) in &metrics.kernel_times {
            if !times.is_empty() {
                let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
                if avg_time > Duration::from_millis(100) {
                    recommendations.push(format!(
                        "Kernel '{}' has high execution time. Consider optimization or splitting.",
                        kernel
                    ));
                }
            }
        }

        // Cache efficiency
        if metrics.cache_hit_rate < 0.8 {
            recommendations
                .push("Low cache hit rate. Consider data locality optimizations.".to_string());
        }

        recommendations
    }
}

/// Performance statistics
#[derive(Clone, Debug)]
pub struct PerformanceStats {
    pub mean: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
    pub percentile_95: Duration,
}

/// Performance report
#[derive(Debug)]
pub struct PerformanceReport {
    pub kernel_stats: HashMap<String, PerformanceStats>,
    pub transfer_stats: HashMap<String, PerformanceStats>,
    pub device_utilization: f64,
    pub memory_bandwidth_util: f64,
    pub compute_throughput: f64,
    pub energy_efficiency: f64,
    pub recommendations: Vec<String>,
}

/// Memory access pattern analyzer
pub struct MemoryAccessAnalyzer {
    /// Access patterns
    patterns: Vec<AccessPattern>,
    /// Coalescing efficiency
    coalescing_efficiency: f64,
    /// Bank conflicts
    bank_conflicts: usize,
}

#[derive(Clone)]
struct AccessPattern {
    /// Access type (read/write)
    access_type: AccessType,
    /// Stride between accesses
    stride: usize,
    /// Access size
    size: usize,
    /// Frequency
    frequency: usize,
}

#[derive(Clone, Copy)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
}

impl Default for MemoryAccessAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAccessAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            coalescing_efficiency: 1.0,
            bank_conflicts: 0,
        }
    }

    /// Analyze memory access pattern
    pub fn analyze_pattern(&mut self, addresses: &[usize], access_type: AccessType) {
        if addresses.len() < 2 {
            return;
        }

        // Calculate stride pattern
        let mut strides = Vec::new();
        for i in 1..addresses.len() {
            strides.push(addresses[i].saturating_sub(addresses[i - 1]));
        }

        // Find most common stride
        let mut stride_counts = HashMap::new();
        for &stride in &strides {
            *stride_counts.entry(stride).or_insert(0) += 1;
        }

        let (common_stride, frequency) = stride_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&stride, &count)| (stride, count))
            .unwrap_or((0, 0));

        self.patterns.push(AccessPattern {
            access_type,
            stride: common_stride,
            size: addresses.len(),
            frequency,
        });

        // Update coalescing efficiency
        self.update_coalescing_efficiency();
    }

    /// Update coalescing efficiency based on patterns
    fn update_coalescing_efficiency(&mut self) {
        let mut total_accesses = 0;
        let mut coalesced_accesses = 0;

        for pattern in &self.patterns {
            total_accesses += pattern.size;

            // Perfect coalescing: stride of 1 (consecutive)
            // Good coalescing: stride of 4/8 (word-aligned)
            // Poor coalescing: random or large strides
            match pattern.stride {
                1 => coalesced_accesses += pattern.size,
                4 | 8 => coalesced_accesses += pattern.size * 3 / 4,
                s if s < 32 => coalesced_accesses += pattern.size / 2,
                _ => {} // No coalescing
            }
        }

        self.coalescing_efficiency = if total_accesses > 0 {
            coalesced_accesses as f64 / total_accesses as f64
        } else {
            1.0
        };
    }

    /// Get optimization suggestions
    pub fn get_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if self.coalescing_efficiency < 0.8 {
            suggestions.push(
                "Poor memory coalescing detected. Consider restructuring data layout.".to_string(),
            );
        }

        // Check for strided patterns
        for pattern in &self.patterns {
            if pattern.stride > 32 && pattern.frequency > pattern.size / 2 {
                suggestions.push(format!(
                    "Large stride pattern detected ({}). Consider data transposition.",
                    pattern.stride
                ));
            }
        }

        if self.bank_conflicts > 0 {
            suggestions.push(format!(
                "Detected {} bank conflicts. Consider padding shared memory.",
                self.bank_conflicts
            ));
        }

        suggestions
    }
}

/// Kernel fusion optimizer
pub struct KernelFusionOptimizer {
    /// Kernel dependency graph
    dependencies: HashMap<String, Vec<String>>,
    /// Kernel characteristics
    kernel_info: HashMap<String, KernelInfo>,
}

struct KernelInfo {
    /// Compute intensity (FLOPS/byte)
    compute_intensity: f64,
    /// Memory requirements
    memory_required: usize,
    /// Can be fused
    fusable: bool,
}

impl Default for KernelFusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelFusionOptimizer {
    /// Create new optimizer
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            kernel_info: HashMap::new(),
        }
    }

    /// Add kernel information
    pub fn add_kernel(
        &mut self,
        name: &str,
        compute_intensity: f64,
        memory_required: usize,
        dependencies: Vec<String>,
    ) {
        self.dependencies.insert(name.to_string(), dependencies);
        self.kernel_info.insert(
            name.to_string(),
            KernelInfo {
                compute_intensity,
                memory_required,
                fusable: true,
            },
        );
    }

    /// Find fusion opportunities
    pub fn find_fusion_opportunities(&self) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();

        // Check pairs of kernels
        for (kernel1, deps1) in &self.dependencies {
            for (kernel2, deps2) in &self.dependencies {
                if kernel1 >= kernel2 {
                    continue;
                }

                // Check if kernels can be fused
                if self.can_fuse(kernel1, kernel2, deps1, deps2) {
                    let benefit = self.calculate_fusion_benefit(kernel1, kernel2);

                    opportunities.push(FusionOpportunity {
                        kernels: vec![kernel1.clone(), kernel2.clone()],
                        benefit_score: benefit,
                        memory_saved: self.estimate_memory_saved(kernel1, kernel2),
                    });
                }
            }
        }

        // Sort by benefit
        opportunities.sort_by(|a, b| b.benefit_score.partial_cmp(&a.benefit_score).unwrap());

        opportunities
    }

    /// Check if two kernels can be fused
    fn can_fuse(&self, kernel1: &str, kernel2: &str, deps1: &[String], deps2: &[String]) -> bool {
        // Check if kernel2 depends on kernel1 or vice versa
        let direct_dep =
            deps2.contains(&kernel1.to_string()) || deps1.contains(&kernel2.to_string());

        // Check if both kernels are fusable
        let both_fusable = self
            .kernel_info
            .get(kernel1)
            .map(|k| k.fusable)
            .unwrap_or(false)
            && self
                .kernel_info
                .get(kernel2)
                .map(|k| k.fusable)
                .unwrap_or(false);

        direct_dep && both_fusable
    }

    /// Calculate benefit of fusing two kernels
    fn calculate_fusion_benefit(&self, kernel1: &str, kernel2: &str) -> f64 {
        let info1 = &self.kernel_info[kernel1];
        let info2 = &self.kernel_info[kernel2];

        // Benefit based on reduced memory transfers and kernel launch overhead
        let memory_benefit = (info1.memory_required + info2.memory_required) as f64 * 0.001;
        let launch_benefit = 1.0; // Fixed benefit for reducing kernel launches
        let intensity_benefit = (info1.compute_intensity + info2.compute_intensity) * 0.1;

        memory_benefit + launch_benefit + intensity_benefit
    }

    /// Estimate memory saved by fusion
    fn estimate_memory_saved(&self, kernel1: &str, kernel2: &str) -> usize {
        let info1 = &self.kernel_info[kernel1];
        let info2 = &self.kernel_info[kernel2];

        // Assume some intermediate results don't need to be stored
        (info1.memory_required + info2.memory_required) / 4
    }
}

/// Fusion opportunity
#[derive(Debug)]
pub struct FusionOpportunity {
    pub kernels: Vec<String>,
    pub benefit_score: f64,
    pub memory_saved: usize,
}

/// Calculate statistics from duration samples
fn calculate_stats(times: &[Duration]) -> PerformanceStats {
    if times.is_empty() {
        return PerformanceStats {
            mean: Duration::ZERO,
            min: Duration::ZERO,
            max: Duration::ZERO,
            std_dev: Duration::ZERO,
            percentile_95: Duration::ZERO,
        };
    }

    let mut sorted_times = times.to_vec();
    sorted_times.sort();

    let sum: Duration = times.iter().sum();
    let mean = sum / times.len() as u32;

    let variance = times
        .iter()
        .map(|&t| {
            let diff = if t > mean {
                (t - mean).as_secs_f64()
            } else {
                (mean - t).as_secs_f64()
            };
            diff * diff
        })
        .sum::<f64>()
        / times.len() as f64;

    let std_dev = Duration::from_secs_f64(variance.sqrt());

    let percentile_95_idx = (times.len() as f64 * 0.95) as usize;
    let percentile_95 = sorted_times[percentile_95_idx.min(sorted_times.len() - 1)];

    PerformanceStats {
        mean,
        min: sorted_times[0],
        max: sorted_times[sorted_times.len() - 1],
        std_dev,
        percentile_95,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_access_analyzer() {
        let mut analyzer = MemoryAccessAnalyzer::new();

        // Test coalesced access pattern
        let addresses: Vec<usize> = (0..32).map(|i| i * 4).collect();
        analyzer.analyze_pattern(&addresses, AccessType::Read);

        // Should have good coalescing
        assert!(analyzer.coalescing_efficiency > 0.7);

        // Test strided access pattern
        let strided: Vec<usize> = (0..32).map(|i| i * 128).collect();
        analyzer.analyze_pattern(&strided, AccessType::Read);

        let suggestions = analyzer.get_suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_kernel_fusion_optimizer() {
        let mut optimizer = KernelFusionOptimizer::new();

        // Add kernels with dependencies
        optimizer.add_kernel("kernel_a", 10.0, 1024, vec![]);
        optimizer.add_kernel("kernel_b", 5.0, 2048, vec!["kernel_a".to_string()]);
        optimizer.add_kernel("kernel_c", 8.0, 512, vec!["kernel_b".to_string()]);

        let opportunities = optimizer.find_fusion_opportunities();
        assert!(!opportunities.is_empty());

        // Should find fusion opportunity between dependent kernels
        let first = &opportunities[0];
        assert!(first.benefit_score > 0.0);
        assert!(first.memory_saved > 0);
    }
}
