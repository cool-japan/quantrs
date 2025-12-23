//! Precision analysis and performance metrics for mixed-precision simulation.
//!
//! This module provides tools for analyzing numerical precision requirements,
//! performance characteristics, and optimal precision selection strategies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::config::QuantumPrecision;

/// Precision analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionAnalysis {
    /// Recommended precision for each operation type
    pub recommended_precisions: HashMap<String, QuantumPrecision>,
    /// Estimated numerical error for each precision
    pub error_estimates: HashMap<QuantumPrecision, f64>,
    /// Performance metrics for each precision
    pub performance_metrics: HashMap<QuantumPrecision, PerformanceMetrics>,
    /// Final precision selection rationale
    pub selection_rationale: String,
    /// Quality score (0-1, higher is better)
    pub quality_score: f64,
}

/// Performance metrics for a specific precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: f64,
    /// Energy efficiency score (operations per joule)
    pub energy_efficiency: f64,
}

/// Precision analyzer for quantum operations
pub struct PrecisionAnalyzer {
    /// Current analysis state
    analysis_state: AnalysisState,
    /// Benchmark results for different precisions
    benchmark_cache: HashMap<QuantumPrecision, PerformanceMetrics>,
    /// Error tracking history
    error_history: Vec<ErrorSample>,
}

/// Internal analysis state
#[derive(Debug, Clone)]
struct AnalysisState {
    /// Number of operations analyzed
    operations_count: usize,
    /// Total analysis time
    total_time: f64,
    /// Current precision being analyzed
    current_precision: QuantumPrecision,
}

/// Error sample for tracking numerical accuracy
#[derive(Debug, Clone)]
struct ErrorSample {
    /// Precision level used
    precision: QuantumPrecision,
    /// Estimated numerical error
    error: f64,
    /// Operation type
    operation_type: String,
    /// Timestamp
    timestamp: Instant,
}

impl PrecisionAnalysis {
    /// Create a new empty analysis
    pub fn new() -> Self {
        Self {
            recommended_precisions: HashMap::new(),
            error_estimates: HashMap::new(),
            performance_metrics: HashMap::new(),
            selection_rationale: String::new(),
            quality_score: 0.0,
        }
    }

    /// Add a precision recommendation for an operation type
    pub fn add_recommendation(&mut self, operation: String, precision: QuantumPrecision) {
        self.recommended_precisions.insert(operation, precision);
    }

    /// Add an error estimate for a precision level
    pub fn add_error_estimate(&mut self, precision: QuantumPrecision, error: f64) {
        self.error_estimates.insert(precision, error);
    }

    /// Add performance metrics for a precision level
    pub fn add_performance_metrics(
        &mut self,
        precision: QuantumPrecision,
        metrics: PerformanceMetrics,
    ) {
        self.performance_metrics.insert(precision, metrics);
    }

    /// Set the selection rationale
    pub fn set_rationale(&mut self, rationale: String) {
        self.selection_rationale = rationale;
    }

    /// Calculate and set the quality score
    pub fn calculate_quality_score(&mut self) {
        let mut score = 0.0;
        let mut count = 0;

        // Score based on error estimates (lower error = higher score)
        for (&precision, &error) in &self.error_estimates {
            let error_score = 1.0 / error.mul_add(1000.0, 1.0); // Normalize error
            score += error_score;
            count += 1;
        }

        // Score based on performance metrics
        for (precision, metrics) in &self.performance_metrics {
            let perf_score = metrics.throughput_ops_per_sec / 1000.0; // Normalize throughput
            let mem_score = 1.0 / (1.0 + metrics.memory_usage_bytes as f64 / 1e9); // Normalize memory
            score += f64::midpoint(perf_score, mem_score);
            count += 1;
        }

        if count > 0 {
            self.quality_score = (score / count as f64).min(1.0);
        }
    }

    /// Get the best precision for a given operation type
    pub fn get_best_precision(&self, operation: &str) -> Option<QuantumPrecision> {
        self.recommended_precisions.get(operation).copied()
    }

    /// Get overall recommended precision
    pub fn get_overall_recommendation(&self) -> QuantumPrecision {
        // Find the most commonly recommended precision
        let mut precision_counts = HashMap::new();
        for &precision in self.recommended_precisions.values() {
            *precision_counts.entry(precision).or_insert(0) += 1;
        }

        precision_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map_or(QuantumPrecision::Single, |(precision, _)| precision)
    }

    /// Check if analysis indicates good quality
    pub fn is_high_quality(&self) -> bool {
        self.quality_score > 0.8
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> AnalysisSummary {
        AnalysisSummary {
            num_operations_analyzed: self.recommended_precisions.len(),
            overall_precision: self.get_overall_recommendation(),
            quality_score: self.quality_score,
            total_error_estimate: self.error_estimates.values().sum(),
            avg_execution_time: self
                .performance_metrics
                .values()
                .map(|m| m.execution_time_ms)
                .sum::<f64>()
                / self.performance_metrics.len().max(1) as f64,
        }
    }
}

/// Summary of precision analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    /// Number of operation types analyzed
    pub num_operations_analyzed: usize,
    /// Overall recommended precision
    pub overall_precision: QuantumPrecision,
    /// Quality score (0-1)
    pub quality_score: f64,
    /// Total estimated numerical error
    pub total_error_estimate: f64,
    /// Average execution time across all precisions
    pub avg_execution_time: f64,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub const fn new(
        execution_time_ms: f64,
        memory_usage_bytes: usize,
        throughput_ops_per_sec: f64,
        energy_efficiency: f64,
    ) -> Self {
        Self {
            execution_time_ms,
            memory_usage_bytes,
            throughput_ops_per_sec,
            energy_efficiency,
        }
    }

    /// Create metrics from execution time and memory usage
    pub fn from_time_and_memory(execution_time_ms: f64, memory_usage_bytes: usize) -> Self {
        let throughput = if execution_time_ms > 0.0 {
            1000.0 / execution_time_ms
        } else {
            0.0
        };

        // Estimate energy efficiency (simplified model)
        let energy_efficiency = throughput / (memory_usage_bytes as f64 / 1e6);

        Self::new(
            execution_time_ms,
            memory_usage_bytes,
            throughput,
            energy_efficiency,
        )
    }

    /// Calculate performance score (0-1, higher is better)
    pub fn performance_score(&self) -> f64 {
        let time_score = 1.0 / (1.0 + self.execution_time_ms / 1000.0);
        let memory_score = 1.0 / (1.0 + self.memory_usage_bytes as f64 / 1e9);
        let throughput_score = self.throughput_ops_per_sec / 1000.0;
        let energy_score = self.energy_efficiency / 1000.0;

        (time_score + memory_score + throughput_score + energy_score) / 4.0
    }

    /// Compare with another metrics instance
    pub fn is_better_than(&self, other: &Self) -> bool {
        self.performance_score() > other.performance_score()
    }
}

impl PrecisionAnalyzer {
    /// Create a new precision analyzer
    pub fn new() -> Self {
        Self {
            analysis_state: AnalysisState {
                operations_count: 0,
                total_time: 0.0,
                current_precision: QuantumPrecision::Single,
            },
            benchmark_cache: HashMap::new(),
            error_history: Vec::new(),
        }
    }

    /// Analyze optimal precision for a given error tolerance
    pub fn analyze_for_tolerance(&mut self, tolerance: f64) -> PrecisionAnalysis {
        let mut analysis = PrecisionAnalysis::new();

        // Test each precision level
        for &precision in &[
            QuantumPrecision::Half,
            QuantumPrecision::Single,
            QuantumPrecision::Double,
        ] {
            let error = precision.typical_error();
            analysis.add_error_estimate(precision, error);

            // Create synthetic performance metrics
            let metrics = self.create_synthetic_metrics(precision);
            analysis.add_performance_metrics(precision, metrics);

            // Add recommendations based on tolerance
            if precision.is_sufficient_for_tolerance(tolerance) {
                analysis.add_recommendation("default".to_string(), precision);
            }
        }

        // Set rationale
        let rationale = format!(
            "Analysis for tolerance {tolerance:.2e}: recommended precision based on error bounds and performance trade-offs"
        );
        analysis.set_rationale(rationale);

        // Calculate quality score
        analysis.calculate_quality_score();

        analysis
    }

    /// Benchmark a specific precision level
    pub fn benchmark_precision(&mut self, precision: QuantumPrecision) -> PerformanceMetrics {
        if let Some(cached) = self.benchmark_cache.get(&precision) {
            return cached.clone();
        }

        let start_time = Instant::now();

        // Simulate some work (in real implementation, this would run actual benchmarks)
        std::thread::sleep(std::time::Duration::from_millis(1));

        let execution_time = start_time.elapsed().as_millis() as f64;
        let memory_usage = self.estimate_memory_usage(precision);

        let metrics = PerformanceMetrics::from_time_and_memory(execution_time, memory_usage);
        self.benchmark_cache.insert(precision, metrics.clone());

        metrics
    }

    /// Add an error sample to the history
    pub fn record_error(
        &mut self,
        precision: QuantumPrecision,
        error: f64,
        operation_type: String,
    ) {
        self.error_history.push(ErrorSample {
            precision,
            error,
            operation_type,
            timestamp: Instant::now(),
        });
    }

    /// Get average error for a precision level
    pub fn get_average_error(&self, precision: QuantumPrecision) -> f64 {
        let errors: Vec<f64> = self
            .error_history
            .iter()
            .filter(|sample| sample.precision == precision)
            .map(|sample| sample.error)
            .collect();

        if errors.is_empty() {
            precision.typical_error()
        } else {
            errors.iter().sum::<f64>() / errors.len() as f64
        }
    }

    /// Clear the analysis state
    pub fn reset(&mut self) {
        self.analysis_state.operations_count = 0;
        self.analysis_state.total_time = 0.0;
        self.error_history.clear();
        self.benchmark_cache.clear();
    }

    /// Create synthetic performance metrics for testing
    fn create_synthetic_metrics(&self, precision: QuantumPrecision) -> PerformanceMetrics {
        let base_time = 100.0; // Base execution time in ms
        let execution_time = base_time * precision.computation_factor();

        let base_memory = 1024 * 1024; // Base memory usage in bytes
        let memory_usage = (base_memory as f64 * precision.memory_factor()) as usize;

        PerformanceMetrics::from_time_and_memory(execution_time, memory_usage)
    }

    /// Estimate memory usage for a precision level
    fn estimate_memory_usage(&self, precision: QuantumPrecision) -> usize {
        let base_memory = 1024 * 1024; // 1MB base
        (base_memory as f64 * precision.memory_factor()) as usize
    }
}

impl Default for PrecisionAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PrecisionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
