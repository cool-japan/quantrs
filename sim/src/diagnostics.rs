//! Comprehensive error handling and diagnostics for quantum simulation
//!
//! This module provides advanced error handling, performance diagnostics,
//! and system health monitoring for the quantum simulation framework.

use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Comprehensive diagnostics system for quantum simulation
#[derive(Debug, Clone)]
pub struct SimulationDiagnostics {
    /// Error tracking and categorization
    error_tracker: Arc<Mutex<ErrorTracker>>,
    /// Performance monitoring
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    /// Memory usage tracking
    memory_tracker: Arc<Mutex<MemoryTracker>>,
    /// Circuit analysis results
    circuit_analyzer: Arc<Mutex<CircuitAnalyzer>>,
}

/// Error tracking and categorization system
#[derive(Debug, Default)]
struct ErrorTracker {
    /// Total error count by category
    error_counts: HashMap<ErrorCategory, usize>,
    /// Recent errors with timestamps
    recent_errors: Vec<(Instant, ErrorInfo)>,
    /// Error patterns and frequencies
    error_patterns: HashMap<String, usize>,
    /// Critical error threshold
    critical_threshold: usize,
}

/// Performance monitoring system
#[derive(Debug, Default)]
struct PerformanceMonitor {
    /// Operation timing statistics
    operation_times: HashMap<String, OperationStats>,
    /// Gate application performance
    gate_performance: HashMap<String, GateStats>,
    /// Memory allocation patterns
    allocation_patterns: Vec<(Instant, usize, String)>,
    /// Simulation throughput metrics
    throughput_metrics: ThroughputMetrics,
}

/// Memory usage tracking
#[derive(Debug, Default)]
struct MemoryTracker {
    /// Peak memory usage per operation
    peak_memory: HashMap<String, usize>,
    /// Memory efficiency metrics
    efficiency_metrics: MemoryEfficiencyMetrics,
    /// Buffer pool statistics
    buffer_pool_stats: BufferPoolStats,
    /// Memory leak detection
    leak_detection: LeakDetectionStats,
    /// Memory allocation patterns
    allocation_patterns: Vec<(Instant, usize, String)>,
}

/// Circuit analysis and optimization recommendations
#[derive(Debug, Default)]
struct CircuitAnalyzer {
    /// Circuit complexity metrics
    complexity_metrics: ComplexityMetrics,
    /// Gate count statistics
    gate_statistics: HashMap<String, usize>,
    /// Optimization opportunities
    optimization_opportunities: Vec<OptimizationRecommendation>,
    /// Circuit health score
    health_score: f64,
}

/// Error categorization
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Memory allocation errors
    Memory,
    /// Invalid circuit structure
    Circuit,
    /// Qubit index out of bounds
    QubitIndex,
    /// Mathematical computation errors
    Computation,
    /// Hardware/GPU related errors
    Hardware,
    /// Configuration errors
    Configuration,
    /// Threading/concurrency errors
    Concurrency,
    /// Unknown error category
    Unknown,
}

/// Detailed error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub category: ErrorCategory,
    pub message: String,
    pub context: HashMap<String, String>,
    pub severity: ErrorSeverity,
    pub suggested_fix: Option<String>,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Operation timing statistics
#[derive(Debug, Default, Clone)]
struct OperationStats {
    total_time: Duration,
    call_count: usize,
    min_time: Option<Duration>,
    max_time: Option<Duration>,
    recent_times: Vec<Duration>,
}

/// Gate-specific performance statistics
#[derive(Debug, Default, Clone)]
struct GateStats {
    total_applications: usize,
    total_time: Duration,
    average_time: Duration,
    qubits_affected: Vec<usize>,
    efficiency_score: f64,
}

/// Throughput metrics
#[derive(Debug, Default, Clone)]
struct ThroughputMetrics {
    gates_per_second: f64,
    qubits_simulated_per_second: f64,
    circuits_completed: usize,
    average_circuit_time: Duration,
}

/// Memory efficiency metrics
#[derive(Debug, Default, Clone)]
struct MemoryEfficiencyMetrics {
    buffer_reuse_rate: f64,
    allocation_efficiency: f64,
    peak_to_average_ratio: f64,
    fragmentation_score: f64,
}

/// Buffer pool statistics
#[derive(Debug, Default, Clone)]
struct BufferPoolStats {
    total_allocations: usize,
    total_reuses: usize,
    cache_hit_rate: f64,
    average_buffer_lifetime: Duration,
}

/// Memory leak detection statistics
#[derive(Debug, Default, Clone)]
struct LeakDetectionStats {
    suspicious_allocations: usize,
    memory_growth_rate: f64,
    long_lived_allocations: usize,
}

/// Circuit complexity metrics
#[derive(Debug, Default, Clone)]
struct ComplexityMetrics {
    total_gates: usize,
    depth: usize,
    width: usize,
    entanglement_measure: f64,
    parallelization_potential: f64,
}

/// Optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: OptimizationCategory,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_difficulty: Difficulty,
    pub priority: Priority,
}

/// Optimization categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    GateFusion,
    CircuitReordering,
    MemoryOptimization,
    ParallelizationOpportunity,
    AlgorithmicImprovement,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Comprehensive diagnostic report
#[derive(Debug, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub timestamp: String,
    pub error_summary: ErrorSummary,
    pub performance_summary: PerformanceSummary,
    pub memory_summary: MemorySummary,
    pub circuit_analysis: CircuitAnalysisSummary,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub overall_health_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorSummary {
    pub total_errors: usize,
    pub errors_by_category: HashMap<ErrorCategory, usize>,
    pub critical_errors: usize,
    pub error_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub average_gate_time: f64,
    pub gates_per_second: f64,
    pub memory_efficiency: f64,
    pub parallelization_efficiency: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemorySummary {
    pub peak_memory_usage: usize,
    pub buffer_pool_efficiency: f64,
    pub memory_leak_risk: f64,
    pub allocation_efficiency: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CircuitAnalysisSummary {
    pub complexity_score: f64,
    pub optimization_potential: f64,
    pub gate_distribution: HashMap<String, usize>,
    pub depth_analysis: DepthAnalysis,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DepthAnalysis {
    pub total_depth: usize,
    pub critical_path_length: usize,
    pub parallelization_opportunities: usize,
}

impl SimulationDiagnostics {
    /// Create a new diagnostics system
    #[must_use]
    pub fn new() -> Self {
        Self {
            error_tracker: Arc::new(Mutex::new(ErrorTracker::default())),
            performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::default())),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::default())),
            circuit_analyzer: Arc::new(Mutex::new(CircuitAnalyzer::default())),
        }
    }

    /// Record an error with detailed context
    pub fn record_error(&self, error: &QuantRS2Error, context: HashMap<String, String>) {
        let error_info = self.categorize_error(error, context);

        if let Ok(mut tracker) = self.error_tracker.lock() {
            tracker.record_error(error_info);
        }
    }

    /// Record operation timing
    pub fn record_operation_time(&self, operation: &str, duration: Duration) {
        if let Ok(mut monitor) = self.performance_monitor.lock() {
            monitor.record_operation(operation.to_string(), duration);
        }
    }

    /// Record gate application performance
    pub fn record_gate_performance(&self, gate_name: &str, qubits: &[usize], duration: Duration) {
        if let Ok(mut monitor) = self.performance_monitor.lock() {
            monitor.record_gate_performance(gate_name.to_string(), qubits.to_vec(), duration);
        }
    }

    /// Record memory allocation
    pub fn record_memory_allocation(&self, size: usize, operation: &str) {
        if let Ok(mut tracker) = self.memory_tracker.lock() {
            tracker.record_allocation(size, operation.to_string());
        }
    }

    /// Analyze circuit complexity and optimization opportunities
    pub fn analyze_circuit<const N: usize>(&self, circuit: &quantrs2_circuit::builder::Circuit<N>) {
        if let Ok(mut analyzer) = self.circuit_analyzer.lock() {
            analyzer.analyze_circuit(circuit);
        }
    }

    /// Generate comprehensive diagnostic report
    #[must_use]
    pub fn generate_report(&self) -> DiagnosticReport {
        let timestamp = chrono::Utc::now().to_rfc3339();

        let error_summary = self
            .error_tracker
            .lock()
            .map(|tracker| tracker.generate_summary())
            .unwrap_or_default();

        let performance_summary = self
            .performance_monitor
            .lock()
            .map(|monitor| monitor.generate_summary())
            .unwrap_or_default();

        let memory_summary = self
            .memory_tracker
            .lock()
            .map(|tracker| tracker.generate_summary())
            .unwrap_or_default();

        let circuit_analysis = self
            .circuit_analyzer
            .lock()
            .map(|analyzer| analyzer.generate_summary())
            .unwrap_or_default();

        let recommendations = self.generate_recommendations();
        let overall_health_score =
            self.calculate_health_score(&error_summary, &performance_summary, &memory_summary);

        DiagnosticReport {
            timestamp,
            error_summary,
            performance_summary,
            memory_summary,
            circuit_analysis,
            recommendations,
            overall_health_score,
        }
    }

    /// Categorize error for better tracking
    fn categorize_error(
        &self,
        error: &QuantRS2Error,
        context: HashMap<String, String>,
    ) -> ErrorInfo {
        let (category, severity, suggested_fix) = match error {
            QuantRS2Error::InvalidQubitId(_) => (
                ErrorCategory::QubitIndex,
                ErrorSeverity::High,
                Some("Check qubit indices are within circuit bounds".to_string()),
            ),
            QuantRS2Error::CircuitValidationFailed(_) => (
                ErrorCategory::Circuit,
                ErrorSeverity::Medium,
                Some("Validate circuit structure before simulation".to_string()),
            ),
            QuantRS2Error::LinalgError(_) => (
                ErrorCategory::Computation,
                ErrorSeverity::High,
                Some("Check matrix dimensions and numerical stability".to_string()),
            ),
            QuantRS2Error::UnsupportedOperation(_) => (
                ErrorCategory::Configuration,
                ErrorSeverity::Medium,
                Some("Use supported gate types for this simulator".to_string()),
            ),
            QuantRS2Error::InvalidInput(_) => (
                ErrorCategory::Configuration,
                ErrorSeverity::Medium,
                Some("Validate input parameters before operation".to_string()),
            ),
            _ => (ErrorCategory::Unknown, ErrorSeverity::Low, None),
        };

        ErrorInfo {
            category,
            message: error.to_string(),
            context,
            severity,
            suggested_fix,
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze performance patterns
        if let Ok(monitor) = self.performance_monitor.lock() {
            if !monitor.gate_performance.is_empty() {
                let avg_gate_time: Duration = monitor
                    .gate_performance
                    .values()
                    .map(|stats| stats.average_time)
                    .sum::<Duration>()
                    / monitor.gate_performance.len() as u32;

                if avg_gate_time > Duration::from_millis(1) {
                    recommendations.push(OptimizationRecommendation {
                        category: OptimizationCategory::GateFusion,
                        description: "Consider gate fusion to reduce operation overhead"
                            .to_string(),
                        expected_improvement: 0.3,
                        implementation_difficulty: Difficulty::Medium,
                        priority: Priority::High,
                    });
                }
            }
        }

        // Analyze memory patterns
        if let Ok(tracker) = self.memory_tracker.lock() {
            if tracker.efficiency_metrics.buffer_reuse_rate < 0.7 {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::MemoryOptimization,
                    description: "Improve buffer pool utilization for better memory efficiency"
                        .to_string(),
                    expected_improvement: 0.25,
                    implementation_difficulty: Difficulty::Easy,
                    priority: Priority::Medium,
                });
            }
        }

        // Analyze circuit structure
        if let Ok(analyzer) = self.circuit_analyzer.lock() {
            if analyzer.complexity_metrics.parallelization_potential > 0.5 {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::ParallelizationOpportunity,
                    description:
                        "Circuit has high parallelization potential - consider parallel execution"
                            .to_string(),
                    expected_improvement: 0.4,
                    implementation_difficulty: Difficulty::Hard,
                    priority: Priority::High,
                });
            }
        }

        recommendations
    }

    /// Calculate overall system health score
    fn calculate_health_score(
        &self,
        error_summary: &ErrorSummary,
        performance_summary: &PerformanceSummary,
        memory_summary: &MemorySummary,
    ) -> f64 {
        let error_score = if error_summary.total_errors == 0 {
            1.0
        } else {
            error_summary.error_rate.min(0.5).mul_add(-2.0, 1.0)
        };

        let performance_score = (performance_summary.gates_per_second / 1000.0).min(1.0);
        let memory_score = memory_summary.buffer_pool_efficiency;

        memory_score.mul_add(0.3, error_score * 0.4 + performance_score * 0.3) * 100.0
    }
}

impl ErrorTracker {
    fn record_error(&mut self, error_info: ErrorInfo) {
        *self.error_counts.entry(error_info.category).or_insert(0) += 1;
        self.recent_errors
            .push((Instant::now(), error_info.clone()));

        // Track error patterns
        let pattern = format!(
            "{:?}:{}",
            error_info.category,
            error_info
                .message
                .split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ")
        );
        *self.error_patterns.entry(pattern).or_insert(0) += 1;

        // Keep only recent errors (last 100)
        if self.recent_errors.len() > 100 {
            self.recent_errors.remove(0);
        }
    }

    fn generate_summary(&self) -> ErrorSummary {
        let total_errors = self.recent_errors.len();
        let critical_errors = self
            .recent_errors
            .iter()
            .filter(|(_, error)| matches!(error.severity, ErrorSeverity::Critical))
            .count();

        let error_rate = if total_errors > 0 {
            critical_errors as f64 / total_errors as f64
        } else {
            0.0
        };

        ErrorSummary {
            total_errors,
            errors_by_category: self.error_counts.clone(),
            critical_errors,
            error_rate,
        }
    }
}

impl PerformanceMonitor {
    fn record_operation(&mut self, operation: String, duration: Duration) {
        let stats = self.operation_times.entry(operation).or_default();
        stats.total_time += duration;
        stats.call_count += 1;

        stats.min_time = Some(stats.min_time.map_or(duration, |min| min.min(duration)));
        stats.max_time = Some(stats.max_time.map_or(duration, |max| max.max(duration)));

        stats.recent_times.push(duration);
        if stats.recent_times.len() > 50 {
            stats.recent_times.remove(0);
        }
    }

    fn record_gate_performance(
        &mut self,
        gate_name: String,
        qubits: Vec<usize>,
        duration: Duration,
    ) {
        let stats = self.gate_performance.entry(gate_name).or_default();
        stats.total_applications += 1;
        stats.total_time += duration;
        stats.average_time = stats.total_time / stats.total_applications as u32;
        stats.qubits_affected.extend(qubits);

        // Calculate efficiency score based on time and qubit count
        stats.efficiency_score =
            1000.0 / (duration.as_nanos() as f64 / stats.qubits_affected.len() as f64);
    }

    fn generate_summary(&self) -> PerformanceSummary {
        let average_gate_time = if self.gate_performance.is_empty() {
            0.0
        } else {
            self.gate_performance
                .values()
                .map(|stats| stats.average_time.as_nanos() as f64)
                .sum::<f64>()
                / self.gate_performance.len() as f64
        };

        let gates_per_second = if average_gate_time > 0.0 {
            1_000_000_000.0 / average_gate_time
        } else {
            0.0
        };

        PerformanceSummary {
            average_gate_time,
            gates_per_second,
            memory_efficiency: 0.85, // Would be calculated from actual metrics
            parallelization_efficiency: 0.75, // Would be calculated from actual metrics
        }
    }
}

impl MemoryTracker {
    fn record_allocation(&mut self, size: usize, operation: String) {
        self.allocation_patterns
            .push((Instant::now(), size, operation.clone()));

        // Update peak memory for operation
        let current_peak = self.peak_memory.entry(operation).or_insert(0);
        *current_peak = (*current_peak).max(size);

        // Keep only recent allocations
        if self.allocation_patterns.len() > 1000 {
            self.allocation_patterns.remove(0);
        }
    }

    fn generate_summary(&self) -> MemorySummary {
        let peak_memory_usage = self.peak_memory.values().max().copied().unwrap_or(0);

        MemorySummary {
            peak_memory_usage,
            buffer_pool_efficiency: 0.85, // Would be calculated from actual pool statistics
            memory_leak_risk: 0.1,        // Would be calculated from growth patterns
            allocation_efficiency: 0.9,   // Would be calculated from reuse patterns
        }
    }
}

impl CircuitAnalyzer {
    fn analyze_circuit<const N: usize>(&mut self, circuit: &quantrs2_circuit::builder::Circuit<N>) {
        self.complexity_metrics.width = N;
        self.complexity_metrics.total_gates = circuit.gates().len();

        // Analyze gate distribution
        for gate in circuit.gates() {
            *self
                .gate_statistics
                .entry(gate.name().to_string())
                .or_insert(0) += 1;
        }

        // Calculate complexity score
        self.complexity_metrics.entanglement_measure = self.calculate_entanglement_measure();
        self.complexity_metrics.parallelization_potential =
            self.calculate_parallelization_potential();

        // Calculate overall health score
        self.health_score = self.calculate_circuit_health();
    }

    fn calculate_entanglement_measure(&self) -> f64 {
        // Simplified entanglement measure based on two-qubit gates
        let two_qubit_gates = self
            .gate_statistics
            .iter()
            .filter(|(name, _)| matches!(name.as_str(), "CNOT" | "CZ" | "SWAP" | "CY" | "CH"))
            .map(|(_, count)| *count)
            .sum::<usize>();

        two_qubit_gates as f64 / self.complexity_metrics.total_gates.max(1) as f64
    }

    fn calculate_parallelization_potential(&self) -> f64 {
        // Simplified calculation based on gate dependencies
        // In a real implementation, this would analyze the circuit DAG
        let single_qubit_gates = self
            .gate_statistics
            .iter()
            .filter(|(name, _)| {
                matches!(
                    name.as_str(),
                    "H" | "X" | "Y" | "Z" | "S" | "T" | "RX" | "RY" | "RZ"
                )
            })
            .map(|(_, count)| *count)
            .sum::<usize>();

        single_qubit_gates as f64 / self.complexity_metrics.total_gates.max(1) as f64
    }

    fn calculate_circuit_health(&self) -> f64 {
        // Health score based on various factors
        let depth_score = if self.complexity_metrics.depth > 0 {
            1.0 / (1.0 + (self.complexity_metrics.depth as f64 / 100.0))
        } else {
            1.0
        };

        let complexity_score = 1.0 - self.complexity_metrics.entanglement_measure.min(1.0);
        let parallelization_score = self.complexity_metrics.parallelization_potential;

        (depth_score + complexity_score + parallelization_score) / 3.0 * 100.0
    }

    fn generate_summary(&self) -> CircuitAnalysisSummary {
        CircuitAnalysisSummary {
            complexity_score: self.complexity_metrics.entanglement_measure * 100.0,
            optimization_potential: self.complexity_metrics.parallelization_potential * 100.0,
            gate_distribution: self.gate_statistics.clone(),
            depth_analysis: DepthAnalysis {
                total_depth: self.complexity_metrics.depth,
                critical_path_length: self.complexity_metrics.depth, // Simplified
                parallelization_opportunities: (self.complexity_metrics.parallelization_potential
                    * 10.0) as usize,
            },
        }
    }
}

impl Default for ErrorSummary {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_category: HashMap::new(),
            critical_errors: 0,
            error_rate: 0.0,
        }
    }
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            average_gate_time: 0.0,
            gates_per_second: 0.0,
            memory_efficiency: 0.0,
            parallelization_efficiency: 0.0,
        }
    }
}

impl Default for MemorySummary {
    fn default() -> Self {
        Self {
            peak_memory_usage: 0,
            buffer_pool_efficiency: 0.0,
            memory_leak_risk: 0.0,
            allocation_efficiency: 0.0,
        }
    }
}

impl Default for CircuitAnalysisSummary {
    fn default() -> Self {
        Self {
            complexity_score: 0.0,
            optimization_potential: 0.0,
            gate_distribution: HashMap::new(),
            depth_analysis: DepthAnalysis {
                total_depth: 0,
                critical_path_length: 0,
                parallelization_opportunities: 0,
            },
        }
    }
}

impl Default for SimulationDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_creation() {
        let diagnostics = SimulationDiagnostics::new();
        let report = diagnostics.generate_report();

        assert_eq!(report.error_summary.total_errors, 0);
        assert!(report.overall_health_score >= 0.0);
    }

    #[test]
    fn test_error_recording() {
        let diagnostics = SimulationDiagnostics::new();
        let error = QuantRS2Error::InvalidQubitId(5);
        let mut context = HashMap::new();
        context.insert("operation".to_string(), "gate_application".to_string());

        diagnostics.record_error(&error, context);

        let report = diagnostics.generate_report();
        assert_eq!(report.error_summary.total_errors, 1);
        assert!(report
            .error_summary
            .errors_by_category
            .contains_key(&ErrorCategory::QubitIndex));
    }

    #[test]
    fn test_performance_recording() {
        let diagnostics = SimulationDiagnostics::new();

        diagnostics.record_operation_time("gate_application", Duration::from_millis(10));
        diagnostics.record_gate_performance("H", &[0], Duration::from_micros(500));

        let report = diagnostics.generate_report();
        assert!(report.performance_summary.average_gate_time > 0.0);
    }
}
