// ! Performance Profiling Utilities for Quantum ML
//!
//! This module provides comprehensive performance profiling tools for quantum
//! machine learning algorithms, helping identify bottlenecks and optimize
//! quantum circuit execution.

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance profiling data for quantum ML operations
#[derive(Debug, Clone)]
pub struct QuantumMLProfiler {
    /// Operation timings
    timings: HashMap<String, Vec<Duration>>,

    /// Memory usage tracking
    memory_snapshots: Vec<MemorySnapshot>,

    /// Quantum circuit metrics
    circuit_metrics: Vec<CircuitMetrics>,

    /// Start time for the current profiling session
    session_start: Option<Instant>,

    /// Profiling configuration
    config: ProfilerConfig,
}

/// Configuration for the profiler
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable detailed timing breakdown
    pub detailed_timing: bool,

    /// Track memory usage
    pub track_memory: bool,

    /// Track quantum circuit metrics
    pub track_circuits: bool,

    /// Sample rate for memory snapshots (every N operations)
    pub memory_sample_rate: usize,

    /// Enable automatic report generation
    pub auto_report: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            detailed_timing: true,
            track_memory: true,
            track_circuits: true,
            memory_sample_rate: 100,
            auto_report: false,
        }
    }
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: Duration,
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub operation: String,
}

/// Quantum circuit performance metrics
#[derive(Debug, Clone)]
pub struct CircuitMetrics {
    pub circuit_name: String,
    pub num_qubits: usize,
    pub circuit_depth: usize,
    pub gate_count: usize,
    pub execution_time: Duration,
    pub shots: usize,
    pub fidelity: Option<f64>,
}

/// Profiling report
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    pub total_duration: Duration,
    pub operation_stats: HashMap<String, OperationStats>,
    pub memory_stats: MemoryStats,
    pub circuit_stats: CircuitStats,
    pub bottlenecks: Vec<Bottleneck>,
    pub recommendations: Vec<String>,
}

/// Statistics for a specific operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub operation_name: String,
    pub call_count: usize,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_dev: Duration,
    pub percentage_of_total: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub peak_memory: usize,
    pub average_memory: usize,
    pub total_allocations: usize,
    pub memory_efficiency: f64,
}

/// Quantum circuit statistics
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub total_circuits_executed: usize,
    pub average_circuit_depth: f64,
    pub average_qubit_count: f64,
    pub total_gate_count: usize,
    pub average_fidelity: Option<f64>,
    pub total_shots: usize,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub operation: String,
    pub severity: BottleneckSeverity,
    pub time_percentage: f64,
    pub description: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckSeverity {
    Critical,   // >50% of total time
    Major,      // 20-50% of total time
    Minor,      // 10-20% of total time
    Negligible, // <10% of total time
}

impl QuantumMLProfiler {
    /// Create a new profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(ProfilerConfig::default())
    }

    /// Create a new profiler with custom configuration
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            timings: HashMap::new(),
            memory_snapshots: Vec::new(),
            circuit_metrics: Vec::new(),
            session_start: None,
            config,
        }
    }

    /// Start a profiling session
    pub fn start_session(&mut self) {
        self.session_start = Some(Instant::now());
        self.timings.clear();
        self.memory_snapshots.clear();
        self.circuit_metrics.clear();
    }

    /// End the profiling session and generate a report
    pub fn end_session(&mut self) -> Result<ProfilingReport> {
        let total_duration = self
            .session_start
            .ok_or_else(|| MLError::InvalidInput("Profiling session not started".to_string()))?
            .elapsed();

        let operation_stats = self.compute_operation_stats(total_duration);
        let memory_stats = self.compute_memory_stats();
        let circuit_stats = self.compute_circuit_stats();
        let bottlenecks = self.identify_bottlenecks(&operation_stats, total_duration);
        let recommendations = self.generate_recommendations(&bottlenecks, &circuit_stats);

        Ok(ProfilingReport {
            total_duration,
            operation_stats,
            memory_stats,
            circuit_stats,
            bottlenecks,
            recommendations,
        })
    }

    /// Time an operation
    pub fn time_operation<F, T>(&mut self, operation_name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        self.timings
            .entry(operation_name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);

        result
    }

    /// Record a memory snapshot
    pub fn record_memory(&mut self, operation: &str, allocated_bytes: usize, peak_bytes: usize) {
        if !self.config.track_memory {
            return;
        }

        let timestamp = self
            .session_start
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO);

        self.memory_snapshots.push(MemorySnapshot {
            timestamp,
            allocated_bytes,
            peak_bytes,
            operation: operation.to_string(),
        });
    }

    /// Record quantum circuit metrics
    pub fn record_circuit_execution(
        &mut self,
        circuit_name: &str,
        num_qubits: usize,
        circuit_depth: usize,
        gate_count: usize,
        execution_time: Duration,
        shots: usize,
        fidelity: Option<f64>,
    ) {
        if !self.config.track_circuits {
            return;
        }

        self.circuit_metrics.push(CircuitMetrics {
            circuit_name: circuit_name.to_string(),
            num_qubits,
            circuit_depth,
            gate_count,
            execution_time,
            shots,
            fidelity,
        });
    }

    /// Compute statistics for all operations
    fn compute_operation_stats(&self, total_duration: Duration) -> HashMap<String, OperationStats> {
        let mut stats = HashMap::new();

        for (operation_name, durations) in &self.timings {
            let call_count = durations.len();
            let total_time: Duration = durations.iter().sum();
            let mean_time = total_time / call_count as u32;
            let min_time = *durations.iter().min().unwrap_or(&Duration::ZERO);
            let max_time = *durations.iter().max().unwrap_or(&Duration::ZERO);

            // Compute standard deviation
            let mean_nanos = mean_time.as_nanos() as f64;
            let variance = durations
                .iter()
                .map(|d| {
                    let diff = d.as_nanos() as f64 - mean_nanos;
                    diff * diff
                })
                .sum::<f64>()
                / call_count as f64;
            let std_dev = Duration::from_nanos(variance.sqrt() as u64);

            let percentage_of_total =
                (total_time.as_secs_f64() / total_duration.as_secs_f64()) * 100.0;

            stats.insert(
                operation_name.clone(),
                OperationStats {
                    operation_name: operation_name.clone(),
                    call_count,
                    total_time,
                    mean_time,
                    min_time,
                    max_time,
                    std_dev,
                    percentage_of_total,
                },
            );
        }

        stats
    }

    /// Compute memory usage statistics
    fn compute_memory_stats(&self) -> MemoryStats {
        if self.memory_snapshots.is_empty() {
            return MemoryStats {
                peak_memory: 0,
                average_memory: 0,
                total_allocations: 0,
                memory_efficiency: 1.0,
            };
        }

        let peak_memory = self
            .memory_snapshots
            .iter()
            .map(|s| s.peak_bytes)
            .max()
            .unwrap_or(0);

        let average_memory = self
            .memory_snapshots
            .iter()
            .map(|s| s.allocated_bytes)
            .sum::<usize>()
            / self.memory_snapshots.len();

        let memory_efficiency = if peak_memory > 0 {
            average_memory as f64 / peak_memory as f64
        } else {
            1.0
        };

        MemoryStats {
            peak_memory,
            average_memory,
            total_allocations: self.memory_snapshots.len(),
            memory_efficiency,
        }
    }

    /// Compute quantum circuit statistics
    fn compute_circuit_stats(&self) -> CircuitStats {
        if self.circuit_metrics.is_empty() {
            return CircuitStats {
                total_circuits_executed: 0,
                average_circuit_depth: 0.0,
                average_qubit_count: 0.0,
                total_gate_count: 0,
                average_fidelity: None,
                total_shots: 0,
            };
        }

        let total_circuits_executed = self.circuit_metrics.len();
        let average_circuit_depth = self
            .circuit_metrics
            .iter()
            .map(|m| m.circuit_depth as f64)
            .sum::<f64>()
            / total_circuits_executed as f64;

        let average_qubit_count = self
            .circuit_metrics
            .iter()
            .map(|m| m.num_qubits as f64)
            .sum::<f64>()
            / total_circuits_executed as f64;

        let total_gate_count = self.circuit_metrics.iter().map(|m| m.gate_count).sum();

        let fidelities: Vec<f64> = self
            .circuit_metrics
            .iter()
            .filter_map(|m| m.fidelity)
            .collect();

        let average_fidelity = if !fidelities.is_empty() {
            Some(fidelities.iter().sum::<f64>() / fidelities.len() as f64)
        } else {
            None
        };

        let total_shots = self.circuit_metrics.iter().map(|m| m.shots).sum();

        CircuitStats {
            total_circuits_executed,
            average_circuit_depth,
            average_qubit_count,
            total_gate_count,
            average_fidelity,
            total_shots,
        }
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(
        &self,
        operation_stats: &HashMap<String, OperationStats>,
        total_duration: Duration,
    ) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        for (_, stats) in operation_stats {
            let severity = if stats.percentage_of_total > 50.0 {
                BottleneckSeverity::Critical
            } else if stats.percentage_of_total > 20.0 {
                BottleneckSeverity::Major
            } else if stats.percentage_of_total > 10.0 {
                BottleneckSeverity::Minor
            } else {
                BottleneckSeverity::Negligible
            };

            if severity != BottleneckSeverity::Negligible {
                let (description, recommendation) =
                    self.analyze_bottleneck(&stats.operation_name, stats);

                bottlenecks.push(Bottleneck {
                    operation: stats.operation_name.clone(),
                    severity,
                    time_percentage: stats.percentage_of_total,
                    description,
                    recommendation,
                });
            }
        }

        // Sort by severity (Critical first)
        bottlenecks.sort_by(|a, b| {
            b.time_percentage
                .partial_cmp(&a.time_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        bottlenecks
    }

    /// Analyze a specific bottleneck and provide recommendations
    fn analyze_bottleneck(&self, operation_name: &str, stats: &OperationStats) -> (String, String) {
        let description = format!(
            "Operation '{}' consumes {:.1}% of total execution time ({} calls, mean: {:?})",
            operation_name, stats.percentage_of_total, stats.call_count, stats.mean_time
        );

        let recommendation = if operation_name.contains("circuit")
            || operation_name.contains("quantum")
        {
            "Consider circuit optimization: reduce circuit depth, use gate compression, or enable SIMD acceleration".to_string()
        } else if operation_name.contains("gradient") || operation_name.contains("backward") {
            "Consider using parameter shift rule caching or analytical gradients where possible"
                .to_string()
        } else if operation_name.contains("measurement") || operation_name.contains("sampling") {
            "Consider reducing shot count or using approximate sampling techniques".to_string()
        } else if stats.call_count > 1000 {
            format!(
                "High call count ({}). Consider batching operations or caching results",
                stats.call_count
            )
        } else {
            "Analyze this operation for optimization opportunities".to_string()
        };

        (description, recommendation)
    }

    /// Generate optimization recommendations based on profiling data
    fn generate_recommendations(
        &self,
        bottlenecks: &[Bottleneck],
        circuit_stats: &CircuitStats,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Circuit-specific recommendations
        if circuit_stats.total_circuits_executed > 0 {
            if circuit_stats.average_circuit_depth > 100.0 {
                recommendations.push(
                    "High average circuit depth detected. Consider circuit optimization or transpilation".to_string()
                );
            }

            if let Some(fidelity) = circuit_stats.average_fidelity {
                if fidelity < 0.9 {
                    recommendations.push(format!(
                        "Low average fidelity ({:.2}). Consider error mitigation strategies",
                        fidelity
                    ));
                }
            }

            if circuit_stats.average_qubit_count > 20.0 {
                recommendations.push(
                    "Large qubit count. Consider using tensor network simulators or real hardware"
                        .to_string(),
                );
            }
        }

        // Memory recommendations
        if !self.memory_snapshots.is_empty() {
            let mem_stats = self.compute_memory_stats();
            if mem_stats.memory_efficiency < 0.5 {
                recommendations.push(
                    format!(
                        "Low memory efficiency ({:.1}%). Consider memory pooling or incremental computation",
                        mem_stats.memory_efficiency * 100.0
                    )
                );
            }
        }

        // Bottleneck-specific recommendations
        for bottleneck in bottlenecks.iter().filter(|b| {
            matches!(
                b.severity,
                BottleneckSeverity::Critical | BottleneckSeverity::Major
            )
        }) {
            recommendations.push(bottleneck.recommendation.clone());
        }

        recommendations
    }

    /// Print a formatted profiling report
    pub fn print_report(&self, report: &ProfilingReport) {
        println!("\n═══════════════════════════════════════════════════════");
        println!("        Quantum ML Performance Profiling Report        ");
        println!("═══════════════════════════════════════════════════════\n");

        println!("Total Execution Time: {:?}\n", report.total_duration);

        // Operation Statistics
        println!("─────────────────────────────────────────────────────");
        println!("Operation Statistics:");
        println!("─────────────────────────────────────────────────────");

        let mut sorted_ops: Vec<_> = report.operation_stats.values().collect();
        sorted_ops.sort_by(|a, b| {
            b.percentage_of_total
                .partial_cmp(&a.percentage_of_total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for stats in sorted_ops.iter().take(10) {
            println!(
                "  {} ({:.1}%): {} calls, mean {:?}, total {:?}",
                stats.operation_name,
                stats.percentage_of_total,
                stats.call_count,
                stats.mean_time,
                stats.total_time
            );
        }

        // Circuit Statistics
        if report.circuit_stats.total_circuits_executed > 0 {
            println!("\n─────────────────────────────────────────────────────");
            println!("Quantum Circuit Statistics:");
            println!("─────────────────────────────────────────────────────");
            println!(
                "  Total Circuits: {}",
                report.circuit_stats.total_circuits_executed
            );
            println!(
                "  Avg Circuit Depth: {:.1}",
                report.circuit_stats.average_circuit_depth
            );
            println!(
                "  Avg Qubit Count: {:.1}",
                report.circuit_stats.average_qubit_count
            );
            println!("  Total Gates: {}", report.circuit_stats.total_gate_count);
            if let Some(fidelity) = report.circuit_stats.average_fidelity {
                println!("  Avg Fidelity: {:.4}", fidelity);
            }
            println!("  Total Shots: {}", report.circuit_stats.total_shots);
        }

        // Memory Statistics
        println!("\n─────────────────────────────────────────────────────");
        println!("Memory Statistics:");
        println!("─────────────────────────────────────────────────────");
        println!(
            "  Peak Memory: {} MB",
            report.memory_stats.peak_memory / 1_000_000
        );
        println!(
            "  Avg Memory: {} MB",
            report.memory_stats.average_memory / 1_000_000
        );
        println!(
            "  Memory Efficiency: {:.1}%",
            report.memory_stats.memory_efficiency * 100.0
        );

        // Bottlenecks
        if !report.bottlenecks.is_empty() {
            println!("\n─────────────────────────────────────────────────────");
            println!("Performance Bottlenecks:");
            println!("─────────────────────────────────────────────────────");

            for bottleneck in &report.bottlenecks {
                println!("  [{:?}] {}", bottleneck.severity, bottleneck.description);
            }
        }

        // Recommendations
        if !report.recommendations.is_empty() {
            println!("\n─────────────────────────────────────────────────────");
            println!("Optimization Recommendations:");
            println!("─────────────────────────────────────────────────────");

            for (i, rec) in report.recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, rec);
            }
        }

        println!("\n═══════════════════════════════════════════════════════\n");
    }
}

impl Default for QuantumMLProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_creation() {
        let profiler = QuantumMLProfiler::new();
        assert!(profiler.session_start.is_none());
        assert!(profiler.timings.is_empty());
    }

    #[test]
    fn test_operation_timing() {
        let mut profiler = QuantumMLProfiler::new();
        profiler.start_session();

        profiler.time_operation("test_op", || {
            thread::sleep(Duration::from_millis(10));
        });

        assert_eq!(
            profiler
                .timings
                .get("test_op")
                .expect("test_op timing should exist")
                .len(),
            1
        );
    }

    #[test]
    fn test_profiling_report() {
        let mut profiler = QuantumMLProfiler::new();
        profiler.start_session();

        profiler.time_operation("fast_op", || {
            thread::sleep(Duration::from_millis(5));
        });

        profiler.time_operation("slow_op", || {
            thread::sleep(Duration::from_millis(20));
        });

        let report = profiler.end_session().expect("End session should succeed");
        assert_eq!(report.operation_stats.len(), 2);
        assert!(report.total_duration >= Duration::from_millis(25));
    }

    #[test]
    fn test_circuit_metrics() {
        let mut profiler = QuantumMLProfiler::new();
        profiler.start_session();

        profiler.record_circuit_execution(
            "test_circuit",
            5,
            10,
            25,
            Duration::from_millis(100),
            1000,
            Some(0.95),
        );

        let report = profiler.end_session().expect("End session should succeed");
        assert_eq!(report.circuit_stats.total_circuits_executed, 1);
        assert_eq!(report.circuit_stats.average_qubit_count, 5.0);
    }
}
