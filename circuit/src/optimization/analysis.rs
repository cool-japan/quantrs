//! Circuit analysis tools
//!
//! This module provides tools for analyzing quantum circuits and generating optimization reports.

use crate::builder::Circuit;
use crate::optimization::gate_properties::get_gate_properties;
use quantrs2_core::error::QuantRS2Result;
use quantrs2_core::gate::GateOp;
use quantrs2_core::qubit::QubitId;
use std::collections::{HashMap, HashSet};

use std::fmt::Write;
/// Metrics for a quantum circuit
#[derive(Debug, Clone)]
pub struct CircuitMetrics {
    /// Total number of gates
    pub gate_count: usize,
    /// Number of each gate type
    pub gate_types: HashMap<String, usize>,
    /// Circuit depth
    pub depth: usize,
    /// Two-qubit gate count
    pub two_qubit_gates: usize,
    /// Number of qubits used
    pub num_qubits: usize,
    /// Critical path length
    pub critical_path: usize,
    /// Total execution time estimate (ns)
    pub execution_time: f64,
    /// Total error estimate
    pub total_error: f64,
    /// Gate density (gates per qubit)
    pub gate_density: f64,
    /// Parallelism factor
    pub parallelism: f64,
}

impl CircuitMetrics {
    /// Calculate improvement percentage compared to another metric
    #[must_use]
    pub fn improvement_from(&self, other: &Self) -> MetricImprovement {
        MetricImprovement {
            gate_count: Self::percent_change(other.gate_count as f64, self.gate_count as f64),
            depth: Self::percent_change(other.depth as f64, self.depth as f64),
            two_qubit_gates: Self::percent_change(
                other.two_qubit_gates as f64,
                self.two_qubit_gates as f64,
            ),
            execution_time: Self::percent_change(other.execution_time, self.execution_time),
            total_error: Self::percent_change(other.total_error, self.total_error),
        }
    }

    fn percent_change(old_val: f64, new_val: f64) -> f64 {
        if old_val == 0.0 {
            0.0
        } else {
            ((old_val - new_val) / old_val) * 100.0
        }
    }
}

/// Improvement metrics
#[derive(Debug, Clone)]
pub struct MetricImprovement {
    pub gate_count: f64,
    pub depth: f64,
    pub two_qubit_gates: f64,
    pub execution_time: f64,
    pub total_error: f64,
}

/// Circuit analyzer
pub struct CircuitAnalyzer {
    analyze_parallelism: bool,
    analyze_critical_path: bool,
}

impl CircuitAnalyzer {
    /// Create a new circuit analyzer
    #[must_use]
    pub const fn new() -> Self {
        Self {
            analyze_parallelism: true,
            analyze_critical_path: true,
        }
    }

    /// Analyze a circuit and compute metrics
    pub fn analyze<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<CircuitMetrics> {
        let stats = circuit.get_stats();

        // Calculate execution time estimate (simplified model)
        let mut execution_time = 0.0;
        for gate in circuit.gates() {
            execution_time += self.estimate_gate_time(gate.as_ref());
        }

        // Calculate total error estimate
        let total_error = self.estimate_total_error(circuit);

        // Calculate parallelism (average gates per layer)
        let parallelism = if stats.depth > 0 {
            stats.total_gates as f64 / stats.depth as f64
        } else {
            0.0
        };

        Ok(CircuitMetrics {
            gate_count: stats.total_gates,
            gate_types: stats.gate_counts,
            depth: stats.depth,
            two_qubit_gates: stats.two_qubit_gates,
            num_qubits: stats.used_qubits,
            critical_path: stats.depth, // For now, same as depth
            execution_time,
            total_error,
            gate_density: stats.gate_density,
            parallelism,
        })
    }

    /// Estimate execution time for a single gate
    fn estimate_gate_time(&self, gate: &dyn GateOp) -> f64 {
        match gate.name() {
            // Single qubit gates (fast)
            "H" | "X" | "Y" | "Z" | "S" | "T" | "RX" | "RY" | "RZ" => 50.0, // nanoseconds
            // Two qubit gates (slower)
            "CNOT" | "CX" | "CZ" | "CY" | "SWAP" | "CRX" | "CRY" | "CRZ" => 200.0,
            // Multi qubit gates (slowest)
            "Toffoli" | "Fredkin" | "CSWAP" => 500.0,
            // Measurements
            "measure" => 1000.0,
            // Unknown gates
            _ => 100.0,
        }
    }

    /// Estimate total error for the circuit
    fn estimate_total_error<const N: usize>(&self, circuit: &Circuit<N>) -> f64 {
        let mut total_error = 0.0;

        for gate in circuit.gates() {
            total_error += self.estimate_gate_error(gate.as_ref());
        }

        // Add coherence errors based on circuit depth and execution time
        let stats = circuit.get_stats();
        let coherence_error = stats.depth as f64 * 0.001; // 0.1% error per depth layer

        total_error + coherence_error
    }

    /// Estimate error for a single gate
    fn estimate_gate_error(&self, gate: &dyn GateOp) -> f64 {
        match gate.name() {
            // Single qubit gates (low error)
            "H" | "X" | "Y" | "Z" | "S" | "T" => 0.0001,
            // Rotation gates (medium error)
            "RX" | "RY" | "RZ" => 0.0005,
            // Two qubit gates (higher error)
            "CNOT" | "CX" | "CZ" | "CY" | "SWAP" => 0.01,
            // Controlled rotations (higher error)
            "CRX" | "CRY" | "CRZ" => 0.015,
            // Multi qubit gates (highest error)
            "Toffoli" | "Fredkin" | "CSWAP" => 0.05,
            // Measurements (readout error)
            "measure" => 0.02,
            // Unknown gates
            _ => 0.01,
        }
    }

    /// Analyze gate sequence (helper for when we have gate list)
    #[must_use]
    pub fn analyze_gates(&self, gates: &[Box<dyn GateOp>], num_qubits: usize) -> CircuitMetrics {
        let mut gate_types = HashMap::new();
        let mut two_qubit_gates = 0;
        let mut execution_time = 0.0;
        let mut total_error = 0.0;

        // Count gates and accumulate costs
        for gate in gates {
            let gate_name = gate.name().to_string();
            *gate_types.entry(gate_name).or_insert(0) += 1;

            if gate.num_qubits() == 2 {
                two_qubit_gates += 1;
            }

            let props = get_gate_properties(gate.as_ref());
            execution_time += props.cost.duration_ns;
            total_error += props.error.total_error();
        }

        let depth = if self.analyze_critical_path {
            self.calculate_depth(gates)
        } else {
            gates.len()
        };

        let critical_path = if self.analyze_critical_path {
            self.calculate_critical_path(gates)
        } else {
            depth
        };

        let parallelism = if self.analyze_parallelism && depth > 0 {
            gates.len() as f64 / depth as f64
        } else {
            1.0
        };

        CircuitMetrics {
            gate_count: gates.len(),
            gate_types,
            depth,
            two_qubit_gates,
            num_qubits,
            critical_path,
            execution_time,
            total_error,
            gate_density: gates.len() as f64 / num_qubits as f64,
            parallelism,
        }
    }

    /// Calculate circuit depth
    fn calculate_depth(&self, gates: &[Box<dyn GateOp>]) -> usize {
        let mut qubit_depths: HashMap<u32, usize> = HashMap::new();
        let mut max_depth = 0;

        for gate in gates {
            let gate_qubits = gate.qubits();

            // Find the maximum depth among involved qubits
            let current_depth = gate_qubits
                .iter()
                .map(|q| qubit_depths.get(&q.id()).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);

            // Update depth for all involved qubits
            let new_depth = current_depth + 1;
            for qubit in gate_qubits {
                qubit_depths.insert(qubit.id(), new_depth);
            }

            max_depth = max_depth.max(new_depth);
        }

        max_depth
    }

    /// Calculate critical path length
    fn calculate_critical_path(&self, gates: &[Box<dyn GateOp>]) -> usize {
        // Build dependency graph
        let mut dependencies: Vec<HashSet<usize>> = vec![HashSet::new(); gates.len()];
        let mut qubit_last_gate: HashMap<u32, usize> = HashMap::new();

        for (i, gate) in gates.iter().enumerate() {
            for qubit in gate.qubits() {
                if let Some(&prev_gate) = qubit_last_gate.get(&qubit.id()) {
                    dependencies[i].insert(prev_gate);
                }
                qubit_last_gate.insert(qubit.id(), i);
            }
        }

        // Calculate longest path
        let mut path_lengths = vec![0; gates.len()];
        let mut max_path = 0;

        for i in 0..gates.len() {
            let max_dep_length = dependencies[i]
                .iter()
                .map(|&j| path_lengths[j])
                .max()
                .unwrap_or(0);

            path_lengths[i] = max_dep_length + 1;
            max_path = max_path.max(path_lengths[i]);
        }

        max_path
    }
}

impl Default for CircuitAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization report
#[derive(Debug)]
pub struct OptimizationReport {
    /// Initial circuit metrics
    pub initial_metrics: CircuitMetrics,
    /// Final circuit metrics
    pub final_metrics: CircuitMetrics,
    /// List of applied optimization passes
    pub applied_passes: Vec<String>,
}

impl OptimizationReport {
    /// Get improvement metrics
    #[must_use]
    pub fn improvement(&self) -> MetricImprovement {
        self.final_metrics.improvement_from(&self.initial_metrics)
    }

    /// Print a summary of the optimization
    pub fn print_summary(&self) {
        println!("=== Circuit Optimization Report ===");
        println!();
        println!("Initial Metrics:");
        println!("  Gate count: {}", self.initial_metrics.gate_count);
        println!("  Depth: {}", self.initial_metrics.depth);
        println!(
            "  Two-qubit gates: {}",
            self.initial_metrics.two_qubit_gates
        );
        println!(
            "  Execution time: {:.2} ns",
            self.initial_metrics.execution_time
        );
        println!("  Total error: {:.6}", self.initial_metrics.total_error);
        println!();
        println!("Final Metrics:");
        println!("  Gate count: {}", self.final_metrics.gate_count);
        println!("  Depth: {}", self.final_metrics.depth);
        println!("  Two-qubit gates: {}", self.final_metrics.two_qubit_gates);
        println!(
            "  Execution time: {:.2} ns",
            self.final_metrics.execution_time
        );
        println!("  Total error: {:.6}", self.final_metrics.total_error);
        println!();
        println!("Improvements:");
        let improvement = self.improvement();
        println!("  Gate count: {:.1}%", improvement.gate_count);
        println!("  Depth: {:.1}%", improvement.depth);
        println!("  Two-qubit gates: {:.1}%", improvement.two_qubit_gates);
        println!("  Execution time: {:.1}%", improvement.execution_time);
        println!("  Total error: {:.1}%", improvement.total_error);
        println!();
        println!("Applied Passes:");
        for pass in &self.applied_passes {
            println!("  - {pass}");
        }
    }

    /// Generate a detailed report as string
    #[must_use]
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Detailed Circuit Optimization Report ===\n\n");

        // Gate type breakdown
        report.push_str("Gate Type Breakdown:\n");
        report.push_str("Initial:\n");
        for (gate_type, count) in &self.initial_metrics.gate_types {
            let _ = writeln!(report, "  {gate_type}: {count}");
        }
        report.push_str("Final:\n");
        for (gate_type, count) in &self.final_metrics.gate_types {
            let _ = writeln!(report, "  {gate_type}: {count}");
        }

        // Additional metrics
        report.push_str("\nGate Density:\n");
        let _ = writeln!(
            report,
            "  Initial: {:.2} gates/qubit",
            self.initial_metrics.gate_density
        );
        let _ = writeln!(
            report,
            "  Final: {:.2} gates/qubit",
            self.final_metrics.gate_density
        );

        report.push_str("\nParallelism Factor:\n");
        let _ = writeln!(report, "  Initial: {:.2}", self.initial_metrics.parallelism);
        let _ = writeln!(report, "  Final: {:.2}", self.final_metrics.parallelism);

        report
    }
}
