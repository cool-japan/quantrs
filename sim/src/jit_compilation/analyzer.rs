//! Pattern analysis for JIT compilation
//!
//! This module provides pattern analysis and complexity analysis for gate sequences.

use std::collections::HashMap;

use crate::circuit_interfaces::{InterfaceGate, InterfaceGateType};

use super::types::{CompilationPriority, OptimizationSuggestion};

/// Pattern analyzer for detecting common gate sequences
pub struct PatternAnalyzer {
    /// Pattern frequency tracking
    pattern_frequencies: HashMap<String, usize>,
    /// Pattern complexity analysis
    complexity_analyzer: ComplexityAnalyzer,
    /// Pattern optimization suggestions
    optimization_suggestions: Vec<OptimizationSuggestion>,
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternAnalyzer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pattern_frequencies: HashMap::new(),
            complexity_analyzer: ComplexityAnalyzer::new(),
            optimization_suggestions: Vec::new(),
        }
    }

    /// Analyze gate sequence for patterns
    pub fn analyze_pattern(&mut self, gates: &[InterfaceGate]) -> PatternAnalysisResult {
        let pattern_signature = Self::compute_pattern_signature(gates);

        // Update frequency
        *self
            .pattern_frequencies
            .entry(pattern_signature.clone())
            .or_insert(0) += 1;

        // Analyze complexity
        let complexity = self.complexity_analyzer.analyze_complexity(gates);

        // Generate optimization suggestions
        let suggestions = self.generate_optimization_suggestions(gates, &complexity);

        let frequency = self.pattern_frequencies[&pattern_signature];

        PatternAnalysisResult {
            pattern_signature,
            frequency,
            complexity,
            optimization_suggestions: suggestions,
            compilation_priority: self.compute_compilation_priority(gates),
        }
    }

    /// Compute pattern signature
    fn compute_pattern_signature(gates: &[InterfaceGate]) -> String {
        gates
            .iter()
            .map(|gate| format!("{:?}", gate.gate_type))
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Generate optimization suggestions
    fn generate_optimization_suggestions(
        &self,
        gates: &[InterfaceGate],
        complexity: &PatternComplexity,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Check for fusion opportunities
        if Self::can_fuse_gates(gates) {
            suggestions.push(OptimizationSuggestion::GateFusion);
        }

        // Check for vectorization opportunities
        if complexity.parallelizable_operations > 0 {
            suggestions.push(OptimizationSuggestion::Vectorization);
        }

        // Check for constant folding opportunities
        if complexity.constant_operations > 0 {
            suggestions.push(OptimizationSuggestion::ConstantFolding);
        }

        suggestions
    }

    /// Check if gates can be fused
    fn can_fuse_gates(gates: &[InterfaceGate]) -> bool {
        if gates.len() < 2 {
            return false;
        }

        // Check for consecutive single-qubit gates on same target
        for window in gates.windows(2) {
            if window[0].qubits.len() == 1
                && window[1].qubits.len() == 1
                && window[0].qubits[0] == window[1].qubits[0]
            {
                return true;
            }
        }

        false
    }

    /// Compute compilation priority
    fn compute_compilation_priority(&self, gates: &[InterfaceGate]) -> CompilationPriority {
        let length = gates.len();
        let complexity = self.complexity_analyzer.analyze_complexity(gates);

        if length > 10 && complexity.computational_cost > 100.0 {
            CompilationPriority::High
        } else if length > 5 && complexity.computational_cost > 50.0 {
            CompilationPriority::Medium
        } else {
            CompilationPriority::Low
        }
    }
}

/// Pattern analysis result
#[derive(Debug, Clone)]
pub struct PatternAnalysisResult {
    /// Pattern signature
    pub pattern_signature: String,
    /// Usage frequency
    pub frequency: usize,
    /// Pattern complexity analysis
    pub complexity: PatternComplexity,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    /// Compilation priority
    pub compilation_priority: CompilationPriority,
}

/// Pattern complexity analysis
#[derive(Debug, Clone)]
pub struct PatternComplexity {
    /// Number of gates in pattern
    pub gate_count: usize,
    /// Computational cost estimate
    pub computational_cost: f64,
    /// Memory usage estimate
    pub memory_usage: usize,
    /// Number of parallelizable operations
    pub parallelizable_operations: usize,
    /// Number of constant operations
    pub constant_operations: usize,
    /// Critical path length
    pub critical_path_length: usize,
}

/// Complexity analyzer
pub struct ComplexityAnalyzer {
    /// Gate cost database
    gate_costs: HashMap<InterfaceGateType, f64>,
}

impl Default for ComplexityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplexityAnalyzer {
    #[must_use]
    pub fn new() -> Self {
        let mut gate_costs = HashMap::new();

        // Initialize gate costs (relative computational complexity)
        gate_costs.insert(InterfaceGateType::PauliX, 1.0);
        gate_costs.insert(InterfaceGateType::PauliY, 1.0);
        gate_costs.insert(InterfaceGateType::PauliZ, 1.0);
        gate_costs.insert(InterfaceGateType::Hadamard, 2.0);
        gate_costs.insert(InterfaceGateType::CNOT, 10.0);

        Self { gate_costs }
    }

    /// Analyze pattern complexity
    #[must_use]
    pub fn analyze_complexity(&self, gates: &[InterfaceGate]) -> PatternComplexity {
        let gate_count = gates.len();
        let computational_cost = self.compute_computational_cost(gates);
        let memory_usage = Self::estimate_memory_usage(gates);
        let parallelizable_operations = Self::count_parallelizable_operations(gates);
        let constant_operations = Self::count_constant_operations(gates);
        let critical_path_length = Self::compute_critical_path_length(gates);

        PatternComplexity {
            gate_count,
            computational_cost,
            memory_usage,
            parallelizable_operations,
            constant_operations,
            critical_path_length,
        }
    }

    /// Compute computational cost
    fn compute_computational_cost(&self, gates: &[InterfaceGate]) -> f64 {
        gates
            .iter()
            .map(|gate| {
                // Handle parameterized gates
                match &gate.gate_type {
                    InterfaceGateType::RX(_)
                    | InterfaceGateType::RY(_)
                    | InterfaceGateType::RZ(_) => 5.0,
                    InterfaceGateType::Phase(_) => 3.0,
                    InterfaceGateType::U1(_) => 4.0,
                    InterfaceGateType::U2(_, _) => 6.0,
                    InterfaceGateType::U3(_, _, _) => 8.0,
                    InterfaceGateType::CRX(_)
                    | InterfaceGateType::CRY(_)
                    | InterfaceGateType::CRZ(_)
                    | InterfaceGateType::CPhase(_) => 12.0,
                    _ => self.gate_costs.get(&gate.gate_type).copied().unwrap_or(1.0),
                }
            })
            .sum()
    }

    /// Estimate memory usage
    fn estimate_memory_usage(gates: &[InterfaceGate]) -> usize {
        // Rough estimate based on gate count and types
        gates.len() * 32 + gates.iter().map(|g| g.qubits.len() * 8).sum::<usize>()
    }

    /// Count parallelizable operations
    fn count_parallelizable_operations(gates: &[InterfaceGate]) -> usize {
        // Operations that don't share targets can be parallelized
        let mut parallelizable = 0;
        let mut used_qubits = std::collections::HashSet::new();

        for gate in gates {
            let mut can_parallelize = true;
            for &target in &gate.qubits {
                if used_qubits.contains(&target) {
                    can_parallelize = false;
                    break;
                }
            }

            if can_parallelize {
                parallelizable += 1;
                for &target in &gate.qubits {
                    used_qubits.insert(target);
                }
            } else {
                used_qubits.clear();
                for &target in &gate.qubits {
                    used_qubits.insert(target);
                }
            }
        }

        parallelizable
    }

    /// Count constant operations
    fn count_constant_operations(gates: &[InterfaceGate]) -> usize {
        gates
            .iter()
            .filter(|gate| {
                // Operations with constant parameters can be optimized
                match &gate.gate_type {
                    InterfaceGateType::RX(angle)
                    | InterfaceGateType::RY(angle)
                    | InterfaceGateType::RZ(angle)
                    | InterfaceGateType::Phase(angle) => {
                        angle.abs() < f64::EPSILON
                            || (angle - std::f64::consts::PI).abs() < f64::EPSILON
                    }
                    _ => true, // Non-parameterized gates are considered constant
                }
            })
            .count()
    }

    /// Compute critical path length
    fn compute_critical_path_length(gates: &[InterfaceGate]) -> usize {
        // Simple heuristic: maximum depth of dependency chain
        let mut qubit_depths = HashMap::new();
        let mut max_depth = 0;

        for gate in gates {
            let mut current_depth = 0;
            for &target in &gate.qubits {
                if let Some(&depth) = qubit_depths.get(&target) {
                    current_depth = current_depth.max(depth);
                }
            }
            current_depth += 1;

            for &target in &gate.qubits {
                qubit_depths.insert(target, current_depth);
            }

            max_depth = max_depth.max(current_depth);
        }

        max_depth
    }
}
