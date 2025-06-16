//! Solution comparison functionality for the solution debugger.

use super::types::{ProblemInfo, Solution};
use serde::Serialize;
use std::collections::HashMap;

/// Solution comparator
pub struct SolutionComparator {
    /// Comparison metrics
    metrics: Vec<ComparisonMetric>,
    /// Reference solutions
    reference_solutions: Vec<Solution>,
}

#[derive(Debug, Clone, Serialize)]
pub enum ComparisonMetric {
    /// Hamming distance
    HammingDistance,
    /// Energy difference
    EnergyDifference,
    /// Constraint satisfaction
    ConstraintSatisfaction,
    /// Structural similarity
    StructuralSimilarity,
    /// Custom metric
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonResult {
    /// Solutions compared
    pub solution1: String,
    pub solution2: String,
    /// Metric results
    pub metrics: HashMap<String, f64>,
    /// Differences
    pub differences: Vec<Difference>,
    /// Similarity score
    pub similarity: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Difference {
    /// Variable name
    pub variable: String,
    /// Value in solution 1
    pub value1: bool,
    /// Value in solution 2
    pub value2: bool,
    /// Impact on objective
    pub objective_impact: f64,
    /// Impact on constraints
    pub constraint_impact: Vec<String>,
}

impl SolutionComparator {
    /// Create new solution comparator
    pub fn new() -> Self {
        Self {
            metrics: vec![
                ComparisonMetric::HammingDistance,
                ComparisonMetric::EnergyDifference,
                ComparisonMetric::ConstraintSatisfaction,
                ComparisonMetric::StructuralSimilarity,
            ],
            reference_solutions: Vec::new(),
        }
    }

    /// Add reference solution
    pub fn add_reference(&mut self, solution: Solution) {
        self.reference_solutions.push(solution);
    }

    /// Compare two solutions
    pub fn compare(
        &self,
        sol1: &Solution,
        sol2: &Solution,
        problem_info: &ProblemInfo,
    ) -> ComparisonResult {
        let mut metrics = HashMap::new();
        let mut differences = Vec::new();

        // Calculate metrics
        for metric in &self.metrics {
            let value = match metric {
                ComparisonMetric::HammingDistance => self.hamming_distance(sol1, sol2),
                ComparisonMetric::EnergyDifference => (sol1.energy - sol2.energy).abs(),
                ComparisonMetric::ConstraintSatisfaction => {
                    self.constraint_satisfaction_diff(sol1, sol2)
                }
                ComparisonMetric::StructuralSimilarity => self.structural_similarity(sol1, sol2),
                ComparisonMetric::Custom { name } => self.custom_metric(sol1, sol2, name),
            };
            metrics.insert(self.metric_name(metric), value);
        }

        // Find differences
        for var in sol1.assignments.keys() {
            let val1 = sol1.assignments.get(var).copied().unwrap_or(false);
            let val2 = sol2.assignments.get(var).copied().unwrap_or(false);

            if val1 != val2 {
                differences.push(Difference {
                    variable: var.clone(),
                    value1: val1,
                    value2: val2,
                    objective_impact: self.calculate_objective_impact(
                        var,
                        val1,
                        val2,
                        problem_info,
                    ),
                    constraint_impact: self.calculate_constraint_impact(
                        var,
                        val1,
                        val2,
                        problem_info,
                    ),
                });
            }
        }

        // Calculate overall similarity (1.0 - normalized hamming distance)
        let max_distance = sol1.assignments.len() as f64;
        let hamming_dist = metrics.get("hamming_distance").copied().unwrap_or(0.0);
        let similarity = 1.0 - (hamming_dist / max_distance);

        ComparisonResult {
            solution1: "sol1".to_string(), // Would use actual solution IDs
            solution2: "sol2".to_string(),
            metrics,
            differences,
            similarity,
        }
    }

    /// Calculate Hamming distance between solutions
    fn hamming_distance(&self, sol1: &Solution, sol2: &Solution) -> f64 {
        let mut distance = 0;

        for var in sol1.assignments.keys() {
            let val1 = sol1.assignments.get(var).copied().unwrap_or(false);
            let val2 = sol2.assignments.get(var).copied().unwrap_or(false);

            if val1 != val2 {
                distance += 1;
            }
        }

        distance as f64
    }

    /// Calculate constraint satisfaction difference
    fn constraint_satisfaction_diff(&self, _sol1: &Solution, _sol2: &Solution) -> f64 {
        // Placeholder - would need actual constraint satisfaction scores
        0.0
    }

    /// Calculate structural similarity
    fn structural_similarity(&self, sol1: &Solution, sol2: &Solution) -> f64 {
        // Calculate similarity based on variable clusters/groups
        // Placeholder implementation
        let total_vars = sol1.assignments.len() as f64;
        let same_vars = sol1
            .assignments
            .iter()
            .filter(|(var, &val1)| sol2.assignments.get(*var).copied().unwrap_or(false) == val1)
            .count() as f64;

        same_vars / total_vars
    }

    /// Calculate custom metric
    fn custom_metric(&self, _sol1: &Solution, _sol2: &Solution, _name: &str) -> f64 {
        // Placeholder for custom metrics
        0.0
    }

    /// Get metric name
    fn metric_name(&self, metric: &ComparisonMetric) -> String {
        match metric {
            ComparisonMetric::HammingDistance => "hamming_distance".to_string(),
            ComparisonMetric::EnergyDifference => "energy_difference".to_string(),
            ComparisonMetric::ConstraintSatisfaction => "constraint_satisfaction".to_string(),
            ComparisonMetric::StructuralSimilarity => "structural_similarity".to_string(),
            ComparisonMetric::Custom { name } => name.clone(),
        }
    }

    /// Calculate objective impact of variable difference
    fn calculate_objective_impact(
        &self,
        _var: &str,
        _val1: bool,
        _val2: bool,
        _problem_info: &ProblemInfo,
    ) -> f64 {
        // Would calculate actual impact based on QUBO matrix
        // Placeholder implementation
        0.0
    }

    /// Calculate constraint impact of variable difference
    fn calculate_constraint_impact(
        &self,
        _var: &str,
        _val1: bool,
        _val2: bool,
        _problem_info: &ProblemInfo,
    ) -> Vec<String> {
        // Would find which constraints are affected by this variable change
        // Placeholder implementation
        Vec::new()
    }
}

impl Default for SolutionComparator {
    fn default() -> Self {
        Self::new()
    }
}
