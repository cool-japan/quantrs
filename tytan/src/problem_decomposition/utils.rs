//! Utility functions for problem decomposition

use super::types::*;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

/// Integration utilities for combining solutions from subproblems
pub struct SolutionIntegrator {
    strategy: IntegrationStrategy,
    weight_scheme: WeightScheme,
    conflict_resolution: ConflictResolution,
}

/// Weighting schemes for solution integration
#[derive(Debug, Clone)]
pub enum WeightScheme {
    /// Equal weights for all subproblems
    Uniform,
    /// Weights based on subproblem size
    SizeBased,
    /// Weights based on solution quality
    QualityBased,
    /// Weights based on confidence/uncertainty
    ConfidenceBased,
    /// Custom weights
    Custom(Vec<f64>),
}

/// Conflict resolution strategies
#[derive(Debug, Clone)]
pub enum ConflictResolution {
    /// Take majority vote
    MajorityVote,
    /// Use highest quality solution
    HighestQuality,
    /// Use most confident solution
    MostConfident,
    /// Random selection
    Random,
    /// Weighted average
    WeightedAverage,
}

impl SolutionIntegrator {
    /// Create new solution integrator
    pub fn new(strategy: IntegrationStrategy) -> Self {
        Self {
            strategy,
            weight_scheme: WeightScheme::Uniform,
            conflict_resolution: ConflictResolution::MajorityVote,
        }
    }

    /// Set weighting scheme
    pub fn with_weight_scheme(mut self, scheme: WeightScheme) -> Self {
        self.weight_scheme = scheme;
        self
    }

    /// Set conflict resolution strategy
    pub fn with_conflict_resolution(mut self, resolution: ConflictResolution) -> Self {
        self.conflict_resolution = resolution;
        self
    }

    /// Integrate solutions from multiple subproblems
    pub fn integrate_solutions(
        &self,
        component_solutions: &[ComponentSolution],
        global_var_map: &HashMap<String, usize>,
    ) -> Result<IntegratedSolution, String> {
        match self.strategy {
            IntegrationStrategy::WeightedVoting => {
                self.weighted_voting_integration(component_solutions, global_var_map)
            }
            IntegrationStrategy::Consensus => {
                self.consensus_integration(component_solutions, global_var_map)
            }
            IntegrationStrategy::BestSelection => {
                self.best_selection_integration(component_solutions, global_var_map)
            }
            IntegrationStrategy::MajorityVoting => {
                self.majority_voting_integration(component_solutions, global_var_map)
            }
        }
    }

    /// Weighted voting integration
    fn weighted_voting_integration(
        &self,
        component_solutions: &[ComponentSolution],
        _global_var_map: &HashMap<String, usize>,
    ) -> Result<IntegratedSolution, String> {
        let weights = self.compute_weights(component_solutions)?;
        let mut integrated_assignment = HashMap::new();
        let mut variable_votes: HashMap<String, f64> = HashMap::new();

        // Collect weighted votes for each variable
        for (i, solution) in component_solutions.iter().enumerate() {
            let weight = weights.get(i).unwrap_or(&1.0);

            for (var_name, &value) in &solution.assignment {
                let vote = if value { *weight } else { -*weight };
                *variable_votes.entry(var_name.clone()).or_insert(0.0) += vote;
            }
        }

        // Make final decisions
        for (var_name, vote_sum) in variable_votes {
            integrated_assignment.insert(var_name, vote_sum > 0.0);
        }

        // Calculate integrated energy
        let energy = self.calculate_integrated_energy(component_solutions, &weights);
        let confidence = self.calculate_integration_confidence(component_solutions, &weights);

        Ok(IntegratedSolution {
            assignment: integrated_assignment,
            energy,
            confidence,
            component_solutions: component_solutions.to_vec(),
        })
    }

    /// Consensus integration
    fn consensus_integration(
        &self,
        component_solutions: &[ComponentSolution],
        _global_var_map: &HashMap<String, usize>,
    ) -> Result<IntegratedSolution, String> {
        let mut consensus_assignment = HashMap::new();
        let mut conflicts = Vec::new();

        // Find variables that appear in multiple solutions
        let mut variable_values: HashMap<String, Vec<bool>> = HashMap::new();

        for solution in component_solutions {
            for (var_name, &value) in &solution.assignment {
                variable_values
                    .entry(var_name.clone())
                    .or_default()
                    .push(value);
            }
        }

        // Build consensus
        for (var_name, values) in variable_values {
            if values.is_empty() {
                continue;
            }

            // Check for consensus
            let first_value = values[0];
            if values.iter().all(|&v| v == first_value) {
                // Consensus found
                consensus_assignment.insert(var_name, first_value);
            } else {
                // Conflict - use resolution strategy
                let resolved_value = self.resolve_conflict(&var_name, &values)?;
                consensus_assignment.insert(var_name.clone(), resolved_value);
                conflicts.push(var_name);
            }
        }

        let energy = component_solutions
            .iter()
            .map(|s| s.energy)
            .fold(0.0, |acc, e| acc + e);
        let confidence = if conflicts.is_empty() { 1.0 } else { 0.5 };

        Ok(IntegratedSolution {
            assignment: consensus_assignment,
            energy,
            confidence,
            component_solutions: component_solutions.to_vec(),
        })
    }

    /// Best selection integration
    fn best_selection_integration(
        &self,
        component_solutions: &[ComponentSolution],
        _global_var_map: &HashMap<String, usize>,
    ) -> Result<IntegratedSolution, String> {
        // Find best solution by energy
        let best_solution = component_solutions
            .iter()
            .min_by(|a, b| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or("No solutions provided")?;

        Ok(IntegratedSolution {
            assignment: best_solution.assignment.clone(),
            energy: best_solution.energy,
            confidence: best_solution.weight,
            component_solutions: component_solutions.to_vec(),
        })
    }

    /// Majority voting integration
    fn majority_voting_integration(
        &self,
        component_solutions: &[ComponentSolution],
        _global_var_map: &HashMap<String, usize>,
    ) -> Result<IntegratedSolution, String> {
        let mut integrated_assignment = HashMap::new();
        let mut variable_votes: HashMap<String, (usize, usize)> = HashMap::new(); // (true_votes, false_votes)

        // Collect votes
        for solution in component_solutions {
            for (var_name, &value) in &solution.assignment {
                let (true_votes, false_votes) =
                    variable_votes.entry(var_name.clone()).or_insert((0, 0));
                if value {
                    *true_votes += 1;
                } else {
                    *false_votes += 1;
                }
            }
        }

        // Make decisions based on majority
        for (var_name, (true_votes, false_votes)) in variable_votes {
            integrated_assignment.insert(var_name, true_votes > false_votes);
        }

        let energy = component_solutions
            .iter()
            .map(|s| s.energy)
            .fold(0.0, |acc, e| acc + e);
        let confidence = 0.8; // Default confidence for majority voting

        Ok(IntegratedSolution {
            assignment: integrated_assignment,
            energy,
            confidence,
            component_solutions: component_solutions.to_vec(),
        })
    }

    /// Compute weights for component solutions
    fn compute_weights(&self, solutions: &[ComponentSolution]) -> Result<Vec<f64>, String> {
        match &self.weight_scheme {
            WeightScheme::Uniform => Ok(vec![1.0; solutions.len()]),
            WeightScheme::SizeBased => {
                let sizes: Vec<f64> = solutions
                    .iter()
                    .map(|s| s.assignment.len() as f64)
                    .collect();
                let total_size: f64 = sizes.iter().sum();
                Ok(sizes.into_iter().map(|s| s / total_size).collect())
            }
            WeightScheme::QualityBased => {
                // Higher weight for lower energy (better quality)
                let max_energy = solutions
                    .iter()
                    .map(|s| s.energy)
                    .fold(f64::NEG_INFINITY, f64::max);

                let weights: Vec<f64> = solutions
                    .iter()
                    .map(|s| max_energy - s.energy + 1.0)
                    .collect();
                let total_weight: f64 = weights.iter().sum();
                Ok(weights.into_iter().map(|w| w / total_weight).collect())
            }
            WeightScheme::ConfidenceBased => {
                let weights: Vec<f64> = solutions.iter().map(|s| s.weight).collect();
                let total_weight: f64 = weights.iter().sum();
                if total_weight > 0.0 {
                    Ok(weights.into_iter().map(|w| w / total_weight).collect())
                } else {
                    Ok(vec![1.0 / solutions.len() as f64; solutions.len()])
                }
            }
            WeightScheme::Custom(weights) => {
                if weights.len() == solutions.len() {
                    Ok(weights.clone())
                } else {
                    Err("Custom weights length mismatch".to_string())
                }
            }
        }
    }

    /// Resolve conflicts between different variable assignments
    fn resolve_conflict(&self, _var_name: &str, values: &[bool]) -> Result<bool, String> {
        match self.conflict_resolution {
            ConflictResolution::MajorityVote => {
                let true_count = values.iter().filter(|&&v| v).count();
                Ok(true_count > values.len() / 2)
            }
            ConflictResolution::Random => {
                use scirs2_core::random::prelude::*;
                let mut rng = thread_rng();
                let index = rng.gen_range(0..values.len());
                Ok(values[index])
            }
            ConflictResolution::HighestQuality => {
                // Would need access to solution qualities - simplified
                Ok(values[0])
            }
            ConflictResolution::MostConfident => {
                // Would need access to confidence scores - simplified
                Ok(values[0])
            }
            ConflictResolution::WeightedAverage => {
                let true_count = values.iter().filter(|&&v| v).count();
                Ok(true_count as f64 / values.len() as f64 > 0.5)
            }
        }
    }

    /// Calculate integrated energy
    fn calculate_integrated_energy(&self, solutions: &[ComponentSolution], weights: &[f64]) -> f64 {
        solutions
            .iter()
            .zip(weights.iter())
            .map(|(solution, weight)| solution.energy * weight)
            .sum()
    }

    /// Calculate integration confidence
    fn calculate_integration_confidence(
        &self,
        solutions: &[ComponentSolution],
        weights: &[f64],
    ) -> f64 {
        if solutions.is_empty() {
            return 0.0;
        }

        // Weighted average of component confidences
        let weighted_confidence: f64 = solutions
            .iter()
            .zip(weights.iter())
            .map(|(solution, weight)| solution.weight * weight)
            .sum();

        weighted_confidence.min(1.0).max(0.0)
    }
}

/// Performance analysis utilities
pub struct DecompositionAnalyzer {
    metrics_history: Vec<DecompositionMetrics>,
}

impl Default for DecompositionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl DecompositionAnalyzer {
    /// Create new decomposition analyzer
    pub fn new() -> Self {
        Self {
            metrics_history: Vec::new(),
        }
    }

    /// Analyze decomposition quality
    pub fn analyze_decomposition(
        &mut self,
        decomposition: &Partitioning,
        _original_qubo: &Array2<f64>,
    ) -> DecompositionMetrics {
        let start_time = std::time::Instant::now();

        let width = self.calculate_decomposition_width(decomposition);
        let num_clusters = decomposition.subproblems.len();
        let balance_factor = self.calculate_balance_factor(decomposition);
        let separator_size = self.calculate_average_separator_size(decomposition);

        let metrics = DecompositionMetrics {
            width,
            num_clusters,
            balance_factor,
            separator_size,
            decomposition_time: start_time.elapsed(),
        };

        self.metrics_history.push(metrics.clone());
        metrics
    }

    /// Calculate decomposition width (maximum subproblem size)
    fn calculate_decomposition_width(&self, decomposition: &Partitioning) -> usize {
        decomposition
            .subproblems
            .iter()
            .map(|sub| sub.variables.len())
            .max()
            .unwrap_or(0)
    }

    /// Calculate balance factor (how evenly distributed subproblems are)
    fn calculate_balance_factor(&self, decomposition: &Partitioning) -> f64 {
        if decomposition.subproblems.is_empty() {
            return 1.0;
        }

        let sizes: Vec<usize> = decomposition
            .subproblems
            .iter()
            .map(|sub| sub.variables.len())
            .collect();

        let mean_size = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let variance = sizes
            .iter()
            .map(|&size| (size as f64 - mean_size).powi(2))
            .sum::<f64>()
            / sizes.len() as f64;

        // Balance factor is inverse of coefficient of variation
        if mean_size > 0.0 {
            mean_size / (variance.sqrt() + 1e-10)
        } else {
            1.0
        }
    }

    /// Calculate average separator size
    fn calculate_average_separator_size(&self, decomposition: &Partitioning) -> f64 {
        if decomposition.coupling_terms.is_empty() {
            return 0.0;
        }

        // Count unique variables involved in coupling terms
        let mut coupling_vars = std::collections::HashSet::new();
        for coupling in &decomposition.coupling_terms {
            coupling_vars.insert(&coupling.var1);
            coupling_vars.insert(&coupling.var2);
        }

        coupling_vars.len() as f64 / decomposition.subproblems.len() as f64
    }

    /// Get performance trends
    pub fn get_performance_trends(&self) -> Vec<(usize, f64, f64)> {
        self.metrics_history
            .iter()
            .enumerate()
            .map(|(i, metrics)| {
                (
                    i,
                    metrics.balance_factor,
                    metrics.decomposition_time.as_secs_f64(),
                )
            })
            .collect()
    }

    /// Generate decomposition report
    pub fn generate_report(&self) -> String {
        if self.metrics_history.is_empty() {
            return "No decomposition metrics available".to_string();
        }

        let latest = &self.metrics_history[self.metrics_history.len() - 1];

        format!(
            "Decomposition Analysis Report\n\
             ============================\n\
             Width (max subproblem size): {}\n\
             Number of clusters: {}\n\
             Balance factor: {:.3}\n\
             Average separator size: {:.3}\n\
             Decomposition time: {:.3}s\n\
             Total decompositions analyzed: {}",
            latest.width,
            latest.num_clusters,
            latest.balance_factor,
            latest.separator_size,
            latest.decomposition_time.as_secs_f64(),
            self.metrics_history.len()
        )
    }
}

/// Validation utilities for decomposition correctness
pub struct DecompositionValidator;

impl DecompositionValidator {
    /// Validate that decomposition preserves problem structure
    pub fn validate_decomposition(
        partitioning: &Partitioning,
        original_qubo: &Array2<f64>,
        original_var_map: &HashMap<String, usize>,
    ) -> Result<ValidationReport, String> {
        let mut issues = Vec::new();

        // Check variable coverage
        let coverage_issue = Self::check_variable_coverage(partitioning, original_var_map);
        if let Some(issue) = coverage_issue {
            issues.push(issue);
        }

        // Check problem equivalence
        let equivalence_issue = Self::check_problem_equivalence(partitioning, original_qubo);
        if let Some(issue) = equivalence_issue {
            issues.push(issue);
        }

        // Check coupling consistency
        let coupling_issue = Self::check_coupling_consistency(partitioning, original_qubo);
        if let Some(issue) = coupling_issue {
            issues.push(issue);
        }

        Ok(ValidationReport {
            is_valid: issues.is_empty(),
            issues,
            coverage_percentage: Self::calculate_coverage_percentage(
                partitioning,
                original_var_map,
            ),
        })
    }

    /// Check that all variables are covered by subproblems
    fn check_variable_coverage(
        partitioning: &Partitioning,
        original_var_map: &HashMap<String, usize>,
    ) -> Option<ValidationIssue> {
        let mut covered_vars = std::collections::HashSet::new();

        for subproblem in &partitioning.subproblems {
            for var in &subproblem.variables {
                covered_vars.insert(var.clone());
            }
        }

        let missing_vars: Vec<_> = original_var_map
            .keys()
            .filter(|var| !covered_vars.contains(*var))
            .cloned()
            .collect();

        if !missing_vars.is_empty() {
            Some(ValidationIssue {
                issue_type: "Missing Variables".to_string(),
                description: format!(
                    "Variables not covered by any subproblem: {:?}",
                    missing_vars
                ),
                severity: ValidationSeverity::Critical,
            })
        } else {
            None
        }
    }

    /// Check that subproblems preserve original problem structure
    fn check_problem_equivalence(
        partitioning: &Partitioning,
        original_qubo: &Array2<f64>,
    ) -> Option<ValidationIssue> {
        // Check that internal energies are preserved
        // This is a simplified check - would need more sophisticated validation

        let total_internal_terms: f64 = partitioning
            .subproblems
            .iter()
            .map(|sub| sub.qubo.sum())
            .sum();

        let total_coupling_terms: f64 = partitioning
            .coupling_terms
            .iter()
            .map(|coupling| coupling.weight.abs())
            .sum();

        let original_total = original_qubo.sum();
        let reconstructed_total = total_internal_terms + total_coupling_terms;

        let relative_error =
            (original_total - reconstructed_total).abs() / original_total.abs().max(1e-10);

        if relative_error > 0.01 {
            Some(ValidationIssue {
                issue_type: "Energy Mismatch".to_string(),
                description: format!(
                    "Relative error in total energy: {:.3}%",
                    relative_error * 100.0
                ),
                severity: ValidationSeverity::Warning,
            })
        } else {
            None
        }
    }

    /// Check consistency of coupling terms
    fn check_coupling_consistency(
        partitioning: &Partitioning,
        _original_qubo: &Array2<f64>,
    ) -> Option<ValidationIssue> {
        // Check that coupling terms match original off-diagonal elements
        let mut inconsistencies = 0;

        for coupling in &partitioning.coupling_terms {
            // This would require mapping back to original indices
            // Simplified check
            if coupling.weight.abs() < 1e-10 {
                inconsistencies += 1;
            }
        }

        if inconsistencies > partitioning.coupling_terms.len() / 10 {
            Some(ValidationIssue {
                issue_type: "Coupling Inconsistency".to_string(),
                description: format!("{} coupling terms have near-zero weights", inconsistencies),
                severity: ValidationSeverity::Warning,
            })
        } else {
            None
        }
    }

    /// Calculate what percentage of variables are covered
    fn calculate_coverage_percentage(
        partitioning: &Partitioning,
        original_var_map: &HashMap<String, usize>,
    ) -> f64 {
        let mut covered_vars = std::collections::HashSet::new();

        for subproblem in &partitioning.subproblems {
            for var in &subproblem.variables {
                covered_vars.insert(var.clone());
            }
        }

        covered_vars.len() as f64 / original_var_map.len() as f64 * 100.0
    }
}

/// Validation report structure
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub coverage_percentage: f64,
}

/// Individual validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub issue_type: String,
    pub description: String,
    pub severity: ValidationSeverity,
}

/// Validation issue severity levels
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution_integrator() {
        let integrator = SolutionIntegrator::new(IntegrationStrategy::WeightedVoting);

        let mut solutions = vec![
            ComponentSolution {
                subproblem_id: 0,
                assignment: [("x0".to_string(), true), ("x1".to_string(), false)]
                    .iter()
                    .cloned()
                    .collect(),
                energy: 1.0,
                weight: 0.8,
            },
            ComponentSolution {
                subproblem_id: 1,
                assignment: [("x0".to_string(), true), ("x1".to_string(), true)]
                    .iter()
                    .cloned()
                    .collect(),
                energy: 2.0,
                weight: 0.6,
            },
        ];

        let global_var_map = [("x0".to_string(), 0), ("x1".to_string(), 1)]
            .iter()
            .cloned()
            .collect();

        let mut result = integrator.integrate_solutions(&solutions, &global_var_map);
        assert!(result.is_ok());

        let integrated = result.unwrap();
        assert_eq!(integrated.assignment.len(), 2);
        assert!(integrated.assignment.contains_key("x0"));
        assert!(integrated.assignment.contains_key("x1"));
    }

    #[test]
    fn test_decomposition_analyzer() {
        let mut analyzer = DecompositionAnalyzer::new();

        // Create mock partitioning
        let partitioning = Partitioning {
            partition_assignment: vec![0, 0, 1, 1],
            subproblems: vec![
                Subproblem {
                    id: 0,
                    variables: vec!["x0".to_string(), "x1".to_string()],
                    qubo: Array2::zeros((2, 2)),
                    var_map: HashMap::new(),
                },
                Subproblem {
                    id: 1,
                    variables: vec!["x2".to_string(), "x3".to_string()],
                    qubo: Array2::zeros((2, 2)),
                    var_map: HashMap::new(),
                },
            ],
            coupling_terms: vec![],
            metrics: PartitionMetrics {
                edge_cut: 1.0,
                balance: 1.0,
                modularity: 0.5,
                conductance: 0.1,
            },
        };

        let mut original_qubo = Array2::zeros((4, 4));
        let mut metrics = analyzer.analyze_decomposition(&partitioning, &original_qubo);

        assert_eq!(metrics.width, 2);
        assert_eq!(metrics.num_clusters, 2);
        assert!(metrics.balance_factor > 0.0);
    }

    #[test]
    fn test_decomposition_validator() {
        let partitioning = Partitioning {
            partition_assignment: vec![0, 0, 1, 1],
            subproblems: vec![
                Subproblem {
                    id: 0,
                    variables: vec!["x0".to_string(), "x1".to_string()],
                    qubo: Array2::zeros((2, 2)),
                    var_map: HashMap::new(),
                },
                Subproblem {
                    id: 1,
                    variables: vec!["x2".to_string(), "x3".to_string()],
                    qubo: Array2::zeros((2, 2)),
                    var_map: HashMap::new(),
                },
            ],
            coupling_terms: vec![],
            metrics: PartitionMetrics {
                edge_cut: 1.0,
                balance: 1.0,
                modularity: 0.5,
                conductance: 0.1,
            },
        };

        let mut original_qubo = Array2::zeros((4, 4));
        let original_var_map = [
            ("x0".to_string(), 0),
            ("x1".to_string(), 1),
            ("x2".to_string(), 2),
            ("x3".to_string(), 3),
        ]
        .iter()
        .cloned()
        .collect();

        let mut report = DecompositionValidator::validate_decomposition(
            &partitioning,
            &original_qubo,
            &original_var_map,
        );
        assert!(report.is_ok());

        let validation_report = report.unwrap();
        assert_eq!(validation_report.coverage_percentage, 100.0);
    }
}
