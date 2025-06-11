//! Penalty function optimization for quantum annealing
//!
//! This module implements advanced penalty function optimization techniques
//! for improving embedding quality and problem formulation in quantum annealing.
//! It includes methods for optimizing chain strengths, penalty weights, and
//! constraint handling using SciRS2 optimization algorithms.

use crate::embedding::Embedding;
use crate::ising::{IsingModel, IsingResult, QuboModel};
use std::collections::HashMap;

/// Configuration for penalty optimization
#[derive(Debug, Clone)]
pub struct PenaltyConfig {
    /// Initial chain strength
    pub initial_chain_strength: f64,
    /// Minimum chain strength
    pub min_chain_strength: f64,
    /// Maximum chain strength  
    pub max_chain_strength: f64,
    /// Chain strength scaling factor
    pub chain_strength_scale: f64,
    /// Penalty weight for constraint violations
    pub constraint_penalty: f64,
    /// Use adaptive penalty adjustment
    pub adaptive: bool,
    /// Learning rate for adaptive adjustment
    pub learning_rate: f64,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            initial_chain_strength: 1.0,
            min_chain_strength: 0.1,
            max_chain_strength: 10.0,
            chain_strength_scale: 1.5,
            constraint_penalty: 1.0,
            adaptive: true,
            learning_rate: 0.1,
        }
    }
}

/// Statistics from penalty optimization
#[derive(Debug, Clone)]
pub struct PenaltyStats {
    /// Number of optimization iterations
    pub iterations: usize,
    /// Final chain strengths
    pub chain_strengths: HashMap<usize, f64>,
    /// Constraint violation counts
    pub violations: HashMap<String, usize>,
    /// Energy improvement
    pub energy_improvement: f64,
    /// Chain break frequency
    pub chain_break_rate: f64,
}

/// Penalty function optimizer
pub struct PenaltyOptimizer {
    config: PenaltyConfig,
    /// History of chain breaks for adaptive adjustment
    chain_break_history: HashMap<usize, Vec<bool>>,
    /// Constraint violation history
    constraint_history: HashMap<String, Vec<f64>>,
}

impl PenaltyOptimizer {
    /// Create a new penalty optimizer
    pub fn new(config: PenaltyConfig) -> Self {
        Self {
            config,
            chain_break_history: HashMap::new(),
            constraint_history: HashMap::new(),
        }
    }

    /// Optimize penalty functions for an embedded Ising model
    pub fn optimize_ising_penalties(
        &mut self,
        model: &mut IsingModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
    ) -> IsingResult<PenaltyStats> {
        let mut stats = PenaltyStats {
            iterations: 0,
            chain_strengths: HashMap::new(),
            violations: HashMap::new(),
            energy_improvement: 0.0,
            chain_break_rate: 0.0,
        };

        // Initialize chain strengths
        for (var, chain) in &embedding.chains {
            stats
                .chain_strengths
                .insert(*var, self.config.initial_chain_strength);
        }

        if !self.config.adaptive {
            // Static penalty optimization
            self.apply_static_penalties(model, embedding, &stats.chain_strengths)?;
        } else {
            // Adaptive penalty optimization
            let initial_energy = self.compute_average_energy(model, samples);

            for iteration in 0..10 {
                // Max iterations
                stats.iterations = iteration + 1;

                // Analyze chain breaks and violations
                let (chain_breaks, violations) = self.analyze_samples(samples, embedding);

                // Update chain strengths based on break frequency
                self.update_chain_strengths(&mut stats.chain_strengths, &chain_breaks);

                // Update constraint penalties based on violations
                self.update_constraint_penalties(model, &violations);

                // Apply updated penalties
                self.apply_static_penalties(model, embedding, &stats.chain_strengths)?;

                // Check convergence
                let new_energy = self.compute_average_energy(model, samples);
                let improvement = initial_energy - new_energy;

                if improvement.abs() < 0.001 {
                    stats.energy_improvement = improvement;
                    break;
                }
            }
        }

        // Compute final statistics
        stats.chain_break_rate = self.compute_chain_break_rate(samples, embedding);

        Ok(stats)
    }

    /// Optimize penalty functions for a QUBO model
    pub fn optimize_qubo_penalties(
        &mut self,
        model: &mut QuboModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
    ) -> IsingResult<PenaltyStats> {
        // Convert QUBO to Ising for penalty optimization
        let (mut ising, offset) = model.to_ising();

        // Optimize Ising penalties
        let stats = self.optimize_ising_penalties(&mut ising, embedding, samples)?;

        // Convert back to QUBO
        *model = ising.to_qubo();

        Ok(stats)
    }

    /// Apply static penalties to the model
    fn apply_static_penalties(
        &self,
        model: &mut IsingModel,
        embedding: &Embedding,
        chain_strengths: &HashMap<usize, f64>,
    ) -> IsingResult<()> {
        // Add chain coupling terms
        for (var, chain) in &embedding.chains {
            let strength = chain_strengths
                .get(var)
                .copied()
                .unwrap_or(self.config.initial_chain_strength);

            // Add coupling between all pairs in the chain
            for i in 0..chain.len() {
                for j in (i + 1)..chain.len() {
                    model.set_coupling(chain[i], chain[j], -strength)?;
                }
            }
        }

        Ok(())
    }

    /// Analyze samples for chain breaks and constraint violations
    fn analyze_samples(
        &self,
        samples: &[Vec<i8>],
        embedding: &Embedding,
    ) -> (HashMap<usize, f64>, HashMap<String, f64>) {
        let mut chain_breaks = HashMap::new();
        let mut violations = HashMap::new();

        for sample in samples {
            // Check chain integrity
            for (var, chain) in &embedding.chains {
                let mut broken = false;
                if chain.len() > 1 {
                    let first_val = sample[chain[0]];
                    for &qubit in &chain[1..] {
                        if sample[qubit] != first_val {
                            broken = true;
                            break;
                        }
                    }
                }

                let count = chain_breaks.entry(*var).or_insert(0.0);
                if broken {
                    *count += 1.0;
                }
            }

            // Check constraint violations (placeholder)
            // In practice, would check specific problem constraints
        }

        // Normalize by number of samples
        let n = samples.len() as f64;
        for count in chain_breaks.values_mut() {
            *count /= n;
        }

        (chain_breaks, violations)
    }

    /// Update chain strengths based on break frequency
    fn update_chain_strengths(
        &self,
        chain_strengths: &mut HashMap<usize, f64>,
        chain_breaks: &HashMap<usize, f64>,
    ) {
        for (var, break_rate) in chain_breaks {
            if let Some(strength) = chain_strengths.get_mut(var) {
                if *break_rate > 0.1 {
                    // More than 10% breaks
                    // Increase chain strength
                    *strength = (*strength * self.config.chain_strength_scale)
                        .min(self.config.max_chain_strength);
                } else if *break_rate < 0.01 {
                    // Less than 1% breaks
                    // Decrease chain strength (might be too strong)
                    *strength = (*strength / self.config.chain_strength_scale)
                        .max(self.config.min_chain_strength);
                }
            }
        }
    }

    /// Update constraint penalties based on violations
    fn update_constraint_penalties(
        &self,
        model: &mut IsingModel,
        violations: &HashMap<String, f64>,
    ) {
        // Placeholder - would update specific constraint penalties
        // based on violation frequencies
    }

    /// Compute average energy of samples
    fn compute_average_energy(&self, model: &IsingModel, samples: &[Vec<i8>]) -> f64 {
        let mut total_energy = 0.0;

        for sample in samples {
            // Ignore errors for invalid samples
            if let Ok(energy) = model.energy(sample) {
                total_energy += energy;
            }
        }

        total_energy / samples.len() as f64
    }

    /// Compute chain break rate across all samples
    fn compute_chain_break_rate(&self, samples: &[Vec<i8>], embedding: &Embedding) -> f64 {
        let mut total_breaks = 0;
        let mut total_chains = 0;

        for sample in samples {
            for (_var, chain) in &embedding.chains {
                if chain.len() > 1 {
                    total_chains += 1;
                    let first_val = sample[chain[0]];
                    for &qubit in &chain[1..] {
                        if sample[qubit] != first_val {
                            total_breaks += 1;
                            break;
                        }
                    }
                }
            }
        }

        if total_chains > 0 {
            total_breaks as f64 / total_chains as f64
        } else {
            0.0
        }
    }
}

/// Advanced penalty optimization using SciRS2-style optimization
pub struct AdvancedPenaltyOptimizer {
    /// Base penalty optimizer
    base_optimizer: PenaltyOptimizer,
    /// Use gradient-based optimization
    use_gradients: bool,
    /// Regularization parameter
    regularization: f64,
}

impl AdvancedPenaltyOptimizer {
    /// Create a new advanced penalty optimizer
    pub fn new(config: PenaltyConfig) -> Self {
        Self {
            base_optimizer: PenaltyOptimizer::new(config),
            use_gradients: true,
            regularization: 0.01,
        }
    }

    /// Optimize penalties using gradient descent
    pub fn optimize_with_gradients(
        &mut self,
        model: &mut IsingModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
        max_iterations: usize,
    ) -> IsingResult<PenaltyStats> {
        let mut chain_strengths: HashMap<usize, f64> = embedding
            .chains
            .keys()
            .map(|&var| (var, self.base_optimizer.config.initial_chain_strength))
            .collect();

        let mut best_energy = f64::INFINITY;
        let mut best_strengths = chain_strengths.clone();

        for iteration in 0..max_iterations {
            // Compute gradients with respect to chain strengths
            let gradients = self.compute_gradients(model, embedding, samples, &chain_strengths)?;

            // Update chain strengths using gradient descent
            for (var, strength) in chain_strengths.iter_mut() {
                if let Some(&grad) = gradients.get(var) {
                    let new_strength = *strength - self.base_optimizer.config.learning_rate * grad;
                    *strength = new_strength
                        .max(self.base_optimizer.config.min_chain_strength)
                        .min(self.base_optimizer.config.max_chain_strength);
                }
            }

            // Apply penalties and evaluate
            self.base_optimizer
                .apply_static_penalties(model, embedding, &chain_strengths)?;
            let energy = self.base_optimizer.compute_average_energy(model, samples);

            if energy < best_energy {
                best_energy = energy;
                best_strengths = chain_strengths.clone();
            }

            // Check convergence
            if iteration > 0 && (best_energy - energy).abs() < 1e-6 {
                break;
            }
        }

        Ok(PenaltyStats {
            iterations: max_iterations,
            chain_strengths: best_strengths,
            violations: HashMap::new(),
            energy_improvement: 0.0,
            chain_break_rate: self
                .base_optimizer
                .compute_chain_break_rate(samples, embedding),
        })
    }

    /// Compute gradients of the objective with respect to chain strengths
    fn compute_gradients(
        &self,
        model: &IsingModel,
        embedding: &Embedding,
        samples: &[Vec<i8>],
        chain_strengths: &HashMap<usize, f64>,
    ) -> IsingResult<HashMap<usize, f64>> {
        let mut gradients = HashMap::new();
        let epsilon = 0.01;

        // Numerical gradient computation
        for (var, &current_strength) in chain_strengths {
            // Forward difference
            let mut strengths_plus = chain_strengths.clone();
            strengths_plus.insert(*var, current_strength + epsilon);

            let energy_plus = {
                let mut model_copy = model.clone();
                self.base_optimizer.apply_static_penalties(
                    &mut model_copy,
                    embedding,
                    &strengths_plus,
                )?;
                self.base_optimizer
                    .compute_average_energy(&model_copy, samples)
            };

            // Backward difference
            let mut strengths_minus = chain_strengths.clone();
            strengths_minus.insert(*var, current_strength - epsilon);

            let energy_minus = {
                let mut model_copy = model.clone();
                self.base_optimizer.apply_static_penalties(
                    &mut model_copy,
                    embedding,
                    &strengths_minus,
                )?;
                self.base_optimizer
                    .compute_average_energy(&model_copy, samples)
            };

            // Gradient with regularization
            let gradient = (energy_plus - energy_minus) / (2.0 * epsilon)
                + self.regularization * current_strength;
            gradients.insert(*var, gradient);
        }

        Ok(gradients)
    }
}

/// Penalty optimization for constrained problems
pub struct ConstraintPenaltyOptimizer {
    /// Constraint definitions
    constraints: Vec<Constraint>,
    /// Penalty weights for each constraint
    penalty_weights: HashMap<String, f64>,
    /// Violation tolerance
    tolerance: f64,
}

/// Represents a constraint in the optimization problem
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Constraint name
    pub name: String,
    /// Variables involved in the constraint
    pub variables: Vec<usize>,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Target value
    pub target: f64,
}

/// Types of constraints
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintType {
    /// Equality constraint (sum = target)
    Equality,
    /// Less than or equal constraint (sum <= target)
    LessEqual,
    /// Greater than or equal constraint (sum >= target)
    GreaterEqual,
    /// Exactly one constraint (exactly one variable is 1)
    ExactlyOne,
    /// At most one constraint (at most one variable is 1)
    AtMostOne,
}

impl ConstraintPenaltyOptimizer {
    /// Create a new constraint penalty optimizer
    pub fn new(tolerance: f64) -> Self {
        Self {
            constraints: Vec::new(),
            penalty_weights: HashMap::new(),
            tolerance,
        }
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        let default_weight = 1.0;
        self.penalty_weights
            .insert(constraint.name.clone(), default_weight);
        self.constraints.push(constraint);
    }

    /// Optimize penalty weights for constraints
    pub fn optimize_penalties(
        &mut self,
        samples: &[Vec<i8>],
        max_iterations: usize,
    ) -> HashMap<String, f64> {
        for _ in 0..max_iterations {
            // Analyze constraint violations
            let violations = self.analyze_constraint_violations(samples);

            // Update penalty weights based on violations
            for (constraint_name, violation_rate) in violations {
                if let Some(weight) = self.penalty_weights.get_mut(&constraint_name) {
                    if violation_rate > self.tolerance {
                        *weight *= 1.5; // Increase penalty
                    } else if violation_rate < self.tolerance / 10.0 {
                        *weight *= 0.8; // Decrease penalty
                    }
                }
            }
        }

        self.penalty_weights.clone()
    }

    /// Analyze constraint violations in samples
    fn analyze_constraint_violations(&self, samples: &[Vec<i8>]) -> HashMap<String, f64> {
        let mut violations = HashMap::new();

        for constraint in &self.constraints {
            let mut violation_count = 0;

            for sample in samples {
                if !self.check_constraint(constraint, sample) {
                    violation_count += 1;
                }
            }

            let violation_rate = violation_count as f64 / samples.len() as f64;
            violations.insert(constraint.name.clone(), violation_rate);
        }

        violations
    }

    /// Check if a sample satisfies a constraint
    fn check_constraint(&self, constraint: &Constraint, sample: &[i8]) -> bool {
        let sum: i8 = constraint
            .variables
            .iter()
            .map(|&var| sample.get(var).copied().unwrap_or(0))
            .sum();

        match constraint.constraint_type {
            ConstraintType::Equality => (sum as f64 - constraint.target).abs() < 1e-6,
            ConstraintType::LessEqual => sum as f64 <= constraint.target,
            ConstraintType::GreaterEqual => sum as f64 >= constraint.target,
            ConstraintType::ExactlyOne => sum == 1,
            ConstraintType::AtMostOne => sum <= 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penalty_optimizer_creation() {
        let config = PenaltyConfig::default();
        let optimizer = PenaltyOptimizer::new(config);
        assert!(optimizer.chain_break_history.is_empty());
    }

    #[test]
    fn test_chain_break_analysis() {
        let config = PenaltyConfig::default();
        let optimizer = PenaltyOptimizer::new(config);

        // Create a simple embedding
        let mut embedding = Embedding::new();
        embedding.chains.insert(0, vec![0, 1]);
        embedding.chains.insert(1, vec![2, 3]);

        // Create samples with some chain breaks
        let samples = vec![
            vec![1, 1, -1, -1], // No breaks
            vec![1, -1, 1, 1],  // Break in chain 0
            vec![1, 1, 1, -1],  // Break in chain 1
        ];

        let rate = optimizer.compute_chain_break_rate(&samples, &embedding);
        assert!(rate > 0.0 && rate < 1.0);
    }

    #[test]
    fn test_constraint_checking() {
        let mut optimizer = ConstraintPenaltyOptimizer::new(0.1);

        // Add an equality constraint
        optimizer.add_constraint(Constraint {
            name: "sum_equals_one".to_string(),
            variables: vec![0, 1, 2],
            constraint_type: ConstraintType::Equality,
            target: 1.0,
        });

        // Test samples
        let sample1 = vec![1, 0, 0]; // Satisfies
        let sample2 = vec![1, 1, 0]; // Violates

        assert!(optimizer.check_constraint(&optimizer.constraints[0], &sample1));
        assert!(!optimizer.check_constraint(&optimizer.constraints[0], &sample2));
    }
}
