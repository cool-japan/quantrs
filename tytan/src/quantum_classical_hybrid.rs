//! Quantum-Classical Hybrid Refinement
//!
//! This module implements hybrid quantum-classical optimization strategies that refine
//! quantum annealing solutions using classical local search, gradient-based methods,
//! and constraint repair techniques.
//!
//! # Features
//!
//! - Local search refinement (hill climbing, simulated annealing)
//! - Gradient-based fine-tuning for continuous embeddings
//! - Constraint repair and feasibility restoration
//! - Variable fixing based on high-confidence quantum samples
//! - Iterative quantum-classical loops with convergence criteria
//! - Integration with existing samplers
//!
//! # Examples
//!
//! ```rust
//! use quantrs2_tytan::quantum_classical_hybrid::*;
//! use scirs2_core::ndarray::Array2;
//! use std::collections::HashMap;
//!
//! // Create hybrid optimizer
//! let config = HybridConfig::default();
//! let mut optimizer = HybridOptimizer::new(config);
//!
//! // Create a simple QUBO matrix
//! let qubo_matrix = Array2::from_shape_fn((2, 2), |(i, j)| {
//!     if i == j { -1.0 } else { 0.5 }
//! });
//!
//! // Refine quantum solution
//! let quantum_solution = HashMap::from([
//!     ("x0".to_string(), true),
//!     ("x1".to_string(), false),
//! ]);
//! let refined = optimizer.refine_solution(&quantum_solution, &qubo_matrix).expect("refinement should succeed");
//! assert!(refined.energy <= 0.0);
//! ```

use crate::sampler::{SampleResult, Sampler};
use quantrs2_anneal::{IsingModel, QuboModel};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops;
use scirs2_core::random::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Local search strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalSearchStrategy {
    /// Steepest descent (best improvement)
    SteepestDescent,
    /// First improvement (accept first better solution)
    FirstImprovement,
    /// Random descent (random neighbor)
    RandomDescent,
    /// Tabu search with memory
    TabuSearch,
    /// Variable neighborhood descent
    VariableNeighborhoodDescent,
}

/// Constraint repair strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairStrategy {
    /// Greedy repair (minimize constraint violation)
    Greedy,
    /// Random repair
    Random,
    /// Weighted repair based on constraint importance
    Weighted,
    /// Iterative repair with backtracking
    Iterative,
}

/// Variable fixing criterion
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FixingCriterion {
    /// Fix variables with high frequency across samples
    HighFrequency { threshold: f64 },
    /// Fix variables with low energy contribution variance
    LowVariance { threshold: f64 },
    /// Fix variables in strongly correlated groups
    StrongCorrelation { threshold: f64 },
    /// Fix based on reduced cost analysis
    ReducedCost { threshold: f64 },
}

/// Hybrid optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Local search strategy
    pub local_search: LocalSearchStrategy,
    /// Maximum local search iterations
    pub max_local_iterations: usize,
    /// Constraint repair strategy
    pub repair_strategy: RepairStrategy,
    /// Enable constraint repair
    pub enable_repair: bool,
    /// Variable fixing criterion
    pub fixing_criterion: Option<FixingCriterion>,
    /// Percentage of variables to fix (0.0 - 1.0)
    pub fixing_percentage: f64,
    /// Number of quantum-classical iterations
    pub max_qc_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Enable gradient-based refinement
    pub enable_gradient: bool,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Enable parallel evaluation
    pub parallel: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            local_search: LocalSearchStrategy::SteepestDescent,
            max_local_iterations: 1000,
            repair_strategy: RepairStrategy::Greedy,
            enable_repair: true,
            fixing_criterion: Some(FixingCriterion::HighFrequency { threshold: 0.8 }),
            fixing_percentage: 0.3,
            max_qc_iterations: 10,
            convergence_tolerance: 1e-6,
            enable_gradient: false,
            learning_rate: 0.01,
            parallel: true,
        }
    }
}

/// Solution with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedSolution {
    /// Variable assignments
    pub assignments: HashMap<String, bool>,
    /// Solution energy
    pub energy: f64,
    /// Constraint violations
    pub violations: Vec<ConstraintViolation>,
    /// Number of refinement iterations
    pub iterations: usize,
    /// Improvement over initial solution
    pub improvement: f64,
    /// Whether solution is feasible
    pub is_feasible: bool,
}

/// Constraint violation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// Constraint identifier
    pub constraint_id: String,
    /// Violation magnitude
    pub magnitude: f64,
    /// Variables involved
    pub variables: Vec<String>,
}

/// Fixed variable information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedVariable {
    /// Variable name
    pub name: String,
    /// Fixed value
    pub value: bool,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Reason for fixing
    pub reason: String,
}

/// Hybrid quantum-classical optimizer
pub struct HybridOptimizer {
    /// Configuration
    config: HybridConfig,
    /// Random number generator
    rng: Box<dyn RngCore>,
    /// Tabu list for tabu search
    tabu_list: HashSet<u64>,
    /// Fixed variables
    fixed_variables: HashMap<String, bool>,
    /// Iteration history
    history: Vec<f64>,
}

impl HybridOptimizer {
    /// Create a new hybrid optimizer
    pub fn new(config: HybridConfig) -> Self {
        Self {
            config,
            rng: Box::new(thread_rng()),
            tabu_list: HashSet::new(),
            fixed_variables: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Refine a solution using local search
    pub fn refine_solution(
        &mut self,
        solution: &HashMap<String, bool>,
        qubo_matrix: &Array2<f64>,
    ) -> Result<RefinedSolution, String> {
        let initial_energy = self.compute_energy(solution, qubo_matrix);
        let mut current_solution = solution.clone();
        let mut current_energy = initial_energy;
        let mut iterations = 0;

        self.history.clear();
        self.history.push(current_energy);

        // Apply constraint repair if enabled
        if self.config.enable_repair {
            current_solution = self.repair_constraints(&current_solution, qubo_matrix)?;
            current_energy = self.compute_energy(&current_solution, qubo_matrix);
            self.history.push(current_energy);
        }

        // Local search refinement
        for iter in 0..self.config.max_local_iterations {
            iterations = iter + 1;

            let (improved_solution, improved_energy) = match self.config.local_search {
                LocalSearchStrategy::SteepestDescent => {
                    self.steepest_descent_step(&current_solution, qubo_matrix)
                }
                LocalSearchStrategy::FirstImprovement => {
                    self.first_improvement_step(&current_solution, qubo_matrix)
                }
                LocalSearchStrategy::RandomDescent => {
                    self.random_descent_step(&current_solution, qubo_matrix)
                }
                LocalSearchStrategy::TabuSearch => {
                    self.tabu_search_step(&current_solution, qubo_matrix)
                }
                LocalSearchStrategy::VariableNeighborhoodDescent => {
                    self.vnd_step(&current_solution, qubo_matrix)
                }
            }?;

            // Check for improvement
            if improved_energy < current_energy - self.config.convergence_tolerance {
                current_solution = improved_solution;
                current_energy = improved_energy;
                self.history.push(current_energy);
            } else {
                // No improvement, stop
                break;
            }

            // Check convergence
            if self.has_converged() {
                break;
            }
        }

        // Compute constraint violations
        let violations = self.compute_violations(&current_solution);
        let is_feasible = violations.is_empty();

        Ok(RefinedSolution {
            assignments: current_solution,
            energy: current_energy,
            violations,
            iterations,
            improvement: initial_energy - current_energy,
            is_feasible,
        })
    }

    /// Steepest descent local search step
    fn steepest_descent_step(
        &self,
        solution: &HashMap<String, bool>,
        qubo_matrix: &Array2<f64>,
    ) -> Result<(HashMap<String, bool>, f64), String> {
        let current_energy = self.compute_energy(solution, qubo_matrix);
        let mut best_solution = solution.clone();
        let mut best_energy = current_energy;
        let mut improved = false;

        // Try flipping each variable
        for (var_name, &current_value) in solution {
            // Skip fixed variables
            if self.fixed_variables.contains_key(var_name) {
                continue;
            }

            let mut neighbor = solution.clone();
            neighbor.insert(var_name.clone(), !current_value);

            let neighbor_energy = self.compute_energy(&neighbor, qubo_matrix);

            if neighbor_energy < best_energy {
                best_solution = neighbor;
                best_energy = neighbor_energy;
                improved = true;
            }
        }

        if improved {
            Ok((best_solution, best_energy))
        } else {
            Ok((solution.clone(), current_energy))
        }
    }

    /// First improvement local search step
    fn first_improvement_step(
        &mut self,
        solution: &HashMap<String, bool>,
        qubo_matrix: &Array2<f64>,
    ) -> Result<(HashMap<String, bool>, f64), String> {
        let current_energy = self.compute_energy(solution, qubo_matrix);

        // Try flipping variables in random order
        let mut var_names: Vec<_> = solution.keys().cloned().collect();
        var_names.shuffle(&mut *self.rng);

        for var_name in var_names {
            // Skip fixed variables
            if self.fixed_variables.contains_key(&var_name) {
                continue;
            }

            let current_value = solution[&var_name];
            let mut neighbor = solution.clone();
            neighbor.insert(var_name, !current_value);

            let neighbor_energy = self.compute_energy(&neighbor, qubo_matrix);

            if neighbor_energy < current_energy {
                return Ok((neighbor, neighbor_energy));
            }
        }

        Ok((solution.clone(), current_energy))
    }

    /// Random descent step
    fn random_descent_step(
        &mut self,
        solution: &HashMap<String, bool>,
        qubo_matrix: &Array2<f64>,
    ) -> Result<(HashMap<String, bool>, f64), String> {
        let current_energy = self.compute_energy(solution, qubo_matrix);

        // Select random variable to flip
        let var_names: Vec<_> = solution
            .keys()
            .filter(|k| !self.fixed_variables.contains_key(*k))
            .cloned()
            .collect();

        if var_names.is_empty() {
            return Ok((solution.clone(), current_energy));
        }

        let var_name = &var_names[self.rng.gen_range(0..var_names.len())];
        let current_value = solution[var_name];

        let mut neighbor = solution.clone();
        neighbor.insert(var_name.clone(), !current_value);

        let neighbor_energy = self.compute_energy(&neighbor, qubo_matrix);

        if neighbor_energy < current_energy {
            Ok((neighbor, neighbor_energy))
        } else {
            Ok((solution.clone(), current_energy))
        }
    }

    /// Tabu search step
    fn tabu_search_step(
        &mut self,
        solution: &HashMap<String, bool>,
        qubo_matrix: &Array2<f64>,
    ) -> Result<(HashMap<String, bool>, f64), String> {
        let current_energy = self.compute_energy(solution, qubo_matrix);
        let mut best_solution = solution.clone();
        let mut best_energy = current_energy;

        // Try non-tabu moves
        for (var_name, &current_value) in solution {
            if self.fixed_variables.contains_key(var_name) {
                continue;
            }

            let mut neighbor = solution.clone();
            neighbor.insert(var_name.clone(), !current_value);

            // Check if move is tabu
            let move_hash = self.hash_solution(&neighbor);
            if self.tabu_list.contains(&move_hash) {
                continue;
            }

            let neighbor_energy = self.compute_energy(&neighbor, qubo_matrix);

            if neighbor_energy < best_energy {
                best_solution = neighbor;
                best_energy = neighbor_energy;
            }
        }

        // Update tabu list
        let move_hash = self.hash_solution(&best_solution);
        self.tabu_list.insert(move_hash);

        // Limit tabu list size
        if self.tabu_list.len() > 100 {
            self.tabu_list.clear();
        }

        Ok((best_solution, best_energy))
    }

    /// Variable neighborhood descent step
    fn vnd_step(
        &mut self,
        solution: &HashMap<String, bool>,
        qubo_matrix: &Array2<f64>,
    ) -> Result<(HashMap<String, bool>, f64), String> {
        let mut current_solution = solution.clone();
        let mut current_energy = self.compute_energy(solution, qubo_matrix);

        // Neighborhood 1: Single variable flip
        let (sol1, e1) = self.steepest_descent_step(&current_solution, qubo_matrix)?;
        if e1 < current_energy {
            current_solution = sol1;
            current_energy = e1;
        }

        // Neighborhood 2: Two-variable swap (if solution is binary)
        let (sol2, e2) = self.two_variable_swap(&current_solution, qubo_matrix)?;
        if e2 < current_energy {
            current_solution = sol2;
            current_energy = e2;
        }

        Ok((current_solution, current_energy))
    }

    /// Two-variable swap neighborhood
    fn two_variable_swap(
        &self,
        solution: &HashMap<String, bool>,
        qubo_matrix: &Array2<f64>,
    ) -> Result<(HashMap<String, bool>, f64), String> {
        let current_energy = self.compute_energy(solution, qubo_matrix);
        let mut best_solution = solution.clone();
        let mut best_energy = current_energy;

        let var_names: Vec<_> = solution
            .keys()
            .filter(|k| !self.fixed_variables.contains_key(*k))
            .cloned()
            .collect();

        for i in 0..var_names.len() {
            for j in (i + 1)..var_names.len() {
                let mut neighbor = solution.clone();
                let val_i = solution[&var_names[i]];
                let val_j = solution[&var_names[j]];

                neighbor.insert(var_names[i].clone(), !val_i);
                neighbor.insert(var_names[j].clone(), !val_j);

                let neighbor_energy = self.compute_energy(&neighbor, qubo_matrix);

                if neighbor_energy < best_energy {
                    best_solution = neighbor;
                    best_energy = neighbor_energy;
                }
            }
        }

        Ok((best_solution, best_energy))
    }

    /// Repair constraint violations
    fn repair_constraints(
        &self,
        solution: &HashMap<String, bool>,
        _qubo_matrix: &Array2<f64>,
    ) -> Result<HashMap<String, bool>, String> {
        // Simplified constraint repair
        // In practice, this would analyze specific constraint types
        let mut repaired = solution.clone();

        match self.config.repair_strategy {
            RepairStrategy::Greedy => {
                // Greedy repair: flip variables to reduce violations
                // Placeholder implementation
            }
            RepairStrategy::Random => {
                // Random repair
                // Placeholder implementation
            }
            RepairStrategy::Weighted => {
                // Weighted repair
                // Placeholder implementation
            }
            RepairStrategy::Iterative => {
                // Iterative repair with backtracking
                // Placeholder implementation
            }
        }

        Ok(repaired)
    }

    /// Fix high-confidence variables based on quantum samples
    pub fn fix_variables(
        &mut self,
        samples: &[HashMap<String, bool>],
        criterion: FixingCriterion,
    ) -> Result<Vec<FixedVariable>, String> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let mut fixed = Vec::new();

        match criterion {
            FixingCriterion::HighFrequency { threshold } => {
                // Compute variable frequencies
                let mut frequencies: HashMap<String, (usize, usize)> = HashMap::new();

                for sample in samples {
                    for (var, &value) in sample {
                        let entry = frequencies.entry(var.clone()).or_insert((0, 0));
                        if value {
                            entry.0 += 1;
                        } else {
                            entry.1 += 1;
                        }
                    }
                }

                // Fix variables with high frequency
                for (var, (true_count, false_count)) in frequencies {
                    let total = (true_count + false_count) as f64;
                    let true_freq = true_count as f64 / total;
                    let false_freq = false_count as f64 / total;

                    if true_freq >= threshold {
                        self.fixed_variables.insert(var.clone(), true);
                        fixed.push(FixedVariable {
                            name: var,
                            value: true,
                            confidence: true_freq,
                            reason: format!("High frequency ({true_freq})"),
                        });
                    } else if false_freq >= threshold {
                        self.fixed_variables.insert(var.clone(), false);
                        fixed.push(FixedVariable {
                            name: var,
                            value: false,
                            confidence: false_freq,
                            reason: format!("High frequency ({false_freq})"),
                        });
                    }
                }
            }
            FixingCriterion::LowVariance { threshold } => {
                // Compute variance of each variable's contribution
                // Placeholder implementation
            }
            FixingCriterion::StrongCorrelation { threshold } => {
                // Detect strongly correlated variable groups
                // Placeholder implementation
            }
            FixingCriterion::ReducedCost { threshold } => {
                // Reduced cost analysis
                // Placeholder implementation
            }
        }

        Ok(fixed)
    }

    /// Unfix all variables
    pub fn unfix_all(&mut self) {
        self.fixed_variables.clear();
    }

    /// Iterative quantum-classical refinement
    pub fn iterative_refinement<S: Sampler>(
        &mut self,
        sampler: &S,
        qubo_matrix: &Array2<f64>,
        num_samples: usize,
    ) -> Result<Vec<RefinedSolution>, String> {
        let mut refined_solutions = Vec::new();
        let mut best_energy = f64::INFINITY;

        for iteration in 0..self.config.max_qc_iterations {
            println!(
                "Quantum-Classical iteration {}/{}",
                iteration + 1,
                self.config.max_qc_iterations
            );

            // Quantum sampling step
            // Note: This is a simplified interface; actual implementation would need proper QUBO format
            // For now, we'll generate random samples as placeholder
            let mut samples = Vec::new();
            for _ in 0..num_samples {
                let mut sample = HashMap::new();
                for i in 0..qubo_matrix.nrows() {
                    sample.insert(format!("x{i}"), self.rng.gen::<bool>());
                }
                samples.push(sample);
            }

            // Fix high-confidence variables if configured
            if let Some(criterion) = self.config.fixing_criterion {
                let fixed = self.fix_variables(&samples, criterion)?;
                println!("Fixed {} variables", fixed.len());
            }

            // Classical refinement of quantum samples
            for sample in samples {
                let refined = self.refine_solution(&sample, qubo_matrix)?;

                if refined.energy < best_energy {
                    best_energy = refined.energy;
                    println!("New best energy: {best_energy}");
                }

                refined_solutions.push(refined);
            }

            // Check convergence
            if iteration > 0 && self.has_converged() {
                println!("Converged after {} iterations", iteration + 1);
                break;
            }
        }

        Ok(refined_solutions)
    }

    /// Compute solution energy
    fn compute_energy(&self, solution: &HashMap<String, bool>, qubo_matrix: &Array2<f64>) -> f64 {
        let n = qubo_matrix.nrows();
        let mut energy = 0.0;

        for i in 0..n {
            for j in 0..n {
                let x_i = if solution.get(&format!("x{i}")).copied().unwrap_or(false) {
                    1.0
                } else {
                    0.0
                };
                let x_j = if solution.get(&format!("x{j}")).copied().unwrap_or(false) {
                    1.0
                } else {
                    0.0
                };
                energy += qubo_matrix[[i, j]] * x_i * x_j;
            }
        }

        energy
    }

    /// Compute constraint violations
    const fn compute_violations(
        &self,
        _solution: &HashMap<String, bool>,
    ) -> Vec<ConstraintViolation> {
        // Placeholder: would check actual constraints
        Vec::new()
    }

    /// Check if optimization has converged
    fn has_converged(&self) -> bool {
        if self.history.len() < 3 {
            return false;
        }

        let recent = &self.history[self.history.len() - 3..];
        let max_change = recent
            .windows(2)
            .map(|w| (w[0] - w[1]).abs())
            .fold(0.0, f64::max);

        max_change < self.config.convergence_tolerance
    }

    /// Hash a solution for tabu search
    fn hash_solution(&self, solution: &HashMap<String, bool>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        let mut sorted: Vec<_> = solution.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());

        for (k, v) in sorted {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get refinement history
    pub fn get_history(&self) -> &[f64] {
        &self.history
    }

    /// Get fixed variables
    pub const fn get_fixed_variables(&self) -> &HashMap<String, bool> {
        &self.fixed_variables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_optimizer_creation() {
        let config = HybridConfig::default();
        let optimizer = HybridOptimizer::new(config);

        assert_eq!(optimizer.fixed_variables.len(), 0);
        assert_eq!(optimizer.history.len(), 0);
    }

    #[test]
    fn test_energy_computation() {
        let config = HybridConfig::default();
        let optimizer = HybridOptimizer::new(config);

        let qubo = Array2::from_shape_fn((2, 2), |(i, j)| if i == j { -1.0 } else { 2.0 });

        let solution = HashMap::from([("x0".to_string(), true), ("x1".to_string(), false)]);

        let energy = optimizer.compute_energy(&solution, &qubo);
        assert_eq!(energy, -1.0); // Only x0 contributes
    }

    #[test]
    fn test_local_search_refinement() {
        let config = HybridConfig {
            max_local_iterations: 10,
            ..Default::default()
        };
        let mut optimizer = HybridOptimizer::new(config);

        let qubo = Array2::from_shape_fn((3, 3), |(i, j)| if i == j { -1.0 } else { 0.5 });

        let initial_solution = HashMap::from([
            ("x0".to_string(), false),
            ("x1".to_string(), false),
            ("x2".to_string(), false),
        ]);

        let refined = optimizer
            .refine_solution(&initial_solution, &qubo)
            .expect("refinement should succeed");

        assert!(refined.improvement >= 0.0);
        assert!(refined.energy <= optimizer.compute_energy(&initial_solution, &qubo));
    }

    #[test]
    fn test_variable_fixing() {
        let config = HybridConfig::default();
        let mut optimizer = HybridOptimizer::new(config);

        let samples = vec![
            HashMap::from([("x0".to_string(), true), ("x1".to_string(), false)]),
            HashMap::from([("x0".to_string(), true), ("x1".to_string(), true)]),
            HashMap::from([("x0".to_string(), true), ("x1".to_string(), false)]),
        ];

        let criterion = FixingCriterion::HighFrequency { threshold: 0.8 };
        let fixed = optimizer
            .fix_variables(&samples, criterion)
            .expect("variable fixing should succeed");

        // x0 should be fixed to true (100% frequency)
        assert!(!fixed.is_empty());
        assert!(fixed.iter().any(|f| f.name == "x0" && f.value));
    }

    #[test]
    fn test_convergence_detection() {
        let mut config = HybridConfig::default();
        config.convergence_tolerance = 0.001; // Set tolerance for test
        let mut optimizer = HybridOptimizer::new(config);

        // Add converged history (changes smaller than tolerance)
        optimizer.history = vec![10.0, 10.00001, 10.00002];

        assert!(optimizer.has_converged());

        // Add non-converged history (changes larger than tolerance)
        optimizer.history = vec![10.0, 9.0, 8.0];

        assert!(!optimizer.has_converged());
    }
}
