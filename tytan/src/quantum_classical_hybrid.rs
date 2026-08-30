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

use crate::optimization::constraints::Constraint;
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
    /// Problem constraints used for feasibility repair/checking
    constraints: Vec<Constraint>,
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
            constraints: Vec::new(),
        }
    }

    /// Attach the problem's constraints so that constraint repair
    /// (`RepairStrategy`) and feasibility checking (`compute_violations`)
    /// operate on the real problem instead of being no-ops.
    pub fn set_constraints(&mut self, constraints: Vec<Constraint>) {
        self.constraints = constraints;
    }

    /// Builder-style variant of [`Self::set_constraints`].
    #[must_use]
    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Get the currently attached constraints
    pub fn get_constraints(&self) -> &[Constraint] {
        &self.constraints
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

        let var_name = &var_names[self.rng.random_range(0..var_names.len())];
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

    /// Total (unsigned) constraint violation across all attached constraints
    fn total_violation(&self, solution: &HashMap<String, bool>) -> f64 {
        self.constraints
            .iter()
            .map(|c| c.violation(solution).abs())
            .sum()
    }

    /// Total constraint violation weighted by each constraint's
    /// `penalty_weight` (defaulting to `1.0` when unset)
    fn weighted_violation(&self, solution: &HashMap<String, bool>) -> f64 {
        self.constraints
            .iter()
            .map(|c| c.penalty_weight.unwrap_or(1.0) * c.violation(solution).abs())
            .sum()
    }

    /// Variables that participate in at least one currently-violated
    /// constraint and are not fixed; these are the only sensible candidates
    /// to flip during constraint repair.
    fn repairable_variables(&self, solution: &HashMap<String, bool>) -> Vec<String> {
        let mut vars: Vec<String> = self
            .constraints
            .iter()
            .filter(|c| c.violation(solution).abs() > 1e-9)
            .flat_map(|c| c.variables.iter().cloned())
            .filter(|v| solution.contains_key(v) && !self.fixed_variables.contains_key(v))
            .collect();
        vars.sort();
        vars.dedup();
        vars
    }

    /// Repair constraint violations
    ///
    /// Actually walks the attached [`Constraint`]s (see [`Self::set_constraints`])
    /// and flips variables to reduce real violation according to the
    /// configured [`RepairStrategy`]. When no constraints are attached this is
    /// a genuine no-op (there is nothing to repair), rather than a fabricated
    /// success.
    fn repair_constraints(
        &mut self,
        solution: &HashMap<String, bool>,
        _qubo_matrix: &Array2<f64>,
    ) -> Result<HashMap<String, bool>, String> {
        if self.constraints.is_empty() {
            return Ok(solution.clone());
        }

        let mut repaired = solution.clone();
        let max_passes = self.config.max_local_iterations.min(200).max(1);

        match self.config.repair_strategy {
            RepairStrategy::Greedy => {
                for _ in 0..max_passes {
                    let current = self.total_violation(&repaired);
                    if current <= 1e-9 {
                        break;
                    }

                    let mut best_var: Option<String> = None;
                    let mut best_value = current;
                    for var in self.repairable_variables(&repaired) {
                        let mut candidate = repaired.clone();
                        let cur_val = candidate[&var];
                        candidate.insert(var.clone(), !cur_val);
                        let candidate_violation = self.total_violation(&candidate);
                        if candidate_violation < best_value {
                            best_value = candidate_violation;
                            best_var = Some(var);
                        }
                    }

                    match best_var {
                        Some(var) => {
                            let cur_val = repaired[&var];
                            repaired.insert(var, !cur_val);
                        }
                        None => break,
                    }
                }
            }
            RepairStrategy::Random => {
                for _ in 0..max_passes {
                    let current = self.total_violation(&repaired);
                    if current <= 1e-9 {
                        break;
                    }

                    let vars = self.repairable_variables(&repaired);
                    if vars.is_empty() {
                        break;
                    }

                    let idx = self.rng.random_range(0..vars.len());
                    let var = vars[idx].clone();
                    let mut candidate = repaired.clone();
                    let cur_val = candidate[&var];
                    candidate.insert(var.clone(), !cur_val);

                    // Random repair still requires the move to not worsen
                    // the total violation, otherwise it is not "repair".
                    if self.total_violation(&candidate) <= current {
                        repaired = candidate;
                    }
                }
            }
            RepairStrategy::Weighted => {
                for _ in 0..max_passes {
                    let current = self.weighted_violation(&repaired);
                    if current <= 1e-9 {
                        break;
                    }

                    let mut best_var: Option<String> = None;
                    let mut best_value = current;
                    for var in self.repairable_variables(&repaired) {
                        let mut candidate = repaired.clone();
                        let cur_val = candidate[&var];
                        candidate.insert(var.clone(), !cur_val);
                        let candidate_violation = self.weighted_violation(&candidate);
                        if candidate_violation < best_value {
                            best_value = candidate_violation;
                            best_var = Some(var);
                        }
                    }

                    match best_var {
                        Some(var) => {
                            let cur_val = repaired[&var];
                            repaired.insert(var, !cur_val);
                        }
                        None => break,
                    }
                }
            }
            RepairStrategy::Iterative => {
                // Greedy single-flip repair; when no single flip improves the
                // solution, search a wider (pairwise) neighborhood before
                // giving up and backtracking to the best solution found.
                let mut best = repaired.clone();
                let mut best_violation = self.total_violation(&best);

                for _ in 0..max_passes {
                    if best_violation <= 1e-9 {
                        break;
                    }

                    let mut improved = false;
                    for var in self.repairable_variables(&best) {
                        let mut candidate = best.clone();
                        let cur_val = candidate[&var];
                        candidate.insert(var.clone(), !cur_val);
                        let candidate_violation = self.total_violation(&candidate);
                        if candidate_violation < best_violation {
                            best = candidate;
                            best_violation = candidate_violation;
                            improved = true;
                            break;
                        }
                    }

                    if improved {
                        continue;
                    }

                    // Backtracking step: widen the neighborhood to pairs of
                    // variables before concluding no further repair is
                    // possible.
                    let vars = self.repairable_variables(&best);
                    let mut found_pair = false;
                    'pairs: for i in 0..vars.len() {
                        for j in (i + 1)..vars.len() {
                            let mut candidate = best.clone();
                            let vi = candidate[&vars[i]];
                            let vj = candidate[&vars[j]];
                            candidate.insert(vars[i].clone(), !vi);
                            candidate.insert(vars[j].clone(), !vj);
                            let candidate_violation = self.total_violation(&candidate);
                            if candidate_violation < best_violation {
                                best = candidate;
                                best_violation = candidate_violation;
                                found_pair = true;
                                break 'pairs;
                            }
                        }
                    }

                    if !found_pair {
                        break;
                    }
                }

                repaired = best;
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
    ///
    /// Each iteration actually invokes the supplied `sampler` (e.g. a real
    /// `SASampler`/`GASampler`/hardware-backed `Sampler`) on `qubo_matrix` to
    /// obtain quantum/annealing samples, then refines each returned sample
    /// with classical local search. Variables are named `x0`, `x1`, ...
    /// matching [`Self::compute_energy`]'s convention.
    pub fn iterative_refinement<S: Sampler>(
        &mut self,
        sampler: &S,
        qubo_matrix: &Array2<f64>,
        num_samples: usize,
    ) -> Result<Vec<RefinedSolution>, String> {
        let n = qubo_matrix.nrows();
        let var_map: HashMap<String, usize> = (0..n).map(|i| (format!("x{i}"), i)).collect();

        let mut refined_solutions = Vec::new();
        let mut best_energy = f64::INFINITY;

        for iteration in 0..self.config.max_qc_iterations {
            println!(
                "Quantum-Classical iteration {}/{}",
                iteration + 1,
                self.config.max_qc_iterations
            );

            // Quantum sampling step: actually run the injected sampler on the
            // real QUBO matrix rather than fabricating uniform-random bits.
            let sample_results = sampler
                .run_qubo(&(qubo_matrix.clone(), var_map.clone()), num_samples)
                .map_err(|e| format!("sampler failed during iterative refinement: {e}"))?;

            let samples: Vec<HashMap<String, bool>> = sample_results
                .into_iter()
                .map(|result| result.assignments)
                .collect();

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
    ///
    /// Evaluates each attached [`Constraint`] (see [`Self::set_constraints`])
    /// against the given assignment and reports the ones that are actually
    /// violated. With no constraints attached this honestly returns an empty
    /// list (there is nothing to violate), rather than fabricating success
    /// for a problem it was never told about.
    fn compute_violations(&self, solution: &HashMap<String, bool>) -> Vec<ConstraintViolation> {
        self.constraints
            .iter()
            .filter_map(|c| {
                let magnitude = c.violation(solution);
                if magnitude.abs() > 1e-9 {
                    Some(ConstraintViolation {
                        constraint_id: c.name.clone(),
                        magnitude,
                        variables: c.variables.clone(),
                    })
                } else {
                    None
                }
            })
            .collect()
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
#[allow(clippy::field_reassign_with_default)]
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

    #[test]
    fn test_iterative_refinement_actually_calls_sampler() {
        use crate::sampler::SASampler;

        let config = HybridConfig {
            max_qc_iterations: 2,
            max_local_iterations: 5,
            fixing_criterion: None,
            ..Default::default()
        };
        let mut optimizer = HybridOptimizer::new(config);
        let sampler = SASampler::new(Some(42));

        // Minimizing QUBO: -1 on the diagonal means x_i = 1 is favorable.
        let qubo = Array2::from_shape_fn((3, 3), |(i, j)| if i == j { -1.0 } else { 0.0 });

        let refined = optimizer
            .iterative_refinement(&sampler, &qubo, 4)
            .expect("iterative refinement should succeed");

        // A real annealer run + classical refinement on this trivial QUBO
        // must reach the true optimum (all variables true, energy = -3.0);
        // the old fabricated "random bits" implementation had no such
        // guarantee since it never actually consulted the sampler.
        assert!(!refined.is_empty());
        let best = refined
            .iter()
            .fold(f64::INFINITY, |acc, sol| acc.min(sol.energy));
        assert!(
            (best - (-3.0)).abs() < 1e-9,
            "expected best energy -3.0 from a real sampler run, got {best}"
        );
    }

    #[test]
    fn test_repair_constraints_fixes_real_violation() {
        use crate::optimization::constraints::{ConstraintType, Expression, Variable};

        let config = HybridConfig {
            repair_strategy: RepairStrategy::Greedy,
            ..Default::default()
        };
        let mut optimizer = HybridOptimizer::new(config);

        // Constraint: x0 == 1 (must be true).
        let expr: Expression = Variable::new("x0".to_string()).into();
        optimizer.set_constraints(vec![Constraint {
            name: "x0_must_be_true".to_string(),
            constraint_type: ConstraintType::Equality { target: 1.0 },
            expression: expr,
            variables: vec!["x0".to_string()],
            penalty_weight: None,
            slack_variables: Vec::new(),
        }]);

        let qubo = Array2::<f64>::zeros((2, 2));
        let solution = HashMap::from([("x0".to_string(), false), ("x1".to_string(), false)]);

        // Before the fix, is_feasible was hardcoded true for every solution
        // regardless of actual constraints.
        assert!(!optimizer.compute_violations(&solution).is_empty());

        let refined = optimizer
            .refine_solution(&solution, &qubo)
            .expect("refinement should succeed");

        assert!(refined.assignments["x0"], "repair should fix x0 to true");
        assert!(refined.is_feasible);
        assert!(refined.violations.is_empty());
    }

    #[test]
    fn test_compute_violations_reports_real_violation_magnitude() {
        use crate::optimization::constraints::{ConstraintType, Expression, Variable};

        let mut optimizer = HybridOptimizer::new(HybridConfig::default());
        let expr: Expression = Variable::new("x0".to_string()).into();
        optimizer.set_constraints(vec![Constraint {
            name: "x0_must_be_true".to_string(),
            constraint_type: ConstraintType::Equality { target: 1.0 },
            expression: expr,
            variables: vec!["x0".to_string()],
            penalty_weight: None,
            slack_variables: Vec::new(),
        }]);

        let violated = HashMap::from([("x0".to_string(), false)]);
        let violations = optimizer.compute_violations(&violated);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].magnitude, -1.0);

        let satisfied = HashMap::from([("x0".to_string(), true)]);
        assert!(optimizer.compute_violations(&satisfied).is_empty());
    }
}
