//! Continuous variable annealing for optimization problems
//!
//! This module extends quantum annealing to continuous variables, enabling
//! the solution of optimization problems with real-valued decision variables
//! using discretization and approximation techniques.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::simulator::{AnnealingParams, AnnealingSolution, TemperatureSchedule};

/// Errors that can occur during continuous variable annealing
#[derive(Error, Debug)]
pub enum ContinuousVariableError {
    /// Invalid variable definition
    #[error("Invalid variable: {0}")]
    InvalidVariable(String),

    /// Invalid constraint
    #[error("Invalid constraint: {0}")]
    InvalidConstraint(String),

    /// Discretization error
    #[error("Discretization error: {0}")]
    DiscretizationError(String),

    /// Optimization failed
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    /// Numerical error
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Result type for continuous variable operations
pub type ContinuousVariableResult<T> = Result<T, ContinuousVariableError>;

/// Continuous variable definition
#[derive(Debug, Clone)]
pub struct ContinuousVariable {
    /// Variable name
    pub name: String,

    /// Lower bound
    pub lower_bound: f64,

    /// Upper bound
    pub upper_bound: f64,

    /// Precision (number of discretization bits)
    pub precision_bits: usize,

    /// Variable description
    pub description: Option<String>,
}

impl ContinuousVariable {
    /// Create a new continuous variable
    pub fn new(
        name: String,
        lower_bound: f64,
        upper_bound: f64,
        precision_bits: usize,
    ) -> ContinuousVariableResult<Self> {
        if lower_bound >= upper_bound {
            return Err(ContinuousVariableError::InvalidVariable(format!(
                "Invalid bounds: {} >= {}",
                lower_bound, upper_bound
            )));
        }

        if precision_bits == 0 || precision_bits > 32 {
            return Err(ContinuousVariableError::InvalidVariable(
                "Precision bits must be between 1 and 32".to_string(),
            ));
        }

        Ok(Self {
            name,
            lower_bound,
            upper_bound,
            precision_bits,
            description: None,
        })
    }

    /// Add description to the variable
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Get the number of discrete levels
    pub fn num_levels(&self) -> usize {
        2_usize.pow(self.precision_bits as u32)
    }

    /// Convert binary representation to continuous value
    pub fn binary_to_continuous(&self, binary_value: u32) -> f64 {
        let max_value = (1u32 << self.precision_bits) - 1;
        let normalized = binary_value as f64 / max_value as f64;
        self.lower_bound + normalized * (self.upper_bound - self.lower_bound)
    }

    /// Convert continuous value to binary representation
    pub fn continuous_to_binary(&self, continuous_value: f64) -> u32 {
        let clamped = continuous_value.clamp(self.lower_bound, self.upper_bound);
        let normalized = (clamped - self.lower_bound) / (self.upper_bound - self.lower_bound);
        let max_value = (1u32 << self.precision_bits) - 1;
        (normalized * max_value as f64).round() as u32
    }

    /// Get the resolution (smallest representable difference)
    pub fn resolution(&self) -> f64 {
        (self.upper_bound - self.lower_bound) / (self.num_levels() - 1) as f64
    }
}

/// Objective function for continuous optimization
pub type ObjectiveFunction = Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync>;

/// Constraint function for continuous optimization
pub type ConstraintFunction = Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync>;

/// Constraint specification
pub struct ContinuousConstraint {
    /// Constraint name
    pub name: String,

    /// Constraint function (should return <= 0 for feasible points)
    pub function: ConstraintFunction,

    /// Penalty weight for constraint violations
    pub penalty_weight: f64,

    /// Constraint tolerance
    pub tolerance: f64,
}

impl std::fmt::Debug for ContinuousConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousConstraint")
            .field("name", &self.name)
            .field("function", &"<function>")
            .field("penalty_weight", &self.penalty_weight)
            .field("tolerance", &self.tolerance)
            .finish()
    }
}

impl ContinuousConstraint {
    /// Create a new constraint
    pub fn new(name: String, function: ConstraintFunction, penalty_weight: f64) -> Self {
        Self {
            name,
            function,
            penalty_weight,
            tolerance: 1e-6,
        }
    }

    /// Set constraint tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Continuous optimization problem
pub struct ContinuousOptimizationProblem {
    /// Variables in the problem
    variables: HashMap<String, ContinuousVariable>,

    /// Objective function to minimize
    objective: ObjectiveFunction,

    /// Constraints
    constraints: Vec<ContinuousConstraint>,

    /// Default penalty weight for constraint violations
    default_penalty_weight: f64,
}

impl std::fmt::Debug for ContinuousOptimizationProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousOptimizationProblem")
            .field("variables", &self.variables)
            .field("objective", &"<function>")
            .field("constraints", &self.constraints)
            .field("default_penalty_weight", &self.default_penalty_weight)
            .finish()
    }
}

impl ContinuousOptimizationProblem {
    /// Create a new continuous optimization problem
    pub fn new(objective: ObjectiveFunction) -> Self {
        Self {
            variables: HashMap::new(),
            objective,
            constraints: Vec::new(),
            default_penalty_weight: 100.0,
        }
    }

    /// Add a variable to the problem
    pub fn add_variable(&mut self, variable: ContinuousVariable) -> ContinuousVariableResult<()> {
        if self.variables.contains_key(&variable.name) {
            return Err(ContinuousVariableError::InvalidVariable(format!(
                "Variable '{}' already exists",
                variable.name
            )));
        }

        self.variables.insert(variable.name.clone(), variable);
        Ok(())
    }

    /// Add a constraint to the problem
    pub fn add_constraint(&mut self, constraint: ContinuousConstraint) {
        self.constraints.push(constraint);
    }

    /// Set default penalty weight
    pub fn set_default_penalty_weight(&mut self, weight: f64) {
        self.default_penalty_weight = weight;
    }

    /// Get total number of binary variables needed
    pub fn total_binary_variables(&self) -> usize {
        self.variables.values().map(|v| v.precision_bits).sum()
    }

    /// Create binary variable mapping
    pub fn create_binary_mapping(&self) -> HashMap<String, Vec<usize>> {
        let mut mapping = HashMap::new();
        let mut current_index = 0;

        for (var_name, var) in &self.variables {
            let indices: Vec<usize> = (current_index..current_index + var.precision_bits).collect();
            mapping.insert(var_name.clone(), indices);
            current_index += var.precision_bits;
        }

        mapping
    }

    /// Convert binary solution to continuous values
    pub fn binary_to_continuous_solution(
        &self,
        binary_solution: &[i8],
    ) -> ContinuousVariableResult<HashMap<String, f64>> {
        let binary_mapping = self.create_binary_mapping();
        let mut continuous_solution = HashMap::new();

        for (var_name, var) in &self.variables {
            let indices = &binary_mapping[var_name];

            if indices.iter().any(|&i| i >= binary_solution.len()) {
                return Err(ContinuousVariableError::DiscretizationError(format!(
                    "Binary solution too short for variable '{}'",
                    var_name
                )));
            }

            // Convert binary bits to integer value
            let mut binary_value = 0u32;
            for (bit_idx, &global_idx) in indices.iter().enumerate() {
                if binary_solution[global_idx] > 0 {
                    binary_value |= 1 << (var.precision_bits - 1 - bit_idx);
                }
            }

            // Convert to continuous value
            let continuous_value = var.binary_to_continuous(binary_value);
            continuous_solution.insert(var_name.clone(), continuous_value);
        }

        Ok(continuous_solution)
    }

    /// Evaluate objective function with penalty for constraint violations
    pub fn evaluate_penalized_objective(&self, continuous_solution: &HashMap<String, f64>) -> f64 {
        let mut objective_value = (self.objective)(continuous_solution);

        // Add constraint penalties
        for constraint in &self.constraints {
            let constraint_value = (constraint.function)(continuous_solution);
            if constraint_value > constraint.tolerance {
                objective_value += constraint.penalty_weight * constraint_value.powi(2);
            }
        }

        objective_value
    }
}

/// Configuration for continuous variable annealing
#[derive(Debug, Clone)]
pub struct ContinuousAnnealingConfig {
    /// Base annealing parameters
    pub annealing_params: AnnealingParams,

    /// Adaptive discretization
    pub adaptive_discretization: bool,

    /// Maximum refinement iterations
    pub max_refinement_iterations: usize,

    /// Convergence tolerance for refinement
    pub refinement_tolerance: f64,

    /// Enable local search post-processing
    pub local_search: bool,

    /// Local search iterations
    pub local_search_iterations: usize,

    /// Local search step size (as fraction of variable range)
    pub local_search_step_size: f64,
}

impl Default for ContinuousAnnealingConfig {
    fn default() -> Self {
        Self {
            annealing_params: AnnealingParams::default(),
            adaptive_discretization: true,
            max_refinement_iterations: 3,
            refinement_tolerance: 1e-4,
            local_search: true,
            local_search_iterations: 100,
            local_search_step_size: 0.01,
        }
    }
}

/// Solution for continuous optimization problem
#[derive(Debug, Clone)]
pub struct ContinuousSolution {
    /// Variable values
    pub variable_values: HashMap<String, f64>,

    /// Objective value
    pub objective_value: f64,

    /// Constraint violations
    pub constraint_violations: Vec<(String, f64)>,

    /// Binary solution used
    pub binary_solution: Vec<i8>,

    /// Solution statistics
    pub stats: ContinuousOptimizationStats,
}

/// Statistics for continuous optimization
#[derive(Debug, Clone)]
pub struct ContinuousOptimizationStats {
    /// Total runtime
    pub total_runtime: Duration,

    /// Discretization time
    pub discretization_time: Duration,

    /// Annealing time
    pub annealing_time: Duration,

    /// Local search time
    pub local_search_time: Duration,

    /// Number of refinement iterations
    pub refinement_iterations: usize,

    /// Final discretization resolution
    pub final_resolution: HashMap<String, f64>,

    /// Convergence achieved
    pub converged: bool,
}

/// Continuous variable annealing solver
pub struct ContinuousVariableAnnealer {
    /// Configuration
    config: ContinuousAnnealingConfig,

    /// Random number generator
    rng: ChaCha8Rng,
}

impl ContinuousVariableAnnealer {
    /// Create a new continuous variable annealer
    pub fn new(config: ContinuousAnnealingConfig) -> Self {
        let rng = match config.annealing_params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };

        Self { config, rng }
    }

    /// Solve a continuous optimization problem
    pub fn solve(
        &mut self,
        problem: &ContinuousOptimizationProblem,
    ) -> ContinuousVariableResult<ContinuousSolution> {
        let total_start = Instant::now();

        // Initial discretization
        let discretize_start = Instant::now();
        let mut current_problem = self.create_discretized_problem(problem)?;
        let discretization_time = discretize_start.elapsed();

        let mut best_solution = None;
        let mut best_objective = f64::INFINITY;
        let mut refinement_iterations = 0;

        // Iterative refinement loop
        for iteration in 0..self.config.max_refinement_iterations {
            // Solve discretized problem
            let anneal_start = Instant::now();
            let binary_solution = self.solve_discretized_problem(&current_problem)?;
            let annealing_time = anneal_start.elapsed();

            // Convert to continuous solution
            let continuous_values = problem.binary_to_continuous_solution(&binary_solution)?;
            let objective_value = problem.evaluate_penalized_objective(&continuous_values);

            // Check for improvement
            let improvement = if best_objective.is_finite() {
                best_objective - objective_value
            } else {
                f64::INFINITY
            };

            if objective_value < best_objective {
                best_objective = objective_value;
                best_solution = Some((binary_solution, continuous_values.clone(), annealing_time));
            }

            refinement_iterations += 1;

            // Check convergence
            if improvement < self.config.refinement_tolerance && iteration > 0 {
                break;
            }

            // Adaptive refinement
            if self.config.adaptive_discretization
                && iteration < self.config.max_refinement_iterations - 1
            {
                current_problem = self.refine_discretization(problem, &continuous_values)?;
            }
        }

        let (final_binary, mut final_continuous, annealing_time) =
            best_solution.ok_or_else(|| {
                ContinuousVariableError::OptimizationFailed("No solution found".to_string())
            })?;

        // Local search post-processing
        let local_search_start = Instant::now();
        let local_search_time = if self.config.local_search {
            self.local_search(problem, &mut final_continuous)?;
            local_search_start.elapsed()
        } else {
            Duration::from_secs(0)
        };

        // Calculate constraint violations
        let constraint_violations =
            self.calculate_constraint_violations(problem, &final_continuous);

        // Calculate final objective (without penalties)
        let final_objective = (problem.objective)(&final_continuous);

        // Calculate final resolutions
        let final_resolution = problem
            .variables
            .iter()
            .map(|(name, var)| (name.clone(), var.resolution()))
            .collect();

        let total_runtime = total_start.elapsed();

        let stats = ContinuousOptimizationStats {
            total_runtime,
            discretization_time,
            annealing_time,
            local_search_time,
            refinement_iterations,
            final_resolution,
            converged: refinement_iterations < self.config.max_refinement_iterations,
        };

        Ok(ContinuousSolution {
            variable_values: final_continuous,
            objective_value: final_objective,
            constraint_violations,
            binary_solution: final_binary,
            stats,
        })
    }

    /// Create discretized QUBO problem
    fn create_discretized_problem(
        &self,
        _problem: &ContinuousOptimizationProblem,
    ) -> ContinuousVariableResult<DiscretizedProblem> {
        // This would create a QUBO representation of the continuous problem
        // For now, return a placeholder
        Ok(DiscretizedProblem {
            num_variables: 0,
            q_matrix: Vec::new(),
        })
    }

    /// Solve discretized problem using annealing
    fn solve_discretized_problem(
        &mut self,
        _problem: &DiscretizedProblem,
    ) -> ContinuousVariableResult<Vec<i8>> {
        // This would solve the QUBO using quantum annealing
        // For now, return a random solution
        let num_vars = 16; // Placeholder
        let solution: Vec<i8> = (0..num_vars)
            .map(|_| if self.rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();

        Ok(solution)
    }

    /// Refine discretization around current solution
    fn refine_discretization(
        &self,
        _problem: &ContinuousOptimizationProblem,
        _current_solution: &HashMap<String, f64>,
    ) -> ContinuousVariableResult<DiscretizedProblem> {
        // This would create a refined discretization focused on the current solution region
        Ok(DiscretizedProblem {
            num_variables: 0,
            q_matrix: Vec::new(),
        })
    }

    /// Perform local search to improve solution
    fn local_search(
        &mut self,
        problem: &ContinuousOptimizationProblem,
        solution: &mut HashMap<String, f64>,
    ) -> ContinuousVariableResult<()> {
        let mut current_objective = problem.evaluate_penalized_objective(solution);

        for _ in 0..self.config.local_search_iterations {
            let mut improved = false;

            // Try small perturbations for each variable
            for (var_name, var) in &problem.variables {
                let current_value = solution[var_name];
                let step_size =
                    (var.upper_bound - var.lower_bound) * self.config.local_search_step_size;

                // Try both directions
                for direction in [-1.0, 1.0] {
                    let new_value = (current_value + direction * step_size)
                        .clamp(var.lower_bound, var.upper_bound);

                    // Temporarily update solution
                    solution.insert(var_name.clone(), new_value);
                    let new_objective = problem.evaluate_penalized_objective(solution);

                    if new_objective < current_objective {
                        current_objective = new_objective;
                        improved = true;
                        break; // Keep this improvement
                    } else {
                        // Revert change
                        solution.insert(var_name.clone(), current_value);
                    }
                }
            }

            // If no improvement found, stop early
            if !improved {
                break;
            }
        }

        Ok(())
    }

    /// Calculate constraint violations
    fn calculate_constraint_violations(
        &self,
        problem: &ContinuousOptimizationProblem,
        solution: &HashMap<String, f64>,
    ) -> Vec<(String, f64)> {
        problem
            .constraints
            .iter()
            .map(|constraint| {
                let violation = (constraint.function)(solution);
                (constraint.name.clone(), violation.max(0.0))
            })
            .collect()
    }
}

/// Placeholder for discretized problem representation
#[derive(Debug)]
struct DiscretizedProblem {
    num_variables: usize,
    q_matrix: Vec<Vec<f64>>,
}

/// Helper functions for common continuous optimization problems

/// Create a quadratic programming problem
pub fn create_quadratic_problem(
    linear_coeffs: &[f64],
    quadratic_matrix: &[Vec<f64>],
    bounds: &[(f64, f64)],
    precision_bits: usize,
) -> ContinuousVariableResult<ContinuousOptimizationProblem> {
    // Objective: 0.5 * x^T * Q * x + c^T * x
    let linear_coeffs = linear_coeffs.to_vec();
    let quadratic_matrix = quadratic_matrix.to_vec();

    let objective: ObjectiveFunction = Box::new(move |vars: &HashMap<String, f64>| {
        let n = linear_coeffs.len();
        let x: Vec<f64> = (0..n).map(|i| vars[&format!("x{}", i)]).collect();

        // Linear term
        let linear_term: f64 = linear_coeffs
            .iter()
            .zip(x.iter())
            .map(|(c, xi)| c * xi)
            .sum();

        // Quadratic term
        let mut quadratic_term = 0.0;
        for i in 0..n {
            for j in 0..n {
                quadratic_term += 0.5 * quadratic_matrix[i][j] * x[i] * x[j];
            }
        }

        linear_term + quadratic_term
    });

    let mut problem = ContinuousOptimizationProblem::new(objective);

    // Add variables
    for (i, &(lower, upper)) in bounds.iter().enumerate() {
        let var = ContinuousVariable::new(format!("x{}", i), lower, upper, precision_bits)?;
        problem.add_variable(var)?;
    }

    Ok(problem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_variable_creation() {
        let var = ContinuousVariable::new("x".to_string(), 0.0, 10.0, 8).unwrap();
        assert_eq!(var.name, "x");
        assert_eq!(var.lower_bound, 0.0);
        assert_eq!(var.upper_bound, 10.0);
        assert_eq!(var.precision_bits, 8);
        assert_eq!(var.num_levels(), 256);
    }

    #[test]
    fn test_binary_continuous_conversion() {
        let var = ContinuousVariable::new("x".to_string(), 0.0, 10.0, 4).unwrap();

        // Test conversion: 0 -> 0.0, 15 -> 10.0
        assert_eq!(var.binary_to_continuous(0), 0.0);
        assert!((var.binary_to_continuous(15) - 10.0).abs() < 1e-10);

        // Test reverse conversion
        assert_eq!(var.continuous_to_binary(0.0), 0);
        assert_eq!(var.continuous_to_binary(10.0), 15);

        // Test round-trip
        let continuous_val = 3.7;
        let binary_val = var.continuous_to_binary(continuous_val);
        let recovered_val = var.binary_to_continuous(binary_val);
        assert!((recovered_val - continuous_val).abs() <= var.resolution());
    }

    #[test]
    fn test_quadratic_problem_creation() {
        let linear_coeffs = vec![1.0, -2.0];
        let quadratic_matrix = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let bounds = vec![(0.0, 5.0), (-3.0, 3.0)];

        let problem =
            create_quadratic_problem(&linear_coeffs, &quadratic_matrix, &bounds, 6).unwrap();
        assert_eq!(problem.variables.len(), 2);
        assert!(problem.variables.contains_key("x0"));
        assert!(problem.variables.contains_key("x1"));
    }

    #[test]
    fn test_constraint_evaluation() {
        let constraint_fn: ConstraintFunction = Box::new(|vars| {
            vars["x"] + vars["y"] - 5.0 // x + y <= 5
        });

        let constraint =
            ContinuousConstraint::new("sum_constraint".to_string(), constraint_fn, 10.0);

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        vars.insert("y".to_string(), 2.0);

        let violation = (constraint.function)(&vars);
        assert_eq!(violation, -1.0); // 2 + 2 - 5 = -1 (feasible)

        vars.insert("y".to_string(), 4.0);
        let violation = (constraint.function)(&vars);
        assert_eq!(violation, 1.0); // 2 + 4 - 5 = 1 (infeasible)
    }
}
