//! Multi-objective optimization framework for quantum annealing
//!
//! This module provides tools for solving multi-objective optimization problems
//! where multiple conflicting objectives need to be optimized simultaneously,
//! generating Pareto-optimal solutions.

use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};
use crate::simulator::{AnnealingParams, AnnealingSolution, QuantumAnnealingSimulator};

/// Errors that can occur during multi-objective optimization
#[derive(Error, Debug)]
pub enum MultiObjectiveError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Invalid objective function
    #[error("Invalid objective: {0}")]
    InvalidObjective(String),

    /// Optimization failed
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    /// Pareto analysis error
    #[error("Pareto analysis error: {0}")]
    ParetoAnalysisError(String),

    /// Scalarization error
    #[error("Scalarization error: {0}")]
    ScalarizationError(String),
}

/// Result type for multi-objective operations
pub type MultiObjectiveResult<T> = Result<T, MultiObjectiveError>;

/// Objective function that evaluates multiple criteria
pub type MultiObjectiveFunction = Box<dyn Fn(&[i8]) -> Vec<f64> + Send + Sync>;

/// Single solution in multi-objective space
#[derive(Debug, Clone)]
pub struct MultiObjectiveSolution {
    /// Variable assignment (spin configuration)
    pub variables: Vec<i8>,

    /// Objective values for each criterion
    pub objectives: Vec<f64>,

    /// Rank in Pareto front (0 = non-dominated)
    pub pareto_rank: usize,

    /// Crowding distance for diversity
    pub crowding_distance: f64,

    /// Solution metadata
    pub metadata: HashMap<String, String>,
}

impl MultiObjectiveSolution {
    /// Create a new multi-objective solution
    pub fn new(variables: Vec<i8>, objectives: Vec<f64>) -> Self {
        Self {
            variables,
            objectives,
            pareto_rank: 0,
            crowding_distance: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Check if this solution dominates another
    pub fn dominates(&self, other: &Self) -> bool {
        if self.objectives.len() != other.objectives.len() {
            return false;
        }

        let mut at_least_one_better = false;
        for (self_obj, other_obj) in self.objectives.iter().zip(other.objectives.iter()) {
            if self_obj > other_obj {
                return false; // Self is worse in this objective
            }
            if self_obj < other_obj {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Check if this solution is non-dominated by another
    pub fn is_non_dominated_by(&self, other: &Self) -> bool {
        !other.dominates(self)
    }
}

/// Scalarization method for converting multi-objective to single-objective
#[derive(Debug, Clone)]
pub enum ScalarizationMethod {
    /// Weighted sum: min Σ(w_i * f_i)
    WeightedSum { weights: Vec<f64> },

    /// Weighted Chebyshev: min max_i(w_i * |f_i - z_i*|)
    WeightedChebyshev {
        weights: Vec<f64>,
        reference_point: Vec<f64>,
    },

    /// Augmented Chebyshev with penalty term
    AugmentedChebyshev {
        weights: Vec<f64>,
        reference_point: Vec<f64>,
        rho: f64,
    },

    /// Achievement scalarizing function
    Achievement {
        weights: Vec<f64>,
        reference_point: Vec<f64>,
    },

    /// ε-constraint: optimize one objective subject to constraints on others
    EpsilonConstraint {
        primary_objective: usize,
        constraints: Vec<f64>,
    },
}

impl ScalarizationMethod {
    /// Apply scalarization to convert multiple objectives to single value
    pub fn scalarize(&self, objectives: &[f64]) -> MultiObjectiveResult<f64> {
        match self {
            ScalarizationMethod::WeightedSum { weights } => {
                if weights.len() != objectives.len() {
                    return Err(MultiObjectiveError::ScalarizationError(
                        "Weights and objectives dimension mismatch".to_string(),
                    ));
                }

                let weighted_sum = weights
                    .iter()
                    .zip(objectives.iter())
                    .map(|(w, obj)| w * obj)
                    .sum();

                Ok(weighted_sum)
            }

            ScalarizationMethod::WeightedChebyshev {
                weights,
                reference_point,
            } => {
                if weights.len() != objectives.len() || reference_point.len() != objectives.len() {
                    return Err(MultiObjectiveError::ScalarizationError(
                        "Dimension mismatch in Chebyshev scalarization".to_string(),
                    ));
                }

                let max_weighted_deviation = weights
                    .iter()
                    .zip(objectives.iter())
                    .zip(reference_point.iter())
                    .map(|((w, obj), ref_val)| w * (obj - ref_val).abs())
                    .fold(0.0, f64::max);

                Ok(max_weighted_deviation)
            }

            ScalarizationMethod::AugmentedChebyshev {
                weights,
                reference_point,
                rho,
            } => {
                let chebyshev_value =
                    self.scalarize_chebyshev(objectives, weights, reference_point)?;

                let augmentation_term = rho
                    * weights
                        .iter()
                        .zip(objectives.iter())
                        .zip(reference_point.iter())
                        .map(|((w, obj), ref_val)| w * (obj - ref_val))
                        .sum::<f64>();

                Ok(chebyshev_value + augmentation_term)
            }

            ScalarizationMethod::Achievement {
                weights,
                reference_point,
            } => {
                let achievement_value = weights
                    .iter()
                    .zip(objectives.iter())
                    .zip(reference_point.iter())
                    .map(|((w, obj), ref_val)| (obj - ref_val) / w.max(1e-8))
                    .fold(f64::NEG_INFINITY, f64::max);

                Ok(achievement_value)
            }

            ScalarizationMethod::EpsilonConstraint {
                primary_objective,
                constraints,
            } => {
                if *primary_objective >= objectives.len() {
                    return Err(MultiObjectiveError::ScalarizationError(
                        "Primary objective index out of bounds".to_string(),
                    ));
                }

                // Check constraint violations
                let mut penalty = 0.0;
                for (i, &constraint_bound) in constraints.iter().enumerate() {
                    if i != *primary_objective && i < objectives.len() {
                        if objectives[i] > constraint_bound {
                            penalty += 1000.0 * (objectives[i] - constraint_bound);
                        }
                    }
                }

                Ok(objectives[*primary_objective] + penalty)
            }
        }
    }

    /// Helper method for Chebyshev scalarization
    fn scalarize_chebyshev(
        &self,
        objectives: &[f64],
        weights: &[f64],
        reference_point: &[f64],
    ) -> MultiObjectiveResult<f64> {
        if weights.len() != objectives.len() || reference_point.len() != objectives.len() {
            return Err(MultiObjectiveError::ScalarizationError(
                "Dimension mismatch in Chebyshev scalarization".to_string(),
            ));
        }

        let max_weighted_deviation = weights
            .iter()
            .zip(objectives.iter())
            .zip(reference_point.iter())
            .map(|((w, obj), ref_val)| w * (obj - ref_val).abs())
            .fold(0.0, f64::max);

        Ok(max_weighted_deviation)
    }
}

/// Configuration for multi-objective optimization
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Base annealing parameters
    pub annealing_params: AnnealingParams,

    /// Scalarization method
    pub scalarization: ScalarizationMethod,

    /// Number of Pareto front approximation runs
    pub num_pareto_runs: usize,

    /// Population size for diversity
    pub population_size: usize,

    /// Enable Pareto ranking and crowding distance
    pub enable_pareto_analysis: bool,

    /// Maximum number of non-dominated solutions to keep
    pub max_pareto_solutions: usize,

    /// Random seed
    pub seed: Option<u64>,

    /// Timeout for optimization
    pub timeout: Option<Duration>,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            annealing_params: AnnealingParams::default(),
            scalarization: ScalarizationMethod::WeightedSum {
                weights: vec![1.0, 1.0],
            },
            num_pareto_runs: 20,
            population_size: 100,
            enable_pareto_analysis: true,
            max_pareto_solutions: 50,
            seed: None,
            timeout: Some(Duration::from_secs(600)),
        }
    }
}

/// Results from multi-objective optimization
#[derive(Debug, Clone)]
pub struct MultiObjectiveResults {
    /// All solutions found
    pub all_solutions: Vec<MultiObjectiveSolution>,

    /// Pareto-optimal solutions (rank 0)
    pub pareto_front: Vec<MultiObjectiveSolution>,

    /// Statistics about the optimization
    pub stats: MultiObjectiveStats,

    /// Objective bounds for normalization
    pub objective_bounds: Vec<(f64, f64)>, // (min, max) for each objective
}

/// Statistics for multi-objective optimization
#[derive(Debug, Clone)]
pub struct MultiObjectiveStats {
    /// Total runtime
    pub total_runtime: Duration,

    /// Number of unique solutions found
    pub unique_solutions: usize,

    /// Size of final Pareto front
    pub pareto_front_size: usize,

    /// Average crowding distance
    pub average_crowding_distance: f64,

    /// Hypervolume indicator (if calculated)
    pub hypervolume: Option<f64>,

    /// Convergence metrics
    pub convergence_metrics: HashMap<String, f64>,
}

/// Multi-objective optimizer using quantum annealing
pub struct MultiObjectiveOptimizer {
    /// Configuration
    config: MultiObjectiveConfig,

    /// Random number generator
    rng: ChaCha8Rng,
}

impl MultiObjectiveOptimizer {
    /// Create a new multi-objective optimizer
    pub fn new(config: MultiObjectiveConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().gen()),
        };

        Self { config, rng }
    }

    /// Solve a multi-objective optimization problem
    pub fn solve(
        &mut self,
        model: &IsingModel,
        objective_function: MultiObjectiveFunction,
        num_objectives: usize,
    ) -> MultiObjectiveResult<MultiObjectiveResults> {
        let start_time = Instant::now();
        let mut all_solutions = Vec::new();

        // Generate multiple scalarizations for Pareto front approximation
        let scalarizations = self.generate_scalarizations(num_objectives)?;

        // Solve for each scalarization
        for (run_idx, scalarization) in scalarizations.into_iter().enumerate() {
            // Check timeout
            if let Some(timeout) = self.config.timeout {
                if start_time.elapsed() > timeout {
                    break;
                }
            }

            println!(
                "Running scalarization {}/{}",
                run_idx + 1,
                self.config.num_pareto_runs
            );

            // Solve single-objective problem
            let solution =
                self.solve_scalarized_problem(model, &objective_function, &scalarization)?;

            // Evaluate true objectives
            let objectives = objective_function(&solution.best_spins);
            let mo_solution = MultiObjectiveSolution::new(solution.best_spins, objectives);

            all_solutions.push(mo_solution);
        }

        // Remove duplicate solutions
        all_solutions = self.remove_duplicates(all_solutions);

        // Perform Pareto analysis if enabled
        if self.config.enable_pareto_analysis {
            self.perform_pareto_analysis(&mut all_solutions)?;
        }

        // Extract Pareto front
        let pareto_front: Vec<MultiObjectiveSolution> = all_solutions
            .iter()
            .filter(|sol| sol.pareto_rank == 0)
            .cloned()
            .collect();

        // Calculate objective bounds
        let objective_bounds = self.calculate_objective_bounds(&all_solutions, num_objectives);

        // Calculate statistics
        let stats = self.calculate_statistics(&all_solutions, &pareto_front, start_time.elapsed());

        Ok(MultiObjectiveResults {
            all_solutions,
            pareto_front,
            stats,
            objective_bounds,
        })
    }

    /// Generate different scalarizations for Pareto front approximation
    fn generate_scalarizations(
        &mut self,
        num_objectives: usize,
    ) -> MultiObjectiveResult<Vec<ScalarizationMethod>> {
        let mut scalarizations = Vec::new();
        let num_runs = self.config.num_pareto_runs;

        match self.config.scalarization.clone() {
            ScalarizationMethod::WeightedSum { .. } => {
                // Generate uniformly distributed weight vectors
                for i in 0..num_runs {
                    let weights = self.generate_weight_vector(num_objectives, i, num_runs);
                    scalarizations.push(ScalarizationMethod::WeightedSum { weights });
                }
            }

            ScalarizationMethod::WeightedChebyshev {
                reference_point, ..
            } => {
                // Generate different weight vectors with same reference point
                for i in 0..num_runs {
                    let weights = self.generate_weight_vector(num_objectives, i, num_runs);
                    scalarizations.push(ScalarizationMethod::WeightedChebyshev {
                        weights,
                        reference_point: reference_point.clone(),
                    });
                }
            }

            ScalarizationMethod::EpsilonConstraint {
                primary_objective, ..
            } => {
                // Generate different constraint bounds
                for i in 0..num_runs {
                    let constraints = self.generate_constraint_bounds(num_objectives, i);
                    scalarizations.push(ScalarizationMethod::EpsilonConstraint {
                        primary_objective,
                        constraints,
                    });
                }
            }

            method => {
                // For other methods, use the original configuration multiple times
                for _ in 0..num_runs {
                    scalarizations.push(method.clone());
                }
            }
        }

        Ok(scalarizations)
    }

    /// Generate weight vector for run i
    fn generate_weight_vector(
        &mut self,
        num_objectives: usize,
        run_index: usize,
        num_runs: usize,
    ) -> Vec<f64> {
        if num_objectives == 2 {
            // For 2 objectives, use uniform distribution
            let alpha = if num_runs > 1 {
                run_index as f64 / (num_runs - 1) as f64
            } else {
                0.5
            };
            vec![alpha, 1.0 - alpha]
        } else {
            // For more objectives, use random weights
            let mut weights: Vec<f64> = (0..num_objectives)
                .map(|_| self.rng.gen_range(0.1..1.0))
                .collect();

            // Normalize weights
            let sum: f64 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= sum;
            }

            weights
        }
    }

    /// Generate constraint bounds for ε-constraint method
    fn generate_constraint_bounds(&mut self, num_objectives: usize, _run_index: usize) -> Vec<f64> {
        // Generate random constraint bounds
        (0..num_objectives)
            .map(|_| self.rng.gen_range(-10.0..10.0))
            .collect()
    }

    /// Solve a scalarized single-objective problem
    fn solve_scalarized_problem(
        &mut self,
        model: &IsingModel,
        objective_function: &MultiObjectiveFunction,
        scalarization: &ScalarizationMethod,
    ) -> MultiObjectiveResult<AnnealingSolution> {
        // Create a new Ising model that incorporates the scalarized objective
        let mut scalarized_model = model.clone();

        // For demonstration, we'll use the original model and evaluate objectives separately
        // In practice, you might want to modify the model to encode the scalarized objective

        let mut simulator = QuantumAnnealingSimulator::new(self.config.annealing_params.clone())
            .map_err(|e| MultiObjectiveError::OptimizationFailed(e.to_string()))?;

        let solution = simulator
            .solve(&scalarized_model)
            .map_err(|e| MultiObjectiveError::OptimizationFailed(e.to_string()))?;

        Ok(solution)
    }

    /// Remove duplicate solutions
    fn remove_duplicates(
        &self,
        mut solutions: Vec<MultiObjectiveSolution>,
    ) -> Vec<MultiObjectiveSolution> {
        solutions.sort_by(|a, b| {
            a.variables
                .partial_cmp(&b.variables)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        solutions.dedup_by(|a, b| a.variables == b.variables);
        solutions
    }

    /// Perform Pareto ranking and crowding distance calculation
    fn perform_pareto_analysis(
        &self,
        solutions: &mut [MultiObjectiveSolution],
    ) -> MultiObjectiveResult<()> {
        // Fast non-dominated sorting
        self.fast_non_dominated_sort(solutions)?;

        // Calculate crowding distances
        self.calculate_crowding_distances(solutions)?;

        Ok(())
    }

    /// Fast non-dominated sorting algorithm
    fn fast_non_dominated_sort(
        &self,
        solutions: &mut [MultiObjectiveSolution],
    ) -> MultiObjectiveResult<()> {
        let n = solutions.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];
        let mut fronts = Vec::new();
        let mut current_front = Vec::new();

        // For each solution, find which solutions it dominates and how many dominate it
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if solutions[i].dominates(&solutions[j]) {
                        dominated_solutions[i].push(j);
                    } else if solutions[j].dominates(&solutions[i]) {
                        domination_count[i] += 1;
                    }
                }
            }

            if domination_count[i] == 0 {
                solutions[i].pareto_rank = 0;
                current_front.push(i);
            }
        }

        fronts.push(current_front.clone());

        // Generate subsequent fronts
        let mut front_index = 0;
        while !fronts[front_index].is_empty() {
            let mut next_front = Vec::new();

            for &p in &fronts[front_index] {
                for &q in &dominated_solutions[p] {
                    domination_count[q] -= 1;
                    if domination_count[q] == 0 {
                        solutions[q].pareto_rank = front_index + 1;
                        next_front.push(q);
                    }
                }
            }

            front_index += 1;
            fronts.push(next_front);
        }

        Ok(())
    }

    /// Calculate crowding distances for diversity preservation
    fn calculate_crowding_distances(
        &self,
        solutions: &mut [MultiObjectiveSolution],
    ) -> MultiObjectiveResult<()> {
        if solutions.is_empty() {
            return Ok(());
        }

        let num_objectives = solutions[0].objectives.len();

        // Initialize crowding distances
        for solution in solutions.iter_mut() {
            solution.crowding_distance = 0.0;
        }

        // Calculate crowding distance for each objective
        for obj_idx in 0..num_objectives {
            // Sort by objective value
            let mut indices: Vec<usize> = (0..solutions.len()).collect();
            indices.sort_by(|&a, &b| {
                solutions[a].objectives[obj_idx]
                    .partial_cmp(&solutions[b].objectives[obj_idx])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Set boundary solutions to infinite distance
            if !indices.is_empty() {
                solutions[indices[0]].crowding_distance = f64::INFINITY;
                if indices.len() > 1 {
                    solutions[indices[indices.len() - 1]].crowding_distance = f64::INFINITY;
                }
            }

            // Calculate range
            if indices.len() > 2 {
                let min_obj = solutions[indices[0]].objectives[obj_idx];
                let max_obj = solutions[indices[indices.len() - 1]].objectives[obj_idx];
                let range = max_obj - min_obj;

                if range > 0.0 {
                    // Calculate crowding distance for intermediate solutions
                    for i in 1..indices.len() - 1 {
                        let distance = (solutions[indices[i + 1]].objectives[obj_idx]
                            - solutions[indices[i - 1]].objectives[obj_idx])
                            / range;
                        solutions[indices[i]].crowding_distance += distance;
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate objective bounds
    fn calculate_objective_bounds(
        &self,
        solutions: &[MultiObjectiveSolution],
        num_objectives: usize,
    ) -> Vec<(f64, f64)> {
        let mut bounds = vec![(f64::INFINITY, f64::NEG_INFINITY); num_objectives];

        for solution in solutions {
            for (obj_idx, &obj_value) in solution.objectives.iter().enumerate() {
                if obj_idx < bounds.len() {
                    bounds[obj_idx].0 = bounds[obj_idx].0.min(obj_value);
                    bounds[obj_idx].1 = bounds[obj_idx].1.max(obj_value);
                }
            }
        }

        bounds
    }

    /// Calculate optimization statistics
    fn calculate_statistics(
        &self,
        all_solutions: &[MultiObjectiveSolution],
        pareto_front: &[MultiObjectiveSolution],
        runtime: Duration,
    ) -> MultiObjectiveStats {
        let unique_solutions = all_solutions.len();
        let pareto_front_size = pareto_front.len();

        let average_crowding_distance = if pareto_front.is_empty() {
            0.0
        } else {
            pareto_front
                .iter()
                .map(|sol| sol.crowding_distance)
                .filter(|&d| d.is_finite())
                .sum::<f64>()
                / pareto_front.len() as f64
        };

        let mut convergence_metrics = HashMap::new();
        convergence_metrics.insert(
            "pareto_ratio".to_string(),
            pareto_front_size as f64 / unique_solutions.max(1) as f64,
        );

        MultiObjectiveStats {
            total_runtime: runtime,
            unique_solutions,
            pareto_front_size,
            average_crowding_distance,
            hypervolume: None, // Would need reference point to calculate
            convergence_metrics,
        }
    }
}

/// Quality metrics for multi-objective optimization results
pub struct QualityMetrics;

impl QualityMetrics {
    /// Calculate hypervolume indicator
    pub fn hypervolume(
        solutions: &[MultiObjectiveSolution],
        reference_point: &[f64],
    ) -> MultiObjectiveResult<f64> {
        if solutions.is_empty() {
            return Ok(0.0);
        }

        let num_objectives = solutions[0].objectives.len();
        if reference_point.len() != num_objectives {
            return Err(MultiObjectiveError::ParetoAnalysisError(
                "Reference point dimension mismatch".to_string(),
            ));
        }

        // Simple hypervolume calculation for 2D case
        if num_objectives == 2 {
            let mut sorted_solutions = solutions.to_vec();
            sorted_solutions.sort_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap());

            let mut volume = 0.0;
            let mut prev_y = 0.0;

            for solution in &sorted_solutions {
                if solution.objectives[0] < reference_point[0]
                    && solution.objectives[1] < reference_point[1]
                {
                    let width = reference_point[0] - solution.objectives[0];
                    let height = solution.objectives[1] - prev_y;
                    if height > 0.0 {
                        volume += width * height;
                    }
                    prev_y = solution.objectives[1];
                }
            }

            Ok(volume)
        } else {
            // For higher dimensions, would need more sophisticated algorithm
            Ok(0.0)
        }
    }

    /// Calculate spacing metric (diversity measure)
    pub fn spacing(solutions: &[MultiObjectiveSolution]) -> f64 {
        if solutions.len() < 2 {
            return 0.0;
        }

        let mut distances = Vec::new();

        for (i, sol1) in solutions.iter().enumerate() {
            let mut min_distance = f64::INFINITY;

            for (j, sol2) in solutions.iter().enumerate() {
                if i != j {
                    let distance: f64 = sol1
                        .objectives
                        .iter()
                        .zip(sol2.objectives.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();

                    min_distance = min_distance.min(distance);
                }
            }

            distances.push(min_distance);
        }

        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance = distances
            .iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f64>()
            / distances.len() as f64;

        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution_dominance() {
        let sol1 = MultiObjectiveSolution::new(vec![1, -1], vec![1.0, 2.0]);
        let sol2 = MultiObjectiveSolution::new(vec![-1, 1], vec![2.0, 1.0]);
        let sol3 = MultiObjectiveSolution::new(vec![1, 1], vec![0.5, 1.5]);

        // sol3 dominates sol1 (both objectives are better: 0.5 < 1.0, 1.5 < 2.0)
        assert!(sol3.dominates(&sol1));
        // sol3 does not dominate sol2 (trade-off: 0.5 < 2.0 but 1.5 > 1.0)
        assert!(!sol3.dominates(&sol2));

        // sol1 and sol2 do not dominate each other (trade-off)
        assert!(!sol1.dominates(&sol2));
        assert!(!sol2.dominates(&sol1));
    }

    #[test]
    fn test_weighted_sum_scalarization() {
        let weights = vec![0.3, 0.7];
        let scalarization = ScalarizationMethod::WeightedSum { weights };

        let objectives = vec![2.0, 4.0];
        let result = scalarization.scalarize(&objectives).unwrap();

        assert_eq!(result, 0.3 * 2.0 + 0.7 * 4.0);
    }

    #[test]
    fn test_chebyshev_scalarization() {
        let weights = vec![1.0, 1.0];
        let reference_point = vec![0.0, 0.0];
        let scalarization = ScalarizationMethod::WeightedChebyshev {
            weights,
            reference_point,
        };

        let objectives = vec![3.0, 1.0];
        let result = scalarization.scalarize(&objectives).unwrap();

        assert_eq!(result, 3.0); // max(1.0 * |3.0 - 0.0|, 1.0 * |1.0 - 0.0|)
    }

    #[test]
    fn test_quality_metrics() {
        let solutions = vec![
            MultiObjectiveSolution::new(vec![1, -1], vec![1.0, 3.0]),
            MultiObjectiveSolution::new(vec![-1, 1], vec![2.0, 1.0]),
            MultiObjectiveSolution::new(vec![1, 1], vec![0.5, 2.0]),
        ];

        let spacing = QualityMetrics::spacing(&solutions);
        assert!(spacing >= 0.0);

        let reference_point = vec![4.0, 4.0];
        let hypervolume = QualityMetrics::hypervolume(&solutions, &reference_point).unwrap();
        assert!(hypervolume >= 0.0);
    }
}
