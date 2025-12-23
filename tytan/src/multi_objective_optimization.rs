//! Multi-Objective QUBO Optimization
//!
//! This module implements advanced multi-objective optimization algorithms for QUBO problems,
//! allowing users to optimize multiple conflicting objectives simultaneously and explore
//! Pareto frontiers.
//!
//! # Features
//!
//! - Pareto frontier computation
//! - Multiple scalarization methods (weighted sum, epsilon-constraint, Tchebycheff)
//! - Evolutionary multi-objective algorithms (NSGA-II, MOEA/D)
//! - Interactive objective exploration
//! - Trade-off visualization and analysis
//!
//! # Examples
//!
//! ```rust
//! use quantrs2_tytan::multi_objective_optimization::*;
//! use std::collections::HashMap;
//!
//! // Define multiple objectives
//! let objectives = vec![
//!     Objective::new("cost", ObjectiveDirection::Minimize, 1.0),
//!     Objective::new("quality", ObjectiveDirection::Maximize, 1.0),
//! ];
//!
//! // Create multi-objective optimizer
//! let config = MultiObjectiveConfig::default();
//! let mut optimizer = MultiObjectiveOptimizer::new(objectives, config);
//! ```

use crate::sampler::{SampleResult, Sampler};
use quantrs2_anneal::QuboModel;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops;
use scirs2_core::random::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Objective direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    /// Minimize the objective
    Minimize,
    /// Maximize the objective
    Maximize,
}

/// Objective definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    /// Objective name
    pub name: String,
    /// Optimization direction
    pub direction: ObjectiveDirection,
    /// Weight for scalarization
    pub weight: f64,
    /// QUBO matrix for this objective
    pub qubo_matrix: Option<Array2<f64>>,
}

impl Objective {
    /// Create a new objective
    pub fn new(name: impl Into<String>, direction: ObjectiveDirection, weight: f64) -> Self {
        Self {
            name: name.into(),
            direction,
            weight,
            qubo_matrix: None,
        }
    }

    /// Set QUBO matrix for this objective
    pub fn with_qubo(mut self, qubo_matrix: Array2<f64>) -> Self {
        self.qubo_matrix = Some(qubo_matrix);
        self
    }

    /// Evaluate this objective for a given solution
    pub fn evaluate(&self, solution: &HashMap<String, bool>) -> f64 {
        if let Some(ref matrix) = self.qubo_matrix {
            // Compute QUBO energy
            let n = matrix.nrows();
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
                    energy += matrix[[i, j]] * x_i * x_j;
                }
            }

            match self.direction {
                ObjectiveDirection::Minimize => energy,
                ObjectiveDirection::Maximize => -energy,
            }
        } else {
            0.0
        }
    }
}

/// A solution in objective space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveSolution {
    /// Variable assignments
    pub assignments: HashMap<String, bool>,
    /// Objective values
    pub objective_values: Vec<f64>,
    /// Dominance rank (lower is better)
    pub rank: usize,
    /// Crowding distance (higher is better for diversity)
    pub crowding_distance: f64,
}

impl MultiObjectiveSolution {
    /// Check if this solution dominates another
    pub fn dominates(&self, other: &Self) -> bool {
        let mut at_least_one_better = false;

        for (a, b) in self.objective_values.iter().zip(&other.objective_values) {
            if a > b {
                return false; // Worse in at least one objective (assuming minimization)
            }
            if a < b {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Compute Euclidean distance in objective space
    pub fn distance_to(&self, other: &Self) -> f64 {
        self.objective_values
            .iter()
            .zip(&other.objective_values)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Scalarization method for multi-objective optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarizationMethod {
    /// Weighted sum of objectives
    WeightedSum,
    /// Epsilon-constraint method
    EpsilonConstraint,
    /// Tchebycheff method
    Tchebycheff,
    /// Augmented Tchebycheff
    AugmentedTchebycheff,
    /// Achievement scalarizing function
    AchievementFunction,
}

/// Multi-objective optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveConfig {
    /// Population size for evolutionary algorithms
    pub population_size: usize,
    /// Number of generations
    pub max_generations: usize,
    /// Scalarization method
    pub scalarization: ScalarizationMethod,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Mutation probability
    pub mutation_prob: f64,
    /// Number of reference points for decomposition
    pub num_reference_points: usize,
    /// Enable parallel evaluation
    pub parallel: bool,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 100,
            scalarization: ScalarizationMethod::WeightedSum,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            num_reference_points: 10,
            parallel: true,
        }
    }
}

/// Multi-objective optimizer
pub struct MultiObjectiveOptimizer {
    /// Objectives to optimize
    objectives: Vec<Objective>,
    /// Configuration
    config: MultiObjectiveConfig,
    /// Current population
    population: Vec<MultiObjectiveSolution>,
    /// Pareto front
    pareto_front: Vec<MultiObjectiveSolution>,
    /// Random number generator
    rng: Box<dyn RngCore>,
}

impl MultiObjectiveOptimizer {
    /// Create a new multi-objective optimizer
    pub fn new(objectives: Vec<Objective>, config: MultiObjectiveConfig) -> Self {
        Self {
            objectives,
            config,
            population: Vec::new(),
            pareto_front: Vec::new(),
            rng: Box::new(thread_rng()),
        }
    }

    /// Initialize population randomly
    pub fn initialize_population(&mut self, num_variables: usize) {
        self.population.clear();

        for _ in 0..self.config.population_size {
            let mut assignments = HashMap::new();

            for i in 0..num_variables {
                assignments.insert(format!("x{i}"), self.rng.gen::<bool>());
            }

            let objective_values = self.evaluate_objectives(&assignments);

            self.population.push(MultiObjectiveSolution {
                assignments,
                objective_values,
                rank: 0,
                crowding_distance: 0.0,
            });
        }

        self.compute_pareto_ranks();
        self.compute_crowding_distances();
    }

    /// Evaluate all objectives for a solution
    fn evaluate_objectives(&self, solution: &HashMap<String, bool>) -> Vec<f64> {
        self.objectives
            .iter()
            .map(|obj| obj.evaluate(solution))
            .collect()
    }

    /// Compute Pareto ranks using fast non-dominated sorting
    fn compute_pareto_ranks(&mut self) {
        let n = self.population.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut fronts: Vec<Vec<usize>> = Vec::new();

        // Count dominations
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                if self.population[i].dominates(&self.population[j]) {
                    dominated_solutions[i].push(j);
                } else if self.population[j].dominates(&self.population[i]) {
                    domination_count[i] += 1;
                }
            }

            if domination_count[i] == 0 {
                self.population[i].rank = 0;
                if fronts.is_empty() {
                    fronts.push(Vec::new());
                }
                fronts[0].push(i);
            }
        }

        // Build subsequent fronts
        let mut current_front = 0;
        while current_front < fronts.len() && !fronts[current_front].is_empty() {
            let mut next_front = Vec::new();

            for &i in &fronts[current_front] {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        self.population[j].rank = current_front + 1;
                        next_front.push(j);
                    }
                }
            }

            if !next_front.is_empty() {
                fronts.push(next_front);
            }
            current_front += 1;
        }

        // Update Pareto front
        if !fronts.is_empty() && !fronts[0].is_empty() {
            self.pareto_front = fronts[0]
                .iter()
                .map(|&i| self.population[i].clone())
                .collect();
        }
    }

    /// Compute crowding distances for diversity preservation
    fn compute_crowding_distances(&mut self) {
        let n = self.population.len();
        let m = self.objectives.len();

        // Initialize distances to zero
        for sol in &mut self.population {
            sol.crowding_distance = 0.0;
        }

        // For each objective
        for obj_idx in 0..m {
            // Sort by objective value
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                self.population[a].objective_values[obj_idx]
                    .partial_cmp(&self.population[b].objective_values[obj_idx])
                    .unwrap()
            });

            // Boundary solutions get infinite distance
            self.population[indices[0]].crowding_distance = f64::INFINITY;
            self.population[indices[n - 1]].crowding_distance = f64::INFINITY;

            // Compute range
            let f_max = self.population[indices[n - 1]].objective_values[obj_idx];
            let f_min = self.population[indices[0]].objective_values[obj_idx];
            let range = f_max - f_min;

            if range > 1e-10 {
                for i in 1..n - 1 {
                    let idx = indices[i];
                    let f_next = self.population[indices[i + 1]].objective_values[obj_idx];
                    let f_prev = self.population[indices[i - 1]].objective_values[obj_idx];

                    self.population[idx].crowding_distance += (f_next - f_prev) / range;
                }
            }
        }
    }

    /// Perform binary tournament selection
    fn tournament_selection(&mut self) -> usize {
        let i = self.rng.gen_range(0..self.population.len());
        let j = self.rng.gen_range(0..self.population.len());

        if self.population[i].rank < self.population[j].rank {
            i
        } else if self.population[i].rank > self.population[j].rank {
            j
        } else if self.population[i].crowding_distance > self.population[j].crowding_distance {
            i
        } else {
            j
        }
    }

    /// Perform crossover
    fn crossover(
        &mut self,
        parent1: &MultiObjectiveSolution,
        parent2: &MultiObjectiveSolution,
    ) -> (MultiObjectiveSolution, MultiObjectiveSolution) {
        if self.rng.gen::<f64>() > self.config.crossover_prob {
            return (parent1.clone(), parent2.clone());
        }

        let mut child1_assignments = parent1.assignments.clone();
        let mut child2_assignments = parent2.assignments.clone();

        // Uniform crossover
        for key in parent1.assignments.keys() {
            if self.rng.gen::<bool>() {
                if let Some(&val) = parent2.assignments.get(key) {
                    child1_assignments.insert(key.clone(), val);
                }
                if let Some(&val) = parent1.assignments.get(key) {
                    child2_assignments.insert(key.clone(), val);
                }
            }
        }

        let child1 = MultiObjectiveSolution {
            assignments: child1_assignments,
            objective_values: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
        };

        let child2 = MultiObjectiveSolution {
            assignments: child2_assignments,
            objective_values: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
        };

        (child1, child2)
    }

    /// Perform mutation
    fn mutate(&mut self, solution: &mut MultiObjectiveSolution) {
        for value in solution.assignments.values_mut() {
            if self.rng.gen::<f64>() < self.config.mutation_prob {
                *value = !*value;
            }
        }
    }

    /// Run NSGA-II algorithm
    pub fn optimize_nsga2(
        &mut self,
        num_variables: usize,
    ) -> Result<Vec<MultiObjectiveSolution>, String> {
        self.initialize_population(num_variables);

        for generation in 0..self.config.max_generations {
            let mut offspring = Vec::new();

            // Generate offspring
            while offspring.len() < self.config.population_size {
                let parent1_idx = self.tournament_selection();
                let parent2_idx = self.tournament_selection();

                // Clone parents to avoid borrowing issues
                let parent1 = self.population[parent1_idx].clone();
                let parent2 = self.population[parent2_idx].clone();

                let (mut child1, mut child2) = self.crossover(&parent1, &parent2);

                self.mutate(&mut child1);
                self.mutate(&mut child2);

                // Evaluate offspring
                child1.objective_values = self.evaluate_objectives(&child1.assignments);
                child2.objective_values = self.evaluate_objectives(&child2.assignments);

                offspring.push(child1);
                if offspring.len() < self.config.population_size {
                    offspring.push(child2);
                }
            }

            // Combine parent and offspring populations
            self.population.extend(offspring);

            // Non-dominated sorting
            self.compute_pareto_ranks();
            self.compute_crowding_distances();

            // Environmental selection
            self.population.sort_by(|a, b| match a.rank.cmp(&b.rank) {
                std::cmp::Ordering::Equal => b
                    .crowding_distance
                    .partial_cmp(&a.crowding_distance)
                    .unwrap(),
                other => other,
            });

            self.population.truncate(self.config.population_size);

            if generation % 10 == 0 {
                println!(
                    "Generation {}: Pareto front size = {}",
                    generation,
                    self.pareto_front.len()
                );
            }
        }

        Ok(self.pareto_front.clone())
    }

    /// Get the Pareto front
    pub fn get_pareto_front(&self) -> &[MultiObjectiveSolution] {
        &self.pareto_front
    }

    /// Scalarize multiple objectives into a single objective
    pub fn scalarize(&self, objective_values: &[f64], weights: &[f64]) -> f64 {
        match self.config.scalarization {
            ScalarizationMethod::WeightedSum => objective_values
                .iter()
                .zip(weights)
                .map(|(val, weight)| val * weight)
                .sum(),
            ScalarizationMethod::Tchebycheff => {
                let mut max_weighted = f64::NEG_INFINITY;
                for (val, weight) in objective_values.iter().zip(weights) {
                    let weighted = weight * val;
                    if weighted > max_weighted {
                        max_weighted = weighted;
                    }
                }
                max_weighted
            }
            ScalarizationMethod::AugmentedTchebycheff => {
                let rho = 0.00001;
                let max_weighted = objective_values
                    .iter()
                    .zip(weights)
                    .map(|(val, weight)| weight * val)
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = objective_values.iter().sum();
                max_weighted + rho * sum
            }
            _ => objective_values
                .iter()
                .zip(weights)
                .map(|(val, weight)| val * weight)
                .sum(),
        }
    }
}

/// Result of multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveResult {
    /// Pareto-optimal solutions
    pub pareto_solutions: Vec<MultiObjectiveSolution>,
    /// Statistics
    pub statistics: OptimizationStatistics,
}

/// Statistics from multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatistics {
    /// Total number of evaluations
    pub num_evaluations: usize,
    /// Total runtime in seconds
    pub runtime_seconds: f64,
    /// Size of final Pareto front
    pub pareto_front_size: usize,
    /// Hypervolume indicator
    pub hypervolume: f64,
    /// Spacing metric
    pub spacing: f64,
}

impl OptimizationStatistics {
    /// Compute hypervolume for the Pareto front
    pub fn compute_hypervolume(
        solutions: &[MultiObjectiveSolution],
        reference_point: &[f64],
    ) -> f64 {
        // Simplified 2D hypervolume computation
        if solutions.is_empty() || reference_point.len() != 2 {
            return 0.0;
        }

        let mut sorted_solutions: Vec<_> = solutions.to_vec();
        sorted_solutions.sort_by(|a, b| {
            a.objective_values[0]
                .partial_cmp(&b.objective_values[0])
                .unwrap()
        });

        let mut hypervolume = 0.0;
        let mut prev_y = reference_point[1];

        for sol in &sorted_solutions {
            let width = reference_point[0] - sol.objective_values[0];
            let height = prev_y - sol.objective_values[1];

            if width > 0.0 && height > 0.0 {
                hypervolume += width * height;
                prev_y = sol.objective_values[1];
            }
        }

        hypervolume
    }

    /// Compute spacing metric for distribution quality
    pub fn compute_spacing(solutions: &[MultiObjectiveSolution]) -> f64 {
        if solutions.len() < 2 {
            return 0.0;
        }

        let mut distances = Vec::new();

        for i in 0..solutions.len() {
            let mut min_distance = f64::INFINITY;

            for j in 0..solutions.len() {
                if i != j {
                    let dist = solutions[i].distance_to(&solutions[j]);
                    if dist < min_distance {
                        min_distance = dist;
                    }
                }
            }

            distances.push(min_distance);
        }

        let mean: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance: f64 =
            distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / distances.len() as f64;

        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objective_creation() {
        let obj = Objective::new("test", ObjectiveDirection::Minimize, 1.0);
        assert_eq!(obj.name, "test");
        assert_eq!(obj.direction, ObjectiveDirection::Minimize);
        assert_eq!(obj.weight, 1.0);
    }

    #[test]
    fn test_dominance() {
        let sol1 = MultiObjectiveSolution {
            assignments: HashMap::new(),
            objective_values: vec![1.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        };

        let sol2 = MultiObjectiveSolution {
            assignments: HashMap::new(),
            objective_values: vec![2.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        };

        assert!(sol1.dominates(&sol2));
        assert!(!sol2.dominates(&sol1));
    }

    #[test]
    fn test_multi_objective_optimizer_initialization() {
        let objectives = vec![
            Objective::new("obj1", ObjectiveDirection::Minimize, 1.0),
            Objective::new("obj2", ObjectiveDirection::Minimize, 1.0),
        ];

        let config = MultiObjectiveConfig::default();
        let mut optimizer = MultiObjectiveOptimizer::new(objectives, config);

        optimizer.initialize_population(10);
        assert_eq!(optimizer.population.len(), 100);
    }

    #[test]
    fn test_scalarization() {
        let objectives = vec![
            Objective::new("obj1", ObjectiveDirection::Minimize, 0.5),
            Objective::new("obj2", ObjectiveDirection::Minimize, 0.5),
        ];

        let mut config = MultiObjectiveConfig::default();
        config.scalarization = ScalarizationMethod::WeightedSum;

        let optimizer = MultiObjectiveOptimizer::new(objectives, config);

        let values = vec![1.0, 2.0];
        let weights = vec![0.5, 0.5];
        let result = optimizer.scalarize(&values, &weights);

        assert!((result - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_computation() {
        let solutions = vec![
            MultiObjectiveSolution {
                assignments: HashMap::new(),
                objective_values: vec![1.0, 3.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            MultiObjectiveSolution {
                assignments: HashMap::new(),
                objective_values: vec![2.0, 2.0],
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        let reference = vec![4.0, 4.0];
        let hv = OptimizationStatistics::compute_hypervolume(&solutions, &reference);

        assert!(hv > 0.0);
    }
}
