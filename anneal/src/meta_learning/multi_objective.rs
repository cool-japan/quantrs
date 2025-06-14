//! Multi-Objective Optimization for Meta-Learning
//!
//! This module contains all Multi-Objective Optimization types and implementations
//! used by the meta-learning optimization system.

use super::config::{
    ConstraintHandling, FrontierUpdateStrategy, MultiObjectiveConfig, OptimizationConfiguration,
    OptimizationObjective, ParetoFrontierConfig, ScalarizationMethod,
};
use crate::applications::ApplicationResult;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Multi-objective optimizer
pub struct MultiObjectiveOptimizer {
    /// Configuration
    pub config: MultiObjectiveConfig,
    /// Pareto frontier
    pub pareto_frontier: ParetoFrontier,
    /// Scalarization methods
    pub scalarizers: Vec<Scalarizer>,
    /// Constraint handlers
    pub constraint_handlers: Vec<ConstraintHandler>,
    /// Decision maker
    pub decision_maker: DecisionMaker,
}

/// Pareto frontier representation
#[derive(Debug)]
pub struct ParetoFrontier {
    /// Non-dominated solutions
    pub solutions: Vec<MultiObjectiveSolution>,
    /// Frontier statistics
    pub statistics: FrontierStatistics,
    /// Update history
    pub update_history: VecDeque<FrontierUpdate>,
}

/// Multi-objective solution
#[derive(Debug, Clone)]
pub struct MultiObjectiveSolution {
    /// Solution identifier
    pub id: String,
    /// Objective values
    pub objective_values: Vec<f64>,
    /// Decision variables
    pub decision_variables: OptimizationConfiguration,
    /// Dominance rank
    pub dominance_rank: usize,
    /// Crowding distance
    pub crowding_distance: f64,
}

/// Frontier statistics
#[derive(Debug, Clone)]
pub struct FrontierStatistics {
    /// Frontier size
    pub size: usize,
    /// Hypervolume
    pub hypervolume: f64,
    /// Spread
    pub spread: f64,
    /// Convergence metric
    pub convergence: f64,
    /// Coverage
    pub coverage: f64,
}

/// Frontier update
#[derive(Debug, Clone)]
pub struct FrontierUpdate {
    /// Update timestamp
    pub timestamp: Instant,
    /// Solutions added
    pub solutions_added: Vec<String>,
    /// Solutions removed
    pub solutions_removed: Vec<String>,
    /// Update reason
    pub reason: UpdateReason,
}

/// Update reasons
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateReason {
    /// New non-dominated solution
    NewNonDominated,
    /// Dominated solution removal
    DominatedRemoval,
    /// Capacity limit reached
    CapacityLimit,
    /// Quality improvement
    QualityImprovement,
}

/// Scalarization function
#[derive(Debug)]
pub struct Scalarizer {
    /// Method used
    pub method: ScalarizationMethod,
    /// Weights or preferences
    pub weights: Vec<f64>,
    /// Reference point
    pub reference_point: Option<Vec<f64>>,
    /// Parameters
    pub parameters: HashMap<String, f64>,
}

/// Constraint handler
#[derive(Debug)]
pub struct ConstraintHandler {
    /// Handling method
    pub method: ConstraintHandling,
    /// Constraints
    pub constraints: Vec<Constraint>,
    /// Penalty parameters
    pub penalty_parameters: HashMap<String, f64>,
}

/// Constraint definition
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint function
    pub function: String,
    /// Bounds
    pub bounds: (f64, f64),
    /// Tolerance
    pub tolerance: f64,
}

/// Constraint types
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// Equality constraint
    Equality,
    /// Inequality constraint
    Inequality,
    /// Box constraint
    Box,
    /// Linear constraint
    Linear,
    /// Nonlinear constraint
    Nonlinear,
}

/// Decision maker for multi-objective problems
#[derive(Debug)]
pub struct DecisionMaker {
    /// Decision strategy
    pub strategy: DecisionStrategy,
    /// Preference information
    pub preferences: UserPreferences,
    /// Decision history
    pub decision_history: VecDeque<Decision>,
}

/// Decision strategies
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionStrategy {
    /// Interactive decision making
    Interactive,
    /// A priori preferences
    APriori,
    /// A posteriori analysis
    APosteriori,
    /// Progressive articulation
    Progressive,
    /// Automated decision
    Automated,
}

/// User preferences
#[derive(Debug, Clone)]
pub struct UserPreferences {
    /// Objective weights
    pub objective_weights: Vec<f64>,
    /// Acceptable trade-offs
    pub trade_offs: HashMap<String, f64>,
    /// Constraints
    pub user_constraints: Vec<Constraint>,
    /// Preference functions
    pub preference_functions: Vec<PreferenceFunction>,
}

/// Preference function
#[derive(Debug, Clone)]
pub struct PreferenceFunction {
    /// Function type
    pub function_type: PreferenceFunctionType,
    /// Parameters
    pub parameters: Vec<f64>,
    /// Applicable objectives
    pub objectives: Vec<usize>,
}

/// Types of preference functions
#[derive(Debug, Clone, PartialEq)]
pub enum PreferenceFunctionType {
    /// Linear preference
    Linear,
    /// Exponential preference
    Exponential,
    /// Logarithmic preference
    Logarithmic,
    /// Threshold-based
    Threshold,
    /// Custom function
    Custom(String),
}

/// Decision record
#[derive(Debug, Clone)]
pub struct Decision {
    /// Decision timestamp
    pub timestamp: Instant,
    /// Selected solution
    pub selected_solution: String,
    /// Decision rationale
    pub rationale: String,
    /// Confidence level
    pub confidence: f64,
    /// User feedback
    pub user_feedback: Option<f64>,
}

impl MultiObjectiveOptimizer {
    pub fn new(config: MultiObjectiveConfig) -> Self {
        Self {
            config: config.clone(),
            pareto_frontier: ParetoFrontier {
                solutions: Vec::new(),
                statistics: FrontierStatistics {
                    size: 0,
                    hypervolume: 0.0,
                    spread: 0.0,
                    convergence: 0.0,
                    coverage: 0.0,
                },
                update_history: VecDeque::new(),
            },
            scalarizers: Vec::new(),
            constraint_handlers: Vec::new(),
            decision_maker: DecisionMaker {
                strategy: DecisionStrategy::Automated,
                preferences: UserPreferences {
                    objective_weights: vec![0.5, 0.3, 0.2],
                    trade_offs: HashMap::new(),
                    user_constraints: Vec::new(),
                    preference_functions: Vec::new(),
                },
                decision_history: VecDeque::new(),
            },
        }
    }

    /// Add solution to Pareto frontier
    pub fn add_solution(&mut self, solution: MultiObjectiveSolution) -> ApplicationResult<bool> {
        // Check if solution is non-dominated
        let is_non_dominated = self.is_non_dominated(&solution);

        if is_non_dominated {
            // Remove dominated solutions
            let solutions_to_keep: Vec<_> = self
                .pareto_frontier
                .solutions
                .iter()
                .filter(|existing| !self.dominates(&solution, existing))
                .cloned()
                .collect();
            self.pareto_frontier.solutions = solutions_to_keep;

            // Add new solution
            self.pareto_frontier.solutions.push(solution.clone());

            // Update statistics
            self.update_frontier_statistics();

            // Record update
            let update = FrontierUpdate {
                timestamp: Instant::now(),
                solutions_added: vec![solution.id],
                solutions_removed: Vec::new(),
                reason: UpdateReason::NewNonDominated,
            };
            self.pareto_frontier.update_history.push_back(update);

            // Limit history size
            if self.pareto_frontier.update_history.len() > 1000 {
                self.pareto_frontier.update_history.pop_front();
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if solution is non-dominated
    fn is_non_dominated(&self, solution: &MultiObjectiveSolution) -> bool {
        for existing in &self.pareto_frontier.solutions {
            if self.dominates(existing, solution) {
                return false;
            }
        }
        true
    }

    /// Check if solution1 dominates solution2
    fn dominates(
        &self,
        solution1: &MultiObjectiveSolution,
        solution2: &MultiObjectiveSolution,
    ) -> bool {
        let mut at_least_one_better = false;

        for (val1, val2) in solution1
            .objective_values
            .iter()
            .zip(&solution2.objective_values)
        {
            if val1 < val2 {
                return false; // Assuming minimization
            }
            if val1 > val2 {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Update frontier statistics
    fn update_frontier_statistics(&mut self) {
        self.pareto_frontier.statistics.size = self.pareto_frontier.solutions.len();

        // Calculate hypervolume (simplified)
        self.pareto_frontier.statistics.hypervolume = self.calculate_hypervolume();

        // Calculate spread
        self.pareto_frontier.statistics.spread = self.calculate_spread();

        // Update convergence metric
        self.pareto_frontier.statistics.convergence = 0.8; // Simplified

        // Update coverage
        self.pareto_frontier.statistics.coverage = 0.9; // Simplified
    }

    /// Calculate hypervolume (simplified implementation)
    fn calculate_hypervolume(&self) -> f64 {
        if self.pareto_frontier.solutions.is_empty() {
            return 0.0;
        }

        // Simple hypervolume calculation
        let mut volume = 0.0;
        for solution in &self.pareto_frontier.solutions {
            let mut point_volume = 1.0;
            for &value in &solution.objective_values {
                point_volume *= value.max(0.0);
            }
            volume += point_volume;
        }

        volume
    }

    /// Calculate spread metric
    fn calculate_spread(&self) -> f64 {
        if self.pareto_frontier.solutions.len() < 2 {
            return 0.0;
        }

        // Simple spread calculation based on distance between solutions
        let mut total_distance = 0.0;
        let num_objectives = self.pareto_frontier.solutions[0].objective_values.len();

        for i in 0..num_objectives {
            let mut values: Vec<f64> = self
                .pareto_frontier
                .solutions
                .iter()
                .map(|s| s.objective_values[i])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if let (Some(&min), Some(&max)) = (values.first(), values.last()) {
                total_distance += max - min;
            }
        }

        total_distance / num_objectives as f64
    }

    /// Scalarize objectives using weighted sum
    pub fn scalarize_weighted_sum(
        &self,
        solution: &MultiObjectiveSolution,
        weights: &[f64],
    ) -> f64 {
        solution
            .objective_values
            .iter()
            .zip(weights)
            .map(|(value, weight)| value * weight)
            .sum()
    }

    /// Select best solution using decision maker preferences
    pub fn select_solution(&mut self) -> ApplicationResult<Option<String>> {
        if self.pareto_frontier.solutions.is_empty() {
            return Ok(None);
        }

        match self.decision_maker.strategy {
            DecisionStrategy::Automated => {
                // Use weighted sum with user preferences
                let weights = &self.decision_maker.preferences.objective_weights;

                let mut best_solution = None;
                let mut best_score = f64::NEG_INFINITY;

                for solution in &self.pareto_frontier.solutions {
                    let score = self.scalarize_weighted_sum(solution, weights);
                    if score > best_score {
                        best_score = score;
                        best_solution = Some(solution.id.clone());
                    }
                }

                if let Some(ref solution_id) = best_solution {
                    // Record decision
                    let decision = Decision {
                        timestamp: Instant::now(),
                        selected_solution: solution_id.clone(),
                        rationale: "Automated selection using weighted sum".to_string(),
                        confidence: 0.8,
                        user_feedback: None,
                    };
                    self.decision_maker.decision_history.push_back(decision);

                    // Limit history size
                    if self.decision_maker.decision_history.len() > 100 {
                        self.decision_maker.decision_history.pop_front();
                    }
                }

                Ok(best_solution)
            }
            _ => {
                // For other strategies, just return the first solution for now
                Ok(self.pareto_frontier.solutions.first().map(|s| s.id.clone()))
            }
        }
    }

    /// Get frontier statistics
    pub fn get_statistics(&self) -> &FrontierStatistics {
        &self.pareto_frontier.statistics
    }

    /// Get all solutions in Pareto frontier
    pub fn get_pareto_solutions(&self) -> &Vec<MultiObjectiveSolution> {
        &self.pareto_frontier.solutions
    }

    /// Clear Pareto frontier
    pub fn clear_frontier(&mut self) {
        self.pareto_frontier.solutions.clear();
        self.update_frontier_statistics();

        let update = FrontierUpdate {
            timestamp: Instant::now(),
            solutions_added: Vec::new(),
            solutions_removed: Vec::new(),
            reason: UpdateReason::QualityImprovement,
        };
        self.pareto_frontier.update_history.push_back(update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_learning::config::*;
    use crate::meta_learning::portfolio::{AlgorithmType, ResourceAllocation};

    #[test]
    fn test_multi_objective_optimizer_creation() {
        let config = MultiObjectiveConfig::default();
        let optimizer = MultiObjectiveOptimizer::new(config);

        assert_eq!(optimizer.pareto_frontier.solutions.len(), 0);
        assert_eq!(optimizer.pareto_frontier.statistics.size, 0);
    }

    #[test]
    fn test_solution_addition() {
        let config = MultiObjectiveConfig::default();
        let mut optimizer = MultiObjectiveOptimizer::new(config);

        let solution = MultiObjectiveSolution {
            id: "test_solution".to_string(),
            objective_values: vec![1.0, 2.0, 3.0],
            decision_variables: OptimizationConfiguration {
                algorithm: AlgorithmType::SimulatedAnnealing,
                hyperparameters: HashMap::new(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(60),
                },
            },
            dominance_rank: 0,
            crowding_distance: 0.0,
        };

        let result = optimizer.add_solution(solution);
        assert!(result.is_ok());
        assert!(result.unwrap());
        assert_eq!(optimizer.pareto_frontier.solutions.len(), 1);
    }

    #[test]
    fn test_dominance_check() {
        let config = MultiObjectiveConfig::default();
        let optimizer = MultiObjectiveOptimizer::new(config);

        let solution1 = MultiObjectiveSolution {
            id: "solution1".to_string(),
            objective_values: vec![1.0, 2.0],
            decision_variables: OptimizationConfiguration {
                algorithm: AlgorithmType::QuantumAnnealing,
                hyperparameters: HashMap::new(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(60),
                },
            },
            dominance_rank: 0,
            crowding_distance: 0.0,
        };

        let solution2 = MultiObjectiveSolution {
            id: "solution2".to_string(),
            objective_values: vec![2.0, 1.0],
            decision_variables: OptimizationConfiguration {
                algorithm: AlgorithmType::TabuSearch,
                hyperparameters: HashMap::new(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(60),
                },
            },
            dominance_rank: 0,
            crowding_distance: 0.0,
        };

        // Neither solution should dominate the other (trade-off)
        assert!(!optimizer.dominates(&solution1, &solution2));
        assert!(!optimizer.dominates(&solution2, &solution1));
    }

    #[test]
    fn test_weighted_sum_scalarization() {
        let config = MultiObjectiveConfig::default();
        let optimizer = MultiObjectiveOptimizer::new(config);

        let solution = MultiObjectiveSolution {
            id: "test_solution".to_string(),
            objective_values: vec![2.0, 3.0, 1.0],
            decision_variables: OptimizationConfiguration {
                algorithm: AlgorithmType::GeneticAlgorithm,
                hyperparameters: HashMap::new(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(60),
                },
            },
            dominance_rank: 0,
            crowding_distance: 0.0,
        };

        let weights = vec![0.5, 0.3, 0.2];
        let score = optimizer.scalarize_weighted_sum(&solution, &weights);

        // Expected: 2.0*0.5 + 3.0*0.3 + 1.0*0.2 = 1.0 + 0.9 + 0.2 = 2.1
        assert!((score - 2.1).abs() < 1e-10);
    }

    #[test]
    fn test_frontier_statistics() {
        let config = MultiObjectiveConfig::default();
        let mut optimizer = MultiObjectiveOptimizer::new(config);

        // Add a solution
        let solution = MultiObjectiveSolution {
            id: "test_solution".to_string(),
            objective_values: vec![1.0, 2.0],
            decision_variables: OptimizationConfiguration {
                algorithm: AlgorithmType::ParticleSwarm,
                hyperparameters: HashMap::new(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(60),
                },
            },
            dominance_rank: 0,
            crowding_distance: 0.0,
        };

        optimizer.add_solution(solution).unwrap();

        let stats = optimizer.get_statistics();
        assert_eq!(stats.size, 1);
        assert!(stats.hypervolume > 0.0);
    }

    #[test]
    fn test_solution_selection() {
        let config = MultiObjectiveConfig::default();
        let mut optimizer = MultiObjectiveOptimizer::new(config);

        // Add solutions
        let solution1 = MultiObjectiveSolution {
            id: "solution1".to_string(),
            objective_values: vec![1.0, 2.0],
            decision_variables: OptimizationConfiguration {
                algorithm: AlgorithmType::AntColony,
                hyperparameters: HashMap::new(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(60),
                },
            },
            dominance_rank: 0,
            crowding_distance: 0.0,
        };

        let solution2 = MultiObjectiveSolution {
            id: "solution2".to_string(),
            objective_values: vec![2.0, 1.0],
            decision_variables: OptimizationConfiguration {
                algorithm: AlgorithmType::VariableNeighborhood,
                hyperparameters: HashMap::new(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(60),
                },
            },
            dominance_rank: 0,
            crowding_distance: 0.0,
        };

        optimizer.add_solution(solution1).unwrap();
        optimizer.add_solution(solution2).unwrap();

        let selected = optimizer.select_solution();
        assert!(selected.is_ok());
        assert!(selected.unwrap().is_some());
    }
}
