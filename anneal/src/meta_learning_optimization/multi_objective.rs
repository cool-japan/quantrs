//! Multi-objective optimization for meta-learning

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::config::*;
use super::feature_extraction::*;

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

impl MultiObjectiveOptimizer {
    pub fn new(config: MultiObjectiveConfig) -> Self {
        Self {
            pareto_frontier: ParetoFrontier::new(&config.pareto_config),
            scalarizers: vec![Scalarizer::new(config.scalarization.clone())],
            constraint_handlers: vec![ConstraintHandler::new(config.constraint_handling.clone())],
            decision_maker: DecisionMaker::new(),
            config,
        }
    }

    pub fn optimize(&mut self, candidates: Vec<OptimizationConfiguration>) -> Result<Vec<MultiObjectiveSolution>, String> {
        let mut solutions = Vec::new();

        for (i, candidate) in candidates.iter().enumerate() {
            let objective_values = self.evaluate_objectives(candidate)?;
            
            let solution = MultiObjectiveSolution {
                id: format!("solution_{}", i),
                objective_values,
                decision_variables: candidate.clone(),
                dominance_rank: 0, // Will be calculated later
                crowding_distance: 0.0, // Will be calculated later
            };
            
            solutions.push(solution);
        }

        // Update Pareto frontier
        self.update_pareto_frontier(&mut solutions)?;
        
        // Calculate dominance ranks and crowding distances
        self.calculate_dominance_ranks(&mut solutions);
        self.calculate_crowding_distances(&mut solutions);

        Ok(solutions)
    }

    fn evaluate_objectives(&self, config: &OptimizationConfiguration) -> Result<Vec<f64>, String> {
        let mut objective_values = Vec::new();

        for objective in &self.config.objectives {
            let value = match objective {
                OptimizationObjective::SolutionQuality => {
                    // Estimate solution quality based on algorithm type
                    match config.algorithm {
                        AlgorithmType::SimulatedAnnealing => 0.8,
                        AlgorithmType::QuantumAnnealing => 0.9,
                        AlgorithmType::TabuSearch => 0.7,
                        _ => 0.6,
                    }
                },
                OptimizationObjective::Runtime => {
                    // Estimate runtime (lower is better, so we use negative)
                    -config.resources.time.as_secs_f64() / 3600.0 // Convert to hours
                },
                OptimizationObjective::ResourceUsage => {
                    // Estimate resource usage (lower is better)
                    -(config.resources.memory as f64 / 1024.0 + config.resources.cpu)
                },
                OptimizationObjective::EnergyConsumption => {
                    // Estimate energy consumption (lower is better)
                    -(config.resources.cpu * config.resources.time.as_secs_f64() / 3600.0)
                },
                OptimizationObjective::Robustness => {
                    // Estimate robustness based on algorithm characteristics
                    match config.algorithm {
                        AlgorithmType::QuantumAnnealing => 0.9,
                        AlgorithmType::SimulatedAnnealing => 0.7,
                        _ => 0.6,
                    }
                },
                OptimizationObjective::Scalability => {
                    // Estimate scalability
                    match config.algorithm {
                        AlgorithmType::TabuSearch => 0.9,
                        AlgorithmType::QuantumAnnealing => 0.6,
                        _ => 0.7,
                    }
                },
                OptimizationObjective::Custom(_) => 0.5, // Default value
            };
            
            objective_values.push(value);
        }

        Ok(objective_values)
    }

    fn update_pareto_frontier(&mut self, solutions: &mut Vec<MultiObjectiveSolution>) -> Result<(), String> {
        let mut new_solutions = Vec::new();
        let mut updated_solutions = Vec::new();

        // Check each solution for dominance
        for solution in solutions.iter() {
            let mut is_dominated = false;
            let mut dominated_solutions = Vec::new();

            // Compare with existing frontier solutions
            for frontier_solution in &self.pareto_frontier.solutions {
                if self.dominates(&frontier_solution.objective_values, &solution.objective_values) {
                    is_dominated = true;
                    break;
                } else if self.dominates(&solution.objective_values, &frontier_solution.objective_values) {
                    dominated_solutions.push(frontier_solution.id.clone());
                }
            }

            if !is_dominated {
                new_solutions.push(solution.clone());
                
                // Remove dominated solutions from frontier
                self.pareto_frontier.solutions.retain(|s| !dominated_solutions.contains(&s.id));
            }
        }

        // Add new non-dominated solutions to frontier
        self.pareto_frontier.solutions.extend(new_solutions);

        // Limit frontier size
        if self.pareto_frontier.solutions.len() > self.config.pareto_config.max_frontier_size {
            self.prune_frontier()?;
        }

        // Update frontier statistics
        self.update_frontier_statistics();

        Ok(())
    }

    fn dominates(&self, obj1: &[f64], obj2: &[f64]) -> bool {
        if obj1.len() != obj2.len() {
            return false;
        }

        let mut at_least_one_better = false;
        for (v1, v2) in obj1.iter().zip(obj2.iter()) {
            if v1 < v2 - self.config.pareto_config.dominance_tolerance {
                return false; // obj1 is worse in this objective
            }
            if v1 > v2 + self.config.pareto_config.dominance_tolerance {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    fn prune_frontier(&mut self) -> Result<(), String> {
        // Use crowding distance to maintain diversity
        self.calculate_crowding_distances_frontier();
        
        // Sort by crowding distance (descending) and keep top solutions
        self.pareto_frontier.solutions.sort_by(|a, b| {
            b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap()
        });
        
        self.pareto_frontier.solutions.truncate(self.config.pareto_config.max_frontier_size);
        
        Ok(())
    }

    fn calculate_crowding_distances_frontier(&mut self) {
        let num_solutions = self.pareto_frontier.solutions.len();
        let num_objectives = if let Some(first) = self.pareto_frontier.solutions.first() {
            first.objective_values.len()
        } else {
            return;
        };

        // Initialize crowding distances
        for solution in &mut self.pareto_frontier.solutions {
            solution.crowding_distance = 0.0;
        }

        // Calculate crowding distance for each objective
        for obj_idx in 0..num_objectives {
            // Sort by objective value
            self.pareto_frontier.solutions.sort_by(|a, b| {
                a.objective_values[obj_idx].partial_cmp(&b.objective_values[obj_idx]).unwrap()
            });

            // Set boundary solutions to infinite distance
            if num_solutions > 2 {
                self.pareto_frontier.solutions[0].crowding_distance = f64::INFINITY;
                self.pareto_frontier.solutions[num_solutions - 1].crowding_distance = f64::INFINITY;

                // Calculate distance for interior solutions
                let obj_range = self.pareto_frontier.solutions[num_solutions - 1].objective_values[obj_idx] - 
                               self.pareto_frontier.solutions[0].objective_values[obj_idx];

                if obj_range > 0.0 {
                    for i in 1..num_solutions - 1 {
                        let distance = (self.pareto_frontier.solutions[i + 1].objective_values[obj_idx] - 
                                       self.pareto_frontier.solutions[i - 1].objective_values[obj_idx]) / obj_range;
                        self.pareto_frontier.solutions[i].crowding_distance += distance;
                    }
                }
            }
        }
    }

    fn calculate_dominance_ranks(&self, solutions: &mut Vec<MultiObjectiveSolution>) {
        // Simple dominance ranking - all solutions get rank 1 for now
        for solution in solutions.iter_mut() {
            solution.dominance_rank = 1;
        }
    }

    fn calculate_crowding_distances(&self, solutions: &mut Vec<MultiObjectiveSolution>) {
        // Simple crowding distance calculation
        for (i, solution) in solutions.iter_mut().enumerate() {
            solution.crowding_distance = i as f64; // Simplified
        }
    }

    fn update_frontier_statistics(&mut self) {
        self.pareto_frontier.statistics.size = self.pareto_frontier.solutions.len();
        
        // Calculate hypervolume (simplified)
        self.pareto_frontier.statistics.hypervolume = self.pareto_frontier.solutions.len() as f64 * 0.1;
        
        // Calculate spread (simplified)
        if self.pareto_frontier.solutions.len() > 1 {
            self.pareto_frontier.statistics.spread = 1.0;
        } else {
            self.pareto_frontier.statistics.spread = 0.0;
        }
        
        // Update convergence and coverage
        self.pareto_frontier.statistics.convergence = 0.8;
        self.pareto_frontier.statistics.coverage = 0.9;
    }

    pub fn make_decision(&mut self, preferences: Option<UserPreferences>) -> Result<MultiObjectiveSolution, String> {
        if self.pareto_frontier.solutions.is_empty() {
            return Err("No solutions in Pareto frontier".to_string());
        }

        self.decision_maker.make_decision(&self.pareto_frontier.solutions, preferences)
    }

    pub fn scalarize(&self, solution: &MultiObjectiveSolution, weights: &[f64]) -> Result<f64, String> {
        if let Some(scalarizer) = self.scalarizers.first() {
            scalarizer.scalarize(&solution.objective_values, weights)
        } else {
            Err("No scalarizer available".to_string())
        }
    }
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

impl ParetoFrontier {
    pub fn new(config: &ParetoFrontierConfig) -> Self {
        Self {
            solutions: Vec::new(),
            statistics: FrontierStatistics {
                size: 0,
                hypervolume: 0.0,
                spread: 0.0,
                convergence: 0.0,
                coverage: 0.0,
            },
            update_history: VecDeque::new(),
        }
    }
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
    /// Dominance detected
    DominanceUpdate,
    /// Crowding-based pruning
    CrowdingPruning,
    /// Size limit reached
    SizeLimitReached,
}

/// Scalarization function
#[derive(Debug)]
pub struct Scalarizer {
    /// Scalarization method
    pub method: ScalarizationMethod,
    /// Reference point (if applicable)
    pub reference_point: Option<Vec<f64>>,
    /// Weights
    pub weights: Vec<f64>,
}

impl Scalarizer {
    pub fn new(method: ScalarizationMethod) -> Self {
        Self {
            method,
            reference_point: None,
            weights: Vec::new(),
        }
    }

    pub fn scalarize(&self, objectives: &[f64], weights: &[f64]) -> Result<f64, String> {
        if objectives.len() != weights.len() {
            return Err("Objectives and weights length mismatch".to_string());
        }

        match &self.method {
            ScalarizationMethod::WeightedSum => {
                Ok(objectives.iter().zip(weights.iter()).map(|(obj, w)| obj * w).sum())
            },
            ScalarizationMethod::WeightedTchebycheff => {
                let reference = self.reference_point.as_ref()
                    .unwrap_or(&vec![0.0; objectives.len()]);
                
                let mut max_weighted_diff = 0.0;
                for ((obj, ref_val), weight) in objectives.iter().zip(reference.iter()).zip(weights.iter()) {
                    let weighted_diff = weight * (ref_val - obj).abs();
                    max_weighted_diff = max_weighted_diff.max(weighted_diff);
                }
                Ok(max_weighted_diff)
            },
            ScalarizationMethod::AchievementScalarizing => {
                // Simplified achievement scalarizing function
                let weighted_sum: f64 = objectives.iter().zip(weights.iter()).map(|(obj, w)| obj * w).sum();
                let max_objective = objectives.iter().fold(0.0, |acc, &obj| acc.max(obj));
                Ok(weighted_sum + 0.01 * max_objective)
            },
            ScalarizationMethod::PenaltyBoundaryIntersection => {
                // Simplified PBI
                let weighted_sum: f64 = objectives.iter().zip(weights.iter()).map(|(obj, w)| obj * w).sum();
                Ok(weighted_sum)
            },
            ScalarizationMethod::ReferencePoint => {
                if let Some(ref_point) = &self.reference_point {
                    let distance: f64 = objectives.iter().zip(ref_point.iter())
                        .map(|(obj, ref_val)| (obj - ref_val).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    Ok(-distance) // Negative because we want to minimize distance
                } else {
                    Err("Reference point not set for reference point method".to_string())
                }
            },
        }
    }
}

/// Constraint handler
#[derive(Debug)]
pub struct ConstraintHandler {
    /// Handling method
    pub method: ConstraintHandling,
    /// Constraints
    pub constraints: Vec<Constraint>,
}

impl ConstraintHandler {
    pub fn new(method: ConstraintHandling) -> Self {
        Self {
            method,
            constraints: Vec::new(),
        }
    }

    pub fn handle_constraints(&self, solution: &MultiObjectiveSolution) -> Result<f64, String> {
        // Simplified constraint handling - return penalty value
        match self.method {
            ConstraintHandling::PenaltyMethod => Ok(0.0), // No penalty for now
            ConstraintHandling::BarrierMethod => Ok(0.0),
            ConstraintHandling::LagrangianMethod => Ok(0.0),
            ConstraintHandling::FeasibilityRules => Ok(0.0),
            ConstraintHandling::MultiObjectiveConstraint => Ok(0.0),
        }
    }
}

/// Constraint definition
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Constraint identifier
    pub id: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Bounds or parameters
    pub parameters: Vec<f64>,
    /// Violation tolerance
    pub tolerance: f64,
}

/// Constraint types
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// Equality constraint
    Equality,
    /// Inequality constraint (<=)
    Inequality,
    /// Bound constraint
    Bound,
    /// Custom constraint
    Custom(String),
}

/// Decision maker for multi-objective problems
#[derive(Debug)]
pub struct DecisionMaker {
    /// Decision strategy
    pub strategy: DecisionStrategy,
    /// User preferences
    pub preferences: Option<UserPreferences>,
    /// Decision history
    pub decision_history: VecDeque<Decision>,
}

impl DecisionMaker {
    pub fn new() -> Self {
        Self {
            strategy: DecisionStrategy::WeightedSum,
            preferences: None,
            decision_history: VecDeque::new(),
        }
    }

    pub fn make_decision(
        &mut self,
        solutions: &[MultiObjectiveSolution],
        preferences: Option<UserPreferences>,
    ) -> Result<MultiObjectiveSolution, String> {
        if solutions.is_empty() {
            return Err("No solutions to choose from".to_string());
        }

        let selected_solution = match &self.strategy {
            DecisionStrategy::WeightedSum => {
                // Use equal weights if no preferences provided
                let weights = if let Some(ref prefs) = preferences {
                    prefs.objective_weights.clone()
                } else {
                    vec![1.0 / solutions[0].objective_values.len() as f64; solutions[0].objective_values.len()]
                };

                // Find solution with best weighted sum
                solutions.iter()
                    .max_by(|a, b| {
                        let score_a: f64 = a.objective_values.iter().zip(weights.iter()).map(|(obj, w)| obj * w).sum();
                        let score_b: f64 = b.objective_values.iter().zip(weights.iter()).map(|(obj, w)| obj * w).sum();
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .ok_or("Failed to select solution")?
            },
            DecisionStrategy::Lexicographic => {
                // Simplified lexicographic ordering - use first objective
                solutions.iter()
                    .max_by(|a, b| a.objective_values[0].partial_cmp(&b.objective_values[0]).unwrap())
                    .ok_or("Failed to select solution")?
            },
            DecisionStrategy::InteractiveMethod => {
                // Simplified - just return first solution
                &solutions[0]
            },
            DecisionStrategy::GoalProgramming => {
                // Simplified goal programming - return solution closest to ideal
                solutions.iter()
                    .min_by(|a, b| {
                        let dist_a: f64 = a.objective_values.iter().map(|obj| (1.0 - obj).powi(2)).sum::<f64>().sqrt();
                        let dist_b: f64 = b.objective_values.iter().map(|obj| (1.0 - obj).powi(2)).sum::<f64>().sqrt();
                        dist_a.partial_cmp(&dist_b).unwrap()
                    })
                    .ok_or("Failed to select solution")?
            },
            DecisionStrategy::TOPSIS => {
                // Simplified TOPSIS
                solutions.iter()
                    .max_by(|a, b| a.crowding_distance.partial_cmp(&b.crowding_distance).unwrap())
                    .ok_or("Failed to select solution")?
            },
        };

        // Record decision
        let decision = Decision {
            timestamp: Instant::now(),
            selected_solution_id: selected_solution.id.clone(),
            strategy_used: self.strategy.clone(),
            preferences_used: preferences.clone(),
            confidence: 0.8,
        };

        self.decision_history.push_back(decision);

        // Limit history size
        while self.decision_history.len() > 100 {
            self.decision_history.pop_front();
        }

        Ok(selected_solution.clone())
    }
}

/// Decision strategies
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionStrategy {
    /// Weighted sum approach
    WeightedSum,
    /// Lexicographic ordering
    Lexicographic,
    /// Interactive methods
    InteractiveMethod,
    /// Goal programming
    GoalProgramming,
    /// TOPSIS
    TOPSIS,
}

/// User preferences
#[derive(Debug, Clone)]
pub struct UserPreferences {
    /// Objective weights
    pub objective_weights: Vec<f64>,
    /// Preference functions
    pub preference_functions: Vec<PreferenceFunction>,
    /// Aspiration levels
    pub aspiration_levels: Vec<f64>,
    /// Reservation levels
    pub reservation_levels: Vec<f64>,
}

/// Preference function
#[derive(Debug, Clone)]
pub struct PreferenceFunction {
    /// Function type
    pub function_type: PreferenceFunctionType,
    /// Parameters
    pub parameters: Vec<f64>,
    /// Objective index
    pub objective_index: usize,
}

/// Types of preference functions
#[derive(Debug, Clone, PartialEq)]
pub enum PreferenceFunctionType {
    /// Linear preference
    Linear,
    /// Piecewise linear
    PiecewiseLinear,
    /// Exponential
    Exponential,
    /// Gaussian
    Gaussian,
    /// Step function
    Step,
}

/// Decision record
#[derive(Debug, Clone)]
pub struct Decision {
    /// Decision timestamp
    pub timestamp: Instant,
    /// Selected solution ID
    pub selected_solution_id: String,
    /// Strategy used
    pub strategy_used: DecisionStrategy,
    /// Preferences used
    pub preferences_used: Option<UserPreferences>,
    /// Decision confidence
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_objective_optimizer_creation() {
        let config = MultiObjectiveConfig::default();
        let optimizer = MultiObjectiveOptimizer::new(config);
        assert!(optimizer.config.enable_multi_objective);
    }

    #[test]
    fn test_pareto_frontier() {
        let config = ParetoFrontierConfig::default();
        let frontier = ParetoFrontier::new(&config);
        assert_eq!(frontier.solutions.len(), 0);
    }

    #[test]
    fn test_scalarizer() {
        let scalarizer = Scalarizer::new(ScalarizationMethod::WeightedSum);
        let objectives = vec![0.8, 0.6, 0.9];
        let weights = vec![0.5, 0.3, 0.2];
        
        let result = scalarizer.scalarize(&objectives, &weights);
        assert!(result.is_ok());
        
        let score = result.unwrap();
        assert!((score - 0.76).abs() < 1e-10); // 0.8*0.5 + 0.6*0.3 + 0.9*0.2 = 0.76
    }

    #[test]
    fn test_decision_maker() {
        let mut decision_maker = DecisionMaker::new();
        
        let solutions = vec![
            MultiObjectiveSolution {
                id: "sol1".to_string(),
                objective_values: vec![0.8, 0.6],
                decision_variables: OptimizationConfiguration {
                    algorithm: AlgorithmType::SimulatedAnnealing,
                    hyperparameters: HashMap::new(),
                    architecture: None,
                    resources: ResourceAllocation {
                        cpu: 1.0,
                        memory: 256,
                        gpu: 0.0,
                        time: std::time::Duration::from_secs(60),
                    },
                },
                dominance_rank: 1,
                crowding_distance: 1.0,
            },
        ];
        
        let result = decision_maker.make_decision(&solutions, None);
        assert!(result.is_ok());
    }
}