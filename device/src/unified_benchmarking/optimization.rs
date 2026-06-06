//! Optimization engine for performance and cost optimization

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;

use super::results::{OptimizationResult, UnifiedBenchmarkResult};

/// Optimization engine for performance and cost optimization
pub struct OptimizationEngine {
    objective_functions: HashMap<String, Box<dyn Fn(&UnifiedBenchmarkResult) -> f64 + Send + Sync>>,
    optimization_history: VecDeque<OptimizationResult>,
    current_strategy: OptimizationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStrategy {
    pub strategy_name: String,
    pub parameters: HashMap<String, f64>,
    pub last_updated: SystemTime,
    pub effectiveness_score: f64,
}

impl Default for OptimizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationEngine {
    pub fn new() -> Self {
        Self {
            objective_functions: HashMap::new(),
            optimization_history: VecDeque::new(),
            current_strategy: OptimizationStrategy {
                strategy_name: "default".to_string(),
                parameters: HashMap::new(),
                last_updated: SystemTime::now(),
                effectiveness_score: 0.0,
            },
        }
    }

    /// Register an objective function that maps a benchmark result to a scalar score.
    ///
    /// The function should return larger values for *better* configurations.
    pub fn register_objective<F>(&mut self, name: impl Into<String>, f: F)
    where
        F: Fn(&UnifiedBenchmarkResult) -> f64 + Send + Sync + 'static,
    {
        self.objective_functions.insert(name.into(), Box::new(f));
    }

    /// Run one optimisation cycle over the supplied benchmark result.
    ///
    /// For each registered objective function the engine evaluates the current
    /// result, compares the score against the best score seen in the history,
    /// and records an `OptimizationResult`.  The overall strategy effectiveness
    /// is updated as the mean improvement across all objectives.
    ///
    /// Returns the list of `OptimizationResult` values produced this cycle.
    pub fn optimize(&mut self, result: &UnifiedBenchmarkResult) -> Vec<OptimizationResult> {
        let mut new_results: Vec<OptimizationResult> = Vec::new();

        let start = std::time::Instant::now();

        for (name, func) in &self.objective_functions {
            let score = func(result);

            // Find the best historical score for this objective.
            let best_historical = self
                .optimization_history
                .iter()
                .filter(|r| r.objective_function == *name)
                .map(|r| r.optimal_value)
                .fold(f64::NEG_INFINITY, f64::max);

            let improvement = if best_historical.is_finite() {
                score - best_historical
            } else {
                0.0
            };

            let opt_result = OptimizationResult {
                objective_function: name.clone(),
                optimal_solution: vec![score],
                optimal_value: score,
                convergence_history: vec![score],
                optimization_time: start.elapsed(),
            };
            new_results.push(opt_result.clone());
            self.optimization_history.push_back(opt_result);
        }

        // Keep history bounded.
        while self.optimization_history.len() > 10_000 {
            self.optimization_history.pop_front();
        }

        // Update effectiveness score: mean improvement this cycle.
        let n = new_results.len();
        if n > 0 {
            let total_improvement: f64 = new_results.iter().map(|r| r.optimal_value).sum();
            self.current_strategy.effectiveness_score = total_improvement / n as f64;
            self.current_strategy.last_updated = SystemTime::now();
        }

        new_results
    }

    /// Return a summary of the current optimisation strategy.
    pub fn current_strategy(&self) -> &OptimizationStrategy {
        &self.current_strategy
    }

    /// Return the full optimisation history.
    pub fn history(&self) -> &VecDeque<OptimizationResult> {
        &self.optimization_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_engine_has_empty_history() {
        let engine = OptimizationEngine::new();
        assert!(
            engine.history().is_empty(),
            "new engine must have empty history"
        );
    }

    #[test]
    fn test_default_engine_matches_new() {
        let e1 = OptimizationEngine::new();
        let e2 = OptimizationEngine::default();
        assert_eq!(
            e1.current_strategy().strategy_name,
            e2.current_strategy().strategy_name,
        );
    }

    #[test]
    fn test_register_objective_increases_objective_count() {
        let mut engine = OptimizationEngine::new();
        assert_eq!(engine.objective_functions.len(), 0);
        engine.register_objective("test_obj", |_| 1.0);
        assert_eq!(engine.objective_functions.len(), 1);
    }

    #[test]
    fn test_current_strategy_initial_effectiveness() {
        let engine = OptimizationEngine::new();
        assert_eq!(
            engine.current_strategy().effectiveness_score,
            0.0,
            "initial effectiveness_score must be 0"
        );
    }
}
