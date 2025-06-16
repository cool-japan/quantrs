//! Adaptive optimization framework that automatically adjusts algorithms and parameters.
//!
//! This module provides intelligent optimization that learns from problem structure
//! and solution history to improve performance over time.

use crate::sampler::{SampleResult, Sampler};
use ndarray::Array2;
use rand::{rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Adaptive optimizer that learns and improves
pub struct AdaptiveOptimizer {
    /// Available samplers
    samplers: Vec<(String, Box<dyn Sampler>)>,
    /// Performance history
    performance_history: PerformanceHistory,
    /// Problem analyzer
    problem_analyzer: ProblemAnalyzer,
    /// Strategy selector
    strategy_selector: StrategySelector,
    /// Parameter tuner
    parameter_tuner: ParameterTuner,
    /// Learning rate
    learning_rate: f64,
    /// Exploration vs exploitation
    exploration_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// Problem features to performance mapping
    feature_performance: HashMap<ProblemFeatures, AlgorithmPerformance>,
    /// Recent runs
    recent_runs: VecDeque<RunRecord>,
    /// Best solutions found
    best_solutions: HashMap<String, BestSolution>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProblemFeatures {
    /// Problem size category
    size_category: SizeCategory,
    /// Density category
    density_category: DensityCategory,
    /// Structure type
    structure_type: StructureType,
    /// Constraint complexity
    constraint_complexity: ConstraintComplexity,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SizeCategory {
    Tiny,      // < 10
    Small,     // 10-50
    Medium,    // 50-200
    Large,     // 200-1000
    VeryLarge, // > 1000
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum DensityCategory {
    Sparse, // < 0.1
    Medium, // 0.1-0.5
    Dense,  // > 0.5
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructureType {
    Random,
    Regular,
    Hierarchical,
    Modular,
    Unknown,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintComplexity {
    None,
    Simple,
    Moderate,
    Complex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmPerformance {
    /// Success rate
    success_rate: f64,
    /// Average solution quality
    avg_quality: f64,
    /// Average time
    avg_time_ms: f64,
    /// Number of runs
    n_runs: usize,
    /// Best parameters found
    best_params: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Problem ID
    problem_id: String,
    /// Algorithm used
    algorithm: String,
    /// Parameters used
    parameters: HashMap<String, f64>,
    /// Solution quality
    quality: f64,
    /// Time taken
    time_ms: f64,
    /// Success
    success: bool,
    /// Problem features
    features: ProblemFeatures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestSolution {
    /// Problem ID
    problem_id: String,
    /// Best energy found
    best_energy: f64,
    /// Algorithm that found it
    algorithm: String,
    /// Time to solution
    time_to_solution: f64,
    /// Solution vector
    solution: HashMap<String, bool>,
}

/// Problem analyzer
pub struct ProblemAnalyzer {
    /// Feature extractors
    extractors: Vec<Box<dyn FeatureExtractor>>,
}

trait FeatureExtractor: Send + Sync {
    fn extract(&self, qubo: &Array2<f64>) -> HashMap<String, f64>;
}

/// Strategy selector
pub struct StrategySelector {
    /// Selection strategy
    strategy: SelectionStrategy,
    /// Performance threshold
    performance_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Thompson sampling
    ThompsonSampling,
    /// Upper confidence bound
    UCB { c: f64 },
    /// Epsilon-greedy
    EpsilonGreedy { epsilon: f64 },
    /// Adaptive
    Adaptive,
}

/// Parameter tuner
pub struct ParameterTuner {
    /// Parameter ranges
    param_ranges: HashMap<String, (f64, f64)>,
    /// Tuning method
    tuning_method: TuningMethod,
    /// History
    tuning_history: HashMap<String, Vec<(HashMap<String, f64>, f64)>>,
}

#[derive(Debug, Clone)]
pub enum TuningMethod {
    /// Grid search
    Grid { resolution: usize },
    /// Random search
    Random { n_trials: usize },
    /// Bayesian optimization
    Bayesian,
    /// Evolutionary
    Evolutionary { population_size: usize },
}

impl AdaptiveOptimizer {
    /// Create new adaptive optimizer
    pub fn new() -> Self {
        Self {
            samplers: vec![
                (
                    "SA".to_string(),
                    Box::new(crate::sampler::SASampler::new(None)),
                ),
                (
                    "GA".to_string(),
                    Box::new(crate::sampler::GASampler::new(None)),
                ),
            ],
            performance_history: PerformanceHistory {
                feature_performance: HashMap::new(),
                recent_runs: VecDeque::with_capacity(1000),
                best_solutions: HashMap::new(),
            },
            problem_analyzer: ProblemAnalyzer::new(),
            strategy_selector: StrategySelector::new(SelectionStrategy::Adaptive),
            parameter_tuner: ParameterTuner::new(TuningMethod::Bayesian),
            learning_rate: 0.1,
            exploration_rate: 0.2,
        }
    }

    /// Add sampler
    pub fn add_sampler(&mut self, name: String, sampler: Box<dyn Sampler>) {
        self.samplers.push((name, sampler));
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Optimize adaptively
    pub fn optimize(
        &mut self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
        time_limit: Duration,
    ) -> Result<OptimizationResult, String> {
        let start_time = Instant::now();
        let problem_id = self.generate_problem_id(qubo);

        // Analyze problem
        let features = self.problem_analyzer.analyze(qubo);

        // Select strategy based on history
        let (algorithm, parameters) = self.select_algorithm_and_params(&features)?;

        // Create configured sampler
        let sampler = self.configure_sampler(&algorithm, &parameters)?;

        // Run optimization with monitoring
        let mut best_result: Option<SampleResult> = None;
        let iterations = 0;
        let improvement_history = Vec::new();

        while start_time.elapsed() < time_limit {
            iterations += 1;

            // Adaptive shot count
            let shots = self.calculate_shot_count(&features, start_time.elapsed(), time_limit);

            // Run sampler
            match sampler.run_qubo(&(qubo.clone(), var_map.clone()), shots) {
                Ok(results) => {
                    for result in results {
                        if best_result.is_none()
                            || result.energy < best_result.as_ref().unwrap().energy
                        {
                            improvement_history
                                .push((start_time.elapsed().as_secs_f64(), result.energy));
                            best_result = Some(result);
                        }
                    }
                }
                Err(e) => {
                    // Record failure
                    self.record_run(RunRecord {
                        problem_id: problem_id.clone(),
                        algorithm: algorithm.clone(),
                        parameters: parameters.clone(),
                        quality: f64::INFINITY,
                        time_ms: start_time.elapsed().as_millis() as f64,
                        success: false,
                        features: features.clone(),
                    });

                    return Err(format!("Sampler error: {:?}", e));
                }
            }

            // Check for convergence
            if self.check_convergence(&improvement_history) {
                break;
            }

            // Adaptive parameter adjustment
            if iterations % 10 == 0 {
                self.adjust_parameters(&mut parameters.clone(), &improvement_history);
            }
        }

        let total_time = start_time.elapsed();

        if let Some(best) = best_result {
            // Record successful run
            self.record_run(RunRecord {
                problem_id: problem_id.clone(),
                algorithm: algorithm.clone(),
                parameters,
                quality: best.energy,
                time_ms: total_time.as_millis() as f64,
                success: true,
                features: features.clone(),
            });

            // Update best solution
            self.update_best_solution(
                problem_id.clone(),
                &best,
                &algorithm,
                total_time.as_secs_f64(),
            );

            Ok(OptimizationResult {
                best_solution: best.assignments,
                best_energy: best.energy,
                algorithm_used: algorithm,
                time_taken: total_time,
                iterations,
                improvement_history,
                features,
            })
        } else {
            Err("No solution found".to_string())
        }
    }

    /// Generate problem ID
    fn generate_problem_id(&self, qubo: &Array2<f64>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hasher = DefaultHasher::new();
        qubo.shape().hash(&mut hasher);

        // Sample some elements for hashing
        let n = qubo.shape()[0];
        for i in (0..n).step_by((n / 10).max(1)) {
            for j in (0..n).step_by((n / 10).max(1)) {
                (qubo[[i, j]].to_bits()).hash(&mut hasher);
            }
        }

        format!("prob_{:x}", hasher.finish())
    }

    /// Select algorithm and parameters
    fn select_algorithm_and_params(
        &self,
        features: &ProblemFeatures,
    ) -> Result<(String, HashMap<String, f64>), String> {
        // Get performance history for these features
        let perf = self.performance_history.feature_performance.get(features);

        match &self.strategy_selector.strategy {
            SelectionStrategy::Adaptive => {
                if let Some(perf) = perf {
                    if perf.n_runs > 10 && perf.success_rate > 0.8 {
                        // Exploit: use best known
                        let algorithm = self.get_best_algorithm_for_features(features);
                        let params = perf.best_params.clone();
                        Ok((algorithm, params))
                    } else {
                        // Explore: try different algorithm
                        self.explore_new_algorithm(features)
                    }
                } else {
                    // No history: use heuristics
                    self.select_by_heuristics(features)
                }
            }
            SelectionStrategy::ThompsonSampling => self.thompson_sampling_select(features),
            SelectionStrategy::UCB { c } => self.ucb_select(features, *c),
            SelectionStrategy::EpsilonGreedy { epsilon } => {
                if rand::random::<f64>() < *epsilon {
                    self.random_select()
                } else {
                    self.greedy_select(features)
                }
            }
        }
    }

    /// Select by heuristics
    fn select_by_heuristics(
        &self,
        features: &ProblemFeatures,
    ) -> Result<(String, HashMap<String, f64>), String> {
        let algorithm = match (&features.size_category, &features.density_category) {
            (SizeCategory::Tiny, _) | (SizeCategory::Small, _) => "SA",
            (_, DensityCategory::Sparse) => "GA",
            (SizeCategory::Medium, DensityCategory::Medium) => "SA",
            _ => "GA",
        };

        let params = self.get_default_params(algorithm);
        Ok((algorithm.to_string(), params))
    }

    /// Get default parameters
    fn get_default_params(&self, algorithm: &str) -> HashMap<String, f64> {
        let params = HashMap::new();

        match algorithm {
            "SA" => {
                params.insert("beta_min".to_string(), 0.1);
                params.insert("beta_max".to_string(), 10.0);
                params.insert("sweeps".to_string(), 1000.0);
            }
            "GA" => {
                params.insert("population_size".to_string(), 100.0);
                params.insert("elite_fraction".to_string(), 0.1);
                params.insert("mutation_rate".to_string(), 0.01);
            }
            _ => {}
        }

        params
    }

    /// Configure sampler with parameters
    fn configure_sampler(
        &self,
        algorithm: &str,
        parameters: &HashMap<String, f64>,
    ) -> Result<Box<dyn Sampler>, String> {
        match algorithm {
            "SA" => {
                let sampler = crate::sampler::SASampler::new(None);

                if let Some(&beta_min) = parameters.get("beta_min") {
                    if let Some(&beta_max) = parameters.get("beta_max") {
                        sampler = sampler.with_beta_range(beta_min, beta_max);
                    }
                }

                if let Some(&sweeps) = parameters.get("sweeps") {
                    sampler = sampler.with_sweeps(sweeps as usize);
                }

                Ok(Box::new(sampler))
            }
            "GA" => {
                let sampler = crate::sampler::GASampler::new(None);

                if let Some(&pop_size) = parameters.get("population_size") {
                    sampler = sampler.with_population_size(pop_size as usize);
                }

                if let Some(&elite) = parameters.get("elite_fraction") {
                    sampler = sampler.with_elite_fraction(elite);
                }

                if let Some(&mutation) = parameters.get("mutation_rate") {
                    sampler = sampler.with_mutation_rate(mutation);
                }

                Ok(Box::new(sampler))
            }
            _ => Err(format!("Unknown algorithm: {}", algorithm)),
        }
    }

    /// Calculate adaptive shot count
    fn calculate_shot_count(
        &self,
        features: &ProblemFeatures,
        elapsed: Duration,
        time_limit: Duration,
    ) -> usize {
        let remaining_fraction = 1.0 - (elapsed.as_secs_f64() / time_limit.as_secs_f64());

        let base_shots = match features.size_category {
            SizeCategory::Tiny => 10,
            SizeCategory::Small => 50,
            SizeCategory::Medium => 100,
            SizeCategory::Large => 200,
            SizeCategory::VeryLarge => 500,
        };

        ((base_shots as f64) * remaining_fraction.sqrt()) as usize
    }

    /// Check convergence
    fn check_convergence(&self, history: &[(f64, f64)]) -> bool {
        if history.len() < 10 {
            return false;
        }

        // Check if no improvement in last N iterations
        let recent = &history[history.len() - 10..];
        let best_recent = recent
            .iter()
            .map(|(_, e)| e)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let best_overall = history
            .iter()
            .map(|(_, e)| e)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        (best_recent - best_overall).abs() < 1e-6
    }

    /// Adjust parameters based on history
    fn adjust_parameters(&mut self, params: &mut HashMap<String, f64>, history: &[(f64, f64)]) {
        if history.len() < 2 {
            return;
        }

        // Simple gradient-based adjustment
        let recent_improvement = history.last().unwrap().1 - history[history.len() - 2].1;

        if recent_improvement < 0.0 {
            // Good direction, increase learning
            for (_, value) in params.iter_mut() {
                *value *= 1.0 + self.learning_rate;
            }
        } else {
            // Bad direction, reverse and decrease
            for (_, value) in params.iter_mut() {
                *value *= 1.0 - self.learning_rate * 0.5;
            }
        }
    }

    /// Record run
    fn record_run(&mut self, record: RunRecord) {
        // Update recent runs
        self.performance_history
            .recent_runs
            .push_back(record.clone());
        if self.performance_history.recent_runs.len() > 1000 {
            self.performance_history.recent_runs.pop_front();
        }

        // Update feature performance
        let perf = self
            .performance_history
            .feature_performance
            .entry(record.features.clone())
            .or_insert(AlgorithmPerformance {
                success_rate: 0.0,
                avg_quality: 0.0,
                avg_time_ms: 0.0,
                n_runs: 0,
                best_params: HashMap::new(),
            });

        // Update statistics
        let n = perf.n_runs as f64;
        perf.avg_quality = (perf.avg_quality * n + record.quality) / (n + 1.0);
        perf.avg_time_ms = (perf.avg_time_ms * n + record.time_ms) / (n + 1.0);
        perf.success_rate =
            (perf.success_rate * n + if record.success { 1.0 } else { 0.0 }) / (n + 1.0);
        perf.n_runs += 1;

        // Update best parameters if better
        if record.success && (perf.best_params.is_empty() || record.quality < perf.avg_quality) {
            perf.best_params = record.parameters;
        }
    }

    /// Update best solution
    fn update_best_solution(
        &mut self,
        problem_id: String,
        result: &SampleResult,
        algorithm: &str,
        time: f64,
    ) {
        let entry = self
            .performance_history
            .best_solutions
            .entry(problem_id.clone())
            .or_insert(BestSolution {
                problem_id,
                best_energy: f64::INFINITY,
                algorithm: String::new(),
                time_to_solution: 0.0,
                solution: HashMap::new(),
            });

        if result.energy < entry.best_energy {
            entry.best_energy = result.energy;
            entry.algorithm = algorithm.to_string();
            entry.time_to_solution = time;
            entry.solution = result.assignments.clone();
        }
    }

    // Strategy implementations

    fn get_best_algorithm_for_features(&self, _features: &ProblemFeatures) -> String {
        // Simple implementation - would be more sophisticated in practice
        "SA".to_string()
    }

    fn explore_new_algorithm(
        &self,
        _features: &ProblemFeatures,
    ) -> Result<(String, HashMap<String, f64>), String> {
        let idx = rng().random_range(0..self.samplers.len());
        let algorithm = self.samplers[idx].0.clone();
        let params = self.get_default_params(&algorithm);
        Ok((algorithm, params))
    }

    fn thompson_sampling_select(
        &self,
        _features: &ProblemFeatures,
    ) -> Result<(String, HashMap<String, f64>), String> {
        // Simplified Thompson sampling
        self.random_select()
    }

    fn ucb_select(
        &self,
        features: &ProblemFeatures,
        _c: f64,
    ) -> Result<(String, HashMap<String, f64>), String> {
        // Simplified UCB
        self.select_by_heuristics(features)
    }

    fn random_select(&self) -> Result<(String, HashMap<String, f64>), String> {
        let idx = rng().random_range(0..self.samplers.len());
        let algorithm = self.samplers[idx].0.clone();
        let params = self.get_default_params(&algorithm);
        Ok((algorithm, params))
    }

    fn greedy_select(
        &self,
        features: &ProblemFeatures,
    ) -> Result<(String, HashMap<String, f64>), String> {
        if let Some(perf) = self.performance_history.feature_performance.get(features) {
            let algorithm = self.get_best_algorithm_for_features(features);
            Ok((algorithm, perf.best_params.clone()))
        } else {
            self.select_by_heuristics(features)
        }
    }

    /// Save history to file
    pub fn save_history(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.performance_history)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load history from file
    pub fn load_history(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        self.performance_history = serde_json::from_str(&json)?;
        Ok(())
    }
}

impl ProblemAnalyzer {
    /// Create new analyzer
    fn new() -> Self {
        Self {
            extractors: vec![
                Box::new(BasicFeatureExtractor),
                Box::new(StructureFeatureExtractor),
            ],
        }
    }

    /// Analyze problem
    fn analyze(&self, qubo: &Array2<f64>) -> ProblemFeatures {
        let n = qubo.shape()[0];

        // Extract all features
        let all_features = HashMap::new();
        for extractor in &self.extractors {
            all_features.extend(extractor.extract(qubo));
        }

        // Categorize
        let size_category = match n {
            0..=9 => SizeCategory::Tiny,
            10..=49 => SizeCategory::Small,
            50..=199 => SizeCategory::Medium,
            200..=999 => SizeCategory::Large,
            _ => SizeCategory::VeryLarge,
        };

        let density = all_features.get("density").copied().unwrap_or(0.5);
        let density_category = match density {
            d if d < 0.1 => DensityCategory::Sparse,
            d if d < 0.5 => DensityCategory::Medium,
            _ => DensityCategory::Dense,
        };

        let structure_score = all_features.get("structure_score").copied().unwrap_or(0.0);
        let structure_type = if structure_score < 0.2 {
            StructureType::Random
        } else if structure_score < 0.5 {
            StructureType::Regular
        } else {
            StructureType::Hierarchical
        };

        ProblemFeatures {
            size_category,
            density_category,
            structure_type,
            constraint_complexity: ConstraintComplexity::None, // Would analyze constraints
        }
    }
}

struct BasicFeatureExtractor;

impl FeatureExtractor for BasicFeatureExtractor {
    fn extract(&self, qubo: &Array2<f64>) -> HashMap<String, f64> {
        let n = qubo.shape()[0];
        let features = HashMap::new();

        // Size
        features.insert("size".to_string(), n as f64);

        // Density
        let non_zeros = qubo.iter().filter(|&&x| x.abs() > 1e-10).count();
        features.insert("density".to_string(), non_zeros as f64 / (n * n) as f64);

        // Statistics
        let values: Vec<f64> = qubo.iter().copied().collect();
        features.insert(
            "mean".to_string(),
            values.iter().sum::<f64>() / values.len() as f64,
        );
        features.insert(
            "max".to_string(),
            values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        );
        features.insert(
            "min".to_string(),
            values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        );

        features
    }
}

struct StructureFeatureExtractor;

impl FeatureExtractor for StructureFeatureExtractor {
    fn extract(&self, qubo: &Array2<f64>) -> HashMap<String, f64> {
        let features = HashMap::new();
        let n = qubo.shape()[0];

        // Diagonal dominance
        let diag_sum = 0.0;
        let total_sum = 0.0;
        for i in 0..n {
            diag_sum += qubo[[i, i]].abs();
            for j in 0..n {
                total_sum += qubo[[i, j]].abs();
            }
        }

        features.insert(
            "diagonal_dominance".to_string(),
            if total_sum > 0.0 {
                diag_sum / total_sum
            } else {
                0.0
            },
        );

        // Symmetry
        let symmetry_score = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                let diff = (qubo[[i, j]] - qubo[[j, i]]).abs();
                let avg = (qubo[[i, j]].abs() + qubo[[j, i]].abs()) / 2.0;
                if avg > 1e-10 {
                    symmetry_score += 1.0 - diff / avg;
                }
            }
        }
        features.insert(
            "symmetry".to_string(),
            symmetry_score / ((n * (n - 1)) / 2) as f64,
        );

        // Structure score (combination)
        let structure_score = features.get("diagonal_dominance").unwrap() * 0.5
            + features.get("symmetry").unwrap() * 0.5;
        features.insert("structure_score".to_string(), structure_score);

        features
    }
}

impl StrategySelector {
    fn new(strategy: SelectionStrategy) -> Self {
        Self {
            strategy,
            performance_threshold: 0.8,
        }
    }
}

impl ParameterTuner {
    fn new(method: TuningMethod) -> Self {
        Self {
            param_ranges: HashMap::new(),
            tuning_method: method,
            tuning_history: HashMap::new(),
        }
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best solution found
    pub best_solution: HashMap<String, bool>,
    /// Best energy
    pub best_energy: f64,
    /// Algorithm used
    pub algorithm_used: String,
    /// Time taken
    pub time_taken: Duration,
    /// Iterations
    pub iterations: usize,
    /// Improvement history
    pub improvement_history: Vec<(f64, f64)>,
    /// Problem features
    pub features: ProblemFeatures,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adaptive_optimizer() {
        let optimizer = AdaptiveOptimizer::new();

        let qubo = array![[0.0, -1.0, 0.5], [-1.0, 0.0, -0.5], [0.5, -0.5, 0.0]];

        let var_map = HashMap::new();
        var_map.insert("x".to_string(), 0);
        var_map.insert("y".to_string(), 1);
        var_map.insert("z".to_string(), 2);

        let result = optimizer
            .optimize(&qubo, &var_map, Duration::from_secs(1))
            .unwrap();

        assert!(!result.best_solution.is_empty());
        assert!(result.best_energy < 0.0);
        assert!(!result.improvement_history.is_empty());
    }

    #[test]
    #[ignore]
    fn test_problem_analyzer() {
        let analyzer = ProblemAnalyzer::new();

        let small_sparse = Array2::eye(10);
        let features = analyzer.analyze(&small_sparse);
        assert_eq!(features.size_category, SizeCategory::Small);
        assert_eq!(features.density_category, DensityCategory::Sparse);

        let large_dense = Array2::ones((500, 500));
        let features = analyzer.analyze(&large_dense);
        assert_eq!(features.size_category, SizeCategory::Large);
        assert_eq!(features.density_category, DensityCategory::Dense);
    }
}
