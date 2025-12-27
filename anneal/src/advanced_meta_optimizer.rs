//! Advanced Meta-Learning Optimization System
//!
//! This module implements a sophisticated meta-learning system that learns from
//! optimization history to improve future optimization runs. It includes:
//! - Optimization history analysis and pattern recognition
//! - Transfer learning between similar problems
//! - Adaptive strategy selection based on problem characteristics
//! - Performance prediction using learned models

use crate::applications::{ApplicationError, ApplicationResult};
use crate::ising::IsingModel;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::ChaCha8Rng;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// Re-export for convenience
pub use crate::ising::Coupling;

/// Problem features extracted for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemFeatures {
    /// Number of variables
    pub num_variables: usize,
    /// Problem density (ratio of non-zero couplings)
    pub density: f64,
    /// Coupling strength statistics
    pub coupling_mean: f64,
    pub coupling_std: f64,
    pub coupling_max: f64,
    /// Bias statistics
    pub bias_mean: f64,
    pub bias_std: f64,
    /// Graph properties
    pub average_degree: f64,
    pub max_degree: usize,
    pub clustering_coefficient: f64,
    /// Energy landscape properties
    pub estimated_barriers: f64,
    pub frustration_index: f64,
    /// Problem symmetry
    pub symmetry_score: f64,
}

/// Optimization run record for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecord {
    /// Problem features
    pub features: ProblemFeatures,
    /// Strategy used
    pub strategy: OptimizationStrategy,
    /// Parameters used
    pub parameters: HashMap<String, f64>,
    /// Performance metrics
    pub best_energy: f64,
    pub convergence_time: Duration,
    pub iterations_to_converge: usize,
    pub success_rate: f64,
    /// Resource usage
    pub cpu_time: Duration,
    pub memory_peak_mb: f64,
    /// Timestamp (not serialized, defaults to now)
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

/// Available optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    ClassicalAnnealing,
    QuantumAnnealing,
    PopulationAnnealing,
    CoherentIsingMachine,
    QuantumWalk,
    HybridQCML,
    AdaptiveSchedule,
    ReversedAnnealing,
}

/// Performance prediction model
#[derive(Debug, Clone)]
pub struct PerformancePredictor {
    /// Feature weights learned from history
    feature_weights: Array1<f64>,
    /// Strategy-specific adjustments
    strategy_adjustments: HashMap<OptimizationStrategy, f64>,
    /// Prediction confidence
    confidence: f64,
}

impl PerformancePredictor {
    /// Create a new predictor
    #[must_use]
    pub fn new(num_features: usize) -> Self {
        Self {
            feature_weights: Array1::zeros(num_features),
            strategy_adjustments: HashMap::new(),
            confidence: 0.0,
        }
    }

    /// Predict performance for given features and strategy
    #[must_use]
    pub fn predict(
        &self,
        features: &ProblemFeatures,
        strategy: OptimizationStrategy,
    ) -> PredictedPerformance {
        let feature_vec = self.features_to_vector(features);
        let base_score = feature_vec.dot(&self.feature_weights);

        let strategy_adj = self
            .strategy_adjustments
            .get(&strategy)
            .copied()
            .unwrap_or(0.0);

        let predicted_quality = (base_score + strategy_adj).tanh();
        let predicted_time = self.estimate_time(features, strategy);

        PredictedPerformance {
            strategy,
            quality_score: predicted_quality,
            estimated_time: predicted_time,
            confidence: self.confidence,
        }
    }

    /// Convert features to vector for prediction
    fn features_to_vector(&self, features: &ProblemFeatures) -> Array1<f64> {
        Array1::from_vec(vec![
            features.num_variables as f64,
            features.density,
            features.coupling_mean,
            features.coupling_std,
            features.average_degree,
            features.clustering_coefficient,
            features.frustration_index,
            features.symmetry_score,
        ])
    }

    /// Estimate execution time based on problem size and strategy
    fn estimate_time(
        &self,
        features: &ProblemFeatures,
        strategy: OptimizationStrategy,
    ) -> Duration {
        let base_complexity = match strategy {
            OptimizationStrategy::ClassicalAnnealing => features.num_variables as f64,
            OptimizationStrategy::QuantumAnnealing => (features.num_variables as f64).powf(1.5),
            OptimizationStrategy::PopulationAnnealing => {
                features.num_variables as f64 * features.density
            }
            OptimizationStrategy::CoherentIsingMachine => (features.num_variables as f64).powi(2),
            OptimizationStrategy::QuantumWalk => {
                features.num_variables as f64 * features.average_degree
            }
            OptimizationStrategy::HybridQCML => features.num_variables as f64 * 1.5,
            OptimizationStrategy::AdaptiveSchedule => features.num_variables as f64 * 0.8,
            OptimizationStrategy::ReversedAnnealing => features.num_variables as f64 * 0.6,
        };

        // Scale by problem difficulty
        let difficulty_factor = 1.0 + features.frustration_index + (1.0 - features.symmetry_score);
        let estimated_ms = base_complexity * difficulty_factor * 10.0;

        Duration::from_millis(estimated_ms as u64)
    }

    /// Update predictor with new observation
    pub fn update(&mut self, record: &OptimizationRecord, learning_rate: f64) {
        let feature_vec = self.features_to_vector(&record.features);

        // Simple gradient update (could be replaced with more sophisticated learning)
        let predicted = self.predict(&record.features, record.strategy);
        let error = record.success_rate - predicted.quality_score;

        for i in 0..self.feature_weights.len() {
            self.feature_weights[i] += learning_rate * error * feature_vec[i];
        }

        // Update strategy adjustment
        let current_adj = self
            .strategy_adjustments
            .entry(record.strategy)
            .or_insert(0.0);
        *current_adj += learning_rate * error * 0.1;

        // Update confidence based on error
        self.confidence = 0.9f64.mul_add(self.confidence, 0.1 * (1.0 - error.abs()));
    }
}

/// Predicted performance metrics
#[derive(Debug, Clone)]
pub struct PredictedPerformance {
    pub strategy: OptimizationStrategy,
    pub quality_score: f64,
    pub estimated_time: Duration,
    pub confidence: f64,
}

/// Transfer learning engine for cross-problem knowledge transfer
#[derive(Debug, Clone)]
pub struct TransferLearningEngine {
    /// Source domain knowledge
    source_records: Vec<OptimizationRecord>,
    /// Similarity metrics between problems
    similarity_cache: HashMap<(usize, usize), f64>,
    /// Transfer weights
    transfer_weights: Vec<f64>,
}

impl TransferLearningEngine {
    /// Create new transfer learning engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            source_records: Vec::new(),
            similarity_cache: HashMap::new(),
            transfer_weights: Vec::new(),
        }
    }

    /// Add source domain knowledge
    pub fn add_source_knowledge(&mut self, records: Vec<OptimizationRecord>) {
        self.source_records.extend(records);
        self.similarity_cache.clear(); // Invalidate cache
    }

    /// Compute similarity between two problems
    #[must_use]
    pub fn compute_similarity(
        &self,
        features1: &ProblemFeatures,
        features2: &ProblemFeatures,
    ) -> f64 {
        // Euclidean distance in normalized feature space
        let diff_size = ((features1.num_variables as f64 - features2.num_variables as f64)
            / features1.num_variables.max(features2.num_variables) as f64)
            .powi(2);
        let diff_density = (features1.density - features2.density).powi(2);
        let diff_frustration = (features1.frustration_index - features2.frustration_index).powi(2);
        let diff_symmetry = (features1.symmetry_score - features2.symmetry_score).powi(2);

        let distance = (diff_size + diff_density + diff_frustration + diff_symmetry).sqrt();

        // Convert distance to similarity (Gaussian kernel)
        let bandwidth: f64 = 0.5;
        (-distance.powi(2) / (2.0 * bandwidth.powi(2))).exp()
    }

    /// Transfer knowledge to target problem
    pub fn transfer_knowledge(
        &self,
        target_features: &ProblemFeatures,
    ) -> Vec<(OptimizationStrategy, f64)> {
        let mut strategy_scores: HashMap<OptimizationStrategy, Vec<f64>> = HashMap::new();

        for record in &self.source_records {
            let similarity = self.compute_similarity(target_features, &record.features);

            // Only transfer from sufficiently similar problems
            if similarity > 0.3 {
                let weighted_score = record.success_rate * similarity;
                strategy_scores
                    .entry(record.strategy)
                    .or_insert_with(Vec::new)
                    .push(weighted_score);
            }
        }

        // Aggregate scores for each strategy
        let mut recommendations: Vec<(OptimizationStrategy, f64)> = strategy_scores
            .into_iter()
            .map(|(strategy, scores)| {
                let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
                (strategy, avg_score)
            })
            .collect();

        // Sort by score descending
        recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        recommendations
    }
}

impl Default for TransferLearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive strategy selector
#[derive(Debug)]
pub struct AdaptiveStrategySelector {
    /// Performance predictor
    predictor: PerformancePredictor,
    /// Transfer learning engine
    transfer_engine: TransferLearningEngine,
    /// Strategy exploration rate
    exploration_rate: f64,
    /// Random number generator
    rng: ChaCha8Rng,
}

impl AdaptiveStrategySelector {
    /// Create new adaptive selector
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            predictor: PerformancePredictor::new(8),
            transfer_engine: TransferLearningEngine::new(),
            exploration_rate: 0.1,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Select best strategy for given problem
    pub fn select_strategy(
        &mut self,
        features: &ProblemFeatures,
        available_strategies: &[OptimizationStrategy],
    ) -> OptimizationStrategy {
        // Exploration: randomly select strategy
        if self.rng.gen::<f64>() < self.exploration_rate {
            return *available_strategies
                .get(self.rng.gen_range(0..available_strategies.len()))
                .expect("random index should be within bounds");
        }

        // Exploitation: use learned knowledge
        let mut best_strategy = available_strategies[0];
        let mut best_score = f64::NEG_INFINITY;

        for &strategy in available_strategies {
            let prediction = self.predictor.predict(features, strategy);
            let transfer_bonus = self.get_transfer_bonus(features, strategy);
            let total_score = 0.3f64.mul_add(transfer_bonus, prediction.quality_score);

            if total_score > best_score {
                best_score = total_score;
                best_strategy = strategy;
            }
        }

        best_strategy
    }

    /// Get transfer learning bonus for strategy
    fn get_transfer_bonus(
        &self,
        features: &ProblemFeatures,
        strategy: OptimizationStrategy,
    ) -> f64 {
        let recommendations = self.transfer_engine.transfer_knowledge(features);

        recommendations
            .iter()
            .find(|(s, _)| *s == strategy)
            .map_or(0.0, |(_, score)| *score)
    }

    /// Update selector with new observation
    pub fn update(&mut self, record: OptimizationRecord) {
        self.predictor.update(&record, 0.01);
        self.transfer_engine.add_source_knowledge(vec![record]);

        // Decay exploration rate
        self.exploration_rate *= 0.999;
    }

    /// Get performance prediction for strategy
    #[must_use]
    pub fn predict_performance(
        &self,
        features: &ProblemFeatures,
        strategy: OptimizationStrategy,
    ) -> PredictedPerformance {
        self.predictor.predict(features, strategy)
    }
}

/// Meta-learning optimizer that learns from optimization history
#[derive(Debug)]
pub struct MetaLearningOptimizer {
    /// Optimization history
    history: VecDeque<OptimizationRecord>,
    /// Maximum history size
    max_history: usize,
    /// Strategy selector
    selector: AdaptiveStrategySelector,
    /// Statistics
    pub total_optimizations: usize,
    pub average_success_rate: f64,
}

impl MetaLearningOptimizer {
    /// Create new meta-learning optimizer
    #[must_use]
    pub fn new(max_history: usize, seed: u64) -> Self {
        Self {
            history: VecDeque::new(),
            max_history,
            selector: AdaptiveStrategySelector::new(seed),
            total_optimizations: 0,
            average_success_rate: 0.0,
        }
    }

    /// Extract features from Ising model
    pub fn extract_features(&self, model: &IsingModel) -> ProblemFeatures {
        let num_variables = model.num_qubits;

        // Calculate coupling statistics using public API
        let couplings = model.couplings();
        let mut coupling_values = Vec::new();
        let mut degrees = vec![0; num_variables];

        for coupling in &couplings {
            coupling_values.push(coupling.strength.abs());
            degrees[coupling.i] += 1;
            degrees[coupling.j] += 1;
        }

        let density = couplings.len() as f64 / (num_variables * (num_variables - 1) / 2) as f64;

        let coupling_mean = if coupling_values.is_empty() {
            0.0
        } else {
            coupling_values.iter().sum::<f64>() / coupling_values.len() as f64
        };

        let coupling_std = if coupling_values.len() > 1 {
            let variance = coupling_values
                .iter()
                .map(|v| (v - coupling_mean).powi(2))
                .sum::<f64>()
                / coupling_values.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let coupling_max = coupling_values.iter().copied().fold(0.0, f64::max);

        // Bias statistics using public API
        let biases = model.biases();
        let bias_values: Vec<f64> = biases.iter().map(|(_, b)| *b).collect();
        let bias_mean = if bias_values.is_empty() {
            0.0
        } else {
            bias_values.iter().sum::<f64>() / bias_values.len() as f64
        };

        let bias_std = if bias_values.len() > 1 {
            let variance = bias_values
                .iter()
                .map(|v| (v - bias_mean).powi(2))
                .sum::<f64>()
                / bias_values.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Graph properties
        let average_degree = if degrees.is_empty() {
            0.0
        } else {
            degrees.iter().sum::<usize>() as f64 / degrees.len() as f64
        };

        let max_degree = degrees.iter().copied().max().unwrap_or(0);

        // Simple clustering coefficient estimate
        let clustering_coefficient = self.estimate_clustering(model);

        // Energy landscape properties (simplified)
        let estimated_barriers = coupling_std / (1.0 + density);
        let frustration_index = self.estimate_frustration(model);

        // Symmetry (simplified - based on bias uniformity)
        let symmetry_score = 1.0 - (bias_std / (1.0 + bias_mean.abs()));

        ProblemFeatures {
            num_variables,
            density,
            coupling_mean,
            coupling_std,
            coupling_max,
            bias_mean,
            bias_std,
            average_degree,
            max_degree,
            clustering_coefficient,
            estimated_barriers,
            frustration_index,
            symmetry_score: symmetry_score.clamp(0.0, 1.0),
        }
    }

    /// Estimate clustering coefficient
    fn estimate_clustering(&self, model: &IsingModel) -> f64 {
        // Simplified clustering coefficient calculation using public API
        let couplings = model.couplings();

        // Build adjacency map
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for coupling in &couplings {
            adj.entry(coupling.i)
                .or_insert_with(Vec::new)
                .push(coupling.j);
            adj.entry(coupling.j)
                .or_insert_with(Vec::new)
                .push(coupling.i);
        }

        let mut triangles = 0;
        let mut triples = 0;

        for i in 0..model.num_qubits {
            if let Some(neighbors) = adj.get(&i) {
                for k in 0..neighbors.len() {
                    for l in (k + 1)..neighbors.len() {
                        triples += 1;
                        let j1 = neighbors[k];
                        let j2 = neighbors[l];

                        // Check if j1 and j2 are connected
                        if let Some(j1_neighbors) = adj.get(&j1) {
                            if j1_neighbors.contains(&j2) {
                                triangles += 1;
                            }
                        }
                    }
                }
            }
        }

        if triples > 0 {
            f64::from(triangles) / f64::from(triples)
        } else {
            0.0
        }
    }

    /// Estimate frustration index
    fn estimate_frustration(&self, model: &IsingModel) -> f64 {
        // Count frustrated interactions (antiferromagnetic couplings) using public API
        let couplings = model.couplings();
        let mut frustrated = 0;
        let total = couplings.len();

        for coupling in &couplings {
            if coupling.strength > 0.0 {
                // Antiferromagnetic
                frustrated += 1;
            }
        }

        if total > 0 {
            f64::from(frustrated) / total as f64
        } else {
            0.0
        }
    }

    /// Select best strategy for problem
    pub fn select_strategy(&mut self, model: &IsingModel) -> OptimizationStrategy {
        let features = self.extract_features(model);
        let available = vec![
            OptimizationStrategy::ClassicalAnnealing,
            OptimizationStrategy::QuantumAnnealing,
            OptimizationStrategy::PopulationAnnealing,
            OptimizationStrategy::AdaptiveSchedule,
        ];

        self.selector.select_strategy(&features, &available)
    }

    /// Record optimization result
    pub fn record_optimization(&mut self, record: OptimizationRecord) {
        self.total_optimizations += 1;

        // Update running average
        self.average_success_rate = self
            .average_success_rate
            .mul_add((self.total_optimizations - 1) as f64, record.success_rate)
            / self.total_optimizations as f64;

        // Update selector
        self.selector.update(record.clone());

        // Add to history
        self.history.push_back(record);

        // Limit history size
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Get recommended strategies for problem
    pub fn recommend_strategies(
        &mut self,
        model: &IsingModel,
        top_k: usize,
    ) -> Vec<(OptimizationStrategy, PredictedPerformance)> {
        let features = self.extract_features(model);
        let strategies = vec![
            OptimizationStrategy::ClassicalAnnealing,
            OptimizationStrategy::QuantumAnnealing,
            OptimizationStrategy::PopulationAnnealing,
            OptimizationStrategy::CoherentIsingMachine,
            OptimizationStrategy::QuantumWalk,
            OptimizationStrategy::HybridQCML,
            OptimizationStrategy::AdaptiveSchedule,
            OptimizationStrategy::ReversedAnnealing,
        ];

        let mut recommendations: Vec<_> = strategies
            .iter()
            .map(|&strategy| {
                let prediction = self.selector.predict_performance(&features, strategy);
                (strategy, prediction)
            })
            .collect();

        // Sort by quality score
        recommendations.sort_by(|a, b| {
            b.1.quality_score
                .partial_cmp(&a.1.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        recommendations.into_iter().take(top_k).collect()
    }

    /// Get optimization statistics
    #[must_use]
    pub fn get_statistics(&self) -> MetaLearningStatistics {
        MetaLearningStatistics {
            total_optimizations: self.total_optimizations,
            average_success_rate: self.average_success_rate,
            history_size: self.history.len(),
            exploration_rate: self.selector.exploration_rate,
        }
    }
}

/// Meta-learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningStatistics {
    pub total_optimizations: usize,
    pub average_success_rate: f64,
    pub history_size: usize,
    pub exploration_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learning_optimizer_creation() {
        let optimizer = MetaLearningOptimizer::new(100, 42);
        assert_eq!(optimizer.total_optimizations, 0);
        assert_eq!(optimizer.average_success_rate, 0.0);
    }

    #[test]
    fn test_feature_extraction() {
        let optimizer = MetaLearningOptimizer::new(100, 42);
        let mut model = IsingModel::new(5);
        model.set_coupling(0, 1, -1.0).expect("should set coupling");
        model.set_coupling(1, 2, -1.0).expect("should set coupling");
        model.set_bias(0, 0.5).expect("should set bias");

        let features = optimizer.extract_features(&model);
        assert_eq!(features.num_variables, 5);
        assert!(features.density > 0.0);
    }

    #[test]
    fn test_strategy_selection() {
        let mut optimizer = MetaLearningOptimizer::new(100, 42);
        let mut model = IsingModel::new(10);
        model.set_coupling(0, 1, -1.0).expect("should set coupling");

        let strategy = optimizer.select_strategy(&model);
        assert!(matches!(
            strategy,
            OptimizationStrategy::ClassicalAnnealing
                | OptimizationStrategy::QuantumAnnealing
                | OptimizationStrategy::PopulationAnnealing
                | OptimizationStrategy::AdaptiveSchedule
        ));
    }

    #[test]
    fn test_performance_predictor() {
        let predictor = PerformancePredictor::new(8);
        let features = ProblemFeatures {
            num_variables: 20,
            density: 0.3,
            coupling_mean: 1.0,
            coupling_std: 0.5,
            coupling_max: 2.0,
            bias_mean: 0.0,
            bias_std: 0.2,
            average_degree: 6.0,
            max_degree: 10,
            clustering_coefficient: 0.3,
            estimated_barriers: 0.5,
            frustration_index: 0.4,
            symmetry_score: 0.7,
        };

        let prediction = predictor.predict(&features, OptimizationStrategy::ClassicalAnnealing);
        assert!(prediction.quality_score.abs() <= 1.0);
        assert!(prediction.estimated_time.as_millis() > 0);
    }

    #[test]
    fn test_transfer_learning() {
        let mut engine = TransferLearningEngine::new();

        let features1 = ProblemFeatures {
            num_variables: 20,
            density: 0.3,
            coupling_mean: 1.0,
            coupling_std: 0.5,
            coupling_max: 2.0,
            bias_mean: 0.0,
            bias_std: 0.2,
            average_degree: 6.0,
            max_degree: 10,
            clustering_coefficient: 0.3,
            estimated_barriers: 0.5,
            frustration_index: 0.4,
            symmetry_score: 0.7,
        };

        let features2 = features1.clone();

        let similarity = engine.compute_similarity(&features1, &features2);
        assert!((similarity - 1.0).abs() < 0.01); // Should be very similar to itself
    }

    #[test]
    fn test_record_optimization() {
        let mut optimizer = MetaLearningOptimizer::new(100, 42);

        let features = ProblemFeatures {
            num_variables: 10,
            density: 0.2,
            coupling_mean: 1.0,
            coupling_std: 0.3,
            coupling_max: 1.5,
            bias_mean: 0.0,
            bias_std: 0.1,
            average_degree: 4.0,
            max_degree: 8,
            clustering_coefficient: 0.2,
            estimated_barriers: 0.3,
            frustration_index: 0.3,
            symmetry_score: 0.8,
        };

        let record = OptimizationRecord {
            features,
            strategy: OptimizationStrategy::ClassicalAnnealing,
            parameters: HashMap::new(),
            best_energy: -10.0,
            convergence_time: Duration::from_secs(1),
            iterations_to_converge: 100,
            success_rate: 0.95,
            cpu_time: Duration::from_secs(1),
            memory_peak_mb: 50.0,
            timestamp: Instant::now(),
        };

        optimizer.record_optimization(record);
        assert_eq!(optimizer.total_optimizations, 1);
        assert_eq!(optimizer.average_success_rate, 0.95);
    }

    #[test]
    fn test_recommend_strategies() {
        let mut optimizer = MetaLearningOptimizer::new(100, 42);
        let mut model = IsingModel::new(15);
        model.set_coupling(0, 1, -1.0).expect("should set coupling");
        model.set_coupling(1, 2, 1.0).expect("should set coupling");

        let recommendations = optimizer.recommend_strategies(&model, 3);
        assert_eq!(recommendations.len(), 3);

        // Should be sorted by quality
        if recommendations.len() >= 2 {
            assert!(recommendations[0].1.quality_score >= recommendations[1].1.quality_score);
        }
    }
}
