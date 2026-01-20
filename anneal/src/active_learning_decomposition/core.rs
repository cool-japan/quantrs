//! Core active learning decomposer implementation

use scirs2_core::ndarray::Array1;
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::{
    ActiveLearningConfig, BoundaryEdge, DecompositionKnowledgeBase, DecompositionMetadata,
    DecompositionResult, DecompositionStrategy, DecompositionStrategyLearner, EvaluationMetric,
    PerformanceEvaluator, PerformanceRecord, ProblemAnalysis, ProblemAnalyzer, QueryStrategy,
    Subproblem, SubproblemGenerator, SubproblemMetadata,
};
use crate::ising::IsingModel;

/// Active learning decomposer for optimization problems
#[derive(Debug, Clone)]
pub struct ActiveLearningDecomposer {
    /// Decomposition strategy learner
    pub strategy_learner: DecompositionStrategyLearner,
    /// Problem analyzer
    pub problem_analyzer: ProblemAnalyzer,
    /// Subproblem generator
    pub subproblem_generator: SubproblemGenerator,
    /// Performance evaluator
    pub performance_evaluator: PerformanceEvaluator,
    /// Knowledge base
    pub knowledge_base: DecompositionKnowledgeBase,
    /// Configuration
    pub config: ActiveLearningConfig,
}

impl ActiveLearningDecomposer {
    /// Create new active learning decomposer
    pub fn new(config: ActiveLearningConfig) -> Result<Self, String> {
        let strategy_learner = DecompositionStrategyLearner::new()?;
        let problem_analyzer = ProblemAnalyzer::new()?;
        let subproblem_generator = SubproblemGenerator::new()?;
        let performance_evaluator = PerformanceEvaluator::new()?;
        let knowledge_base = DecompositionKnowledgeBase::new()?;

        Ok(Self {
            strategy_learner,
            problem_analyzer,
            subproblem_generator,
            performance_evaluator,
            knowledge_base,
            config,
        })
    }

    /// Decompose problem using active learning
    pub fn decompose_problem(
        &mut self,
        problem: &IsingModel,
    ) -> Result<DecompositionResult, String> {
        // Analyze problem structure and characteristics
        let problem_analysis = self.analyze_problem(problem)?;

        // Select decomposition strategy using active learning
        let strategy = self.select_strategy(problem, &problem_analysis)?;

        // Generate subproblems
        let subproblems = self.generate_subproblems(problem, &strategy, &problem_analysis)?;

        // Validate decomposition quality
        let quality_score = self.validate_decomposition_quality(&subproblems, problem)?;

        // Update knowledge base if learning is enabled
        if self.config.enable_online_learning {
            self.update_knowledge_base(problem, &strategy, quality_score)?;
        }

        Ok(DecompositionResult {
            subproblems,
            strategy_used: strategy,
            quality_score,
            analysis: problem_analysis,
            metadata: DecompositionMetadata::new(),
        })
    }

    /// Analyze problem structure and characteristics
    pub fn analyze_problem(&mut self, problem: &IsingModel) -> Result<ProblemAnalysis, String> {
        // Calculate graph metrics
        let graph_metrics = self
            .problem_analyzer
            .graph_analyzer
            .calculate_metrics(problem)?;

        // Detect communities
        let communities = self
            .problem_analyzer
            .graph_analyzer
            .detect_communities(problem)?;

        // Detect structures
        let structures = self
            .problem_analyzer
            .structure_detector
            .detect_structures(problem)?;

        // Estimate complexity
        let complexity = self
            .problem_analyzer
            .complexity_estimator
            .estimate_complexity(problem)?;

        // Score decomposability
        let decomposability = self
            .problem_analyzer
            .decomposability_scorer
            .score_decomposability(problem)?;

        Ok(ProblemAnalysis {
            graph_metrics,
            communities,
            structures,
            complexity,
            decomposability,
            problem_features: self.extract_problem_features(problem)?,
        })
    }

    /// Extract problem features for learning
    pub fn extract_problem_features(&self, problem: &IsingModel) -> Result<Array1<f64>, String> {
        let mut features = Vec::new();

        // Basic features
        features.push(problem.num_qubits as f64);

        // Count non-zero couplings
        let mut num_couplings = 0;
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    num_couplings += 1;
                }
            }
        }
        features.push(f64::from(num_couplings));

        // Density
        let max_couplings = problem.num_qubits * (problem.num_qubits - 1) / 2;
        let density = if max_couplings > 0 {
            f64::from(num_couplings) / max_couplings as f64
        } else {
            0.0
        };
        features.push(density);

        // Bias statistics
        let mut bias_sum = 0.0;
        let mut bias_var = 0.0;
        for i in 0..problem.num_qubits {
            let bias = problem.get_bias(i).unwrap_or(0.0);
            bias_sum += bias;
            bias_var += bias * bias;
        }
        let bias_mean = bias_sum / problem.num_qubits as f64;
        bias_var = bias_var / problem.num_qubits as f64 - bias_mean * bias_mean;
        features.extend_from_slice(&[bias_mean, bias_var.sqrt()]);

        // Coupling statistics
        let mut coupling_sum = 0.0;
        let mut coupling_var = 0.0;
        if num_couplings > 0 {
            for i in 0..problem.num_qubits {
                for j in (i + 1)..problem.num_qubits {
                    let coupling = problem.get_coupling(i, j).unwrap_or(0.0);
                    if coupling.abs() > 1e-10 {
                        coupling_sum += coupling;
                        coupling_var += coupling * coupling;
                    }
                }
            }
            let coupling_mean = coupling_sum / f64::from(num_couplings);
            coupling_var = coupling_var / f64::from(num_couplings) - coupling_mean * coupling_mean;
            features.extend_from_slice(&[coupling_mean, coupling_var.sqrt()]);
        } else {
            features.extend_from_slice(&[0.0, 0.0]);
        }

        // Pad to fixed size
        features.resize(20, 0.0);
        Ok(Array1::from_vec(features))
    }

    /// Select decomposition strategy using active learning
    fn select_strategy(
        &self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<DecompositionStrategy, String> {
        // Get strategy recommendation from model
        let recommendation = self
            .strategy_learner
            .recommend_strategy(problem, analysis)?;

        // Decide whether to explore or exploit
        let should_explore = self.should_explore(&analysis.problem_features)?;

        if should_explore {
            // Exploration: try a different strategy or query for feedback
            self.explore_strategy(problem, analysis, &recommendation)
        } else {
            // Exploitation: use the recommended strategy
            Ok(recommendation)
        }
    }

    /// Decide whether to explore or exploit
    fn should_explore(&self, features: &Array1<f64>) -> Result<bool, String> {
        // Get uncertainty estimate for this problem
        let uncertainty = self
            .strategy_learner
            .selection_model
            .get_uncertainty(features)?;

        // Compare with threshold and exploration rate
        let explore_threshold = self.config.exploration_rate;
        let uncertainty_threshold = 0.5; // High uncertainty threshold

        Ok(uncertainty > uncertainty_threshold || thread_rng().gen::<f64>() < explore_threshold)
    }

    /// Explore strategy selection
    fn explore_strategy(
        &self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
        base_recommendation: &DecompositionStrategy,
    ) -> Result<DecompositionStrategy, String> {
        // Select query strategy
        match self.strategy_learner.query_selector.query_strategy {
            QueryStrategy::UncertaintySampling => self.uncertainty_sampling_exploration(analysis),
            QueryStrategy::DiversitySampling => self.diversity_sampling_exploration(analysis),
            _ => {
                // Default: slightly perturb the base recommendation
                self.perturb_strategy(base_recommendation)
            }
        }
    }

    /// Uncertainty sampling exploration
    fn uncertainty_sampling_exploration(
        &self,
        analysis: &ProblemAnalysis,
    ) -> Result<DecompositionStrategy, String> {
        // Find strategy with highest uncertainty
        let strategies = &self.knowledge_base.strategy_database.strategies;
        let mut best_strategy = DecompositionStrategy::GraphPartitioning;
        let mut highest_uncertainty = 0.0;

        for strategy in strategies {
            let uncertainty = self
                .strategy_learner
                .selection_model
                .get_strategy_uncertainty(strategy, &analysis.problem_features)?;

            if uncertainty > highest_uncertainty {
                highest_uncertainty = uncertainty;
                best_strategy = strategy.clone();
            }
        }

        Ok(best_strategy)
    }

    /// Diversity sampling exploration
    fn diversity_sampling_exploration(
        &self,
        analysis: &ProblemAnalysis,
    ) -> Result<DecompositionStrategy, String> {
        // Find strategy most different from recently used strategies
        let recent_strategies: Vec<_> = self
            .strategy_learner
            .query_selector
            .query_history
            .iter()
            .rev()
            .take(10)
            .map(|record| &record.recommended_strategy)
            .collect();

        let strategies = &self.knowledge_base.strategy_database.strategies;
        let mut best_strategy = DecompositionStrategy::GraphPartitioning;
        let mut max_diversity = 0.0;

        for strategy in strategies {
            let diversity = self.calculate_strategy_diversity(strategy, &recent_strategies)?;

            if diversity > max_diversity {
                max_diversity = diversity;
                best_strategy = strategy.clone();
            }
        }

        Ok(best_strategy)
    }

    /// Calculate strategy diversity
    fn calculate_strategy_diversity(
        &self,
        strategy: &DecompositionStrategy,
        recent_strategies: &[&DecompositionStrategy],
    ) -> Result<f64, String> {
        if recent_strategies.is_empty() {
            return Ok(1.0);
        }

        let mut total_distance = 0.0;
        for &recent_strategy in recent_strategies {
            total_distance += self.strategy_distance(strategy, recent_strategy)?;
        }

        Ok(total_distance / recent_strategies.len() as f64)
    }

    /// Calculate distance between strategies
    fn strategy_distance(
        &self,
        strategy1: &DecompositionStrategy,
        strategy2: &DecompositionStrategy,
    ) -> Result<f64, String> {
        // Simplified strategy distance
        if strategy1 == strategy2 {
            Ok(0.0)
        } else {
            Ok(1.0)
        }
    }

    /// Perturb strategy selection
    fn perturb_strategy(
        &self,
        base_strategy: &DecompositionStrategy,
    ) -> Result<DecompositionStrategy, String> {
        let strategies = &self.knowledge_base.strategy_database.strategies;
        let mut rng = thread_rng();

        // Select a random strategy with some probability
        if rng.gen::<f64>() < 0.3 {
            let random_idx = rng.gen_range(0..strategies.len());
            Ok(strategies[random_idx].clone())
        } else {
            Ok(base_strategy.clone())
        }
    }

    /// Generate subproblems using selected strategy
    fn generate_subproblems(
        &self,
        problem: &IsingModel,
        strategy: &DecompositionStrategy,
        analysis: &ProblemAnalysis,
    ) -> Result<Vec<Subproblem>, String> {
        match strategy {
            DecompositionStrategy::GraphPartitioning => {
                self.graph_partitioning_decomposition(problem, analysis)
            }
            DecompositionStrategy::CommunityDetection => {
                self.community_detection_decomposition(problem, analysis)
            }
            DecompositionStrategy::SpectralClustering => {
                self.spectral_clustering_decomposition(problem, analysis)
            }
            DecompositionStrategy::NoDecomposition => {
                Ok(vec![Subproblem::from_full_problem(problem)])
            }
            _ => {
                // Default to graph partitioning
                self.graph_partitioning_decomposition(problem, analysis)
            }
        }
    }

    /// Graph partitioning decomposition
    fn graph_partitioning_decomposition(
        &self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<Vec<Subproblem>, String> {
        // Simple bisection for demonstration
        let n = problem.num_qubits;
        let mid = n / 2;

        let partition1: Vec<usize> = (0..mid).collect();
        let partition2: Vec<usize> = (mid..n).collect();

        let subproblem1 = self.create_subproblem(problem, &partition1, 0)?;
        let subproblem2 = self.create_subproblem(problem, &partition2, 1)?;

        Ok(vec![subproblem1, subproblem2])
    }

    /// Community detection decomposition
    fn community_detection_decomposition(
        &self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<Vec<Subproblem>, String> {
        // Use detected communities from analysis
        let communities = &analysis.communities;
        let mut subproblems = Vec::new();

        for (i, community) in communities.iter().enumerate() {
            if community.vertices.len() >= self.config.min_subproblem_size {
                let subproblem = self.create_subproblem(problem, &community.vertices, i)?;
                subproblems.push(subproblem);
            }
        }

        // If no valid communities, fall back to graph partitioning
        if subproblems.is_empty() {
            self.graph_partitioning_decomposition(problem, analysis)
        } else {
            Ok(subproblems)
        }
    }

    /// Spectral clustering decomposition
    fn spectral_clustering_decomposition(
        &self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<Vec<Subproblem>, String> {
        // Simplified spectral clustering - in practice would use eigendecomposition
        self.graph_partitioning_decomposition(problem, analysis)
    }

    /// Create subproblem from vertex subset
    pub fn create_subproblem(
        &self,
        problem: &IsingModel,
        vertices: &[usize],
        subproblem_id: usize,
    ) -> Result<Subproblem, String> {
        // Create Ising model for subproblem
        let mut subproblem_model = IsingModel::new(vertices.len());

        // Map vertex indices
        let vertex_map: HashMap<usize, usize> = vertices
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx))
            .collect();

        // Copy biases
        for (new_idx, &old_idx) in vertices.iter().enumerate() {
            let bias = problem.get_bias(old_idx).unwrap_or(0.0);
            subproblem_model
                .set_bias(new_idx, bias)
                .map_err(|e| e.to_string())?;
        }

        // Copy couplings within subproblem
        for (i, &old_i) in vertices.iter().enumerate() {
            for (j, &old_j) in vertices.iter().enumerate().skip(i + 1) {
                let coupling = problem.get_coupling(old_i, old_j).unwrap_or(0.0);
                if coupling.abs() > 1e-10 {
                    subproblem_model
                        .set_coupling(i, j, coupling)
                        .map_err(|e| e.to_string())?;
                }
            }
        }

        // Identify boundary edges (edges connecting to other subproblems)
        let mut boundary_edges = Vec::new();
        for &vertex in vertices {
            for other_vertex in 0..problem.num_qubits {
                if !vertices.contains(&other_vertex) {
                    let coupling = problem.get_coupling(vertex, other_vertex).unwrap_or(0.0);
                    if coupling.abs() > 1e-10 {
                        boundary_edges.push(BoundaryEdge {
                            internal_vertex: vertex_map[&vertex],
                            external_vertex: other_vertex,
                            coupling_strength: coupling,
                        });
                    }
                }
            }
        }

        Ok(Subproblem {
            id: subproblem_id,
            model: subproblem_model,
            vertices: vertices.to_vec(),
            boundary_edges,
            metadata: SubproblemMetadata::new(),
        })
    }

    /// Validate decomposition quality
    pub fn validate_decomposition_quality(
        &self,
        subproblems: &[Subproblem],
        original_problem: &IsingModel,
    ) -> Result<f64, String> {
        let mut total_score = 0.0;
        let mut num_criteria = 0;

        // Size balance
        let sizes: Vec<usize> = subproblems.iter().map(|sp| sp.vertices.len()).collect();
        let avg_size = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let size_variance = sizes
            .iter()
            .map(|&size| (size as f64 - avg_size).powi(2))
            .sum::<f64>()
            / sizes.len() as f64;
        let size_balance_score = 1.0 / (1.0 + size_variance / avg_size);
        total_score += size_balance_score;
        num_criteria += 1;

        // Cut quality (minimize boundary edges)
        let total_boundary_edges: usize =
            subproblems.iter().map(|sp| sp.boundary_edges.len()).sum();
        let total_edges = original_problem.num_qubits * (original_problem.num_qubits - 1) / 2;
        let cut_quality_score = 1.0 - (total_boundary_edges as f64 / total_edges as f64);
        total_score += cut_quality_score;
        num_criteria += 1;

        // Coverage (all vertices included)
        let covered_vertices: HashSet<usize> = subproblems
            .iter()
            .flat_map(|sp| sp.vertices.iter())
            .copied()
            .collect();
        let coverage_score = covered_vertices.len() as f64 / original_problem.num_qubits as f64;
        total_score += coverage_score;
        num_criteria += 1;

        Ok(total_score / f64::from(num_criteria))
    }

    /// Update knowledge base with new experience
    fn update_knowledge_base(
        &mut self,
        problem: &IsingModel,
        strategy: &DecompositionStrategy,
        quality_score: f64,
    ) -> Result<(), String> {
        // Update strategy performance history
        let performance_record = PerformanceRecord {
            timestamp: Instant::now(),
            problem_id: format!("problem_{}", problem.num_qubits),
            strategy_used: strategy.clone(),
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert(EvaluationMetric::SolutionQuality, quality_score);
                metrics
            },
            overall_score: quality_score,
        };

        self.strategy_learner
            .performance_history
            .entry(format!("{strategy:?}"))
            .or_insert_with(Vec::new)
            .push(performance_record.clone());

        self.performance_evaluator
            .performance_history
            .push(performance_record);

        // Update learning statistics
        self.strategy_learner.learning_stats.total_queries += 1;
        if quality_score > self.config.performance_threshold {
            self.strategy_learner.learning_stats.successful_predictions += 1;
        }

        // Update success rates
        let strategy_key = format!("{strategy:?}");
        let history = &self.strategy_learner.performance_history[&strategy_key];
        let success_count = history
            .iter()
            .filter(|record| record.overall_score > self.config.performance_threshold)
            .count();
        let success_rate = success_count as f64 / history.len() as f64;

        self.knowledge_base
            .strategy_database
            .success_rates
            .insert(strategy.clone(), success_rate);

        Ok(())
    }
}
