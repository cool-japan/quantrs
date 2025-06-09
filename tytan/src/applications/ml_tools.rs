//! Machine learning tools: Feature selection and optimization.
//!
//! This module provides quantum optimization tools for machine learning
//! including feature selection, hyperparameter optimization, and model selection.

use crate::sampler::{Sampler, SampleResult, SamplerError, SamplerResult};
use ndarray::{Array, Array1, Array2, Array3, IxDyn};
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use rand::prelude::*;
use rand::thread_rng;

/// Feature selector using quantum optimization
pub struct QuantumFeatureSelector {
    /// Feature data
    features: FeatureData,
    /// Selection method
    method: SelectionMethod,
    /// Evaluation criteria
    criteria: EvaluationCriteria,
    /// Constraints
    constraints: SelectionConstraints,
    /// Cross-validation strategy
    cv_strategy: CrossValidationStrategy,
}

#[derive(Debug, Clone)]
pub struct FeatureData {
    /// Feature matrix (samples × features)
    pub data: Array2<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Target variable
    pub target: Array1<f64>,
    /// Feature types
    pub feature_types: Vec<FeatureType>,
    /// Feature statistics
    pub statistics: FeatureStatistics,
}

#[derive(Debug, Clone)]
pub enum FeatureType {
    /// Continuous numeric
    Continuous,
    /// Discrete numeric
    Discrete { levels: usize },
    /// Binary
    Binary,
    /// Categorical
    Categorical { categories: Vec<String> },
    /// Ordinal
    Ordinal { levels: Vec<String> },
    /// Text
    Text,
    /// Time series
    TimeSeries { frequency: String },
}

#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    /// Mean values
    pub means: Array1<f64>,
    /// Standard deviations
    pub stds: Array1<f64>,
    /// Correlations with target
    pub target_correlations: Array1<f64>,
    /// Feature-feature correlations
    pub feature_correlations: Array2<f64>,
    /// Missing value counts
    pub missing_counts: Array1<usize>,
    /// Unique value counts
    pub unique_counts: Array1<usize>,
}

#[derive(Debug, Clone)]
pub enum SelectionMethod {
    /// Filter methods
    Filter {
        metric: FilterMetric,
        threshold: f64,
    },
    /// Wrapper methods
    Wrapper {
        model: MLModel,
        search_strategy: SearchStrategy,
    },
    /// Embedded methods
    Embedded {
        regularization: RegularizationType,
        strength: f64,
    },
    /// Hybrid approach
    Hybrid {
        filter_metric: FilterMetric,
        wrapper_model: MLModel,
        balance: f64,
    },
    /// Quantum-inspired
    QuantumInspired {
        entanglement_penalty: f64,
        coherence_bonus: f64,
    },
}

#[derive(Debug, Clone)]
pub enum FilterMetric {
    /// Mutual information
    MutualInformation,
    /// Chi-squared test
    ChiSquared,
    /// ANOVA F-value
    ANOVA,
    /// Correlation coefficient
    Correlation,
    /// Information gain
    InformationGain,
    /// Variance threshold
    VarianceThreshold { threshold: f64 },
    /// Relief algorithm
    Relief,
}

#[derive(Debug, Clone)]
pub struct MLModel {
    /// Model type
    pub model_type: ModelType,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, f64>,
    /// Training parameters
    pub training_params: TrainingParameters,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    /// Linear regression
    LinearRegression,
    /// Logistic regression
    LogisticRegression,
    /// Support vector machine
    SVM { kernel: String },
    /// Random forest
    RandomForest { n_trees: usize },
    /// Neural network
    NeuralNetwork { architecture: Vec<usize> },
    /// Gradient boosting
    GradientBoosting { n_estimators: usize },
    /// K-nearest neighbors
    KNN { k: usize },
}

#[derive(Debug, Clone)]
pub struct TrainingParameters {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping
    pub early_stopping: bool,
    /// Patience
    pub patience: usize,
}

#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exhaustive search
    Exhaustive,
    /// Forward selection
    ForwardSelection,
    /// Backward elimination
    BackwardElimination,
    /// Bidirectional search
    Bidirectional,
    /// Genetic algorithm
    Genetic { population_size: usize, generations: usize },
    /// Simulated annealing
    SimulatedAnnealing { temperature: f64, cooling_rate: f64 },
}

#[derive(Debug, Clone)]
pub enum RegularizationType {
    /// L1 (Lasso)
    L1,
    /// L2 (Ridge)
    L2,
    /// Elastic net
    ElasticNet { l1_ratio: f64 },
    /// Group lasso
    GroupLasso { groups: Vec<Vec<usize>> },
    /// Fused lasso
    FusedLasso,
}

#[derive(Debug, Clone)]
pub struct EvaluationCriteria {
    /// Primary metric
    pub primary_metric: EvaluationMetric,
    /// Secondary metrics
    pub secondary_metrics: Vec<EvaluationMetric>,
    /// Metric weights
    pub weights: HashMap<String, f64>,
    /// Target performance
    pub target_performance: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum EvaluationMetric {
    /// Accuracy
    Accuracy,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// F1 score
    F1Score,
    /// AUC-ROC
    AUCROC,
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
    /// R-squared
    R2,
    /// Log loss
    LogLoss,
    /// Custom metric
    Custom { name: String },
}

#[derive(Debug, Clone)]
pub struct SelectionConstraints {
    /// Minimum features
    pub min_features: Option<usize>,
    /// Maximum features
    pub max_features: Option<usize>,
    /// Must-include features
    pub must_include: Vec<usize>,
    /// Must-exclude features
    pub must_exclude: Vec<usize>,
    /// Feature groups (select all or none)
    pub feature_groups: Vec<Vec<usize>>,
    /// Budget constraint
    pub feature_costs: Option<HashMap<usize, f64>>,
    /// Maximum cost
    pub max_cost: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum CrossValidationStrategy {
    /// K-fold
    KFold { k: usize, shuffle: bool },
    /// Stratified K-fold
    StratifiedKFold { k: usize },
    /// Leave-one-out
    LeaveOneOut,
    /// Time series split
    TimeSeriesSplit { n_splits: usize },
    /// Group K-fold
    GroupKFold { k: usize, groups: Vec<usize> },
    /// Monte Carlo
    MonteCarlo { n_splits: usize, test_size: f64 },
}

impl QuantumFeatureSelector {
    /// Create new feature selector
    pub fn new(
        features: FeatureData,
        method: SelectionMethod,
    ) -> Self {
        Self {
            features,
            method,
            criteria: EvaluationCriteria {
                primary_metric: EvaluationMetric::Accuracy,
                secondary_metrics: vec![],
                weights: HashMap::new(),
                target_performance: None,
            },
            constraints: SelectionConstraints::default(),
            cv_strategy: CrossValidationStrategy::KFold { k: 5, shuffle: true },
        }
    }
    
    /// Set evaluation criteria
    pub fn with_criteria(mut self, criteria: EvaluationCriteria) -> Self {
        self.criteria = criteria;
        self
    }
    
    /// Set constraints
    pub fn with_constraints(mut self, constraints: SelectionConstraints) -> Self {
        self.constraints = constraints;
        self
    }
    
    /// Set cross-validation strategy
    pub fn with_cv_strategy(mut self, strategy: CrossValidationStrategy) -> Self {
        self.cv_strategy = strategy;
        self
    }
    
    /// Build QUBO for feature selection
    pub fn build_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let n_features = self.features.feature_names.len();
        let mut qubo = Array2::zeros((n_features, n_features));
        let mut var_map = HashMap::new();
        
        // Create variable mapping
        for (i, name) in self.features.feature_names.iter().enumerate() {
            var_map.insert(format!("feature_{}", i), i);
        }
        
        // Add objective based on method
        match &self.method {
            SelectionMethod::Filter { metric, threshold } => {
                self.add_filter_objective(&mut qubo, metric, *threshold)?;
            }
            SelectionMethod::Wrapper { model, .. } => {
                self.add_wrapper_objective(&mut qubo, model)?;
            }
            SelectionMethod::Embedded { regularization, strength } => {
                self.add_embedded_objective(&mut qubo, regularization, *strength)?;
            }
            SelectionMethod::Hybrid { filter_metric, wrapper_model, balance } => {
                self.add_hybrid_objective(&mut qubo, filter_metric, wrapper_model, *balance)?;
            }
            SelectionMethod::QuantumInspired { entanglement_penalty, coherence_bonus } => {
                self.add_quantum_objective(&mut qubo, *entanglement_penalty, *coherence_bonus)?;
            }
        }
        
        // Add constraints
        self.add_selection_constraints(&mut qubo)?;
        
        Ok((qubo, var_map))
    }
    
    /// Add filter-based objective
    fn add_filter_objective(
        &self,
        qubo: &mut Array2<f64>,
        metric: &FilterMetric,
        threshold: f64,
    ) -> Result<(), String> {
        match metric {
            FilterMetric::MutualInformation => {
                // Use pre-computed mutual information scores
                for i in 0..self.features.feature_names.len() {
                    let mi_score = self.compute_mutual_information(i)?;
                    qubo[[i, i]] -= mi_score;
                }
            }
            FilterMetric::Correlation => {
                // Use correlation with target
                for i in 0..self.features.feature_names.len() {
                    let corr = self.features.statistics.target_correlations[i].abs();
                    if corr >= threshold {
                        qubo[[i, i]] -= corr;
                    }
                }
            }
            FilterMetric::VarianceThreshold { threshold } => {
                // Favor high-variance features
                for i in 0..self.features.feature_names.len() {
                    let variance = self.features.statistics.stds[i].powi(2);
                    if variance >= *threshold {
                        qubo[[i, i]] -= 1.0;
                    }
                }
            }
            _ => {
                // Other metrics
                for i in 0..self.features.feature_names.len() {
                    qubo[[i, i]] -= 1.0; // Default score
                }
            }
        }
        
        // Penalize correlated features
        self.add_correlation_penalty(qubo)?;
        
        Ok(())
    }
    
    /// Compute mutual information
    fn compute_mutual_information(&self, feature_idx: usize) -> Result<f64, String> {
        // Simplified mutual information calculation
        // In practice, would use proper MI estimation
        
        let feature = self.features.data.column(feature_idx);
        let target = &self.features.target;
        
        // Discretize if continuous
        let n_bins = 10;
        let feature_discrete = self.discretize_array(feature, n_bins);
        let target_discrete = self.discretize_array(target, n_bins);
        
        // Compute joint and marginal probabilities
        let mut joint_counts = Array2::zeros((n_bins, n_bins));
        for (i, (f, t)) in feature_discrete.iter().zip(target_discrete.iter()).enumerate() {
            joint_counts[[*f, *t]] += 1.0;
        }
        
        let joint_probs = &joint_counts / feature.len() as f64;
        let feature_probs = joint_probs.sum_axis(ndarray::Axis(1));
        let target_probs = joint_probs.sum_axis(ndarray::Axis(0));
        
        // Compute MI
        let mut mi = 0.0;
        for i in 0..n_bins {
            for j in 0..n_bins {
                if joint_probs[[i, j]] > 0.0 {
                    mi += joint_probs[[i, j]] * (joint_probs[[i, j]] / 
                        (feature_probs[i] * target_probs[j])).ln();
                }
            }
        }
        
        Ok(mi)
    }
    
    /// Discretize continuous array
    fn discretize_array(&self, array: &Array1<f64>, n_bins: usize) -> Vec<usize> {
        let min = array.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max = array.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let bin_width = (max - min) / n_bins as f64;
        
        array.iter()
            .map(|&x| ((x - min) / bin_width).floor() as usize)
            .map(|b| b.min(n_bins - 1))
            .collect()
    }
    
    /// Add correlation penalty
    fn add_correlation_penalty(&self, qubo: &mut Array2<f64>) -> Result<(), String> {
        let corr_threshold = 0.9;
        let penalty = 10.0;
        
        let corr_matrix = &self.features.statistics.feature_correlations;
        
        for i in 0..corr_matrix.shape()[0] {
            for j in i+1..corr_matrix.shape()[1] {
                if corr_matrix[[i, j]].abs() > corr_threshold {
                    // Penalize selecting both highly correlated features
                    qubo[[i, j]] += penalty;
                    qubo[[j, i]] += penalty;
                }
            }
        }
        
        Ok(())
    }
    
    /// Add wrapper-based objective
    fn add_wrapper_objective(
        &self,
        qubo: &mut Array2<f64>,
        model: &MLModel,
    ) -> Result<(), String> {
        // Wrapper methods require model evaluation
        // Use surrogate model or pre-computed scores
        
        // Simplified: use feature importance from a preliminary model
        let importances = self.compute_feature_importances(model)?;
        
        for (i, &importance) in importances.iter().enumerate() {
            qubo[[i, i]] -= importance;
        }
        
        // Add interaction terms based on feature synergy
        self.add_feature_interactions(qubo, model)?;
        
        Ok(())
    }
    
    /// Compute feature importances
    fn compute_feature_importances(&self, model: &MLModel) -> Result<Array1<f64>, String> {
        // Simplified: return random importances
        // In practice, would train model and extract importances
        
        let n_features = self.features.feature_names.len();
        let mut rng = thread_rng();
        
        Ok(Array1::from_shape_fn(n_features, |_| rng.gen::<f64>()))
    }
    
    /// Add feature interaction terms
    fn add_feature_interactions(
        &self,
        qubo: &mut Array2<f64>,
        model: &MLModel,
    ) -> Result<(), String> {
        // Add synergy bonus for features that work well together
        // Simplified: use correlation structure
        
        let synergy_bonus = -5.0;
        let corr_matrix = &self.features.statistics.feature_correlations;
        
        for i in 0..corr_matrix.shape()[0] {
            for j in i+1..corr_matrix.shape()[1] {
                // Moderate correlation might indicate synergy
                let corr = corr_matrix[[i, j]].abs();
                if corr > 0.3 && corr < 0.7 {
                    qubo[[i, j]] += synergy_bonus * corr;
                    qubo[[j, i]] += synergy_bonus * corr;
                }
            }
        }
        
        Ok(())
    }
    
    /// Add embedded method objective
    fn add_embedded_objective(
        &self,
        qubo: &mut Array2<f64>,
        regularization: &RegularizationType,
        strength: f64,
    ) -> Result<(), String> {
        match regularization {
            RegularizationType::L1 => {
                // L1 penalty on each feature
                for i in 0..self.features.feature_names.len() {
                    qubo[[i, i]] += strength;
                }
            }
            RegularizationType::L2 => {
                // L2 penalty (quadratic)
                for i in 0..self.features.feature_names.len() {
                    qubo[[i, i]] += strength;
                    // Add quadratic term (simplified)
                    for j in 0..self.features.feature_names.len() {
                        if i != j {
                            qubo[[i, j]] += strength * 0.1;
                        }
                    }
                }
            }
            RegularizationType::ElasticNet { l1_ratio } => {
                // Combination of L1 and L2
                let l1_strength = strength * l1_ratio;
                let l2_strength = strength * (1.0 - l1_ratio);
                
                for i in 0..self.features.feature_names.len() {
                    qubo[[i, i]] += l1_strength + l2_strength;
                }
            }
            RegularizationType::GroupLasso { groups } => {
                // Penalize groups together
                for group in groups {
                    let group_penalty = strength / group.len() as f64;
                    for &i in group {
                        for &j in group {
                            if i < self.features.feature_names.len() && 
                               j < self.features.feature_names.len() {
                                qubo[[i, j]] += group_penalty;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Add hybrid objective
    fn add_hybrid_objective(
        &self,
        qubo: &mut Array2<f64>,
        filter_metric: &FilterMetric,
        wrapper_model: &MLModel,
        balance: f64,
    ) -> Result<(), String> {
        // Combine filter and wrapper approaches
        
        // Filter component
        let mut filter_qubo = Array2::zeros(qubo.shape());
        self.add_filter_objective(&mut filter_qubo, filter_metric, 0.0)?;
        
        // Wrapper component
        let mut wrapper_qubo = Array2::zeros(qubo.shape());
        self.add_wrapper_objective(&mut wrapper_qubo, wrapper_model)?;
        
        // Combine with balance
        *qubo = &filter_qubo * balance + &wrapper_qubo * (1.0 - balance);
        
        Ok(())
    }
    
    /// Add quantum-inspired objective
    fn add_quantum_objective(
        &self,
        qubo: &mut Array2<f64>,
        entanglement_penalty: f64,
        coherence_bonus: f64,
    ) -> Result<(), String> {
        // Quantum-inspired feature selection
        // Based on quantum information theory concepts
        
        // Entanglement penalty for redundant features
        let corr_matrix = &self.features.statistics.feature_correlations;
        
        for i in 0..corr_matrix.shape()[0] {
            for j in i+1..corr_matrix.shape()[1] {
                let correlation = corr_matrix[[i, j]].abs();
                // High correlation = high entanglement
                let entanglement = correlation.powi(2);
                qubo[[i, j]] += entanglement_penalty * entanglement;
                qubo[[j, i]] += entanglement_penalty * entanglement;
            }
        }
        
        // Coherence bonus for informative features
        for i in 0..self.features.feature_names.len() {
            let target_corr = self.features.statistics.target_correlations[i].abs();
            let variance = self.features.statistics.stds[i].powi(2);
            
            // High correlation with target and high variance = high coherence
            let coherence = target_corr * variance.sqrt();
            qubo[[i, i]] -= coherence_bonus * coherence;
        }
        
        // Quantum superposition principle: favor diverse feature sets
        self.add_diversity_bonus(qubo, coherence_bonus * 0.5)?;
        
        Ok(())
    }
    
    /// Add diversity bonus
    fn add_diversity_bonus(&self, qubo: &mut Array2<f64>, bonus: f64) -> Result<(), String> {
        // Favor selecting features from different types/groups
        
        // Group features by type
        let mut type_groups: HashMap<String, Vec<usize>> = HashMap::new();
        
        for (i, ftype) in self.features.feature_types.iter().enumerate() {
            let type_key = match ftype {
                FeatureType::Continuous => "continuous",
                FeatureType::Discrete { .. } => "discrete",
                FeatureType::Binary => "binary",
                FeatureType::Categorical { .. } => "categorical",
                _ => "other",
            };
            
            type_groups.entry(type_key.to_string())
                .or_insert_with(Vec::new)
                .push(i);
        }
        
        // Bonus for selecting from different groups
        for (_, group1) in &type_groups {
            for (_, group2) in &type_groups {
                if group1 != group2 {
                    for &i in group1 {
                        for &j in group2 {
                            if i < j {
                                qubo[[i, j]] -= bonus;
                                qubo[[j, i]] -= bonus;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Add selection constraints
    fn add_selection_constraints(&self, qubo: &mut Array2<f64>) -> Result<(), String> {
        let penalty = 100.0;
        
        // Must-include features
        for &feature_idx in &self.constraints.must_include {
            qubo[[feature_idx, feature_idx]] -= penalty * 10.0;
        }
        
        // Must-exclude features
        for &feature_idx in &self.constraints.must_exclude {
            qubo[[feature_idx, feature_idx]] += penalty * 10.0;
        }
        
        // Feature groups (all or none)
        for group in &self.constraints.feature_groups {
            for &i in group {
                for &j in group {
                    if i != j {
                        qubo[[i, j]] -= penalty;
                    }
                }
            }
        }
        
        // Cost constraint
        if let (Some(costs), Some(max_cost)) = (&self.constraints.feature_costs, self.constraints.max_cost) {
            // Soft constraint: penalize expensive features
            for (&feature_idx, &cost) in costs {
                if feature_idx < qubo.shape()[0] {
                    qubo[[feature_idx, feature_idx]] += (cost / max_cost) * penalty;
                }
            }
        }
        
        Ok(())
    }
    
    /// Decode solution to selected features
    pub fn decode_solution(&self, solution: &HashMap<String, bool>) -> SelectedFeatures {
        let mut selected_indices = Vec::new();
        let mut selected_names = Vec::new();
        
        for (i, name) in self.features.feature_names.iter().enumerate() {
            let var_name = format!("feature_{}", i);
            if *solution.get(&var_name).unwrap_or(&false) {
                selected_indices.push(i);
                selected_names.push(name.clone());
            }
        }
        
        SelectedFeatures {
            indices: selected_indices,
            names: selected_names,
            performance_estimate: self.estimate_performance(&solution),
            importance_scores: self.calculate_importance_scores(&solution),
        }
    }
    
    /// Estimate performance of selected features
    fn estimate_performance(&self, solution: &HashMap<String, bool>) -> f64 {
        // Simplified: use correlation with target
        let mut total_score = 0.0;
        let mut count = 0;
        
        for (i, _) in self.features.feature_names.iter().enumerate() {
            let var_name = format!("feature_{}", i);
            if *solution.get(&var_name).unwrap_or(&false) {
                total_score += self.features.statistics.target_correlations[i].abs();
                count += 1;
            }
        }
        
        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate importance scores
    fn calculate_importance_scores(&self, solution: &HashMap<String, bool>) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        
        for (i, name) in self.features.feature_names.iter().enumerate() {
            let var_name = format!("feature_{}", i);
            if *solution.get(&var_name).unwrap_or(&false) {
                let score = self.features.statistics.target_correlations[i].abs();
                scores.insert(name.clone(), score);
            }
        }
        
        scores
    }
}

impl Default for SelectionConstraints {
    fn default() -> Self {
        Self {
            min_features: None,
            max_features: None,
            must_include: Vec::new(),
            must_exclude: Vec::new(),
            feature_groups: Vec::new(),
            feature_costs: None,
            max_cost: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SelectedFeatures {
    pub indices: Vec<usize>,
    pub names: Vec<String>,
    pub performance_estimate: f64,
    pub importance_scores: HashMap<String, f64>,
}

/// Hyperparameter optimizer
pub struct HyperparameterOptimizer {
    /// Model to optimize
    model: MLModel,
    /// Parameter space
    param_space: ParameterSpace,
    /// Optimization strategy
    strategy: OptimizationStrategy,
    /// Evaluation method
    evaluation: HyperparameterEvaluation,
}

#[derive(Debug, Clone)]
pub struct ParameterSpace {
    /// Continuous parameters
    pub continuous: HashMap<String, ContinuousParam>,
    /// Discrete parameters
    pub discrete: HashMap<String, DiscreteParam>,
    /// Categorical parameters
    pub categorical: HashMap<String, CategoricalParam>,
    /// Conditional parameters
    pub conditional: Vec<ConditionalParam>,
}

#[derive(Debug, Clone)]
pub struct ContinuousParam {
    pub min: f64,
    pub max: f64,
    pub scale: ScaleType,
    pub default: f64,
}

#[derive(Debug, Clone)]
pub enum ScaleType {
    Linear,
    Log,
    Exponential,
}

#[derive(Debug, Clone)]
pub struct DiscreteParam {
    pub values: Vec<i32>,
    pub default: i32,
}

#[derive(Debug, Clone)]
pub struct CategoricalParam {
    pub choices: Vec<String>,
    pub default: String,
}

#[derive(Debug, Clone)]
pub struct ConditionalParam {
    pub parameter: String,
    pub condition: String,
    pub condition_value: String,
}

#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Grid search
    GridSearch,
    /// Random search
    RandomSearch { n_trials: usize },
    /// Bayesian optimization
    BayesianOptimization {
        acquisition: AcquisitionFunction,
        n_initial: usize,
    },
    /// Evolutionary strategy
    EvolutionaryStrategy {
        population_size: usize,
        mutation_rate: f64,
    },
    /// Quantum-inspired
    QuantumOptimization {
        tunneling_rate: f64,
        superposition_size: usize,
    },
}

#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound { kappa: f64 },
    EntropySearch,
}

#[derive(Debug, Clone)]
pub struct HyperparameterEvaluation {
    /// Evaluation metric
    pub metric: EvaluationMetric,
    /// Cross-validation
    pub cv_strategy: CrossValidationStrategy,
    /// Resource constraints
    pub constraints: ResourceConstraints,
}

#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum time per trial
    pub max_time_per_trial: Option<std::time::Duration>,
    /// Maximum total time
    pub max_total_time: Option<std::time::Duration>,
    /// Maximum memory
    pub max_memory: Option<usize>,
    /// Early stopping
    pub early_stopping: bool,
}

impl HyperparameterOptimizer {
    /// Build QUBO for hyperparameter optimization
    pub fn build_qubo(&self) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        // Discretize parameter space
        let discretized = self.discretize_parameters()?;
        
        let n_vars = discretized.total_combinations();
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();
        
        // Create variable mapping
        self.create_parameter_variables(&mut var_map, &discretized)?;
        
        // Add objective
        match &self.strategy {
            OptimizationStrategy::QuantumOptimization { tunneling_rate, superposition_size } => {
                self.add_quantum_hyperopt_objective(
                    &mut qubo,
                    &var_map,
                    &discretized,
                    *tunneling_rate,
                    *superposition_size,
                )?;
            }
            _ => {
                self.add_standard_hyperopt_objective(&mut qubo, &var_map, &discretized)?;
            }
        }
        
        Ok((qubo, var_map))
    }
    
    /// Discretize parameter space
    fn discretize_parameters(&self) -> Result<DiscretizedSpace, String> {
        let mut discretized = DiscretizedSpace {
            parameters: Vec::new(),
            grid_points: Vec::new(),
        };
        
        // Discretize continuous parameters
        for (name, param) in &self.param_space.continuous {
            let n_points = 10; // Resolution
            let mut points = Vec::new();
            
            for i in 0..n_points {
                let t = i as f64 / (n_points - 1) as f64;
                let value = match param.scale {
                    ScaleType::Linear => param.min + t * (param.max - param.min),
                    ScaleType::Log => {
                        let log_min = param.min.ln();
                        let log_max = param.max.ln();
                        (log_min + t * (log_max - log_min)).exp()
                    }
                    ScaleType::Exponential => {
                        param.min * (param.max / param.min).powf(t)
                    }
                };
                points.push(value);
            }
            
            discretized.parameters.push(name.clone());
            discretized.grid_points.push(points);
        }
        
        // Add discrete parameters
        for (name, param) in &self.param_space.discrete {
            discretized.parameters.push(name.clone());
            discretized.grid_points.push(
                param.values.iter().map(|&v| v as f64).collect()
            );
        }
        
        Ok(discretized)
    }
    
    /// Create parameter variables
    fn create_parameter_variables(
        &self,
        var_map: &mut HashMap<String, usize>,
        discretized: &DiscretizedSpace,
    ) -> Result<(), String> {
        let mut var_idx = 0;
        
        // Create variables for each parameter value combination
        for (param_idx, param_name) in discretized.parameters.iter().enumerate() {
            for (value_idx, _) in discretized.grid_points[param_idx].iter().enumerate() {
                let var_name = format!("param_{}_{}", param_name, value_idx);
                var_map.insert(var_name, var_idx);
                var_idx += 1;
            }
        }
        
        Ok(())
    }
    
    /// Add quantum hyperparameter optimization objective
    fn add_quantum_hyperopt_objective(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        discretized: &DiscretizedSpace,
        tunneling_rate: f64,
        superposition_size: usize,
    ) -> Result<(), String> {
        // Quantum-inspired optimization with tunneling and superposition
        
        // Add performance landscape (simplified)
        for (var_name, &var_idx) in var_map {
            // Estimate performance for this parameter setting
            let performance = self.estimate_parameter_performance(var_name, discretized)?;
            qubo[[var_idx, var_idx]] -= performance;
        }
        
        // Add quantum tunneling terms
        self.add_tunneling_terms(qubo, var_map, tunneling_rate)?;
        
        // Add superposition bonus for exploring multiple configurations
        self.add_superposition_bonus(qubo, var_map, superposition_size)?;
        
        Ok(())
    }
    
    /// Estimate parameter performance
    fn estimate_parameter_performance(
        &self,
        var_name: &str,
        discretized: &DiscretizedSpace,
    ) -> Result<f64, String> {
        // Simplified: use surrogate model or prior knowledge
        // In practice, would use Gaussian process or similar
        
        // Random performance for demonstration
        let mut rng = thread_rng();
        Ok(rng.gen::<f64>())
    }
    
    /// Add tunneling terms
    fn add_tunneling_terms(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        tunneling_rate: f64,
    ) -> Result<(), String> {
        // Allow transitions between nearby parameter values
        
        for (var1, &idx1) in var_map {
            for (var2, &idx2) in var_map {
                if var1 != var2 && self.are_neighbors(var1, var2) {
                    // Tunneling term
                    qubo[[idx1, idx2]] -= tunneling_rate;
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if parameters are neighbors
    fn are_neighbors(&self, var1: &str, var2: &str) -> bool {
        // Check if two parameter settings are adjacent
        // Simplified logic
        let parts1: Vec<&str> = var1.split('_').collect();
        let parts2: Vec<&str> = var2.split('_').collect();
        
        if parts1.len() >= 3 && parts2.len() >= 3 {
            // Same parameter, adjacent values
            if parts1[1] == parts2[1] {
                let idx1: usize = parts1[2].parse().unwrap_or(0);
                let idx2: usize = parts2[2].parse().unwrap_or(0);
                return (idx1 as i32 - idx2 as i32).abs() == 1;
            }
        }
        
        false
    }
    
    /// Add superposition bonus
    fn add_superposition_bonus(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        superposition_size: usize,
    ) -> Result<(), String> {
        // Bonus for maintaining superposition of multiple good configurations
        // This encourages exploration
        
        let bonus = -0.1;
        
        // Add small negative bias to encourage selecting multiple options
        for &idx in var_map.values() {
            qubo[[idx, idx]] += bonus;
        }
        
        Ok(())
    }
    
    /// Add standard hyperparameter optimization objective
    fn add_standard_hyperopt_objective(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        discretized: &DiscretizedSpace,
    ) -> Result<(), String> {
        // Standard objective based on estimated performance
        
        for (var_name, &var_idx) in var_map {
            let performance = self.estimate_parameter_performance(var_name, discretized)?;
            qubo[[var_idx, var_idx]] -= performance;
        }
        
        // Add regularization for smooth parameter landscapes
        self.add_smoothness_regularization(qubo, var_map)?;
        
        Ok(())
    }
    
    /// Add smoothness regularization
    fn add_smoothness_regularization(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(), String> {
        let regularization_strength = 0.01;
        
        // Penalize large jumps in parameter space
        for (var1, &idx1) in var_map {
            for (var2, &idx2) in var_map {
                if var1 != var2 && self.are_neighbors(var1, var2) {
                    // Encourage smooth transitions
                    qubo[[idx1, idx2]] -= regularization_strength;
                }
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct DiscretizedSpace {
    parameters: Vec<String>,
    grid_points: Vec<Vec<f64>>,
}

impl DiscretizedSpace {
    fn total_combinations(&self) -> usize {
        self.grid_points.iter()
            .map(|points| points.len())
            .sum()
    }
}

/// Model selector for choosing best ML model
pub struct ModelSelector {
    /// Candidate models
    candidates: Vec<CandidateModel>,
    /// Selection criteria
    criteria: ModelSelectionCriteria,
    /// Ensemble options
    ensemble_options: EnsembleOptions,
}

#[derive(Debug, Clone)]
pub struct CandidateModel {
    /// Model specification
    pub model: MLModel,
    /// Prior performance
    pub prior_performance: Option<f64>,
    /// Complexity measure
    pub complexity: f64,
    /// Training time estimate
    pub training_time: f64,
}

#[derive(Debug, Clone)]
pub struct ModelSelectionCriteria {
    /// Performance weight
    pub performance_weight: f64,
    /// Complexity penalty
    pub complexity_penalty: f64,
    /// Training time penalty
    pub time_penalty: f64,
    /// Interpretability requirement
    pub interpretability: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct EnsembleOptions {
    /// Allow ensemble
    pub allow_ensemble: bool,
    /// Maximum ensemble size
    pub max_size: usize,
    /// Ensemble method
    pub method: EnsembleMethod,
    /// Diversity requirement
    pub min_diversity: f64,
}

#[derive(Debug, Clone)]
pub enum EnsembleMethod {
    /// Simple averaging
    Averaging,
    /// Weighted averaging
    WeightedAveraging,
    /// Stacking
    Stacking { meta_model: Box<MLModel> },
    /// Boosting
    Boosting,
    /// Bagging
    Bagging,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_selector() {
        let n_samples = 100;
        let n_features = 10;
        
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>());
        let target = Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>());
        
        let feature_names: Vec<_> = (0..n_features)
            .map(|i| format!("feature_{}", i))
            .collect();
        
        let feature_types = vec![FeatureType::Continuous; n_features];
        
        let statistics = FeatureStatistics {
            means: data.mean_axis(ndarray::Axis(0)).unwrap(),
            stds: data.std_axis(ndarray::Axis(0), 0.0),
            target_correlations: Array1::from_shape_fn(n_features, |_| rng.gen::<f64>()),
            feature_correlations: Array2::from_shape_fn((n_features, n_features), |(i, j)| {
                if i == j { 1.0 } else { rng.gen::<f64>() * 0.5 }
            }),
            missing_counts: Array1::zeros(n_features),
            unique_counts: Array1::from_elem(n_features, n_samples),
        };
        
        let features = FeatureData {
            data,
            feature_names,
            target,
            feature_types,
            statistics,
        };
        
        let selector = QuantumFeatureSelector::new(
            features,
            SelectionMethod::Filter {
                metric: FilterMetric::Correlation,
                threshold: 0.3,
            },
        );
        
        let result = selector.build_qubo();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_hyperparameter_optimizer() {
        let model = MLModel {
            model_type: ModelType::RandomForest { n_trees: 100 },
            hyperparameters: HashMap::new(),
            training_params: TrainingParameters {
                learning_rate: 0.01,
                epochs: 100,
                batch_size: 32,
                early_stopping: true,
                patience: 10,
            },
        };
        
        let param_space = ParameterSpace {
            continuous: {
                let mut params = HashMap::new();
                params.insert("learning_rate".to_string(), ContinuousParam {
                    min: 0.001,
                    max: 0.1,
                    scale: ScaleType::Log,
                    default: 0.01,
                });
                params
            },
            discrete: {
                let mut params = HashMap::new();
                params.insert("n_trees".to_string(), DiscreteParam {
                    values: vec![50, 100, 200, 500],
                    default: 100,
                });
                params
            },
            categorical: HashMap::new(),
            conditional: Vec::new(),
        };
        
        let evaluation = HyperparameterEvaluation {
            metric: EvaluationMetric::Accuracy,
            cv_strategy: CrossValidationStrategy::KFold { k: 5, shuffle: true },
            constraints: ResourceConstraints {
                max_time_per_trial: None,
                max_total_time: None,
                max_memory: None,
                early_stopping: true,
            },
        };
        
        let optimizer = HyperparameterOptimizer {
            model,
            param_space,
            strategy: OptimizationStrategy::QuantumOptimization {
                tunneling_rate: 0.1,
                superposition_size: 5,
            },
            evaluation,
        };
        
        let result = optimizer.build_qubo();
        assert!(result.is_ok());
    }
}