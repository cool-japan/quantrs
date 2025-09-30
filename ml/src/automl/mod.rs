//! Quantum Automated Machine Learning (AutoML) Framework
//!
//! This module provides comprehensive automated machine learning capabilities for quantum
//! computing, including automated model selection, hyperparameter optimization, pipeline
//! construction, and quantum-specific optimizations.
//!
//! The framework has been refactored from a single 3,471-line file into a modular architecture
//! to address configuration explosion and mixed responsibilities.

pub mod analysis;
pub mod config;
pub mod pipeline;
pub mod resource;
pub mod search;

pub use config::*;

use crate::anomaly_detection::QuantumAnomalyDetector;
use crate::classification::Classifier;
use crate::clustering::QuantumClusterer;
use crate::dimensionality_reduction::QuantumDimensionalityReducer;
use crate::error::{MLError, Result};
use crate::optimization::OptimizationMethod;
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use crate::quantum_nas::{ArchitectureCandidate, QuantumNAS, SearchStrategy};
use crate::time_series::QuantumTimeSeriesForecaster;
use fastrand;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

use analysis::{AutoMLResults, PerformanceTracker};
use pipeline::{AutomatedPipelineConstructor, QuantumMLPipeline};
use resource::QuantumResourceOptimizer;
use search::{QuantumHyperparameterOptimizer, QuantumModelSelector, SearchHistory};

/// Main Quantum AutoML framework
#[derive(Debug, Clone)]
pub struct QuantumAutoML {
    /// AutoML configuration
    config: QuantumAutoMLConfig,

    /// Automated pipeline constructor
    pipeline_constructor: AutomatedPipelineConstructor,

    /// Hyperparameter optimizer
    hyperparameter_optimizer: QuantumHyperparameterOptimizer,

    /// Model selector
    model_selector: QuantumModelSelector,

    /// Ensemble manager
    ensemble_manager: QuantumEnsembleManager,

    /// Performance tracker
    performance_tracker: PerformanceTracker,

    /// Resource optimizer
    resource_optimizer: QuantumResourceOptimizer,

    /// Search history
    search_history: SearchHistory,

    /// Best pipeline found
    best_pipeline: Option<QuantumMLPipeline>,

    /// Current experiment results
    experiment_results: AutoMLResults,
}

/// Quantum ensemble manager
#[derive(Debug, Clone)]
pub struct QuantumEnsembleManager {
    /// Maximum ensemble size
    max_ensemble_size: usize,

    /// Current ensemble members
    ensemble_members: Vec<QuantumMLPipeline>,

    /// Diversity strategies
    diversity_strategies: Vec<EnsembleDiversityStrategy>,

    /// Combination methods
    combination_methods: Vec<EnsembleCombinationMethod>,

    /// Performance weights
    performance_weights: Vec<f64>,
}

impl QuantumAutoML {
    /// Create a new Quantum AutoML instance
    pub fn new(config: QuantumAutoMLConfig) -> Self {
        Self {
            pipeline_constructor: AutomatedPipelineConstructor::new(&config),
            hyperparameter_optimizer: QuantumHyperparameterOptimizer::new(
                &config.search_space.hyperparameters,
            ),
            model_selector: QuantumModelSelector::new(&config.search_space.algorithms),
            ensemble_manager: QuantumEnsembleManager::new(&config.search_space.ensembles),
            performance_tracker: PerformanceTracker::new(&config.evaluation_config),
            resource_optimizer: QuantumResourceOptimizer::new(&config.quantum_constraints),
            search_history: SearchHistory::new(),
            best_pipeline: None,
            experiment_results: AutoMLResults::new(),
            config,
        }
    }

    /// Create AutoML with basic configuration
    pub fn basic() -> Self {
        Self::new(QuantumAutoMLConfig::basic())
    }

    /// Create AutoML with comprehensive configuration
    pub fn comprehensive() -> Self {
        Self::new(QuantumAutoMLConfig::comprehensive())
    }

    /// Create AutoML with production configuration
    pub fn production() -> Self {
        Self::new(QuantumAutoMLConfig::production())
    }

    /// Fit the AutoML system to training data
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        // Task detection if not specified
        if self.config.task_type.is_none() {
            self.config.task_type = Some(self.detect_task_type(X, y)?);
        }

        // Initialize search process
        self.search_history.start_search();

        // Main AutoML search loop
        for trial_id in 0..self.config.search_budget.max_trials {
            if self.should_stop_search(trial_id)? {
                break;
            }

            // Construct candidate pipeline
            let pipeline = self
                .pipeline_constructor
                .construct_pipeline(X, y, &self.config)?;

            // Optimize hyperparameters
            let optimized_pipeline = self.hyperparameter_optimizer.optimize(pipeline, X, y)?;

            // Evaluate pipeline
            let performance = self.evaluate_pipeline(&optimized_pipeline, X, y)?;

            // Update search history
            self.search_history
                .record_trial(trial_id, &optimized_pipeline, performance);

            // Update best pipeline if better
            if self.is_better_pipeline(&optimized_pipeline, performance)? {
                self.best_pipeline = Some(optimized_pipeline.clone());
                self.performance_tracker
                    .update_best_performance(performance);
            }

            // Update ensemble if enabled
            if self.config.search_space.ensembles.enabled {
                self.ensemble_manager
                    .consider_pipeline(optimized_pipeline, performance)?;
            }
        }

        // Finalize results
        self.finalize_search()?;

        Ok(())
    }

    /// Predict using the best found pipeline
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>> {
        match &self.best_pipeline {
            Some(pipeline) => pipeline.predict(X),
            None => Err(MLError::ModelNotTrained(
                "AutoML has not been fitted yet".to_string(),
            )),
        }
    }

    /// Get the best pipeline found
    pub fn best_pipeline(&self) -> Option<&QuantumMLPipeline> {
        self.best_pipeline.as_ref()
    }

    /// Get search results and analysis
    pub fn get_results(&self) -> &AutoMLResults {
        &self.experiment_results
    }

    /// Get search history
    pub fn get_search_history(&self) -> &SearchHistory {
        &self.search_history
    }

    /// Get performance tracker
    pub fn get_performance_tracker(&self) -> &PerformanceTracker {
        &self.performance_tracker
    }

    // Private methods

    fn detect_task_type(&self, X: &Array2<f64>, y: &Array1<f64>) -> Result<MLTaskType> {
        // Simple task type detection based on target values
        let unique_values = {
            let mut values: Vec<f64> = y.iter().cloned().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();
            values
        };

        // Check if all values are integers (classification)
        let all_integers = unique_values.iter().all(|&v| (v.fract()).abs() < 1e-10);

        if all_integers && unique_values.len() <= 2 {
            Ok(MLTaskType::BinaryClassification)
        } else if all_integers && unique_values.len() <= 20 {
            Ok(MLTaskType::MultiClassification {
                num_classes: unique_values.len(),
            })
        } else {
            Ok(MLTaskType::Regression)
        }
    }

    fn should_stop_search(&self, trial_id: usize) -> Result<bool> {
        // Check time budget
        if let Some(elapsed) = self.search_history.elapsed_time() {
            if elapsed > self.config.search_budget.max_time_seconds {
                return Ok(true);
            }
        }

        // Check early stopping
        if self.config.search_budget.early_stopping.enabled {
            if let Some(best_performance) = self.performance_tracker.best_performance() {
                let trials_without_improvement = self.search_history.trials_without_improvement();
                if trials_without_improvement >= self.config.search_budget.early_stopping.patience {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn evaluate_pipeline(
        &self,
        pipeline: &QuantumMLPipeline,
        X: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<f64> {
        // Perform cross-validation evaluation
        match &self.config.evaluation_config.cv_strategy {
            CrossValidationStrategy::KFold { k } => self.evaluate_k_fold(pipeline, X, y, *k),
            CrossValidationStrategy::HoldOut { test_size } => {
                self.evaluate_holdout(pipeline, X, y, *test_size)
            }
            _ => {
                // For other strategies, use simple holdout for now
                self.evaluate_holdout(pipeline, X, y, self.config.evaluation_config.test_size)
            }
        }
    }

    fn evaluate_k_fold(
        &self,
        pipeline: &QuantumMLPipeline,
        X: &Array2<f64>,
        y: &Array1<f64>,
        k: usize,
    ) -> Result<f64> {
        let n_samples = X.nrows();
        let fold_size = n_samples / k;
        let mut scores = Vec::new();

        for fold in 0..k {
            let start_idx = fold * fold_size;
            let end_idx = if fold == k - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test split
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_samples {
                if i >= start_idx && i < end_idx {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            // Extract train/test data
            let X_train = X.select(Axis(0), &train_indices);
            let y_train = y.select(Axis(0), &train_indices);
            let X_test = X.select(Axis(0), &test_indices);
            let y_test = y.select(Axis(0), &test_indices);

            // Train and evaluate
            let mut pipeline_copy = pipeline.clone();
            pipeline_copy.fit(&X_train, &y_train)?;
            let predictions = pipeline_copy.predict(&X_test)?;

            // Calculate score based on task type
            let score = self.calculate_score(&predictions, &y_test)?;
            scores.push(score);
        }

        // Return mean score
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    fn evaluate_holdout(
        &self,
        pipeline: &QuantumMLPipeline,
        X: &Array2<f64>,
        y: &Array1<f64>,
        test_size: f64,
    ) -> Result<f64> {
        let n_samples = X.nrows();
        let n_test = (n_samples as f64 * test_size) as usize;
        let n_train = n_samples - n_test;

        // Simple train/test split
        let X_train = X.slice(s![0..n_train, ..]).to_owned();
        let y_train = y.slice(s![0..n_train]).to_owned();
        let X_test = X.slice(s![n_train.., ..]).to_owned();
        let y_test = y.slice(s![n_train..]).to_owned();

        // Train and evaluate
        let mut pipeline_copy = pipeline.clone();
        pipeline_copy.fit(&X_train, &y_train)?;
        let predictions = pipeline_copy.predict(&X_test)?;

        self.calculate_score(&predictions, &y_test)
    }

    fn calculate_score(&self, predictions: &Array1<f64>, y_true: &Array1<f64>) -> Result<f64> {
        match &self.config.task_type {
            Some(MLTaskType::BinaryClassification)
            | Some(MLTaskType::MultiClassification { .. }) => {
                // Calculate accuracy
                let correct = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(pred, true_val)| {
                        if (pred.round() - true_val).abs() < 1e-10 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>();
                Ok(correct / predictions.len() as f64)
            }
            Some(MLTaskType::Regression) => {
                // Calculate R2 score
                let mean_true = y_true.mean().unwrap();
                let ss_tot = y_true.iter().map(|&y| (y - mean_true).powi(2)).sum::<f64>();
                let ss_res = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(pred, true_val)| (true_val - pred).powi(2))
                    .sum::<f64>();
                Ok(1.0 - (ss_res / ss_tot))
            }
            _ => {
                // Default to MSE for unknown tasks
                let mse = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(pred, true_val)| (pred - true_val).powi(2))
                    .sum::<f64>()
                    / predictions.len() as f64;
                Ok(-mse) // Negative because we want to maximize
            }
        }
    }

    fn is_better_pipeline(&self, pipeline: &QuantumMLPipeline, performance: f64) -> Result<bool> {
        match self.performance_tracker.best_performance() {
            Some(best_perf) => {
                Ok(performance
                    > best_perf + self.config.search_budget.early_stopping.min_improvement)
            }
            None => Ok(true), // First pipeline is always better
        }
    }

    fn finalize_search(&mut self) -> Result<()> {
        // Generate final results
        self.experiment_results = self.generate_final_results()?;

        // Update ensemble if enabled
        if self.config.search_space.ensembles.enabled {
            self.ensemble_manager.finalize_ensemble()?;
        }

        Ok(())
    }

    fn generate_final_results(&self) -> Result<AutoMLResults> {
        let mut results = AutoMLResults::new();

        // Set best pipeline info
        if let Some(pipeline) = &self.best_pipeline {
            results.set_best_pipeline_info(pipeline);
        }

        // Set search statistics
        results.set_search_statistics(&self.search_history);

        // Set performance analysis
        results.set_performance_analysis(&self.performance_tracker);

        Ok(results)
    }
}

impl QuantumEnsembleManager {
    fn new(ensemble_config: &EnsembleSearchSpace) -> Self {
        Self {
            max_ensemble_size: ensemble_config.max_ensemble_size,
            ensemble_members: Vec::new(),
            diversity_strategies: ensemble_config.diversity_strategies.clone(),
            combination_methods: ensemble_config.combination_methods.clone(),
            performance_weights: Vec::new(),
        }
    }

    fn consider_pipeline(&mut self, pipeline: QuantumMLPipeline, performance: f64) -> Result<()> {
        // Simple strategy: keep top N performing pipelines
        if self.ensemble_members.len() < self.max_ensemble_size {
            self.ensemble_members.push(pipeline);
            self.performance_weights.push(performance);
        } else {
            // Replace worst performer if this is better
            if let Some((worst_idx, _)) = self
                .performance_weights
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                if performance > self.performance_weights[worst_idx] {
                    self.ensemble_members[worst_idx] = pipeline;
                    self.performance_weights[worst_idx] = performance;
                }
            }
        }

        Ok(())
    }

    fn finalize_ensemble(&mut self) -> Result<()> {
        // Normalize performance weights
        let total_weight: f64 = self.performance_weights.iter().sum();
        if total_weight > 0.0 {
            for weight in &mut self.performance_weights {
                *weight /= total_weight;
            }
        }

        Ok(())
    }
}

impl Default for QuantumAutoML {
    fn default() -> Self {
        Self::basic()
    }
}

// Helper functions for creating configurations

/// Create a default AutoML configuration
pub fn create_default_automl_config() -> QuantumAutoMLConfig {
    QuantumAutoMLConfig::basic()
}

/// Create a comprehensive AutoML configuration
pub fn create_comprehensive_automl_config() -> QuantumAutoMLConfig {
    QuantumAutoMLConfig::comprehensive()
}
