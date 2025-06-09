//! Scikit-learn compatibility layer for QuantRS2-ML
//!
//! This module provides a compatibility layer that mimics scikit-learn APIs,
//! allowing easy integration of quantum ML models with existing scikit-learn
//! workflows and pipelines.

use crate::error::{MLError, Result};
use crate::qsvm::{QSVM, QSVMParams, FeatureMapType};
use crate::qnn::{QuantumNeuralNetwork, QNNBuilder};
use crate::clustering::{QuantumClusterer, ClusteringAlgorithm};
use crate::classification::{Classifier, ClassificationMetrics};
use crate::simulator_backends::{SimulatorBackend, StatevectorBackend, BackendCapabilities};
use ndarray::{Array1, Array2, ArrayD, Axis};
use std::collections::HashMap;
use std::sync::Arc;

/// Base estimator trait following scikit-learn conventions
pub trait SklearnEstimator: Send + Sync {
    /// Fit the model to training data
    fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()>;
    
    /// Get model parameters
    fn get_params(&self) -> HashMap<String, String>;
    
    /// Set model parameters
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()>;
    
    /// Check if model is fitted
    fn is_fitted(&self) -> bool;
    
    /// Get feature names
    fn get_feature_names_out(&self) -> Vec<String> {
        vec![]
    }
}

/// Classifier mixin trait
pub trait SklearnClassifier: SklearnEstimator {
    /// Predict class labels
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>>;
    
    /// Predict class probabilities
    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>>;
    
    /// Get unique class labels
    fn classes(&self) -> &[i32];
    
    /// Score the model (accuracy by default)
    fn score(&self, X: &Array2<f64>, y: &Array1<i32>) -> Result<f64> {
        let predictions = self.predict(X)?;
        let correct = predictions.iter()
            .zip(y.iter())
            .filter(|(&pred, &true_label)| pred == true_label)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }
}

/// Regressor mixin trait
pub trait SklearnRegressor: SklearnEstimator {
    /// Predict continuous values
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>>;
    
    /// Score the model (R² by default)
    fn score(&self, X: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(X)?;
        let y_mean = y.mean().unwrap();
        
        let ss_res: f64 = y.iter()
            .zip(predictions.iter())
            .map(|(&true_val, &pred)| (true_val - pred).powi(2))
            .sum();
        
        let ss_tot: f64 = y.iter()
            .map(|&val| (val - y_mean).powi(2))
            .sum();
        
        Ok(1.0 - ss_res / ss_tot)
    }
}

/// Clusterer mixin trait
pub trait SklearnClusterer: SklearnEstimator {
    /// Predict cluster labels
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>>;
    
    /// Fit and predict in one step
    fn fit_predict(&mut self, X: &Array2<f64>) -> Result<Array1<i32>> {
        self.fit(X, None)?;
        self.predict(X)
    }
    
    /// Get cluster centers (if applicable)
    fn cluster_centers(&self) -> Option<&Array2<f64>> {
        None
    }
}

/// Quantum Support Vector Machine (sklearn-compatible)
pub struct QuantumSVC {
    /// Internal QSVM
    qsvm: Option<QSVM>,
    /// SVM parameters
    params: QSVMParams,
    /// Feature map type
    feature_map: FeatureMapType,
    /// Backend
    backend: Arc<dyn SimulatorBackend>,
    /// Fitted flag
    fitted: bool,
    /// Unique classes
    classes: Vec<i32>,
    /// Regularization parameter
    C: f64,
    /// Kernel gamma parameter
    gamma: f64,
}

impl QuantumSVC {
    /// Create new Quantum SVC
    pub fn new() -> Self {
        Self {
            qsvm: None,
            params: QSVMParams::default(),
            feature_map: FeatureMapType::ZZFeatureMap,
            backend: Arc::new(StatevectorBackend::new(BackendCapabilities::default())),
            fitted: false,
            classes: Vec::new(),
            C: 1.0,
            gamma: 1.0,
        }
    }
    
    /// Set regularization parameter
    pub fn set_C(mut self, C: f64) -> Self {
        self.C = C;
        self
    }
    
    /// Set kernel gamma parameter
    pub fn set_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }
    
    /// Set feature map
    pub fn set_kernel(mut self, feature_map: FeatureMapType) -> Self {
        self.feature_map = feature_map;
        self
    }
    
    /// Set quantum backend
    pub fn set_backend(mut self, backend: Arc<dyn SimulatorBackend>) -> Self {
        self.backend = backend;
        self
    }
}

impl SklearnEstimator for QuantumSVC {
    fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()> {
        let y = y.ok_or_else(|| MLError::InvalidConfiguration(
            "Labels required for supervised learning".to_string(),
        ))?;
        
        // Convert continuous labels to integer classes
        let y_int: Array1<i32> = y.mapv(|val| val.round() as i32);
        
        // Find unique classes
        let mut classes = Vec::new();
        for &label in y_int.iter() {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        classes.sort();
        self.classes = classes;
        
        // Update QSVM parameters
        self.params.feature_map = self.feature_map;
        self.params.regularization = self.C;
        
        // Create and train QSVM
        let mut qsvm = QSVM::new(self.params.clone());
        qsvm.fit(X, &y_int)?;
        
        self.qsvm = Some(qsvm);
        self.fitted = true;
        
        Ok(())
    }
    
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("C".to_string(), self.C.to_string());
        params.insert("gamma".to_string(), self.gamma.to_string());
        params.insert("kernel".to_string(), format!("{:?}", self.feature_map));
        params
    }
    
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "C" => {
                    self.C = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid C parameter: {}", value),
                    ))?;
                }
                "gamma" => {
                    self.gamma = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid gamma parameter: {}", value),
                    ))?;
                }
                "kernel" => {
                    self.feature_map = match value.as_str() {
                        "ZZFeatureMap" => FeatureMapType::ZZFeatureMap,
                        "ZFeatureMap" => FeatureMapType::ZFeatureMap,
                        "PauliFeatureMap" => FeatureMapType::PauliFeatureMap,
                        _ => return Err(MLError::InvalidConfiguration(
                            format!("Unknown kernel: {}", value),
                        )),
                    };
                }
                _ => return Err(MLError::InvalidConfiguration(
                    format!("Unknown parameter: {}", key),
                )),
            }
        }
        Ok(())
    }
    
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClassifier for QuantumSVC {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained);
        }
        
        let qsvm = self.qsvm.as_ref().unwrap();
        qsvm.predict(X)
    }
    
    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained);
        }
        
        let predictions = self.predict(X)?;
        let n_samples = X.nrows();
        let n_classes = self.classes.len();
        
        let mut probabilities = Array2::zeros((n_samples, n_classes));
        
        // Convert hard predictions to probabilities (placeholder)
        for (i, &prediction) in predictions.iter().enumerate() {
            for (j, &class) in self.classes.iter().enumerate() {
                probabilities[[i, j]] = if prediction == class { 1.0 } else { 0.0 };
            }
        }
        
        Ok(probabilities)
    }
    
    fn classes(&self) -> &[i32] {
        &self.classes
    }
}

/// Quantum Neural Network Classifier (sklearn-compatible)
pub struct QuantumMLPClassifier {
    /// Internal QNN
    qnn: Option<QuantumNeuralNetwork>,
    /// Network configuration
    hidden_layer_sizes: Vec<usize>,
    /// Activation function
    activation: String,
    /// Solver
    solver: String,
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Random state
    random_state: Option<u64>,
    /// Backend
    backend: Arc<dyn SimulatorBackend>,
    /// Fitted flag
    fitted: bool,
    /// Unique classes
    classes: Vec<i32>,
}

impl QuantumMLPClassifier {
    /// Create new Quantum MLP Classifier
    pub fn new() -> Self {
        Self {
            qnn: None,
            hidden_layer_sizes: vec![10],
            activation: "relu".to_string(),
            solver: "adam".to_string(),
            learning_rate: 0.001,
            max_iter: 200,
            random_state: None,
            backend: Arc::new(StatevectorBackend::new(BackendCapabilities::default())),
            fitted: false,
            classes: Vec::new(),
        }
    }
    
    /// Set hidden layer sizes
    pub fn set_hidden_layer_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_layer_sizes = sizes;
        self
    }
    
    /// Set activation function
    pub fn set_activation(mut self, activation: String) -> Self {
        self.activation = activation;
        self
    }
    
    /// Set learning rate
    pub fn set_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    
    /// Set maximum iterations
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl SklearnEstimator for QuantumMLPClassifier {
    fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()> {
        let y = y.ok_or_else(|| MLError::InvalidConfiguration(
            "Labels required for supervised learning".to_string(),
        ))?;
        
        // Convert continuous labels to integer classes
        let y_int: Array1<i32> = y.mapv(|val| val.round() as i32);
        
        // Find unique classes
        let mut classes = Vec::new();
        for &label in y_int.iter() {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        classes.sort();
        self.classes = classes;
        
        // Build QNN
        let input_size = X.ncols();
        let output_size = self.classes.len();
        
        let mut builder = QNNBuilder::new(input_size);
        
        // Add hidden layers
        for &size in &self.hidden_layer_sizes {
            builder = builder.add_layer(size);
        }
        
        // Add output layer
        builder = builder.add_layer(output_size);
        
        let mut qnn = builder.build()?;
        
        // Train QNN
        let y_one_hot = self.to_one_hot(&y_int)?;
        qnn.train(X, &y_one_hot, self.max_iter, self.learning_rate)?;
        
        self.qnn = Some(qnn);
        self.fitted = true;
        
        Ok(())
    }
    
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("hidden_layer_sizes".to_string(), format!("{:?}", self.hidden_layer_sizes));
        params.insert("activation".to_string(), self.activation.clone());
        params.insert("solver".to_string(), self.solver.clone());
        params.insert("learning_rate".to_string(), self.learning_rate.to_string());
        params.insert("max_iter".to_string(), self.max_iter.to_string());
        params
    }
    
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "learning_rate" => {
                    self.learning_rate = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid learning_rate: {}", value),
                    ))?;
                }
                "max_iter" => {
                    self.max_iter = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid max_iter: {}", value),
                    ))?;
                }
                "activation" => {
                    self.activation = value;
                }
                "solver" => {
                    self.solver = value;
                }
                _ => {
                    // Skip unknown parameters
                }
            }
        }
        Ok(())
    }
    
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl QuantumMLPClassifier {
    /// Convert integer labels to one-hot encoding
    fn to_one_hot(&self, y: &Array1<i32>) -> Result<Array2<f64>> {
        let n_samples = y.len();
        let n_classes = self.classes.len();
        let mut one_hot = Array2::zeros((n_samples, n_classes));
        
        for (i, &label) in y.iter().enumerate() {
            if let Some(class_idx) = self.classes.iter().position(|&c| c == label) {
                one_hot[[i, class_idx]] = 1.0;
            }
        }
        
        Ok(one_hot)
    }
    
    /// Convert one-hot predictions to class labels
    fn from_one_hot(&self, predictions: &Array2<f64>) -> Array1<i32> {
        predictions.axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.classes[max_idx]
            })
            .collect()
    }
}

impl SklearnClassifier for QuantumMLPClassifier {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained);
        }
        
        let qnn = self.qnn.as_ref().unwrap();
        let predictions = qnn.predict(X)?;
        Ok(self.from_one_hot(&predictions))
    }
    
    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained);
        }
        
        let qnn = self.qnn.as_ref().unwrap();
        qnn.predict(X)
    }
    
    fn classes(&self) -> &[i32] {
        &self.classes
    }
}

/// Quantum Regressor (sklearn-compatible)
pub struct QuantumMLPRegressor {
    /// Internal QNN
    qnn: Option<QuantumNeuralNetwork>,
    /// Network configuration
    hidden_layer_sizes: Vec<usize>,
    /// Activation function
    activation: String,
    /// Solver
    solver: String,
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Random state
    random_state: Option<u64>,
    /// Backend
    backend: Arc<dyn SimulatorBackend>,
    /// Fitted flag
    fitted: bool,
}

impl QuantumMLPRegressor {
    /// Create new Quantum MLP Regressor
    pub fn new() -> Self {
        Self {
            qnn: None,
            hidden_layer_sizes: vec![10],
            activation: "relu".to_string(),
            solver: "adam".to_string(),
            learning_rate: 0.001,
            max_iter: 200,
            random_state: None,
            backend: Arc::new(StatevectorBackend::new(BackendCapabilities::default())),
            fitted: false,
        }
    }
    
    /// Set hidden layer sizes
    pub fn set_hidden_layer_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_layer_sizes = sizes;
        self
    }
    
    /// Set learning rate
    pub fn set_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    
    /// Set maximum iterations
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl SklearnEstimator for QuantumMLPRegressor {
    fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()> {
        let y = y.ok_or_else(|| MLError::InvalidConfiguration(
            "Target values required for regression".to_string(),
        ))?;
        
        // Build QNN for regression
        let input_size = X.ncols();
        let output_size = 1; // Single output for regression
        
        let mut builder = QNNBuilder::new(input_size);
        
        // Add hidden layers
        for &size in &self.hidden_layer_sizes {
            builder = builder.add_layer(size);
        }
        
        // Add output layer
        builder = builder.add_layer(output_size);
        
        let mut qnn = builder.build()?;
        
        // Reshape target for training
        let y_reshaped = y.clone().into_shape((y.len(), 1)).unwrap();
        
        // Train QNN
        qnn.train(X, &y_reshaped, self.max_iter, self.learning_rate)?;
        
        self.qnn = Some(qnn);
        self.fitted = true;
        
        Ok(())
    }
    
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("hidden_layer_sizes".to_string(), format!("{:?}", self.hidden_layer_sizes));
        params.insert("activation".to_string(), self.activation.clone());
        params.insert("solver".to_string(), self.solver.clone());
        params.insert("learning_rate".to_string(), self.learning_rate.to_string());
        params.insert("max_iter".to_string(), self.max_iter.to_string());
        params
    }
    
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "learning_rate" => {
                    self.learning_rate = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid learning_rate: {}", value),
                    ))?;
                }
                "max_iter" => {
                    self.max_iter = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid max_iter: {}", value),
                    ))?;
                }
                "activation" => {
                    self.activation = value;
                }
                "solver" => {
                    self.solver = value;
                }
                _ => {
                    // Skip unknown parameters
                }
            }
        }
        Ok(())
    }
    
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnRegressor for QuantumMLPRegressor {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained);
        }
        
        let qnn = self.qnn.as_ref().unwrap();
        let predictions = qnn.predict(X)?;
        
        // Extract single column for regression
        Ok(predictions.column(0).to_owned())
    }
}

/// Quantum K-Means (sklearn-compatible)
pub struct QuantumKMeans {
    /// Internal clusterer
    clusterer: Option<QuantumClusterer>,
    /// Number of clusters
    n_clusters: usize,
    /// Maximum iterations
    max_iter: usize,
    /// Tolerance
    tol: f64,
    /// Random state
    random_state: Option<u64>,
    /// Backend
    backend: Arc<dyn SimulatorBackend>,
    /// Fitted flag
    fitted: bool,
    /// Cluster centers
    cluster_centers_: Option<Array2<f64>>,
    /// Labels
    labels_: Option<Array1<i32>>,
}

impl QuantumKMeans {
    /// Create new Quantum K-Means
    pub fn new(n_clusters: usize) -> Self {
        Self {
            clusterer: None,
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            random_state: None,
            backend: Arc::new(StatevectorBackend::new(BackendCapabilities::default())),
            fitted: false,
            cluster_centers_: None,
            labels_: None,
        }
    }
    
    /// Set maximum iterations
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    
    /// Set tolerance
    pub fn set_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    
    /// Set random state
    pub fn set_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl SklearnEstimator for QuantumKMeans {
    fn fit(&mut self, X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        let mut clusterer = QuantumClusterer::new(
            ClusteringAlgorithm::QuantumKMeans,
            self.n_clusters,
        );
        
        let result = clusterer.fit_predict(X)?;
        self.labels_ = Some(result.labels);
        self.cluster_centers_ = result.cluster_centers;
        
        self.clusterer = Some(clusterer);
        self.fitted = true;
        
        Ok(())
    }
    
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_clusters".to_string(), self.n_clusters.to_string());
        params.insert("max_iter".to_string(), self.max_iter.to_string());
        params.insert("tol".to_string(), self.tol.to_string());
        if let Some(rs) = self.random_state {
            params.insert("random_state".to_string(), rs.to_string());
        }
        params
    }
    
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "n_clusters" => {
                    self.n_clusters = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid n_clusters: {}", value),
                    ))?;
                }
                "max_iter" => {
                    self.max_iter = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid max_iter: {}", value),
                    ))?;
                }
                "tol" => {
                    self.tol = value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid tol: {}", value),
                    ))?;
                }
                "random_state" => {
                    self.random_state = Some(value.parse().map_err(|_| MLError::InvalidConfiguration(
                        format!("Invalid random_state: {}", value),
                    ))?);
                }
                _ => {
                    // Skip unknown parameters
                }
            }
        }
        Ok(())
    }
    
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClusterer for QuantumKMeans {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained);
        }
        
        let clusterer = self.clusterer.as_ref().unwrap();
        clusterer.predict(X)
    }
    
    fn cluster_centers(&self) -> Option<&Array2<f64>> {
        self.cluster_centers_.as_ref()
    }
}

/// Model selection utilities (sklearn-compatible)
pub mod model_selection {
    use super::*;
    use rand::seq::SliceRandom;
    
    /// Cross-validation score
    pub fn cross_val_score<E>(
        estimator: &mut E,
        X: &Array2<f64>,
        y: &Array1<f64>,
        cv: usize,
    ) -> Result<Array1<f64>>
    where
        E: SklearnClassifier,
    {
        let n_samples = X.nrows();
        let fold_size = n_samples / cv;
        let mut scores = Array1::zeros(cv);
        
        // Create fold indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rand::thread_rng());
        
        for fold in 0..cv {
            let start_test = fold * fold_size;
            let end_test = if fold == cv - 1 { n_samples } else { (fold + 1) * fold_size };
            
            // Create train/test splits
            let test_indices = &indices[start_test..end_test];
            let train_indices: Vec<usize> = indices.iter()
                .enumerate()
                .filter(|(i, _)| *i < start_test || *i >= end_test)
                .map(|(_, &idx)| idx)
                .collect();
            
            // Extract train/test data
            let X_train = X.select(Axis(0), &train_indices);
            let y_train = y.select(Axis(0), &train_indices);
            let X_test = X.select(Axis(0), test_indices);
            let y_test = y.select(Axis(0), test_indices);
            
            // Convert to i32 for classification
            let y_train_int = y_train.mapv(|x| x.round() as i32);
            let y_test_int = y_test.mapv(|x| x.round() as i32);
            
            // Train and evaluate
            estimator.fit(&X_train, Some(&y_train))?;
            scores[fold] = estimator.score(&X_test, &y_test_int)?;
        }
        
        Ok(scores)
    }
    
    /// Train-test split
    pub fn train_test_split(
        X: &Array2<f64>,
        y: &Array1<f64>,
        test_size: f64,
        random_state: Option<u64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>)> {
        let n_samples = X.nrows();
        let n_test = (n_samples as f64 * test_size).round() as usize;
        
        // Create indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        if let Some(seed) = random_state {
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        } else {
            indices.shuffle(&mut rand::thread_rng());
        }
        
        let test_indices = &indices[..n_test];
        let train_indices = &indices[n_test..];
        
        let X_train = X.select(Axis(0), train_indices);
        let X_test = X.select(Axis(0), test_indices);
        let y_train = y.select(Axis(0), train_indices);
        let y_test = y.select(Axis(0), test_indices);
        
        Ok((X_train, X_test, y_train, y_test))
    }
    
    /// Grid search for hyperparameter tuning
    pub struct GridSearchCV<E> {
        /// Base estimator
        estimator: E,
        /// Parameter grid
        param_grid: HashMap<String, Vec<String>>,
        /// Cross-validation folds
        cv: usize,
        /// Best parameters
        best_params_: Option<HashMap<String, String>>,
        /// Best score
        best_score_: f64,
        /// Fitted flag
        fitted: bool,
    }
    
    impl<E> GridSearchCV<E>
    where
        E: SklearnClassifier + Clone,
    {
        /// Create new grid search
        pub fn new(estimator: E, param_grid: HashMap<String, Vec<String>>, cv: usize) -> Self {
            Self {
                estimator,
                param_grid,
                cv,
                best_params_: None,
                best_score_: f64::NEG_INFINITY,
                fitted: false,
            }
        }
        
        /// Fit grid search
        pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
            let param_combinations = self.generate_param_combinations();
            
            for params in param_combinations {
                let mut estimator = self.estimator.clone();
                estimator.set_params(params.clone())?;
                
                let scores = cross_val_score(&mut estimator, X, y, self.cv)?;
                let mean_score = scores.mean().unwrap();
                
                if mean_score > self.best_score_ {
                    self.best_score_ = mean_score;
                    self.best_params_ = Some(params);
                }
            }
            
            // Fit best estimator
            if let Some(ref best_params) = self.best_params_ {
                self.estimator.set_params(best_params.clone())?;
                self.estimator.fit(X, Some(y))?;
            }
            
            self.fitted = true;
            Ok(())
        }
        
        /// Generate all parameter combinations
        fn generate_param_combinations(&self) -> Vec<HashMap<String, String>> {
            let mut combinations = vec![HashMap::new()];
            
            for (param_name, param_values) in &self.param_grid {
                let mut new_combinations = Vec::new();
                
                for combination in &combinations {
                    for value in param_values {
                        let mut new_combination = combination.clone();
                        new_combination.insert(param_name.clone(), value.clone());
                        new_combinations.push(new_combination);
                    }
                }
                
                combinations = new_combinations;
            }
            
            combinations
        }
        
        /// Get best parameters
        pub fn best_params(&self) -> Option<&HashMap<String, String>> {
            self.best_params_.as_ref()
        }
        
        /// Get best score
        pub fn best_score(&self) -> f64 {
            self.best_score_
        }
        
        /// Predict with best estimator
        pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
            if !self.fitted {
                return Err(MLError::ModelNotTrained);
            }
            self.estimator.predict(X)
        }
    }
}

/// Pipeline utilities (sklearn-compatible)
pub mod pipeline {
    use super::*;
    
    /// Transformer trait
    pub trait SklearnTransformer: Send + Sync {
        /// Fit transformer
        fn fit(&mut self, X: &Array2<f64>) -> Result<()>;
        
        /// Transform data
        fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>>;
        
        /// Fit and transform
        fn fit_transform(&mut self, X: &Array2<f64>) -> Result<Array2<f64>> {
            self.fit(X)?;
            self.transform(X)
        }
    }
    
    /// Quantum feature scaler
    pub struct QuantumStandardScaler {
        /// Feature means
        mean_: Option<Array1<f64>>,
        /// Feature standard deviations
        scale_: Option<Array1<f64>>,
        /// Fitted flag
        fitted: bool,
    }
    
    impl QuantumStandardScaler {
        /// Create new scaler
        pub fn new() -> Self {
            Self {
                mean_: None,
                scale_: None,
                fitted: false,
            }
        }
    }
    
    impl SklearnTransformer for QuantumStandardScaler {
        fn fit(&mut self, X: &Array2<f64>) -> Result<()> {
            let mean = X.mean_axis(Axis(0)).unwrap();
            let std = X.std_axis(Axis(0), 0.0);
            
            self.mean_ = Some(mean);
            self.scale_ = Some(std);
            self.fitted = true;
            
            Ok(())
        }
        
        fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
            if !self.fitted {
                return Err(MLError::ModelNotTrained);
            }
            
            let mean = self.mean_.as_ref().unwrap();
            let scale = self.scale_.as_ref().unwrap();
            
            let mut X_scaled = X.clone();
            for mut row in X_scaled.axis_iter_mut(Axis(0)) {
                row -= mean;
                row /= scale;
            }
            
            Ok(X_scaled)
        }
    }
    
    /// Quantum pipeline
    pub struct QuantumPipeline {
        /// Pipeline steps
        steps: Vec<(String, PipelineStep)>,
        /// Fitted flag
        fitted: bool,
    }
    
    /// Pipeline step enum
    pub enum PipelineStep {
        /// Transformer step
        Transformer(Box<dyn SklearnTransformer>),
        /// Classifier step
        Classifier(Box<dyn SklearnClassifier>),
        /// Regressor step
        Regressor(Box<dyn SklearnRegressor>),
        /// Clusterer step
        Clusterer(Box<dyn SklearnClusterer>),
    }
    
    impl QuantumPipeline {
        /// Create new pipeline
        pub fn new() -> Self {
            Self {
                steps: Vec::new(),
                fitted: false,
            }
        }
        
        /// Add transformer step
        pub fn add_transformer(mut self, name: String, transformer: Box<dyn SklearnTransformer>) -> Self {
            self.steps.push((name, PipelineStep::Transformer(transformer)));
            self
        }
        
        /// Add classifier step
        pub fn add_classifier(mut self, name: String, classifier: Box<dyn SklearnClassifier>) -> Self {
            self.steps.push((name, PipelineStep::Classifier(classifier)));
            self
        }
        
        /// Fit pipeline
        pub fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()> {
            let mut current_X = X.clone();
            
            for (_name, step) in &mut self.steps {
                match step {
                    PipelineStep::Transformer(transformer) => {
                        current_X = transformer.fit_transform(&current_X)?;
                    }
                    PipelineStep::Classifier(classifier) => {
                        classifier.fit(&current_X, y)?;
                    }
                    PipelineStep::Regressor(regressor) => {
                        regressor.fit(&current_X, y)?;
                    }
                    PipelineStep::Clusterer(clusterer) => {
                        clusterer.fit(&current_X, y)?;
                    }
                }
            }
            
            self.fitted = true;
            Ok(())
        }
        
        /// Predict with pipeline
        pub fn predict(&self, X: &Array2<f64>) -> Result<ArrayD<f64>> {
            if !self.fitted {
                return Err(MLError::ModelNotTrained);
            }
            
            let mut current_X = X.clone();
            
            for (_name, step) in &self.steps {
                match step {
                    PipelineStep::Transformer(transformer) => {
                        current_X = transformer.transform(&current_X)?;
                    }
                    PipelineStep::Classifier(classifier) => {
                        let predictions = classifier.predict(&current_X)?;
                        return Ok(predictions.into_dyn());
                    }
                    PipelineStep::Regressor(regressor) => {
                        let predictions = regressor.predict(&current_X)?;
                        return Ok(predictions.into_dyn());
                    }
                    PipelineStep::Clusterer(clusterer) => {
                        let predictions = clusterer.predict(&current_X)?;
                        return Ok(predictions.into_dyn());
                    }
                }
            }
            
            Ok(current_X.into_dyn())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_quantum_svc() {
        let mut svc = QuantumSVC::new()
            .set_C(1.0)
            .set_gamma(0.1);
        
        let X = Array::from_shape_vec((4, 2), vec![
            1.0, 1.0,
            1.0, -1.0,
            -1.0, 1.0,
            -1.0, -1.0,
        ]).unwrap();
        let y = Array::from_vec(vec![1.0, -1.0, -1.0, 1.0]);
        
        assert!(svc.fit(&X, Some(&y)).is_ok());
        assert!(svc.is_fitted());
        
        let predictions = svc.predict(&X);
        assert!(predictions.is_ok());
    }
    
    #[test]
    fn test_quantum_mlp_classifier() {
        let mut mlp = QuantumMLPClassifier::new()
            .set_hidden_layer_sizes(vec![5])
            .set_learning_rate(0.01)
            .set_max_iter(10);
        
        let X = Array::from_shape_vec((4, 2), vec![
            1.0, 1.0,
            1.0, -1.0,
            -1.0, 1.0,
            -1.0, -1.0,
        ]).unwrap();
        let y = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]);
        
        assert!(mlp.fit(&X, Some(&y)).is_ok());
        assert!(mlp.is_fitted());
        
        let predictions = mlp.predict(&X);
        assert!(predictions.is_ok());
        
        let probas = mlp.predict_proba(&X);
        assert!(probas.is_ok());
    }
    
    #[test]
    fn test_quantum_kmeans() {
        let mut kmeans = QuantumKMeans::new(2)
            .set_max_iter(50)
            .set_tol(1e-4);
        
        let X = Array::from_shape_vec((4, 2), vec![
            1.0, 1.0,
            1.1, 1.1,
            -1.0, -1.0,
            -1.1, -1.1,
        ]).unwrap();
        
        assert!(kmeans.fit(&X, None).is_ok());
        assert!(kmeans.is_fitted());
        
        let predictions = kmeans.predict(&X);
        assert!(predictions.is_ok());
        
        assert!(kmeans.cluster_centers().is_some());
    }
    
    #[test]
    fn test_model_selection() {
        use model_selection::train_test_split;
        
        let X = Array::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = Array::from_vec((0..10).map(|x| x as f64).collect());
        
        let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.3, Some(42)).unwrap();
        
        assert_eq!(X_train.nrows() + X_test.nrows(), X.nrows());
        assert_eq!(y_train.len() + y_test.len(), y.len());
    }
    
    #[test]
    fn test_pipeline() {
        use pipeline::{QuantumPipeline, QuantumStandardScaler};
        
        let mut pipeline = QuantumPipeline::new()
            .add_transformer("scaler".to_string(), Box::new(QuantumStandardScaler::new()));
        
        let X = Array::from_shape_vec((4, 2), vec![
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
            4.0, 4.0,
        ]).unwrap();
        
        assert!(pipeline.fit(&X, None).is_ok());
        
        let transformed = pipeline.predict(&X);
        assert!(transformed.is_ok());
    }
}