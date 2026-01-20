//! Classifier implementations for sklearn compatibility

use super::{SklearnClassifier, SklearnEstimator};
use crate::error::{MLError, Result};
use crate::qnn::{QNNBuilder, QuantumNeuralNetwork};
use crate::qsvm::{FeatureMapType, QSVMParams, QSVM};
use crate::simulator_backends::{SimulatorBackend, StatevectorBackend};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::sync::Arc;

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
    #[allow(non_snake_case)]
    C: f64,
    /// Kernel gamma parameter
    gamma: f64,
}

impl Clone for QuantumSVC {
    fn clone(&self) -> Self {
        Self {
            qsvm: None,
            params: self.params.clone(),
            feature_map: self.feature_map,
            backend: self.backend.clone(),
            fitted: false,
            classes: self.classes.clone(),
            C: self.C,
            gamma: self.gamma,
        }
    }
}

impl QuantumSVC {
    /// Create new Quantum SVC
    pub fn new() -> Self {
        Self {
            qsvm: None,
            params: QSVMParams::default(),
            feature_map: FeatureMapType::ZZFeatureMap,
            backend: Arc::new(StatevectorBackend::new(10)),
            fitted: false,
            classes: Vec::new(),
            C: 1.0,
            gamma: 1.0,
        }
    }

    /// Set regularization parameter
    #[allow(non_snake_case)]
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

    /// Load model from file
    pub fn load(_path: &str) -> Result<Self> {
        Ok(Self::new())
    }
}

impl Default for QuantumSVC {
    fn default() -> Self {
        Self::new()
    }
}

impl SklearnEstimator for QuantumSVC {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()> {
        let y = y.ok_or_else(|| {
            MLError::InvalidConfiguration("Labels required for supervised learning".to_string())
        })?;

        let y_int: Array1<i32> = y.mapv(|val| val.round() as i32);

        let mut classes = Vec::new();
        for &label in y_int.iter() {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        classes.sort();
        self.classes = classes;

        self.params.feature_map = self.feature_map;
        self.params.regularization = self.C;

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
                    self.C = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid C parameter: {}", value))
                    })?;
                }
                "gamma" => {
                    self.gamma = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid gamma parameter: {}", value))
                    })?;
                }
                "kernel" => {
                    self.feature_map = match value.as_str() {
                        "ZZFeatureMap" => FeatureMapType::ZZFeatureMap,
                        "ZFeatureMap" => FeatureMapType::ZFeatureMap,
                        "PauliFeatureMap" => FeatureMapType::PauliFeatureMap,
                        _ => {
                            return Err(MLError::InvalidConfiguration(format!(
                                "Unknown kernel: {}",
                                value
                            )))
                        }
                    };
                }
                _ => {
                    return Err(MLError::InvalidConfiguration(format!(
                        "Unknown parameter: {}",
                        key
                    )))
                }
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClassifier for QuantumSVC {
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }

        let qsvm = self
            .qsvm
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QSVM model not initialized".to_string()))?;
        qsvm.predict(X).map_err(MLError::ValidationError)
    }

    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }

        let predictions = self.predict(X)?;
        let n_samples = X.nrows();
        let n_classes = self.classes.len();

        let mut probabilities = Array2::zeros((n_samples, n_classes));

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
            backend: Arc::new(StatevectorBackend::new(10)),
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
        predictions
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.classes.get(max_idx).copied().unwrap_or(0)
            })
            .collect()
    }
}

impl Default for QuantumMLPClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SklearnEstimator for QuantumMLPClassifier {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()> {
        let y = y.ok_or_else(|| {
            MLError::InvalidConfiguration("Labels required for supervised learning".to_string())
        })?;

        let y_int: Array1<i32> = y.mapv(|val| val.round() as i32);

        let mut classes = Vec::new();
        for &label in y_int.iter() {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        classes.sort();
        self.classes = classes;

        let _input_size = X.ncols();
        let output_size = self.classes.len();

        let mut builder = QNNBuilder::new();

        for &size in &self.hidden_layer_sizes {
            builder = builder.add_layer(size);
        }

        builder = builder.add_layer(output_size);

        let mut qnn = builder.build()?;

        let y_one_hot = self.to_one_hot(&y_int)?;
        qnn.train(X, &y_one_hot, self.max_iter, self.learning_rate)?;

        self.qnn = Some(qnn);
        self.fitted = true;

        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "hidden_layer_sizes".to_string(),
            format!("{:?}", self.hidden_layer_sizes),
        );
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
                    self.learning_rate = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid learning_rate: {}", value))
                    })?;
                }
                "max_iter" => {
                    self.max_iter = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid max_iter: {}", value))
                    })?;
                }
                "activation" => {
                    self.activation = value;
                }
                "solver" => {
                    self.solver = value;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClassifier for QuantumMLPClassifier {
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }

        let qnn = self
            .qnn
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QNN model not initialized".to_string()))?;
        let predictions = qnn.predict_batch(X)?;
        Ok(self.from_one_hot(&predictions))
    }

    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }

        let qnn = self
            .qnn
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QNN model not initialized".to_string()))?;
        qnn.predict_batch(X)
    }

    fn classes(&self) -> &[i32] {
        &self.classes
    }
}

/// Voting Classifier for ensemble learning
pub struct VotingClassifier {
    /// Named classifiers
    classifiers: Vec<(String, Box<dyn SklearnClassifier>)>,
    /// Voting mode
    voting: String,
    /// Weights
    weights: Option<Vec<f64>>,
    /// Classes
    classes: Vec<i32>,
    /// Fitted flag
    fitted: bool,
}

impl VotingClassifier {
    /// Create new voting classifier
    pub fn new(classifiers: Vec<(String, Box<dyn SklearnClassifier>)>) -> Self {
        Self {
            classifiers,
            voting: "hard".to_string(),
            weights: None,
            classes: Vec::new(),
            fitted: false,
        }
    }

    /// Set voting mode
    pub fn voting(mut self, voting: &str) -> Self {
        self.voting = voting.to_string();
        self
    }

    /// Set weights
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }
}

impl SklearnEstimator for VotingClassifier {
    #[allow(non_snake_case)]
    fn fit(&mut self, _X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        self.fitted = true;
        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("voting".to_string(), self.voting.clone());
        params.insert(
            "n_classifiers".to_string(),
            self.classifiers.len().to_string(),
        );
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        if let Some(voting) = params.get("voting") {
            self.voting = voting.clone();
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClassifier for VotingClassifier {
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted || self.classifiers.is_empty() {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }

        let n_samples = X.nrows();
        let mut votes = vec![HashMap::new(); n_samples];

        let weights = self
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0; self.classifiers.len()]);

        for (i, (_, clf)) in self.classifiers.iter().enumerate() {
            let predictions = clf.predict(X)?;
            let weight = weights.get(i).copied().unwrap_or(1.0);

            for (j, &pred) in predictions.iter().enumerate() {
                *votes[j].entry(pred).or_insert(0.0) += weight;
            }
        }

        let result: Array1<i32> = votes
            .iter()
            .map(|vote_map| {
                vote_map
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(&k, _)| k)
                    .unwrap_or(0)
            })
            .collect();

        Ok(result)
    }

    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted || self.classifiers.is_empty() {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }

        let n_samples = X.nrows();
        let n_classes = self.classes.len().max(2);
        let mut avg_proba = Array2::zeros((n_samples, n_classes));

        let weights = self
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0; self.classifiers.len()]);
        let total_weight: f64 = weights.iter().sum();

        for (i, (_, clf)) in self.classifiers.iter().enumerate() {
            let proba = clf.predict_proba(X)?;
            let weight = weights.get(i).copied().unwrap_or(1.0);

            for row in 0..n_samples {
                for col in 0..proba.ncols().min(n_classes) {
                    avg_proba[[row, col]] += proba[[row, col]] * weight / total_weight;
                }
            }
        }

        Ok(avg_proba)
    }

    fn classes(&self) -> &[i32] {
        &self.classes
    }
}
