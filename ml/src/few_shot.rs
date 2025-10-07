//! Quantum Few-Shot Learning
//!
//! This module implements quantum few-shot learning algorithms that enable quantum models
//! to learn from very limited training examples. It includes support for meta-learning,
//! prototypical networks, and metric learning approaches adapted for quantum circuits.

use crate::autodiff::optimizers::Optimizer;
use crate::error::{MLError, Result};
use crate::kernels::QuantumKernel;
use crate::optimization::OptimizationMethod;
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::gate::{
    single::{RotationX, RotationY, RotationZ},
    GateOp,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::SliceRandomExt;
use std::collections::HashMap;

/// Few-shot learning algorithm types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FewShotMethod {
    /// Prototypical networks in quantum feature space
    PrototypicalNetworks,

    /// Model-Agnostic Meta-Learning (MAML) for quantum circuits
    MAML { inner_steps: usize, inner_lr: f64 },

    /// Metric learning with quantum kernels
    MetricLearning,

    /// Siamese networks with quantum encoders
    SiameseNetworks,

    /// Matching networks with quantum attention
    MatchingNetworks,
}

/// Episode configuration for few-shot learning
#[derive(Debug, Clone)]
pub struct Episode {
    /// Support set (few labeled examples per class)
    pub support_set: Vec<(Array1<f64>, usize)>,

    /// Query set (examples to classify)
    pub query_set: Vec<(Array1<f64>, usize)>,

    /// Number of classes in this episode (N-way)
    pub num_classes: usize,

    /// Number of examples per class in support set (K-shot)
    pub k_shot: usize,
}

/// Quantum prototypical network for few-shot learning
pub struct QuantumPrototypicalNetwork {
    /// Quantum encoder network
    encoder: QuantumNeuralNetwork,

    /// Feature dimension in quantum space
    feature_dim: usize,

    /// Distance metric to use
    distance_metric: DistanceMetric,
}

/// Distance metrics for prototype comparison
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance in feature space
    Euclidean,

    /// Cosine similarity
    Cosine,

    /// Quantum kernel distance
    QuantumKernel,
}

impl QuantumPrototypicalNetwork {
    /// Create a new quantum prototypical network
    pub fn new(
        encoder: QuantumNeuralNetwork,
        feature_dim: usize,
        distance_metric: DistanceMetric,
    ) -> Self {
        Self {
            encoder,
            feature_dim,
            distance_metric,
        }
    }

    /// Encode data into quantum feature space
    pub fn encode(&self, data: &Array1<f64>) -> Result<Array1<f64>> {
        // Placeholder - would use quantum circuit for encoding
        let features = self.extract_features_placeholder()?;

        Ok(features)
    }

    /// Extract features from quantum state (placeholder)
    fn extract_features_placeholder(&self) -> Result<Array1<f64>> {
        // Placeholder - would measure specific observables
        let features = Array1::zeros(self.feature_dim);
        Ok(features)
    }

    /// Compute prototype for a class from support examples
    pub fn compute_prototype(&self, support_examples: &[Array1<f64>]) -> Result<Array1<f64>> {
        let mut prototype = Array1::zeros(self.feature_dim);

        // Encode and average support examples
        for example in support_examples {
            let encoded = self.encode(example)?;
            prototype = prototype + encoded;
        }

        prototype = prototype / support_examples.len() as f64;
        Ok(prototype)
    }

    /// Classify query example based on prototypes
    pub fn classify(&self, query: &Array1<f64>, prototypes: &[Array1<f64>]) -> Result<usize> {
        let query_encoded = self.encode(query)?;

        // Find nearest prototype
        let mut min_distance = f64::INFINITY;
        let mut predicted_class = 0;

        for (class_idx, prototype) in prototypes.iter().enumerate() {
            let distance = match self.distance_metric {
                DistanceMetric::Euclidean => {
                    (&query_encoded - prototype).mapv(|x| x * x).sum().sqrt()
                }
                DistanceMetric::Cosine => {
                    let dot = (&query_encoded * prototype).sum();
                    let norm_q = query_encoded.mapv(|x| x * x).sum().sqrt();
                    let norm_p = prototype.mapv(|x| x * x).sum().sqrt();
                    1.0 - dot / (norm_q * norm_p + 1e-8)
                }
                DistanceMetric::QuantumKernel => {
                    // Use quantum kernel distance
                    self.quantum_distance(&query_encoded, prototype)?
                }
            };

            if distance < min_distance {
                min_distance = distance;
                predicted_class = class_idx;
            }
        }

        Ok(predicted_class)
    }

    /// Compute quantum kernel distance
    fn quantum_distance(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64> {
        // Placeholder - would compute quantum kernel
        Ok((x - y).mapv(|v| v * v).sum().sqrt())
    }

    /// Train on an episode
    pub fn train_episode(
        &mut self,
        episode: &Episode,
        optimizer: &mut dyn Optimizer,
    ) -> Result<f64> {
        // Compute prototypes for each class
        let mut prototypes = Vec::new();
        let mut class_examples = HashMap::new();

        // Group support examples by class
        for (data, label) in &episode.support_set {
            class_examples
                .entry(*label)
                .or_insert(Vec::new())
                .push(data.clone());
        }

        // Compute prototype for each class
        for class_id in 0..episode.num_classes {
            if let Some(examples) = class_examples.get(&class_id) {
                let prototype = self.compute_prototype(examples)?;
                prototypes.push(prototype);
            }
        }

        // Evaluate on query set
        let mut correct = 0;
        let mut total_loss = 0.0;

        for (query, true_label) in &episode.query_set {
            let predicted = self.classify(query, &prototypes)?;

            if predicted == *true_label {
                correct += 1;
            }

            // Compute loss
            let query_encoded = self.encode(query)?;
            let loss = self.prototypical_loss(&query_encoded, &prototypes, *true_label)?;
            total_loss += loss;
        }

        let accuracy = correct as f64 / episode.query_set.len() as f64;
        let avg_loss = total_loss / episode.query_set.len() as f64;

        // Update encoder parameters
        self.update_parameters(optimizer, avg_loss)?;

        Ok(accuracy)
    }

    /// Compute prototypical loss
    fn prototypical_loss(
        &self,
        query: &Array1<f64>,
        prototypes: &[Array1<f64>],
        true_label: usize,
    ) -> Result<f64> {
        let mut distances = Vec::new();

        // Compute distances to all prototypes
        for prototype in prototypes {
            let distance = match self.distance_metric {
                DistanceMetric::Euclidean => (query - prototype).mapv(|x| x * x).sum(),
                _ => {
                    // Other metrics
                    (query - prototype).mapv(|x| x * x).sum()
                }
            };
            distances.push(-distance); // Negative for softmax
        }

        // Softmax and cross-entropy loss
        let max_val = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = distances.iter().map(|&d| (d - max_val).exp()).sum();
        let log_prob = distances[true_label] - max_val - exp_sum.ln();

        Ok(-log_prob)
    }

    /// Update encoder parameters
    fn update_parameters(&mut self, optimizer: &mut dyn Optimizer, loss: f64) -> Result<()> {
        // Placeholder - would compute gradients and update
        Ok(())
    }
}

/// Quantum MAML for few-shot learning
pub struct QuantumMAML {
    /// Base quantum model
    model: QuantumNeuralNetwork,

    /// Inner loop learning rate
    inner_lr: f64,

    /// Number of inner loop steps
    inner_steps: usize,

    /// Task-specific parameters
    task_params: HashMap<String, Array1<f64>>,
}

impl QuantumMAML {
    /// Create a new Quantum MAML instance
    pub fn new(model: QuantumNeuralNetwork, inner_lr: f64, inner_steps: usize) -> Self {
        Self {
            model,
            inner_lr,
            inner_steps,
            task_params: HashMap::new(),
        }
    }

    /// Inner loop adaptation for a specific task
    pub fn adapt_to_task(
        &mut self,
        support_set: &[(Array1<f64>, usize)],
        task_id: &str,
    ) -> Result<()> {
        // Clone current parameters
        let mut adapted_params = self.model.parameters.clone();

        // Perform inner loop updates
        for _ in 0..self.inner_steps {
            // Compute gradients on support set
            let gradients = self.compute_task_gradients(support_set, &adapted_params)?;

            // Update parameters
            adapted_params = adapted_params - self.inner_lr * &gradients;
        }

        // Store task-specific parameters
        self.task_params.insert(task_id.to_string(), adapted_params);

        Ok(())
    }

    /// Compute gradients for a specific task
    fn compute_task_gradients(
        &self,
        support_set: &[(Array1<f64>, usize)],
        params: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Placeholder - would compute actual quantum gradients
        Ok(Array1::zeros(params.len()))
    }

    /// Predict using task-adapted parameters
    pub fn predict_adapted(&self, query: &Array1<f64>, task_id: &str) -> Result<usize> {
        let params = self
            .task_params
            .get(task_id)
            .ok_or(MLError::ModelCreationError("Task not adapted".to_string()))?;

        // Use adapted parameters for prediction
        // Placeholder implementation
        Ok(0)
    }

    /// Meta-train on multiple tasks
    pub fn meta_train(
        &mut self,
        tasks: &[Episode],
        meta_optimizer: &mut dyn Optimizer,
        meta_epochs: usize,
    ) -> Result<Vec<f64>> {
        let mut meta_losses = Vec::new();

        for epoch in 0..meta_epochs {
            let mut epoch_loss = 0.0;

            for (task_idx, episode) in tasks.iter().enumerate() {
                let task_id = format!("task_{}", task_idx);

                // Inner loop: adapt to task
                self.adapt_to_task(&episode.support_set, &task_id)?;

                // Outer loop: evaluate on query set
                let mut task_loss = 0.0;
                for (query, label) in &episode.query_set {
                    let predicted = self.predict_adapted(query, &task_id)?;
                    task_loss += if predicted == *label { 0.0 } else { 1.0 };
                }

                epoch_loss += task_loss / episode.query_set.len() as f64;
            }

            // Meta-update
            let meta_loss = epoch_loss / tasks.len() as f64;
            meta_losses.push(meta_loss);

            // Update base model parameters
            self.meta_update(meta_optimizer, meta_loss)?;
        }

        Ok(meta_losses)
    }

    /// Perform meta-update on base model
    fn meta_update(&mut self, optimizer: &mut dyn Optimizer, loss: f64) -> Result<()> {
        // Placeholder - would compute meta-gradients
        Ok(())
    }
}

/// Few-shot learning manager
pub struct FewShotLearner {
    /// Learning method
    method: FewShotMethod,

    /// Base model
    model: QuantumNeuralNetwork,

    /// Training history
    history: Vec<f64>,
}

impl FewShotLearner {
    /// Create a new few-shot learner
    pub fn new(method: FewShotMethod, model: QuantumNeuralNetwork) -> Self {
        Self {
            method,
            model,
            history: Vec::new(),
        }
    }

    /// Generate episode from dataset
    pub fn generate_episode(
        data: &Array2<f64>,
        labels: &Array1<usize>,
        num_classes: usize,
        k_shot: usize,
        query_per_class: usize,
    ) -> Result<Episode> {
        let mut support_set = Vec::new();
        let mut query_set = Vec::new();

        // Sample classes
        let selected_classes: Vec<usize> = (0..num_classes).collect();

        for class_id in selected_classes {
            // Find all examples of this class
            let class_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == class_id)
                .map(|(i, _)| i)
                .collect();

            if class_indices.len() < k_shot + query_per_class {
                return Err(MLError::ModelCreationError(format!(
                    "Not enough examples for class {}",
                    class_id
                )));
            }

            // Sample support and query examples
            let mut rng = thread_rng();
            let mut shuffled = class_indices.clone();
            shuffled.shuffle(&mut rng);

            // Support set
            for i in 0..k_shot {
                let idx = shuffled[i];
                support_set.push((data.row(idx).to_owned(), class_id));
            }

            // Query set
            for i in k_shot..(k_shot + query_per_class) {
                let idx = shuffled[i];
                query_set.push((data.row(idx).to_owned(), class_id));
            }
        }

        Ok(Episode {
            support_set,
            query_set,
            num_classes,
            k_shot,
        })
    }

    /// Train the few-shot learner
    pub fn train(
        &mut self,
        episodes: &[Episode],
        optimizer: &mut dyn Optimizer,
        epochs: usize,
    ) -> Result<Vec<f64>> {
        match self.method {
            FewShotMethod::PrototypicalNetworks => {
                let mut proto_net = QuantumPrototypicalNetwork::new(
                    self.model.clone(),
                    16, // feature dimension
                    DistanceMetric::Euclidean,
                );

                for epoch in 0..epochs {
                    let mut epoch_acc = 0.0;

                    for episode in episodes {
                        let acc = proto_net.train_episode(episode, optimizer)?;
                        epoch_acc += acc;
                    }

                    let avg_acc = epoch_acc / episodes.len() as f64;
                    self.history.push(avg_acc);
                }
            }
            FewShotMethod::MAML {
                inner_steps,
                inner_lr,
            } => {
                let mut maml = QuantumMAML::new(self.model.clone(), inner_lr, inner_steps);

                let losses = maml.meta_train(episodes, optimizer, epochs)?;
                self.history.extend(losses);
            }
            _ => {
                return Err(MLError::ModelCreationError(
                    "Method not implemented".to_string(),
                ));
            }
        }

        Ok(self.history.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::optimizers::Adam;
    use crate::qnn::QNNLayerType;

    #[test]
    fn test_episode_generation() {
        let num_samples = 100;
        let num_features = 4;
        let num_classes = 5;

        // Generate synthetic data
        let data = Array2::from_shape_fn((num_samples, num_features), |(i, j)| {
            (i as f64 * 0.1 + j as f64 * 0.2).sin()
        });
        let labels = Array1::from_shape_fn(num_samples, |i| i % num_classes);

        // Generate episode
        let episode = FewShotLearner::generate_episode(
            &data, &labels, 3, // 3-way
            5, // 5-shot
            5, // 5 query per class
        )
        .unwrap();

        assert_eq!(episode.num_classes, 3);
        assert_eq!(episode.k_shot, 5);
        assert_eq!(episode.support_set.len(), 15); // 3 classes * 5 shots
        assert_eq!(episode.query_set.len(), 15); // 3 classes * 5 queries
    }

    #[test]
    fn test_prototypical_network() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let qnn = QuantumNeuralNetwork::new(layers, 4, 4, 2).unwrap();
        let proto_net = QuantumPrototypicalNetwork::new(qnn, 8, DistanceMetric::Euclidean);

        // Test encoding
        let data = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let encoded = proto_net.encode(&data).unwrap();
        assert_eq!(encoded.len(), 8);

        // Test prototype computation
        let examples = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
            Array1::from_vec(vec![0.2, 0.3, 0.4, 0.5]),
        ];
        let prototype = proto_net.compute_prototype(&examples).unwrap();
        assert_eq!(prototype.len(), 8);
    }

    #[test]
    fn test_maml_adaptation() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 6 },
        ];

        let qnn = QuantumNeuralNetwork::new(layers, 4, 4, 2).unwrap();
        let mut maml = QuantumMAML::new(qnn, 0.01, 5);

        // Create support set
        let support_set = vec![
            (Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]), 0),
            (Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]), 1),
        ];

        // Adapt to task
        maml.adapt_to_task(&support_set, "test_task").unwrap();

        // Check that task parameters were stored
        assert!(maml.task_params.contains_key("test_task"));
    }
}
