//! Quantum Meta-Learning Algorithms
//!
//! This module implements various meta-learning algorithms adapted for quantum circuits,
//! enabling quantum models to learn how to learn from limited data across multiple tasks.

use crate::autodiff::optimizers::Optimizer;
use crate::error::{MLError, Result};
use crate::optimization::OptimizationMethod;
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use ndarray::{s, Array1, Array2, Array3, Axis};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::gate::{
    single::{RotationX, RotationY, RotationZ},
    GateOp,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::collections::HashMap;

/// Meta-learning algorithm types
#[derive(Debug, Clone, Copy)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML {
        inner_steps: usize,
        inner_lr: f64,
        first_order: bool,
    },

    /// Reptile algorithm
    Reptile { inner_steps: usize, inner_lr: f64 },

    /// Prototypical MAML
    ProtoMAML {
        inner_steps: usize,
        inner_lr: f64,
        proto_weight: f64,
    },

    /// Meta-SGD with learnable inner learning rates
    MetaSGD { inner_steps: usize },

    /// Almost No Inner Loop (ANIL)
    ANIL { inner_steps: usize, inner_lr: f64 },
}

/// Task definition for meta-learning
#[derive(Debug, Clone)]
pub struct MetaTask {
    /// Task identifier
    pub id: String,

    /// Training data (support set)
    pub train_data: Vec<(Array1<f64>, usize)>,

    /// Test data (query set)
    pub test_data: Vec<(Array1<f64>, usize)>,

    /// Number of classes
    pub num_classes: usize,

    /// Task-specific metadata
    pub metadata: HashMap<String, f64>,
}

/// Base quantum meta-learner
pub struct QuantumMetaLearner {
    /// Meta-learning algorithm
    algorithm: MetaLearningAlgorithm,

    /// Base quantum model
    model: QuantumNeuralNetwork,

    /// Meta-parameters
    meta_params: Array1<f64>,

    /// Per-parameter learning rates (for Meta-SGD)
    per_param_lr: Option<Array1<f64>>,

    /// Task embeddings
    task_embeddings: HashMap<String, Array1<f64>>,

    /// Training history
    history: MetaLearningHistory,
}

/// Training history for meta-learning
#[derive(Debug, Clone)]
pub struct MetaLearningHistory {
    /// Meta-train losses
    pub meta_train_losses: Vec<f64>,

    /// Meta-validation accuracies
    pub meta_val_accuracies: Vec<f64>,

    /// Per-task performance
    pub task_performance: HashMap<String, Vec<f64>>,
}

impl QuantumMetaLearner {
    /// Create a new quantum meta-learner
    pub fn new(algorithm: MetaLearningAlgorithm, model: QuantumNeuralNetwork) -> Self {
        let num_params = model.parameters.len();
        let meta_params = model.parameters.clone();

        let per_param_lr = match algorithm {
            MetaLearningAlgorithm::MetaSGD { .. } => Some(Array1::from_elem(num_params, 0.01)),
            _ => None,
        };

        Self {
            algorithm,
            model,
            meta_params,
            per_param_lr,
            task_embeddings: HashMap::new(),
            history: MetaLearningHistory {
                meta_train_losses: Vec::new(),
                meta_val_accuracies: Vec::new(),
                task_performance: HashMap::new(),
            },
        }
    }

    /// Meta-train on multiple tasks
    pub fn meta_train(
        &mut self,
        tasks: &[MetaTask],
        meta_optimizer: &mut dyn Optimizer,
        meta_epochs: usize,
        tasks_per_batch: usize,
    ) -> Result<()> {
        println!("Starting meta-training with {} tasks...", tasks.len());

        for epoch in 0..meta_epochs {
            let mut epoch_loss = 0.0;
            let mut epoch_acc = 0.0;

            // Sample batch of tasks
            let task_batch = self.sample_task_batch(tasks, tasks_per_batch);

            // Perform meta-update based on algorithm
            match self.algorithm {
                MetaLearningAlgorithm::MAML { .. } => {
                    let (loss, acc) = self.maml_update(&task_batch, meta_optimizer)?;
                    epoch_loss += loss;
                    epoch_acc += acc;
                }
                MetaLearningAlgorithm::Reptile { .. } => {
                    let (loss, acc) = self.reptile_update(&task_batch, meta_optimizer)?;
                    epoch_loss += loss;
                    epoch_acc += acc;
                }
                MetaLearningAlgorithm::ProtoMAML { .. } => {
                    let (loss, acc) = self.protomaml_update(&task_batch, meta_optimizer)?;
                    epoch_loss += loss;
                    epoch_acc += acc;
                }
                MetaLearningAlgorithm::MetaSGD { .. } => {
                    let (loss, acc) = self.metasgd_update(&task_batch, meta_optimizer)?;
                    epoch_loss += loss;
                    epoch_acc += acc;
                }
                MetaLearningAlgorithm::ANIL { .. } => {
                    let (loss, acc) = self.anil_update(&task_batch, meta_optimizer)?;
                    epoch_loss += loss;
                    epoch_acc += acc;
                }
            }

            // Update history
            self.history.meta_train_losses.push(epoch_loss);
            self.history.meta_val_accuracies.push(epoch_acc);

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Loss = {:.4}, Accuracy = {:.2}%",
                    epoch,
                    epoch_loss,
                    epoch_acc * 100.0
                );
            }
        }

        Ok(())
    }

    /// MAML update step
    fn maml_update(
        &mut self,
        tasks: &[MetaTask],
        optimizer: &mut dyn Optimizer,
    ) -> Result<(f64, f64)> {
        let (inner_steps, inner_lr, first_order) = match self.algorithm {
            MetaLearningAlgorithm::MAML {
                inner_steps,
                inner_lr,
                first_order,
            } => (inner_steps, inner_lr, first_order),
            _ => unreachable!(),
        };

        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut meta_gradients = Array1::zeros(self.meta_params.len());

        for task in tasks {
            // Clone meta-parameters for inner loop
            let mut task_params = self.meta_params.clone();

            // Inner loop: adapt to task
            for _ in 0..inner_steps {
                let grad = self.compute_task_gradient(&task.train_data, &task_params)?;
                task_params = task_params - inner_lr * &grad;
            }

            // Compute loss on query set with adapted parameters
            let (query_loss, query_acc) = self.evaluate_task(&task.test_data, &task_params)?;
            total_loss += query_loss;
            total_acc += query_acc;

            // Compute meta-gradient
            if !first_order {
                // Full second-order MAML gradient
                let meta_grad = self.compute_maml_gradient(task, &task_params, inner_lr)?;
                meta_gradients = meta_gradients + meta_grad;
            } else {
                // First-order approximation (FO-MAML)
                let grad = self.compute_task_gradient(&task.test_data, &task_params)?;
                meta_gradients = meta_gradients + grad;
            }
        }

        // Average gradients and update meta-parameters
        meta_gradients = meta_gradients / tasks.len() as f64;
        self.meta_params = self.meta_params.clone() - 0.001 * &meta_gradients; // Meta learning rate

        Ok((
            total_loss / tasks.len() as f64,
            total_acc / tasks.len() as f64,
        ))
    }

    /// Reptile update step
    fn reptile_update(
        &mut self,
        tasks: &[MetaTask],
        optimizer: &mut dyn Optimizer,
    ) -> Result<(f64, f64)> {
        let (inner_steps, inner_lr) = match self.algorithm {
            MetaLearningAlgorithm::Reptile {
                inner_steps,
                inner_lr,
            } => (inner_steps, inner_lr),
            _ => unreachable!(),
        };

        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let epsilon = 0.1; // Reptile step size

        for task in tasks {
            // Clone meta-parameters
            let mut task_params = self.meta_params.clone();

            // Perform multiple SGD steps on task
            for _ in 0..inner_steps {
                let grad = self.compute_task_gradient(&task.train_data, &task_params)?;
                task_params = task_params - inner_lr * &grad;
            }

            // Evaluate adapted model
            let (loss, acc) = self.evaluate_task(&task.test_data, &task_params)?;
            total_loss += loss;
            total_acc += acc;

            // Reptile update: move meta-params toward task-adapted params
            let direction = &task_params - &self.meta_params;
            self.meta_params = &self.meta_params + epsilon * &direction;
        }

        Ok((
            total_loss / tasks.len() as f64,
            total_acc / tasks.len() as f64,
        ))
    }

    /// ProtoMAML update step
    fn protomaml_update(
        &mut self,
        tasks: &[MetaTask],
        optimizer: &mut dyn Optimizer,
    ) -> Result<(f64, f64)> {
        let (inner_steps, inner_lr, proto_weight) = match self.algorithm {
            MetaLearningAlgorithm::ProtoMAML {
                inner_steps,
                inner_lr,
                proto_weight,
            } => (inner_steps, inner_lr, proto_weight),
            _ => unreachable!(),
        };

        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for task in tasks {
            // Compute prototypes for each class
            let prototypes = self.compute_prototypes(&task.train_data, task.num_classes)?;

            // Clone parameters for adaptation
            let mut task_params = self.meta_params.clone();

            // Inner loop with prototype regularization
            for _ in 0..inner_steps {
                let grad = self.compute_task_gradient(&task.train_data, &task_params)?;
                let proto_reg =
                    self.prototype_regularization(&task.train_data, &prototypes, &task_params)?;
                task_params = task_params - inner_lr * (&grad + proto_weight * &proto_reg);
            }

            // Evaluate with prototypical classification
            let (loss, acc) =
                self.evaluate_with_prototypes(&task.test_data, &prototypes, &task_params)?;
            total_loss += loss;
            total_acc += acc;
        }

        Ok((
            total_loss / tasks.len() as f64,
            total_acc / tasks.len() as f64,
        ))
    }

    /// Meta-SGD update step
    fn metasgd_update(
        &mut self,
        tasks: &[MetaTask],
        optimizer: &mut dyn Optimizer,
    ) -> Result<(f64, f64)> {
        let inner_steps = match self.algorithm {
            MetaLearningAlgorithm::MetaSGD { inner_steps } => inner_steps,
            _ => unreachable!(),
        };

        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut meta_lr_gradients = Array1::zeros(self.meta_params.len());

        for task in tasks {
            let mut task_params = self.meta_params.clone();

            // Inner loop with learnable per-parameter learning rates
            for _ in 0..inner_steps {
                let grad = self.compute_task_gradient(&task.train_data, &task_params)?;
                let lr = self.per_param_lr.as_ref().unwrap();
                task_params = task_params - lr * &grad;
            }

            // Evaluate and compute gradients w.r.t. both parameters and learning rates
            let (loss, acc) = self.evaluate_task(&task.test_data, &task_params)?;
            total_loss += loss;
            total_acc += acc;

            // Compute gradient w.r.t. learning rates
            let lr_grad = self.compute_lr_gradient(task, &task_params)?;
            meta_lr_gradients = meta_lr_gradients + lr_grad;
        }

        // Update both parameters and learning rates
        if let Some(ref mut lr) = self.per_param_lr {
            *lr = lr.clone() - &(0.001 * &meta_lr_gradients / tasks.len() as f64);
        }

        Ok((
            total_loss / tasks.len() as f64,
            total_acc / tasks.len() as f64,
        ))
    }

    /// ANIL update step
    fn anil_update(
        &mut self,
        tasks: &[MetaTask],
        optimizer: &mut dyn Optimizer,
    ) -> Result<(f64, f64)> {
        let (inner_steps, inner_lr) = match self.algorithm {
            MetaLearningAlgorithm::ANIL {
                inner_steps,
                inner_lr,
            } => (inner_steps, inner_lr),
            _ => unreachable!(),
        };

        // ANIL: Only adapt the final layer(s) in inner loop
        let num_params = self.meta_params.len();
        let final_layer_start = (num_params * 3) / 4; // Last 25% of parameters

        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for task in tasks {
            let mut task_params = self.meta_params.clone();

            // Inner loop: only update final layer parameters
            for _ in 0..inner_steps {
                let grad = self.compute_task_gradient(&task.train_data, &task_params)?;

                // Only update final layer
                for i in final_layer_start..num_params {
                    task_params[i] -= inner_lr * grad[i];
                }
            }

            let (loss, acc) = self.evaluate_task(&task.test_data, &task_params)?;
            total_loss += loss;
            total_acc += acc;
        }

        Ok((
            total_loss / tasks.len() as f64,
            total_acc / tasks.len() as f64,
        ))
    }

    /// Compute gradient for a task
    fn compute_task_gradient(
        &self,
        data: &[(Array1<f64>, usize)],
        params: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Placeholder - would compute actual quantum gradients
        Ok(Array1::zeros(params.len()))
    }

    /// Evaluate model on task data
    fn evaluate_task(
        &self,
        data: &[(Array1<f64>, usize)],
        params: &Array1<f64>,
    ) -> Result<(f64, f64)> {
        // Placeholder - would evaluate quantum model
        let loss = 0.5 + 0.5 * rand::random::<f64>();
        let acc = 0.5 + 0.3 * rand::random::<f64>();
        Ok((loss, acc))
    }

    /// Compute MAML gradient (second-order)
    fn compute_maml_gradient(
        &self,
        task: &MetaTask,
        adapted_params: &Array1<f64>,
        inner_lr: f64,
    ) -> Result<Array1<f64>> {
        // Placeholder - would compute Hessian-vector products
        Ok(Array1::zeros(self.meta_params.len()))
    }

    /// Compute prototypes for ProtoMAML
    fn compute_prototypes(
        &self,
        data: &[(Array1<f64>, usize)],
        num_classes: usize,
    ) -> Result<Vec<Array1<f64>>> {
        let feature_dim = 16; // Placeholder dimension
        let mut prototypes = vec![Array1::zeros(feature_dim); num_classes];
        let mut counts = vec![0; num_classes];

        // Placeholder - would encode data and compute class means
        for (x, label) in data {
            counts[*label] += 1;
        }

        Ok(prototypes)
    }

    /// Prototype regularization
    fn prototype_regularization(
        &self,
        data: &[(Array1<f64>, usize)],
        prototypes: &[Array1<f64>],
        params: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Placeholder - would compute prototype-based regularization
        Ok(Array1::zeros(params.len()))
    }

    /// Evaluate with prototypical classification
    fn evaluate_with_prototypes(
        &self,
        data: &[(Array1<f64>, usize)],
        prototypes: &[Array1<f64>],
        params: &Array1<f64>,
    ) -> Result<(f64, f64)> {
        // Placeholder
        Ok((0.3, 0.7))
    }

    /// Compute gradient w.r.t. learning rates for Meta-SGD
    fn compute_lr_gradient(
        &self,
        task: &MetaTask,
        adapted_params: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Placeholder
        Ok(Array1::zeros(self.meta_params.len()))
    }

    /// Sample batch of tasks
    fn sample_task_batch(&self, tasks: &[MetaTask], batch_size: usize) -> Vec<MetaTask> {
        let mut batch = Vec::new();
        let mut rng = fastrand::Rng::new();

        for _ in 0..batch_size.min(tasks.len()) {
            let idx = rng.usize(0..tasks.len());
            batch.push(tasks[idx].clone());
        }

        batch
    }

    /// Adapt to new task
    pub fn adapt_to_task(&mut self, task: &MetaTask) -> Result<Array1<f64>> {
        let adapted_params = match self.algorithm {
            MetaLearningAlgorithm::MAML {
                inner_steps,
                inner_lr,
                ..
            }
            | MetaLearningAlgorithm::Reptile {
                inner_steps,
                inner_lr,
            }
            | MetaLearningAlgorithm::ProtoMAML {
                inner_steps,
                inner_lr,
                ..
            }
            | MetaLearningAlgorithm::ANIL {
                inner_steps,
                inner_lr,
            } => {
                let mut params = self.meta_params.clone();
                for _ in 0..inner_steps {
                    let grad = self.compute_task_gradient(&task.train_data, &params)?;
                    params = params - inner_lr * &grad;
                }
                params
            }
            MetaLearningAlgorithm::MetaSGD { inner_steps } => {
                let mut params = self.meta_params.clone();
                let lr = self.per_param_lr.as_ref().unwrap();
                for _ in 0..inner_steps {
                    let grad = self.compute_task_gradient(&task.train_data, &params)?;
                    params = params - lr * &grad;
                }
                params
            }
        };

        Ok(adapted_params)
    }

    /// Get task embedding
    pub fn get_task_embedding(&self, task_id: &str) -> Option<&Array1<f64>> {
        self.task_embeddings.get(task_id)
    }

    /// Get meta parameters
    pub fn meta_params(&self) -> &Array1<f64> {
        &self.meta_params
    }

    /// Get per-parameter learning rates
    pub fn per_param_lr(&self) -> Option<&Array1<f64>> {
        self.per_param_lr.as_ref()
    }
}

/// Continual meta-learning with memory
pub struct ContinualMetaLearner {
    /// Base meta-learner
    meta_learner: QuantumMetaLearner,

    /// Memory buffer for past tasks
    memory_buffer: Vec<MetaTask>,

    /// Maximum memory size
    memory_capacity: usize,

    /// Replay ratio
    replay_ratio: f64,
}

impl ContinualMetaLearner {
    /// Create new continual meta-learner
    pub fn new(
        meta_learner: QuantumMetaLearner,
        memory_capacity: usize,
        replay_ratio: f64,
    ) -> Self {
        Self {
            meta_learner,
            memory_buffer: Vec::new(),
            memory_capacity,
            replay_ratio,
        }
    }

    /// Learn new task while preserving old knowledge
    pub fn learn_task(&mut self, new_task: MetaTask) -> Result<()> {
        // Add to memory with reservoir sampling
        if self.memory_buffer.len() < self.memory_capacity {
            self.memory_buffer.push(new_task.clone());
        } else {
            let idx = fastrand::usize(0..self.memory_buffer.len());
            self.memory_buffer[idx] = new_task.clone();
        }

        // Create mixed batch with replay
        let num_replay = (self.memory_buffer.len() as f64 * self.replay_ratio) as usize;
        let mut task_batch = vec![new_task];

        for _ in 0..num_replay {
            let idx = fastrand::usize(0..self.memory_buffer.len());
            task_batch.push(self.memory_buffer[idx].clone());
        }

        // Update meta-learner
        let mut dummy_optimizer = crate::autodiff::optimizers::Adam::new(0.001);
        self.meta_learner
            .meta_train(&task_batch, &mut dummy_optimizer, 10, task_batch.len())?;

        Ok(())
    }

    /// Get memory buffer length
    pub fn memory_buffer_len(&self) -> usize {
        self.memory_buffer.len()
    }
}

/// Task generator for meta-learning experiments
pub struct TaskGenerator {
    /// Feature dimension
    feature_dim: usize,

    /// Number of classes per task
    num_classes: usize,

    /// Task distribution parameters
    task_params: HashMap<String, f64>,
}

impl TaskGenerator {
    /// Create new task generator
    pub fn new(feature_dim: usize, num_classes: usize) -> Self {
        Self {
            feature_dim,
            num_classes,
            task_params: HashMap::new(),
        }
    }

    /// Generate sinusoid regression task
    pub fn generate_sinusoid_task(&self, num_samples: usize) -> MetaTask {
        let amplitude = 0.1 + 4.9 * rand::random::<f64>();
        let phase = 2.0 * std::f64::consts::PI * rand::random::<f64>();

        let mut train_data = Vec::new();
        let mut test_data = Vec::new();

        // Generate samples
        for i in 0..num_samples {
            let x = -5.0 + 10.0 * rand::random::<f64>();
            let y = amplitude * (x + phase).sin();

            let input = Array1::from_vec(vec![x]);
            let label = if y > 0.0 { 1 } else { 0 }; // Binarize for classification

            if i < num_samples / 2 {
                train_data.push((input, label));
            } else {
                test_data.push((input, label));
            }
        }

        MetaTask {
            id: format!("sin_a{:.2}_p{:.2}", amplitude, phase),
            train_data,
            test_data,
            num_classes: 2,
            metadata: vec![
                ("amplitude".to_string(), amplitude),
                ("phase".to_string(), phase),
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Generate classification task with rotated features
    pub fn generate_rotation_task(&self, num_samples: usize) -> MetaTask {
        let angle = 2.0 * std::f64::consts::PI * rand::random::<f64>();
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let mut train_data = Vec::new();
        let mut test_data = Vec::new();

        for i in 0..num_samples {
            // Generate base features
            let mut features = Array1::zeros(self.feature_dim);
            let label = i % self.num_classes;

            // Class-specific pattern
            for j in 0..self.feature_dim {
                features[j] = if j % self.num_classes == label {
                    1.0
                } else {
                    0.0
                };
                features[j] += 0.1 * rand::random::<f64>();
            }

            // Apply rotation (simplified for first 2 dims)
            if self.feature_dim >= 2 {
                let x = features[0];
                let y = features[1];
                features[0] = cos_a * x - sin_a * y;
                features[1] = sin_a * x + cos_a * y;
            }

            if i < num_samples / 2 {
                train_data.push((features, label));
            } else {
                test_data.push((features, label));
            }
        }

        MetaTask {
            id: format!("rot_{:.2}", angle),
            train_data,
            test_data,
            num_classes: self.num_classes,
            metadata: vec![("rotation_angle".to_string(), angle)]
                .into_iter()
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::optimizers::Adam;
    use crate::qnn::QNNLayerType;

    #[test]
    fn test_task_generator() {
        let generator = TaskGenerator::new(4, 2);

        let sin_task = generator.generate_sinusoid_task(20);
        assert_eq!(sin_task.train_data.len(), 10);
        assert_eq!(sin_task.test_data.len(), 10);

        let rot_task = generator.generate_rotation_task(30);
        assert_eq!(rot_task.train_data.len(), 15);
        assert_eq!(rot_task.test_data.len(), 15);
    }

    #[test]
    fn test_meta_learner_creation() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let qnn = QuantumNeuralNetwork::new(layers, 4, 4, 2).unwrap();

        let maml_algo = MetaLearningAlgorithm::MAML {
            inner_steps: 5,
            inner_lr: 0.01,
            first_order: true,
        };

        let meta_learner = QuantumMetaLearner::new(maml_algo, qnn);
        assert!(meta_learner.per_param_lr.is_none());

        // Test Meta-SGD
        let layers2 = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
        ];
        let qnn2 = QuantumNeuralNetwork::new(layers2, 4, 4, 2).unwrap();

        let metasgd_algo = MetaLearningAlgorithm::MetaSGD { inner_steps: 3 };
        let meta_sgd = QuantumMetaLearner::new(metasgd_algo, qnn2);
        assert!(meta_sgd.per_param_lr.is_some());
    }

    #[test]
    fn test_task_adaptation() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 2 },
            QNNLayerType::VariationalLayer { num_params: 6 },
        ];

        let qnn = QuantumNeuralNetwork::new(layers, 4, 2, 2).unwrap();
        let algo = MetaLearningAlgorithm::Reptile {
            inner_steps: 5,
            inner_lr: 0.01,
        };

        let mut meta_learner = QuantumMetaLearner::new(algo, qnn);

        // Generate task
        let generator = TaskGenerator::new(2, 2);
        let task = generator.generate_rotation_task(20);

        // Adapt to task
        let adapted_params = meta_learner.adapt_to_task(&task).unwrap();
        assert_eq!(adapted_params.len(), meta_learner.meta_params.len());
    }
}
