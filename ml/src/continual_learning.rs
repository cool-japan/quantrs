//! Quantum Continual Learning
//!
//! This module implements continual learning algorithms for quantum neural networks,
//! enabling models to learn new tasks sequentially while preserving knowledge from
//! previous tasks and avoiding catastrophic forgetting.

use crate::autodiff::optimizers::Optimizer;
use crate::error::{MLError, Result};
use crate::optimization::OptimizationMethod;
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::gate::{
    single::{RotationX, RotationY, RotationZ},
    GateOp,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

/// Continual learning strategies for quantum models
#[derive(Debug, Clone)]
pub enum ContinualLearningStrategy {
    /// Elastic Weight Consolidation (EWC) for quantum circuits
    ElasticWeightConsolidation {
        importance_weight: f64,
        fisher_samples: usize,
    },

    /// Progressive Neural Networks with quantum modules
    ProgressiveNetworks {
        lateral_connections: bool,
        adaptation_layers: usize,
    },

    /// Memory replay with episodic buffer
    ExperienceReplay {
        buffer_size: usize,
        replay_ratio: f64,
        memory_selection: MemorySelectionStrategy,
    },

    /// Parameter isolation and expansion
    ParameterIsolation {
        allocation_strategy: ParameterAllocationStrategy,
        growth_threshold: f64,
    },

    /// Gradient episodic memory
    GradientEpisodicMemory {
        memory_strength: f64,
        violation_threshold: f64,
    },

    /// Learning without forgetting (LwF)
    LearningWithoutForgetting {
        distillation_weight: f64,
        temperature: f64,
    },

    /// Quantum-specific regularization
    QuantumRegularization {
        entanglement_preservation: f64,
        parameter_drift_penalty: f64,
    },
}

/// Memory selection strategies for experience replay
#[derive(Debug, Clone)]
pub enum MemorySelectionStrategy {
    /// Random sampling
    Random,
    /// Gradient-based importance
    GradientImportance,
    /// Uncertainty-based selection
    Uncertainty,
    /// Diverse sampling
    Diversity,
    /// Quantum-specific metrics
    QuantumMetrics,
}

/// Parameter allocation strategies
#[derive(Debug, Clone)]
pub enum ParameterAllocationStrategy {
    /// Add new parameters for new tasks
    Expansion,
    /// Mask existing parameters for different tasks
    Masking,
    /// Hierarchical parameter sharing
    Hierarchical,
    /// Quantum-specific allocation
    QuantumAware,
}

/// Task definition for continual learning
#[derive(Debug, Clone)]
pub struct ContinualTask {
    /// Task identifier
    pub task_id: String,

    /// Task type/domain
    pub task_type: TaskType,

    /// Training data
    pub train_data: Array2<f64>,

    /// Training labels
    pub train_labels: Array1<usize>,

    /// Validation data
    pub val_data: Array2<f64>,

    /// Validation labels
    pub val_labels: Array1<usize>,

    /// Number of classes
    pub num_classes: usize,

    /// Task-specific metadata
    pub metadata: HashMap<String, f64>,
}

/// Task types for continual learning
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// Classification task
    Classification { num_classes: usize },
    /// Regression task
    Regression { output_dim: usize },
    /// Quantum state preparation
    StatePreparation { target_states: usize },
    /// Quantum optimization
    Optimization { problem_type: String },
}

/// Memory buffer for experience replay
#[derive(Debug, Clone)]
pub struct MemoryBuffer {
    /// Stored experiences
    experiences: VecDeque<Experience>,

    /// Maximum buffer size
    max_size: usize,

    /// Selection strategy
    selection_strategy: MemorySelectionStrategy,

    /// Task-wise organization
    task_memories: HashMap<String, Vec<usize>>,
}

/// Individual experience/memory
#[derive(Debug, Clone)]
pub struct Experience {
    /// Input data
    pub input: Array1<f64>,

    /// Target output
    pub target: Array1<f64>,

    /// Task identifier
    pub task_id: String,

    /// Importance score
    pub importance: f64,

    /// Gradient information (optional)
    pub gradient_info: Option<Array1<f64>>,

    /// Uncertainty measure
    pub uncertainty: Option<f64>,
}

/// Quantum continual learner
pub struct QuantumContinualLearner {
    /// Base quantum model
    model: QuantumNeuralNetwork,

    /// Continual learning strategy
    strategy: ContinualLearningStrategy,

    /// Task sequence and history
    task_history: Vec<ContinualTask>,

    /// Current task index
    current_task: Option<usize>,

    /// Memory buffer
    memory_buffer: Option<MemoryBuffer>,

    /// Fisher information (for EWC)
    fisher_information: Option<Array1<f64>>,

    /// Previous task parameters (for EWC)
    previous_parameters: Option<Array1<f64>>,

    /// Progressive modules (for Progressive Networks)
    progressive_modules: Vec<QuantumNeuralNetwork>,

    /// Parameter masks (for Parameter Isolation)
    parameter_masks: HashMap<String, Array1<bool>>,

    /// Performance metrics per task
    task_metrics: HashMap<String, TaskMetrics>,

    /// Forgetting metrics
    forgetting_metrics: ForgettingMetrics,
}

/// Metrics for individual tasks
#[derive(Debug, Clone)]
pub struct TaskMetrics {
    /// Accuracy on current task
    pub current_accuracy: f64,

    /// Accuracy after learning subsequent tasks
    pub retained_accuracy: f64,

    /// Learning speed (epochs to convergence)
    pub learning_speed: usize,

    /// Backward transfer (improvement from future tasks)
    pub backward_transfer: f64,

    /// Forward transfer (help to future tasks)
    pub forward_transfer: f64,
}

/// Overall forgetting and transfer metrics
#[derive(Debug, Clone)]
pub struct ForgettingMetrics {
    /// Average accuracy across all seen tasks
    pub average_accuracy: f64,

    /// Catastrophic forgetting measure
    pub forgetting_measure: f64,

    /// Backward transfer coefficient
    pub backward_transfer: f64,

    /// Forward transfer coefficient
    pub forward_transfer: f64,

    /// Overall continual learning score
    pub continual_learning_score: f64,

    /// Per-task forgetting
    pub per_task_forgetting: HashMap<String, f64>,
}

impl QuantumContinualLearner {
    /// Create a new quantum continual learner
    pub fn new(model: QuantumNeuralNetwork, strategy: ContinualLearningStrategy) -> Self {
        let memory_buffer = match &strategy {
            ContinualLearningStrategy::ExperienceReplay { buffer_size, .. } => Some(
                MemoryBuffer::new(*buffer_size, MemorySelectionStrategy::Random),
            ),
            ContinualLearningStrategy::GradientEpisodicMemory { .. } => Some(MemoryBuffer::new(
                1000,
                MemorySelectionStrategy::GradientImportance,
            )),
            _ => None,
        };

        Self {
            model,
            strategy,
            task_history: Vec::new(),
            current_task: None,
            memory_buffer,
            fisher_information: None,
            previous_parameters: None,
            progressive_modules: Vec::new(),
            parameter_masks: HashMap::new(),
            task_metrics: HashMap::new(),
            forgetting_metrics: ForgettingMetrics {
                average_accuracy: 0.0,
                forgetting_measure: 0.0,
                backward_transfer: 0.0,
                forward_transfer: 0.0,
                continual_learning_score: 0.0,
                per_task_forgetting: HashMap::new(),
            },
        }
    }

    /// Learn a new task
    pub fn learn_task(
        &mut self,
        task: ContinualTask,
        optimizer: &mut dyn Optimizer,
        epochs: usize,
    ) -> Result<TaskMetrics> {
        println!("Learning task: {}", task.task_id);

        // Store task in history
        self.task_history.push(task.clone());
        self.current_task = Some(self.task_history.len() - 1);

        // Apply continual learning strategy before training
        self.apply_pre_training_strategy(&task)?;

        // Train on the new task
        let start_time = std::time::Instant::now();
        let learning_losses = self.train_on_task(&task, optimizer, epochs)?;
        let learning_time = start_time.elapsed();

        // Apply post-training strategy
        self.apply_post_training_strategy(&task)?;

        // Evaluate on current task
        let current_accuracy = self.evaluate_task(&task)?;

        // Update memory buffer if applicable
        if self.memory_buffer.is_some() {
            let mut buffer = self
                .memory_buffer
                .take()
                .expect("memory_buffer verified to be Some above");
            self.update_memory_buffer(&mut buffer, &task)?;
            self.memory_buffer = Some(buffer);
        }

        // Compute task metrics
        let task_metrics = TaskMetrics {
            current_accuracy,
            retained_accuracy: current_accuracy, // Will be updated later
            learning_speed: epochs,              // Simplified - could track convergence
            backward_transfer: 0.0,              // Will be computed later
            forward_transfer: 0.0,               // Will be computed when future tasks are learned
        };

        self.task_metrics
            .insert(task.task_id.clone(), task_metrics.clone());

        // Update overall metrics
        self.update_forgetting_metrics()?;

        println!(
            "Task {} learned with accuracy: {:.3}",
            task.task_id, current_accuracy
        );

        Ok(task_metrics)
    }

    /// Train on a specific task
    fn train_on_task(
        &mut self,
        task: &ContinualTask,
        optimizer: &mut dyn Optimizer,
        epochs: usize,
    ) -> Result<Vec<f64>> {
        let mut losses = Vec::new();
        let batch_size = 32;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let num_batches = (task.train_data.nrows() + batch_size - 1) / batch_size;

            for batch_idx in 0..num_batches {
                let batch_start = batch_idx * batch_size;
                let batch_end = (batch_start + batch_size).min(task.train_data.nrows());

                let batch_data = task
                    .train_data
                    .slice(s![batch_start..batch_end, ..])
                    .to_owned();
                let batch_labels = task
                    .train_labels
                    .slice(s![batch_start..batch_end])
                    .to_owned();

                // Create combined training batch with replay if applicable
                let (final_data, final_labels) =
                    self.create_training_batch(&batch_data, &batch_labels, task)?;

                // Compute loss with continual learning regularization
                let batch_loss = self.compute_continual_loss(&final_data, &final_labels, task)?;
                epoch_loss += batch_loss;

                // Update model parameters (simplified)
                // In practice, this would use proper backpropagation
            }

            epoch_loss /= num_batches as f64;
            losses.push(epoch_loss);

            if epoch % 10 == 0 {
                println!("  Epoch {}: Loss = {:.4}", epoch, epoch_loss);
            }
        }

        Ok(losses)
    }

    /// Apply pre-training strategy
    fn apply_pre_training_strategy(&mut self, task: &ContinualTask) -> Result<()> {
        let strategy = self.strategy.clone();
        match strategy {
            ContinualLearningStrategy::ElasticWeightConsolidation { .. } => {
                if !self.task_history.is_empty() {
                    // Store current parameters and compute Fisher information
                    self.previous_parameters = Some(self.model.parameters.clone());
                    self.compute_fisher_information()?;
                }
            }

            ContinualLearningStrategy::ProgressiveNetworks {
                lateral_connections,
                adaptation_layers,
            } => {
                // Create new column for the new task
                self.create_progressive_column(adaptation_layers)?;
            }

            ContinualLearningStrategy::ParameterIsolation {
                allocation_strategy,
                ..
            } => {
                // Allocate parameters for the new task
                self.allocate_parameters_for_task(task, &allocation_strategy)?;
            }

            _ => {}
        }

        Ok(())
    }

    /// Apply post-training strategy
    fn apply_post_training_strategy(&mut self, task: &ContinualTask) -> Result<()> {
        match &self.strategy {
            ContinualLearningStrategy::ExperienceReplay { .. } => {
                // Memory buffer already updated during training
            }

            ContinualLearningStrategy::GradientEpisodicMemory { .. } => {
                // Compute and store gradient information
                self.compute_gradient_memory(task)?;
            }

            _ => {}
        }

        Ok(())
    }

    /// Create training batch with replay if applicable
    fn create_training_batch(
        &self,
        current_data: &Array2<f64>,
        current_labels: &Array1<usize>,
        task: &ContinualTask,
    ) -> Result<(Array2<f64>, Array1<usize>)> {
        match &self.strategy {
            ContinualLearningStrategy::ExperienceReplay { replay_ratio, .. } => {
                if let Some(ref buffer) = self.memory_buffer {
                    let num_replay = (current_data.nrows() as f64 * replay_ratio) as usize;
                    let replay_experiences = buffer.sample(num_replay);

                    // Combine current and replay data
                    let mut combined_data = current_data.clone();
                    let mut combined_labels = current_labels.clone();

                    for experience in replay_experiences {
                        // Add replay data (simplified)
                        // In practice, would properly combine arrays
                    }

                    Ok((combined_data, combined_labels))
                } else {
                    Ok((current_data.clone(), current_labels.clone()))
                }
            }
            _ => Ok((current_data.clone(), current_labels.clone())),
        }
    }

    /// Compute continual learning loss with regularization
    fn compute_continual_loss(
        &self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
        task: &ContinualTask,
    ) -> Result<f64> {
        // Base loss (simplified)
        let mut total_loss = 0.0;

        for (input, &label) in data.outer_iter().zip(labels.iter()) {
            let output = self.model.forward(&input.to_owned())?;
            total_loss += self.cross_entropy_loss(&output, label);
        }

        let base_loss = total_loss / data.nrows() as f64;

        // Add continual learning regularization
        let regularization = match &self.strategy {
            ContinualLearningStrategy::ElasticWeightConsolidation {
                importance_weight, ..
            } => self.compute_ewc_regularization(*importance_weight),

            ContinualLearningStrategy::LearningWithoutForgetting {
                distillation_weight,
                temperature,
            } => self.compute_lwf_regularization(*distillation_weight, *temperature, data)?,

            ContinualLearningStrategy::QuantumRegularization {
                entanglement_preservation,
                parameter_drift_penalty,
            } => self.compute_quantum_regularization(
                *entanglement_preservation,
                *parameter_drift_penalty,
            ),

            _ => 0.0,
        };

        Ok(base_loss + regularization)
    }

    /// Compute EWC regularization term
    fn compute_ewc_regularization(&self, importance_weight: f64) -> f64 {
        if let (Some(ref fisher), Some(ref prev_params)) =
            (&self.fisher_information, &self.previous_parameters)
        {
            let param_diff = &self.model.parameters - prev_params;
            let ewc_term = fisher * &param_diff.mapv(|x| x.powi(2));
            importance_weight * ewc_term.sum() / 2.0
        } else {
            0.0
        }
    }

    /// Compute Learning without Forgetting regularization
    fn compute_lwf_regularization(
        &self,
        distillation_weight: f64,
        temperature: f64,
        data: &Array2<f64>,
    ) -> Result<f64> {
        if self.task_history.len() <= 1 {
            return Ok(0.0);
        }

        // Compute distillation loss (simplified)
        let mut distillation_loss = 0.0;

        for input in data.outer_iter() {
            let current_output = self.model.forward(&input.to_owned())?;

            // Get "teacher" output from previous model state (simplified)
            // In practice, would store previous model or compute with masked parameters
            let teacher_output = current_output.clone(); // Placeholder

            // Compute KL divergence with temperature scaling
            let student_probs = self.softmax_with_temperature(&current_output, temperature);
            let teacher_probs = self.softmax_with_temperature(&teacher_output, temperature);

            for (s, t) in student_probs.iter().zip(teacher_probs.iter()) {
                if *t > 1e-10 {
                    distillation_loss += t * (t / s).ln();
                }
            }
        }

        Ok(distillation_weight * distillation_loss / data.nrows() as f64)
    }

    /// Compute quantum-specific regularization
    fn compute_quantum_regularization(
        &self,
        entanglement_preservation: f64,
        parameter_drift_penalty: f64,
    ) -> f64 {
        let mut regularization = 0.0;

        // Entanglement preservation penalty
        if let Some(ref prev_params) = self.previous_parameters {
            let param_diff = &self.model.parameters - prev_params;

            // Penalize changes that might reduce entanglement capability
            let entanglement_penalty = param_diff.mapv(|x| x.abs()).sum();
            regularization += entanglement_preservation * entanglement_penalty;
        }

        // Parameter drift penalty (encourage small changes)
        if let Some(ref prev_params) = self.previous_parameters {
            let drift = (&self.model.parameters - prev_params)
                .mapv(|x| x.powi(2))
                .sum();
            regularization += parameter_drift_penalty * drift;
        }

        regularization
    }

    /// Compute Fisher information matrix for EWC
    fn compute_fisher_information(&mut self) -> Result<()> {
        if let ContinualLearningStrategy::ElasticWeightConsolidation { fisher_samples, .. } =
            &self.strategy
        {
            let mut fisher = Array1::zeros(self.model.parameters.len());

            // Sample data from previous tasks for Fisher computation
            if let Some(current_task_idx) = self.current_task {
                if current_task_idx > 0 {
                    // Use previous task data (simplified)
                    let prev_task = &self.task_history[current_task_idx - 1];

                    for i in 0..*fisher_samples {
                        let idx = i % prev_task.train_data.nrows();
                        let input = prev_task.train_data.row(idx).to_owned();
                        let label = prev_task.train_labels[idx];

                        // Compute gradient (simplified - would use automatic differentiation)
                        let gradient = self.compute_parameter_gradient(&input, label)?;
                        fisher = fisher + &gradient.mapv(|x| x.powi(2));
                    }

                    fisher = fisher / *fisher_samples as f64;
                }
            }

            self.fisher_information = Some(fisher);
        }

        Ok(())
    }

    /// Create progressive network column
    fn create_progressive_column(&mut self, adaptation_layers: usize) -> Result<()> {
        // Create a new small network for the new task
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 6 },
        ];

        let progressive_module = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;
        self.progressive_modules.push(progressive_module);

        Ok(())
    }

    /// Allocate parameters for new task
    fn allocate_parameters_for_task(
        &mut self,
        task: &ContinualTask,
        strategy: &ParameterAllocationStrategy,
    ) -> Result<()> {
        match strategy {
            ParameterAllocationStrategy::Masking => {
                // Create mask for this task
                let mask = Array1::from_elem(self.model.parameters.len(), true);
                // In practice, would compute optimal mask
                self.parameter_masks.insert(task.task_id.clone(), mask);
            }

            ParameterAllocationStrategy::Expansion => {
                // Expand model capacity if needed
                // This would require modifying the model architecture
            }

            _ => {}
        }

        Ok(())
    }

    /// Compute gradient memory for GEM
    fn compute_gradient_memory(&mut self, task: &ContinualTask) -> Result<()> {
        if self.memory_buffer.is_some() {
            let mut buffer = self
                .memory_buffer
                .take()
                .expect("memory_buffer verified to be Some above");

            // Store representative examples with their gradients
            for i in 0..task.train_data.nrows().min(100) {
                let input = task.train_data.row(i).to_owned();
                let label = task.train_labels[i];

                let gradient = self.compute_parameter_gradient(&input, label)?;

                let experience = Experience {
                    input,
                    target: Array1::from_elem(task.num_classes, 0.0), // Simplified
                    task_id: task.task_id.clone(),
                    importance: 1.0,
                    gradient_info: Some(gradient),
                    uncertainty: None,
                };

                buffer.add_experience(experience);
            }

            self.memory_buffer = Some(buffer);
        }

        Ok(())
    }

    /// Update memory buffer with new experiences
    fn update_memory_buffer(&self, buffer: &mut MemoryBuffer, task: &ContinualTask) -> Result<()> {
        // Add experiences from the new task
        for i in 0..task.train_data.nrows() {
            let input = task.train_data.row(i).to_owned();
            let target = Array1::from_elem(task.num_classes, 0.0); // Simplified encoding

            let experience = Experience {
                input,
                target,
                task_id: task.task_id.clone(),
                importance: 1.0,
                gradient_info: None,
                uncertainty: None,
            };

            buffer.add_experience(experience);
        }

        Ok(())
    }

    /// Evaluate model on a specific task
    fn evaluate_task(&self, task: &ContinualTask) -> Result<f64> {
        let mut correct = 0;
        let total = task.val_data.nrows();

        for (input, &label) in task.val_data.outer_iter().zip(task.val_labels.iter()) {
            let output = self.model.forward(&input.to_owned())?;
            let predicted = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if predicted == label {
                correct += 1;
            }
        }

        Ok(correct as f64 / total as f64)
    }

    /// Evaluate all previous tasks to measure forgetting
    pub fn evaluate_all_tasks(&mut self) -> Result<HashMap<String, f64>> {
        let mut accuracies = HashMap::new();

        for task in &self.task_history {
            let accuracy = self.evaluate_task(task)?;
            accuracies.insert(task.task_id.clone(), accuracy);

            // Update retained accuracy in task metrics
            if let Some(metrics) = self.task_metrics.get_mut(&task.task_id) {
                metrics.retained_accuracy = accuracy;
            }
        }

        Ok(accuracies)
    }

    /// Update forgetting metrics
    fn update_forgetting_metrics(&mut self) -> Result<()> {
        if self.task_history.is_empty() {
            return Ok(());
        }

        // Evaluate all tasks
        let accuracies = self.evaluate_all_tasks()?;

        // Compute average accuracy
        let avg_accuracy = accuracies.values().sum::<f64>() / accuracies.len() as f64;
        self.forgetting_metrics.average_accuracy = avg_accuracy;

        // Compute forgetting measure
        let mut total_forgetting = 0.0;
        let mut num_comparisons = 0;

        for (task_id, metrics) in &self.task_metrics {
            let current_acc = accuracies.get(task_id).unwrap_or(&0.0);
            let original_acc = metrics.current_accuracy;

            if original_acc > 0.0 {
                let forgetting = (original_acc - current_acc).max(0.0);
                total_forgetting += forgetting;
                num_comparisons += 1;

                self.forgetting_metrics
                    .per_task_forgetting
                    .insert(task_id.clone(), forgetting);
            }
        }

        if num_comparisons > 0 {
            self.forgetting_metrics.forgetting_measure = total_forgetting / num_comparisons as f64;
        }

        // Compute continual learning score (simplified)
        self.forgetting_metrics.continual_learning_score =
            avg_accuracy - self.forgetting_metrics.forgetting_measure;

        Ok(())
    }

    /// Compute parameter gradient (simplified)
    fn compute_parameter_gradient(&self, input: &Array1<f64>, label: usize) -> Result<Array1<f64>> {
        // Placeholder for gradient computation
        // In practice, would use automatic differentiation
        Ok(Array1::zeros(self.model.parameters.len()))
    }

    /// Cross-entropy loss
    fn cross_entropy_loss(&self, output: &Array1<f64>, label: usize) -> f64 {
        let predicted_prob = output[label].max(1e-10);
        -predicted_prob.ln()
    }

    /// Softmax with temperature
    fn softmax_with_temperature(&self, logits: &Array1<f64>, temperature: f64) -> Array1<f64> {
        let scaled_logits = logits / temperature;
        let max_logit = scaled_logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_logits = scaled_logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        exp_logits / sum_exp
    }

    /// Get forgetting metrics
    pub fn get_forgetting_metrics(&self) -> &ForgettingMetrics {
        &self.forgetting_metrics
    }

    /// Get task metrics
    pub fn get_task_metrics(&self) -> &HashMap<String, TaskMetrics> {
        &self.task_metrics
    }

    /// Get current model
    pub fn get_model(&self) -> &QuantumNeuralNetwork {
        &self.model
    }

    /// Reset for new task sequence
    pub fn reset(&mut self) {
        self.task_history.clear();
        self.current_task = None;
        self.fisher_information = None;
        self.previous_parameters = None;
        self.progressive_modules.clear();
        self.parameter_masks.clear();
        self.task_metrics.clear();

        if let Some(ref mut buffer) = self.memory_buffer {
            buffer.clear();
        }
    }
}

impl MemoryBuffer {
    /// Create new memory buffer
    pub fn new(max_size: usize, strategy: MemorySelectionStrategy) -> Self {
        Self {
            experiences: VecDeque::new(),
            max_size,
            selection_strategy: strategy,
            task_memories: HashMap::new(),
        }
    }

    /// Add experience to buffer
    pub fn add_experience(&mut self, experience: Experience) {
        // Add to main buffer
        if self.experiences.len() >= self.max_size {
            let removed = self
                .experiences
                .pop_front()
                .expect("Buffer is non-empty when len >= max_size");
            self.remove_from_task_index(&removed);
        }

        let experience_idx = self.experiences.len();
        self.experiences.push_back(experience.clone());

        // Update task index
        self.task_memories
            .entry(experience.task_id.clone())
            .or_insert_with(Vec::new)
            .push(experience_idx);
    }

    /// Sample experiences from buffer
    pub fn sample(&self, num_samples: usize) -> Vec<Experience> {
        let mut samples = Vec::new();

        let available = self.experiences.len().min(num_samples);

        match self.selection_strategy {
            MemorySelectionStrategy::Random => {
                for _ in 0..available {
                    let idx = fastrand::usize(0..self.experiences.len());
                    samples.push(self.experiences[idx].clone());
                }
            }

            MemorySelectionStrategy::GradientImportance => {
                // Sort by gradient importance and sample top experiences
                let mut indexed_experiences: Vec<_> = self.experiences.iter().enumerate().collect();

                indexed_experiences.sort_by(|a, b| {
                    let importance_a = a.1.importance;
                    let importance_b = b.1.importance;
                    importance_b
                        .partial_cmp(&importance_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for (_, experience) in indexed_experiences.into_iter().take(available) {
                    samples.push(experience.clone());
                }
            }

            _ => {
                // Fallback to random sampling
                for _ in 0..available {
                    let idx = fastrand::usize(0..self.experiences.len());
                    samples.push(self.experiences[idx].clone());
                }
            }
        }

        samples
    }

    /// Remove experience from task index
    fn remove_from_task_index(&mut self, experience: &Experience) {
        if let Some(indices) = self.task_memories.get_mut(&experience.task_id) {
            // This is simplified - in practice would need to update all indices
            indices.clear();
        }
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.experiences.clear();
        self.task_memories.clear();
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.experiences.len()
    }
}

/// Helper function to create a simple continual task
pub fn create_continual_task(
    task_id: String,
    task_type: TaskType,
    data: Array2<f64>,
    labels: Array1<usize>,
    train_ratio: f64,
) -> ContinualTask {
    let train_size = (data.nrows() as f64 * train_ratio) as usize;

    let train_data = data.slice(s![0..train_size, ..]).to_owned();
    let train_labels = labels.slice(s![0..train_size]).to_owned();

    let val_data = data.slice(s![train_size.., ..]).to_owned();
    let val_labels = labels.slice(s![train_size..]).to_owned();

    let num_classes = labels.iter().max().unwrap_or(&0) + 1;

    ContinualTask {
        task_id,
        task_type,
        train_data,
        train_labels,
        val_data,
        val_labels,
        num_classes,
        metadata: HashMap::new(),
    }
}

/// Helper function to generate synthetic task sequence
pub fn generate_task_sequence(
    num_tasks: usize,
    samples_per_task: usize,
    feature_dim: usize,
) -> Vec<ContinualTask> {
    let mut tasks = Vec::new();

    for i in 0..num_tasks {
        // Generate task-specific data with some variation
        let data = Array2::from_shape_fn((samples_per_task, feature_dim), |(row, col)| {
            let task_shift = i as f64 * 0.5;
            let base_value = row as f64 / samples_per_task as f64 + col as f64 / feature_dim as f64;
            0.5 + 0.3 * (base_value * 2.0 * PI + task_shift).sin() + 0.1 * (fastrand::f64() - 0.5)
        });

        let labels = Array1::from_shape_fn(samples_per_task, |row| {
            // Binary classification based on sum of features
            let sum = data.row(row).sum();
            if sum > feature_dim as f64 * 0.5 {
                1
            } else {
                0
            }
        });

        let task = create_continual_task(
            format!("task_{}", i),
            TaskType::Classification { num_classes: 2 },
            data,
            labels,
            0.8, // 80% training, 20% validation
        );

        tasks.push(task);
    }

    tasks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::optimizers::Adam;
    use crate::qnn::QNNLayerType;

    #[test]
    fn test_memory_buffer() {
        let mut buffer = MemoryBuffer::new(5, MemorySelectionStrategy::Random);

        for i in 0..10 {
            let experience = Experience {
                input: Array1::from_vec(vec![i as f64]),
                target: Array1::from_vec(vec![(i % 2) as f64]),
                task_id: format!("task_{}", i / 3),
                importance: i as f64,
                gradient_info: None,
                uncertainty: None,
            };

            buffer.add_experience(experience);
        }

        assert_eq!(buffer.size(), 5);

        let samples = buffer.sample(3);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_continual_task_creation() {
        let data = Array2::from_shape_fn((100, 4), |(i, j)| (i as f64 + j as f64) / 50.0);
        let labels = Array1::from_shape_fn(100, |i| i % 3);

        let task = create_continual_task(
            "test_task".to_string(),
            TaskType::Classification { num_classes: 3 },
            data,
            labels,
            0.7,
        );

        assert_eq!(task.task_id, "test_task");
        assert_eq!(task.train_data.nrows(), 70);
        assert_eq!(task.val_data.nrows(), 30);
        assert_eq!(task.num_classes, 3);
    }

    #[test]
    fn test_continual_learner_creation() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let model = QuantumNeuralNetwork::new(layers, 4, 4, 2).expect("Failed to create model");

        let strategy = ContinualLearningStrategy::ElasticWeightConsolidation {
            importance_weight: 1000.0,
            fisher_samples: 100,
        };

        let learner = QuantumContinualLearner::new(model, strategy);

        assert_eq!(learner.task_history.len(), 0);
        assert!(learner.current_task.is_none());
    }

    #[test]
    fn test_task_sequence_generation() {
        let tasks = generate_task_sequence(3, 50, 4);

        assert_eq!(tasks.len(), 3);

        for (i, task) in tasks.iter().enumerate() {
            assert_eq!(task.task_id, format!("task_{}", i));
            assert_eq!(task.train_data.nrows(), 40); // 80% of 50
            assert_eq!(task.val_data.nrows(), 10); // 20% of 50
            assert_eq!(task.train_data.ncols(), 4);
        }
    }

    #[test]
    fn test_forgetting_metrics() {
        let metrics = ForgettingMetrics {
            average_accuracy: 0.75,
            forgetting_measure: 0.15,
            backward_transfer: 0.05,
            forward_transfer: 0.1,
            continual_learning_score: 0.6,
            per_task_forgetting: HashMap::new(),
        };

        assert_eq!(metrics.average_accuracy, 0.75);
        assert_eq!(metrics.forgetting_measure, 0.15);
        assert!(metrics.continual_learning_score > 0.5);
    }
}
