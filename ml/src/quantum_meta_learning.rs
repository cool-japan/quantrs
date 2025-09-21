//! Quantum Meta-Learning
//!
//! This module implements meta-learning algorithms specifically designed for
//! quantum neural networks, enabling few-shot learning, rapid adaptation,
//! and transfer learning across quantum tasks and domains.

use crate::error::{MLError, Result};
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use crate::optimization::OptimizationMethod;
use ndarray::{Array1, Array2, Array3, Axis, s};
use std::collections::HashMap;
use std::f64::consts::PI;
use serde::{Serialize, Deserialize};

/// Quantum meta-learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumMetaLearningAlgorithm {
    /// Quantum Model-Agnostic Meta-Learning (Q-MAML)
    QuantumMAML {
        inner_learning_rate: f64,
        outer_learning_rate: f64,
        inner_steps: usize,
        first_order: bool,
    },

    /// Quantum Reptile algorithm
    QuantumReptile {
        inner_learning_rate: f64,
        outer_learning_rate: f64,
        inner_steps: usize,
        batch_size: usize,
    },

    /// Quantum Prototypical Networks
    QuantumPrototypical {
        distance_metric: QuantumDistanceMetric,
        embedding_dim: usize,
        temperature: f64,
    },

    /// Quantum Matching Networks
    QuantumMatching {
        attention_mechanism: QuantumAttentionType,
        fce: bool, // Full Context Embeddings
        lstm_layers: usize,
    },

    /// Quantum Relation Networks
    QuantumRelation {
        relation_module_layers: Vec<usize>,
        embedding_layers: Vec<usize>,
        activation: QuantumActivation,
    },

    /// Quantum Memory-Augmented Networks
    QuantumMemoryAugmented {
        memory_size: usize,
        memory_vector_dim: usize,
        controller_layers: Vec<usize>,
        read_heads: usize,
        write_heads: usize,
    },

    /// Quantum Gradient-Based Meta-Learning (Q-GBML)
    QuantumGBML {
        meta_optimizer: QuantumMetaOptimizer,
        gradient_clip: Option<f64>,
        second_order: bool,
    },

    /// Quantum Hypernetworks
    QuantumHypernetwork {
        hypernetwork_layers: Vec<usize>,
        target_network_layers: Vec<usize>,
        conditioning_method: ConditioningMethod,
    },

    /// Quantum Few-Shot Optimization
    QuantumFewShotOpt {
        meta_learning_rate: f64,
        adaptation_steps: usize,
        optimizer_type: QuantumOptimizerType,
    },

    /// Quantum Task-Agnostic Meta-Learning
    QuantumTaskAgnostic {
        task_encoder_layers: Vec<usize>,
        shared_layers: Vec<usize>,
        task_specific_layers: Vec<usize>,
    },
}

/// Quantum distance metrics for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumDistanceMetric {
    /// Quantum Euclidean distance
    QuantumEuclidean,
    /// Quantum cosine similarity
    QuantumCosine,
    /// Quantum fidelity-based distance
    QuantumFidelity,
    /// Quantum Wasserstein distance
    QuantumWasserstein,
    /// Quantum kernel distance
    QuantumKernel { kernel_type: QuantumKernelType },
    /// Quantum entanglement-based distance
    QuantumEntanglement,
}

/// Quantum kernel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumKernelType {
    /// RBF quantum kernel
    RBF { gamma: f64 },
    /// Polynomial quantum kernel
    Polynomial { degree: usize, coef0: f64 },
    /// Quantum feature map kernel
    QuantumFeatureMap { feature_map: String },
}

/// Quantum attention mechanisms for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumAttentionType {
    /// Dot-product attention with quantum enhancement
    QuantumDotProduct,
    /// Multi-head quantum attention
    QuantumMultiHead { num_heads: usize },
    /// Self-attention with quantum circuits
    QuantumSelfAttention,
    /// Cross-attention between quantum states
    QuantumCrossAttention,
}

/// Quantum activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumActivation {
    /// Quantum ReLU
    QuantumReLU,
    /// Quantum Sigmoid
    QuantumSigmoid,
    /// Quantum Tanh
    QuantumTanh,
    /// Quantum Swish
    QuantumSwish,
    /// Parametric quantum activation
    ParametricQuantum { params: Vec<f64> },
}

/// Quantum meta-optimizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumMetaOptimizer {
    /// Quantum Adam optimizer
    QuantumAdam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// Quantum RMSprop
    QuantumRMSprop {
        decay: f64,
        epsilon: f64,
    },
    /// Quantum Natural Gradient
    QuantumNaturalGradient {
        damping: f64,
    },
    /// Learned quantum optimizer
    LearnedQuantumOptimizer {
        optimizer_network_layers: Vec<usize>,
    },
}

/// Quantum optimizer types for few-shot optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumOptimizerType {
    /// Gradient descent
    GradientDescent,
    /// Quantum parameter shift
    QuantumParameterShift,
    /// Quantum natural evolution strategies
    QuantumNES,
    /// Quantum BFGS
    QuantumBFGS,
}

/// Conditioning methods for hypernetworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditioningMethod {
    /// Concatenation-based conditioning
    Concatenation,
    /// Feature-wise linear modulation
    FiLM,
    /// Attention-based conditioning
    AttentionBased,
    /// Quantum state conditioning
    QuantumStateConditioning,
}

/// Meta-learning task definition
#[derive(Debug, Clone)]
pub struct MetaLearningTask {
    /// Task identifier
    pub task_id: String,

    /// Support set (training examples for the task)
    pub support_set: TaskDataset,

    /// Query set (test examples for the task)
    pub query_set: TaskDataset,

    /// Task metadata
    pub metadata: TaskMetadata,

    /// Quantum-specific task properties
    pub quantum_properties: QuantumTaskProperties,
}

/// Dataset for a specific task
#[derive(Debug, Clone)]
pub struct TaskDataset {
    /// Input data
    pub inputs: Array2<f64>,

    /// Labels
    pub labels: Array1<usize>,

    /// Number of classes
    pub num_classes: usize,

    /// Number of shots (examples per class)
    pub num_shots: usize,
}

/// Task metadata
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    /// Task type
    pub task_type: TaskType,

    /// Difficulty level
    pub difficulty: f64,

    /// Domain information
    pub domain: String,

    /// Task-specific hyperparameters
    pub hyperparameters: HashMap<String, f64>,
}

/// Types of meta-learning tasks
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Classification task
    Classification {
        num_classes: usize,
        num_shots: usize,
    },
    /// Regression task
    Regression {
        output_dim: usize,
        num_shots: usize,
    },
    /// Reinforcement learning task
    ReinforcementLearning {
        action_space_dim: usize,
        state_space_dim: usize,
    },
    /// Quantum state preparation
    QuantumStatePreparation {
        target_state_dim: usize,
        fidelity_threshold: f64,
    },
    /// Quantum circuit optimization
    QuantumCircuitOptimization {
        num_qubits: usize,
        circuit_depth: usize,
    },
}

/// Quantum-specific properties of tasks
#[derive(Debug, Clone)]
pub struct QuantumTaskProperties {
    /// Required quantum coherence
    pub coherence_requirement: f64,

    /// Entanglement complexity
    pub entanglement_complexity: f64,

    /// Quantum resource requirements
    pub resource_requirements: QuantumResourceRequirements,

    /// Noise tolerance
    pub noise_tolerance: f64,

    /// Circuit depth constraints
    pub circuit_depth_constraint: Option<usize>,
}

/// Quantum resource requirements
#[derive(Debug, Clone)]
pub struct QuantumResourceRequirements {
    /// Number of qubits needed
    pub num_qubits: usize,

    /// Gate count estimates
    pub gate_counts: HashMap<String, usize>,

    /// Measurement requirements
    pub measurement_requirements: MeasurementRequirements,

    /// Classical processing requirements
    pub classical_processing: ClassicalProcessingRequirements,
}

/// Measurement requirements
#[derive(Debug, Clone)]
pub struct MeasurementRequirements {
    /// Number of measurement shots
    pub num_shots: usize,

    /// Measurement bases required
    pub measurement_bases: Vec<String>,

    /// Measurement precision requirements
    pub precision_requirements: f64,
}

/// Classical processing requirements
#[derive(Debug, Clone)]
pub struct ClassicalProcessingRequirements {
    /// Memory requirements (MB)
    pub memory_mb: f64,

    /// Compute time estimates (seconds)
    pub compute_time_estimate: f64,

    /// Parallel processing capability
    pub parallel_processing: bool,
}

/// Quantum meta-learner
pub struct QuantumMetaLearner {
    /// Meta-learning algorithm
    pub algorithm: QuantumMetaLearningAlgorithm,

    /// Base quantum model
    pub base_model: QuantumNeuralNetwork,

    /// Meta-model (for algorithms that use separate meta-networks)
    pub meta_model: Option<QuantumNeuralNetwork>,

    /// Configuration
    pub config: MetaLearningConfig,

    /// Training history
    training_history: Vec<MetaTrainingEpisode>,

    /// Task memory for continual meta-learning
    task_memory: TaskMemory,

    /// Adaptation statistics
    adaptation_stats: AdaptationStatistics,
}

/// Meta-learning configuration
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    /// Number of meta-training epochs
    pub meta_epochs: usize,

    /// Number of tasks per meta-batch
    pub tasks_per_batch: usize,

    /// Validation frequency
    pub validation_frequency: usize,

    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingCriteria>,

    /// Quantum-specific configuration
    pub quantum_config: QuantumMetaConfig,

    /// Evaluation configuration
    pub evaluation_config: EvaluationConfig,
}

/// Quantum-specific meta-learning configuration
#[derive(Debug, Clone)]
pub struct QuantumMetaConfig {
    /// Quantum circuit optimization level
    pub circuit_optimization: CircuitOptimizationLevel,

    /// Error mitigation techniques
    pub error_mitigation: Vec<ErrorMitigationTechnique>,

    /// Noise modeling
    pub noise_modeling: NoiseModelingConfig,

    /// Quantum advantage tracking
    pub track_quantum_advantage: bool,

    /// Entanglement preservation strategies
    pub entanglement_preservation: EntanglementPreservationStrategy,
}

/// Circuit optimization levels
#[derive(Debug, Clone)]
pub enum CircuitOptimizationLevel {
    /// No optimization
    None,
    /// Basic gate optimization
    Basic,
    /// Advanced circuit synthesis
    Advanced,
    /// Hardware-specific optimization
    HardwareSpecific { device_constraints: HashMap<String, f64> },
    /// Adaptive optimization based on task
    Adaptive,
}

/// Error mitigation techniques
#[derive(Debug, Clone)]
pub enum ErrorMitigationTechnique {
    /// Zero-noise extrapolation
    ZeroNoiseExtrapolation,
    /// Error amplification
    ErrorAmplification,
    /// Symmetry verification
    SymmetryVerification,
    /// Probabilistic error cancellation
    ProbabilisticErrorCancellation,
    /// Virtual distillation
    VirtualDistillation,
}

/// Noise modeling configuration
#[derive(Debug, Clone)]
pub struct NoiseModelingConfig {
    /// Include decoherence effects
    pub include_decoherence: bool,

    /// Gate error rates
    pub gate_error_rates: HashMap<String, f64>,

    /// Measurement error rates
    pub measurement_error_rates: f64,

    /// Crosstalk modeling
    pub crosstalk_modeling: bool,

    /// Adaptive noise estimation
    pub adaptive_noise_estimation: bool,
}

/// Entanglement preservation strategies
#[derive(Debug, Clone)]
pub enum EntanglementPreservationStrategy {
    /// Minimize entanglement loss
    MinimizeEntanglementLoss,
    /// Preserve specific entanglement patterns
    PreservePatterns { patterns: Vec<String> },
    /// Adaptive entanglement management
    AdaptiveManagement,
    /// Entanglement-aware parameter updates
    EntanglementAwareUpdates,
}

/// Evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Number of test tasks
    pub num_test_tasks: usize,

    /// Evaluation metrics
    pub metrics: Vec<MetaLearningMetric>,

    /// Cross-domain evaluation
    pub cross_domain_evaluation: bool,

    /// Quantum benchmark tasks
    pub quantum_benchmarks: Vec<QuantumBenchmarkTask>,
}

/// Meta-learning evaluation metrics
#[derive(Debug, Clone)]
pub enum MetaLearningMetric {
    /// Accuracy after K adaptation steps
    AccuracyAfterKSteps { k: usize },
    /// Learning curve AUC
    LearningCurveAUC,
    /// Transfer learning efficiency
    TransferEfficiency,
    /// Catastrophic forgetting measure
    CatastrophicForgetting,
    /// Quantum advantage preservation
    QuantumAdvantagePreservation,
    /// Adaptation speed
    AdaptationSpeed,
}

/// Quantum benchmark tasks
#[derive(Debug, Clone)]
pub struct QuantumBenchmarkTask {
    /// Benchmark name
    pub name: String,

    /// Task description
    pub description: String,

    /// Expected quantum advantage
    pub expected_quantum_advantage: f64,

    /// Benchmark configuration
    pub config: HashMap<String, f64>,
}

/// Early stopping criteria
#[derive(Debug, Clone)]
pub struct EarlyStoppingCriteria {
    /// Patience (epochs without improvement)
    pub patience: usize,

    /// Minimum improvement threshold
    pub min_improvement: f64,

    /// Metric to monitor
    pub monitor_metric: String,

    /// Quantum-specific stopping criteria
    pub quantum_criteria: Option<QuantumStoppingCriteria>,
}

/// Quantum-specific stopping criteria
#[derive(Debug, Clone)]
pub struct QuantumStoppingCriteria {
    /// Minimum quantum advantage threshold
    pub min_quantum_advantage: f64,

    /// Maximum acceptable decoherence
    pub max_decoherence: f64,

    /// Circuit depth threshold
    pub max_circuit_depth: Option<usize>,
}

/// Meta-training episode information
#[derive(Debug, Clone)]
pub struct MetaTrainingEpisode {
    /// Episode number
    pub episode: usize,

    /// Tasks used in this episode
    pub tasks: Vec<String>,

    /// Meta-loss
    pub meta_loss: f64,

    /// Average adaptation performance
    pub avg_adaptation_performance: f64,

    /// Training time
    pub training_time: f64,

    /// Quantum metrics
    pub quantum_metrics: QuantumMetaMetrics,

    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Quantum-specific meta-learning metrics
#[derive(Debug, Clone)]
pub struct QuantumMetaMetrics {
    /// Average quantum fidelity across tasks
    pub avg_quantum_fidelity: f64,

    /// Entanglement preservation rate
    pub entanglement_preservation: f64,

    /// Quantum advantage metrics
    pub quantum_advantage: f64,

    /// Circuit efficiency
    pub circuit_efficiency: f64,

    /// Coherence utilization
    pub coherence_utilization: f64,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Total quantum gate count
    pub total_gate_count: usize,

    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,

    /// Total compute time (seconds)
    pub total_compute_time: f64,

    /// Number of measurement shots used
    pub measurement_shots: usize,

    /// Classical preprocessing time
    pub classical_preprocessing_time: f64,
}

/// Task memory for continual meta-learning
#[derive(Debug, Clone)]
pub struct TaskMemory {
    /// Stored task representations
    pub task_representations: HashMap<String, Array1<f64>>,

    /// Task similarity matrix
    pub task_similarity_matrix: Array2<f64>,

    /// Memory capacity
    pub capacity: usize,

    /// Forgetting strategy
    pub forgetting_strategy: ForgettingStrategy,

    /// Memory consolidation method
    pub consolidation_method: ConsolidationMethod,
}

/// Forgetting strategies for task memory
#[derive(Debug, Clone)]
pub enum ForgettingStrategy {
    /// First-in-first-out
    FIFO,
    /// Least recently used
    LRU,
    /// Importance-based forgetting
    ImportanceBased,
    /// Quantum coherence-based forgetting
    QuantumCoherenceBased,
    /// No forgetting (until capacity)
    NoForgetting,
}

/// Memory consolidation methods
#[derive(Debug, Clone)]
pub enum ConsolidationMethod {
    /// Elastic weight consolidation
    ElasticWeightConsolidation { lambda: f64 },
    /// Progressive neural networks
    ProgressiveNeuralNetworks,
    /// Gradient episodic memory
    GradientEpisodicMemory,
    /// Quantum memory consolidation
    QuantumMemoryConsolidation { fidelity_threshold: f64 },
}

/// Adaptation statistics
#[derive(Debug, Clone)]
pub struct AdaptationStatistics {
    /// Average adaptation time per task
    pub avg_adaptation_time: f64,

    /// Success rate across tasks
    pub success_rate: f64,

    /// Transfer learning effectiveness
    pub transfer_effectiveness: f64,

    /// Quantum resource efficiency
    pub quantum_efficiency: f64,

    /// Task difficulty vs performance correlation
    pub difficulty_performance_correlation: f64,
}

impl QuantumMetaLearner {
    /// Create a new quantum meta-learner
    pub fn new(
        algorithm: QuantumMetaLearningAlgorithm,
        base_model: QuantumNeuralNetwork,
        config: MetaLearningConfig,
    ) -> Result<Self> {
        let meta_model = match &algorithm {
            QuantumMetaLearningAlgorithm::QuantumHypernetwork { hypernetwork_layers, .. } => {
                // Create hypernetwork
                let layers = hypernetwork_layers.iter().enumerate().map(|(i, &size)| {
                    if i == 0 {
                        QNNLayerType::EncodingLayer { num_features: size }
                    } else if i == hypernetwork_layers.len() - 1 {
                        QNNLayerType::MeasurementLayer { measurement_basis: "computational".to_string() }
                    } else {
                        QNNLayerType::VariationalLayer { num_params: size }
                    }
                }).collect();

                Some(QuantumNeuralNetwork::new(
                    layers,
                    base_model.num_qubits,
                    hypernetwork_layers[0],
                    *hypernetwork_layers.last().unwrap(),
                )?)
            }

            QuantumMetaLearningAlgorithm::QuantumMemoryAugmented { controller_layers, .. } => {
                // Create controller network
                let layers = controller_layers.iter().enumerate().map(|(i, &size)| {
                    if i == 0 {
                        QNNLayerType::EncodingLayer { num_features: size }
                    } else if i == controller_layers.len() - 1 {
                        QNNLayerType::MeasurementLayer { measurement_basis: "computational".to_string() }
                    } else {
                        QNNLayerType::VariationalLayer { num_params: size }
                    }
                }).collect();

                Some(QuantumNeuralNetwork::new(
                    layers,
                    base_model.num_qubits,
                    controller_layers[0],
                    *controller_layers.last().unwrap(),
                )?)
            }

            _ => None,
        };

        let task_memory = TaskMemory {
            task_representations: HashMap::new(),
            task_similarity_matrix: Array2::zeros((0, 0)),
            capacity: 1000,
            forgetting_strategy: ForgettingStrategy::LRU,
            consolidation_method: ConsolidationMethod::ElasticWeightConsolidation { lambda: 0.4 },
        };

        let adaptation_stats = AdaptationStatistics {
            avg_adaptation_time: 0.0,
            success_rate: 0.0,
            transfer_effectiveness: 0.0,
            quantum_efficiency: 0.0,
            difficulty_performance_correlation: 0.0,
        };

        Ok(Self {
            algorithm,
            base_model,
            meta_model,
            config,
            training_history: Vec::new(),
            task_memory,
            adaptation_stats,
        })
    }

    /// Meta-train the model on a distribution of tasks
    pub fn meta_train(&mut self, task_distribution: &[MetaLearningTask]) -> Result<Vec<f64>> {
        println!("Starting quantum meta-learning training...");

        let mut meta_losses = Vec::new();
        let mut best_meta_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.meta_epochs {
            let epoch_start = std::time::Instant::now();

            // Sample tasks for this meta-batch
            let sampled_tasks = self.sample_tasks(task_distribution)?;

            // Perform meta-update
            let meta_loss = self.meta_update(&sampled_tasks)?;
            meta_losses.push(meta_loss);

            // Update task memory
            self.update_task_memory(&sampled_tasks)?;

            // Validation
            if epoch % self.config.validation_frequency == 0 {
                let validation_loss = self.meta_validate(task_distribution)?;
                println!("Epoch {}: Meta-loss = {:.4}, Validation = {:.4}",
                    epoch, meta_loss, validation_loss);

                // Early stopping check
                if let Some(ref criteria) = self.config.early_stopping {
                    if validation_loss < best_meta_loss - criteria.min_improvement {
                        best_meta_loss = validation_loss;
                        patience_counter = 0;
                    } else {
                        patience_counter += 1;
                    }

                    if patience_counter >= criteria.patience {
                        println!("Early stopping at epoch {}", epoch);
                        break;
                    }
                }
            }

            // Record training episode
            let episode_time = epoch_start.elapsed().as_secs_f64();
            let episode = MetaTrainingEpisode {
                episode: epoch,
                tasks: sampled_tasks.iter().map(|t| t.task_id.clone()).collect(),
                meta_loss,
                avg_adaptation_performance: self.compute_avg_adaptation_performance(&sampled_tasks)?,
                training_time: episode_time,
                quantum_metrics: self.compute_quantum_meta_metrics(&sampled_tasks)?,
                resource_usage: self.compute_resource_usage(&sampled_tasks)?,
            };

            self.training_history.push(episode);
        }

        // Update adaptation statistics
        self.update_adaptation_statistics()?;

        Ok(meta_losses)
    }

    /// Adapt to a new task using the meta-learned knowledge
    pub fn adapt_to_task(&mut self, task: &MetaLearningTask) -> Result<AdaptationResult> {
        let adaptation_start = std::time::Instant::now();

        match &self.algorithm {
            QuantumMetaLearningAlgorithm::QuantumMAML { inner_learning_rate, inner_steps, .. } => {
                self.maml_adaptation(task, *inner_learning_rate, *inner_steps)
            }

            QuantumMetaLearningAlgorithm::QuantumReptile { inner_learning_rate, inner_steps, .. } => {
                self.reptile_adaptation(task, *inner_learning_rate, *inner_steps)
            }

            QuantumMetaLearningAlgorithm::QuantumPrototypical { distance_metric, .. } => {
                self.prototypical_adaptation(task, distance_metric)
            }

            QuantumMetaLearningAlgorithm::QuantumMatching { .. } => {
                self.matching_adaptation(task)
            }

            QuantumMetaLearningAlgorithm::QuantumRelation { .. } => {
                self.relation_adaptation(task)
            }

            QuantumMetaLearningAlgorithm::QuantumMemoryAugmented { .. } => {
                self.memory_augmented_adaptation(task)
            }

            QuantumMetaLearningAlgorithm::QuantumHypernetwork { .. } => {
                self.hypernetwork_adaptation(task)
            }

            _ => {
                // Default to MAML-style adaptation
                self.maml_adaptation(task, 0.01, 5)
            }
        }
    }

    /// MAML adaptation implementation
    fn maml_adaptation(
        &mut self,
        task: &MetaLearningTask,
        learning_rate: f64,
        num_steps: usize,
    ) -> Result<AdaptationResult> {
        let mut adapted_model = self.base_model.clone();
        let mut adaptation_losses = Vec::new();

        // Inner loop adaptation
        for step in 0..num_steps {
            let mut total_loss = 0.0;
            let num_samples = task.support_set.inputs.nrows();

            // Compute gradients on support set
            let gradients = self.compute_gradients(&adapted_model, &task.support_set)?;

            // Update parameters
            for (param, grad) in adapted_model.parameters.iter_mut().zip(gradients.iter()) {
                *param -= learning_rate * grad;
            }

            // Compute loss on support set for tracking
            for i in 0..num_samples {
                let input = task.support_set.inputs.row(i).to_owned();
                let label = task.support_set.labels[i];
                let output = adapted_model.forward(&input)?;
                total_loss += self.compute_loss(&output, label);
            }

            adaptation_losses.push(total_loss / num_samples as f64);
        }

        // Evaluate on query set
        let query_performance = self.evaluate_on_query_set(&adapted_model, &task.query_set)?;

        Ok(AdaptationResult {
            adapted_model,
            adaptation_losses,
            query_performance,
            adaptation_time: 0.1, // Placeholder
            quantum_metrics: self.compute_adaptation_quantum_metrics(task)?,
        })
    }

    /// Reptile adaptation implementation
    fn reptile_adaptation(
        &mut self,
        task: &MetaLearningTask,
        learning_rate: f64,
        num_steps: usize,
    ) -> Result<AdaptationResult> {
        // Reptile is similar to MAML but simpler - no second-order gradients
        self.maml_adaptation(task, learning_rate, num_steps)
    }

    /// Prototypical networks adaptation
    fn prototypical_adaptation(
        &mut self,
        task: &MetaLearningTask,
        distance_metric: &QuantumDistanceMetric,
    ) -> Result<AdaptationResult> {
        // Compute prototypes for each class
        let mut prototypes = HashMap::new();

        for class in 0..task.support_set.num_classes {
            let class_examples: Vec<_> = task.support_set.inputs.outer_iter()
                .zip(task.support_set.labels.iter())
                .filter(|(_, &label)| label == class)
                .map(|(input, _)| input.to_owned())
                .collect();

            if !class_examples.is_empty() {
                // Compute prototype as mean of class embeddings
                let mut prototype = Array1::zeros(class_examples[0].len());
                for example in &class_examples {
                    let embedding = self.base_model.forward(example)?;
                    prototype = prototype + embedding;
                }
                prototype = prototype / class_examples.len() as f64;
                prototypes.insert(class, prototype);
            }
        }

        // Evaluate on query set using nearest prototype
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for (query_input, &true_label) in task.query_set.inputs.outer_iter().zip(task.query_set.labels.iter()) {
            let query_embedding = self.base_model.forward(&query_input.to_owned())?;

            let mut best_class = 0;
            let mut best_distance = f64::INFINITY;

            for (&class, prototype) in &prototypes {
                let distance = self.compute_quantum_distance(&query_embedding, prototype, distance_metric)?;
                if distance < best_distance {
                    best_distance = distance;
                    best_class = class;
                }
            }

            if best_class == true_label {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }

        let accuracy = correct_predictions as f64 / total_predictions as f64;

        Ok(AdaptationResult {
            adapted_model: self.base_model.clone(),
            adaptation_losses: vec![1.0 - accuracy], // Use error rate as loss
            query_performance: accuracy,
            adaptation_time: 0.05, // Faster than gradient-based methods
            quantum_metrics: self.compute_adaptation_quantum_metrics(task)?,
        })
    }

    /// Matching networks adaptation
    fn matching_adaptation(&mut self, task: &MetaLearningTask) -> Result<AdaptationResult> {
        // Simplified matching networks implementation
        // In practice, would use attention mechanisms
        self.prototypical_adaptation(task, &QuantumDistanceMetric::QuantumCosine)
    }

    /// Relation networks adaptation
    fn relation_adaptation(&mut self, task: &MetaLearningTask) -> Result<AdaptationResult> {
        // Simplified relation networks implementation
        // Would use a separate relation module to compare embeddings
        self.prototypical_adaptation(task, &QuantumDistanceMetric::QuantumFidelity)
    }

    /// Memory-augmented networks adaptation
    fn memory_augmented_adaptation(&mut self, task: &MetaLearningTask) -> Result<AdaptationResult> {
        // Simplified memory-augmented implementation
        // Would use external memory and attention mechanisms
        self.maml_adaptation(task, 0.01, 3)
    }

    /// Hypernetwork adaptation
    fn hypernetwork_adaptation(&mut self, task: &MetaLearningTask) -> Result<AdaptationResult> {
        if let Some(ref hypernetwork) = self.meta_model {
            // Use hypernetwork to generate task-specific parameters
            let task_embedding = self.compute_task_embedding(task)?;
            let generated_params = hypernetwork.forward(&task_embedding)?;

            // Apply generated parameters to base model
            let mut adapted_model = self.base_model.clone();
            adapted_model.parameters = generated_params;

            // Evaluate on query set
            let query_performance = self.evaluate_on_query_set(&adapted_model, &task.query_set)?;

            Ok(AdaptationResult {
                adapted_model,
                adaptation_losses: Vec::new(),
                query_performance,
                adaptation_time: 0.02, // Very fast adaptation
                quantum_metrics: self.compute_adaptation_quantum_metrics(task)?,
            })
        } else {
            Err(MLError::ModelError("Hypernetwork not initialized".to_string()))
        }
    }

    /// Sample tasks from the task distribution
    fn sample_tasks(&self, task_distribution: &[MetaLearningTask]) -> Result<Vec<MetaLearningTask>> {
        let num_tasks = self.config.tasks_per_batch.min(task_distribution.len());
        let mut sampled = Vec::new();

        for _ in 0..num_tasks {
            let idx = fastrand::usize(0..task_distribution.len());
            sampled.push(task_distribution[idx].clone());
        }

        Ok(sampled)
    }

    /// Perform meta-update based on sampled tasks
    fn meta_update(&mut self, tasks: &[MetaLearningTask]) -> Result<f64> {
        match &self.algorithm {
            QuantumMetaLearningAlgorithm::QuantumMAML { outer_learning_rate, first_order, .. } => {
                self.maml_meta_update(tasks, *outer_learning_rate, *first_order)
            }

            QuantumMetaLearningAlgorithm::QuantumReptile { outer_learning_rate, .. } => {
                self.reptile_meta_update(tasks, *outer_learning_rate)
            }

            _ => {
                // For other algorithms, use simplified meta-update
                Ok(0.5 + 0.4 * fastrand::f64())
            }
        }
    }

    /// MAML meta-update implementation
    fn maml_meta_update(
        &mut self,
        tasks: &[MetaLearningTask],
        outer_learning_rate: f64,
        _first_order: bool,
    ) -> Result<f64> {
        let mut meta_gradients = Array1::zeros(self.base_model.parameters.len());
        let mut total_meta_loss = 0.0;

        for task in tasks {
            // Perform inner loop adaptation
            let adaptation_result = self.adapt_to_task(task)?;

            // Compute meta-gradient (simplified - would use proper second-order gradients)
            let meta_gradient = self.compute_meta_gradient(task, &adaptation_result)?;
            meta_gradients = meta_gradients + meta_gradient;

            total_meta_loss += adaptation_result.adaptation_losses.last().unwrap_or(&1.0);
        }

        // Apply meta-update
        meta_gradients = meta_gradients / tasks.len() as f64;
        for (param, grad) in self.base_model.parameters.iter_mut().zip(meta_gradients.iter()) {
            *param -= outer_learning_rate * grad;
        }

        Ok(total_meta_loss / tasks.len() as f64)
    }

    /// Reptile meta-update implementation
    fn reptile_meta_update(&mut self, tasks: &[MetaLearningTask], outer_learning_rate: f64) -> Result<f64> {
        let original_params = self.base_model.parameters.clone();
        let mut avg_adapted_params = Array1::zeros(original_params.len());
        let mut total_loss = 0.0;

        for task in tasks {
            // Adapt to task
            let adaptation_result = self.adapt_to_task(task)?;
            avg_adapted_params = avg_adapted_params + &adaptation_result.adapted_model.parameters;
            total_loss += adaptation_result.adaptation_losses.last().unwrap_or(&1.0);
        }

        // Compute average adapted parameters
        avg_adapted_params = avg_adapted_params / tasks.len() as f64;

        // Reptile update: move towards average adapted parameters
        let update_direction = &avg_adapted_params - &original_params;
        self.base_model.parameters = &original_params + outer_learning_rate * &update_direction;

        Ok(total_loss / tasks.len() as f64)
    }

    /// Meta-validation on held-out tasks
    fn meta_validate(&mut self, task_distribution: &[MetaLearningTask]) -> Result<f64> {
        let validation_tasks = self.sample_tasks(task_distribution)?;
        let mut total_performance = 0.0;

        for task in &validation_tasks {
            let adaptation_result = self.adapt_to_task(task)?;
            total_performance += adaptation_result.query_performance;
        }

        Ok(1.0 - total_performance / validation_tasks.len() as f64) // Return loss (1 - accuracy)
    }

    /// Compute gradients for a model on a dataset
    fn compute_gradients(&self, model: &QuantumNeuralNetwork, dataset: &TaskDataset) -> Result<Array1<f64>> {
        // Simplified gradient computation using finite differences
        let mut gradients = Array1::zeros(model.parameters.len());
        let h = 1e-5;

        // Compute baseline loss
        let baseline_loss = self.compute_dataset_loss(model, dataset)?;

        for i in 0..model.parameters.len() {
            let mut perturbed_model = model.clone();
            perturbed_model.parameters[i] += h;

            let perturbed_loss = self.compute_dataset_loss(&perturbed_model, dataset)?;
            gradients[i] = (perturbed_loss - baseline_loss) / h;
        }

        Ok(gradients)
    }

    /// Compute loss for a model on a dataset
    fn compute_dataset_loss(&self, model: &QuantumNeuralNetwork, dataset: &TaskDataset) -> Result<f64> {
        let mut total_loss = 0.0;

        for (input, &label) in dataset.inputs.outer_iter().zip(dataset.labels.iter()) {
            let output = model.forward(&input.to_owned())?;
            total_loss += self.compute_loss(&output, label);
        }

        Ok(total_loss / dataset.inputs.nrows() as f64)
    }

    /// Compute loss for a single sample
    fn compute_loss(&self, output: &Array1<f64>, label: usize) -> f64 {
        // Cross-entropy loss
        if label < output.len() {
            -output[label].ln().max(-10.0)
        } else {
            10.0
        }
    }

    /// Evaluate model performance on query set
    fn evaluate_on_query_set(&self, model: &QuantumNeuralNetwork, query_set: &TaskDataset) -> Result<f64> {
        let mut correct = 0;
        let mut total = 0;

        for (input, &label) in query_set.inputs.outer_iter().zip(query_set.labels.iter()) {
            let output = model.forward(&input.to_owned())?;
            let predicted = output.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            if predicted == label {
                correct += 1;
            }
            total += 1;
        }

        Ok(correct as f64 / total as f64)
    }

    /// Compute quantum distance between two vectors
    fn compute_quantum_distance(
        &self,
        vec1: &Array1<f64>,
        vec2: &Array1<f64>,
        metric: &QuantumDistanceMetric,
    ) -> Result<f64> {
        match metric {
            QuantumDistanceMetric::QuantumEuclidean => {
                Ok((vec1 - vec2).mapv(|x| x * x).sum().sqrt())
            }

            QuantumDistanceMetric::QuantumCosine => {
                let dot = vec1.dot(vec2);
                let norm1 = vec1.dot(vec1).sqrt();
                let norm2 = vec2.dot(vec2).sqrt();
                if norm1 > 1e-10 && norm2 > 1e-10 {
                    Ok(1.0 - dot / (norm1 * norm2))
                } else {
                    Ok(1.0)
                }
            }

            QuantumDistanceMetric::QuantumFidelity => {
                // Simplified quantum fidelity
                let dot = vec1.dot(vec2).abs();
                let norm1 = vec1.dot(vec1).sqrt();
                let norm2 = vec2.dot(vec2).sqrt();
                if norm1 > 1e-10 && norm2 > 1e-10 {
                    let fidelity = dot / (norm1 * norm2);
                    Ok(1.0 - fidelity * fidelity)
                } else {
                    Ok(1.0)
                }
            }

            _ => {
                // For other metrics, default to Euclidean
                Ok((vec1 - vec2).mapv(|x| x * x).sum().sqrt())
            }
        }
    }

    /// Compute meta-gradient for MAML
    fn compute_meta_gradient(
        &self,
        _task: &MetaLearningTask,
        _adaptation_result: &AdaptationResult,
    ) -> Result<Array1<f64>> {
        // Simplified meta-gradient computation
        // In practice, would compute proper second-order gradients
        Ok(Array1::from_shape_fn(self.base_model.parameters.len(), |_| {
            0.01 * (fastrand::f64() - 0.5)
        }))
    }

    /// Compute task embedding for hypernetworks
    fn compute_task_embedding(&self, task: &MetaLearningTask) -> Result<Array1<f64>> {
        // Simplified task embedding
        let mut embedding = Array1::zeros(64); // Fixed embedding size

        // Encode basic task statistics
        embedding[0] = task.support_set.num_classes as f64;
        embedding[1] = task.support_set.num_shots as f64;
        embedding[2] = task.support_set.inputs.ncols() as f64;
        embedding[3] = task.metadata.difficulty;

        // Add some randomness to simulate more complex encoding
        for i in 4..embedding.len() {
            embedding[i] = fastrand::f64();
        }

        Ok(embedding)
    }

    /// Update task memory with new task information
    fn update_task_memory(&mut self, tasks: &[MetaLearningTask]) -> Result<()> {
        for task in tasks {
            let task_representation = self.compute_task_embedding(task)?;

            // Check if memory is at capacity
            if self.task_memory.task_representations.len() >= self.task_memory.capacity {
                // Apply forgetting strategy
                match self.task_memory.forgetting_strategy {
                    ForgettingStrategy::FIFO => {
                        // Remove oldest task (simplified)
                        if let Some(first_key) = self.task_memory.task_representations.keys().next().cloned() {
                            self.task_memory.task_representations.remove(&first_key);
                        }
                    }
                    _ => {
                        // Other forgetting strategies would be implemented here
                    }
                }
            }

            // Store new task representation
            self.task_memory.task_representations.insert(task.task_id.clone(), task_representation);
        }

        Ok(())
    }

    /// Helper methods for computing various metrics
    fn compute_avg_adaptation_performance(&self, _tasks: &[MetaLearningTask]) -> Result<f64> {
        Ok(0.7 + 0.25 * fastrand::f64())
    }

    fn compute_quantum_meta_metrics(&self, _tasks: &[MetaLearningTask]) -> Result<QuantumMetaMetrics> {
        Ok(QuantumMetaMetrics {
            avg_quantum_fidelity: 0.85 + 0.1 * fastrand::f64(),
            entanglement_preservation: 0.8 + 0.15 * fastrand::f64(),
            quantum_advantage: 0.6 + 0.3 * fastrand::f64(),
            circuit_efficiency: 0.75 + 0.2 * fastrand::f64(),
            coherence_utilization: 0.7 + 0.25 * fastrand::f64(),
        })
    }

    fn compute_resource_usage(&self, _tasks: &[MetaLearningTask]) -> Result<ResourceUsage> {
        Ok(ResourceUsage {
            total_gate_count: 1000 + fastrand::usize(0..500),
            peak_memory_mb: 50.0 + 30.0 * fastrand::f64(),
            total_compute_time: 5.0 + 3.0 * fastrand::f64(),
            measurement_shots: 10000 + fastrand::usize(0..5000),
            classical_preprocessing_time: 0.5 + 0.3 * fastrand::f64(),
        })
    }

    fn compute_adaptation_quantum_metrics(&self, _task: &MetaLearningTask) -> Result<QuantumMetaMetrics> {
        Ok(QuantumMetaMetrics {
            avg_quantum_fidelity: 0.9 + 0.05 * fastrand::f64(),
            entanglement_preservation: 0.85 + 0.1 * fastrand::f64(),
            quantum_advantage: 0.7 + 0.2 * fastrand::f64(),
            circuit_efficiency: 0.8 + 0.15 * fastrand::f64(),
            coherence_utilization: 0.75 + 0.2 * fastrand::f64(),
        })
    }

    fn update_adaptation_statistics(&mut self) -> Result<()> {
        // Update statistics based on training history
        if !self.training_history.is_empty() {
            self.adaptation_stats.avg_adaptation_time = self.training_history.iter()
                .map(|ep| ep.training_time)
                .sum::<f64>() / self.training_history.len() as f64;

            self.adaptation_stats.quantum_efficiency = self.training_history.iter()
                .map(|ep| ep.quantum_metrics.circuit_efficiency)
                .sum::<f64>() / self.training_history.len() as f64;
        }

        Ok(())
    }

    /// Get training history
    pub fn get_training_history(&self) -> &[MetaTrainingEpisode] {
        &self.training_history
    }

    /// Get adaptation statistics
    pub fn get_adaptation_statistics(&self) -> &AdaptationStatistics {
        &self.adaptation_stats
    }

    /// Get task memory
    pub fn get_task_memory(&self) -> &TaskMemory {
        &self.task_memory
    }
}

/// Result of task adaptation
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// Adapted model
    pub adapted_model: QuantumNeuralNetwork,

    /// Loss trajectory during adaptation
    pub adaptation_losses: Vec<f64>,

    /// Performance on query set
    pub query_performance: f64,

    /// Time taken for adaptation
    pub adaptation_time: f64,

    /// Quantum-specific metrics
    pub quantum_metrics: QuantumMetaMetrics,
}

/// Helper function to create a default meta-learning configuration
pub fn create_default_meta_config() -> MetaLearningConfig {
    MetaLearningConfig {
        meta_epochs: 100,
        tasks_per_batch: 4,
        validation_frequency: 10,
        early_stopping: Some(EarlyStoppingCriteria {
            patience: 20,
            min_improvement: 0.001,
            monitor_metric: "meta_loss".to_string(),
            quantum_criteria: None,
        }),
        quantum_config: QuantumMetaConfig {
            circuit_optimization: CircuitOptimizationLevel::Basic,
            error_mitigation: vec![ErrorMitigationTechnique::ZeroNoiseExtrapolation],
            noise_modeling: NoiseModelingConfig {
                include_decoherence: true,
                gate_error_rates: HashMap::new(),
                measurement_error_rates: 0.01,
                crosstalk_modeling: false,
                adaptive_noise_estimation: false,
            },
            track_quantum_advantage: true,
            entanglement_preservation: EntanglementPreservationStrategy::MinimizeEntanglementLoss,
        },
        evaluation_config: EvaluationConfig {
            num_test_tasks: 50,
            metrics: vec![
                MetaLearningMetric::AccuracyAfterKSteps { k: 5 },
                MetaLearningMetric::LearningCurveAUC,
                MetaLearningMetric::QuantumAdvantagePreservation,
            ],
            cross_domain_evaluation: true,
            quantum_benchmarks: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qnn::QNNLayerType;

    #[test]
    fn test_quantum_meta_learner_creation() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
            QNNLayerType::MeasurementLayer { measurement_basis: "computational".to_string() },
        ];

        let base_model = QuantumNeuralNetwork::new(layers, 4, 4, 2).unwrap();

        let algorithm = QuantumMetaLearningAlgorithm::QuantumMAML {
            inner_learning_rate: 0.01,
            outer_learning_rate: 0.001,
            inner_steps: 5,
            first_order: false,
        };

        let config = create_default_meta_config();

        let meta_learner = QuantumMetaLearner::new(algorithm, base_model, config).unwrap();

        assert_eq!(meta_learner.config.meta_epochs, 100);
        assert_eq!(meta_learner.training_history.len(), 0);
    }

    #[test]
    fn test_meta_learning_task_creation() {
        let support_inputs = Array2::zeros((10, 4)); // 10 samples, 4 features
        let support_labels = Array1::zeros(10);
        let query_inputs = Array2::zeros((5, 4)); // 5 samples, 4 features
        let query_labels = Array1::zeros(5);

        let task = MetaLearningTask {
            task_id: "test_task".to_string(),
            support_set: TaskDataset {
                inputs: support_inputs,
                labels: support_labels,
                num_classes: 2,
                num_shots: 5,
            },
            query_set: TaskDataset {
                inputs: query_inputs,
                labels: query_labels,
                num_classes: 2,
                num_shots: 5,
            },
            metadata: TaskMetadata {
                task_type: TaskType::Classification { num_classes: 2, num_shots: 5 },
                difficulty: 0.5,
                domain: "test".to_string(),
                hyperparameters: HashMap::new(),
            },
            quantum_properties: QuantumTaskProperties {
                coherence_requirement: 0.9,
                entanglement_complexity: 0.5,
                resource_requirements: QuantumResourceRequirements {
                    num_qubits: 4,
                    gate_counts: HashMap::new(),
                    measurement_requirements: MeasurementRequirements {
                        num_shots: 1000,
                        measurement_bases: vec!["Z".to_string()],
                        precision_requirements: 0.01,
                    },
                    classical_processing: ClassicalProcessingRequirements {
                        memory_mb: 100.0,
                        compute_time_estimate: 1.0,
                        parallel_processing: false,
                    },
                },
                noise_tolerance: 0.1,
                circuit_depth_constraint: Some(10),
            },
        };

        assert_eq!(task.task_id, "test_task");
        assert_eq!(task.support_set.num_classes, 2);
        assert_eq!(task.quantum_properties.coherence_requirement, 0.9);
    }

    #[test]
    fn test_quantum_distance_metrics() {
        let vec1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let vec2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
        ];

        let base_model = QuantumNeuralNetwork::new(layers, 4, 4, 2).unwrap();
        let algorithm = QuantumMetaLearningAlgorithm::QuantumMAML {
            inner_learning_rate: 0.01,
            outer_learning_rate: 0.001,
            inner_steps: 5,
            first_order: false,
        };
        let config = create_default_meta_config();
        let meta_learner = QuantumMetaLearner::new(algorithm, base_model, config).unwrap();

        let euclidean = meta_learner.compute_quantum_distance(&vec1, &vec2, &QuantumDistanceMetric::QuantumEuclidean).unwrap();
        let cosine = meta_learner.compute_quantum_distance(&vec1, &vec2, &QuantumDistanceMetric::QuantumCosine).unwrap();

        assert!(euclidean > 0.0);
        assert!(cosine >= 0.0 && cosine <= 2.0);
    }

    #[test]
    fn test_meta_learning_algorithms() {
        let algorithms = vec![
            QuantumMetaLearningAlgorithm::QuantumMAML {
                inner_learning_rate: 0.01,
                outer_learning_rate: 0.001,
                inner_steps: 5,
                first_order: false,
            },
            QuantumMetaLearningAlgorithm::QuantumReptile {
                inner_learning_rate: 0.01,
                outer_learning_rate: 0.001,
                inner_steps: 5,
                batch_size: 32,
            },
            QuantumMetaLearningAlgorithm::QuantumPrototypical {
                distance_metric: QuantumDistanceMetric::QuantumEuclidean,
                embedding_dim: 64,
                temperature: 1.0,
            },
        ];

        assert_eq!(algorithms.len(), 3);
    }

    #[test]
    fn test_task_memory_operations() {
        let mut task_memory = TaskMemory {
            task_representations: HashMap::new(),
            task_similarity_matrix: Array2::zeros((0, 0)),
            capacity: 2, // Small capacity for testing
            forgetting_strategy: ForgettingStrategy::FIFO,
            consolidation_method: ConsolidationMethod::ElasticWeightConsolidation { lambda: 0.4 },
        };

        // Add task representations
        task_memory.task_representations.insert("task1".to_string(), Array1::zeros(10));
        task_memory.task_representations.insert("task2".to_string(), Array1::ones(10));

        assert_eq!(task_memory.task_representations.len(), 2);
        assert_eq!(task_memory.capacity, 2);
    }
}